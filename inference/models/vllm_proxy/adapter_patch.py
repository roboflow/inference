"""Deterministic transform of Roboflow LoRA adapter packages into a form vLLM accepts.

Roboflow model packages for qwen3_5 fine-tunes contain `adapter_config.json`
and `adapter_model.safetensors` produced by PEFT during training. Two issues
prevent serving them directly with vLLM's dynamic LoRA loading:

1. **Weight-key layout** - Roboflow adapters store language-model keys like
   `...model.layers.N...`, while the qwen3_5 VL architecture names them
   `...model.language_model.layers.N...` (see `refactor_adapter_weights_key`
   in `inference_models/models/qwen3_5/qwen3_5_hf.py`, which performs the
   equivalent remap when loading via PEFT in-process). The exact layout vLLM
   v0.22.1 accepts is empirically unconfirmed, so the remap target is a
   configurable template (`VLLM_ADAPTER_KEY_TEMPLATE`).
2. **DoRA** - production adapters use `use_dora: true`; stock vLLM may reject
   DoRA adapters. The `policy` parameter controls handling: `reject`
   (default), `strip` (drop magnitude vectors), or `svd` (convert DoRA to a
   plain LoRA via the real merge math + truncated SVD - requires base
   weights).

This module is intentionally limited to pure file/tensor operations
(safetensors + json + torch CPU). It performs no network access.
"""

import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file, save_file

from inference.core import logger
from inference.models.vllm_proxy.config import (
    get_vllm_adapter_key_template,
    get_vllm_max_lora_rank,
    get_vllm_served_base_name,
    get_vllm_served_base_variant,
    get_vllm_vision_lora_norm_threshold,
)
from inference.models.vllm_proxy.errors import AdapterNotServableError

ADAPTER_CONFIG_FILE = "adapter_config.json"
ADAPTER_WEIGHTS_FILE = "adapter_model.safetensors"
PATCH_REPORT_FILE = "patch_report.json"

# Outcomes of the `base_model_name_or_path` vs served-base cross-check,
# recorded in `PatchReport.base_model_check` / `patch_report.json`.
BASE_MODEL_CHECK_MATCH = "match"
BASE_MODEL_CHECK_SKIPPED = "skipped-missing-base_model_name_or_path"

DORA_POLICIES = ("reject", "strip", "svd")

# PEFT serialises adapter keys with this prefix; vLLM's LoRA loader expects it
# to be preserved.
PEFT_KEY_PREFIX = "base_model.model."

# Markers identifying tensors that belong to the vision encoder of qwen VL
# architectures (vision tower blocks + the vision->language merger).
VISION_MODULE_MARKERS = ("visual.", "vision_tower", "merger")

# Language-model modules vLLM supports for qwen3_5 LoRA adapters.
SUPPORTED_LANGUAGE_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Fields PEFT/vLLM choke on - same set the existing HF paths strip before
# constructing `LoraConfig` (see `inference/models/qwen25vl/qwen25vl.py`).
ADAPTER_CONFIG_KEYS_TO_STRIP = ("eva_config", "lora_bias", "exclude_modules")

_LORA_TENSOR_MARKERS = (
    ".lora_A.",
    ".lora_B.",
    ".lora_embedding_A",
    ".lora_embedding_B",
)
_MAGNITUDE_MARKER = "lora_magnitude_vector"

_LANGUAGE_LAYERS_PREFIXES = (
    "model.language_model.layers.",
    "model.layers.",
)


@dataclass
class PatchReport:
    """Summary of a `patch_adapter` run, persisted as `patch_report.json`."""

    source_dir: str
    dst_dir: str
    policy: str
    source_use_dora: bool
    lora_rank: int
    key_template: str
    target_modules: List[str] = field(default_factory=list)
    total_source_tensors: int = 0
    remapped_keys: int = 0
    dropped_vision_keys: List[str] = field(default_factory=list)
    dropped_magnitude_keys: List[str] = field(default_factory=list)
    svd_rank: Optional[int] = None
    source_weights_digest: Optional[str] = None
    patched_weights_digest: Optional[str] = None
    base_model_name_or_path: Optional[str] = None
    base_model_check: str = BASE_MODEL_CHECK_SKIPPED
    # Registry modelVariant as recorded by the weights provider - advisory
    # only (sometimes misregistered); kept alongside base_model_name_or_path
    # so registry/adapter drift can be audited from the patch report.
    registry_variant: Optional[str] = None
    notes: List[str] = field(default_factory=list)


def patch_adapter(
    src_dir: str,
    dst_dir: str,
    policy: str = "reject",
    base_dir: Optional[str] = None,
    max_lora_rank: Optional[int] = None,
    vision_norm_threshold: Optional[float] = None,
    key_template: Optional[str] = None,
    model_id: Optional[str] = None,
    registry_variant: Optional[str] = None,
) -> PatchReport:
    """Transforms the adapter in `src_dir` into a vLLM-servable one in `dst_dir`.

    Pipeline:
        1. Validate `adapter_config.json` (`modules_to_save` empty, rank within
           `VLLM_MAX_LORA_RANK`), and cross-check the adapter's own
           `base_model_name_or_path` against the served base
           (`VLLM_SERVED_BASE_VARIANT` / `VLLM_SERVED_BASE_NAME`) - registry
           variant metadata can be wrong, and loading an adapter trained on a
           different base fails deep inside vLLM with an opaque tensor-shape
           error.
        2. Drop vision-tower tensors; raise if any dropped vision `lora_B`
           tensor has a norm above `VLLM_VISION_LORA_NORM_THRESHOLD` (a
           meaningfully trained vision adapter cannot be approximated by a
           language-only LoRA).
        3. Apply the DoRA `policy` (`reject` / `strip` / `svd`).
        4. Remap weight keys to the configured vLLM/PEFT layout.
        5. Rewrite `adapter_config.json` (intersect `target_modules` with the
           supported language-module set, strip unsupported fields).
        6. Write patched weights + config + `patch_report.json` into `dst_dir`.

    Args:
        src_dir: Directory with the downloaded Roboflow adapter artifacts.
        dst_dir: Output directory for the patched adapter.
        policy: DoRA handling policy, one of `reject` / `strip` / `svd`.
        base_dir: Directory holding base model safetensors - required for the
            `svd` policy. Runtime AdapterManager calls intentionally download
            adapter-only artifacts, so `svd` is reserved for offline/lab
            conversion paths that pass base weights explicitly.
        max_lora_rank: Maximum accepted LoRA rank (defaults to
            `VLLM_MAX_LORA_RANK`).
        vision_norm_threshold: Norm threshold above which a vision `lora_B`
            tensor marks the adapter as not servable (defaults to
            `VLLM_VISION_LORA_NORM_THRESHOLD`).
        key_template: Remap target template with a `{suffix}` placeholder
            (defaults to `VLLM_ADAPTER_KEY_TEMPLATE`).
        model_id: Roboflow model id the adapter belongs to - only used to
            make error/log messages actionable.
        registry_variant: Registry `modelVariant` as recorded by the weights
            provider - advisory only, recorded in the patch report for
            registry/adapter drift auditing.

    Raises:
        AdapterNotServableError: If the adapter cannot be made servable.
    """
    if policy not in DORA_POLICIES:
        raise ValueError(
            f"Unknown DoRA policy: {policy!r} - expected one of {DORA_POLICIES}."
        )
    if max_lora_rank is None:
        max_lora_rank = get_vllm_max_lora_rank()
    if vision_norm_threshold is None:
        vision_norm_threshold = get_vllm_vision_lora_norm_threshold()
    if key_template is None:
        key_template = get_vllm_adapter_key_template()

    config = _load_adapter_config(adapter_dir=src_dir)
    _validate_adapter_config(config=config, max_lora_rank=max_lora_rank)
    declared_base, base_model_check = cross_check_base_model(
        config=config, model_id=model_id
    )
    source_use_dora = bool(config.get("use_dora", False))
    lora_rank = int(config["r"])

    weights_path = os.path.join(src_dir, ADAPTER_WEIGHTS_FILE)
    if not os.path.isfile(weights_path):
        raise AdapterNotServableError(
            f"Adapter package in {src_dir} is missing {ADAPTER_WEIGHTS_FILE}."
        )
    tensors = load_file(weights_path)

    report = PatchReport(
        source_dir=os.path.abspath(src_dir),
        dst_dir=os.path.abspath(dst_dir),
        policy=policy,
        source_use_dora=source_use_dora,
        lora_rank=lora_rank,
        key_template=key_template,
        total_source_tensors=len(tensors),
        source_weights_digest=_sha256_of_file(weights_path),
        base_model_name_or_path=declared_base,
        base_model_check=base_model_check,
        registry_variant=registry_variant,
    )

    tensors = _drop_vision_tensors(
        tensors=tensors,
        vision_norm_threshold=vision_norm_threshold,
        report=report,
    )

    if source_use_dora:
        if policy == "reject":
            raise AdapterNotServableError(
                "Adapter uses DoRA (`use_dora: true`), which is rejected under "
                "the configured `VLLM_DORA_POLICY=reject`. Set the policy to "
                "`strip` at runtime, or run offline `svd` conversion with "
                "base weights."
            )
        if policy == "strip":
            tensors = _strip_magnitude_vectors(tensors=tensors, report=report)
            report.notes.append(
                "DoRA magnitude vectors stripped - served adapter approximates "
                "the trained DoRA adapter."
            )
        elif policy == "svd":
            if base_dir is None:
                raise AdapterNotServableError(
                    "DoRA policy `svd` requires base model weights "
                    "(`base_dir` was not provided). Runtime AdapterManager "
                    "downloads adapter-only artifacts; reserve `svd` for "
                    "offline conversion paths that pass base weights explicitly."
                )
            svd_rank = min(lora_rank, max_lora_rank)
            base_weight_lookup = _build_base_weight_lookup(base_dir=base_dir)
            tensors = _convert_dora_tensors_to_plain_lora(
                tensors=tensors,
                config=config,
                base_weight_lookup=base_weight_lookup,
                rank=svd_rank,
            )
            report.svd_rank = svd_rank
            report.notes.append(
                "DoRA adapter converted to plain LoRA via merged-weight SVD "
                f"truncation (rank={svd_rank})."
            )
            config = _rewrite_config_for_svd(config=config, rank=svd_rank)
            lora_rank = svd_rank
    else:
        # Plain-LoRA adapters may still carry stray magnitude keys - drop them.
        tensors = _strip_magnitude_vectors(tensors=tensors, report=report)

    remapped_tensors = {}
    for key, tensor in tensors.items():
        new_key = remap_adapter_weight_key(key=key, key_template=key_template)
        if new_key != key:
            report.remapped_keys += 1
        remapped_tensors[new_key] = tensor

    config = _rewrite_adapter_config(config=config, policy=policy)
    report.target_modules = list(config["target_modules"])
    if not remapped_tensors:
        raise AdapterNotServableError(
            "Adapter contains no servable LoRA tensors after filtering."
        )

    dst_dir = os.path.abspath(dst_dir)
    dst_parent = os.path.dirname(dst_dir)
    os.makedirs(dst_parent, exist_ok=True)
    staging_dir = tempfile.mkdtemp(
        prefix=f".{os.path.basename(dst_dir)}-", dir=dst_parent
    )
    try:
        patched_weights_path = os.path.join(staging_dir, ADAPTER_WEIGHTS_FILE)
        save_file(remapped_tensors, patched_weights_path)
        report.patched_weights_digest = _sha256_of_file(patched_weights_path)
        with open(os.path.join(staging_dir, ADAPTER_CONFIG_FILE), "w") as f:
            json.dump(config, f, indent=2, sort_keys=True)
        report.notes.append(
            "The exact adapter key layout accepted by vLLM v0.22.1 is empirically "
            "unconfirmed - adjust VLLM_ADAPTER_KEY_TEMPLATE if vLLM rejects the "
            "patched adapter."
        )
        with open(os.path.join(staging_dir, PATCH_REPORT_FILE), "w") as f:
            json.dump(dataclasses.asdict(report), f, indent=2)
        _replace_directory(src_dir=staging_dir, dst_dir=dst_dir)
        staging_dir = None
    finally:
        if staging_dir is not None:
            _remove_path(staging_dir)
    return report


def _replace_directory(src_dir: str, dst_dir: str) -> None:
    """Publishes `src_dir` at `dst_dir` without exposing partial contents."""
    old_dir = None
    if os.path.exists(dst_dir):
        old_dir = tempfile.mkdtemp(
            prefix=f".{os.path.basename(dst_dir)}-old-",
            dir=os.path.dirname(os.path.abspath(dst_dir)),
        )
        os.rmdir(old_dir)
        os.replace(dst_dir, old_dir)
    try:
        os.replace(src_dir, dst_dir)
    except Exception:
        if (
            old_dir is not None
            and os.path.exists(old_dir)
            and not os.path.exists(dst_dir)
        ):
            os.replace(old_dir, dst_dir)
        raise
    finally:
        if old_dir is not None and os.path.exists(old_dir):
            _remove_path(old_dir)


def _remove_path(path: str) -> None:
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def svd_convert(base_dir: str, adapter_dir: str, dst_dir: str, rank: int) -> None:
    """Converts a DoRA adapter into a plain LoRA via merged-weight SVD.

    Math (mirrors PEFT's DoRA merge - `peft/tuners/lora/dora.py` /
    `Linear.merge`):

        scale     = lora_alpha / r            (or lora_alpha / sqrt(r) with rsLoRA)
        W         = W0 + scale * (B @ A)      # candidate direction matrix
        col_norm  = ||W||_2 along dim=1       # one norm per output row, i.e.
                                              # "column-wise" norm of W^T as in
                                              # the DoRA paper
        W_merged  = (m / col_norm).view(-1, 1) * W   # m = lora_magnitude_vector
        dW        = W_merged - W0

    `dW` is then SVD-truncated: `U S V^T = svd(dW)`,
    `B_new = U[:, :rank] sqrt(S[:rank])`, `A_new = sqrt(S[:rank]) V[:rank]^T`,
    and the emitted config uses `lora_alpha = rank` so the effective scale is
    1 and `B_new @ A_new` directly approximates `dW`. Note that `dW` is not
    low-rank in general (the per-row rescaling perturbs all of `W0`), so this
    is an approximation; it is exact when the magnitude vector equals the
    column norm (i.e. the DoRA rescaling is a no-op).

    Runs on CPU; written for clarity rather than speed.

    Args:
        base_dir: Directory holding base model safetensors (the `base/` dir of
            a Roboflow model package).
        adapter_dir: Directory holding the DoRA adapter
            (`adapter_config.json` + `adapter_model.safetensors`).
        dst_dir: Output directory for the plain-LoRA adapter.
        rank: Rank of the emitted LoRA.
    """
    config = _load_adapter_config(adapter_dir=adapter_dir)
    tensors = load_file(os.path.join(adapter_dir, ADAPTER_WEIGHTS_FILE))
    base_weight_lookup = _build_base_weight_lookup(base_dir=base_dir)
    converted = _convert_dora_tensors_to_plain_lora(
        tensors=tensors,
        config=config,
        base_weight_lookup=base_weight_lookup,
        rank=rank,
    )
    config = _rewrite_config_for_svd(config=config, rank=rank)
    os.makedirs(dst_dir, exist_ok=True)
    save_file(converted, os.path.join(dst_dir, ADAPTER_WEIGHTS_FILE))
    with open(os.path.join(dst_dir, ADAPTER_CONFIG_FILE), "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def remap_adapter_weight_key(key: str, key_template: str) -> str:
    """Remaps a Roboflow adapter weight key to the vLLM/PEFT expected layout.

    Roboflow qwen3_5 adapters store language-model keys as
    `base_model.model.model.layers.N....` while the qwen3_5 VL architecture
    names those modules `model.language_model.layers.N....`. This function
    rewrites any key under `model.layers.` / `model.language_model.layers.`
    (with or without the PEFT `base_model.model.` prefix) using
    `key_template`, which receives the part after the layers prefix as
    `{suffix}`. The default template preserves the standard PEFT
    `base_model.model.` prefix convention:

        base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
            -> base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight

    Keys that do not point into the language-model layers are returned
    unchanged.
    """
    core = key
    if core.startswith(PEFT_KEY_PREFIX):
        core = core[len(PEFT_KEY_PREFIX) :]
    for layers_prefix in _LANGUAGE_LAYERS_PREFIXES:
        if core.startswith(layers_prefix):
            suffix = core[len(layers_prefix) :]
            return key_template.format(suffix=suffix)
    return key


def normalize_base_model_reference(value: str) -> str:
    """Normalises a base-model reference for the served-base cross-check.

    Lowercases, strips any org prefix (`qwen/qwen3_5-2b` -> `qwen3_5-2b`) and
    drops separator characters, so `qwen/qwen3_5-0.8b`, `Qwen3.5-0.8B` and
    `qwen3_5-0.8b` all compare equal, while genuinely different bases
    (`qwen3_5-2b` vs `qwen3_5-0.8b`) stay distinct.
    """
    value = value.strip().lower()
    if "/" in value:
        value = value.rsplit("/", 1)[-1]
    return re.sub(r"[^a-z0-9]+", "", value)


def cross_check_base_model(
    config: dict, model_id: Optional[str] = None
) -> Tuple[Optional[str], str]:
    """Cross-checks the adapter's declared base against the served base.

    Registry variant metadata occasionally contradicts the adapter's own
    `adapter_config.json` (incident 2026-06-10: `image-text/223` was recorded
    as a `0.8b-peft` fine-tune but its adapter config declared
    `qwen/qwen3_5-2b`). Without this pre-flight check the mismatch only
    surfaces as an opaque tensor-shape `RuntimeError` inside vLLM's
    `/v1/load_lora_adapter`.

    Returns `(declared_base, check_result)` where `check_result` is one of
    `BASE_MODEL_CHECK_MATCH` / `BASE_MODEL_CHECK_SKIPPED`. The check is
    skipped (with a warning) when `base_model_name_or_path` is missing/empty.

    Raises:
        AdapterNotServableError: When the declared base matches neither
            `VLLM_SERVED_BASE_VARIANT` nor `VLLM_SERVED_BASE_NAME`.
    """
    declared_base = (config.get("base_model_name_or_path") or "").strip()
    if not declared_base:
        logger.warning(
            "Adapter config for model %s declares no base_model_name_or_path "
            "- skipping the served-base cross-check.",
            model_id or "<unknown>",
        )
        return None, BASE_MODEL_CHECK_SKIPPED
    served_base_variant = get_vllm_served_base_variant()
    served_base_name = get_vllm_served_base_name()
    normalized_declared = normalize_base_model_reference(declared_base)
    if normalized_declared in {
        normalize_base_model_reference(served_base_variant),
        normalize_base_model_reference(served_base_name),
    }:
        return declared_base, BASE_MODEL_CHECK_MATCH
    raise AdapterNotServableError(
        f"Adapter for model {model_id or '<unknown>'} declares "
        f"base_model_name_or_path={declared_base!r} in its "
        f"adapter_config.json, but this vLLM deployment serves base "
        f"{served_base_variant!r} (VLLM_SERVED_BASE_VARIANT; served name "
        f"{served_base_name!r}). The model's registry variant metadata "
        f"contradicts the adapter's own config - this is a registry data "
        f"bug: fix the recorded modelVariant for "
        f"{model_id or 'this model'} so it matches the adapter's true base "
        f"{declared_base!r}."
    )


def extract_module_path(key: str) -> Optional[str]:
    """Returns the module path of a LoRA tensor key, or None for other keys.

    E.g. `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`
    -> `base_model.model.model.layers.0.self_attn.q_proj`.
    """
    for marker in _LORA_TENSOR_MARKERS:
        index = key.find(marker)
        if index != -1:
            return key[:index]
    index = key.find(f".{_MAGNITUDE_MARKER}")
    if index != -1:
        return key[:index]
    return None


def is_vision_tensor_key(key: str) -> bool:
    """True if the tensor's module path belongs to the vision encoder."""
    module_path = extract_module_path(key) or key
    return any(marker in module_path for marker in VISION_MODULE_MARKERS)


def _load_adapter_config(adapter_dir: str) -> dict:
    config_path = os.path.join(adapter_dir, ADAPTER_CONFIG_FILE)
    if not os.path.isfile(config_path):
        raise AdapterNotServableError(
            f"Adapter package in {adapter_dir} is missing {ADAPTER_CONFIG_FILE}."
        )
    with open(config_path) as f:
        return json.load(f)


def _validate_adapter_config(config: dict, max_lora_rank: int) -> None:
    modules_to_save = config.get("modules_to_save") or []
    if modules_to_save:
        raise AdapterNotServableError(
            "Adapter declares `modules_to_save` "
            f"({modules_to_save}) - full-module replacements cannot be served "
            "via vLLM dynamic LoRA."
        )
    rank = config.get("r")
    if rank is None:
        raise AdapterNotServableError(
            "Adapter config does not declare a LoRA rank (`r`)."
        )
    if int(rank) > max_lora_rank:
        raise AdapterNotServableError(
            f"Adapter rank r={rank} exceeds the maximum rank supported by the "
            f"vLLM deployment (VLLM_MAX_LORA_RANK={max_lora_rank})."
        )
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise AdapterNotServableError(
            "Adapter uses per-module `rank_pattern`/`alpha_pattern`, which is "
            "not supported by the vLLM proxy transform."
        )


def _drop_vision_tensors(
    tensors: Dict[str, torch.Tensor],
    vision_norm_threshold: float,
    report: PatchReport,
) -> Dict[str, torch.Tensor]:
    kept: Dict[str, torch.Tensor] = {}
    for key, tensor in tensors.items():
        if not is_vision_tensor_key(key):
            kept[key] = tensor
            continue
        if ".lora_B." in key:
            norm = float(torch.linalg.norm(tensor.float()))
            if norm > vision_norm_threshold:
                raise AdapterNotServableError(
                    f"Adapter meaningfully trained the vision encoder "
                    f"(|{key}| = {norm:.6f} > threshold "
                    f"{vision_norm_threshold}) - dropping the vision tensors "
                    "would change model behaviour, so the adapter cannot be "
                    "served via the language-only vLLM LoRA path."
                )
        report.dropped_vision_keys.append(key)
    return kept


def _strip_magnitude_vectors(
    tensors: Dict[str, torch.Tensor], report: PatchReport
) -> Dict[str, torch.Tensor]:
    kept: Dict[str, torch.Tensor] = {}
    for key, tensor in tensors.items():
        if _MAGNITUDE_MARKER in key:
            report.dropped_magnitude_keys.append(key)
            continue
        kept[key] = tensor
    return kept


def _rewrite_adapter_config(config: dict, policy: str) -> dict:
    config = dict(config)
    target_modules = config.get("target_modules") or []
    supported = [
        module
        for module in SUPPORTED_LANGUAGE_TARGET_MODULES
        if module in target_modules
    ]
    if not supported:
        raise AdapterNotServableError(
            f"Adapter target_modules {target_modules} share no modules with "
            f"the supported language-module set "
            f"{list(SUPPORTED_LANGUAGE_TARGET_MODULES)}."
        )
    config["target_modules"] = supported
    for config_key in ADAPTER_CONFIG_KEYS_TO_STRIP:
        config.pop(config_key, None)
    if policy in ("strip", "svd"):
        config["use_dora"] = False
    # `use_rslora` is intentionally preserved - it changes the effective
    # scaling (alpha / sqrt(r)) and must survive the transform.
    return config


def _rewrite_config_for_svd(config: dict, rank: int) -> dict:
    config = dict(config)
    config["r"] = rank
    # lora_alpha = rank makes the effective scale exactly 1, so the emitted
    # B @ A directly reconstructs the truncated delta-W.
    config["lora_alpha"] = rank
    config["use_dora"] = False
    config["use_rslora"] = False
    return config


def _compute_lora_scaling(config: dict) -> float:
    rank = int(config["r"])
    alpha = float(config.get("lora_alpha", rank))
    if config.get("use_rslora", False):
        return alpha / math.sqrt(rank)
    return alpha / rank


def _convert_dora_tensors_to_plain_lora(
    tensors: Dict[str, torch.Tensor],
    config: dict,
    base_weight_lookup: Callable[[str], torch.Tensor],
    rank: int,
) -> Dict[str, torch.Tensor]:
    """Tensor-level DoRA -> plain-LoRA conversion (see `svd_convert` docstring)."""
    scaling = _compute_lora_scaling(config=config)
    modules = _group_tensors_by_module(tensors=tensors)
    converted: Dict[str, torch.Tensor] = {}
    for module_path, module_tensors in modules.items():
        lora_a = module_tensors.get("lora_A")
        lora_b = module_tensors.get("lora_B")
        if lora_a is None or lora_b is None:
            raise AdapterNotServableError(
                f"Adapter module {module_path} is missing lora_A/lora_B "
                "tensors - cannot run SVD conversion."
            )
        magnitude = module_tensors.get("magnitude")
        base_weight = base_weight_lookup(module_path)
        merged = _dora_merged_weight(
            base_weight=base_weight.float(),
            lora_a=lora_a.float(),
            lora_b=lora_b.float(),
            magnitude=None if magnitude is None else magnitude.float(),
            scaling=scaling,
        )
        delta = merged - base_weight.float()
        new_a, new_b = _svd_truncate(delta=delta, rank=rank)
        dtype = lora_a.dtype
        converted[f"{module_path}.lora_A.weight"] = new_a.to(dtype).contiguous()
        converted[f"{module_path}.lora_B.weight"] = new_b.to(dtype).contiguous()
    return converted


def _dora_merged_weight(
    base_weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    magnitude: Optional[torch.Tensor],
    scaling: float,
) -> torch.Tensor:
    """Computes the merged weight for one DoRA module (PEFT merge semantics).

    `base_weight` has shape (out, in), `lora_a` (r, in), `lora_b` (out, r),
    `magnitude` (out,). When `magnitude` is None the module is treated as
    plain LoRA (merged = W0 + scale * B @ A).
    """
    candidate = base_weight + scaling * (lora_b @ lora_a)
    if magnitude is None:
        return candidate
    # PEFT computes the per-output-row L2 norm (torch.linalg.norm(..., dim=1))
    # of the candidate matrix - "column-wise" w.r.t. W^T as in the DoRA paper.
    column_norm = torch.linalg.norm(candidate, dim=1)
    dora_factor = (magnitude.reshape(-1) / column_norm).view(-1, 1)
    return dora_factor * candidate


def _svd_truncate(delta: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """SVD-truncates `delta` to `rank`, returning (lora_A, lora_B).

    With the emitted config's effective scale of 1, `lora_B @ lora_A`
    reconstructs the rank-truncated `delta`. Singular values are split evenly
    (sqrt) between the two factors for numerical balance.
    """
    u, s, vh = torch.linalg.svd(delta, full_matrices=False)
    rank = min(rank, s.shape[0])
    sqrt_s = torch.sqrt(s[:rank])
    lora_b = u[:, :rank] * sqrt_s.unsqueeze(0)
    lora_a = sqrt_s.unsqueeze(1) * vh[:rank, :]
    return lora_a, lora_b


def _group_tensors_by_module(
    tensors: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, torch.Tensor]]:
    modules: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in tensors.items():
        module_path = extract_module_path(key)
        if module_path is None:
            raise AdapterNotServableError(
                f"Unrecognised adapter tensor key: {key} - cannot run SVD "
                "conversion."
            )
        entry = modules.setdefault(module_path, {})
        if ".lora_A." in key:
            entry["lora_A"] = tensor
        elif ".lora_B." in key:
            entry["lora_B"] = tensor
        elif _MAGNITUDE_MARKER in key:
            # Saved either as `...lora_magnitude_vector` (older PEFT) or
            # `...lora_magnitude_vector.weight` (newer PEFT).
            entry["magnitude"] = tensor
    return modules


def _build_base_weight_lookup(base_dir: str) -> Callable[[str], torch.Tensor]:
    """Builds a lookup from adapter module paths to base model weights.

    Handles the `model.layers.` vs `model.language_model.layers.` naming
    quirk by trying both candidates, and resolves sharded checkpoints via
    `model.safetensors.index.json` when present.
    """
    index_path = os.path.join(base_dir, "model.safetensors.index.json")
    key_to_file: Dict[str, str] = {}
    if os.path.isfile(index_path):
        with open(index_path) as f:
            weight_map = json.load(f).get("weight_map", {})
        key_to_file = {
            key: os.path.join(base_dir, file_name)
            for key, file_name in weight_map.items()
        }
    else:
        from safetensors import safe_open

        for file_name in sorted(os.listdir(base_dir)):
            if not file_name.endswith(".safetensors"):
                continue
            file_path = os.path.join(base_dir, file_name)
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    key_to_file[key] = file_path

    def lookup(module_path: str) -> torch.Tensor:
        for candidate in _base_weight_key_candidates(module_path=module_path):
            file_path = key_to_file.get(candidate)
            if file_path is None:
                continue
            from safetensors import safe_open

            with safe_open(file_path, framework="pt", device="cpu") as f:
                return f.get_tensor(candidate)
        raise AdapterNotServableError(
            f"Could not locate base weight for adapter module {module_path} "
            f"in {base_dir}."
        )

    return lookup


def _base_weight_key_candidates(module_path: str) -> List[str]:
    core = module_path
    if core.startswith(PEFT_KEY_PREFIX):
        core = core[len(PEFT_KEY_PREFIX) :]
    candidates = [f"{core}.weight"]
    swapped = re.sub(r"^model\.layers\.", "model.language_model.layers.", core)
    if swapped != core:
        candidates.append(f"{swapped}.weight")
    swapped_back = re.sub(r"^model\.language_model\.layers\.", "model.layers.", core)
    if swapped_back != core:
        candidates.append(f"{swapped_back}.weight")
    return candidates


def _sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
