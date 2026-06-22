"""Resolution + registration of Roboflow models against the vLLM sidecar.

Maps a Roboflow `model_id` to the name it is served under in vLLM:

- Base model ids (matching `VLLM_SERVED_BASE_VARIANT` or
  `VLLM_SERVED_BASE_NAME`) require no registration - vLLM already serves the
  base model.
- Registry metadata is ADVISORY for fine-tunes: `modelArchitecture` gates
  pre-download (clearly-unsupported families like florence never download),
  but `modelVariant` is sometimes misregistered (e.g. `image-text/223` was
  recorded as `0.8b-peft` while its adapter_config.json declared
  `qwen/qwen3_5-2b`), so a variant mismatch against
  `VLLM_SERVED_BASE_VARIANT` only logs a warning. The AUTHORITATIVE
  accept/reject is the adapter's own `adapter_config.json`
  `base_model_name_or_path`, cross-checked against the served base inside
  `patch_adapter` (`cross_check_base_model`).
- Fine-tunes resolve their model package via the Roboflow weights provider
  (the api_key is passed through so registry-side access control applies),
  download ONLY the adapter artifacts, run `patch_adapter`, and register the
  patched adapter with vLLM's dynamic LoRA endpoint.

Cache keys combine model_id + package_id + a content digest because package
ids are NOT unique per model version.

Registration semantics (multi-process): several gunicorn workers each hold
their own AdapterManager but instruct ONE shared vLLM engine, so the
per-process map can go stale (worker recycles, vLLM restarts). The map is
therefore only trusted to skip the expensive download/patch work; the cheap,
idempotent `load_lora_adapter` call is ALWAYS re-issued so vLLM remains the
source of truth. The manager never unloads adapters: vLLM's own
`--max-cpu-loras` LRU bounds host memory and refills from disk, and disk
growth is handled outside (pod recycle / future GC).
`VLLM_MAX_REGISTERED_ADAPTERS` is a warn-only threshold.
"""

import hashlib
import os
import re
import threading
from dataclasses import dataclass
from typing import Dict, Optional

from filelock import FileLock

from inference.core import logger
from inference.models.vllm_proxy.adapter_patch import (
    ADAPTER_CONFIG_FILE,
    ADAPTER_WEIGHTS_FILE,
    BASE_MODEL_CHECK_MATCH,
    PATCH_REPORT_FILE,
    patch_adapter,
)
from inference.models.vllm_proxy.config import (
    get_model_cache_dir,
    get_vllm_dora_policy,
    get_vllm_max_registered_adapters,
    get_vllm_served_base_name,
    get_vllm_served_base_variant,
)
from inference.models.vllm_proxy.errors import (
    AdapterNotServableError,
    NotServableOnVLLMError,
    VLLMHTTPError,
)
from inference.models.vllm_proxy.vllm_client import VLLMClient
from inference_models.models.auto_loaders.entities import BackendType
from inference_models.utils.download import download_files_to_directory
from inference_models.weights_providers.core import get_model_from_provider

ADAPTERS_CACHE_SUBDIR = "vllm-adapters"
ADAPTER_CACHE_LOCK_FILE = ".registration.lock"
ADAPTER_CACHE_LOCK_TIMEOUT_SECONDS = 120
BASE_PACKAGE_DIR_PREFIX = "base/"
SUPPORTED_MODEL_ARCHITECTURES = ("qwen3_5", "qwen3vl")
PATCHED_ADAPTER_REQUIRED_FILES = (
    ADAPTER_CONFIG_FILE,
    ADAPTER_WEIGHTS_FILE,
    PATCH_REPORT_FILE,
)

# Registry fine-tune variants carry this suffix; a fine-tune of base X is
# servable on a pool whose vLLM container serves X.
PEFT_VARIANT_SUFFIX = "-peft"


@dataclass
class AdapterRegistration:
    """Bookkeeping record for an adapter registered with vLLM."""

    served_name: str
    model_id: str
    package_id: str
    content_digest: str
    source_dir: str
    patched_dir: str


def sanitize_for_slug(value: str) -> str:
    """Lowercases and reduces a value to `[a-z0-9-]` for cache/served names."""
    slug = re.sub(r"[^a-z0-9_-]+", "-", value.lower())
    slug = re.sub(r"[_-]{2,}", "-", slug)
    return slug.strip("-")


def normalize_base_variant(
    model_architecture: Optional[str], model_variant: Optional[str]
) -> Optional[str]:
    """Maps registry metadata to the canonical `<architecture>-<variant>` form.

    `VLLM_SERVED_BASE_VARIANT` is configured as
    `<architecture>-<variant-with-peft-suffix-stripped>` (e.g.
    `qwen3_5-0.8b`, `qwen3vl-2b`). Registry metadata is inconsistent across
    families: qwen3_5 models report variants that already carry the
    architecture prefix (`qwen3_5-0.8b`), while qwen3vl fine-tunes report
    bare variants (`2b-peft`). Normalisation: lowercase, strip a trailing
    `-peft` (a fine-tune of base X is servable on the pool serving X), then
    prefix with `<architecture>-` unless the variant already starts with it.

    Returns None when either field is missing.
    """
    if not model_architecture or not model_variant:
        return None
    architecture = model_architecture.strip().lower()
    variant = model_variant.strip().lower()
    if variant.endswith(PEFT_VARIANT_SUFFIX):
        variant = variant[: -len(PEFT_VARIANT_SUFFIX)]
    if variant == architecture or variant.startswith(f"{architecture}-"):
        return variant
    return f"{architecture}-{variant}"


class AdapterManager:
    """Thread-safe, idempotent adapter registration (registration-only).

    Never unloads adapters from vLLM - the engine's own `--max-cpu-loras`
    LRU bounds memory. The local map only short-circuits download/patch
    work; the idempotent vLLM registration call is always re-issued.
    """

    def __init__(self, client: Optional[VLLMClient] = None):
        self._client = client or VLLMClient()
        self._lock = threading.Lock()
        self._registered: Dict[str, AdapterRegistration] = {}

    @property
    def client(self) -> VLLMClient:
        return self._client

    def resolve_and_register(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        weights_provider_extra_headers: Optional[Dict[str, str]] = None,
    ) -> str:
        """Resolves `model_id` to the name it is served under in vLLM.

        Base model ids return the served base name without registration.
        Fine-tunes are downloaded, patched and registered. Idempotent, with
        a multi-process twist: when the slug is already in the local map and
        its patched dir is still on disk, the expensive download/patch work
        is skipped but the cheap `load_lora_adapter` call is ALWAYS
        re-issued - the per-process map may be stale relative to the shared
        vLLM engine (another worker's actions, vLLM restarts), and vLLM
        treats re-registration as success.

        Registry `modelArchitecture` gates pre-download (unsupported families
        are rejected before any artifact download); registry `modelVariant`
        is advisory only - a mismatch against the served base logs a warning
        and defers to the adapter's own `adapter_config.json`
        `base_model_name_or_path`, cross-checked inside `patch_adapter`.

        Raises:
            NotServableOnVLLMError: If the model's architecture is not
                supported by the vLLM proxy, or no HF package is exposed.
            AdapterNotServableError: If the adapter's declared base does not
                match the served base, or it cannot be patched into a
                vLLM-servable form.
        """
        served_base_variant = get_vllm_served_base_variant()
        served_base_name = get_vllm_served_base_name()
        # Base-model ids short-circuit: ids equal to the configured variant
        # (qwen3_5-0.8b) or to the served base name (qwen3vl-2b-instruct)
        # require no registration - vLLM already serves the base model.
        if model_id.lower() in {
            served_base_variant.lower(),
            served_base_name.lower(),
        }:
            return served_base_name
        metadata = get_model_from_provider(
            model_id=model_id,
            provider="roboflow",
            api_key=api_key,
            weights_provider_extra_headers=weights_provider_extra_headers,
        )
        if (
            not (metadata.model_architecture or "")
            .lower()
            .startswith(SUPPORTED_MODEL_ARCHITECTURES)
        ):
            raise NotServableOnVLLMError(
                f"Model {model_id} has architecture "
                f"{metadata.model_architecture!r} which is not servable by the "
                f"vLLM proxy (expected one of {SUPPORTED_MODEL_ARCHITECTURES!r})."
            )
        normalized_variant = normalize_base_variant(
            model_architecture=metadata.model_architecture,
            model_variant=metadata.model_variant,
        )
        registry_variant_matches = normalized_variant == served_base_variant.lower()
        if not registry_variant_matches:
            # ADVISORY ONLY: registry modelVariant is sometimes misregistered
            # (e.g. image-text/223 recorded as 0.8b-peft while its
            # adapter_config.json declared qwen/qwen3_5-2b). The adapter's
            # own adapter_config.json is authoritative - the cross-check in
            # `patch_adapter` accepts/rejects after download.
            logger.warning(
                "Registry variant %r for model %s (architecture %r, variant "
                "%r) does not match served base %r - deferring to "
                "adapter_config.json.",
                normalized_variant,
                model_id,
                metadata.model_architecture,
                metadata.model_variant,
                served_base_variant,
            )
        package = self._select_model_package(model_id=model_id, metadata=metadata)
        adapter_files = [
            artefact
            for artefact in package.package_artefacts
            if not artefact.file_handle.startswith(BASE_PACKAGE_DIR_PREFIX)
        ]
        if not any(
            artefact.file_handle == ADAPTER_CONFIG_FILE for artefact in adapter_files
        ):
            # No adapter artifacts - this is a base-model package of the
            # served variant.
            return get_vllm_served_base_name()
        content_digest = self._compute_content_digest(adapter_files=adapter_files)
        slug = self._build_slug(
            model_id=metadata.model_id,
            package_id=package.package_id,
            content_digest=content_digest,
        )
        with self._lock:
            existing = self._registered.get(slug)
            if existing is not None and self._try_load_existing_registration(existing):
                # Skip ONLY the expensive download/patch work. The vLLM
                # registration call must still happen: this process's map
                # may be stale (shared engine, NUM_WORKERS>1) and the call
                # is idempotent and ~ms with files already on disk.
                return slug
            registration = self._download_patch_and_load(
                slug=slug,
                model_id=metadata.model_id,
                package_id=package.package_id,
                content_digest=content_digest,
                adapter_files=adapter_files,
                registry_variant=metadata.model_variant,
                registry_variant_matches=registry_variant_matches,
            )
            self._registered[slug] = registration
            self._warn_if_over_max_registered()
        return slug

    def invalidate(self, served_name: str) -> None:
        """Drops `served_name` from the local registration map.

        Used by the request-path self-heal when vLLM reports the adapter
        unknown despite local bookkeeping (vLLM restart, desync across
        gunicorn workers): the next `resolve_and_register` re-runs the full
        path (files already on disk make it near-instant).
        """
        with self._lock:
            self._registered.pop(served_name, None)

    def get_registration(self, served_name: str) -> Optional[AdapterRegistration]:
        with self._lock:
            return self._registered.get(served_name)

    def _select_model_package(self, model_id: str, metadata):
        hf_packages = [
            package
            for package in metadata.model_packages
            if package.backend is BackendType.HF
        ]
        if not hf_packages:
            raise NotServableOnVLLMError(
                f"Model {model_id} exposes no HF model package - the vLLM "
                "proxy requires HF (PEFT adapter) packages."
            )
        return hf_packages[0]

    @staticmethod
    def _compute_content_digest(adapter_files) -> str:
        """Digest over the adapter artifact hashes (no download required).

        Package ids are not unique per model version, so the digest
        disambiguates cache entries when package content changes.
        """
        digest = hashlib.sha256()
        for artefact in sorted(adapter_files, key=lambda a: a.file_handle):
            digest.update(artefact.file_handle.encode("utf-8"))
            digest.update((artefact.md5_hash or artefact.download_url).encode("utf-8"))
        return digest.hexdigest()[:8]

    @staticmethod
    def _build_slug(model_id: str, package_id: str, content_digest: str) -> str:
        return (
            f"{sanitize_for_slug(model_id)}-{sanitize_for_slug(package_id)}"
            f"-{content_digest}"
        )

    def _download_patch_and_load(
        self,
        slug: str,
        model_id: str,
        package_id: str,
        content_digest: str,
        adapter_files,
        registry_variant: Optional[str] = None,
        registry_variant_matches: bool = True,
    ) -> AdapterRegistration:
        adapter_cache_dir = os.path.join(
            get_model_cache_dir(), ADAPTERS_CACHE_SUBDIR, slug
        )
        source_dir = os.path.join(adapter_cache_dir, "src")
        patched_dir = os.path.join(adapter_cache_dir, "patched")
        os.makedirs(adapter_cache_dir, exist_ok=True)
        with self._adapter_cache_lock(adapter_cache_dir):
            if not self._patched_adapter_ready(patched_dir):
                dora_policy = get_vllm_dora_policy()
                if dora_policy == "svd":
                    raise NotServableOnVLLMError(
                        "VLLM_DORA_POLICY=svd is not supported in the runtime "
                        "adapter manager: the manager downloads adapter-only "
                        "artifacts and intentionally prunes base/ weights. Use "
                        "VLLM_DORA_POLICY=strip or reject at runtime, or run "
                        "offline SVD conversion with an explicit base_dir."
                    )
                os.makedirs(source_dir, exist_ok=True)
                download_files_to_directory(
                    target_dir=source_dir,
                    files_specs=[
                        (artefact.file_handle, artefact.download_url, artefact.md5_hash)
                        for artefact in adapter_files
                    ],
                    verbose=False,
                )
                report = patch_adapter(
                    src_dir=source_dir,
                    dst_dir=patched_dir,
                    policy=dora_policy,
                    model_id=model_id,
                    registry_variant=registry_variant,
                )
                if (
                    not registry_variant_matches
                    and report.base_model_check == BASE_MODEL_CHECK_MATCH
                ):
                    # Drift audit: the adapter is servable here (its own config
                    # matches the served base) but the registry disagrees - flag the
                    # registry record for correction.
                    logger.warning(
                        "Registry modelVariant misregistered for %s: registry says "
                        "%r, adapter declares %r.",
                        model_id,
                        registry_variant,
                        report.base_model_name_or_path,
                    )
            self._load_adapter_into_vllm(
                slug=slug, model_id=model_id, patched_dir=patched_dir
            )
        logger.info(
            "Registered LoRA adapter %s (model_id=%s, package_id=%s) with vLLM",
            slug,
            model_id,
            package_id,
        )
        return AdapterRegistration(
            served_name=slug,
            model_id=model_id,
            package_id=package_id,
            content_digest=content_digest,
            source_dir=source_dir,
            patched_dir=patched_dir,
        )

    def _try_load_existing_registration(
        self, registration: AdapterRegistration
    ) -> bool:
        adapter_cache_dir = os.path.dirname(registration.patched_dir)
        os.makedirs(adapter_cache_dir, exist_ok=True)
        with self._adapter_cache_lock(adapter_cache_dir):
            if not self._patched_adapter_ready(registration.patched_dir):
                return False
            self._load_adapter_into_vllm(
                slug=registration.served_name,
                model_id=registration.model_id,
                patched_dir=registration.patched_dir,
            )
            return True

    @staticmethod
    def _adapter_cache_lock(adapter_cache_dir: str) -> FileLock:
        return FileLock(
            os.path.join(adapter_cache_dir, ADAPTER_CACHE_LOCK_FILE),
            timeout=ADAPTER_CACHE_LOCK_TIMEOUT_SECONDS,
        )

    @staticmethod
    def _patched_adapter_ready(patched_dir: str) -> bool:
        return os.path.isdir(patched_dir) and all(
            os.path.isfile(os.path.join(patched_dir, file_name))
            for file_name in PATCHED_ADAPTER_REQUIRED_FILES
        )

    def _load_adapter_into_vllm(
        self, slug: str, model_id: str, patched_dir: str
    ) -> None:
        """Loads the patched adapter, surfacing 5xx load failures as typed errors.

        vLLM returns HTTP 500 when an adapter passes local validation but is
        rejected at load time (e.g. tensor-shape mismatch against the served
        base) - re-raised as `AdapterNotServableError` naming the adapter and
        excerpting vLLM's response so on-call sees the real cause instead of
        an opaque proxy 500. Connection errors (`VLLMConnectionError`) and
        non-5xx HTTP errors propagate unchanged: they indicate sidecar /
        request problems, not a broken adapter, and stay retryable.
        """
        try:
            self._client.load_lora_adapter(name=slug, path=patched_dir)
        except VLLMHTTPError as error:
            if error.status_code < 500:
                raise
            body_excerpt = (error.response_body or "").strip()[:500]
            raise AdapterNotServableError(
                f"vLLM rejected LoRA adapter {slug} (model_id={model_id}) at "
                f"load time with HTTP {error.status_code}. The adapter passed "
                f"local validation but could not be loaded into the served "
                f"base model - vLLM said: {body_excerpt!r}"
            ) from error

    def _warn_if_over_max_registered(self) -> None:
        """Warn-only threshold - the manager NEVER unloads adapters.

        With NUM_WORKERS>1 every gunicorn worker instructs the same shared
        vLLM engine; unloading from one worker's bookkeeping would yank
        adapters other workers still serve. vLLM's own `--max-cpu-loras`
        LRU bounds host memory and refills from disk; disk growth is
        handled outside (pod recycle / future GC).
        """
        max_registered = get_vllm_max_registered_adapters()
        if len(self._registered) > max_registered:
            logger.warning(
                "Registered LoRA adapter count %d exceeds "
                "VLLM_MAX_REGISTERED_ADAPTERS=%d. No adapter is unloaded "
                "(vLLM's --max-cpu-loras LRU bounds memory); consider "
                "recycling the pod if disk growth becomes a concern.",
                len(self._registered),
                max_registered,
            )


_ADAPTER_MANAGER: Optional[AdapterManager] = None
_ADAPTER_MANAGER_LOCK = threading.Lock()


def get_adapter_manager() -> AdapterManager:
    """Returns the process-wide AdapterManager singleton."""
    global _ADAPTER_MANAGER
    with _ADAPTER_MANAGER_LOCK:
        if _ADAPTER_MANAGER is None:
            _ADAPTER_MANAGER = AdapterManager()
        return _ADAPTER_MANAGER
