import json
import os

import pytest
import torch
from safetensors.torch import load_file

from inference.models.vllm_proxy.adapter_patch import (
    BASE_MODEL_CHECK_MATCH,
    BASE_MODEL_CHECK_SKIPPED,
    _dora_merged_weight,
    normalize_base_model_reference,
    patch_adapter,
    remap_adapter_weight_key,
    svd_convert,
)
from inference.models.vllm_proxy.config import DEFAULT_VLLM_ADAPTER_KEY_TEMPLATE
from inference.models.vllm_proxy.errors import AdapterNotServableError
from tests.inference.unit_tests.models.vllm_proxy.common import (
    BASE_WEIGHT_KEY,
    IN_FEATURES,
    LANGUAGE_MODULE_PATH,
    LORA_ALPHA,
    LORA_RANK,
    OUT_FEATURES,
    VISION_MODULE_PATH,
    build_adapter_config,
    build_adapter_tensors,
    write_adapter_package,
    write_base_package,
)


class TestRemapAdapterWeightKey:
    def test_language_layers_key_is_remapped_to_language_model_path(self) -> None:
        # given
        key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"

        # when
        result = remap_adapter_weight_key(
            key=key, key_template=DEFAULT_VLLM_ADAPTER_KEY_TEMPLATE
        )

        # then
        assert result == (
            "base_model.model.model.language_model.layers.0"
            ".self_attn.q_proj.lora_A.weight"
        )

    def test_already_remapped_key_is_idempotent(self) -> None:
        # given
        key = (
            "base_model.model.model.language_model.layers.0"
            ".self_attn.q_proj.lora_A.weight"
        )

        # when
        result = remap_adapter_weight_key(
            key=key, key_template=DEFAULT_VLLM_ADAPTER_KEY_TEMPLATE
        )

        # then
        assert result == key

    def test_key_without_peft_prefix_is_remapped_into_template(self) -> None:
        # given
        key = "model.layers.3.mlp.gate_proj.lora_B.weight"

        # when
        result = remap_adapter_weight_key(
            key=key, key_template=DEFAULT_VLLM_ADAPTER_KEY_TEMPLATE
        )

        # then
        assert result == (
            "base_model.model.model.language_model.layers.3"
            ".mlp.gate_proj.lora_B.weight"
        )

    def test_vision_key_is_left_unchanged(self) -> None:
        # given
        key = "base_model.model.model.visual.blocks.0.attn.qkv.lora_A.weight"

        # when
        result = remap_adapter_weight_key(
            key=key, key_template=DEFAULT_VLLM_ADAPTER_KEY_TEMPLATE
        )

        # then
        assert result == key

    def test_custom_template_is_applied(self) -> None:
        # given
        key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"

        # when
        result = remap_adapter_weight_key(
            key=key, key_template="language_model.layers.{suffix}"
        )

        # then
        assert result == "language_model.layers.0.self_attn.q_proj.lora_A.weight"


class TestPatchAdapterDoraPolicies:
    def test_dora_adapter_is_rejected_under_reject_policy(self, tmp_path) -> None:
        # given
        src_dir = write_adapter_package(
            target_dir=str(tmp_path / "src"),
            config=build_adapter_config(use_dora=True),
            tensors=build_adapter_tensors(use_dora=True),
        )

        # when / then
        with pytest.raises(AdapterNotServableError):
            patch_adapter(
                src_dir=src_dir, dst_dir=str(tmp_path / "dst"), policy="reject"
            )

    def test_strip_policy_drops_magnitude_vectors_and_disables_dora(
        self, tmp_path
    ) -> None:
        # given
        src_dir = write_adapter_package(
            target_dir=str(tmp_path / "src"),
            config=build_adapter_config(use_dora=True),
            tensors=build_adapter_tensors(use_dora=True),
        )
        dst_dir = str(tmp_path / "dst")

        # when
        report = patch_adapter(src_dir=src_dir, dst_dir=dst_dir, policy="strip")

        # then
        patched_tensors = load_file(os.path.join(dst_dir, "adapter_model.safetensors"))
        assert not any("lora_magnitude_vector" in key for key in patched_tensors)
        with open(os.path.join(dst_dir, "adapter_config.json")) as f:
            patched_config = json.load(f)
        assert patched_config["use_dora"] is False
        assert len(report.dropped_magnitude_keys) == 1

    def test_plain_lora_adapter_passes_under_reject_policy(self, tmp_path) -> None:
        # given
        src_dir = write_adapter_package(
            target_dir=str(tmp_path / "src"),
            config=build_adapter_config(use_dora=False),
            tensors=build_adapter_tensors(use_dora=False),
        )
        dst_dir = str(tmp_path / "dst")

        # when
        report = patch_adapter(src_dir=src_dir, dst_dir=dst_dir, policy="reject")

        # then
        assert os.path.isfile(os.path.join(dst_dir, "patch_report.json"))
        assert report.remapped_keys == 2

    def test_patch_adapter_publishes_complete_directory(self, tmp_path) -> None:
        # given - an existing output dir must never be observed half-written.
        src_dir = write_adapter_package(target_dir=str(tmp_path / "src"))
        dst_path = tmp_path / "dst"
        dst_path.mkdir()
        (dst_path / "stale-file").write_text("old")

        # when
        report = patch_adapter(src_dir=src_dir, dst_dir=str(dst_path))

        # then
        assert report.dst_dir == str(dst_path)
        assert not (dst_path / "stale-file").exists()
        assert (dst_path / "adapter_model.safetensors").is_file()
        assert (dst_path / "adapter_config.json").is_file()
        assert (dst_path / "patch_report.json").is_file()
        assert list(tmp_path.glob(".dst-*")) == []

    def test_unknown_policy_raises(self, tmp_path) -> None:
        # given
        src_dir = write_adapter_package(target_dir=str(tmp_path / "src"))

        # when / then
        with pytest.raises(ValueError):
            patch_adapter(
                src_dir=src_dir, dst_dir=str(tmp_path / "dst"), policy="invalid"
            )


class TestPatchAdapterValidation:
    def test_modules_to_save_raises(self, tmp_path) -> None:
        # given
        src_dir = write_adapter_package(
            target_dir=str(tmp_path / "src"),
            config=build_adapter_config(modules_to_save=["lm_head"]),
        )

        # when / then
        with pytest.raises(AdapterNotServableError):
            patch_adapter(src_dir=src_dir, dst_dir=str(tmp_path / "dst"))

    def test_rank_above_max_raises(self, tmp_path) -> None:
        # given
        src_dir = write_adapter_package(target_dir=str(tmp_path / "src"))

        # when / then
        with pytest.raises(AdapterNotServableError):
            patch_adapter(
                src_dir=src_dir,
                dst_dir=str(tmp_path / "dst"),
                max_lora_rank=LORA_RANK - 1,
            )


class TestNormalizeBaseModelReference:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("qwen/qwen3_5-2b", "qwen352b"),
            ("qwen3_5-0.8b", "qwen3508b"),
            ("Qwen/Qwen3-VL-2B-Instruct", "qwen3vl2binstruct"),
            ("qwen3vl-2b-instruct", "qwen3vl2binstruct"),
            ("  Qwen3.5-0.8B  ", "qwen3508b"),
        ],
    )
    def test_normalization(self, value: str, expected: str) -> None:
        assert normalize_base_model_reference(value) == expected


class TestBaseModelCrossCheck:
    """Pre-flight cross-check of adapter_config.json's base against the pool.

    Guards against registry metadata bugs (incident 2026-06-10:
    image-text/223 recorded as 0.8b-peft, adapter trained on qwen3_5-2b).
    """

    def test_mismatched_base_raises_naming_both_values(
        self, monkeypatch, tmp_path
    ) -> None:
        # given - served pool is qwen3_5-0.8b, adapter declares qwen/qwen3_5-2b
        monkeypatch.setenv("VLLM_SERVED_BASE_VARIANT", "qwen3_5-0.8b")
        src_dir = write_adapter_package(
            target_dir=str(tmp_path / "src"),
            config=build_adapter_config(base_model_name_or_path="qwen/qwen3_5-2b"),
        )

        # when / then
        with pytest.raises(AdapterNotServableError) as error:
            patch_adapter(
                src_dir=src_dir,
                dst_dir=str(tmp_path / "dst"),
                model_id="image-text/223",
            )
        message = str(error.value)
        assert "qwen/qwen3_5-2b" in message
        assert "qwen3_5-0.8b" in message
        assert "image-text/223" in message
        assert "registry" in message

    def test_matching_base_passes_and_is_recorded_in_patch_report(
        self, monkeypatch, tmp_path
    ) -> None:
        # given - org-prefixed config value matching the served variant
        monkeypatch.setenv("VLLM_SERVED_BASE_VARIANT", "qwen3_5-0.8b")
        src_dir = write_adapter_package(
            target_dir=str(tmp_path / "src"),
            config=build_adapter_config(base_model_name_or_path="qwen/qwen3_5-0.8b"),
        )
        dst_dir = str(tmp_path / "dst")

        # when
        report = patch_adapter(src_dir=src_dir, dst_dir=dst_dir)

        # then
        assert report.base_model_check == BASE_MODEL_CHECK_MATCH
        assert report.base_model_name_or_path == "qwen/qwen3_5-0.8b"
        with open(os.path.join(dst_dir, "patch_report.json")) as f:
            persisted = json.load(f)
        assert persisted["base_model_check"] == BASE_MODEL_CHECK_MATCH
        assert persisted["base_model_name_or_path"] == "qwen/qwen3_5-0.8b"

    def test_base_matching_served_base_name_passes(self, monkeypatch, tmp_path) -> None:
        # given - HF-style reference matches VLLM_SERVED_BASE_NAME (not the
        # variant), with different separators/casing
        monkeypatch.setenv("VLLM_SERVED_BASE_VARIANT", "qwen3vl-2b")
        monkeypatch.setenv("VLLM_SERVED_BASE_NAME", "qwen3vl-2b-instruct")
        src_dir = write_adapter_package(
            target_dir=str(tmp_path / "src"),
            config=build_adapter_config(
                base_model_name_or_path="Qwen/Qwen3-VL-2B-Instruct"
            ),
        )

        # when
        report = patch_adapter(src_dir=src_dir, dst_dir=str(tmp_path / "dst"))

        # then
        assert report.base_model_check == BASE_MODEL_CHECK_MATCH

    def test_missing_base_model_field_skips_check_with_warning(
        self, monkeypatch, tmp_path
    ) -> None:
        # given - fixture config carries no base_model_name_or_path
        from unittest.mock import MagicMock

        from inference.models.vllm_proxy import adapter_patch as adapter_patch_module

        logger_mock = MagicMock()
        monkeypatch.setattr(adapter_patch_module, "logger", logger_mock)
        src_dir = write_adapter_package(target_dir=str(tmp_path / "src"))
        dst_dir = str(tmp_path / "dst")

        # when
        report = patch_adapter(src_dir=src_dir, dst_dir=dst_dir)

        # then
        assert report.base_model_check == BASE_MODEL_CHECK_SKIPPED
        assert report.base_model_name_or_path is None
        logger_mock.warning.assert_called_once()
        with open(os.path.join(dst_dir, "patch_report.json")) as f:
            persisted = json.load(f)
        assert persisted["base_model_check"] == BASE_MODEL_CHECK_SKIPPED


class TestPatchAdapterVisionFiltering:
    def test_untrained_vision_tensors_are_dropped_and_recorded(self, tmp_path) -> None:
        # given
        src_dir = write_adapter_package(
            target_dir=str(tmp_path / "src"),
            config=build_adapter_config(),
            tensors=build_adapter_tensors(
                include_vision=True, vision_lora_b_nonzero=False
            ),
        )
        dst_dir = str(tmp_path / "dst")

        # when
        report = patch_adapter(src_dir=src_dir, dst_dir=dst_dir)

        # then
        assert sorted(report.dropped_vision_keys) == [
            f"{VISION_MODULE_PATH}.lora_A.weight",
            f"{VISION_MODULE_PATH}.lora_B.weight",
        ]
        patched_tensors = load_file(os.path.join(dst_dir, "adapter_model.safetensors"))
        assert not any("visual" in key for key in patched_tensors)

    def test_trained_vision_lora_b_raises(self, tmp_path) -> None:
        # given
        src_dir = write_adapter_package(
            target_dir=str(tmp_path / "src"),
            config=build_adapter_config(),
            tensors=build_adapter_tensors(
                include_vision=True, vision_lora_b_nonzero=True
            ),
        )

        # when / then
        with pytest.raises(AdapterNotServableError):
            patch_adapter(src_dir=src_dir, dst_dir=str(tmp_path / "dst"))

    def test_trained_vision_lora_b_passes_with_high_threshold(self, tmp_path) -> None:
        # given
        src_dir = write_adapter_package(
            target_dir=str(tmp_path / "src"),
            config=build_adapter_config(),
            tensors=build_adapter_tensors(
                include_vision=True, vision_lora_b_nonzero=True
            ),
        )

        # when
        report = patch_adapter(
            src_dir=src_dir,
            dst_dir=str(tmp_path / "dst"),
            vision_norm_threshold=1e6,
        )

        # then
        assert len(report.dropped_vision_keys) == 2


class TestPatchAdapterConfigRewrite:
    def test_target_modules_are_intersected_and_unsupported_fields_stripped(
        self, tmp_path
    ) -> None:
        # given
        src_dir = write_adapter_package(
            target_dir=str(tmp_path / "src"),
            config=build_adapter_config(
                target_modules=["q_proj", "v_proj", "qkv", "lm_head"],
                use_rslora=True,
            ),
        )
        dst_dir = str(tmp_path / "dst")

        # when
        _ = patch_adapter(src_dir=src_dir, dst_dir=dst_dir)

        # then
        with open(os.path.join(dst_dir, "adapter_config.json")) as f:
            patched_config = json.load(f)
        assert patched_config["target_modules"] == ["q_proj", "v_proj"]
        assert "eva_config" not in patched_config
        assert "lora_bias" not in patched_config
        assert "exclude_modules" not in patched_config
        assert patched_config["use_rslora"] is True

    def test_no_supported_target_modules_raises(self, tmp_path) -> None:
        # given
        src_dir = write_adapter_package(
            target_dir=str(tmp_path / "src"),
            config=build_adapter_config(target_modules=["qkv", "proj_out"]),
        )

        # when / then
        with pytest.raises(AdapterNotServableError):
            patch_adapter(src_dir=src_dir, dst_dir=str(tmp_path / "dst"))


class TestSVDConversion:
    def test_svd_round_trip_reconstructs_delta_when_rank_is_sufficient(
        self, tmp_path
    ) -> None:
        # given - a DoRA adapter whose magnitude equals the column norm of
        # W0 + scale * B @ A, so delta-W is exactly rank LORA_RANK and the
        # SVD truncation is lossless.
        base_dir = str(tmp_path / "base")
        base_weight = write_base_package(target_dir=base_dir)
        tensors = build_adapter_tensors(use_dora=True)
        lora_a = tensors[f"{LANGUAGE_MODULE_PATH}.lora_A.weight"]
        lora_b = tensors[f"{LANGUAGE_MODULE_PATH}.lora_B.weight"]
        scaling = LORA_ALPHA / LORA_RANK
        candidate = base_weight + scaling * (lora_b @ lora_a)
        tensors[f"{LANGUAGE_MODULE_PATH}.lora_magnitude_vector.weight"] = (
            torch.linalg.norm(candidate, dim=1)
        )
        adapter_dir = write_adapter_package(
            target_dir=str(tmp_path / "adapter"),
            config=build_adapter_config(use_dora=True),
            tensors=tensors,
        )
        dst_dir = str(tmp_path / "dst")
        expected_delta = scaling * (lora_b @ lora_a)

        # when
        svd_convert(
            base_dir=base_dir,
            adapter_dir=adapter_dir,
            dst_dir=dst_dir,
            rank=LORA_RANK,
        )

        # then
        converted = load_file(os.path.join(dst_dir, "adapter_model.safetensors"))
        new_a = converted[f"{LANGUAGE_MODULE_PATH}.lora_A.weight"]
        new_b = converted[f"{LANGUAGE_MODULE_PATH}.lora_B.weight"]
        # emitted config has lora_alpha == r == rank, so effective scale is 1
        reconstructed = new_b @ new_a
        assert torch.allclose(reconstructed, expected_delta, atol=1e-4)
        with open(os.path.join(dst_dir, "adapter_config.json")) as f:
            converted_config = json.load(f)
        assert converted_config["r"] == LORA_RANK
        assert converted_config["lora_alpha"] == LORA_RANK
        assert converted_config["use_dora"] is False
        assert converted_config["use_rslora"] is False

    def test_svd_with_full_rank_reconstructs_general_dora_delta(self, tmp_path) -> None:
        # given - generic magnitude vector (delta-W is full-rank), but the
        # SVD rank equals min(out, in) so reconstruction is still exact.
        base_dir = str(tmp_path / "base")
        base_weight = write_base_package(target_dir=base_dir)
        tensors = build_adapter_tensors(use_dora=True)
        adapter_dir = write_adapter_package(
            target_dir=str(tmp_path / "adapter"),
            config=build_adapter_config(use_dora=True),
            tensors=tensors,
        )
        dst_dir = str(tmp_path / "dst")
        full_rank = min(OUT_FEATURES, IN_FEATURES)
        expected_merged = _dora_merged_weight(
            base_weight=base_weight,
            lora_a=tensors[f"{LANGUAGE_MODULE_PATH}.lora_A.weight"],
            lora_b=tensors[f"{LANGUAGE_MODULE_PATH}.lora_B.weight"],
            magnitude=tensors[f"{LANGUAGE_MODULE_PATH}.lora_magnitude_vector.weight"],
            scaling=LORA_ALPHA / LORA_RANK,
        )

        # when
        svd_convert(
            base_dir=base_dir,
            adapter_dir=adapter_dir,
            dst_dir=dst_dir,
            rank=full_rank,
        )

        # then
        converted = load_file(os.path.join(dst_dir, "adapter_model.safetensors"))
        reconstructed = (
            converted[f"{LANGUAGE_MODULE_PATH}.lora_B.weight"]
            @ converted[f"{LANGUAGE_MODULE_PATH}.lora_A.weight"]
        )
        assert torch.allclose(base_weight + reconstructed, expected_merged, atol=1e-4)

    def test_patch_adapter_with_svd_policy_runs_end_to_end(self, tmp_path) -> None:
        # given
        base_dir = str(tmp_path / "base")
        write_base_package(target_dir=base_dir)
        src_dir = write_adapter_package(
            target_dir=str(tmp_path / "src"),
            config=build_adapter_config(use_dora=True),
            tensors=build_adapter_tensors(use_dora=True),
        )
        dst_dir = str(tmp_path / "dst")

        # when
        report = patch_adapter(
            src_dir=src_dir, dst_dir=dst_dir, policy="svd", base_dir=base_dir
        )

        # then
        assert report.svd_rank == LORA_RANK
        patched_tensors = load_file(os.path.join(dst_dir, "adapter_model.safetensors"))
        assert sorted(patched_tensors.keys()) == [
            "base_model.model.model.language_model.layers.0"
            ".self_attn.q_proj.lora_A.weight",
            "base_model.model.model.language_model.layers.0"
            ".self_attn.q_proj.lora_B.weight",
        ]

    def test_svd_policy_without_base_dir_raises(self, tmp_path) -> None:
        # given
        src_dir = write_adapter_package(
            target_dir=str(tmp_path / "src"),
            config=build_adapter_config(use_dora=True),
            tensors=build_adapter_tensors(use_dora=True),
        )

        # when / then
        with pytest.raises(AdapterNotServableError):
            patch_adapter(src_dir=src_dir, dst_dir=str(tmp_path / "dst"), policy="svd")


class TestDoraMergeMathAgainstPeft:
    def test_merged_weight_matches_peft_merge_and_unload(self) -> None:
        # given - this is the strongest check of the DoRA merge math: build a
        # tiny PeftModel with DoRA, randomize its parameters, and compare
        # peft's own merge_and_unload result against _dora_merged_weight.
        peft = pytest.importorskip("peft")
        import torch.nn as nn

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)

            def forward(self, x):
                return self.lin(x)

        torch.manual_seed(3)
        lora_config = peft.LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=["lin"],
            use_dora=True,
            init_lora_weights=False,
            bias="none",
        )
        peft_model = peft.get_peft_model(TinyModel(), lora_config)
        layer = peft_model.base_model.model.lin
        with torch.no_grad():
            layer.lora_A["default"].weight.copy_(
                torch.randn_like(layer.lora_A["default"].weight)
            )
            layer.lora_B["default"].weight.copy_(
                torch.randn_like(layer.lora_B["default"].weight)
            )
            layer.lora_magnitude_vector["default"].weight.copy_(
                torch.rand(OUT_FEATURES) + 0.5
            )
        base_weight = layer.base_layer.weight.detach().clone()
        lora_a = layer.lora_A["default"].weight.detach().clone()
        lora_b = layer.lora_B["default"].weight.detach().clone()
        magnitude = layer.lora_magnitude_vector["default"].weight.detach().clone()

        # when
        merged_model = peft_model.merge_and_unload()
        peft_merged_weight = merged_model.lin.weight.detach()
        our_merged_weight = _dora_merged_weight(
            base_weight=base_weight,
            lora_a=lora_a,
            lora_b=lora_b,
            magnitude=magnitude,
            scaling=LORA_ALPHA / LORA_RANK,
        )

        # then
        assert torch.allclose(our_merged_weight, peft_merged_weight, atol=1e-5)


def test_base_weight_lookup_handles_layers_naming_quirk(tmp_path) -> None:
    # given - adapter keys say `model.layers.` but the base checkpoint names
    # the module `model.language_model.layers.` (the qwen3_5 VL quirk).
    from inference.models.vllm_proxy.adapter_patch import _build_base_weight_lookup

    base_dir = str(tmp_path / "base")
    base_weight = write_base_package(target_dir=base_dir)
    lookup = _build_base_weight_lookup(base_dir=base_dir)

    # when
    resolved = lookup(LANGUAGE_MODULE_PATH)

    # then
    assert torch.equal(resolved, base_weight)
    assert BASE_WEIGHT_KEY.startswith("model.language_model.layers.")
