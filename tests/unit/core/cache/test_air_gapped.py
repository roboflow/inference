"""Tests for inference.core.cache.air_gapped scanning utilities."""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_model_type_json(
    cache_dir: str,
    model_id: str,
    metadata: dict,
) -> None:
    """Write a ``model_type.json`` marker inside *cache_dir/model_id*."""
    model_dir = os.path.join(cache_dir, model_id)
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "model_type.json"), "w") as fh:
        json.dump(metadata, fh)


def _make_block_spec(
    identifier: str,
    manifest_cls: type,
):
    """Build a minimal BlockSpecification-like object."""

    @dataclass
    class _FakeBlockSpec:
        block_source: str
        identifier: str
        block_class: Any
        manifest_class: Any

    return _FakeBlockSpec(
        block_source="test",
        identifier=identifier,
        block_class=MagicMock(),
        manifest_class=manifest_cls,
    )


# ---------------------------------------------------------------------------
# scan_cached_models
# ---------------------------------------------------------------------------


class TestScanTraditionalCache:
    """model_type.json written with ``project_task_type`` / ``model_type`` keys."""

    def test_scan_traditional_cache(self, tmp_path):
        from inference.core.cache.air_gapped import scan_cached_models

        cache = str(tmp_path)
        _write_model_type_json(
            cache,
            "my-workspace/my-project/3",
            {"project_task_type": "object-detection", "model_type": "yolov8n"},
        )

        result = scan_cached_models(cache)

        assert len(result) == 1
        m = result[0]
        assert m["model_id"] == "my-workspace/my-project/3"
        assert m["task_type"] == "object-detection"
        assert m["model_architecture"] == "yolov8n"
        assert m["is_foundation"] is False


class TestScanInferenceModelsCache:
    """model_type.json written with ``taskType`` / ``modelArchitecture`` keys."""

    def test_scan_inference_models_cache(self, tmp_path):
        from inference.core.cache.air_gapped import scan_cached_models

        cache = str(tmp_path)
        _write_model_type_json(
            cache,
            "coco/1",
            {"taskType": "instance-segmentation", "modelArchitecture": "yolact"},
        )

        result = scan_cached_models(cache)

        assert len(result) == 1
        m = result[0]
        assert m["model_id"] == "coco/1"
        assert m["task_type"] == "instance-segmentation"
        assert m["model_architecture"] == "yolact"
        assert m["is_foundation"] is False


class TestSkipNonModelDirs:
    """Ensure ``workflow/`` and ``_file_locks/`` are not traversed."""

    def test_skip_non_model_dirs(self, tmp_path):
        from inference.core.cache.air_gapped import scan_cached_models

        cache = str(tmp_path)

        # These should be skipped.
        _write_model_type_json(
            cache,
            "workflow/some-workflow",
            {"project_task_type": "object-detection", "model_type": "yolov8n"},
        )
        _write_model_type_json(
            cache,
            "_file_locks/lock-dir",
            {"project_task_type": "object-detection", "model_type": "yolov8n"},
        )

        # This should be found.
        _write_model_type_json(
            cache,
            "real-workspace/real-project/1",
            {"project_task_type": "classification", "model_type": "vit"},
        )

        result = scan_cached_models(cache)

        assert len(result) == 1
        assert result[0]["model_id"] == "real-workspace/real-project/1"


# ---------------------------------------------------------------------------
# Foundation model detection
# ---------------------------------------------------------------------------


class _DefaultManifestMixin:
    """Provide the base-class defaults that get_cached_foundation_models relies on."""

    @classmethod
    def get_air_gapped_availability(cls):
        from inference.core.workflows.prototypes.block import AirGappedAvailability

        return AirGappedAvailability(available=True)

    @classmethod
    def get_supported_model_variants(cls):
        return None

    @classmethod
    def get_compatible_task_types(cls):
        return None


class TestFoundationModelDetection:
    """Block with ``get_supported_model_variants()`` whose files exist."""

    def test_foundation_model_detected_when_cached(self, tmp_path):
        from inference.core.cache.air_gapped import get_cached_foundation_models

        cache = str(tmp_path)

        # Create a cache directory for the model variant with a file.
        variant_dir = os.path.join(cache, "foundation", "clip")
        os.makedirs(variant_dir, exist_ok=True)
        open(os.path.join(variant_dir, "weights.pt"), "w").close()

        class FakeManifest(_DefaultManifestMixin):
            @classmethod
            def get_supported_model_variants(cls):
                return ["foundation/clip"]

            @classmethod
            def model_json_schema(cls) -> dict:
                return {"name": "CLIP"}

        block = _make_block_spec("roboflow_core/clip@v1", FakeManifest)

        with patch("inference.core.cache.air_gapped.MODEL_CACHE_DIR", cache), patch(
            "inference.core.cache.air_gapped.USE_INFERENCE_MODELS", True
        ):
            result = get_cached_foundation_models(blocks=[block])

        assert len(result) == 1
        m = result[0]
        assert m["model_id"] == "foundation/clip"
        assert m["is_foundation"] is True
        assert m["name"] == "CLIP"


class TestFoundationModelMissing:
    """Block with ``get_supported_model_variants()`` whose files do NOT exist."""

    def test_foundation_model_not_detected_when_missing(self, tmp_path):
        from inference.core.cache.air_gapped import get_cached_foundation_models

        cache = str(tmp_path)

        class FakeManifest(_DefaultManifestMixin):
            @classmethod
            def get_supported_model_variants(cls):
                return ["foundation/sam"]

            @classmethod
            def model_json_schema(cls) -> dict:
                return {"name": "SAM"}

        block = _make_block_spec("roboflow_core/sam@v1", FakeManifest)

        with patch("inference.core.cache.air_gapped.MODEL_CACHE_DIR", cache), patch(
            "inference.core.cache.air_gapped.USE_INFERENCE_MODELS", True
        ):
            result = get_cached_foundation_models(blocks=[block])

        assert len(result) == 0


class TestFoundationModelListFormat:
    """Block with ``get_supported_model_variants()`` returning multiple variant IDs."""

    def test_detected_when_any_variant_cached(self, tmp_path):
        from inference.core.cache.air_gapped import get_cached_foundation_models

        cache = str(tmp_path)

        # Create a cache directory for one of the variants with a file.
        variant_dir = os.path.join(cache, "clip", "ViT-B-32")
        os.makedirs(variant_dir, exist_ok=True)
        open(os.path.join(variant_dir, "visual.onnx"), "w").close()

        class FakeManifest(_DefaultManifestMixin):
            @classmethod
            def get_supported_model_variants(cls):
                return ["clip/RN50", "clip/ViT-B-32", "clip/ViT-L-14"]

            @classmethod
            def model_json_schema(cls) -> dict:
                return {"name": "CLIP"}

        block = _make_block_spec("roboflow_core/clip@v1", FakeManifest)

        with patch("inference.core.cache.air_gapped.MODEL_CACHE_DIR", cache), patch(
            "inference.core.cache.air_gapped.USE_INFERENCE_MODELS", True
        ):
            result = get_cached_foundation_models(blocks=[block])

        assert len(result) == 1
        m = result[0]
        assert m["is_foundation"] is True
        assert m["name"] == "CLIP"

    def test_not_detected_when_no_variant_cached(self, tmp_path):
        from inference.core.cache.air_gapped import get_cached_foundation_models

        cache = str(tmp_path)

        class FakeManifest(_DefaultManifestMixin):
            @classmethod
            def get_supported_model_variants(cls):
                return ["clip/RN50", "clip/ViT-B-32"]

            @classmethod
            def model_json_schema(cls) -> dict:
                return {"name": "CLIP"}

        block = _make_block_spec("roboflow_core/clip@v1", FakeManifest)

        with patch("inference.core.cache.air_gapped.MODEL_CACHE_DIR", cache), patch(
            "inference.core.cache.air_gapped.USE_INFERENCE_MODELS", True
        ):
            result = get_cached_foundation_models(blocks=[block])

        assert len(result) == 0

    def test_not_detected_when_variant_dir_empty(self, tmp_path):
        from inference.core.cache.air_gapped import get_cached_foundation_models

        cache = str(tmp_path)

        # Create a cache directory but with no files in it.
        variant_dir = os.path.join(cache, "clip", "ViT-B-32")
        os.makedirs(variant_dir, exist_ok=True)

        class FakeManifest(_DefaultManifestMixin):
            @classmethod
            def get_supported_model_variants(cls):
                return ["clip/ViT-B-32"]

            @classmethod
            def model_json_schema(cls) -> dict:
                return {"name": "CLIP"}

        block = _make_block_spec("roboflow_core/clip@v1", FakeManifest)

        with patch("inference.core.cache.air_gapped.MODEL_CACHE_DIR", cache), patch(
            "inference.core.cache.air_gapped.USE_INFERENCE_MODELS", True
        ):
            result = get_cached_foundation_models(blocks=[block])


# ---------------------------------------------------------------------------
# scan_cached_models — model_config.json (inference-models cache layout)
# ---------------------------------------------------------------------------


def _write_model_config_json(
    cache_dir: str,
    slug_dir: str,
    package_id: str,
    config: dict,
) -> None:
    """Write a ``model_config.json`` inside the inference-models cache layout."""
    package_dir = os.path.join(cache_dir, "models-cache", slug_dir, package_id)
    os.makedirs(package_dir, exist_ok=True)
    with open(os.path.join(package_dir, "model_config.json"), "w") as fh:
        json.dump(config, fh)


class TestScanModelConfigJson:
    """model_config.json written by dump_model_config_for_offline_use."""

    def test_uses_canonical_model_id_from_config(self, tmp_path):
        """When model_config.json has model_id, use it instead of directory path."""
        from inference.core.cache.air_gapped import scan_cached_models

        cache = str(tmp_path)
        _write_model_config_json(
            cache,
            slug_dir="coco-22-abcd1234",
            package_id="pkg-001",
            config={
                "model_id": "coco/22",
                "task_type": "object-detection",
                "model_architecture": "yolov10b",
                "backend_type": "onnxruntime",
            },
        )

        result = scan_cached_models(cache)

        assert len(result) == 1
        m = result[0]
        assert m["model_id"] == "coco/22"
        assert m["task_type"] == "object-detection"
        assert m["model_architecture"] == "yolov10b"
        assert m["is_foundation"] is False

    def test_deduplicates_by_model_id(self, tmp_path):
        """Two cache entries with the same canonical model_id produce one result."""
        from inference.core.cache.air_gapped import scan_cached_models

        cache = str(tmp_path)
        # Same model in inference-models layout
        _write_model_config_json(
            cache,
            slug_dir="coco-22-abcd1234",
            package_id="pkg-001",
            config={
                "model_id": "coco/22",
                "task_type": "object-detection",
                "model_architecture": "yolov10b",
                "backend_type": "onnxruntime",
            },
        )
        # Same model also present in traditional layout
        _write_model_type_json(
            cache,
            "coco/22",
            {"project_task_type": "object-detection", "model_type": "yolov10b"},
        )

        result = scan_cached_models(cache)

        assert len(result) == 1
        assert result[0]["model_id"] == "coco/22"

    def test_skips_config_without_model_id(self, tmp_path):
        """model_config.json missing model_id falls back to model_type.json."""
        from inference.core.cache.air_gapped import scan_cached_models

        cache = str(tmp_path)
        # model_config.json without model_id — should not be picked up
        _write_model_config_json(
            cache,
            slug_dir="some-slug-abcd1234",
            package_id="pkg-001",
            config={
                "task_type": "object-detection",
                "model_architecture": "yolov8n",
                "backend_type": "onnxruntime",
            },
        )

        result = scan_cached_models(cache)

        assert len(result) == 0


# ---------------------------------------------------------------------------
# is_model_cached — inference-models layout delegation
# ---------------------------------------------------------------------------


class TestIsModelCachedInferenceModels:
    """is_model_cached delegates to find_cached_model_package_dir for the
    inference-models cache layout."""

    def test_returns_true_when_package_dir_found(self, tmp_path):
        from inference.core.cache.air_gapped import is_model_cached

        cache = str(tmp_path)
        fake_find = MagicMock(return_value="/some/cached/dir")
        fake_module = MagicMock()
        fake_module.find_cached_model_package_dir = fake_find

        with patch("inference.core.cache.air_gapped.MODEL_CACHE_DIR", cache), patch(
            "inference.core.cache.air_gapped.USE_INFERENCE_MODELS", True
        ), patch.dict(
            "sys.modules",
            {"inference_models.models.auto_loaders.core": fake_module},
        ):
            assert is_model_cached("my-model") is True
        fake_find.assert_called_once_with("my-model")

    def test_returns_false_when_no_cache_hit(self):
        from inference.core.cache.air_gapped import is_model_cached

        fake_find = MagicMock(return_value=None)
        fake_module = MagicMock()
        fake_module.find_cached_model_package_dir = fake_find

        with patch(
            "inference.core.cache.air_gapped.MODEL_CACHE_DIR", "/nonexistent"
        ), patch(
            "inference.core.cache.air_gapped.USE_INFERENCE_MODELS", True
        ), patch.dict(
            "sys.modules",
            {"inference_models.models.auto_loaders.core": fake_module},
        ):
            assert is_model_cached("no-such-model") is False

    def test_returns_false_when_inference_models_not_installed(self):
        from inference.core.cache.air_gapped import is_model_cached

        with patch(
            "inference.core.cache.air_gapped.MODEL_CACHE_DIR", "/nonexistent"
        ), patch(
            "inference.core.cache.air_gapped.USE_INFERENCE_MODELS", True
        ), patch.dict(
            "sys.modules",
            {"inference_models.models.auto_loaders.core": None},
        ):
            assert is_model_cached("some-model") is False
