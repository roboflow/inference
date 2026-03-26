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


class TestFoundationModelDetection:
    """Block with ``get_required_cache_artifacts()`` whose files exist."""

    def test_foundation_model_detected_when_cached(self, tmp_path):
        from inference.core.cache.air_gapped import get_cached_foundation_models

        cache = str(tmp_path)
        model_id = "foundation/clip"

        # Create the required files in the cache.
        model_dir = os.path.join(cache, model_id)
        os.makedirs(model_dir, exist_ok=True)
        for fname in ["weights.pt", "config.json"]:
            open(os.path.join(model_dir, fname), "w").close()

        class FakeManifest:
            """Minimal manifest: only model_id and files are required.

            Name is derived from model_json_schema() metadata.
            """

            @classmethod
            def get_required_cache_artifacts(cls) -> Dict[str, Any]:
                return {
                    "model_id": "foundation/clip",
                    "files": ["weights.pt", "config.json"],
                }

            @classmethod
            def model_json_schema(cls) -> dict:
                return {"name": "CLIP"}

        block = _make_block_spec("roboflow_core/clip@v1", FakeManifest)

        with patch(
            "inference.core.cache.air_gapped.MODEL_CACHE_DIR",
            cache,
        ), patch(
            "inference.core.cache.model_artifacts.MODEL_CACHE_DIR",
            cache,
        ):
            result = get_cached_foundation_models(blocks=[block])

        assert len(result) == 1
        m = result[0]
        assert m["model_id"] == "foundation/clip"
        assert m["is_foundation"] is True
        assert m["name"] == "CLIP"


class TestFoundationModelMissing:
    """Block with ``get_required_cache_artifacts()`` whose files do NOT exist."""

    def test_foundation_model_not_detected_when_missing(self, tmp_path):
        from inference.core.cache.air_gapped import get_cached_foundation_models

        cache = str(tmp_path)

        class FakeManifest:
            @classmethod
            def get_required_cache_artifacts(cls) -> Dict[str, Any]:
                return {
                    "model_id": "foundation/sam",
                    "files": ["encoder.pt", "decoder.pt"],
                }

            @classmethod
            def model_json_schema(cls) -> dict:
                return {"name": "SAM"}

        block = _make_block_spec("roboflow_core/sam@v1", FakeManifest)

        with patch(
            "inference.core.cache.air_gapped.MODEL_CACHE_DIR",
            cache,
        ), patch(
            "inference.core.cache.model_artifacts.MODEL_CACHE_DIR",
            cache,
        ):
            result = get_cached_foundation_models(blocks=[block])

        assert len(result) == 0


class TestFoundationModelListFormat:
    """Block with ``get_required_cache_artifacts()`` returning a list of model_ids."""

    def test_detected_when_any_variant_cached(self, tmp_path):
        from inference.core.cache.air_gapped import get_cached_foundation_models

        cache = str(tmp_path)

        # Create a cache directory for one of the variants with a file.
        variant_dir = os.path.join(cache, "clip", "ViT-B-32")
        os.makedirs(variant_dir, exist_ok=True)
        open(os.path.join(variant_dir, "visual.onnx"), "w").close()

        class FakeManifest:
            @classmethod
            def get_required_cache_artifacts(cls) -> List[str]:
                return ["clip/RN50", "clip/ViT-B-32", "clip/ViT-L-14"]

            @classmethod
            def model_json_schema(cls) -> dict:
                return {"name": "CLIP"}

        block = _make_block_spec("roboflow_core/clip@v1", FakeManifest)

        with patch("inference.core.cache.air_gapped.MODEL_CACHE_DIR", cache):
            result = get_cached_foundation_models(blocks=[block])

        assert len(result) == 1
        m = result[0]
        assert m["is_foundation"] is True
        assert m["name"] == "CLIP"

    def test_not_detected_when_no_variant_cached(self, tmp_path):
        from inference.core.cache.air_gapped import get_cached_foundation_models

        cache = str(tmp_path)

        class FakeManifest:
            @classmethod
            def get_required_cache_artifacts(cls) -> List[str]:
                return ["clip/RN50", "clip/ViT-B-32"]

            @classmethod
            def model_json_schema(cls) -> dict:
                return {"name": "CLIP"}

        block = _make_block_spec("roboflow_core/clip@v1", FakeManifest)

        with patch("inference.core.cache.air_gapped.MODEL_CACHE_DIR", cache):
            result = get_cached_foundation_models(blocks=[block])

        assert len(result) == 0

    def test_not_detected_when_variant_dir_empty(self, tmp_path):
        from inference.core.cache.air_gapped import get_cached_foundation_models

        cache = str(tmp_path)

        # Create a cache directory but with no files in it.
        variant_dir = os.path.join(cache, "clip", "ViT-B-32")
        os.makedirs(variant_dir, exist_ok=True)

        class FakeManifest:
            @classmethod
            def get_required_cache_artifacts(cls) -> List[str]:
                return ["clip/ViT-B-32"]

            @classmethod
            def model_json_schema(cls) -> dict:
                return {"name": "CLIP"}

        block = _make_block_spec("roboflow_core/clip@v1", FakeManifest)

        with patch("inference.core.cache.air_gapped.MODEL_CACHE_DIR", cache):
            result = get_cached_foundation_models(blocks=[block])

        assert len(result) == 0


# ── Cross-validation: _slugify_model_id must match inference_models ──────────

_SLUGIFY_TEST_IDS = [
    "clip/ViT-B-16",
    "coco/40",
    "rfdetr-medium",
    "sam3/sam3_final",
    "florence-pretrains/3",
    "depth-anything-v3/small",
    "smolvlm2/smolvlm-2.2b-instruct",
    "qwen-pretrains/1",
    "a" * 100,  # long model id
    "special!!!chars###here",
]


@pytest.mark.parametrize("model_id", _SLUGIFY_TEST_IDS)
def test_slugify_matches_inference_models(model_id: str):
    """Ensure _slugify_model_id stays in sync with the canonical implementation."""
    try:
        from inference_models.models.auto_loaders.core import (
            slugify_model_id_to_os_safe_format,
        )
    except ImportError:
        pytest.skip("inference_models not installed")

    from inference.core.cache.air_gapped import _slugify_model_id

    assert _slugify_model_id(model_id) == slugify_model_id_to_os_safe_format(model_id)
