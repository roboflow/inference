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
        assert m["model_id"] == "clip/ViT-B-32"
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

        assert result == []


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


def _write_auto_resolution_cache_entry(
    cache_dir: str,
    model_id: str,
    package_id: str,
) -> None:
    """Write metadata that links a legacy package slug to its canonical model ID."""
    resolution_cache_dir = os.path.join(cache_dir, "auto-resolution-cache")
    os.makedirs(resolution_cache_dir, exist_ok=True)
    with open(os.path.join(resolution_cache_dir, "resolution.json"), "w") as file:
        json.dump(
            {
                "model_id": model_id,
                "model_package_id": package_id,
            },
            file,
        )


class TestScanModelConfigJson:
    """model_config.json written by dump_model_config_for_offline_use."""

    def test_uses_canonical_model_id_from_config(self, tmp_path):
        """When model_config.json has model_id, use it instead of directory path."""
        from inference.core.cache.air_gapped import scan_cached_models
        from inference_models.models.auto_loaders.model_cache_paths import (
            slugify_model_id_to_os_safe_format,
        )

        cache = str(tmp_path)
        model_id = "coco/22"
        _write_model_config_json(
            cache,
            slug_dir=slugify_model_id_to_os_safe_format(model_id=model_id),
            package_id="pkg001",
            config={
                "model_id": model_id,
                "task_type": "object-detection",
                "model_architecture": "yolov10b",
                "backend_type": "onnx",
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
        from inference_models.models.auto_loaders.model_cache_paths import (
            slugify_model_id_to_os_safe_format,
        )

        cache = str(tmp_path)
        model_id = "coco/22"
        # Same model in inference-models layout
        _write_model_config_json(
            cache,
            slug_dir=slugify_model_id_to_os_safe_format(model_id=model_id),
            package_id="pkg001",
            config={
                "model_id": model_id,
                "task_type": "object-detection",
                "model_architecture": "yolov10b",
                "backend_type": "onnx",
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
        """model_config.json missing model_id is not reported."""
        from inference.core.cache.air_gapped import scan_cached_models

        cache = str(tmp_path)
        _write_model_config_json(
            cache,
            slug_dir="some-slug-abcd1234",
            package_id="pkg001",
            config={
                "task_type": "object-detection",
                "model_architecture": "yolov8n",
                "backend_type": "onnx",
            },
        )

        result = scan_cached_models(cache)

        assert len(result) == 0

    @pytest.mark.parametrize("invalid_model_id", ["", [], {}])
    def test_skips_config_with_invalid_model_id(self, tmp_path, invalid_model_id):
        from inference.core.cache.air_gapped import scan_cached_models

        cache = str(tmp_path)
        _write_model_config_json(
            cache,
            slug_dir="some-slug-abcd1234",
            package_id="pkg001",
            config={
                "model_id": invalid_model_id,
                "task_type": "object-detection",
                "model_architecture": "yolov8n",
                "backend_type": "onnx",
            },
        )

        assert scan_cached_models(cache) == []

    def test_uses_resolution_metadata_for_legacy_config(self, tmp_path):
        """Pre-upgrade configs are listed using their auto-resolution metadata."""
        from inference.core.cache.air_gapped import scan_cached_models
        from inference_models.models.auto_loaders.model_cache_paths import (
            slugify_model_id_to_os_safe_format,
        )

        cache = str(tmp_path)
        model_id = "workspace/my-project/3"
        package_id = "pkg001"
        model_slug = slugify_model_id_to_os_safe_format(model_id=model_id)
        _write_model_config_json(
            cache,
            slug_dir=model_slug,
            package_id=package_id,
            config={
                "task_type": "object-detection",
                "model_architecture": "yolov8n",
                "backend_type": "onnx",
            },
        )
        _write_auto_resolution_cache_entry(
            cache_dir=cache,
            model_id=model_id,
            package_id=package_id,
        )

        result = scan_cached_models(cache)

        assert result == [
            {
                "model_id": model_id,
                "name": model_id,
                "task_type": "object-detection",
                "model_architecture": "yolov8n",
                "is_foundation": False,
            }
        ]

    def test_legacy_scan_tolerates_missing_inference_models_package(self, tmp_path):
        from inference.core.cache.air_gapped import scan_cached_models

        cache = str(tmp_path)
        _write_auto_resolution_cache_entry(
            cache_dir=cache,
            model_id="workspace/my-project/3",
            package_id="pkg001",
        )

        with patch.dict(
            "sys.modules",
            {"inference_models.models.auto_loaders.model_cache_paths": None},
        ):
            assert scan_cached_models(cache) == []

    def test_skips_model_config_in_wrong_slug(self, tmp_path):
        from inference.core.cache.air_gapped import scan_cached_models

        _write_model_config_json(
            str(tmp_path),
            slug_dir="wrong-slug",
            package_id="pkg001",
            config={
                "model_id": "workspace/project/3",
                "task_type": "object-detection",
                "model_architecture": "yolov8n",
                "backend_type": "onnx",
            },
        )

        assert scan_cached_models(str(tmp_path)) == []

    @pytest.mark.parametrize(
        "field,value",
        [
            ("task_type", ["object-detection"]),
            ("model_architecture", {"name": "yolov8n"}),
        ],
    )
    def test_skips_malformed_model_config_metadata(
        self, tmp_path, field, value
    ):
        from inference.core.cache.air_gapped import scan_cached_models
        from inference_models.models.auto_loaders.model_cache_paths import (
            slugify_model_id_to_os_safe_format,
        )

        model_id = "workspace/project/3"
        config = {
            "model_id": model_id,
            "task_type": "object-detection",
            "model_architecture": "yolov8n",
            "backend_type": "onnx",
        }
        config[field] = value
        _write_model_config_json(
            str(tmp_path),
            slug_dir=slugify_model_id_to_os_safe_format(model_id=model_id),
            package_id="pkg001",
            config=config,
        )

        assert scan_cached_models(str(tmp_path)) == []

    def test_skips_symlinked_model_config_and_legacy_metadata(self, tmp_path):
        from inference.core.cache.air_gapped import scan_cached_models
        from inference_models.models.auto_loaders.model_cache_paths import (
            slugify_model_id_to_os_safe_format,
        )

        cache = tmp_path / "cache"
        model_id = "workspace/project/3"
        model_slug = slugify_model_id_to_os_safe_format(model_id=model_id)
        package_dir = cache / "models-cache" / model_slug / "pkg001"
        package_dir.mkdir(parents=True)
        external_config = tmp_path / "external-model-config.json"
        external_config.write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "task_type": "object-detection",
                    "model_architecture": "yolov8n",
                    "backend_type": "onnx",
                }
            )
        )
        (package_dir / "model_config.json").symlink_to(external_config)

        legacy_package = cache / "models-cache" / model_slug / "pkg002"
        legacy_package.mkdir()
        (legacy_package / "model_config.json").write_text(
            json.dumps(
                {
                    "task_type": "object-detection",
                    "model_architecture": "yolov8n",
                    "backend_type": "onnx",
                }
            )
        )
        resolution_dir = cache / "auto-resolution-cache"
        resolution_dir.mkdir()
        external_resolution = tmp_path / "external-resolution.json"
        external_resolution.write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "model_package_id": "pkg002",
                }
            )
        )
        (resolution_dir / "entry.json").symlink_to(external_resolution)

        assert scan_cached_models(str(cache)) == []

    def test_supports_symlinked_models_cache_root(self, tmp_path):
        from inference.core.cache.air_gapped import scan_cached_models
        from inference_models.models.auto_loaders.model_cache_paths import (
            slugify_model_id_to_os_safe_format,
        )

        cache = tmp_path / "cache"
        cache.mkdir()
        external_models_cache = tmp_path / "external-models-cache"
        external_models_cache.mkdir()
        (cache / "models-cache").symlink_to(
            external_models_cache,
            target_is_directory=True,
        )
        model_id = "workspace/project/3"
        package_dir = (
            external_models_cache
            / slugify_model_id_to_os_safe_format(model_id=model_id)
            / "pkg001"
        )
        package_dir.mkdir(parents=True)
        (package_dir / "model_config.json").write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "task_type": "object-detection",
                    "model_architecture": "yolov8n",
                    "backend_type": "onnx",
                }
            )
        )

        result = scan_cached_models(str(cache))

        assert [entry["model_id"] for entry in result] == [model_id]

    def test_rejects_models_cache_symlink_cycle(self, tmp_path):
        from inference.core.cache.air_gapped import scan_cached_models

        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "models-cache").symlink_to(cache, target_is_directory=True)

        assert scan_cached_models(str(cache)) == []

    def test_conflicting_package_metadata_is_not_reported(self, tmp_path):
        from inference.core.cache.air_gapped import scan_cached_models
        from inference_models.models.auto_loaders.model_cache_paths import (
            slugify_model_id_to_os_safe_format,
        )

        model_id = "workspace/project/3"
        model_slug = slugify_model_id_to_os_safe_format(model_id=model_id)
        for package_id, task_type in (
            ("pkg001", "object-detection"),
            ("pkg002", "classification"),
        ):
            _write_model_config_json(
                str(tmp_path),
                slug_dir=model_slug,
                package_id=package_id,
                config={
                    "model_id": model_id,
                    "task_type": task_type,
                    "model_architecture": "yolov8n",
                    "backend_type": "onnx",
                },
            )

        assert scan_cached_models(str(tmp_path)) == []

    def test_excludes_nested_cache_root(self, tmp_path):
        from inference.core.cache.air_gapped import scan_cached_models

        cache = tmp_path / "cache"
        nested_cache = cache / "nested"
        _write_model_type_json(
            str(cache),
            "outer/model/1",
            {
                "project_task_type": "object-detection",
                "model_type": "yolov8n",
            },
        )
        _write_model_type_json(
            str(nested_cache),
            "inner/model/1",
            {
                "project_task_type": "classification",
                "model_type": "vit",
            },
        )

        result = scan_cached_models(
            str(cache),
            excluded_cache_roots=[str(nested_cache)],
        )

        assert [entry["model_id"] for entry in result] == ["outer/model/1"]

    def test_skips_symlinked_traditional_model_type(self, tmp_path):
        from inference.core.cache.air_gapped import scan_cached_models

        model_dir = tmp_path / "workspace" / "project" / "3"
        model_dir.mkdir(parents=True)
        external_metadata = tmp_path / "external-model-type.json"
        external_metadata.write_text(
            json.dumps(
                {
                    "project_task_type": "object-detection",
                    "model_type": "yolov8n",
                }
            )
        )
        (model_dir / "model_type.json").symlink_to(external_metadata)

        assert scan_cached_models(str(tmp_path)) == []

    def test_does_not_treat_model_type_inside_models_cache_as_traditional(
        self, tmp_path
    ):
        from inference.core.cache.air_gapped import scan_cached_models

        package_dir = tmp_path / "models-cache" / "wrong-slug" / "pkg001"
        package_dir.mkdir(parents=True)
        (package_dir / "model_type.json").write_text(
            json.dumps(
                {
                    "project_task_type": "object-detection",
                    "model_type": "yolov8n",
                }
            )
        )

        assert scan_cached_models(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# is_model_cached — inference-models layout delegation
# ---------------------------------------------------------------------------


class TestIsModelCachedTraditionalLayout:
    @pytest.mark.parametrize("model_id", ["", ".", "nested/.."])
    def test_does_not_treat_cache_root_as_a_model(self, tmp_path, model_id):
        from inference.core.cache.air_gapped import is_model_cached

        (tmp_path / "unrelated-cache-entry").write_text("not a model")

        with patch(
            "inference.core.cache.air_gapped.MODEL_CACHE_DIR", str(tmp_path)
        ), patch("inference.core.cache.air_gapped.USE_INFERENCE_MODELS", False):
            assert is_model_cached(model_id) is False

    def test_rejects_intermediate_symlink_below_cache_root(self, tmp_path):
        from inference.core.cache.air_gapped import is_model_cached

        cache_root = tmp_path / "cache"
        cache_root.mkdir()
        outside_workspace = tmp_path / "outside-workspace"
        model_directory = outside_workspace / "project" / "1"
        model_directory.mkdir(parents=True)
        (model_directory / "weights.onnx").write_text("outside")
        (cache_root / "workspace").symlink_to(
            outside_workspace,
            target_is_directory=True,
        )

        with patch(
            "inference.core.cache.air_gapped.MODEL_CACHE_DIR",
            str(cache_root),
        ), patch("inference.core.cache.air_gapped.USE_INFERENCE_MODELS", False):
            assert is_model_cached("workspace/project/1") is False

    def test_allows_configured_cache_root_to_be_a_symlink(self, tmp_path):
        from inference.core.cache.air_gapped import is_model_cached

        real_cache_root = tmp_path / "real-cache"
        model_directory = real_cache_root / "workspace" / "project" / "1"
        model_directory.mkdir(parents=True)
        (model_directory / "weights.onnx").write_text("cached")
        mounted_cache_root = tmp_path / "mounted-cache"
        mounted_cache_root.symlink_to(
            real_cache_root,
            target_is_directory=True,
        )

        with patch(
            "inference.core.cache.air_gapped.MODEL_CACHE_DIR",
            str(mounted_cache_root),
        ), patch("inference.core.cache.air_gapped.USE_INFERENCE_MODELS", False):
            assert is_model_cached("workspace/project/1") is True


class TestIsModelCachedInferenceModels:
    """is_model_cached delegates to find_cached_model_package_dir for the
    inference-models cache layout."""

    def test_returns_true_when_package_dir_found(self, tmp_path):
        from inference.core.cache.air_gapped import is_model_cached
        from inference_models.models.auto_loaders import core as auto_loaders

        cache = str(tmp_path)
        cached_package_dir = tmp_path / "models-cache" / "my-model" / "package"
        cached_package_dir.mkdir(parents=True)
        (cached_package_dir / "model_config.json").write_text("{}")
        fake_find = MagicMock(return_value=str(cached_package_dir))

        with patch("inference.core.cache.air_gapped.MODEL_CACHE_DIR", cache), patch(
            "inference.core.cache.air_gapped.USE_INFERENCE_MODELS", True
        ), patch.object(
            auto_loaders,
            "find_cached_model_package_dir",
            fake_find,
        ):
            assert is_model_cached("my-model") is True
        fake_find.assert_called_once_with(model_id="my-model")

    def test_returns_false_when_no_cache_hit(self):
        from inference.core.cache.air_gapped import is_model_cached
        from inference_models.models.auto_loaders import core as auto_loaders

        fake_find = MagicMock(return_value=None)

        with patch(
            "inference.core.cache.air_gapped.MODEL_CACHE_DIR", "/nonexistent"
        ), patch(
            "inference.core.cache.air_gapped.USE_INFERENCE_MODELS", True
        ), patch.object(
            auto_loaders,
            "find_cached_model_package_dir",
            fake_find,
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
