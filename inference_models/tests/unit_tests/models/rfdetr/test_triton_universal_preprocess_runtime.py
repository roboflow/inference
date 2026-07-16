import numpy as np
import pytest
import torch

from inference_models import PreProcessingOverrides
from inference_models.errors import ModelRuntimeError
from inference_models.models.common.roboflow.model_packages import (
    ColorMode,
    Contrast,
    ContrastType,
    Grayscale,
    ImagePreProcessing,
    NetworkInputDefinition,
    ResizeMode,
    StaticCrop,
    TrainingInputSize,
)
from inference_models.models.rfdetr.optimization.catalog import (
    RFDETR_PREPROCESSOR_IMPLEMENTATIONS,
)
from inference_models.models.rfdetr.optimization.ids import (
    RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1,
)
from inference_models.models.rfdetr.pre_processing import (
    resolve_rfdetr_preprocessor_max_workers,
)
from inference_models.models.rfdetr.triton_universal_preprocess_runtime import (
    UniversalFastPreprocessRuntime,
    _build_metadata_batch,
    _canonicalize_batch,
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _network_input(
    *,
    resize_mode: ResizeMode = ResizeMode.STRETCH_TO,
) -> NetworkInputDefinition:
    return NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=64),
        dataset_version_resize_dimensions=None,
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=resize_mode,
        input_channels=3,
        scaling_factor=255,
        normalization=[list(_IMAGENET_MEAN), list(_IMAGENET_STD)],
    )


@pytest.mark.parametrize(
    "images",
    [
        np.zeros((2, 8, 9, 3), dtype=np.uint8),
        [np.zeros((8, 9, 3), dtype=np.uint8) for _ in range(2)],
        torch.zeros((2, 3, 8, 9), dtype=torch.uint8),
        torch.zeros((2, 8, 9, 3), dtype=torch.uint8),
    ],
)
def test_canonicalize_uint8_cpu_batches(images) -> None:
    batch = _canonicalize_batch(images)

    assert batch.kind == "uint8"
    assert (batch.height, batch.width) == (8, 9)
    assert len(batch.items) == 2
    assert all(tuple(item.shape) == (8, 9, 3) for item in batch.items)


@pytest.mark.parametrize(
    "images",
    [
        torch.zeros((2, 3, 8, 9), dtype=torch.float32),
        torch.zeros((2, 8, 9, 3), dtype=torch.float16),
    ],
)
def test_canonicalize_float_cpu_batches(images) -> None:
    batch = _canonicalize_batch(images)

    assert batch.kind == "float"
    assert (batch.height, batch.width) == (8, 9)
    assert all(tuple(item.shape) == (3, 8, 9) for item in batch.items)


def test_canonicalize_float_numpy_matches_reference_uint8_conversion() -> None:
    image = np.array([[[0.0, 0.5, 1.5]]], dtype=np.float32)

    batch = _canonicalize_batch(image)

    np.testing.assert_array_equal(
        batch.items[0],
        np.array([[[0, 127, 255]]], dtype=np.uint8),
    )


def test_canonicalize_rejects_mixed_semantics() -> None:
    with pytest.raises(ModelRuntimeError, match="homogeneous batch"):
        _canonicalize_batch(
            [
                torch.zeros((3, 8, 9), dtype=torch.uint8),
                torch.zeros((3, 8, 9), dtype=torch.float32),
            ]
        )


def test_canonicalize_rejects_mixed_source_dimensions() -> None:
    with pytest.raises(ModelRuntimeError, match="equal source dimensions"):
        _canonicalize_batch(
            [
                np.zeros((8, 9, 3), dtype=np.uint8),
                np.zeros((10, 9, 3), dtype=np.uint8),
            ]
        )


def test_metadata_batch_describes_stretch() -> None:
    metadata = _build_metadata_batch(
        batch_size=2,
        source_h=8,
        source_w=10,
        target_h=16,
        target_w=40,
    )

    assert len(metadata) == 2
    assert metadata[0].scale_height == 2
    assert metadata[0].scale_width == 4
    assert metadata[0].static_crop_offset.crop_width == 10
    assert metadata[0].static_crop_offset.crop_height == 8


def test_universal_candidate_is_explicitly_selectable() -> None:
    metadata = RFDETR_PREPROCESSOR_IMPLEMENTATIONS[
        RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1
    ]
    assert metadata.validated_environments == ()
    assert metadata.fallback_id == "base"


@pytest.mark.parametrize(
    ("image_pre_processing", "reason"),
    [
        (
            ImagePreProcessing.model_validate(
                {
                    "static-crop": StaticCrop(
                        enabled=True,
                        x_min=10,
                        x_max=90,
                        y_min=10,
                        y_max=90,
                    )
                }
            ),
            "static crop",
        ),
        (ImagePreProcessing(grayscale=Grayscale(enabled=True)), "grayscale"),
        (
            ImagePreProcessing(
                contrast=Contrast(
                    enabled=True,
                    type=ContrastType.CONTRAST_STRETCHING,
                )
            ),
            "contrast",
        ),
    ],
)
def test_model_compatibility_reports_base_supported_transformations(
    image_pre_processing: ImagePreProcessing,
    reason: str,
) -> None:
    compatibility = UniversalFastPreprocessRuntime.check_model_compatibility(
        image_pre_processing=image_pre_processing,
        network_input=_network_input(),
    )

    assert not compatibility.supported
    assert reason in compatibility.reasons


def test_model_compatibility_reports_non_stretch_resize() -> None:
    compatibility = UniversalFastPreprocessRuntime.check_model_compatibility(
        image_pre_processing=ImagePreProcessing(),
        network_input=_network_input(resize_mode=ResizeMode.LETTERBOX),
    )

    assert not compatibility.supported
    assert any(reason.startswith("resize_mode=") for reason in compatibility.reasons)


def test_request_compatibility_reports_overrides_and_heterogeneous_shapes() -> None:
    compatibility = UniversalFastPreprocessRuntime.check_request_compatibility(
        images=[
            np.zeros((8, 9, 3), dtype=np.uint8),
            np.zeros((10, 9, 3), dtype=np.uint8),
        ],
        pre_processing_overrides=PreProcessingOverrides(disable_static_crop=True),
    )

    assert not compatibility.supported
    assert "pre-processing overrides" in compatibility.reasons
    assert any(
        reason.startswith("heterogeneous source dimensions")
        for reason in compatibility.reasons
    )


def test_supported_request_is_compatible() -> None:
    compatibility = UniversalFastPreprocessRuntime.check_request_compatibility(
        images=np.zeros((2, 8, 9, 3), dtype=np.uint8),
        pre_processing_overrides=None,
    )

    assert compatibility.supported


def test_preprocessor_worker_limit_can_be_selected_from_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv("INFERENCE_MODELS_RFDETR_PREPROCESSOR_MAX_WORKERS", "7")

    assert resolve_rfdetr_preprocessor_max_workers() == 7


def test_explicit_preprocessor_worker_limit_overrides_environment(monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_MODELS_RFDETR_PREPROCESSOR_MAX_WORKERS", "7")

    assert resolve_rfdetr_preprocessor_max_workers(2) == 2


def test_preprocessor_worker_limit_rejects_non_positive_environment_value(
    monkeypatch,
) -> None:
    monkeypatch.setenv("INFERENCE_MODELS_RFDETR_PREPROCESSOR_MAX_WORKERS", "0")

    with pytest.raises(ModelRuntimeError, match="must be at least 1"):
        resolve_rfdetr_preprocessor_max_workers()


def test_universal_runtime_requires_cuda_device() -> None:
    with pytest.raises(ModelRuntimeError, match="requires a CUDA target"):
        UniversalFastPreprocessRuntime(device=torch.device("cpu"))
