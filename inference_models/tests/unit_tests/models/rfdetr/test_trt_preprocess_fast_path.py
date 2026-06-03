import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from inference_models.entities import ImageDimensions
from inference_models.models.common.roboflow.model_packages import (
    ColorMode,
    ImagePreProcessing,
    NetworkInputDefinition,
    ResizeMode,
    TrainingInputSize,
)
from inference_models.models.rfdetr import triton_preprocess

pytest.importorskip("tensorrt")
pytest.importorskip("pycuda.driver")

from inference_models.models.rfdetr import (
    rfdetr_instance_segmentation_trt,
)  # noqa: E402

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _network_input(
    target_h: int = 64,
    target_w: int = 64,
) -> NetworkInputDefinition:
    return NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=target_h, width=target_w),
        dataset_version_resize_dimensions=None,
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
        scaling_factor=255,
        normalization=[list(_IMAGENET_MEAN), list(_IMAGENET_STD)],
    )


def _adapter_for_fast_preprocess(network_input: NetworkInputDefinition):
    model = object.__new__(
        rfdetr_instance_segmentation_trt.RFDetrForInstanceSegmentationTRT
    )
    model._inference_config = SimpleNamespace(
        image_pre_processing=ImagePreProcessing(),
        network_input=network_input,
    )
    model._device = torch.device("cuda")
    model._pre_process_cuda_stream = torch.cuda.Stream(device=model._device)
    model._fast_path_state = None
    model._fast_preprocess_warned_reasons = set()
    return model


def _reference_preprocess(image_rgb: np.ndarray, target_h: int, target_w: int):
    resized = TF.resize(
        Image.fromarray(image_rgb),
        (target_h, target_w),
        antialias=True,
    )
    tensor = TF.to_tensor(resized)
    tensor = TF.normalize(
        tensor,
        mean=list(_IMAGENET_MEAN),
        std=list(_IMAGENET_STD),
    )
    return tensor.unsqueeze(0)


def test_trt_fast_preprocess_warns_once_for_unsupported_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rfdetr_instance_segmentation_trt, "_FAST_PATH_ENABLED", True)
    monkeypatch.setattr(rfdetr_instance_segmentation_trt, "_TRITON_AVAILABLE", True)
    model = object.__new__(
        rfdetr_instance_segmentation_trt.RFDetrForInstanceSegmentationTRT
    )
    model._inference_config = SimpleNamespace(
        image_pre_processing=ImagePreProcessing(),
        network_input=_network_input(),
    )
    model._fast_preprocess_warned_reasons = set()
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.warns(RuntimeWarning, match="only batch size 1 is supported"):
        assert (
            model._try_fast_preprocess(
                images=[image, image],
                input_color_format="bgr",
                image_size=None,
                pre_processing_overrides=None,
            )
            is None
        )

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        assert (
            model._try_fast_preprocess(
                images=[image, image],
                input_color_format="bgr",
                image_size=None,
                pre_processing_overrides=None,
            )
            is None
        )
    assert recorded == []


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_preprocess.TRITON_AVAILABLE,
    reason="CUDA and Triton are required",
)
def test_trt_fast_preprocess_matches_reference_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rfdetr_instance_segmentation_trt, "_FAST_PATH_ENABLED", True)
    monkeypatch.setattr(rfdetr_instance_segmentation_trt, "_TRITON_AVAILABLE", True)
    target_h, target_w = 64, 64
    model = _adapter_for_fast_preprocess(
        network_input=_network_input(target_h=target_h, target_w=target_w),
    )
    rng = np.random.default_rng(seed=71)
    image_rgb = rng.integers(0, 256, size=(96, 80, 3), dtype=np.uint8)
    image_bgr = image_rgb[:, :, ::-1].copy()

    actual, metadata = model._try_fast_preprocess(
        images=image_bgr,
        input_color_format="bgr",
        image_size=None,
        pre_processing_overrides=None,
    )
    actual._trt_ready_event.synchronize()  # type: ignore[attr-defined]

    expected = _reference_preprocess(image_rgb, target_h=target_h, target_w=target_w)
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-6, rtol=0)

    assert metadata[0].original_size == ImageDimensions(height=96, width=80)
    assert metadata[0].size_after_pre_processing == ImageDimensions(
        height=96,
        width=80,
    )
    assert metadata[0].inference_size == ImageDimensions(
        height=target_h,
        width=target_w,
    )
    assert actual._pre_processing_meta == metadata  # type: ignore[attr-defined]
