import warnings

import numpy as np
import pytest
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from inference_models.entities import ImageDimensions
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
from inference_models.models.rfdetr import (
    triton_preprocess,
    triton_preprocess_runtime,
)
from inference_models.models.rfdetr.triton_preprocess_runtime import (
    FastPreprocessRuntime,
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_DEFAULT_NORMALIZATION = object()


def _network_input(
    target_h: int = 64,
    target_w: int = 64,
    dataset_version_resize_dimensions=None,
    resize_mode: ResizeMode = ResizeMode.STRETCH_TO,
    input_channels: int = 3,
    scaling_factor=255,
    normalization=_DEFAULT_NORMALIZATION,
) -> NetworkInputDefinition:
    if normalization is _DEFAULT_NORMALIZATION:
        normalization = [list(_IMAGENET_MEAN), list(_IMAGENET_STD)]
    return NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=target_h, width=target_w),
        dataset_version_resize_dimensions=dataset_version_resize_dimensions,
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=resize_mode,
        input_channels=input_channels,
        scaling_factor=scaling_factor,
        normalization=normalization,
    )


def _call_fast_preprocess(
    runtime: FastPreprocessRuntime,
    *,
    images=None,
    image_size=None,
    image_pre_processing=None,
    network_input=None,
):
    if images is None:
        images = np.zeros((8, 8, 3), dtype=np.uint8)
    if image_pre_processing is None:
        image_pre_processing = ImagePreProcessing()
    if network_input is None:
        network_input = _network_input()
    return runtime.try_preprocess(
        images=images,
        input_color_format="bgr",
        image_size=image_size,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        stream=None,
    )


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
    monkeypatch.setattr(triton_preprocess_runtime, "_FAST_PATH_ENABLED", True)
    monkeypatch.setattr(triton_preprocess_runtime, "_TRITON_AVAILABLE", True)
    runtime = FastPreprocessRuntime(device=torch.device("cuda"))
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.warns(RuntimeWarning, match="only batch size 1 is supported"):
        assert (
            runtime.try_preprocess(
                images=[image, image],
                input_color_format="bgr",
                image_size=None,
                image_pre_processing=ImagePreProcessing(),
                network_input=_network_input(),
                stream=None,
            )
            is None
        )

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        assert (
            runtime.try_preprocess(
                images=[image, image],
                input_color_format="bgr",
                image_size=None,
                image_pre_processing=ImagePreProcessing(),
                network_input=_network_input(),
                stream=None,
            )
            is None
        )
    assert recorded == []


def test_trt_fast_preprocess_flag_disabled_returns_none_without_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(triton_preprocess_runtime, "_FAST_PATH_ENABLED", False)
    monkeypatch.setattr(triton_preprocess_runtime, "_TRITON_AVAILABLE", True)
    runtime = FastPreprocessRuntime(device=torch.device("cuda"))

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        assert _call_fast_preprocess(runtime) is None
    assert recorded == []


@pytest.mark.parametrize(
    ("runtime_device", "kwargs", "reason"),
    [
        (
            torch.device("cuda"),
            {},
            "triton is not installed",
        ),
    ],
)
def test_trt_fast_preprocess_warns_for_unavailable_runtime(
    monkeypatch: pytest.MonkeyPatch,
    runtime_device: torch.device,
    kwargs,
    reason: str,
) -> None:
    monkeypatch.setattr(triton_preprocess_runtime, "_FAST_PATH_ENABLED", True)
    monkeypatch.setattr(triton_preprocess_runtime, "_TRITON_AVAILABLE", False)
    runtime = FastPreprocessRuntime(device=runtime_device)

    with pytest.warns(RuntimeWarning, match=reason):
        assert _call_fast_preprocess(runtime, **kwargs) is None


@pytest.mark.parametrize(
    ("runtime_device", "kwargs", "reason"),
    [
        (
            torch.device("cpu"),
            {},
            "CUDA device is required",
        ),
        (
            torch.device("cuda"),
            {"image_size": (32, 32)},
            "custom image_size overrides are not supported",
        ),
        (
            torch.device("cuda"),
            {
                "image_pre_processing": ImagePreProcessing.model_validate(
                    {
                        "static-crop": StaticCrop(
                            enabled=True,
                            x_min=0,
                            x_max=8,
                            y_min=0,
                            y_max=8,
                        )
                    }
                )
            },
            "static crop, contrast, and grayscale preprocessing are unsupported",
        ),
        (
            torch.device("cuda"),
            {
                "image_pre_processing": ImagePreProcessing(
                    contrast=Contrast(
                        enabled=True,
                        type=ContrastType.CONTRAST_STRETCHING,
                    )
                )
            },
            "static crop, contrast, and grayscale preprocessing are unsupported",
        ),
        (
            torch.device("cuda"),
            {
                "image_pre_processing": ImagePreProcessing(
                    grayscale=Grayscale(enabled=True)
                )
            },
            "static crop, contrast, and grayscale preprocessing are unsupported",
        ),
        (
            torch.device("cuda"),
            {
                "network_input": _network_input(
                    dataset_version_resize_dimensions=TrainingInputSize(
                        height=8,
                        width=8,
                    )
                )
            },
            "dataset-version resize is unsupported",
        ),
        (
            torch.device("cuda"),
            {"network_input": _network_input(input_channels=1)},
            "only 3-channel inputs are supported",
        ),
        (
            torch.device("cuda"),
            {"network_input": _network_input(scaling_factor=1)},
            "only scaling_factor None or 255 is supported",
        ),
        (
            torch.device("cuda"),
            {"network_input": _network_input(normalization=None)},
            "normalization is required",
        ),
        (
            torch.device("cuda"),
            {"network_input": _network_input(resize_mode=ResizeMode.LETTERBOX)},
            "resize mode",
        ),
        (
            torch.device("cuda"),
            {"network_input": _network_input(resize_mode=ResizeMode.CENTER_CROP)},
            "resize mode",
        ),
        (
            torch.device("cuda"),
            {
                "network_input": _network_input(
                    resize_mode=ResizeMode.LETTERBOX_REFLECT_EDGES
                )
            },
            "resize mode",
        ),
        (
            torch.device("cuda"),
            {"network_input": _network_input(resize_mode=ResizeMode.FIT_LONGER_EDGE)},
            "resize mode",
        ),
        (
            torch.device("cuda"),
            {"images": torch.zeros((8, 8, 3), dtype=torch.uint8)},
            "only numpy ndarray inputs are supported",
        ),
        (
            torch.device("cuda"),
            {"images": np.zeros((8, 8, 3), dtype=np.float32)},
            "input must be uint8 HWC with 3 channels",
        ),
        (
            torch.device("cuda"),
            {"images": np.zeros((8, 8), dtype=np.uint8)},
            "input must be uint8 HWC with 3 channels",
        ),
        (
            torch.device("cuda"),
            {"images": np.zeros((8, 8, 1), dtype=np.uint8)},
            "input must be uint8 HWC with 3 channels",
        ),
    ],
)
def test_trt_fast_preprocess_warns_for_unsupported_requests(
    monkeypatch: pytest.MonkeyPatch,
    runtime_device: torch.device,
    kwargs,
    reason: str,
) -> None:
    monkeypatch.setattr(triton_preprocess_runtime, "_FAST_PATH_ENABLED", True)
    monkeypatch.setattr(triton_preprocess_runtime, "_TRITON_AVAILABLE", True)
    runtime = FastPreprocessRuntime(device=runtime_device)

    with pytest.warns(RuntimeWarning, match=reason):
        assert _call_fast_preprocess(runtime, **kwargs) is None


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_preprocess.TRITON_AVAILABLE,
    reason="CUDA and Triton are required",
)
def test_trt_fast_preprocess_matches_reference_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(triton_preprocess_runtime, "_FAST_PATH_ENABLED", True)
    monkeypatch.setattr(triton_preprocess_runtime, "_TRITON_AVAILABLE", True)
    target_h, target_w = 64, 64
    runtime = FastPreprocessRuntime(device=torch.device("cuda"))
    stream = torch.cuda.Stream(device=torch.device("cuda"))
    rng = np.random.default_rng(seed=71)
    image_rgb = rng.integers(0, 256, size=(96, 80, 3), dtype=np.uint8)
    image_bgr = image_rgb[:, :, ::-1].copy()

    result = runtime.try_preprocess(
        images=image_bgr,
        input_color_format="bgr",
        image_size=None,
        image_pre_processing=ImagePreProcessing(),
        network_input=_network_input(target_h=target_h, target_w=target_w),
        stream=stream,
    )
    assert result is not None
    result.ready_event.synchronize()

    expected = _reference_preprocess(image_rgb, target_h=target_h, target_w=target_w)
    torch.testing.assert_close(result.tensor.cpu(), expected, atol=1e-6, rtol=0)

    metadata = result.metadata
    assert metadata[0].original_size == ImageDimensions(height=96, width=80)
    assert metadata[0].size_after_pre_processing == ImageDimensions(
        height=96,
        width=80,
    )
    assert metadata[0].inference_size == ImageDimensions(
        height=target_h,
        width=target_w,
    )
    assert result.tensor._pre_processing_meta == metadata  # type: ignore[attr-defined]
