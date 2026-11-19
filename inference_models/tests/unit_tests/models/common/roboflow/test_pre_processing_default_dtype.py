from typing import Generator, List, Optional, Tuple, Union

import numpy as np
import pytest
import torch

from inference_models.models.common.roboflow.model_packages import (
    ColorMode,
    ImagePreProcessing,
    NetworkInputDefinition,
    ResizeMode,
    TrainingInputSize,
)
from inference_models.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)

ImagesInput = Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]


def _numpy_image() -> np.ndarray:
    return np.random.default_rng(seed=0).integers(
        0, 256, size=(192, 168, 3), dtype=np.uint8
    )


def _numpy_input() -> ImagesInput:
    return _numpy_image()


def _numpy_list_input() -> ImagesInput:
    return [_numpy_image(), _numpy_image()]


def _tensor_input() -> ImagesInput:
    return torch.from_numpy(_numpy_image()).permute(2, 0, 1).contiguous()


def _tensor_list_input() -> ImagesInput:
    return [_tensor_input(), _tensor_input()]


INPUT_FACTORIES = {
    "numpy": _numpy_input,
    "numpy-list": _numpy_list_input,
    "tensor": _tensor_input,
    "tensor-list": _tensor_list_input,
}

# Stretch, center-crop and fit-longer-edge divide the resized integer image by
# `scaling_factor` directly. Letterbox copies it into a float32 canvas first and was
# never affected; it is kept as a control.
RESIZE_MODES_AND_INPUTS = [
    (ResizeMode.STRETCH_TO, "numpy"),
    (ResizeMode.STRETCH_TO, "numpy-list"),
    (ResizeMode.STRETCH_TO, "tensor"),
    (ResizeMode.STRETCH_TO, "tensor-list"),
    (ResizeMode.CENTER_CROP, "numpy"),
    (ResizeMode.CENTER_CROP, "numpy-list"),
    (ResizeMode.CENTER_CROP, "tensor"),
    (ResizeMode.CENTER_CROP, "tensor-list"),
    (ResizeMode.FIT_LONGER_EDGE, "numpy"),
    (ResizeMode.FIT_LONGER_EDGE, "tensor"),
    (ResizeMode.LETTERBOX, "numpy"),
    (ResizeMode.LETTERBOX, "tensor-list"),
]


@pytest.fixture
def bfloat16_default_dtype() -> Generator[None, None, None]:
    # Simulates the state a racing transformers `from_pretrained(dtype=bf16)` leaves.
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        yield
    finally:
        torch.set_default_dtype(original_dtype)


def _network_input(
    resize_mode: ResizeMode,
    normalization: Optional[Tuple[List[float], List[float]]],
) -> NetworkInputDefinition:
    return NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=resize_mode,
        input_channels=3,
        scaling_factor=255,
        normalization=normalization,
    )


def _pre_process(
    images: ImagesInput,
    network_input: NetworkInputDefinition,
) -> torch.Tensor:
    result_image, _ = pre_process_network_input(
        images=images,
        image_pre_processing=ImagePreProcessing(),
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="bgr",
    )

    return result_image


@pytest.mark.parametrize(
    "normalization",
    [None, ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
    ids=["no-normalization", "normalization"],
)
@pytest.mark.parametrize("resize_mode, input_kind", RESIZE_MODES_AND_INPUTS)
def test_pre_process_network_input_returns_float32_when_default_dtype_is_bfloat16(
    resize_mode: ResizeMode,
    input_kind: str,
    normalization: Optional[Tuple[List[float], List[float]]],
    bfloat16_default_dtype: None,
) -> None:
    # given
    network_input = _network_input(resize_mode=resize_mode, normalization=normalization)
    images = INPUT_FACTORIES[input_kind]()

    # when
    result_image = _pre_process(images=images, network_input=network_input)

    # then
    assert result_image.dtype == torch.float32


@pytest.mark.parametrize("resize_mode, input_kind", RESIZE_MODES_AND_INPUTS)
def test_pre_process_network_input_values_do_not_depend_on_default_dtype(
    resize_mode: ResizeMode,
    input_kind: str,
) -> None:
    # given
    network_input = _network_input(resize_mode=resize_mode, normalization=None)
    expected_image = _pre_process(
        images=INPUT_FACTORIES[input_kind](), network_input=network_input
    )
    original_dtype = torch.get_default_dtype()

    # when
    torch.set_default_dtype(torch.bfloat16)
    try:
        result_image = _pre_process(
            images=INPUT_FACTORIES[input_kind](), network_input=network_input
        )
    finally:
        torch.set_default_dtype(original_dtype)

    # then
    assert expected_image.dtype == torch.float32
    assert torch.equal(result_image, expected_image)
