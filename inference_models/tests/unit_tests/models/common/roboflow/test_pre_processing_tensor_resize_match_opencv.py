import numpy as np
import pytest
import torch

from inference_models.models.common.roboflow import (
    pre_processing as pre_processing_module,
)
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

RESIZE_MODES = [
    ResizeMode.STRETCH_TO,
    ResizeMode.LETTERBOX,
    ResizeMode.FIT_LONGER_EDGE,
]
LIST_RESIZE_MODES = [ResizeMode.STRETCH_TO, ResizeMode.LETTERBOX]
SIZES = [
    ((640, 640), (448, 448)),
    ((486, 480), (640, 640)),
    ((1280, 1280), (640, 640)),
    ((375, 517), (320, 320)),
]


def _network_input(resize_mode: ResizeMode, target_hw: tuple) -> NetworkInputDefinition:
    return NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=target_hw[0], width=target_hw[1]),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=resize_mode,
        padding_value=114,
        input_channels=3,
        scaling_factor=255,
    )


def _random_image(height: int, width: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(
        0, 256, (height, width, 3), dtype=np.uint8
    )


@pytest.fixture
def match_opencv_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pre_processing_module,
        "INFERENCE_MODELS_TENSOR_RESIZE_MATCH_OPENCV",
        True,
        raising=False,
    )


@pytest.fixture
def match_opencv_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pre_processing_module,
        "INFERENCE_MODELS_TENSOR_RESIZE_MATCH_OPENCV",
        False,
        raising=False,
    )


@pytest.mark.parametrize("resize_mode", RESIZE_MODES)
@pytest.mark.parametrize("source_hw, target_hw", SIZES)
def test_uint8_tensor_matches_numpy_input_when_flag_enabled(
    match_opencv_enabled: None,
    resize_mode: ResizeMode,
    source_hw: tuple,
    target_hw: tuple,
) -> None:
    # given
    network_input = _network_input(resize_mode=resize_mode, target_hw=target_hw)
    image = _random_image(*source_hw)
    tensor = torch.from_numpy(image).permute(2, 0, 1)

    # when
    numpy_result, numpy_meta = pre_process_network_input(
        images=image,
        image_pre_processing=ImagePreProcessing(),
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="bgr",
    )
    tensor_result, tensor_meta = pre_process_network_input(
        images=tensor,
        image_pre_processing=ImagePreProcessing(),
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="bgr",
    )

    # then
    assert tensor_result.dtype == numpy_result.dtype
    assert torch.equal(tensor_result, numpy_result)
    assert tensor_meta == numpy_meta


@pytest.mark.parametrize("resize_mode", LIST_RESIZE_MODES)
def test_list_of_uint8_tensors_matches_numpy_list_when_flag_enabled(
    match_opencv_enabled: None,
    resize_mode: ResizeMode,
) -> None:
    # given
    network_input = _network_input(resize_mode=resize_mode, target_hw=(640, 640))
    images = [_random_image(1000, 701, seed=1), _random_image(375, 517, seed=2)]
    tensors = [torch.from_numpy(image).permute(2, 0, 1) for image in images]

    # when
    numpy_result, _ = pre_process_network_input(
        images=images,
        image_pre_processing=ImagePreProcessing(),
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="bgr",
    )
    tensor_result, _ = pre_process_network_input(
        images=tensors,
        image_pre_processing=ImagePreProcessing(),
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="bgr",
    )

    # then
    assert torch.equal(tensor_result, numpy_result)


@pytest.mark.parametrize("resize_mode", RESIZE_MODES)
def test_uint8_tensor_resize_unchanged_when_flag_disabled(
    match_opencv_disabled: None,
    resize_mode: ResizeMode,
) -> None:
    # given
    network_input = _network_input(resize_mode=resize_mode, target_hw=(448, 448))
    tensor = torch.from_numpy(_random_image(640, 600)).permute(2, 0, 1)
    resized_by_torch = torch.nn.functional.interpolate(
        tensor.unsqueeze(0), size=[448, 420], mode="bilinear"
    )

    # when
    result, _ = pre_process_network_input(
        images=tensor,
        image_pre_processing=ImagePreProcessing(),
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="bgr",
    )

    # then
    if resize_mode is ResizeMode.STRETCH_TO:
        expected = torch.nn.functional.interpolate(
            tensor.unsqueeze(0), size=[448, 448], mode="bilinear"
        )
        assert torch.equal(result, expected / 255)
    elif resize_mode is ResizeMode.LETTERBOX:
        assert torch.equal(result[:, :, :, 14:434], resized_by_torch / 255)
    else:
        assert torch.equal(result, resized_by_torch / 255)


@pytest.mark.parametrize("resize_mode", RESIZE_MODES)
def test_float_tensor_resize_unchanged_when_flag_enabled(
    match_opencv_enabled: None,
    resize_mode: ResizeMode,
) -> None:
    # given
    network_input = _network_input(resize_mode=resize_mode, target_hw=(448, 448))
    tensor = torch.from_numpy(_random_image(448, 448)).permute(2, 0, 1).float()
    tensor = torch.nn.functional.pad(tensor, (0, 0, 0, 32))

    # when
    result, _ = pre_process_network_input(
        images=tensor,
        image_pre_processing=ImagePreProcessing(),
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="bgr",
    )

    # then
    target_hw = [448, 448] if resize_mode is ResizeMode.STRETCH_TO else [448, 418]
    expected = torch.nn.functional.interpolate(
        tensor.unsqueeze(0), size=target_hw, mode="bilinear"
    )
    if resize_mode is ResizeMode.LETTERBOX:
        result = result[:, :, :, 15:433]
    assert torch.equal(result, expected / 255)
