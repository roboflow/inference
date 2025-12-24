import numpy as np
import pytest
import torch
from PIL.Image import Image

from inference_models.entities import ImageDimensions
from inference_models.errors import ModelRuntimeError
from inference_models.models.common.roboflow.model_packages import (
    ColorMode,
    Contrast,
    ContrastType,
    Grayscale,
    ImagePreProcessing,
    NetworkInputDefinition,
    PreProcessingMetadata,
    ResizeMode,
    StaticCrop,
    StaticCropOffset,
    TrainingInputSize,
)
from inference_models.models.common.roboflow.pre_processing import (
    extract_input_images_dimensions,
    images_to_pillow,
    make_the_value_divisible,
    pre_process_network_input,
)


def test_images_to_pillow_when_input_is_np_array_in_bgr_and_model_input_is_in_rgb() -> (
    None
):
    # given
    images = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    expected_result = (np.ones((192, 168, 3), dtype=np.uint8) * (30, 20, 10)).astype(
        np.uint8
    )

    # when
    result_images, images_dimensions = images_to_pillow(
        images=images,
    )

    # then
    assert len(result_images) == 1
    assert len(images_dimensions) == 1
    assert isinstance(result_images[0], Image)
    assert result_images[0].size == (168, 192)
    assert np.allclose(np.asarray(result_images[0]), expected_result)
    assert (images_dimensions[0].height, images_dimensions[0].width) == (192, 168)


def test_images_to_pillow_when_input_is_np_array_in_rgb_and_model_input_is_in_rgb() -> (
    None
):
    # given
    images = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    # when
    result_images, images_dimensions = images_to_pillow(
        images=images,
        input_color_format="rgb",
    )

    # then
    assert len(result_images) == 1
    assert len(images_dimensions) == 1
    assert isinstance(result_images[0], Image)
    assert result_images[0].size == (168, 192)
    assert np.allclose(np.asarray(result_images[0]), images)
    assert (images_dimensions[0].height, images_dimensions[0].width) == (192, 168)


def test_images_to_pillow_when_input_is_np_array_in_bgr_and_model_input_is_in_bgr() -> (
    None
):
    # given
    images = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    # when
    result_images, images_dimensions = images_to_pillow(
        images=images,
        model_color_format="bgr",
    )

    # then
    assert len(result_images) == 1
    assert len(images_dimensions) == 1
    assert isinstance(result_images[0], Image)
    assert result_images[0].size == (168, 192)
    assert np.allclose(np.asarray(result_images[0]), images)
    assert (images_dimensions[0].height, images_dimensions[0].width) == (192, 168)


def test_images_to_pillow_when_input_is_np_array_in_rgb_and_model_input_is_in_bgr() -> (
    None
):
    # given
    images = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    expected_result = (np.ones((192, 168, 3), dtype=np.uint8) * (30, 20, 10)).astype(
        np.uint8
    )

    # when
    result_images, images_dimensions = images_to_pillow(
        images=images,
        input_color_format="rgb",
        model_color_format="bgr",
    )

    # then
    assert len(result_images) == 1
    assert len(images_dimensions) == 1
    assert isinstance(result_images[0], Image)
    assert result_images[0].size == (168, 192)
    assert np.allclose(np.asarray(result_images[0]), expected_result)
    assert (images_dimensions[0].height, images_dimensions[0].width) == (192, 168)


def test_images_to_pillow_when_input_is_list_of_np_array_in_bgr_and_model_input_is_in_rgb() -> (
    None
):
    # given
    images = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    expected_result = (np.ones((192, 168, 3), dtype=np.uint8) * (30, 20, 10)).astype(
        np.uint8
    )

    # when
    result_images, images_dimensions = images_to_pillow(
        images=[images, images],
    )

    # then
    assert len(result_images) == 2
    assert len(images_dimensions) == 2
    assert isinstance(result_images[0], Image)
    assert isinstance(result_images[1], Image)
    assert result_images[0].size == (168, 192)
    assert result_images[1].size == (168, 192)
    assert np.allclose(np.asarray(result_images[0]), expected_result)
    assert np.allclose(np.asarray(result_images[1]), expected_result)
    assert (images_dimensions[0].height, images_dimensions[0].width) == (192, 168)
    assert (images_dimensions[1].height, images_dimensions[1].width) == (192, 168)


def test_images_to_pillow_when_input_is_list_of_np_array_in_rgb_and_model_input_is_in_rgb() -> (
    None
):
    # given
    images = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    # when
    result_images, images_dimensions = images_to_pillow(
        images=[images, images],
        input_color_format="rgb",
    )

    # then
    assert len(result_images) == 2
    assert len(images_dimensions) == 2
    assert isinstance(result_images[0], Image)
    assert isinstance(result_images[1], Image)
    assert result_images[0].size == (168, 192)
    assert result_images[1].size == (168, 192)
    assert np.allclose(np.asarray(result_images[0]), images)
    assert np.allclose(np.asarray(result_images[1]), images)
    assert (images_dimensions[0].height, images_dimensions[0].width) == (192, 168)
    assert (images_dimensions[1].height, images_dimensions[1].width) == (192, 168)


def test_images_to_pillow_when_input_is_list_of_np_array_in_bgr_and_model_input_is_in_bgr() -> (
    None
):
    # given
    images = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    # when
    result_images, images_dimensions = images_to_pillow(
        images=[images, images],
        model_color_format="bgr",
    )

    # then
    assert len(result_images) == 2
    assert len(images_dimensions) == 2
    assert isinstance(result_images[0], Image)
    assert isinstance(result_images[1], Image)
    assert result_images[0].size == (168, 192)
    assert result_images[1].size == (168, 192)
    assert np.allclose(np.asarray(result_images[0]), images)
    assert np.allclose(np.asarray(result_images[1]), images)
    assert (images_dimensions[0].height, images_dimensions[0].width) == (192, 168)
    assert (images_dimensions[1].height, images_dimensions[1].width) == (192, 168)


def test_images_to_pillow_when_input_is_list_of_np_array_in_rgb_and_model_input_is_in_bgr() -> (
    None
):
    # given
    images = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    expected_result = (np.ones((192, 168, 3), dtype=np.uint8) * (30, 20, 10)).astype(
        np.uint8
    )

    # when
    result_images, images_dimensions = images_to_pillow(
        images=[images, images],
        input_color_format="rgb",
        model_color_format="bgr",
    )

    # then
    assert len(result_images) == 2
    assert len(images_dimensions) == 2
    assert isinstance(result_images[0], Image)
    assert isinstance(result_images[1], Image)
    assert result_images[0].size == (168, 192)
    assert result_images[1].size == (168, 192)
    assert np.allclose(np.asarray(result_images[0]), expected_result)
    assert np.allclose(np.asarray(result_images[1]), expected_result)
    assert (images_dimensions[0].height, images_dimensions[0].width) == (192, 168)
    assert (images_dimensions[1].height, images_dimensions[1].width) == (192, 168)


def test_images_to_pillow_when_input_is_tensor_in_bgr_and_model_input_is_in_rgb() -> (
    None
):
    # given
    images = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(2, 0, 1)
    expected_result = (np.ones((192, 168, 3), dtype=np.uint8) * (30, 20, 10)).astype(
        np.uint8
    )

    # when
    result_images, images_dimensions = images_to_pillow(
        images=images, input_color_format="bgr"
    )

    # then
    assert len(result_images) == 1
    assert len(images_dimensions) == 1
    assert isinstance(result_images[0], Image)
    assert result_images[0].size == (168, 192)
    assert np.allclose(np.asarray(result_images[0]), expected_result)
    assert (images_dimensions[0].height, images_dimensions[0].width) == (192, 168)


def test_images_to_pillow_when_input_is_tensor_in_rgb_and_model_input_is_in_rgb() -> (
    None
):
    # given
    images = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(2, 0, 1)
    expected_result = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(
        np.uint8
    )

    # when
    result_images, images_dimensions = images_to_pillow(
        images=images,
    )

    # then
    assert len(result_images) == 1
    assert len(images_dimensions) == 1
    assert isinstance(result_images[0], Image)
    assert result_images[0].size == (168, 192)
    assert np.allclose(np.asarray(result_images[0]), expected_result)
    assert (images_dimensions[0].height, images_dimensions[0].width) == (192, 168)


def test_images_to_pillow_when_input_is_tensor_in_bgr_and_model_input_is_in_bgr() -> (
    None
):
    # given
    images = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(2, 0, 1)
    expected_result = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(
        np.uint8
    )

    # when
    result_images, images_dimensions = images_to_pillow(
        images=images,
        input_color_format="bgr",
        model_color_format="bgr",
    )

    # then
    assert len(result_images) == 1
    assert len(images_dimensions) == 1
    assert isinstance(result_images[0], Image)
    assert result_images[0].size == (168, 192)
    assert np.allclose(np.asarray(result_images[0]), expected_result)
    assert (images_dimensions[0].height, images_dimensions[0].width) == (192, 168)


def test_images_to_pillow_when_input_is_tensor_in_rgb_and_model_input_is_in_bgr() -> (
    None
):
    # given
    images = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(2, 0, 1)
    expected_result = (np.ones((192, 168, 3), dtype=np.uint8) * (30, 20, 10)).astype(
        np.uint8
    )

    # when
    result_images, images_dimensions = images_to_pillow(
        images=images,
        input_color_format="rgb",
        model_color_format="bgr",
    )

    # then
    assert len(result_images) == 1
    assert len(images_dimensions) == 1
    assert isinstance(result_images[0], Image)
    assert result_images[0].size == (168, 192)
    assert np.allclose(np.asarray(result_images[0]), expected_result)
    assert (images_dimensions[0].height, images_dimensions[0].width) == (192, 168)


def test_images_to_pillow_when_input_is_tensor_in_bgr_and_model_input_is_in_rgb() -> (
    None
):
    # given
    images = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(2, 0, 1)
    expected_result = (np.ones((192, 168, 3), dtype=np.uint8) * (30, 20, 10)).astype(
        np.uint8
    )

    # when
    result_images, images_dimensions = images_to_pillow(
        images=images, input_color_format="bgr"
    )

    # then
    assert len(result_images) == 1
    assert len(images_dimensions) == 1
    assert isinstance(result_images[0], Image)
    assert result_images[0].size == (168, 192)
    assert np.allclose(np.asarray(result_images[0]), expected_result)
    assert (images_dimensions[0].height, images_dimensions[0].width) == (192, 168)


def test_images_to_pillow_when_input_is_list_of_tensors_in_rgb_and_model_input_is_in_rgb() -> (
    None
):
    # given
    images = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(2, 0, 1)
    expected_result = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(
        np.uint8
    )

    # when
    result_images, images_dimensions = images_to_pillow(
        images=[images, images],
    )

    # then
    assert len(result_images) == 2
    assert len(images_dimensions) == 2
    assert isinstance(result_images[0], Image)
    assert isinstance(result_images[1], Image)
    assert result_images[0].size == (168, 192)
    assert result_images[1].size == (168, 192)
    assert np.allclose(np.asarray(result_images[0]), expected_result)
    assert np.allclose(np.asarray(result_images[1]), expected_result)
    assert (images_dimensions[0].height, images_dimensions[0].width) == (192, 168)
    assert (images_dimensions[1].height, images_dimensions[1].width) == (192, 168)


def test_images_to_pillow_when_input_is_list_of_tensors_in_bgr_and_model_input_is_in_bgr() -> (
    None
):
    # given
    images = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(2, 0, 1)
    expected_result = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(
        np.uint8
    )

    # when
    result_images, images_dimensions = images_to_pillow(
        images=[images, images],
        input_color_format="bgr",
        model_color_format="bgr",
    )

    # then
    assert len(result_images) == 2
    assert len(images_dimensions) == 2
    assert isinstance(result_images[0], Image)
    assert isinstance(result_images[1], Image)
    assert result_images[0].size == (168, 192)
    assert result_images[1].size == (168, 192)
    assert np.allclose(np.asarray(result_images[0]), expected_result)
    assert np.allclose(np.asarray(result_images[1]), expected_result)
    assert (images_dimensions[0].height, images_dimensions[0].width) == (192, 168)
    assert (images_dimensions[1].height, images_dimensions[1].width) == (192, 168)


def test_images_to_pillow_when_input_is_list_of_tensors_in_rgb_and_model_input_is_in_bgr() -> (
    None
):
    # given
    images = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(2, 0, 1)
    expected_result = (np.ones((192, 168, 3), dtype=np.uint8) * (30, 20, 10)).astype(
        np.uint8
    )

    # when
    result_images, images_dimensions = images_to_pillow(
        images=[images, images],
        input_color_format="rgb",
        model_color_format="bgr",
    )

    # then
    assert len(result_images) == 2
    assert len(images_dimensions) == 2
    assert isinstance(result_images[0], Image)
    assert isinstance(result_images[1], Image)
    assert result_images[0].size == (168, 192)
    assert result_images[1].size == (168, 192)
    assert np.allclose(np.asarray(result_images[0]), expected_result)
    assert np.allclose(np.asarray(result_images[1]), expected_result)
    assert (images_dimensions[0].height, images_dimensions[0].width) == (192, 168)
    assert (images_dimensions[1].height, images_dimensions[1].width) == (192, 168)


def test_images_to_pillow_when_input_is_multi_element_tensor_in_bgr_and_model_input_is_in_rgb() -> (
    None
):
    # given
    images = torch.from_numpy(
        (np.ones((2, 192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(0, 3, 1, 2)
    expected_result = (np.ones((192, 168, 3), dtype=np.uint8) * (30, 20, 10)).astype(
        np.uint8
    )

    # when
    result_images, images_dimensions = images_to_pillow(
        images=images, input_color_format="bgr"
    )

    # then
    assert len(result_images) == 2
    assert len(images_dimensions) == 2
    assert isinstance(result_images[0], Image)
    assert isinstance(result_images[1], Image)
    assert result_images[0].size == (168, 192)
    assert result_images[1].size == (168, 192)
    assert np.allclose(np.asarray(result_images[0]), expected_result)
    assert np.allclose(np.asarray(result_images[1]), expected_result)
    assert (images_dimensions[0].height, images_dimensions[0].width) == (192, 168)
    assert (images_dimensions[1].height, images_dimensions[1].width) == (192, 168)


def test_images_to_pillow_when_input_is_multi_element_tensor_in_rgb_and_model_input_is_in_rgb() -> (
    None
):
    # given
    images = torch.from_numpy(
        (np.ones((2, 192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(0, 3, 1, 2)
    expected_result = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(
        np.uint8
    )

    # when
    result_images, images_dimensions = images_to_pillow(
        images=images,
    )

    # then
    assert len(result_images) == 2
    assert len(images_dimensions) == 2
    assert isinstance(result_images[0], Image)
    assert isinstance(result_images[1], Image)
    assert result_images[0].size == (168, 192)
    assert result_images[1].size == (168, 192)
    assert np.allclose(np.asarray(result_images[0]), expected_result)
    assert np.allclose(np.asarray(result_images[1]), expected_result)
    assert (images_dimensions[0].height, images_dimensions[0].width) == (192, 168)
    assert (images_dimensions[1].height, images_dimensions[1].width) == (192, 168)


def test_images_to_pillow_when_input_is_multi_element_tensor_in_bgr_and_model_input_is_in_bgr() -> (
    None
):
    # given
    images = torch.from_numpy(
        (np.ones((2, 192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(0, 3, 1, 2)
    expected_result = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(
        np.uint8
    )

    # when
    result_images, images_dimensions = images_to_pillow(
        images=images,
        input_color_format="bgr",
        model_color_format="bgr",
    )

    # then
    assert len(result_images) == 2
    assert len(images_dimensions) == 2
    assert isinstance(result_images[0], Image)
    assert isinstance(result_images[1], Image)
    assert result_images[0].size == (168, 192)
    assert result_images[1].size == (168, 192)
    assert np.allclose(np.asarray(result_images[0]), expected_result)
    assert np.allclose(np.asarray(result_images[1]), expected_result)
    assert (images_dimensions[0].height, images_dimensions[0].width) == (192, 168)
    assert (images_dimensions[1].height, images_dimensions[1].width) == (192, 168)


def test_images_to_pillow_when_input_is_multi_element_tensor_in_rgb_and_model_input_is_in_bgr() -> (
    None
):
    # given
    images = torch.from_numpy(
        (np.ones((2, 192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(0, 3, 1, 2)
    expected_result = (np.ones((192, 168, 3), dtype=np.uint8) * (30, 20, 10)).astype(
        np.uint8
    )

    # when
    result_images, images_dimensions = images_to_pillow(
        images=images,
        input_color_format="rgb",
        model_color_format="bgr",
    )

    # then
    assert len(result_images) == 2
    assert len(images_dimensions) == 2
    assert isinstance(result_images[0], Image)
    assert isinstance(result_images[1], Image)
    assert result_images[0].size == (168, 192)
    assert result_images[1].size == (168, 192)
    assert np.allclose(np.asarray(result_images[0]), expected_result)
    assert np.allclose(np.asarray(result_images[1]), expected_result)
    assert (images_dimensions[0].height, images_dimensions[0].width) == (192, 168)
    assert (images_dimensions[1].height, images_dimensions[1].width) == (192, 168)


def test_images_to_pillow_when_empty_batch_provided() -> None:
    # when
    with pytest.raises(ModelRuntimeError):
        _ = images_to_pillow(images=[])


def test_images_to_pillow_when_invalid_input_provided() -> None:
    # when
    with pytest.raises(ModelRuntimeError):
        _ = images_to_pillow(images="invalid")


def test_images_to_pillow_when_ilist_of_nvalid_input_provided() -> None:
    # when
    with pytest.raises(ModelRuntimeError):
        _ = images_to_pillow(images=["invalid"])


def test_extract_input_images_dimensions_when_numpy_image_provided() -> None:
    # given
    images = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    # when
    result = extract_input_images_dimensions(images=images)

    # then
    assert len(result) == 1
    assert result[0] == ImageDimensions(height=192, width=168)


def test_extract_input_images_dimensions_when_list_of_numpy_images_provided() -> None:
    # given
    images = [
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8),
        (np.ones((480, 640, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8),
    ]

    # when
    result = extract_input_images_dimensions(images=images)

    # then
    assert len(result) == 2
    assert result[0] == ImageDimensions(height=192, width=168)
    assert result[1] == ImageDimensions(height=480, width=640)


def test_extract_input_images_dimensions_when_tensor_provided() -> None:
    # given
    images = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(2, 0, 1)

    # when
    result = extract_input_images_dimensions(images=images)

    # then
    assert len(result) == 1
    assert result[0] == ImageDimensions(height=192, width=168)


def test_extract_input_images_dimensions_when_multi_image_tensor_provided() -> None:
    # given
    images = torch.from_numpy(
        (np.ones((2, 192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(0, 3, 1, 2)

    # when
    result = extract_input_images_dimensions(images=images)

    # then
    assert len(result) == 2
    assert result[0] == ImageDimensions(height=192, width=168)
    assert result[1] == ImageDimensions(height=192, width=168)


def test_extract_input_images_dimensions_when_list_of_tensor_images_provided() -> None:
    # given
    image_1 = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(2, 0, 1)
    image_2 = torch.from_numpy(
        (np.ones((480, 640, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(2, 0, 1)

    # when
    result = extract_input_images_dimensions(images=[image_1, image_2])

    # then
    assert len(result) == 2
    assert result[0] == ImageDimensions(height=192, width=168)
    assert result[1] == ImageDimensions(height=480, width=640)


def test_extract_input_images_dimensions_when_empty_batch_provided() -> None:
    # when
    with pytest.raises(ModelRuntimeError):
        _ = extract_input_images_dimensions(images=[])


def test_extract_input_images_dimensions_when_invalid_data_provided() -> None:
    # when
    with pytest.raises(ModelRuntimeError):
        _ = extract_input_images_dimensions(images="some")


def test_extract_input_images_dimensions_when_invalid_batch_element_provided() -> None:
    # when
    with pytest.raises(ModelRuntimeError):
        _ = extract_input_images_dimensions(images=["some"])


def test_pre_process_numpy_image_with_stretch() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0][0] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][2] == 10)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_numpy_images_list_with_stretch() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][0] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][2] == 10)
    assert torch.all(result_image[1][0] == 30)
    assert torch.all(result_image[1][1] == 20)
    assert torch.all(result_image[1][2] == 10)
    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_3d_torch_image_with_stretch() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    )
    image = torch.permute(image, (2, 0, 1))

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0][0] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][2] == 10)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_3d_torch_image_not_permuted_with_stretch() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    )

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0][0] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][2] == 10)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_list_of_3d_torch_image_with_stretch() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    )
    image = torch.permute(image, (2, 0, 1))

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][0] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][2] == 10)
    assert torch.all(result_image[1][0] == 30)
    assert torch.all(result_image[1][1] == 20)
    assert torch.all(result_image[1][2] == 10)
    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_list_of_3d_not_permuted_torch_image_with_stretch() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    )

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][0] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][2] == 10)
    assert torch.all(result_image[1][0] == 30)
    assert torch.all(result_image[1][1] == 20)
    assert torch.all(result_image[1][2] == 10)
    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_list_of_4d_torch_image_with_stretch() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    )
    image = torch.permute(image, (2, 0, 1))
    image = torch.unsqueeze(image, dim=0)

    # when
    with pytest.raises(ModelRuntimeError):
        _ = pre_process_network_input(
            images=[image, image],
            image_pre_processing=image_pre_processing,
            network_input=network_input,
            target_device=torch.device("cpu"),
            input_color_format="rgb",
        )


def test_pre_process_4d_torch_image_with_stretch() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    )
    image = torch.permute(image, (2, 0, 1))
    image = torch.stack([image, image], dim=0)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][0] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][2] == 10)
    assert torch.all(result_image[1][0] == 30)
    assert torch.all(result_image[1][1] == 20)
    assert torch.all(result_image[1][2] == 10)
    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_4d_torch_image_not_permuted_with_stretch() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    )
    image = torch.stack([image, image], dim=0)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][0] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][2] == 10)
    assert torch.all(result_image[1][0] == 30)
    assert torch.all(result_image[1][1] == 20)
    assert torch.all(result_image[1][2] == 10)
    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_numpy_image_with_stretch_and_crop() -> None:
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = np.ones((200, 100, 3), dtype=np.uint8)
    image[40:160, 10:90, :] = (image[40:160, 10:90, :] * (10, 20, 30)).astype(np.uint8)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0][2] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][0] == 10)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=200, width=100),
        size_after_pre_processing=ImageDimensions(height=120, width=80),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 80,
        scale_height=64 / 120,
        static_crop_offset=StaticCropOffset(
            offset_x=10,
            offset_y=40,
            crop_width=80,
            crop_height=120,
        ),
    )


def test_pre_process_numpy_images_list_with_stretch_and_crop() -> None:
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = np.ones((200, 100, 3), dtype=np.uint8)
    image[40:160, 10:90, :] = (image[40:160, 10:90, :] * (10, 20, 30)).astype(np.uint8)

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][2] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][0] == 10)
    assert torch.all(result_image[1][2] == 30)
    assert torch.all(result_image[1][1] == 20)
    assert torch.all(result_image[1][0] == 10)
    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=200, width=100),
        size_after_pre_processing=ImageDimensions(height=120, width=80),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 80,
        scale_height=64 / 120,
        static_crop_offset=StaticCropOffset(
            offset_x=10,
            offset_y=40,
            crop_width=80,
            crop_height=120,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_3d_image_with_stretch_and_crop() -> None:
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = np.ones((200, 100, 3), dtype=np.uint8)
    image[40:160, 10:90, :] = (image[40:160, 10:90, :] * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0][2] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][0] == 10)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=200, width=100),
        size_after_pre_processing=ImageDimensions(height=120, width=80),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 80,
        scale_height=64 / 120,
        static_crop_offset=StaticCropOffset(
            offset_x=10,
            offset_y=40,
            crop_width=80,
            crop_height=120,
        ),
    )


def test_pre_process_list_of_torch_3d_image_with_stretch_and_crop() -> None:
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = np.ones((200, 100, 3), dtype=np.uint8)
    image[40:160, 10:90, :] = (image[40:160, 10:90, :] * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][2] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][0] == 10)
    assert torch.all(result_image[1][2] == 30)
    assert torch.all(result_image[1][1] == 20)
    assert torch.all(result_image[1][0] == 10)
    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=200, width=100),
        size_after_pre_processing=ImageDimensions(height=120, width=80),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 80,
        scale_height=64 / 120,
        static_crop_offset=StaticCropOffset(
            offset_x=10,
            offset_y=40,
            crop_width=80,
            crop_height=120,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_list_of_torch_3d_not_permuted_image_with_stretch_and_crop() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = np.ones((200, 100, 3), dtype=np.uint8)
    image[40:160, 10:90, :] = (image[40:160, 10:90, :] * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][2] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][0] == 10)
    assert torch.all(result_image[1][2] == 30)
    assert torch.all(result_image[1][1] == 20)
    assert torch.all(result_image[1][0] == 10)
    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=200, width=100),
        size_after_pre_processing=ImageDimensions(height=120, width=80),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 80,
        scale_height=64 / 120,
        static_crop_offset=StaticCropOffset(
            offset_x=10,
            offset_y=40,
            crop_width=80,
            crop_height=120,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_list_of_torch_4d_image_with_stretch_and_crop() -> None:
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = np.ones((200, 100, 3), dtype=np.uint8)
    image[40:160, 10:90, :] = (image[40:160, 10:90, :] * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))
    image = torch.unsqueeze(image, dim=0)

    # when
    with pytest.raises(ModelRuntimeError):
        _ = pre_process_network_input(
            images=[image, image],
            image_pre_processing=image_pre_processing,
            network_input=network_input,
            target_device=torch.device("cpu"),
            input_color_format="rgb",
        )


def test_pre_process_torch_3d_not_permuted_image_with_stretch_and_crop() -> None:
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = np.ones((200, 100, 3), dtype=np.uint8)
    image[40:160, 10:90, :] = (image[40:160, 10:90, :] * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0][2] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][0] == 10)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=200, width=100),
        size_after_pre_processing=ImageDimensions(height=120, width=80),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 80,
        scale_height=64 / 120,
        static_crop_offset=StaticCropOffset(
            offset_x=10,
            offset_y=40,
            crop_width=80,
            crop_height=120,
        ),
    )


def test_pre_process_torch_4d_image_with_stretch_and_crop() -> None:
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = np.ones((200, 100, 3), dtype=np.uint8)
    image[40:160, 10:90, :] = (image[40:160, 10:90, :] * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))
    image = torch.stack([image, image], dim=0)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][2] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][0] == 10)
    assert torch.all(result_image[1][2] == 30)
    assert torch.all(result_image[1][1] == 20)
    assert torch.all(result_image[1][0] == 10)
    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=200, width=100),
        size_after_pre_processing=ImageDimensions(height=120, width=80),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 80,
        scale_height=64 / 120,
        static_crop_offset=StaticCropOffset(
            offset_x=10,
            offset_y=40,
            crop_width=80,
            crop_height=120,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_4d_not_permuted_image_with_stretch_and_crop() -> None:
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = np.ones((200, 100, 3), dtype=np.uint8)
    image[40:160, 10:90, :] = (image[40:160, 10:90, :] * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.stack([image, image], dim=0)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][2] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][0] == 10)
    assert torch.all(result_image[1][2] == 30)
    assert torch.all(result_image[1][1] == 20)
    assert torch.all(result_image[1][0] == 10)
    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=200, width=100),
        size_after_pre_processing=ImageDimensions(height=120, width=80),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 80,
        scale_height=64 / 120,
        static_crop_offset=StaticCropOffset(
            offset_x=10,
            offset_y=40,
            crop_width=80,
            crop_height=120,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_numpy_image_with_stretch_and_rescaling() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0][0] == 3.0)
    assert torch.all(result_image[0][1] == 2.0)
    assert torch.all(result_image[0][2] == 1.0)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_numpy_images_list_with_stretch_and_rescaling() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][0] == 3.0)
    assert torch.all(result_image[0][1] == 2.0)
    assert torch.all(result_image[0][2] == 1.0)
    assert torch.all(result_image[1][0] == 3.0)
    assert torch.all(result_image[1][1] == 2.0)
    assert torch.all(result_image[1][2] == 1.0)
    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[0] == expected_meta


def test_pre_process_torch_3d_image_with_stretch_and_rescaling() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0][0] == 3.0)
    assert torch.all(result_image[0][1] == 2.0)
    assert torch.all(result_image[0][2] == 1.0)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_torch_3d_not_permuted_image_with_stretch_and_rescaling() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0][0] == 3.0)
    assert torch.all(result_image[0][1] == 2.0)
    assert torch.all(result_image[0][2] == 1.0)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_list_of_torch_3d_image_with_stretch_and_rescaling() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][0] == 3.0)
    assert torch.all(result_image[0][1] == 2.0)
    assert torch.all(result_image[0][2] == 1.0)
    assert torch.all(result_image[1][0] == 3.0)
    assert torch.all(result_image[1][1] == 2.0)
    assert torch.all(result_image[1][2] == 1.0)
    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_list_of_torch_3d_image_not_permuted_with_stretch_and_rescaling() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][0] == 3.0)
    assert torch.all(result_image[0][1] == 2.0)
    assert torch.all(result_image[0][2] == 1.0)
    assert torch.all(result_image[1][0] == 3.0)
    assert torch.all(result_image[1][1] == 2.0)
    assert torch.all(result_image[1][2] == 1.0)
    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_list_of_torch_4d_image_with_stretch_and_rescaling() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))
    image = torch.unsqueeze(image, dim=0)

    # when
    with pytest.raises(ModelRuntimeError):
        _ = pre_process_network_input(
            images=[image, image],
            image_pre_processing=image_pre_processing,
            network_input=network_input,
            target_device=torch.device("cpu"),
            input_color_format="rgb",
        )


def test_pre_process_torch_4d_image_with_stretch_and_rescaling() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))
    image = torch.stack([image, image], dim=0)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][0] == 3.0)
    assert torch.all(result_image[0][1] == 2.0)
    assert torch.all(result_image[0][2] == 1.0)
    assert torch.all(result_image[1][0] == 3.0)
    assert torch.all(result_image[1][1] == 2.0)
    assert torch.all(result_image[1][2] == 1.0)
    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_numpy_image_with_stretch_rescaling_and_normalization() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
        scaling_factor=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0][0] == 1 / 6)
    assert torch.all(result_image[0][1] == 0.0)
    assert torch.all(result_image[0][2] == -1 / 6)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_numpy_images_list_with_stretch_rescaling_and_normalization() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
        scaling_factor=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][0] == 1 / 6)
    assert torch.all(result_image[0][1] == 0.0)
    assert torch.all(result_image[0][2] == -1 / 6)
    assert torch.all(result_image[1][0] == 1 / 6)
    assert torch.all(result_image[1][1] == 0.0)
    assert torch.all(result_image[1][2] == -1 / 6)
    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_3d_image_with_stretch_rescaling_and_normalization() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
        scaling_factor=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0][0] == 1 / 6)
    assert torch.all(result_image[0][1] == 0.0)
    assert torch.all(result_image[0][2] == -1 / 6)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_list_of_torch_3d_image_with_stretch_rescaling_and_normalization() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
        scaling_factor=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][0] == 1 / 6)
    assert torch.all(result_image[0][1] == 0.0)
    assert torch.all(result_image[0][2] == -1 / 6)
    assert torch.all(result_image[1][0] == 1 / 6)
    assert torch.all(result_image[1][1] == 0.0)
    assert torch.all(result_image[1][2] == -1 / 6)
    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_list_of_torch_3d_image_not_permuted_with_stretch_rescaling_and_normalization() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
        scaling_factor=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][0] == 1 / 6)
    assert torch.all(result_image[0][1] == 0.0)
    assert torch.all(result_image[0][2] == -1 / 6)
    assert torch.all(result_image[1][0] == 1 / 6)
    assert torch.all(result_image[1][1] == 0.0)
    assert torch.all(result_image[1][2] == -1 / 6)
    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_list_of_torch_4d_image_with_stretch_rescaling_and_normalization() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
        scaling_factor=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))
    image = torch.unsqueeze(image, dim=0)

    # when
    with pytest.raises(ModelRuntimeError):
        _ = pre_process_network_input(
            images=[image, image],
            image_pre_processing=image_pre_processing,
            network_input=network_input,
            target_device=torch.device("cpu"),
            input_color_format="rgb",
        )


def test_pre_process_torch_3d_not_permuted_image_with_stretch_rescaling_and_normalization() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
        scaling_factor=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0][0] == 1 / 6)
    assert torch.all(result_image[0][1] == 0.0)
    assert torch.all(result_image[0][2] == -1 / 6)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_torch_4d_image_with_stretch_rescaling_and_normalization() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
        scaling_factor=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))
    image = torch.stack([image, image], dim=0)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][0] == 1 / 6)
    assert torch.all(result_image[0][1] == 0.0)
    assert torch.all(result_image[0][2] == -1 / 6)
    assert torch.all(result_image[1][0] == 1 / 6)
    assert torch.all(result_image[1][1] == 0.0)
    assert torch.all(result_image[1][2] == -1 / 6)
    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_4d_image_not_permuted_with_stretch_rescaling_and_normalization() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
        scaling_factor=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.stack([image, image], dim=0)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][0] == 1 / 6)
    assert torch.all(result_image[0][1] == 0.0)
    assert torch.all(result_image[0][2] == -1 / 6)
    assert torch.all(result_image[1][0] == 1 / 6)
    assert torch.all(result_image[1][1] == 0.0)
    assert torch.all(result_image[1][2] == -1 / 6)
    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_numpy_image_with_letterbox_selected() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0, :, :, :20] == 0)
    assert torch.all(result_image[0, :, :, 76:] == 0)
    assert torch.all(result_image[0, 0, :, 20:76] == 30)
    assert torch.all(result_image[0, 1, :, 20:76] == 20)
    assert torch.all(result_image[0, 2, :, 20:76] == 10)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=20,
        pad_top=0,
        pad_right=20,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=64 / 192,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_numpy_images_list_with_letterbox_selected() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0, :, :, :20] == 0)
    assert torch.all(result_image[0, :, :, 76:] == 0)
    assert torch.all(result_image[0, 0, :, 20:76] == 30)
    assert torch.all(result_image[0, 1, :, 20:76] == 20)
    assert torch.all(result_image[0, 2, :, 20:76] == 10)
    assert torch.all(result_image[1, :, :, :20] == 0)
    assert torch.all(result_image[1, :, :, 76:] == 0)
    assert torch.all(result_image[1, 0, :, 20:76] == 30)
    assert torch.all(result_image[1, 1, :, 20:76] == 20)
    assert torch.all(result_image[1, 2, :, 20:76] == 10)
    expected_meta = PreProcessingMetadata(
        pad_left=20,
        pad_top=0,
        pad_right=20,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=64 / 192,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_3d_image_with_letterbox_selected() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0, :, :, :20] == 0)
    assert torch.all(result_image[0, :, :, 76:] == 0)
    assert torch.all(result_image[0, 0, :, 20:76] == 30)
    assert torch.all(result_image[0, 1, :, 20:76] == 20)
    assert torch.all(result_image[0, 2, :, 20:76] == 10)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=20,
        pad_top=0,
        pad_right=20,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=64 / 192,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_torch_3d_not_permuted_image_with_letterbox_selected() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0, :, :, :20] == 0)
    assert torch.all(result_image[0, :, :, 76:] == 0)
    assert torch.all(result_image[0, 0, :, 20:76] == 30)
    assert torch.all(result_image[0, 1, :, 20:76] == 20)
    assert torch.all(result_image[0, 2, :, 20:76] == 10)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=20,
        pad_top=0,
        pad_right=20,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=64 / 192,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_list_of_torch_3d_image_with_letterbox_selected() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0, :, :, :20] == 0)
    assert torch.all(result_image[0, :, :, 76:] == 0)
    assert torch.all(result_image[0, 0, :, 20:76] == 30)
    assert torch.all(result_image[0, 1, :, 20:76] == 20)
    assert torch.all(result_image[0, 2, :, 20:76] == 10)
    assert torch.all(result_image[1, :, :, :20] == 0)
    assert torch.all(result_image[1, :, :, 76:] == 0)
    assert torch.all(result_image[1, 0, :, 20:76] == 30)
    assert torch.all(result_image[1, 1, :, 20:76] == 20)
    assert torch.all(result_image[1, 2, :, 20:76] == 10)
    assert abs(result_meta[0].scale_width - 1 / 3) < 1e-5
    assert abs(result_meta[0].scale_height - 1 / 3) < 1e-5
    expected_meta = PreProcessingMetadata(
        pad_left=20,
        pad_top=0,
        pad_right=20,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=result_meta[0].scale_width,
        scale_height=result_meta[0].scale_height,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_list_of_torch_3d_image_not_permuted_with_letterbox_selected() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0, :, :, :20] == 0)
    assert torch.all(result_image[0, :, :, 76:] == 0)
    assert torch.all(result_image[0, 0, :, 20:76] == 30)
    assert torch.all(result_image[0, 1, :, 20:76] == 20)
    assert torch.all(result_image[0, 2, :, 20:76] == 10)
    assert torch.all(result_image[1, :, :, :20] == 0)
    assert torch.all(result_image[1, :, :, 76:] == 0)
    assert torch.all(result_image[1, 0, :, 20:76] == 30)
    assert torch.all(result_image[1, 1, :, 20:76] == 20)
    assert torch.all(result_image[1, 2, :, 20:76] == 10)
    assert abs(result_meta[0].scale_width - 1 / 3) < 1e-5
    assert abs(result_meta[0].scale_height - 1 / 3) < 1e-5
    expected_meta = PreProcessingMetadata(
        pad_left=20,
        pad_top=0,
        pad_right=20,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=result_meta[0].scale_width,
        scale_height=result_meta[0].scale_height,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_list_of_torch_4d_image_with_letterbox_selected() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))
    image = torch.unsqueeze(image, dim=0)

    # when
    with pytest.raises(ModelRuntimeError):
        _ = pre_process_network_input(
            images=[image, image],
            image_pre_processing=image_pre_processing,
            network_input=network_input,
            target_device=torch.device("cpu"),
        )


def test_pre_process_torch_4d_image_with_letterbox_selected() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))
    image = torch.stack([image, image], dim=0)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0, :, :, :20] == 0)
    assert torch.all(result_image[0, :, :, 76:] == 0)
    assert torch.all(result_image[0, 0, :, 20:76] == 30)
    assert torch.all(result_image[0, 1, :, 20:76] == 20)
    assert torch.all(result_image[0, 2, :, 20:76] == 10)
    assert torch.all(result_image[1, :, :, :20] == 0)
    assert torch.all(result_image[1, :, :, 76:] == 0)
    assert torch.all(result_image[1, 0, :, 20:76] == 30)
    assert torch.all(result_image[1, 1, :, 20:76] == 20)
    assert torch.all(result_image[1, 2, :, 20:76] == 10)
    expected_meta = PreProcessingMetadata(
        pad_left=20,
        pad_top=0,
        pad_right=20,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=64 / 192,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_4d_not_permuted_image_with_letterbox_selected() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.stack([image, image], dim=0)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0, :, :, :20] == 0)
    assert torch.all(result_image[0, :, :, 76:] == 0)
    assert torch.all(result_image[0, 0, :, 20:76] == 30)
    assert torch.all(result_image[0, 1, :, 20:76] == 20)
    assert torch.all(result_image[0, 2, :, 20:76] == 10)
    assert torch.all(result_image[1, :, :, :20] == 0)
    assert torch.all(result_image[1, :, :, 76:] == 0)
    assert torch.all(result_image[1, 0, :, 20:76] == 30)
    assert torch.all(result_image[1, 1, :, 20:76] == 20)
    assert torch.all(result_image[1, 2, :, 20:76] == 10)
    expected_meta = PreProcessingMetadata(
        pad_left=20,
        pad_top=0,
        pad_right=20,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=64 / 192,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_numpy_image_with_letterbox_and_static_crop_selected() -> None:
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
    )
    image = np.ones((200, 100, 3), dtype=np.uint8)
    image[40:160, 10:90, :] = (image[40:160, 10:90, :] * (10, 20, 30)).astype(np.uint8)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0, :, :, :27] == 0)
    assert torch.all(result_image[0, :, :, 69:] == 0)
    assert torch.all(result_image[0, 0, :, 27:69] == 30)
    assert torch.all(result_image[0, 1, :, 27:69] == 20)
    assert torch.all(result_image[0, 2, :, 27:69] == 10)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=27,
        pad_top=0,
        pad_right=27,
        pad_bottom=0,
        original_size=ImageDimensions(height=200, width=100),
        size_after_pre_processing=ImageDimensions(height=120, width=80),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=64 / 120,
        scale_height=64 / 120,
        static_crop_offset=StaticCropOffset(
            offset_x=10,
            offset_y=40,
            crop_width=80,
            crop_height=120,
        ),
    )


def test_pre_process_numpy_images_list_with_letterbox_and_static_crop_selected() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
    )
    image = np.ones((200, 100, 3), dtype=np.uint8)
    image[40:160, 10:90, :] = (image[40:160, 10:90, :] * (10, 20, 30)).astype(np.uint8)

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0, :, :, :27] == 0)
    assert torch.all(result_image[0, :, :, 69:] == 0)
    assert torch.all(result_image[0, 0, :, 27:69] == 30)
    assert torch.all(result_image[0, 1, :, 27:69] == 20)
    assert torch.all(result_image[0, 2, :, 27:69] == 10)
    assert torch.all(result_image[1, :, :, :27] == 0)
    assert torch.all(result_image[1, :, :, 69:] == 0)
    assert torch.all(result_image[1, 0, :, 27:69] == 30)
    assert torch.all(result_image[1, 1, :, 27:69] == 20)
    assert torch.all(result_image[1, 2, :, 27:69] == 10)
    expected_meta = PreProcessingMetadata(
        pad_left=27,
        pad_top=0,
        pad_right=27,
        pad_bottom=0,
        original_size=ImageDimensions(height=200, width=100),
        size_after_pre_processing=ImageDimensions(height=120, width=80),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=64 / 120,
        scale_height=64 / 120,
        static_crop_offset=StaticCropOffset(
            offset_x=10,
            offset_y=40,
            crop_width=80,
            crop_height=120,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_3d_image_with_letterbox_and_static_crop_selected() -> None:
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
    )
    image = np.ones((200, 100, 3), dtype=np.uint8)
    image[40:160, 10:90, :] = (image[40:160, 10:90, :] * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0, :, :, :27] == 0)
    assert torch.all(result_image[0, :, :, 69:] == 0)
    assert torch.all(result_image[0, 0, :, 27:69] == 30)
    assert torch.all(result_image[0, 1, :, 27:69] == 20)
    assert torch.all(result_image[0, 2, :, 27:69] == 10)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=27,
        pad_top=0,
        pad_right=27,
        pad_bottom=0,
        original_size=ImageDimensions(height=200, width=100),
        size_after_pre_processing=ImageDimensions(height=120, width=80),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=64 / 120,
        scale_height=64 / 120,
        static_crop_offset=StaticCropOffset(
            offset_x=10,
            offset_y=40,
            crop_width=80,
            crop_height=120,
        ),
    )


def test_pre_process_torch_3d_not_permuted_image_with_letterbox_and_static_crop_selected() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
    )
    image = np.ones((200, 100, 3), dtype=np.uint8)
    image[40:160, 10:90, :] = (image[40:160, 10:90, :] * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0, :, :, :27] == 0)
    assert torch.all(result_image[0, :, :, 69:] == 0)
    assert torch.all(result_image[0, 0, :, 27:69] == 30)
    assert torch.all(result_image[0, 1, :, 27:69] == 20)
    assert torch.all(result_image[0, 2, :, 27:69] == 10)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=27,
        pad_top=0,
        pad_right=27,
        pad_bottom=0,
        original_size=ImageDimensions(height=200, width=100),
        size_after_pre_processing=ImageDimensions(height=120, width=80),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=64 / 120,
        scale_height=64 / 120,
        static_crop_offset=StaticCropOffset(
            offset_x=10,
            offset_y=40,
            crop_width=80,
            crop_height=120,
        ),
    )


def test_pre_process_torch_4d_image_with_letterbox_and_static_crop_selected() -> None:
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
    )
    image = np.ones((200, 100, 3), dtype=np.uint8)
    image[40:160, 10:90, :] = (image[40:160, 10:90, :] * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))
    image = torch.stack([image, image], dim=0)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0, :, :, :27] == 0)
    assert torch.all(result_image[0, :, :, 69:] == 0)
    assert torch.all(result_image[0, 0, :, 27:69] == 30)
    assert torch.all(result_image[0, 1, :, 27:69] == 20)
    assert torch.all(result_image[0, 2, :, 27:69] == 10)
    assert torch.all(result_image[1, :, :, :27] == 0)
    assert torch.all(result_image[1, :, :, 69:] == 0)
    assert torch.all(result_image[1, 0, :, 27:69] == 30)
    assert torch.all(result_image[1, 1, :, 27:69] == 20)
    assert torch.all(result_image[1, 2, :, 27:69] == 10)
    expected_meta = PreProcessingMetadata(
        pad_left=27,
        pad_top=0,
        pad_right=27,
        pad_bottom=0,
        original_size=ImageDimensions(height=200, width=100),
        size_after_pre_processing=ImageDimensions(height=120, width=80),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=64 / 120,
        scale_height=64 / 120,
        static_crop_offset=StaticCropOffset(
            offset_x=10,
            offset_y=40,
            crop_width=80,
            crop_height=120,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_4d_not_permuted_image_with_letterbox_and_static_crop_selected() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
    )
    image = np.ones((200, 100, 3), dtype=np.uint8)
    image[40:160, 10:90, :] = (image[40:160, 10:90, :] * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.stack([image, image], dim=0)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0, :, :, :27] == 0)
    assert torch.all(result_image[0, :, :, 69:] == 0)
    assert torch.all(result_image[0, 0, :, 27:69] == 30)
    assert torch.all(result_image[0, 1, :, 27:69] == 20)
    assert torch.all(result_image[0, 2, :, 27:69] == 10)
    assert torch.all(result_image[1, :, :, :27] == 0)
    assert torch.all(result_image[1, :, :, 69:] == 0)
    assert torch.all(result_image[1, 0, :, 27:69] == 30)
    assert torch.all(result_image[1, 1, :, 27:69] == 20)
    assert torch.all(result_image[1, 2, :, 27:69] == 10)
    expected_meta = PreProcessingMetadata(
        pad_left=27,
        pad_top=0,
        pad_right=27,
        pad_bottom=0,
        original_size=ImageDimensions(height=200, width=100),
        size_after_pre_processing=ImageDimensions(height=120, width=80),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=64 / 120,
        scale_height=64 / 120,
        static_crop_offset=StaticCropOffset(
            offset_x=10,
            offset_y=40,
            crop_width=80,
            crop_height=120,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_numpy_image_with_letterbox_and_rescaling() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0, :, :, :20] == 0)
    assert torch.all(result_image[0, :, :, 76:] == 0)
    assert torch.all(result_image[0, 0, :, 20:76] == 3.0)
    assert torch.all(result_image[0, 1, :, 20:76] == 2.0)
    assert torch.all(result_image[0, 2, :, 20:76] == 1.0)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=20,
        pad_top=0,
        pad_right=20,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=64 / 192,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_numpy_images_list_with_letterbox_and_rescaling() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0, :, :, :20] == 0)
    assert torch.all(result_image[0, :, :, 76:] == 0)
    assert torch.all(result_image[0, 0, :, 20:76] == 3.0)
    assert torch.all(result_image[0, 1, :, 20:76] == 2.0)
    assert torch.all(result_image[0, 2, :, 20:76] == 1.0)
    assert torch.all(result_image[1, :, :, :20] == 0)
    assert torch.all(result_image[1, :, :, 76:] == 0)
    assert torch.all(result_image[1, 0, :, 20:76] == 3.0)
    assert torch.all(result_image[1, 1, :, 20:76] == 2.0)
    assert torch.all(result_image[1, 2, :, 20:76] == 1.0)
    expected_meta = PreProcessingMetadata(
        pad_left=20,
        pad_top=0,
        pad_right=20,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=64 / 192,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_3d_image_with_letterbox_and_rescaling() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0, :, :, :20] == 0)
    assert torch.all(result_image[0, :, :, 76:] == 0)
    assert torch.all(result_image[0, 0, :, 20:76] == 3.0)
    assert torch.all(result_image[0, 1, :, 20:76] == 2.0)
    assert torch.all(result_image[0, 2, :, 20:76] == 1.0)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=20,
        pad_top=0,
        pad_right=20,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=64 / 192,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_torch_3d_not_permuted_image_with_letterbox_and_rescaling() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0, :, :, :20] == 0)
    assert torch.all(result_image[0, :, :, 76:] == 0)
    assert torch.all(result_image[0, 0, :, 20:76] == 3.0)
    assert torch.all(result_image[0, 1, :, 20:76] == 2.0)
    assert torch.all(result_image[0, 2, :, 20:76] == 1.0)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=20,
        pad_top=0,
        pad_right=20,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=64 / 192,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_torch_4d_image_with_letterbox_and_rescaling() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))
    image = torch.stack([image, image], dim=0)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0, :, :, :20] == 0)
    assert torch.all(result_image[0, :, :, 76:] == 0)
    assert torch.all(result_image[0, 0, :, 20:76] == 3.0)
    assert torch.all(result_image[0, 1, :, 20:76] == 2.0)
    assert torch.all(result_image[0, 2, :, 20:76] == 1.0)
    assert torch.all(result_image[1, :, :, :20] == 0)
    assert torch.all(result_image[1, :, :, 76:] == 0)
    assert torch.all(result_image[1, 0, :, 20:76] == 3.0)
    assert torch.all(result_image[1, 1, :, 20:76] == 2.0)
    assert torch.all(result_image[1, 2, :, 20:76] == 1.0)
    expected_meta = PreProcessingMetadata(
        pad_left=20,
        pad_top=0,
        pad_right=20,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=64 / 192,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_4d_not_permuted_image_with_letterbox_and_rescaling() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.stack([image, image], dim=0)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0, :, :, :20] == 0)
    assert torch.all(result_image[0, :, :, 76:] == 0)
    assert torch.all(result_image[0, 0, :, 20:76] == 3.0)
    assert torch.all(result_image[0, 1, :, 20:76] == 2.0)
    assert torch.all(result_image[0, 2, :, 20:76] == 1.0)
    assert torch.all(result_image[1, :, :, :20] == 0)
    assert torch.all(result_image[1, :, :, 76:] == 0)
    assert torch.all(result_image[1, 0, :, 20:76] == 3.0)
    assert torch.all(result_image[1, 1, :, 20:76] == 2.0)
    assert torch.all(result_image[1, 2, :, 20:76] == 1.0)
    expected_meta = PreProcessingMetadata(
        pad_left=20,
        pad_top=0,
        pad_right=20,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=64 / 192,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_numpy_image_with_letterbox_rescaling_and_normalization() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
        scaling_factor=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0, :, :, :20] == -1 / 3)
    assert torch.all(result_image[0, :, :, 76:] == -1 / 3)
    assert torch.all(result_image[0, 0, :, 20:76] == 1 / 6)
    assert torch.all(result_image[0, 1, :, 20:76] == 0.0)
    assert torch.all(result_image[0, 2, :, 20:76] == -1 / 6)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=20,
        pad_top=0,
        pad_right=20,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=64 / 192,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_numpy_images_list_with_letterbox_rescaling_and_normalization() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
        scaling_factor=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0, :, :, :20] == -1 / 3)
    assert torch.all(result_image[0, :, :, 76:] == -1 / 3)
    assert torch.all(result_image[0, 0, :, 20:76] == 1 / 6)
    assert torch.all(result_image[0, 1, :, 20:76] == 0.0)
    assert torch.all(result_image[0, 2, :, 20:76] == -1 / 6)
    assert torch.all(result_image[1, :, :, :20] == -1 / 3)
    assert torch.all(result_image[1, :, :, 76:] == -1 / 3)
    assert torch.all(result_image[1, 0, :, 20:76] == 1 / 6)
    assert torch.all(result_image[1, 1, :, 20:76] == 0.0)
    assert torch.all(result_image[1, 2, :, 20:76] == -1 / 6)
    expected_meta = PreProcessingMetadata(
        pad_left=20,
        pad_top=0,
        pad_right=20,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=64 / 192,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_numpy_image_with_center_crop_selected_and_crop_fitting_inside_original_image() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0][0] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][2] == 10)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=-36,
        pad_top=-64,
        pad_right=-36,
        pad_bottom=-64,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_numpy_images_list_with_center_crop_selected_and_crop_fitting_inside_original_image() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][0] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][2] == 10)
    assert torch.all(result_image[1][0] == 30)
    assert torch.all(result_image[1][1] == 20)
    assert torch.all(result_image[1][2] == 10)
    expected_meta = PreProcessingMetadata(
        pad_left=-36,
        pad_top=-64,
        pad_right=-36,
        pad_bottom=-64,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_3d_image_with_center_crop_selected_and_crop_fitting_inside_original_image() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0][0] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][2] == 10)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=-36,
        pad_top=-64,
        pad_right=-36,
        pad_bottom=-64,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_torch_3d_not_permuted_image_with_center_crop_selected_and_crop_fitting_inside_original_image() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0][0] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][2] == 10)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=-36,
        pad_top=-64,
        pad_right=-36,
        pad_bottom=-64,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_torch_4d_image_with_center_crop_selected_and_crop_fitting_inside_original_image() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))
    image = torch.stack([image, image], dim=0)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][0] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][2] == 10)
    assert torch.all(result_image[1][0] == 30)
    assert torch.all(result_image[1][1] == 20)
    assert torch.all(result_image[1][2] == 10)
    expected_meta = PreProcessingMetadata(
        pad_left=-36,
        pad_top=-64,
        pad_right=-36,
        pad_bottom=-64,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_4d_not_permuted_image_with_center_crop_selected_and_crop_fitting_inside_original_image() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.stack([image, image], dim=0)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][0] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][2] == 10)
    assert torch.all(result_image[1][0] == 30)
    assert torch.all(result_image[1][1] == 20)
    assert torch.all(result_image[1][2] == 10)
    expected_meta = PreProcessingMetadata(
        pad_left=-36,
        pad_top=-64,
        pad_right=-36,
        pad_bottom=-64,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_numpy_image_with_center_crop_selected_and_crop_not_fitting_inside_original_image() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=292, width=268),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 292, 268)
    assert torch.all(result_image[0][0, 50:242, 50:218] == 30)
    assert torch.all(result_image[0][1, 50:242, 50:218] == 20)
    assert torch.all(result_image[0][2, 50:242, 50:218] == 10)
    assert torch.all(result_image[0][:, 0:50, :] == 0)
    assert torch.all(result_image[0][:, 242:, :] == 0)
    assert torch.all(result_image[0][:, :, :50] == 0)
    assert torch.all(result_image[0][:, :, 218:] == 0)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=50,
        pad_top=50,
        pad_right=50,
        pad_bottom=50,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=292, width=268),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_numpy_images_list_with_center_crop_selected_and_crop_not_fitting_inside_original_image() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=292, width=268),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 292, 268)
    assert torch.all(result_image[0][0, 50:242, 50:218] == 30)
    assert torch.all(result_image[0][1, 50:242, 50:218] == 20)
    assert torch.all(result_image[0][2, 50:242, 50:218] == 10)
    assert torch.all(result_image[0][:, 0:50, :] == 0)
    assert torch.all(result_image[0][:, 242:, :] == 0)
    assert torch.all(result_image[0][:, :, :50] == 0)
    assert torch.all(result_image[0][:, :, 218:] == 0)
    assert torch.all(result_image[1][0, 50:242, 50:218] == 30)
    assert torch.all(result_image[1][1, 50:242, 50:218] == 20)
    assert torch.all(result_image[1][2, 50:242, 50:218] == 10)
    assert torch.all(result_image[1][:, 0:50, :] == 0)
    assert torch.all(result_image[1][:, 242:, :] == 0)
    assert torch.all(result_image[1][:, :, :50] == 0)
    assert torch.all(result_image[1][:, :, 218:] == 0)
    expected_meta = PreProcessingMetadata(
        pad_left=50,
        pad_top=50,
        pad_right=50,
        pad_bottom=50,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=292, width=268),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_3d_image_with_center_crop_selected_and_crop_not_fitting_inside_original_image() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=292, width=268),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 292, 268)
    assert torch.all(result_image[0][0, 50:242, 50:218] == 30)
    assert torch.all(result_image[0][1, 50:242, 50:218] == 20)
    assert torch.all(result_image[0][2, 50:242, 50:218] == 10)
    assert torch.all(result_image[0][:, 0:50, :] == 0)
    assert torch.all(result_image[0][:, 242:, :] == 0)
    assert torch.all(result_image[0][:, :, :50] == 0)
    assert torch.all(result_image[0][:, :, 218:] == 0)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=50,
        pad_top=50,
        pad_right=50,
        pad_bottom=50,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=292, width=268),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_torch_3d_not_permuted_image_with_center_crop_selected_and_crop_not_fitting_inside_original_image() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=292, width=268),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 292, 268)
    assert torch.all(result_image[0][0, 50:242, 50:218] == 30)
    assert torch.all(result_image[0][1, 50:242, 50:218] == 20)
    assert torch.all(result_image[0][2, 50:242, 50:218] == 10)
    assert torch.all(result_image[0][:, 0:50, :] == 0)
    assert torch.all(result_image[0][:, 242:, :] == 0)
    assert torch.all(result_image[0][:, :, :50] == 0)
    assert torch.all(result_image[0][:, :, 218:] == 0)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=50,
        pad_top=50,
        pad_right=50,
        pad_bottom=50,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=292, width=268),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_list_of_torch_3d_image_with_center_crop_selected_and_crop_not_fitting_inside_original_image() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=292, width=268),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 292, 268)
    assert torch.all(result_image[0][0, 50:242, 50:218] == 30)
    assert torch.all(result_image[0][1, 50:242, 50:218] == 20)
    assert torch.all(result_image[0][2, 50:242, 50:218] == 10)
    assert torch.all(result_image[0][:, 0:50, :] == 0)
    assert torch.all(result_image[0][:, 242:, :] == 0)
    assert torch.all(result_image[0][:, :, :50] == 0)
    assert torch.all(result_image[0][:, :, 218:] == 0)
    assert torch.all(result_image[1][0, 50:242, 50:218] == 30)
    assert torch.all(result_image[1][1, 50:242, 50:218] == 20)
    assert torch.all(result_image[1][2, 50:242, 50:218] == 10)
    assert torch.all(result_image[1][:, 0:50, :] == 0)
    assert torch.all(result_image[1][:, 242:, :] == 0)
    assert torch.all(result_image[1][:, :, :50] == 0)
    assert torch.all(result_image[1][:, :, 218:] == 0)
    expected_meta = PreProcessingMetadata(
        pad_left=50,
        pad_top=50,
        pad_right=50,
        pad_bottom=50,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=292, width=268),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_list_of_torch_3d_image_not_permuted_with_center_crop_selected_and_crop_not_fitting_inside_original_image() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=292, width=268),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 292, 268)
    assert torch.all(result_image[0][0, 50:242, 50:218] == 30)
    assert torch.all(result_image[0][1, 50:242, 50:218] == 20)
    assert torch.all(result_image[0][2, 50:242, 50:218] == 10)
    assert torch.all(result_image[0][:, 0:50, :] == 0)
    assert torch.all(result_image[0][:, 242:, :] == 0)
    assert torch.all(result_image[0][:, :, :50] == 0)
    assert torch.all(result_image[0][:, :, 218:] == 0)
    assert torch.all(result_image[1][0, 50:242, 50:218] == 30)
    assert torch.all(result_image[1][1, 50:242, 50:218] == 20)
    assert torch.all(result_image[1][2, 50:242, 50:218] == 10)
    assert torch.all(result_image[1][:, 0:50, :] == 0)
    assert torch.all(result_image[1][:, 242:, :] == 0)
    assert torch.all(result_image[1][:, :, :50] == 0)
    assert torch.all(result_image[1][:, :, 218:] == 0)
    expected_meta = PreProcessingMetadata(
        pad_left=50,
        pad_top=50,
        pad_right=50,
        pad_bottom=50,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=292, width=268),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_list_of_torch_4d_image_with_center_crop_selected_and_crop_not_fitting_inside_original_image() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=292, width=268),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))
    image = torch.unsqueeze(image, dim=0)

    # when
    with pytest.raises(ModelRuntimeError):
        _ = pre_process_network_input(
            images=[image, image],
            image_pre_processing=image_pre_processing,
            network_input=network_input,
            target_device=torch.device("cpu"),
        )


def test_pre_process_torch_4d_image_with_center_crop_selected_and_crop_not_fitting_inside_original_image() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=292, width=268),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))
    image = torch.stack([image, image], dim=0)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 292, 268)
    assert torch.all(result_image[0][0, 50:242, 50:218] == 30)
    assert torch.all(result_image[0][1, 50:242, 50:218] == 20)
    assert torch.all(result_image[0][2, 50:242, 50:218] == 10)
    assert torch.all(result_image[0][:, 0:50, :] == 0)
    assert torch.all(result_image[0][:, 242:, :] == 0)
    assert torch.all(result_image[0][:, :, :50] == 0)
    assert torch.all(result_image[0][:, :, 218:] == 0)
    assert torch.all(result_image[1][0, 50:242, 50:218] == 30)
    assert torch.all(result_image[1][1, 50:242, 50:218] == 20)
    assert torch.all(result_image[1][2, 50:242, 50:218] == 10)
    assert torch.all(result_image[1][:, 0:50, :] == 0)
    assert torch.all(result_image[1][:, 242:, :] == 0)
    assert torch.all(result_image[1][:, :, :50] == 0)
    assert torch.all(result_image[1][:, :, 218:] == 0)
    expected_meta = PreProcessingMetadata(
        pad_left=50,
        pad_top=50,
        pad_right=50,
        pad_bottom=50,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=292, width=268),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta


def test_pre_process_torch_4d_not_permuted_image_with_center_crop_selected_and_crop_not_fitting_inside_original_image() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=292, width=268),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.stack([image, image], dim=0)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 292, 268)
    # image 0
    assert torch.all(result_image[0][0, 50:242, 50:218] == 30)
    assert torch.all(result_image[0][1, 50:242, 50:218] == 20)
    assert torch.all(result_image[0][2, 50:242, 50:218] == 10)
    assert torch.all(result_image[0][:, 0:50, :] == 0)
    assert torch.all(result_image[0][:, 242:, :] == 0)
    assert torch.all(result_image[0][:, :, :50] == 0)
    assert torch.all(result_image[0][:, :, 218:] == 0)
    # image 1
    assert torch.all(result_image[1][0, 50:242, 50:218] == 30)
    assert torch.all(result_image[1][1, 50:242, 50:218] == 20)
    assert torch.all(result_image[1][2, 50:242, 50:218] == 10)
    assert torch.all(result_image[1][:, 0:50, :] == 0)
    assert torch.all(result_image[1][:, 242:, :] == 0)
    assert torch.all(result_image[1][:, :, :50] == 0)
    assert torch.all(result_image[1][:, :, 218:] == 0)

    expected_meta = PreProcessingMetadata(
        pad_left=50,
        pad_top=50,
        pad_right=50,
        pad_bottom=50,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=292, width=268),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_numpy_image_with_center_crop_selected_and_crop_not_fitting_inside_original_image_along_width_dimension() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=292, width=100),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = (np.ones((192, 169, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 292, 100)
    assert torch.all(result_image[0][0, 50:242, :] == 30)
    assert torch.all(result_image[0][1, 50:242, :] == 20)
    assert torch.all(result_image[0][2, 50:242, :] == 10)
    assert torch.all(result_image[0][:, 0:50, :] == 0)
    assert torch.all(result_image[0][:, 242:, :] == 0)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=-34,
        pad_top=50,
        pad_right=-35,
        pad_bottom=50,
        original_size=ImageDimensions(height=192, width=169),
        size_after_pre_processing=ImageDimensions(height=192, width=169),
        inference_size=ImageDimensions(height=292, width=100),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=169,
            crop_height=192,
        ),
    )


def test_pre_process_numpy_images_list_with_center_crop_selected_and_crop_not_fitting_inside_original_image_along_width_dimension() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=292, width=100),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = (np.ones((192, 169, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 292, 100)
    # image 0: content band (no horizontal padding, only vertical)
    assert torch.all(result_image[0][0, 50:242, :] == 30)
    assert torch.all(result_image[0][1, 50:242, :] == 20)
    assert torch.all(result_image[0][2, 50:242, :] == 10)
    assert torch.all(result_image[0][:, 0:50, :] == 0)
    assert torch.all(result_image[0][:, 242:, :] == 0)
    # image 1
    assert torch.all(result_image[1][0, 50:242, :] == 30)
    assert torch.all(result_image[1][1, 50:242, :] == 20)
    assert torch.all(result_image[1][2, 50:242, :] == 10)
    assert torch.all(result_image[1][:, 0:50, :] == 0)
    assert torch.all(result_image[1][:, 242:, :] == 0)

    expected_meta = PreProcessingMetadata(
        pad_left=-34,
        pad_top=50,
        pad_right=-35,
        pad_bottom=50,
        original_size=ImageDimensions(height=192, width=169),
        size_after_pre_processing=ImageDimensions(height=192, width=169),
        inference_size=ImageDimensions(height=292, width=100),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=169,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_3d_image_with_center_crop_selected_and_crop_not_fitting_inside_original_image_along_width_dimension() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=292, width=100),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = (np.ones((192, 169, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))  # CHW

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 292, 100)
    assert torch.all(result_image[0][0, 50:242, :] == 30)
    assert torch.all(result_image[0][1, 50:242, :] == 20)
    assert torch.all(result_image[0][2, 50:242, :] == 10)
    assert torch.all(result_image[0][:, 0:50, :] == 0)
    assert torch.all(result_image[0][:, 242:, :] == 0)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=-34,
        pad_top=50,
        pad_right=-35,
        pad_bottom=50,
        original_size=ImageDimensions(height=192, width=169),
        size_after_pre_processing=ImageDimensions(height=192, width=169),
        inference_size=ImageDimensions(height=292, width=100),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=169,
            crop_height=192,
        ),
    )


def test_pre_process_torch_3d_not_permuted_image_with_center_crop_selected_and_crop_not_fitting_inside_original_image_along_width_dimension() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=292, width=100),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = (np.ones((192, 169, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)  # HWC

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 292, 100)
    assert torch.all(result_image[0][0, 50:242, :] == 30)
    assert torch.all(result_image[0][1, 50:242, :] == 20)
    assert torch.all(result_image[0][2, 50:242, :] == 10)
    assert torch.all(result_image[0][:, 0:50, :] == 0)
    assert torch.all(result_image[0][:, 242:, :] == 0)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=-34,
        pad_top=50,
        pad_right=-35,
        pad_bottom=50,
        original_size=ImageDimensions(height=192, width=169),
        size_after_pre_processing=ImageDimensions(height=192, width=169),
        inference_size=ImageDimensions(height=292, width=100),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=169,
            crop_height=192,
        ),
    )


def test_pre_process_torch_4d_image_with_center_crop_selected_and_crop_not_fitting_inside_original_image_along_width_dimension() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=292, width=100),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = (np.ones((192, 169, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))  # CHW
    image = torch.stack([image, image], dim=0)  # NCHW

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 292, 100)
    # image 0
    assert torch.all(result_image[0][0, 50:242, :] == 30)
    assert torch.all(result_image[0][1, 50:242, :] == 20)
    assert torch.all(result_image[0][2, 50:242, :] == 10)
    assert torch.all(result_image[0][:, 0:50, :] == 0)
    assert torch.all(result_image[0][:, 242:, :] == 0)
    # image 1
    assert torch.all(result_image[1][0, 50:242, :] == 30)
    assert torch.all(result_image[1][1, 50:242, :] == 20)
    assert torch.all(result_image[1][2, 50:242, :] == 10)
    assert torch.all(result_image[1][:, 0:50, :] == 0)
    assert torch.all(result_image[1][:, 242:, :] == 0)

    expected_meta = PreProcessingMetadata(
        pad_left=-34,
        pad_top=50,
        pad_right=-35,
        pad_bottom=50,
        original_size=ImageDimensions(height=192, width=169),
        size_after_pre_processing=ImageDimensions(height=192, width=169),
        inference_size=ImageDimensions(height=292, width=100),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=169,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_4d_not_permuted_image_with_center_crop_selected_and_crop_not_fitting_inside_original_image_along_width_dimension() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=292, width=100),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = (np.ones((192, 169, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.stack([image, image], dim=0)  # NHWC

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 292, 100)
    # image 0
    assert torch.all(result_image[0][0, 50:242, :] == 30)
    assert torch.all(result_image[0][1, 50:242, :] == 20)
    assert torch.all(result_image[0][2, 50:242, :] == 10)
    assert torch.all(result_image[0][:, 0:50, :] == 0)
    assert torch.all(result_image[0][:, 242:, :] == 0)
    # image 1
    assert torch.all(result_image[1][0, 50:242, :] == 30)
    assert torch.all(result_image[1][1, 50:242, :] == 20)
    assert torch.all(result_image[1][2, 50:242, :] == 10)
    assert torch.all(result_image[1][:, 0:50, :] == 0)
    assert torch.all(result_image[1][:, 242:, :] == 0)

    expected_meta = PreProcessingMetadata(
        pad_left=-34,
        pad_top=50,
        pad_right=-35,
        pad_bottom=50,
        original_size=ImageDimensions(height=192, width=169),
        size_after_pre_processing=ImageDimensions(height=192, width=169),
        inference_size=ImageDimensions(height=292, width=100),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=169,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_numpy_image_with_center_crop_selected_and_crop_not_fitting_inside_original_image_along_heigght_dimension() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=100, width=268),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 100, 268)
    assert torch.all(result_image[0][0, :, 50:218] == 30)
    assert torch.all(result_image[0][1, :, 50:218] == 20)
    assert torch.all(result_image[0][2, :, 50:218] == 10)
    assert torch.all(result_image[0][:, :, :50] == 0)
    assert torch.all(result_image[0][:, :, 218:] == 0)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=50,
        pad_top=-46,
        pad_right=50,
        pad_bottom=-46,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=100, width=268),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_numpy_images_list_with_center_crop_selected_and_crop_not_fitting_inside_original_image_along_heigght_dimension() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=100, width=268),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 100, 268)
    # image 0
    assert torch.all(result_image[0][0, :, 50:218] == 30)
    assert torch.all(result_image[0][1, :, 50:218] == 20)
    assert torch.all(result_image[0][2, :, 50:218] == 10)
    assert torch.all(result_image[0][:, :, :50] == 0)
    assert torch.all(result_image[0][:, :, 218:] == 0)
    # image 1
    assert torch.all(result_image[1][0, :, 50:218] == 30)
    assert torch.all(result_image[1][1, :, 50:218] == 20)
    assert torch.all(result_image[1][2, :, 50:218] == 10)
    assert torch.all(result_image[1][:, :, :50] == 0)
    assert torch.all(result_image[1][:, :, 218:] == 0)

    expected_meta = PreProcessingMetadata(
        pad_left=50,
        pad_top=-46,
        pad_right=50,
        pad_bottom=-46,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=100, width=268),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_3d_image_with_center_crop_selected_and_crop_not_fitting_inside_original_image_along_heigght_dimension() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=100, width=268),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))  # CHW

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 100, 268)
    assert torch.all(result_image[0][0, :, 50:218] == 30)
    assert torch.all(result_image[0][1, :, 50:218] == 20)
    assert torch.all(result_image[0][2, :, 50:218] == 10)
    assert torch.all(result_image[0][:, :, :50] == 0)
    assert torch.all(result_image[0][:, :, 218:] == 0)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=50,
        pad_top=-46,
        pad_right=50,
        pad_bottom=-46,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=100, width=268),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_torch_3d_not_permuted_image_with_center_crop_selected_and_crop_not_fitting_inside_original_image_along_heigght_dimension() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=100, width=268),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)  # HWC

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 100, 268)
    assert torch.all(result_image[0][0, :, 50:218] == 30)
    assert torch.all(result_image[0][1, :, 50:218] == 20)
    assert torch.all(result_image[0][2, :, 50:218] == 10)
    assert torch.all(result_image[0][:, :, :50] == 0)
    assert torch.all(result_image[0][:, :, 218:] == 0)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=50,
        pad_top=-46,
        pad_right=50,
        pad_bottom=-46,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=100, width=268),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_torch_4d_image_with_center_crop_selected_and_crop_not_fitting_inside_original_image_along_heigght_dimension() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=100, width=268),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))  # CHW
    image = torch.stack([image, image], dim=0)  # NCHW

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 100, 268)
    # image 0
    assert torch.all(result_image[0][0, :, 50:218] == 30)
    assert torch.all(result_image[0][1, :, 50:218] == 20)
    assert torch.all(result_image[0][2, :, 50:218] == 10)
    assert torch.all(result_image[0][:, :, :50] == 0)
    assert torch.all(result_image[0][:, :, 218:] == 0)
    # image 1
    assert torch.all(result_image[1][0, :, 50:218] == 30)
    assert torch.all(result_image[1][1, :, 50:218] == 20)
    assert torch.all(result_image[1][2, :, 50:218] == 10)
    assert torch.all(result_image[1][:, :, :50] == 0)
    assert torch.all(result_image[1][:, :, 218:] == 0)

    expected_meta = PreProcessingMetadata(
        pad_left=50,
        pad_top=-46,
        pad_right=50,
        pad_bottom=-46,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=100, width=268),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_4d_not_permuted_image_with_center_crop_selected_and_crop_not_fitting_inside_original_image_along_heigght_dimension() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=100, width=268),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.stack([image, image], dim=0)  # NHWC

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 100, 268)
    # image 0
    assert torch.all(result_image[0][0, :, 50:218] == 30)
    assert torch.all(result_image[0][1, :, 50:218] == 20)
    assert torch.all(result_image[0][2, :, 50:218] == 10)
    assert torch.all(result_image[0][:, :, :50] == 0)
    assert torch.all(result_image[0][:, :, 218:] == 0)
    # image 1
    assert torch.all(result_image[1][0, :, 50:218] == 30)
    assert torch.all(result_image[1][1, :, 50:218] == 20)
    assert torch.all(result_image[1][2, :, 50:218] == 10)
    assert torch.all(result_image[1][:, :, :50] == 0)
    assert torch.all(result_image[1][:, :, 218:] == 0)

    expected_meta = PreProcessingMetadata(
        pad_left=50,
        pad_top=-46,
        pad_right=50,
        pad_bottom=-46,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=100, width=268),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_numpy_image_with_static_crop_and_center_crop_selected_and_crop_fitting_inside_original_image() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    # after center crop - image of size (120, 160, 3) available

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0][0] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][2] == 10)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=-32,
        pad_top=-28,
        pad_right=-32,
        pad_bottom=-28,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )


def test_pre_process_numpy_images_list_with_static_crop_and_center_crop_selected_and_crop_fitting_inside_original_image() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    # after center crop - image of size (120, 160, 3) available

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][0] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][2] == 10)
    assert torch.all(result_image[1][0] == 30)
    assert torch.all(result_image[1][1] == 20)
    assert torch.all(result_image[1][2] == 10)
    expected_meta = PreProcessingMetadata(
        pad_left=-32,
        pad_top=-28,
        pad_right=-32,
        pad_bottom=-28,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_3d_image_with_static_crop_and_center_crop_selected_and_crop_fitting_inside_original_image() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))
    # after center crop - image of size (120, 160, 3) available

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0][0] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][2] == 10)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=-32,
        pad_top=-28,
        pad_right=-32,
        pad_bottom=-28,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )


def test_pre_process_torch_3d_not_permuted_image_with_static_crop_and_center_crop_selected_and_crop_fitting_inside_original_image() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    image = torch.from_numpy(image)
    # after center crop - image of size (120, 160, 3) available

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0][0] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][2] == 10)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=-32,
        pad_top=-28,
        pad_right=-32,
        pad_bottom=-28,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )


def test_pre_process_torch_4d_image_with_static_crop_and_center_crop_selected_and_crop_fitting_inside_original_image() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))
    image = torch.stack([image, image], dim=0)
    # after center crop - image of size (120, 160, 3) available

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][0] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][2] == 10)
    assert torch.all(result_image[1][0] == 30)
    assert torch.all(result_image[1][1] == 20)
    assert torch.all(result_image[1][2] == 10)
    expected_meta = PreProcessingMetadata(
        pad_left=-32,
        pad_top=-28,
        pad_right=-32,
        pad_bottom=-28,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_4d_image_not_permuted_with_static_crop_and_center_crop_selected_and_crop_fitting_inside_original_image() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    image = torch.from_numpy(image)
    image = torch.stack([image, image], dim=0)
    # after center crop - image of size (120, 160, 3) available

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][0] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][2] == 10)
    assert torch.all(result_image[1][0] == 30)
    assert torch.all(result_image[1][1] == 20)
    assert torch.all(result_image[1][2] == 10)
    expected_meta = PreProcessingMetadata(
        pad_left=-32,
        pad_top=-28,
        pad_right=-32,
        pad_bottom=-28,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_numpy_image_with_static_crop_and_center_crop_selected_and_crop_not_fitting_inside_original_image() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=200, width=200),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    # after center crop - image of size (120, 160, 3) available

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 200, 200)
    assert torch.all(result_image[0][0, 40:160, 20:180] == 30)
    assert torch.all(result_image[0][1, 40:160, 20:180] == 20)
    assert torch.all(result_image[0][2, 40:160, 20:180] == 10)
    assert torch.all(result_image[0][:, :40, :] == 0)
    assert torch.all(result_image[0][:, 160:, :] == 0)
    assert torch.all(result_image[0][:, :, :20] == 0)
    assert torch.all(result_image[0][:, :, 180:] == 0)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=20,
        pad_top=40,
        pad_right=20,
        pad_bottom=40,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=200, width=200),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )


def test_pre_process_numpy_images_list_with_static_crop_and_center_crop_selected_and_crop_not_fitting_inside_original_image() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=200, width=200),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    # after center crop - image of size (120, 160, 3) available

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 200, 200)
    assert torch.all(result_image[0][0, 40:160, 20:180] == 30)
    assert torch.all(result_image[0][1, 40:160, 20:180] == 20)
    assert torch.all(result_image[0][2, 40:160, 20:180] == 10)
    assert torch.all(result_image[0][:, :40, :] == 0)
    assert torch.all(result_image[0][:, 160:, :] == 0)
    assert torch.all(result_image[0][:, :, :20] == 0)
    assert torch.all(result_image[0][:, :, 180:] == 0)
    assert torch.all(result_image[1][0, 40:160, 20:180] == 30)
    assert torch.all(result_image[1][1, 40:160, 20:180] == 20)
    assert torch.all(result_image[1][2, 40:160, 20:180] == 10)
    assert torch.all(result_image[1][:, :40, :] == 0)
    assert torch.all(result_image[1][:, 160:, :] == 0)
    assert torch.all(result_image[1][:, :, :20] == 0)
    assert torch.all(result_image[1][:, :, 180:] == 0)
    expected_meta = PreProcessingMetadata(
        pad_left=20,
        pad_top=40,
        pad_right=20,
        pad_bottom=40,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=200, width=200),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_3d_image_with_static_crop_and_center_crop_selected_and_crop_not_fitting_inside_original_image() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=200, width=200),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))
    # after center crop - image of size (120, 160, 3) available

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 200, 200)
    assert torch.all(result_image[0][0, 40:160, 20:180] == 30)
    assert torch.all(result_image[0][1, 40:160, 20:180] == 20)
    assert torch.all(result_image[0][2, 40:160, 20:180] == 10)
    assert torch.all(result_image[0][:, :40, :] == 0)
    assert torch.all(result_image[0][:, 160:, :] == 0)
    assert torch.all(result_image[0][:, :, :20] == 0)
    assert torch.all(result_image[0][:, :, 180:] == 0)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=20,
        pad_top=40,
        pad_right=20,
        pad_bottom=40,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=200, width=200),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )


def test_pre_process_torch_3d_not_permuted_image_with_static_crop_and_center_crop_selected_and_crop_not_fitting_inside_original_image() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=200, width=200),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    image = torch.from_numpy(image)
    # after center crop - image of size (120, 160, 3) available

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 200, 200)
    assert torch.all(result_image[0][0, 40:160, 20:180] == 30)
    assert torch.all(result_image[0][1, 40:160, 20:180] == 20)
    assert torch.all(result_image[0][2, 40:160, 20:180] == 10)
    assert torch.all(result_image[0][:, :40, :] == 0)
    assert torch.all(result_image[0][:, 160:, :] == 0)
    assert torch.all(result_image[0][:, :, :20] == 0)
    assert torch.all(result_image[0][:, :, 180:] == 0)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=20,
        pad_top=40,
        pad_right=20,
        pad_bottom=40,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=200, width=200),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )


def test_pre_process_torch_4d_image_with_static_crop_and_center_crop_selected_and_crop_not_fitting_inside_original_image() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=200, width=200),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))
    image = torch.stack([image, image], dim=0)
    # after center crop - image of size (120, 160, 3) available

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 200, 200)
    assert torch.all(result_image[0][0, 40:160, 20:180] == 30)
    assert torch.all(result_image[0][1, 40:160, 20:180] == 20)
    assert torch.all(result_image[0][2, 40:160, 20:180] == 10)
    assert torch.all(result_image[0][:, :40, :] == 0)
    assert torch.all(result_image[0][:, 160:, :] == 0)
    assert torch.all(result_image[0][:, :, :20] == 0)
    assert torch.all(result_image[0][:, :, 180:] == 0)
    assert torch.all(result_image[1][0, 40:160, 20:180] == 30)
    assert torch.all(result_image[1][1, 40:160, 20:180] == 20)
    assert torch.all(result_image[1][2, 40:160, 20:180] == 10)
    assert torch.all(result_image[1][:, :40, :] == 0)
    assert torch.all(result_image[1][:, 160:, :] == 0)
    assert torch.all(result_image[1][:, :, :20] == 0)
    assert torch.all(result_image[1][:, :, 180:] == 0)
    expected_meta = PreProcessingMetadata(
        pad_left=20,
        pad_top=40,
        pad_right=20,
        pad_bottom=40,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=200, width=200),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_4d_not_permuted_image_with_static_crop_and_center_crop_selected_and_crop_not_fitting_inside_original_image() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=200, width=200),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    image = torch.from_numpy(image)
    image = torch.stack([image, image], dim=0)
    # after center crop - image of size (120, 160, 3) available

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 200, 200)
    assert torch.all(result_image[0][0, 40:160, 20:180] == 30)
    assert torch.all(result_image[0][1, 40:160, 20:180] == 20)
    assert torch.all(result_image[0][2, 40:160, 20:180] == 10)
    assert torch.all(result_image[0][:, :40, :] == 0)
    assert torch.all(result_image[0][:, 160:, :] == 0)
    assert torch.all(result_image[0][:, :, :20] == 0)
    assert torch.all(result_image[0][:, :, 180:] == 0)
    assert torch.all(result_image[1][0, 40:160, 20:180] == 30)
    assert torch.all(result_image[1][1, 40:160, 20:180] == 20)
    assert torch.all(result_image[1][2, 40:160, 20:180] == 10)
    assert torch.all(result_image[1][:, :40, :] == 0)
    assert torch.all(result_image[1][:, 160:, :] == 0)
    assert torch.all(result_image[1][:, :, :20] == 0)
    assert torch.all(result_image[1][:, :, 180:] == 0)
    expected_meta = PreProcessingMetadata(
        pad_left=20,
        pad_top=40,
        pad_right=20,
        pad_bottom=40,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=200, width=200),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_numpy_image_with_static_crop_and_center_crop_selected_and_crop_fitting_inside_original_image_with_scaling() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    # after center crop - image of size (120, 160, 3) available

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0][0] == 3.0)
    assert torch.all(result_image[0][1] == 2.0)
    assert torch.all(result_image[0][2] == 1.0)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=-32,
        pad_top=-28,
        pad_right=-32,
        pad_bottom=-28,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )


def test_pre_process_numpy_images_list_with_static_crop_and_center_crop_selected_and_crop_fitting_inside_original_image_with_scaling() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    # img 0
    assert torch.all(result_image[0][0] == 3.0)
    assert torch.all(result_image[0][1] == 2.0)
    assert torch.all(result_image[0][2] == 1.0)
    # img 1
    assert torch.all(result_image[1][0] == 3.0)
    assert torch.all(result_image[1][1] == 2.0)
    assert torch.all(result_image[1][2] == 1.0)

    expected_meta = PreProcessingMetadata(
        pad_left=-32,
        pad_top=-28,
        pad_right=-32,
        pad_bottom=-28,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_3d_image_with_static_crop_and_center_crop_selected_and_crop_fitting_inside_original_image_with_scaling() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))  # CHW

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0][0] == 3.0)
    assert torch.all(result_image[0][1] == 2.0)
    assert torch.all(result_image[0][2] == 1.0)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=-32,
        pad_top=-28,
        pad_right=-32,
        pad_bottom=-28,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )


def test_pre_process_torch_3d_image_with_static_crop_and_center_crop_selected_and_crop_fitting_inside_original_image_with_scaling() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))  # CHW

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0][0] == 3.0)
    assert torch.all(result_image[0][1] == 2.0)
    assert torch.all(result_image[0][2] == 1.0)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=-32,
        pad_top=-28,
        pad_right=-32,
        pad_bottom=-28,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )


def test_pre_process_torch_3d_not_permuted_image_with_static_crop_and_center_crop_selected_and_crop_fitting_inside_original_image_with_scaling() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    image = torch.from_numpy(image)  # HWC

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0][0] == 3.0)
    assert torch.all(result_image[0][1] == 2.0)
    assert torch.all(result_image[0][2] == 1.0)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=-32,
        pad_top=-28,
        pad_right=-32,
        pad_bottom=-28,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )


def test_pre_process_torch_4d_image_with_static_crop_and_center_crop_selected_and_crop_fitting_inside_original_image_with_scaling() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))  # CHW
    image = torch.stack([image, image], dim=0)  # NCHW

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    # img 0
    assert torch.all(result_image[0][0] == 3.0)
    assert torch.all(result_image[0][1] == 2.0)
    assert torch.all(result_image[0][2] == 1.0)
    # img 1
    assert torch.all(result_image[1][0] == 3.0)
    assert torch.all(result_image[1][1] == 2.0)
    assert torch.all(result_image[1][2] == 1.0)

    expected_meta = PreProcessingMetadata(
        pad_left=-32,
        pad_top=-28,
        pad_right=-32,
        pad_bottom=-28,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_4d_not_permuted_image_with_static_crop_and_center_crop_selected_and_crop_fitting_inside_original_image_with_scaling() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    image = torch.from_numpy(image)
    image = torch.stack([image, image], dim=0)  # NHWC

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    # img 0
    assert torch.all(result_image[0][0] == 3.0)
    assert torch.all(result_image[0][1] == 2.0)
    assert torch.all(result_image[0][2] == 1.0)
    # img 1
    assert torch.all(result_image[1][0] == 3.0)
    assert torch.all(result_image[1][1] == 2.0)
    assert torch.all(result_image[1][2] == 1.0)

    expected_meta = PreProcessingMetadata(
        pad_left=-32,
        pad_top=-28,
        pad_right=-32,
        pad_bottom=-28,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_numpy_image_with_static_crop_and_center_crop_selected_and_crop_not_fitting_inside_original_image_with_scaling() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=200, width=200),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    # after center crop - image of size (120, 160, 3) available

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 200, 200)
    assert torch.all(result_image[0][0, 40:160, 20:180] == 3.0)
    assert torch.all(result_image[0][1, 40:160, 20:180] == 2.0)
    assert torch.all(result_image[0][2, 40:160, 20:180] == 1.0)
    assert torch.all(result_image[0][:, :40, :] == 0)
    assert torch.all(result_image[0][:, 160:, :] == 0)
    assert torch.all(result_image[0][:, :, :20] == 0)
    assert torch.all(result_image[0][:, :, 180:] == 0)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=20,
        pad_top=40,
        pad_right=20,
        pad_bottom=40,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=200, width=200),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )


def test_pre_process_numpy_images_list_with_static_crop_and_center_crop_selected_and_crop_not_fitting_inside_original_image_with_scaling() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=200, width=200),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    images = [image, image]

    # when
    result_image, result_meta = pre_process_network_input(
        images=images,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 200, 200)

    # check both items
    for i in range(2):
        assert torch.all(result_image[i][0, 40:160, 20:180] == 3.0)
        assert torch.all(result_image[i][1, 40:160, 20:180] == 2.0)
        assert torch.all(result_image[i][2, 40:160, 20:180] == 1.0)
        assert torch.all(result_image[i][:, :40, :] == 0)
        assert torch.all(result_image[i][:, 160:, :] == 0)
        assert torch.all(result_image[i][:, :, :20] == 0)
        assert torch.all(result_image[i][:, :, 180:] == 0)

    expected_meta = PreProcessingMetadata(
        pad_left=20,
        pad_top=40,
        pad_right=20,
        pad_bottom=40,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=200, width=200),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_3d_image_with_static_crop_and_center_crop_selected_and_crop_not_fitting_inside_original_image_with_scaling() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=200, width=200),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))  # CHW

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 200, 200)
    assert torch.all(result_image[0][0, 40:160, 20:180] == 3.0)
    assert torch.all(result_image[0][1, 40:160, 20:180] == 2.0)
    assert torch.all(result_image[0][2, 40:160, 20:180] == 1.0)
    assert torch.all(result_image[0][:, :40, :] == 0)
    assert torch.all(result_image[0][:, 160:, :] == 0)
    assert torch.all(result_image[0][:, :, :20] == 0)
    assert torch.all(result_image[0][:, :, 180:] == 0)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=20,
        pad_top=40,
        pad_right=20,
        pad_bottom=40,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=200, width=200),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )


def test_pre_process_torch_3d_not_permuted_image_with_static_crop_and_center_crop_selected_and_crop_not_fitting_inside_original_image_with_scaling() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=200, width=200),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    image = torch.from_numpy(image)  # HWC

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 200, 200)
    assert torch.all(result_image[0][0, 40:160, 20:180] == 3.0)
    assert torch.all(result_image[0][1, 40:160, 20:180] == 2.0)
    assert torch.all(result_image[0][2, 40:160, 20:180] == 1.0)
    assert torch.all(result_image[0][:, :40, :] == 0)
    assert torch.all(result_image[0][:, 160:, :] == 0)
    assert torch.all(result_image[0][:, :, :20] == 0)
    assert torch.all(result_image[0][:, :, 180:] == 0)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=20,
        pad_top=40,
        pad_right=20,
        pad_bottom=40,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=200, width=200),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )


def test_pre_process_torch_4d_image_with_static_crop_and_center_crop_selected_and_crop_not_fitting_inside_original_image_with_scaling() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=200, width=200),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
        scaling_factor=10.0,
    )
    img_np = np.ones((200, 200, 3), dtype=np.uint8)
    img_np[40:160, 20:180, :] = (img_np[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    img_t = torch.from_numpy(img_np)
    img_t = torch.permute(img_t, (2, 0, 1))  # CHW
    image = torch.stack([img_t, img_t], dim=0)  # NCHW

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 200, 200)

    for i in range(2):
        assert torch.all(result_image[i][0, 40:160, 20:180] == 3.0)
        assert torch.all(result_image[i][1, 40:160, 20:180] == 2.0)
        assert torch.all(result_image[i][2, 40:160, 20:180] == 1.0)
        assert torch.all(result_image[i][:, :40, :] == 0)
        assert torch.all(result_image[i][:, 160:, :] == 0)
        assert torch.all(result_image[i][:, :, :20] == 0)
        assert torch.all(result_image[i][:, :, 180:] == 0)

    expected_meta = PreProcessingMetadata(
        pad_left=20,
        pad_top=40,
        pad_right=20,
        pad_bottom=40,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=200, width=200),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_4d_not_permuted_image_with_static_crop_and_center_crop_selected_and_crop_not_fitting_inside_original_image_with_scaling() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=200, width=200),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
        scaling_factor=10.0,
    )
    img_np = np.ones((200, 200, 3), dtype=np.uint8)
    img_np[40:160, 20:180, :] = (img_np[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    img_t = torch.from_numpy(img_np)  # HWC
    image = torch.stack([img_t, img_t], dim=0)  # NHWC

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 200, 200)

    for i in range(2):
        assert torch.all(result_image[i][0, 40:160, 20:180] == 3.0)
        assert torch.all(result_image[i][1, 40:160, 20:180] == 2.0)
        assert torch.all(result_image[i][2, 40:160, 20:180] == 1.0)
        assert torch.all(result_image[i][:, :40, :] == 0)
        assert torch.all(result_image[i][:, 160:, :] == 0)
        assert torch.all(result_image[i][:, :, :20] == 0)
        assert torch.all(result_image[i][:, :, 180:] == 0)

    expected_meta = PreProcessingMetadata(
        pad_left=20,
        pad_top=40,
        pad_right=20,
        pad_bottom=40,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=200, width=200),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_numpy_image_with_static_crop_and_center_crop_selected_and_crop_fitting_inside_original_image_with_scaling_and_normalisation() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
        scaling_factor=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    # after center crop - image of size (120, 160, 3) available

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0][0] == 1 / 6)
    assert torch.all(result_image[0][1] == 0.0)
    assert torch.all(result_image[0][2] == -1 / 6)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=-32,
        pad_top=-28,
        pad_right=-32,
        pad_bottom=-28,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )


def test_pre_process_numpy_images_list_with_static_crop_and_center_crop_selected_and_crop_fitting_inside_original_image_with_scaling_and_normalisation() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
        scaling_factor=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert result_image.shape == (2, 3, 64, 96)
    # img 0
    assert torch.all(result_image[0][0] == 1 / 6)
    assert torch.all(result_image[0][1] == 0.0)
    assert torch.all(result_image[0][2] == -1 / 6)
    # img 1
    assert torch.all(result_image[1][0] == 1 / 6)
    assert torch.all(result_image[1][1] == 0.0)
    assert torch.all(result_image[1][2] == -1 / 6)

    expected_meta = PreProcessingMetadata(
        pad_left=-32,
        pad_top=-28,
        pad_right=-32,
        pad_bottom=-28,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20, offset_y=40, crop_width=160, crop_height=120
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_3d_image_with_static_crop_and_center_crop_selected_and_crop_fitting_inside_original_image_with_scaling_and_normalisation() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
        scaling_factor=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))  # CHW

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert result_image.shape == (1, 3, 64, 96)
    assert torch.all(result_image[0][0] == 1 / 6)
    assert torch.all(result_image[0][1] == 0.0)
    assert torch.all(result_image[0][2] == -1 / 6)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=-32,
        pad_top=-28,
        pad_right=-32,
        pad_bottom=-28,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20, offset_y=40, crop_width=160, crop_height=120
        ),
    )


def test_pre_process_torch_3d_not_permuted_image_with_static_crop_and_center_crop_selected_and_crop_fitting_inside_original_image_with_scaling_and_normalisation() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
        scaling_factor=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    image = torch.from_numpy(image)  # HWC

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert result_image.shape == (1, 3, 64, 96)
    assert torch.all(result_image[0][0] == 1 / 6)
    assert torch.all(result_image[0][1] == 0.0)
    assert torch.all(result_image[0][2] == -1 / 6)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=-32,
        pad_top=-28,
        pad_right=-32,
        pad_bottom=-28,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20, offset_y=40, crop_width=160, crop_height=120
        ),
    )


def test_pre_process_torch_4d_image_with_static_crop_and_center_crop_selected_and_crop_fitting_inside_original_image_with_scaling_and_normalisation() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
        scaling_factor=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))  # CHW
    image = torch.stack([image, image], dim=0)  # NCHW

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert result_image.shape == (2, 3, 64, 96)
    # img 0
    assert torch.all(result_image[0][0] == 1 / 6)
    assert torch.all(result_image[0][1] == 0.0)
    assert torch.all(result_image[0][2] == -1 / 6)
    # img 1
    assert torch.all(result_image[1][0] == 1 / 6)
    assert torch.all(result_image[1][1] == 0.0)
    assert torch.all(result_image[1][2] == -1 / 6)

    expected_meta = PreProcessingMetadata(
        pad_left=-32,
        pad_top=-28,
        pad_right=-32,
        pad_bottom=-28,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20, offset_y=40, crop_width=160, crop_height=120
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_4d_not_permuted_image_with_static_crop_and_center_crop_selected_and_crop_fitting_inside_original_image_with_scaling_and_normalisation() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
        scaling_factor=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    image = torch.from_numpy(image)
    image = torch.stack([image, image], dim=0)  # NHWC

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert result_image.shape == (2, 3, 64, 96)
    # img 0
    assert torch.all(result_image[0][0] == 1 / 6)
    assert torch.all(result_image[0][1] == 0.0)
    assert torch.all(result_image[0][2] == -1 / 6)
    # img 1
    assert torch.all(result_image[1][0] == 1 / 6)
    assert torch.all(result_image[1][1] == 0.0)
    assert torch.all(result_image[1][2] == -1 / 6)

    expected_meta = PreProcessingMetadata(
        pad_left=-32,
        pad_top=-28,
        pad_right=-32,
        pad_bottom=-28,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20, offset_y=40, crop_width=160, crop_height=120
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_numpy_image_with_static_crop_and_center_crop_selected_and_crop_not_fitting_inside_original_image_with_scaling_and_normalisation() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=200, width=200),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
        scaling_factor=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    # after center crop - image of size (120, 160, 3) available

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 200, 200)
    assert torch.all(result_image[0][0, 40:160, 20:180] == 1 / 6)
    assert torch.all(result_image[0][1, 40:160, 20:180] == 0.0)
    assert torch.all(result_image[0][2, 40:160, 20:180] == -1 / 6)
    assert torch.all(result_image[0][:, :40, :] == -1 / 3)
    assert torch.all(result_image[0][:, 160:, :] == -1 / 3)
    assert torch.all(result_image[0][:, :, :20] == -1 / 3)
    assert torch.all(result_image[0][:, :, 180:] == -1 / 3)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=20,
        pad_top=40,
        pad_right=20,
        pad_bottom=40,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=200, width=200),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )


def test_pre_process_numpy_images_list_with_static_crop_and_center_crop_selected_and_crop_not_fitting_inside_original_image_with_scaling_and_normalisation() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=200, width=200),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
        scaling_factor=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    # after center crop - image of size (120, 160, 3) available

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 200, 200)
    assert torch.all(result_image[0][0, 40:160, 20:180] == 1 / 6)
    assert torch.all(result_image[0][1, 40:160, 20:180] == 0.0)
    assert torch.all(result_image[0][2, 40:160, 20:180] == -1 / 6)
    assert torch.all(result_image[0][:, :40, :] == -1 / 3)
    assert torch.all(result_image[0][:, 160:, :] == -1 / 3)
    assert torch.all(result_image[0][:, :, :20] == -1 / 3)
    assert torch.all(result_image[0][:, :, 180:] == -1 / 3)
    assert torch.all(result_image[1][0, 40:160, 20:180] == 1 / 6)
    assert torch.all(result_image[1][1, 40:160, 20:180] == 0.0)
    assert torch.all(result_image[1][2, 40:160, 20:180] == -1 / 6)
    assert torch.all(result_image[1][:, :40, :] == -1 / 3)
    assert torch.all(result_image[1][:, 160:, :] == -1 / 3)
    assert torch.all(result_image[1][:, :, :20] == -1 / 3)
    assert torch.all(result_image[1][:, :, 180:] == -1 / 3)
    expected_meta = PreProcessingMetadata(
        pad_left=20,
        pad_top=40,
        pad_right=20,
        pad_bottom=40,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=200, width=200),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_3d_image_with_static_crop_and_center_crop_selected_and_crop_not_fitting_inside_original_image_with_scaling_and_normalisation() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=200, width=200),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
        scaling_factor=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))
    # after center crop - image of size (120, 160, 3) available

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 200, 200)
    assert torch.all(result_image[0][0, 40:160, 20:180] == 1 / 6)
    assert torch.all(result_image[0][1, 40:160, 20:180] == 0.0)
    assert torch.all(result_image[0][2, 40:160, 20:180] == -1 / 6)
    assert torch.all(result_image[0][:, :40, :] == -1 / 3)
    assert torch.all(result_image[0][:, 160:, :] == -1 / 3)
    assert torch.all(result_image[0][:, :, :20] == -1 / 3)
    assert torch.all(result_image[0][:, :, 180:] == -1 / 3)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=20,
        pad_top=40,
        pad_right=20,
        pad_bottom=40,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=200, width=200),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )


def test_pre_process_torch_3d_not_permuted_image_with_static_crop_and_center_crop_selected_and_crop_not_fitting_inside_original_image_with_scaling_and_normalisation() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=200, width=200),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
        scaling_factor=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    image = torch.from_numpy(image)
    # after center crop - image of size (120, 160, 3) available

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 200, 200)
    assert torch.all(result_image[0][0, 40:160, 20:180] == 1 / 6)
    assert torch.all(result_image[0][1, 40:160, 20:180] == 0.0)
    assert torch.all(result_image[0][2, 40:160, 20:180] == -1 / 6)
    assert torch.all(result_image[0][:, :40, :] == -1 / 3)
    assert torch.all(result_image[0][:, 160:, :] == -1 / 3)
    assert torch.all(result_image[0][:, :, :20] == -1 / 3)
    assert torch.all(result_image[0][:, :, 180:] == -1 / 3)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=20,
        pad_top=40,
        pad_right=20,
        pad_bottom=40,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=200, width=200),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )


def test_pre_process_torch_4d_image_with_static_crop_and_center_crop_selected_and_crop_not_fitting_inside_original_image_with_scaling_and_normalisation() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(
        **{
            "static-crop": StaticCrop(
                enabled=True, x_min=10, x_max=90, y_min=20, y_max=80
            )
        }
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=200, width=200),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.CENTER_CROP,
        input_channels=3,
        scaling_factor=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )
    image = np.ones((200, 200, 3), dtype=np.uint8)
    image[40:160, 20:180, :] = (image[40:160, 20:180, :] * (10, 20, 30)).astype(
        np.uint8
    )
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))
    image = torch.stack([image, image])
    # after center crop - image of size (120, 160, 3) available

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 200, 200)
    assert torch.all(result_image[0][0, 40:160, 20:180] == 1 / 6)
    assert torch.all(result_image[0][1, 40:160, 20:180] == 0.0)
    assert torch.all(result_image[0][2, 40:160, 20:180] == -1 / 6)
    assert torch.all(result_image[0][:, :40, :] == -1 / 3)
    assert torch.all(result_image[0][:, 160:, :] == -1 / 3)
    assert torch.all(result_image[0][:, :, :20] == -1 / 3)
    assert torch.all(result_image[0][:, :, 180:] == -1 / 3)
    assert torch.all(result_image[1][0, 40:160, 20:180] == 1 / 6)
    assert torch.all(result_image[1][1, 40:160, 20:180] == 0.0)
    assert torch.all(result_image[1][2, 40:160, 20:180] == -1 / 6)
    assert torch.all(result_image[1][:, :40, :] == -1 / 3)
    assert torch.all(result_image[1][:, 160:, :] == -1 / 3)
    assert torch.all(result_image[1][:, :, :20] == -1 / 3)
    assert torch.all(result_image[1][:, :, 180:] == -1 / 3)
    expected_meta = PreProcessingMetadata(
        pad_left=20,
        pad_top=40,
        pad_right=20,
        pad_bottom=40,
        original_size=ImageDimensions(height=200, width=200),
        size_after_pre_processing=ImageDimensions(height=120, width=160),
        inference_size=ImageDimensions(height=200, width=200),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=20,
            offset_y=40,
            crop_width=160,
            crop_height=120,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_numpy_image_with_longer_edge_fit_selected() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.FIT_LONGER_EDGE,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 56)
    assert torch.all(result_image[0][0] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][2] == 10)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=56),
        scale_width=64 / 192,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_numpy_images_list_with_longer_edge_fit_selected() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.FIT_LONGER_EDGE,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    images = [image, image]

    # when
    with pytest.raises(ModelRuntimeError):
        _ = pre_process_network_input(
            images=images,
            image_pre_processing=image_pre_processing,
            network_input=network_input,
            target_device=torch.device("cpu"),
        )


def test_pre_process_torch_3d_image_with_longer_edge_fit_selected() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.FIT_LONGER_EDGE,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))  # CHW

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 56)
    assert torch.all(result_image[0][0] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][2] == 10)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=56),
        scale_width=64 / 192,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_torch_3d_not_permuted_image_with_longer_edge_fit_selected() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.FIT_LONGER_EDGE,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)  # HWC

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 56)
    assert torch.all(result_image[0][0] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][2] == 10)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=56),
        scale_width=64 / 192,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_torch_4d_image_with_longer_edge_fit_selected() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.FIT_LONGER_EDGE,
        input_channels=3,
    )
    img_np = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    img_t = torch.from_numpy(img_np)
    img_t = torch.permute(img_t, (2, 0, 1))  # CHW
    image = torch.stack([img_t, img_t], dim=0)  # NCHW

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 56)
    for i in range(2):
        assert torch.all(result_image[i][0] == 30)
        assert torch.all(result_image[i][1] == 20)
        assert torch.all(result_image[i][2] == 10)

    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=56),
        scale_width=64 / 192,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_4d_not_permuted_image_with_longer_edge_fit_selected() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.FIT_LONGER_EDGE,
        input_channels=3,
    )
    img_np = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    img_t = torch.from_numpy(img_np)  # HWC
    image = torch.stack([img_t, img_t], dim=0)  # NHWC

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 56)
    for i in range(2):
        assert torch.all(result_image[i][0] == 30)
        assert torch.all(result_image[i][1] == 20)
        assert torch.all(result_image[i][2] == 10)

    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=56),
        scale_width=64 / 192,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_numpy_image_with_longer_edge_fit_selected_with_scaling() -> None:
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.FIT_LONGER_EDGE,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 56)
    assert torch.all(result_image[0][0] == 3.0)
    assert torch.all(result_image[0][1] == 2.0)
    assert torch.all(result_image[0][2] == 1.0)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=56),
        scale_width=64 / 192,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_numpy_images_list_with_longer_edge_fit_selected_with_scaling() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.FIT_LONGER_EDGE,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    # when
    with pytest.raises(ModelRuntimeError):
        _ = pre_process_network_input(
            images=[image, image],
            image_pre_processing=image_pre_processing,
            network_input=network_input,
            target_device=torch.device("cpu"),
        )


def test_pre_process_torch_images_list_with_longer_edge_fit_selected_with_scaling() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.FIT_LONGER_EDGE,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)

    # when
    with pytest.raises(ModelRuntimeError):
        _ = pre_process_network_input(
            images=[image, image],
            image_pre_processing=image_pre_processing,
            network_input=network_input,
            target_device=torch.device("cpu"),
        )


def test_pre_process_torch_3d_image_with_longer_edge_fit_selected_with_scaling() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,  # mirrors your other torch tests
        resize_mode=ResizeMode.FIT_LONGER_EDGE,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))  # CHW

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert result_image.shape == (1, 3, 64, 56)
    assert torch.all(result_image[0][0] == 3.0)
    assert torch.all(result_image[0][1] == 2.0)
    assert torch.all(result_image[0][2] == 1.0)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=56),
        scale_width=64 / 192,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0, offset_y=0, crop_width=168, crop_height=192
        ),
    )


def test_pre_process_torch_3d_not_permuted_image_with_longer_edge_fit_selected_with_scaling() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.FIT_LONGER_EDGE,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)  # HWC

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert result_image.shape == (1, 3, 64, 56)
    assert torch.all(result_image[0][0] == 3.0)
    assert torch.all(result_image[0][1] == 2.0)
    assert torch.all(result_image[0][2] == 1.0)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=56),
        scale_width=64 / 192,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0, offset_y=0, crop_width=168, crop_height=192
        ),
    )


def test_pre_process_torch_4d_image_with_longer_edge_fit_selected_with_scaling() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.FIT_LONGER_EDGE,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))  # CHW
    image = torch.stack([image, image], dim=0)  # NCHW

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert result_image.shape == (2, 3, 64, 56)
    # img 0
    assert torch.all(result_image[0][0] == 3.0)
    assert torch.all(result_image[0][1] == 2.0)
    assert torch.all(result_image[0][2] == 1.0)
    # img 1
    assert torch.all(result_image[1][0] == 3.0)
    assert torch.all(result_image[1][1] == 2.0)
    assert torch.all(result_image[1][2] == 1.0)

    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=56),
        scale_width=64 / 192,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0, offset_y=0, crop_width=168, crop_height=192
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_4d_not_permuted_image_with_longer_edge_fit_selected_with_scaling() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.FIT_LONGER_EDGE,
        input_channels=3,
        scaling_factor=10.0,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.stack([image, image], dim=0)  # NHWC

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert result_image.shape == (2, 3, 64, 56)
    # img 0
    assert torch.all(result_image[0][0] == 3.0)
    assert torch.all(result_image[0][1] == 2.0)
    assert torch.all(result_image[0][2] == 1.0)
    # img 1
    assert torch.all(result_image[1][0] == 3.0)
    assert torch.all(result_image[1][1] == 2.0)
    assert torch.all(result_image[1][2] == 1.0)

    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=56),
        scale_width=64 / 192,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0, offset_y=0, crop_width=168, crop_height=192
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_numpy_image_with_longer_edge_fit_selected_with_scaling_and_normalisation() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.FIT_LONGER_EDGE,
        input_channels=3,
        scaling_factor=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 56)
    assert torch.all(result_image[0][0] == 1 / 6)
    assert torch.all(result_image[0][1] == 0.0)
    assert torch.all(result_image[0][2] == -1 / 6)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=56),
        scale_width=64 / 192,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_numpy_images_list_with_longer_edge_fit_selected_with_scaling_and_normalisation() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.FIT_LONGER_EDGE,
        input_channels=3,
        scaling_factor=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    images = [image, image]

    # when
    with pytest.raises(ModelRuntimeError):
        _ = pre_process_network_input(
            images=images,
            image_pre_processing=image_pre_processing,
            network_input=network_input,
            target_device=torch.device("cpu"),
        )


def test_pre_process_torch_3d_image_with_longer_edge_fit_selected_with_scaling_and_normalisation() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.FIT_LONGER_EDGE,
        input_channels=3,
        scaling_factor=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))  # CHW

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 56)
    assert torch.all(result_image[0][0] == 1 / 6)
    assert torch.all(result_image[0][1] == 0.0)
    assert torch.all(result_image[0][2] == -1 / 6)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=56),
        scale_width=64 / 192,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_torch_3d_not_permuted_image_with_longer_edge_fit_selected_with_scaling_and_normalisation() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.FIT_LONGER_EDGE,
        input_channels=3,
        scaling_factor=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)  # HWC

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 56)
    assert torch.all(result_image[0][0] == 1 / 6)
    assert torch.all(result_image[0][1] == 0.0)
    assert torch.all(result_image[0][2] == -1 / 6)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=56),
        scale_width=64 / 192,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_pre_process_torch_4d_image_with_longer_edge_fit_selected_with_scaling_and_normalisation() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.FIT_LONGER_EDGE,
        input_channels=3,
        scaling_factor=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )
    img_np = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    img_t = torch.from_numpy(img_np)
    img_t = torch.permute(img_t, (2, 0, 1))  # CHW
    image = torch.stack([img_t, img_t], dim=0)  # NCHW

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 56)
    for i in range(2):
        assert torch.all(result_image[i][0] == 1 / 6)
        assert torch.all(result_image[i][1] == 0.0)
        assert torch.all(result_image[i][2] == -1 / 6)

    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=56),
        scale_width=64 / 192,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_pre_process_torch_4d_not_permuted_image_with_longer_edge_fit_selected_with_scaling_and_normalisation() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.FIT_LONGER_EDGE,
        input_channels=3,
        scaling_factor=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )
    img_np = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    img_t = torch.from_numpy(img_np)  # HWC
    image = torch.stack([img_t, img_t], dim=0)  # NHWC

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 56)
    for i in range(2):
        assert torch.all(result_image[i][0] == 1 / 6)
        assert torch.all(result_image[i][1] == 0.0)
        assert torch.all(result_image[i][2] == -1 / 6)

    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=56),
        scale_width=64 / 192,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_grayscale_pre_processing_for_numpy_image() -> None:
    # given
    image_pre_processing = ImagePreProcessing(grayscale=Grayscale(enabled=True))
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0][0] == 18)
    assert torch.all(result_image[0][1] == 18)
    assert torch.all(result_image[0][2] == 18)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


@pytest.mark.parametrize(
    "contrast_type",
    [
        ContrastType.ADAPTIVE_EQUALIZATION,
        ContrastType.CONTRAST_STRETCHING,
        ContrastType.HISTOGRAM_EQUALIZATION,
    ],
)
def test_contrast_pre_processing_for_numpy_image(contrast_type: ContrastType) -> None:
    # given
    image_pre_processing = ImagePreProcessing(
        contrast=Contrast(enabled=True, type=contrast_type)
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


@pytest.mark.parametrize(
    "contrast_type",
    [
        ContrastType.ADAPTIVE_EQUALIZATION,
        ContrastType.CONTRAST_STRETCHING,
        ContrastType.HISTOGRAM_EQUALIZATION,
    ],
)
def test_contrast_and_grayscale_pre_processing_for_numpy_image(
    contrast_type: ContrastType,
) -> None:
    # given
    image_pre_processing = ImagePreProcessing(
        contrast=Contrast(enabled=True, type=contrast_type),
        grayscale=Grayscale(enabled=True),
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_grayscale_pre_processing_for_list_of_numpy_images() -> None:
    # given
    image_pre_processing = ImagePreProcessing(grayscale=Grayscale(enabled=True))
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][0] == 18)
    assert torch.all(result_image[0][1] == 18)
    assert torch.all(result_image[0][2] == 18)
    assert torch.all(result_image[1][0] == 18)
    assert torch.all(result_image[1][1] == 18)
    assert torch.all(result_image[1][2] == 18)
    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


@pytest.mark.parametrize(
    "contrast_type",
    [
        ContrastType.ADAPTIVE_EQUALIZATION,
        ContrastType.CONTRAST_STRETCHING,
        ContrastType.HISTOGRAM_EQUALIZATION,
    ],
)
def test_contrast_pre_processing_for_list_of_numpy_images(
    contrast_type: ContrastType,
) -> None:
    # given
    image_pre_processing = ImagePreProcessing(
        contrast=Contrast(enabled=True, type=contrast_type),
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


@pytest.mark.parametrize(
    "contrast_type",
    [
        ContrastType.ADAPTIVE_EQUALIZATION,
        ContrastType.CONTRAST_STRETCHING,
        ContrastType.HISTOGRAM_EQUALIZATION,
    ],
)
def test_contrast_and_grayscale_pre_processing_for_list_of_numpy_images(
    contrast_type: ContrastType,
) -> None:
    # given
    image_pre_processing = ImagePreProcessing(
        contrast=Contrast(enabled=True, type=contrast_type),
        grayscale=Grayscale(enabled=True),
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_grayscale_pre_processing_for_3d_torch_tensor_image() -> None:
    # given
    image_pre_processing = ImagePreProcessing(grayscale=Grayscale(enabled=True))
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0][0] == 18)
    assert torch.all(result_image[0][1] == 18)
    assert torch.all(result_image[0][2] == 18)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


@pytest.mark.parametrize(
    "contrast_type",
    [
        ContrastType.ADAPTIVE_EQUALIZATION,
        ContrastType.CONTRAST_STRETCHING,
        ContrastType.HISTOGRAM_EQUALIZATION,
    ],
)
def test_contrast_pre_processing_for_3d_torch_tensor_image(
    contrast_type: ContrastType,
) -> None:
    # given
    image_pre_processing = ImagePreProcessing(
        contrast=Contrast(enabled=True, type=contrast_type)
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


@pytest.mark.parametrize(
    "contrast_type",
    [
        ContrastType.ADAPTIVE_EQUALIZATION,
        ContrastType.CONTRAST_STRETCHING,
        ContrastType.HISTOGRAM_EQUALIZATION,
    ],
)
def test_contrast_and_grayscale_pre_processing_for_3d_torch_tensor_image(
    contrast_type: ContrastType,
) -> None:
    # given
    image_pre_processing = ImagePreProcessing(
        contrast=Contrast(enabled=True, type=contrast_type),
        grayscale=Grayscale(enabled=True),
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_grayscale_pre_processing_for_3d_torch_tensor_not_permuted_image() -> None:
    # given
    image_pre_processing = ImagePreProcessing(grayscale=Grayscale(enabled=True))
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert torch.all(result_image[0][0] == 18)
    assert torch.all(result_image[0][1] == 18)
    assert torch.all(result_image[0][2] == 18)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


@pytest.mark.parametrize(
    "contrast_type",
    [
        ContrastType.ADAPTIVE_EQUALIZATION,
        ContrastType.CONTRAST_STRETCHING,
        ContrastType.HISTOGRAM_EQUALIZATION,
    ],
)
def test_contrast_pre_processing_for_3d_torch_tensor_not_permuted_image(
    contrast_type: ContrastType,
) -> None:
    # given
    image_pre_processing = ImagePreProcessing(
        contrast=Contrast(enabled=True, type=contrast_type)
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


@pytest.mark.parametrize(
    "contrast_type",
    [
        ContrastType.ADAPTIVE_EQUALIZATION,
        ContrastType.CONTRAST_STRETCHING,
        ContrastType.HISTOGRAM_EQUALIZATION,
    ],
)
def test_contrast_and_grayscalepre_processing_for_3d_torch_tensor_not_permuted_image(
    contrast_type: ContrastType,
) -> None:
    # given
    image_pre_processing = ImagePreProcessing(
        contrast=Contrast(enabled=True, type=contrast_type),
        grayscale=Grayscale(enabled=True),
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 96)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )


def test_grayscale_pre_processing_for_4d_torch_tensor_image() -> None:
    # given
    image_pre_processing = ImagePreProcessing(grayscale=Grayscale(enabled=True))
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))
    image = torch.stack([image, image], dim=0)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][0] == 18)
    assert torch.all(result_image[0][1] == 18)
    assert torch.all(result_image[0][2] == 18)
    assert torch.all(result_image[1][0] == 18)
    assert torch.all(result_image[1][1] == 18)
    assert torch.all(result_image[1][2] == 18)
    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


@pytest.mark.parametrize(
    "contrast_type",
    [
        ContrastType.ADAPTIVE_EQUALIZATION,
        ContrastType.CONTRAST_STRETCHING,
        ContrastType.HISTOGRAM_EQUALIZATION,
    ],
)
def test_contrast_pre_processing_for_4d_torch_tensor_image(
    contrast_type: ContrastType,
) -> None:
    # given
    image_pre_processing = ImagePreProcessing(
        contrast=Contrast(enabled=True, type=contrast_type),
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))
    image = torch.stack([image, image], dim=0)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


@pytest.mark.parametrize(
    "contrast_type",
    [
        ContrastType.ADAPTIVE_EQUALIZATION,
        ContrastType.CONTRAST_STRETCHING,
        ContrastType.HISTOGRAM_EQUALIZATION,
    ],
)
def test_contrast_and_grayscale_pre_processing_for_4d_torch_tensor_image(
    contrast_type: ContrastType,
) -> None:
    # given
    image_pre_processing = ImagePreProcessing(
        contrast=Contrast(enabled=True, type=contrast_type),
        grayscale=Grayscale(enabled=True),
    )
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))
    image = torch.stack([image, image], dim=0)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_grayscale_pre_processing_for_4d_torch_tensor_not_permuted_image() -> None:
    # given
    image_pre_processing = ImagePreProcessing(grayscale=Grayscale(enabled=True))
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.stack([image, image], dim=0)

    # when
    result_image, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][0] == 18)
    assert torch.all(result_image[0][1] == 18)
    assert torch.all(result_image[0][2] == 18)
    assert torch.all(result_image[1][0] == 18)
    assert torch.all(result_image[1][1] == 18)
    assert torch.all(result_image[1][2] == 18)
    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_grayscale_pre_processing_for_list_of_3d_torch_tensor_images() -> None:
    # given
    image_pre_processing = ImagePreProcessing(grayscale=Grayscale(enabled=True))
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][0] == 18)
    assert torch.all(result_image[0][1] == 18)
    assert torch.all(result_image[0][2] == 18)
    assert torch.all(result_image[1][0] == 18)
    assert torch.all(result_image[1][1] == 18)
    assert torch.all(result_image[1][2] == 18)
    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_grayscale_pre_processing_for_list_of_3d_torch_tensor_not_permuted_images() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(grayscale=Grayscale(enabled=True))
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)

    # when
    result_image, result_meta = pre_process_network_input(
        images=[image, image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 96)
    assert torch.all(result_image[0][0] == 18)
    assert torch.all(result_image[0][1] == 18)
    assert torch.all(result_image[0][2] == 18)
    assert torch.all(result_image[1][0] == 18)
    assert torch.all(result_image[1][1] == 18)
    assert torch.all(result_image[1][2] == 18)
    expected_meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        size_after_pre_processing=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=96),
        scale_width=96 / 168,
        scale_height=64 / 192,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=168,
            crop_height=192,
        ),
    )
    assert result_meta[0] == expected_meta
    assert result_meta[1] == expected_meta


def test_grayscale_pre_processing_for_list_of_4d_torch_tensor_images() -> None:
    # given
    image_pre_processing = ImagePreProcessing(grayscale=Grayscale(enabled=True))
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))
    image = torch.unsqueeze(image, dim=0)

    # when
    with pytest.raises(ModelRuntimeError):
        _ = pre_process_network_input(
            images=[image, image],
            image_pre_processing=image_pre_processing,
            network_input=network_input,
            target_device=torch.device("cpu"),
            input_color_format="rgb",
        )


def test_grayscale_pre_processing_for_list_of_4d_torch_tensor_not_permuted_images() -> (
    None
):
    # given
    image_pre_processing = ImagePreProcessing(grayscale=Grayscale(enabled=True))
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.BGR,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    image = torch.from_numpy(image)
    image = torch.unsqueeze(image, dim=0)

    # when
    with pytest.raises(ModelRuntimeError):
        _ = pre_process_network_input(
            images=[image, image],
            image_pre_processing=image_pre_processing,
            network_input=network_input,
            target_device=torch.device("cpu"),
            input_color_format="rgb",
        )


def test_make_the_value_divisible_when_value_already_divisible() -> None:
    # when
    result = make_the_value_divisible(4, 2)

    # then
    assert result == 4


def test_make_the_value_divisible_when_value_not_divisible() -> None:
    # when
    result = make_the_value_divisible(13, 2)

    # then
    assert result == 14
