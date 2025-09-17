import numpy as np
import pytest
import torch
from inference_exp.entities import ImageDimensions
from inference_exp.errors import ModelRuntimeError
from inference_exp.models.common.roboflow.model_packages import (
    ColorMode,
    PreProcessingMetadata,
)
from inference_exp.models.common.roboflow.pre_processing import (
    extract_input_images_dimensions,
    images_to_pillow,
    pre_process_images_tensor,
    pre_process_images_tensor_list,
    pre_process_numpy_image,
    pre_process_numpy_images_list,
)
from PIL.Image import Image


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
    pre_processing_config = PreProcessingConfig(
        mode=PreProcessingMode.STRETCH, target_size=ImageDimensions(height=64, width=64)
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    # when
    result_image, result_meta = pre_process_numpy_image(
        image=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_mode=ColorMode.RGB,
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 64)
    assert torch.all(result_image[0][0] == 30)
    assert torch.all(result_image[0][1] == 20)
    assert torch.all(result_image[0][2] == 10)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=64),
        scale_width=64 / 168,
        scale_height=64 / 192,
    )


def test_pre_process_numpy_image_with_stretch_and_rescaling() -> None:
    # given
    pre_processing_config = PreProcessingConfig(
        mode=PreProcessingMode.STRETCH,
        target_size=ImageDimensions(height=64, width=64),
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    # when
    result_image, result_meta = pre_process_numpy_image(
        image=image,
        pre_processing_config=pre_processing_config,
        expected_network_color_format="rgb",
        target_device=torch.device("cpu"),
        rescaling_constant=10.0,
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 64)
    assert torch.all(result_image[0][0] == 3.0)
    assert torch.all(result_image[0][1] == 2.0)
    assert torch.all(result_image[0][2] == 1.0)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=64),
        scale_width=64 / 168,
        scale_height=64 / 192,
    )


def test_pre_process_numpy_image_with_stretch_rescaling_and_normalization() -> None:
    # given
    pre_processing_config = PreProcessingConfig(
        mode=PreProcessingMode.STRETCH,
        target_size=ImageDimensions(height=64, width=64),
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    # when
    result_image, result_meta = pre_process_numpy_image(
        image=image,
        pre_processing_config=pre_processing_config,
        expected_network_color_format="rgb",
        target_device=torch.device("cpu"),
        rescaling_constant=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 64)
    assert torch.all(result_image[0][0] == 1 / 6)
    assert torch.all(result_image[0][1] == 0.0)
    assert torch.all(result_image[0][2] == -1 / 6)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=64),
        scale_width=64 / 168,
        scale_height=64 / 192,
    )


def test_pre_process_numpy_image_with_letterbox_selected_with_invalid_config() -> None:
    # given
    pre_processing_config = PreProcessingConfig(
        mode=PreProcessingMode.LETTERBOX,
        target_size=ImageDimensions(height=64, width=64),
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    # when
    with pytest.raises(ModelRuntimeError):
        _ = pre_process_numpy_image(
            image=image,
            pre_processing_config=pre_processing_config,
            expected_network_color_format="rgb",
            target_device=torch.device("cpu"),
            rescaling_constant=None,
        )


def test_pre_process_numpy_image_with_letterbox_selected() -> None:
    # given
    pre_processing_config = PreProcessingConfig(
        mode=PreProcessingMode.LETTERBOX,
        target_size=ImageDimensions(height=64, width=64),
        padding_value=0,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    # when
    result_image, result_meta = pre_process_numpy_image(
        image=image,
        pre_processing_config=pre_processing_config,
        expected_network_color_format="rgb",
        target_device=torch.device("cpu"),
        rescaling_constant=None,
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 64)
    assert torch.all(result_image[0, :, :, :4] == 0)
    assert torch.all(result_image[0, :, :, 60:] == 0)
    assert torch.all(result_image[0, 0, :, 4:60] == 30)
    assert torch.all(result_image[0, 1, :, 4:60] == 20)
    assert torch.all(result_image[0, 2, :, 4:60] == 10)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=4,
        pad_top=0,
        pad_right=4,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=64),
        scale_width=1 / 3,
        scale_height=1 / 3,
    )


def test_pre_process_numpy_image_with_letterbox_and_rescaling() -> None:
    # given
    pre_processing_config = PreProcessingConfig(
        mode=PreProcessingMode.LETTERBOX,
        target_size=ImageDimensions(height=64, width=64),
        padding_value=0,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    # when
    result_image, result_meta = pre_process_numpy_image(
        image=image,
        pre_processing_config=pre_processing_config,
        expected_network_color_format="rgb",
        target_device=torch.device("cpu"),
        rescaling_constant=10.0,
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 64)
    assert tuple(result_image.shape) == (1, 3, 64, 64)
    assert torch.all(result_image[0, :, :, :4] == 0)
    assert torch.all(result_image[0, :, :, 60:] == 0)
    assert torch.all(result_image[0, 0, :, 4:60] == 3)
    assert torch.all(result_image[0, 1, :, 4:60] == 2)
    assert torch.all(result_image[0, 2, :, 4:60] == 1)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=4,
        pad_top=0,
        pad_right=4,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=64),
        scale_width=1 / 3,
        scale_height=1 / 3,
    )


def test_pre_process_numpy_image_with_letterbox_rescaling_and_normalization() -> None:
    # given
    pre_processing_config = PreProcessingConfig(
        mode=PreProcessingMode.LETTERBOX,
        target_size=ImageDimensions(height=64, width=64),
        padding_value=0,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    # when
    result_image, result_meta = pre_process_numpy_image(
        image=image,
        pre_processing_config=pre_processing_config,
        expected_network_color_format="rgb",
        target_device=torch.device("cpu"),
        rescaling_constant=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (1, 3, 64, 64)
    assert torch.all(result_image[0, :, :, :4] == -1 / 3)
    assert torch.all(result_image[0, :, :, 60:] == -1 / 3)
    assert torch.all(result_image[0, 0, :, 4:60] == 1 / 6)
    assert torch.all(result_image[0, 1, :, 4:60] == 0)
    assert torch.all(result_image[0, 2, :, 4:60] == -1 / 6)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=4,
        pad_top=0,
        pad_right=4,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=64),
        scale_width=1 / 3,
        scale_height=1 / 3,
    )


def test_pre_process_numpy_images_list() -> None:
    # given
    pre_processing_config = PreProcessingConfig(
        mode=PreProcessingMode.STRETCH,
        target_size=ImageDimensions(height=64, width=64),
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    # when
    result_image, result_meta = pre_process_numpy_images_list(
        images=[image, image],
        pre_processing_config=pre_processing_config,
        expected_network_color_format="rgb",
        target_device=torch.device("cpu"),
        rescaling_constant=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )

    # then
    assert isinstance(result_image, torch.Tensor)
    assert tuple(result_image.shape) == (2, 3, 64, 64)
    assert torch.all(result_image[0][0] == 1 / 6)
    assert torch.all(result_image[0][1] == 0.0)
    assert torch.all(result_image[0][2] == -1 / 6)
    assert torch.all(result_image[1][0] == 1 / 6)
    assert torch.all(result_image[1][1] == 0.0)
    assert torch.all(result_image[1][2] == -1 / 6)
    assert len(result_meta) == 2
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=64),
        scale_width=64 / 168,
        scale_height=64 / 192,
    )
    assert result_meta[1] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=64),
        scale_width=64 / 168,
        scale_height=64 / 192,
    )


def test_pre_process_images_tensor_list_when_invalid_pre_processing_config_provided() -> (
    None
):
    # given
    pre_processing_config = PreProcessingConfig(
        mode=PreProcessingMode.NONE,
    )
    image = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(2, 0, 1)

    # when
    with pytest.raises(ModelRuntimeError):
        _ = pre_process_images_tensor_list(
            images=[image],
            pre_processing_config=pre_processing_config,
            expected_network_color_format="rgb",
            target_device=torch.device("cpu"),
        )


def test_pre_process_images_tensor_list_when_stretch_config_provided_and_hwc_format_provided() -> (
    None
):
    # given
    pre_processing_config = PreProcessingConfig(
        mode=PreProcessingMode.STRETCH,
        target_size=ImageDimensions(height=64, width=64),
    )
    image = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    )

    # when
    result_tensor, result_meta = pre_process_images_tensor_list(
        images=[image],
        pre_processing_config=pre_processing_config,
        expected_network_color_format="rgb",
        target_device=torch.device("cpu"),
        rescaling_constant=None,
    )

    # then
    assert tuple(result_tensor.shape) == (1, 3, 64, 64)
    assert len(result_meta) == 1
    assert torch.all(result_tensor[0][2] == 30)
    assert torch.all(result_tensor[0][1] == 20)
    assert torch.all(result_tensor[0][0] == 10)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=64),
        scale_width=64 / 168,
        scale_height=64 / 192,
    )


def test_pre_process_images_tensor_list_when_stretch_config_provided_and_chw_format_provided() -> (
    None
):
    # given
    pre_processing_config = PreProcessingConfig(
        mode=PreProcessingMode.STRETCH,
        target_size=ImageDimensions(height=64, width=64),
    )
    image = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(2, 0, 1)

    # when
    result_tensor, result_meta = pre_process_images_tensor_list(
        images=[image],
        pre_processing_config=pre_processing_config,
        expected_network_color_format="rgb",
        target_device=torch.device("cpu"),
        rescaling_constant=None,
    )

    # then
    assert tuple(result_tensor.shape) == (1, 3, 64, 64)
    assert len(result_meta) == 1
    assert torch.all(result_tensor[0][2] == 30)
    assert torch.all(result_tensor[0][1] == 20)
    assert torch.all(result_tensor[0][0] == 10)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=64),
        scale_width=64 / 168,
        scale_height=64 / 192,
    )


def test_pre_process_images_tensor_list_when_stretch_config_provided_and_rescaling_selected() -> (
    None
):
    # given
    pre_processing_config = PreProcessingConfig(
        mode=PreProcessingMode.STRETCH,
        target_size=ImageDimensions(height=64, width=64),
    )
    image = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(2, 0, 1)

    # when
    result_tensor, result_meta = pre_process_images_tensor_list(
        images=[image],
        pre_processing_config=pre_processing_config,
        expected_network_color_format="rgb",
        target_device=torch.device("cpu"),
        rescaling_constant=10.0,
    )

    # then
    assert tuple(result_tensor.shape) == (1, 3, 64, 64)
    assert len(result_meta) == 1
    assert torch.all(result_tensor[0][2] == 3)
    assert torch.all(result_tensor[0][1] == 2)
    assert torch.all(result_tensor[0][0] == 1)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=64),
        scale_width=64 / 168,
        scale_height=64 / 192,
    )


def test_pre_process_images_tensor_list_when_stretch_config_provided_and_rescaling_with_normalization_selected() -> (
    None
):
    # given
    pre_processing_config = PreProcessingConfig(
        mode=PreProcessingMode.STRETCH,
        target_size=ImageDimensions(height=64, width=64),
    )
    image = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(2, 0, 1)

    # when
    result_tensor, result_meta = pre_process_images_tensor_list(
        images=[image],
        pre_processing_config=pre_processing_config,
        expected_network_color_format="rgb",
        target_device=torch.device("cpu"),
        rescaling_constant=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )

    # then
    assert tuple(result_tensor.shape) == (1, 3, 64, 64)
    assert len(result_meta) == 1
    assert torch.all(result_tensor[0][2] == 1 / 6)
    assert torch.all(result_tensor[0][1] == 0)
    assert torch.all(result_tensor[0][0] == -1 / 6)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=64),
        scale_width=64 / 168,
        scale_height=64 / 192,
    )


def test_pre_process_images_tensor_list_when_invalid_letterbox_config_provided() -> (
    None
):
    # given
    pre_processing_config = PreProcessingConfig(
        mode=PreProcessingMode.LETTERBOX,
        target_size=ImageDimensions(height=64, width=64),
    )
    image = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(2, 0, 1)

    # when
    with pytest.raises(ModelRuntimeError):
        _ = pre_process_images_tensor_list(
            images=[image],
            pre_processing_config=pre_processing_config,
            expected_network_color_format="rgb",
            target_device=torch.device("cpu"),
            rescaling_constant=10.0,
            normalization=([2, 2, 2], [6, 6, 6]),
        )


def test_pre_process_images_tensor_list_when_letterbox_selected() -> None:
    # given
    pre_processing_config = PreProcessingConfig(
        mode=PreProcessingMode.LETTERBOX,
        target_size=ImageDimensions(height=64, width=64),
        padding_value=0,
    )
    image = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(2, 0, 1)

    # when
    result_tensor, result_meta = pre_process_images_tensor_list(
        images=[image],
        pre_processing_config=pre_processing_config,
        expected_network_color_format="rgb",
        target_device=torch.device("cpu"),
        rescaling_constant=None,
    )

    # then
    assert tuple(result_tensor.shape) == (1, 3, 64, 64)
    assert len(result_meta) == 1
    assert torch.all(result_tensor[0, :, :, :4] == 0)
    assert torch.all(result_tensor[0, :, :, 60:] == 0)
    assert torch.all(result_tensor[0, 0, :, 4:60] == 10)
    assert torch.all(result_tensor[0, 1, :, 4:60] == 20)
    assert torch.allclose(
        result_tensor[0, 2, :, 4:60], torch.ones_like(result_tensor[0, 2, :, 4:60]) * 30
    )
    assert (result_meta[0].pad_left, result_meta[0].pad_right) == (4, 4)
    assert (result_meta[0].pad_top, result_meta[0].pad_bottom) == (0, 0)
    assert abs(result_meta[0].scale_width - 1 / 3) < 1e-5
    assert abs(result_meta[0].scale_height - 1 / 3) < 1e-5


def test_pre_process_images_tensor_list_when_letterbox_with_rescaling_selected() -> (
    None
):
    # given
    pre_processing_config = PreProcessingConfig(
        mode=PreProcessingMode.LETTERBOX,
        target_size=ImageDimensions(height=64, width=64),
        padding_value=0,
    )
    image = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(2, 0, 1)

    # when
    result_tensor, result_meta = pre_process_images_tensor_list(
        images=[image],
        pre_processing_config=pre_processing_config,
        expected_network_color_format="rgb",
        target_device=torch.device("cpu"),
        rescaling_constant=10.0,
    )

    # then
    assert tuple(result_tensor.shape) == (1, 3, 64, 64)
    assert len(result_meta) == 1
    assert torch.all(result_tensor[0, :, :, :4] == 0)
    assert torch.all(result_tensor[0, :, :, 60:] == 0)
    assert torch.allclose(
        result_tensor[0, 0, :, 4:60], torch.ones_like(result_tensor[0, 0, :, 4:60])
    )
    assert torch.allclose(
        result_tensor[0, 1, :, 4:60], torch.ones_like(result_tensor[0, 1, :, 4:60]) * 2
    )
    assert torch.allclose(
        result_tensor[0, 2, :, 4:60],
        torch.ones_like(result_tensor[0, 2, :, 4:60]) * 3.0,
    )
    assert (result_meta[0].pad_left, result_meta[0].pad_right) == (4, 4)
    assert (result_meta[0].pad_top, result_meta[0].pad_bottom) == (0, 0)
    assert abs(result_meta[0].scale_width - 1 / 3) < 1e-5
    assert abs(result_meta[0].scale_height - 1 / 3) < 1e-5


def test_pre_process_images_tensor_list_when_letterbox_with_rescaling_and_normalization_selected() -> (
    None
):
    # given
    pre_processing_config = PreProcessingConfig(
        mode=PreProcessingMode.LETTERBOX,
        target_size=ImageDimensions(height=64, width=64),
        padding_value=0,
    )
    image = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(2, 0, 1)

    # when
    result_tensor, result_meta = pre_process_images_tensor_list(
        images=[image],
        pre_processing_config=pre_processing_config,
        expected_network_color_format="rgb",
        target_device=torch.device("cpu"),
        rescaling_constant=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )

    # then
    assert tuple(result_tensor.shape) == (1, 3, 64, 64)
    assert len(result_meta) == 1
    assert torch.allclose(
        result_tensor[0, :, :, :4], torch.ones_like(result_tensor[0, :, :, :4]) * -1 / 3
    )
    assert torch.allclose(
        result_tensor[0, :, :, 60:],
        torch.ones_like(result_tensor[0, :, :, :4]) * -1 / 3,
    )
    assert torch.allclose(
        result_tensor[0, 0, :, 4:60],
        torch.ones_like(result_tensor[0, 0, :, 4:60]) * -1 / 6,
    )
    assert torch.allclose(
        result_tensor[0, 1, :, 4:60], torch.zeros_like(result_tensor[0, 1, :, 4:60])
    )
    assert torch.allclose(
        result_tensor[0, 2, :, 4:60],
        torch.ones_like(result_tensor[0, 2, :, 4:60]) * 1 / 6,
    )
    assert (result_meta[0].pad_left, result_meta[0].pad_right) == (4, 4)
    assert (result_meta[0].pad_top, result_meta[0].pad_bottom) == (0, 0)
    assert abs(result_meta[0].scale_width - 1 / 3) < 1e-5
    assert abs(result_meta[0].scale_height - 1 / 3) < 1e-5


def test_pre_process_images_tensor_list_when_unsupported_pre_processing_mode_selected() -> (
    None
):
    # given
    pre_processing_config = PreProcessingConfig(
        mode="some",
        target_size=ImageDimensions(height=64, width=64),
        padding_value=0,
    )
    image = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(2, 0, 1)

    # when
    with pytest.raises(ModelRuntimeError):
        _ = pre_process_images_tensor_list(
            images=[image],
            pre_processing_config=pre_processing_config,
            expected_network_color_format="rgb",
            target_device=torch.device("cpu"),
            rescaling_constant=10.0,
        )


def test_pre_process_images_tensor_when_pre_processing_was_misconfigured() -> None:
    # given
    pre_processing_config = PreProcessingConfig(
        mode=PreProcessingMode.NONE,
    )
    images = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(2, 0, 1)

    # when
    with pytest.raises(ModelRuntimeError):
        _ = pre_process_images_tensor(
            images=images,
            pre_processing_config=pre_processing_config,
            expected_network_color_format="rgb",
            target_device=torch.device("cpu"),
            rescaling_constant=10.0,
        )


def test_pre_process_images_tensor_when_stretch_config_provided_and_hwc_format_provided() -> (
    None
):
    # given
    pre_processing_config = PreProcessingConfig(
        mode=PreProcessingMode.STRETCH,
        target_size=ImageDimensions(height=64, width=64),
    )
    images = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    )

    # when
    result_tensor, result_meta = pre_process_images_tensor(
        images=images,
        pre_processing_config=pre_processing_config,
        expected_network_color_format="rgb",
        target_device=torch.device("cpu"),
        rescaling_constant=None,
    )

    # then
    assert tuple(result_tensor.shape) == (1, 3, 64, 64)
    assert len(result_meta) == 1
    assert torch.all(result_tensor[0][2] == 30)
    assert torch.all(result_tensor[0][1] == 20)
    assert torch.all(result_tensor[0][0] == 10)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=64),
        scale_width=64 / 168,
        scale_height=64 / 192,
    )


def test_pre_process_images_tensor_when_stretch_config_provided_and_chw_format_provided() -> (
    None
):
    # given
    pre_processing_config = PreProcessingConfig(
        mode=PreProcessingMode.STRETCH,
        target_size=ImageDimensions(height=64, width=64),
    )
    images = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(2, 0, 1)

    # when
    result_tensor, result_meta = pre_process_images_tensor(
        images=images,
        pre_processing_config=pre_processing_config,
        expected_network_color_format="rgb",
        target_device=torch.device("cpu"),
        rescaling_constant=None,
    )

    # then
    assert tuple(result_tensor.shape) == (1, 3, 64, 64)
    assert len(result_meta) == 1
    assert torch.all(result_tensor[0][2] == 30)
    assert torch.all(result_tensor[0][1] == 20)
    assert torch.all(result_tensor[0][0] == 10)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=64),
        scale_width=64 / 168,
        scale_height=64 / 192,
    )


def test_pre_process_images_tensor_when_stretch_config_provided_with_rescaling() -> (
    None
):
    # given
    pre_processing_config = PreProcessingConfig(
        mode=PreProcessingMode.STRETCH,
        target_size=ImageDimensions(height=64, width=64),
    )
    images = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(2, 0, 1)

    # when
    result_tensor, result_meta = pre_process_images_tensor(
        images=images,
        pre_processing_config=pre_processing_config,
        expected_network_color_format="rgb",
        target_device=torch.device("cpu"),
        rescaling_constant=10.0,
    )

    # then
    assert tuple(result_tensor.shape) == (1, 3, 64, 64)
    assert len(result_meta) == 1
    assert torch.all(result_tensor[0][2] == 3)
    assert torch.all(result_tensor[0][1] == 2)
    assert torch.all(result_tensor[0][0] == 1)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=64),
        scale_width=64 / 168,
        scale_height=64 / 192,
    )


def test_pre_process_images_tensor_when_stretch_config_provided_with_rescaling_and_normalization() -> (
    None
):
    # given
    pre_processing_config = PreProcessingConfig(
        mode=PreProcessingMode.STRETCH,
        target_size=ImageDimensions(height=64, width=64),
    )
    images = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(2, 0, 1)

    # when
    result_tensor, result_meta = pre_process_images_tensor(
        images=images,
        pre_processing_config=pre_processing_config,
        expected_network_color_format="rgb",
        target_device=torch.device("cpu"),
        rescaling_constant=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )

    # then
    assert tuple(result_tensor.shape) == (1, 3, 64, 64)
    assert len(result_meta) == 1
    assert torch.all(result_tensor[0][2] == 1 / 6)
    assert torch.all(result_tensor[0][1] == 0)
    assert torch.all(result_tensor[0][0] == -1 / 6)
    assert result_meta[0] == PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=192, width=168),
        inference_size=ImageDimensions(height=64, width=64),
        scale_width=64 / 168,
        scale_height=64 / 192,
    )


def test_pre_process_images_tensor_when_letterbox_config_provided_without_padding() -> (
    None
):
    # given
    pre_processing_config = PreProcessingConfig(
        mode=PreProcessingMode.LETTERBOX,
        target_size=ImageDimensions(height=64, width=64),
    )
    images = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(2, 0, 1)

    # when
    with pytest.raises(ModelRuntimeError):
        _ = pre_process_images_tensor(
            images=images,
            pre_processing_config=pre_processing_config,
            expected_network_color_format="rgb",
            target_device=torch.device("cpu"),
            rescaling_constant=None,
        )


def test_pre_process_images_tensor_when_letterbox_config_provided() -> None:
    # given
    pre_processing_config = PreProcessingConfig(
        mode=PreProcessingMode.LETTERBOX,
        target_size=ImageDimensions(height=64, width=64),
        padding_value=0,
    )
    images = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(2, 0, 1)

    # when
    result_tensor, result_meta = pre_process_images_tensor(
        images=images,
        pre_processing_config=pre_processing_config,
        expected_network_color_format="rgb",
        target_device=torch.device("cpu"),
        rescaling_constant=None,
    )

    # then
    assert tuple(result_tensor.shape) == (1, 3, 64, 64)
    assert len(result_meta) == 1
    assert torch.all(result_tensor[0, :, :, :4] == 0)
    assert torch.all(result_tensor[0, :, :, 60:] == 0)
    assert torch.allclose(
        result_tensor[0, 0, :, 4:60], torch.ones_like(result_tensor[0, 0, :, 4:60]) * 10
    )
    assert torch.allclose(
        result_tensor[0, 1, :, 4:60], torch.ones_like(result_tensor[0, 1, :, 4:60]) * 20
    )
    assert torch.allclose(
        result_tensor[0, 2, :, 4:60], torch.ones_like(result_tensor[0, 2, :, 4:60]) * 30
    )
    assert (result_meta[0].pad_left, result_meta[0].pad_right) == (4, 4)
    assert (result_meta[0].pad_top, result_meta[0].pad_bottom) == (0, 0)
    assert abs(result_meta[0].scale_width - 1 / 3) < 1e-5
    assert abs(result_meta[0].scale_height - 1 / 3) < 1e-5


def test_pre_process_images_tensor_when_letterbox_config_with_rescaling_provided() -> (
    None
):
    # given
    pre_processing_config = PreProcessingConfig(
        mode=PreProcessingMode.LETTERBOX,
        target_size=ImageDimensions(height=64, width=64),
        padding_value=0,
    )
    images = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(2, 0, 1)

    # when
    result_tensor, result_meta = pre_process_images_tensor(
        images=images,
        pre_processing_config=pre_processing_config,
        expected_network_color_format="rgb",
        target_device=torch.device("cpu"),
        rescaling_constant=10.0,
    )

    # then
    assert tuple(result_tensor.shape) == (1, 3, 64, 64)
    assert len(result_meta) == 1
    assert torch.all(result_tensor[0, :, :, :4] == 0)
    assert torch.all(result_tensor[0, :, :, 60:] == 0)
    assert torch.allclose(
        result_tensor[0, 0, :, 4:60], torch.ones_like(result_tensor[0, 0, :, 4:60])
    )
    assert torch.allclose(
        result_tensor[0, 1, :, 4:60], torch.ones_like(result_tensor[0, 1, :, 4:60]) * 2
    )
    assert torch.allclose(
        result_tensor[0, 2, :, 4:60], torch.ones_like(result_tensor[0, 2, :, 4:60]) * 3
    )
    assert (result_meta[0].pad_left, result_meta[0].pad_right) == (4, 4)
    assert (result_meta[0].pad_top, result_meta[0].pad_bottom) == (0, 0)
    assert abs(result_meta[0].scale_width - 1 / 3) < 1e-5
    assert abs(result_meta[0].scale_height - 1 / 3) < 1e-5


def test_pre_process_images_tensor_when_letterbox_config_with_rescaling_and_normalization_provided() -> (
    None
):
    # given
    pre_processing_config = PreProcessingConfig(
        mode=PreProcessingMode.LETTERBOX,
        target_size=ImageDimensions(height=64, width=64),
        padding_value=0,
    )
    images = torch.from_numpy(
        (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)
    ).permute(2, 0, 1)

    # when
    result_tensor, result_meta = pre_process_images_tensor(
        images=images,
        pre_processing_config=pre_processing_config,
        expected_network_color_format="rgb",
        target_device=torch.device("cpu"),
        rescaling_constant=10.0,
        normalization=([2, 2, 2], [6, 6, 6]),
    )

    # then
    assert tuple(result_tensor.shape) == (1, 3, 64, 64)
    assert len(result_meta) == 1
    assert torch.allclose(
        result_tensor[0, :, :, :4], torch.ones_like(result_tensor[0, :, :, :4]) * -1 / 3
    )
    assert torch.allclose(
        result_tensor[0, :, :, 60:],
        torch.ones_like(result_tensor[0, :, :, :4]) * -1 / 3,
    )
    assert torch.allclose(
        result_tensor[0, 0, :, 4:60],
        torch.ones_like(result_tensor[0, 0, :, 4:60]) * -1 / 6,
    )
    assert torch.allclose(
        result_tensor[0, 1, :, 4:60], torch.zeros_like(result_tensor[0, 1, :, 4:60])
    )
    assert torch.allclose(
        result_tensor[0, 2, :, 4:60],
        torch.ones_like(result_tensor[0, 2, :, 4:60]) * 1 / 6,
    )
    assert (result_meta[0].pad_left, result_meta[0].pad_right) == (4, 4)
    assert (result_meta[0].pad_top, result_meta[0].pad_bottom) == (0, 0)
    assert abs(result_meta[0].scale_width - 1 / 3) < 1e-5
    assert abs(result_meta[0].scale_height - 1 / 3) < 1e-5
