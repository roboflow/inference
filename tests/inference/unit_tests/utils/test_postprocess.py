import numpy as np
import pytest

from inference.core.exceptions import PostProcessingError
from inference.core.utils.postprocess import (
    cosine_similarity,
    crop_mask,
    standardise_static_crop,
    get_static_crop_dimensions,
)


def test_cosine_similarity_against_vectors() -> None:
    # given
    a = np.array([1, 0])
    b = np.array([0, 1])

    # when
    result = cosine_similarity(a=a, b=b)

    # then
    assert abs(result) < 1e-5


def test_cosine_similarity_orthogonal_vectors() -> None:
    # given
    a = np.array([1, 0])
    b = np.array([-2, 0])

    # when
    result = cosine_similarity(a=a, b=b)

    # then
    assert abs(result + 1) < 1e-5


def test_cosine_similarity_against_semi_aligned_vectors() -> None:
    # given
    a = np.array([5, 5])
    b = np.array([1, 0])

    # when
    result = cosine_similarity(a=a, b=b)

    # then
    assert abs(result - np.sqrt(2) / 2) < 1e-5


def test_cosine_similarity_against_bunch_of_vectors() -> None:
    # given
    a = np.array([[1, 0], [1, 0], [5, 5]])
    b = np.array([[0, 1], [-2, 0], [1, 0]])

    # when
    with pytest.raises(ValueError):
        # does not work against multiple vectors at once
        _ = cosine_similarity(a=a, b=b)


def test_crop_mask() -> None:
    # given
    masks = np.ones((2, 128, 128))
    boxes = np.array([[32, 32, 64, 64], [64, 64, 96, 96]])
    expected_result = np.zeros((2, 128, 128))
    expected_result[0, 32:64, 32:64] = 1
    expected_result[1, 64:96, 64:96] = 1

    # when
    result = crop_mask(masks=masks, boxes=boxes)

    # then
    assert result.shape == expected_result.shape
    assert np.allclose(result, expected_result)


def test_standardise_static_crop() -> None:
    # when
    result = standardise_static_crop(
        static_crop_config={"x_min": 15, "y_min": 20, "x_max": 50, "y_max": 75}
    )

    # then
    assert np.allclose(result, [0.15, 0.2, 0.5, 0.75])


def test_get_static_crop_dimensions_when_misconfiguration_happens() -> None:
    # when
    with pytest.raises(PostProcessingError):
        _ = get_static_crop_dimensions(
            orig_shape=(256, 256),
            preproc={"static-crop": {}},
        )


def test_get_static_crop_dimensions_when_no_crop_should_be_applied() -> None:
    # when
    result = get_static_crop_dimensions(
        orig_shape=(256, 256),
        preproc={},
    )

    # then
    assert result == ((0, 0), (256, 256))


def test_get_static_crop_dimensions_when_crop_should_be_applied() -> None:
    # when
    result = get_static_crop_dimensions(
        orig_shape=(100, 200),
        preproc={
            "static-crop": {
                "enabled": True,
                "x_min": 15,
                "y_min": 20,
                "x_max": 50,
                "y_max": 75,
            }
        },
    )

    # then
    assert result == ((15, 40), (35, 110))
