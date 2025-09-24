from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.core.exceptions import PreProcessingError
from inference.core.utils import preprocess
from inference.core.utils.preprocess import (
    ContrastAdjustmentType,
    apply_contrast_adjustment,
    contrast_adjustments_should_be_applied,
    grayscale_conversion_should_be_applied,
    prepare,
    static_crop_should_be_applied,
    take_static_crop,
)


@mock.patch.object(preprocess, "DISABLE_PREPROC_STATIC_CROP", False)
def test_static_crop_should_be_applied_when_config_does_not_specify_this_preprocessing() -> (
    None
):
    # when
    result = static_crop_should_be_applied(
        preprocessing_config={},
        disable_preproc_static_crop=False,
    )

    # then
    assert result is False


@mock.patch.object(preprocess, "DISABLE_PREPROC_STATIC_CROP", False)
def test_static_crop_should_be_applied_when_config_disables_this_preprocessing() -> (
    None
):
    # when
    result = static_crop_should_be_applied(
        preprocessing_config={"static-crop": {"enabled": False}},
        disable_preproc_static_crop=False,
    )

    # then
    assert result is False


@mock.patch.object(preprocess, "DISABLE_PREPROC_STATIC_CROP", True)
def test_static_crop_should_be_applied_when_env_disables_this_preprocessing() -> None:
    # when
    result = static_crop_should_be_applied(
        preprocessing_config={"static-crop": {"enabled": True}},
        disable_preproc_static_crop=False,
    )

    # then
    assert result is False


@mock.patch.object(preprocess, "DISABLE_PREPROC_STATIC_CROP", False)
def test_static_crop_should_be_applied_when_function_parameter_this_preprocessing() -> (
    None
):
    # when
    result = static_crop_should_be_applied(
        preprocessing_config={"static-crop": {"enabled": True}},
        disable_preproc_static_crop=True,
    )

    # then
    assert result is False


@mock.patch.object(preprocess, "DISABLE_PREPROC_STATIC_CROP", False)
def test_static_crop_should_be_applied_when_preprocessing_should_be_applied() -> None:
    # when
    result = static_crop_should_be_applied(
        preprocessing_config={"static-crop": {"enabled": True}},
        disable_preproc_static_crop=False,
    )

    # then
    assert result is True


@mock.patch.object(preprocess, "DISABLE_PREPROC_CONTRAST", False)
def test_contrast_adjustments_should_be_applied_when_config_does_not_specify_this_preprocessing() -> (
    None
):
    # when
    result = contrast_adjustments_should_be_applied(
        preprocessing_config={},
        disable_preproc_contrast=False,
    )

    # then
    assert result is False


@mock.patch.object(preprocess, "DISABLE_PREPROC_CONTRAST", False)
def test_contrast_adjustments_should_be_applied_when_config_disables_this_preprocessing() -> (
    None
):
    # when
    result = contrast_adjustments_should_be_applied(
        preprocessing_config={"contrast": {"enabled": False}},
        disable_preproc_contrast=False,
    )

    # then
    assert result is False


@mock.patch.object(preprocess, "DISABLE_PREPROC_CONTRAST", True)
def test_contrast_adjustments_should_be_applied_when_env_disables_this_preprocessing() -> (
    None
):
    # when
    result = contrast_adjustments_should_be_applied(
        preprocessing_config={"contrast": {"enabled": True}},
        disable_preproc_contrast=False,
    )

    # then
    assert result is False


@mock.patch.object(preprocess, "DISABLE_PREPROC_CONTRAST", False)
def test_contrast_adjustments_should_be_applied_when_function_parameter_this_preprocessing() -> (
    None
):
    # when
    result = contrast_adjustments_should_be_applied(
        preprocessing_config={"contrast": {"enabled": True}},
        disable_preproc_contrast=True,
    )

    # then
    assert result is False


@mock.patch.object(preprocess, "DISABLE_PREPROC_CONTRAST", False)
def test_contrast_adjustments_should_be_applied_when_preprocessing_should_be_applied() -> (
    None
):
    # when
    result = contrast_adjustments_should_be_applied(
        preprocessing_config={"contrast": {"enabled": True}},
        disable_preproc_contrast=False,
    )

    # then
    assert result is True


@mock.patch.object(preprocess, "DISABLE_PREPROC_GRAYSCALE", False)
def test_grayscale_conversion_should_be_applied_when_config_does_not_specify_this_preprocessing() -> (
    None
):
    # when
    result = grayscale_conversion_should_be_applied(
        preprocessing_config={},
        disable_preproc_grayscale=False,
    )

    # then
    assert result is False


@mock.patch.object(preprocess, "DISABLE_PREPROC_GRAYSCALE", False)
def test_grayscale_conversion_should_be_applied_when_config_disables_this_preprocessing() -> (
    None
):
    # when
    result = grayscale_conversion_should_be_applied(
        preprocessing_config={"grayscale": {"enabled": False}},
        disable_preproc_grayscale=False,
    )

    # then
    assert result is False


@mock.patch.object(preprocess, "DISABLE_PREPROC_GRAYSCALE", True)
def test_grayscale_conversion_should_be_applied_when_env_disables_this_preprocessing() -> (
    None
):
    # when
    result = grayscale_conversion_should_be_applied(
        preprocessing_config={"grayscale": {"enabled": True}},
        disable_preproc_grayscale=False,
    )

    # then
    assert result is False


@mock.patch.object(preprocess, "DISABLE_PREPROC_GRAYSCALE", False)
def test_grayscale_conversion_should_be_applied_when_function_parameter_this_preprocessing() -> (
    None
):
    # when
    result = grayscale_conversion_should_be_applied(
        preprocessing_config={"grayscale": {"enabled": True}},
        disable_preproc_grayscale=True,
    )

    # then
    assert result is False


@mock.patch.object(preprocess, "DISABLE_PREPROC_GRAYSCALE", False)
def test_grayscale_conversion_should_be_applied_when_preprocessing_should_be_applied() -> (
    None
):
    # when
    result = grayscale_conversion_should_be_applied(
        preprocessing_config={"grayscale": {"enabled": True}},
        disable_preproc_grayscale=False,
    )

    # then
    assert result is True


def test_take_static_crop_when_config_is_complete() -> None:
    # given
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[32:64, 32:96, :] = 255
    expected_result = np.ones((32, 64, 3), dtype=np.uint8) * 255

    # when
    result = take_static_crop(
        image=image,
        crop_parameters={"x_min": 16, "x_max": 48, "y_min": 32, "y_max": 64},
    )

    # then
    assert result.shape == expected_result.shape
    assert np.allclose(result, expected_result)


def test_take_static_crop_when_config_is_not_complete() -> None:
    # given
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    # when
    with pytest.raises(KeyError):
        _ = take_static_crop(
            image=image, crop_parameters={"x_min": 16, "x_max": 48, "y_max": 64}
        )


@mock.patch.dict(
    preprocess.CONTRAST_ADJUSTMENTS_METHODS,
    {
        ContrastAdjustmentType.CONTRAST_STRETCHING: lambda i: "A",
        ContrastAdjustmentType.HISTOGRAM_EQUALISATION: lambda i: "B",
        ContrastAdjustmentType.ADAPTIVE_EQUALISATION: lambda i: "C",
    },
    clear=True,
)
@pytest.mark.parametrize(
    "adjustment_type, expected_outcome",
    [
        (ContrastAdjustmentType.CONTRAST_STRETCHING, "A"),
        (ContrastAdjustmentType.HISTOGRAM_EQUALISATION, "B"),
        (ContrastAdjustmentType.ADAPTIVE_EQUALISATION, "C"),
    ],
)
def test_apply_contrast_adjustment(
    adjustment_type: ContrastAdjustmentType,
    expected_outcome: str,
) -> None:
    # given
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    # when
    result = apply_contrast_adjustment(image=image, adjustment_type=adjustment_type)

    # then
    assert result == expected_outcome


@mock.patch.object(preprocess, "DISABLE_PREPROC_STATIC_CROP", False)
@mock.patch.object(preprocess, "take_static_crop")
def test_prepare_when_misconfiguration_error_is_encountered(
    take_static_crop_mock: MagicMock,
) -> None:
    # given
    take_static_crop_mock.side_effect = KeyError()

    # when
    with pytest.raises(PreProcessingError):
        _ = prepare(
            image=np.zeros((128, 128, 3), dtype=np.uint8),
            preproc={"static-crop": {"enabled": True}},
        )
