from unittest import mock

from inference.core.utils.preprocess import (
    static_crop_should_be_applied,
    contrast_adjustments_should_be_applied,
    grayscale_conversion_should_be_applied,
)
from inference.core.utils import preprocess


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
