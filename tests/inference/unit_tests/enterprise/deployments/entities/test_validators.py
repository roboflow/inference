from typing import Any

import numpy as np
import pytest

from inference.enterprise.deployments.entities.inputs import (
    InferenceImage,
    InferenceParameter,
)
from inference.enterprise.deployments.entities.steps import Crop, ObjectDetectionModel
from inference.enterprise.deployments.entities.validators import (
    get_last_selector_chunk,
    is_selector,
    validate_field_has_given_type,
    validate_field_is_empty_or_selector_or_list_of_string,
    validate_field_is_in_range_zero_one_or_empty_or_selector,
    validate_field_is_list_of_string,
    validate_field_is_one_of_selected_values,
    validate_field_is_selector_or_has_given_type,
    validate_field_is_selector_or_one_of_values,
    validate_image_biding,
    validate_image_is_valid_selector,
    validate_selector_holds_detections,
    validate_selector_holds_image,
    validate_selector_is_inference_parameter,
    validate_value_is_empty_or_number_in_range_zero_one,
    validate_value_is_empty_or_positive_number,
    validate_value_is_empty_or_selector_or_positive_number,
)
from inference.enterprise.deployments.errors import (
    InvalidStepInputDetected,
    VariableTypeError,
)


def test_get_last_selector_chunk_against_step_output() -> None:
    # when
    result = get_last_selector_chunk(selector="$steps.my_step.predictions")

    # then
    assert (
        result == "predictions"
    ), "Last chunk of selector is expected to be `predictions`"


def test_get_last_selector_chunk_against_input_name() -> None:
    # when
    result = get_last_selector_chunk(selector="$inputs.image")

    # then
    assert result == "image", "Last chunk of selector is expected to be `image`"


@pytest.mark.parametrize("value", [1, 1.3, None, [], {}, set()])
def test_is_selector_when_non_string_input_provided(value: Any) -> None:
    # when
    result = is_selector(selector_or_value=value)

    # then
    assert result is False, "Selector must be string"


def test_is_selector_when_string_input_provided_but_not_in_selector_scheme() -> None:
    # when
    result = is_selector(selector_or_value="some")

    # then
    assert result is False, "Selectors are assumed to start from $"


def test_is_selector_when_selector_provided() -> None:
    # when
    result = is_selector(selector_or_value="$inputs.my_image")

    # then
    assert result is True, "Valid selector expected to be detected"


def test_validate_selector_holds_detections_when_not_applicable_field_tested() -> None:
    # when
    validate_selector_holds_detections(
        step_name="some",
        image_selector="$inputs.image",
        detections_selector="$steps.detect.predictions",
        field_name="some_field",
        input_step=Crop(
            type="Crop",
            name="my_crop",
            image="$inputs.image",
            detections="$steps.detect_2.predictions",
        ),
    )

    # no error raised, as validated field is not `detections`


def test_validate_selector_holds_detections_when_detections_field_contains_invalid_step() -> (
    None
):
    # when
    with pytest.raises(InvalidStepInputDetected):
        validate_selector_holds_detections(
            step_name="some",
            image_selector="$inputs.image",
            detections_selector="$steps.detect.predictions",
            field_name="detections",
            input_step=Crop(
                type="Crop",
                name="my_crop",
                image="$inputs.image",
                detections="$steps.detect_2.predictions",
            ),
        )


def test_validate_selector_holds_detections_when_detections_selector_does_not_point_predictions_property() -> (
    None
):
    # when
    with pytest.raises(InvalidStepInputDetected):
        validate_selector_holds_detections(
            step_name="some",
            image_selector="$inputs.image",
            detections_selector="$steps.detect.invalid",
            field_name="detections",
            input_step=ObjectDetectionModel(
                type="ObjectDetectionModel",
                name="my_crop",
                image="$inputs.image",
                model_id="some/1",
            ),
        )


def test_validate_selector_holds_detections_when_image_selector_given_and_does_not_match_input_image() -> (
    None
):
    # when
    with pytest.raises(InvalidStepInputDetected):
        validate_selector_holds_detections(
            step_name="some",
            image_selector="$inputs.image",
            detections_selector="$steps.detect.predictions",
            field_name="detections",
            input_step=ObjectDetectionModel(
                type="ObjectDetectionModel",
                name="my_crop",
                image="$inputs.other_image",
                model_id="some/1",
            ),
        )


def test_validate_selector_holds_detections_when_validation_passes_without_verification_of_image() -> (
    None
):
    # when
    validate_selector_holds_detections(
        step_name="some",
        image_selector=None,
        detections_selector="$steps.detect.predictions",
        field_name="detections",
        input_step=ObjectDetectionModel(
            type="ObjectDetectionModel",
            name="my_crop",
            image="$inputs.image",
            model_id="some/1",
        ),
    )


def test_validate_selector_holds_detections_when_validation_passes_with_verification_of_image() -> (
    None
):
    # when
    validate_selector_holds_detections(
        step_name="some",
        image_selector="$inputs.image",
        detections_selector="$steps.detect.predictions",
        field_name="detections",
        input_step=ObjectDetectionModel(
            type="ObjectDetectionModel",
            name="my_crop",
            image="$inputs.image",
            model_id="some/1",
        ),
    )


def test_validate_selector_holds_detections_when_validation_passes_with_change_of_applicable_fields() -> (
    None
):
    # when
    validate_selector_holds_detections(
        step_name="some",
        image_selector="$inputs.image",
        detections_selector="$steps.detect.predictions",
        field_name="predictions",
        input_step=ObjectDetectionModel(
            type="ObjectDetectionModel",
            name="my_crop",
            image="$inputs.image",
            model_id="some/1",
        ),
        applicable_fields={"predictions"},
    )


def test_validate_selector_holds_image_when_validating_not_applicable_field() -> None:
    # when
    validate_selector_holds_image(
        step_type="Crop",
        field_name="some",
        input_step=InferenceImage(type="InferenceImage", name="image"),
    )


def test_validate_selector_holds_image_when_valid_input_step_provided() -> None:
    # when
    validate_selector_holds_image(
        step_type="Crop",
        field_name="image",
        input_step=InferenceImage(type="InferenceImage", name="image"),
    )


def test_validate_selector_holds_image_when_invalid_input_step_provided_and_applicable_field_pointed() -> (
    None
):
    # when
    with pytest.raises(InvalidStepInputDetected):
        validate_selector_holds_image(
            step_type="Crop",
            field_name="image",
            input_step=InferenceParameter(type="InferenceParameter", name="parameter"),
        )


def test_validate_selector_holds_image_when_invalid_input_step_provided() -> None:
    # when
    with pytest.raises(InvalidStepInputDetected):
        validate_selector_holds_image(
            step_type="Crop",
            field_name="some",
            input_step=InferenceParameter(type="InferenceParameter", name="image"),
            applicable_fields={"some"},
        )


def test_validate_selector_is_inference_parameter_when_validating_not_relevant_field() -> (
    None
):
    validate_selector_is_inference_parameter(
        step_type="Crop",
        field_name="some",
        input_step=InferenceImage(type="InferenceImage", name="image"),
        applicable_fields={"my_parameter"},
    )


def test_validate_selector_is_inference_parameter_when_input_step_is_valid() -> None:
    validate_selector_is_inference_parameter(
        step_type="Crop",
        field_name="my_parameter",
        input_step=InferenceParameter(type="InferenceParameter", name="parameter"),
        applicable_fields={"my_parameter"},
    )


def test_validate_selector_is_inference_parameter_when_input_step_is_invalid() -> None:
    # when
    with pytest.raises(InvalidStepInputDetected):
        validate_selector_is_inference_parameter(
            step_type="Crop",
            field_name="my_parameter",
            input_step=InferenceImage(type="InferenceImage", name="image"),
            applicable_fields={"my_parameter"},
        )


def test_validate_image_biding_when_single_valid_image_provided() -> None:
    # when
    validate_image_biding(value={"type": "url", "value": "https://my.com/image.jpg"})


def test_validate_image_biding_when_multiple_valid_images_provided() -> None:
    # when
    validate_image_biding(
        value=[
            {"type": "url", "value": "https://my.com/image.jpg"},
            {"type": "url", "value": "https://my.com/image.jpg"},
        ]
    )


def test_validate_image_biding_single_invalid_image_provided() -> None:
    # when
    with pytest.raises(VariableTypeError):
        validate_image_biding(value=np.zeros((192, 168, 3)))


def test_validate_image_biding_one_of_provided_images_is_invalid() -> None:
    # when
    with pytest.raises(VariableTypeError):
        validate_image_biding(
            value=[
                {"type": "url", "value": "https://my.com/image.jpg"},
                {"type": "url", "value": "https://my.com/image.jpg"},
                {"value": "https://my.com/image.jpg"},
            ]
        )


def test_validate_field_has_given_type_when_field_matches() -> None:
    # when
    validate_field_has_given_type(
        field_name="some",
        allowed_types=[type(None), str],
        value=None,
    )


def test_validate_field_has_given_type_when_field_does_not_match() -> None:
    with pytest.raises(ValueError):
        validate_field_has_given_type(
            field_name="some",
            allowed_types=[str],
            value=None,
        )


class MyError(Exception):
    pass


def test_validate_field_has_given_type_when_field_does_not_match_and_custom_error_to_be_thrown() -> (
    None
):
    with pytest.raises(MyError):
        validate_field_has_given_type(
            field_name="some",
            allowed_types=[str],
            value=None,
            error=MyError,
        )


def test_validate_field_is_selector_or_has_given_type_when_selector_given() -> None:
    # when
    validate_field_is_selector_or_has_given_type(
        field_name="some",
        allowed_types=[int],
        value="$inputs.image",
    )


def test_validate_field_is_selector_or_has_given_type_when_matching_value_given() -> (
    None
):
    # when
    validate_field_is_selector_or_has_given_type(
        field_name="some",
        allowed_types=[int],
        value=4,
    )


def test_validate_field_is_selector_or_has_given_type_when_validation_fails() -> None:
    # when
    with pytest.raises(ValueError):
        validate_field_is_selector_or_has_given_type(
            field_name="some", allowed_types=[int], value="invalid"
        )


def test_validate_field_is_one_of_selected_values_when_value_matches() -> None:
    # when
    validate_field_is_one_of_selected_values(
        value="some", field_name="my_field", selected_values={"some", "other"}
    )


def test_validate_field_is_one_of_selected_values_when_value_does_not_match() -> None:
    # when
    with pytest.raises(ValueError):
        validate_field_is_one_of_selected_values(
            value="invalid", field_name="my_field", selected_values={"some", "other"}
        )


def test_validate_field_is_one_of_selected_values_when_value_does_not_match_and_error_is_given() -> (
    None
):
    # when
    with pytest.raises(MyError):
        validate_field_is_one_of_selected_values(
            value="invalid",
            field_name="my_field",
            selected_values={"some", "other"},
            error=MyError,
        )


def test_validate_field_is_selector_or_one_of_values_when_input_is_selector() -> None:
    # when
    validate_field_is_selector_or_one_of_values(
        value="$inputs.image",
        field_name="my_field",
        selected_values={"some", "other"},
    )


def test_validate_field_is_selector_or_one_of_values_when_input_is_matching_value() -> (
    None
):
    # when
    validate_field_is_selector_or_one_of_values(
        value="some",
        field_name="my_field",
        selected_values={"some", "other"},
    )


def test_validate_field_is_selector_or_one_of_values_when_input_is_not_matching_value() -> (
    None
):
    # when
    with pytest.raises(ValueError):
        validate_field_is_selector_or_one_of_values(
            value=4,
            field_name="my_field",
            selected_values={"some", "other"},
        )


@pytest.mark.parametrize("value", [{}, 1.0, 1, "some", set()])
def test_validate_field_is_list_of_string_when_not_a_list_provided(value: Any) -> None:
    # when
    with pytest.raises(ValueError):
        validate_field_is_list_of_string(value=value, field_name="some")


def test_validate_field_is_list_of_string_when_provided_list_byt_not_all_elements_are_string() -> (
    None
):
    # when
    with pytest.raises(ValueError):
        validate_field_is_list_of_string(value=["some", "other", 1], field_name="some")


def test_validate_field_is_list_of_string_when_validation_fails_with_custom_error() -> (
    None
):
    # when
    with pytest.raises(MyError):
        validate_field_is_list_of_string(value=1, field_name="some", error=MyError)


def test_validate_field_is_empty_or_selector_or_list_of_string_when_empty_input_given() -> (
    None
):
    # when
    validate_field_is_empty_or_selector_or_list_of_string(value=None, field_name="some")


def test_validate_field_is_empty_or_selector_or_list_of_string_when_selector_given() -> (
    None
):
    # when
    validate_field_is_empty_or_selector_or_list_of_string(
        value="$inputs.some", field_name="some"
    )


def test_validate_field_is_empty_or_selector_or_list_of_string_when_valid_list_given() -> (
    None
):
    # when
    validate_field_is_empty_or_selector_or_list_of_string(
        value=["some", "other"], field_name="some"
    )


def test_validate_field_is_empty_or_selector_or_list_of_string_when_invalid_list_given() -> (
    None
):
    # when
    with pytest.raises(ValueError):
        validate_field_is_empty_or_selector_or_list_of_string(
            value=["some", 3], field_name="some"
        )


@pytest.mark.parametrize("value", ["invalid", [], {}, set()])
def test_validate_value_is_empty_or_positive_number_when_invalid_types_given(
    value: Any,
) -> None:
    # when
    with pytest.raises(ValueError):
        validate_value_is_empty_or_positive_number(
            value=value,
            field_name="some",
        )


@pytest.mark.parametrize("value", [0.1, 1, 2.0])
def test_validate_value_is_empty_or_positive_number_when_positive_number_given(
    value: Any,
) -> None:
    # when
    validate_value_is_empty_or_positive_number(
        value=value,
        field_name="some",
    )


@pytest.mark.parametrize("value", [0.0, -0.1, -1, -2.0])
def test_validate_value_is_empty_or_positive_number_when_negative_number_given(
    value: Any,
) -> None:
    # when
    with pytest.raises(ValueError):
        validate_value_is_empty_or_positive_number(
            value=value,
            field_name="some",
        )


def test_validate_value_is_empty_or_positive_number_when_invalid_value_given_and_custom_error_provided() -> (
    None
):
    # when
    with pytest.raises(MyError):
        validate_value_is_empty_or_positive_number(
            value="invalid",
            field_name="some",
            error=MyError,
        )


def test_validate_value_is_empty_or_positive_number_when_empty_value_provided() -> None:
    # when
    validate_value_is_empty_or_positive_number(
        value=None,
        field_name="some",
        error=MyError,
    )


def test_validate_value_is_empty_or_selector_or_positive_number_when_selector_given() -> (
    None
):
    # when
    validate_value_is_empty_or_selector_or_positive_number(
        value="$inputs.image",
        field_name="some",
    )


def test_validate_value_is_empty_or_selector_or_positive_number_when_correct_value_given() -> (
    None
):
    validate_value_is_empty_or_selector_or_positive_number(
        value=3,
        field_name="some",
    )


def test_validate_value_is_empty_or_selector_or_positive_number_when_invalid__value_given() -> (
    None
):
    with pytest.raises(ValueError):
        validate_value_is_empty_or_selector_or_positive_number(
            value=0,
            field_name="some",
        )


def test_validate_value_is_empty_or_number_in_range_zero_one_when_empty_value_given() -> (
    None
):
    # when
    validate_value_is_empty_or_number_in_range_zero_one(value=None)


def test_validate_value_is_empty_or_number_in_range_zero_one_when_invalid_value_given_and_custom_error_pointed() -> (
    None
):
    # when
    with pytest.raises(MyError):
        validate_value_is_empty_or_number_in_range_zero_one(value="some", error=MyError)


@pytest.mark.parametrize("value", ["a", [], {}, set()])
def test_validate_value_is_empty_or_number_in_range_zero_one_when_value_with_invalid_type_given(
    value: Any,
) -> None:
    # when
    with pytest.raises(ValueError):
        validate_value_is_empty_or_number_in_range_zero_one(value=value)


@pytest.mark.parametrize("value", [-1, -0.1, 1.05, 10])
def test_validate_value_is_empty_or_number_in_range_zero_one_when_invalid_number_given(
    value: Any,
) -> None:
    # when
    with pytest.raises(ValueError):
        validate_value_is_empty_or_number_in_range_zero_one(value=value)


@pytest.mark.parametrize("value", [0, 1, 0.1, 0.5])
def test_validate_value_is_empty_or_number_in_range_zero_one_when_valid_number_given(
    value: Any,
) -> None:
    # when
    validate_value_is_empty_or_number_in_range_zero_one(value=value)


def test_validate_field_is_in_range_zero_one_or_empty_or_selector_when_selector_given() -> (
    None
):
    # when
    validate_field_is_in_range_zero_one_or_empty_or_selector(value="$inputs.confidence")


def test_validate_field_is_in_range_zero_one_or_empty_or_selector_when_correct_value_given() -> (
    None
):
    # when
    validate_field_is_in_range_zero_one_or_empty_or_selector(value=0.3)


def test_validate_field_is_in_range_zero_one_or_empty_or_selector_when_invalid_value_given() -> (
    None
):
    # when
    with pytest.raises(ValueError):
        validate_field_is_in_range_zero_one_or_empty_or_selector(value=1.1)


def test_validate_image_is_valid_selector_when_single_selector_given() -> None:
    # when
    validate_image_is_valid_selector(value="$inputs.image")


def test_validate_image_is_valid_selector_when_multiple_selectors_given() -> None:
    validate_image_is_valid_selector(value=["$inputs.image", "$inputs.image2"])


def test_validate_image_is_valid_selector_when_invalid_value_provided() -> None:
    with pytest.raises(ValueError):
        validate_image_is_valid_selector(value=np.zeros((192, 168, 3)))
