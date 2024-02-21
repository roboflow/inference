from typing import Any, List, Optional, Set, Type

from pydantic import ValidationError

from inference.core.entities.requests.inference import InferenceRequestImage
from inference.enterprise.workflows.entities.base import GraphNone
from inference.enterprise.workflows.errors import (
    InvalidStepInputDetected,
    VariableTypeError,
)

STEPS_WITH_IMAGE = {
    "InferenceImage",
    "Crop",
    "AbsoluteStaticCrop",
    "RelativeStaticCrop",
}


def validate_image_is_valid_selector(value: Any, field_name: str = "image") -> None:
    if issubclass(type(value), list):
        if any(not is_selector(selector_or_value=e) for e in value):
            raise ValueError(f"`{field_name}` field can only contain selector values")
    elif not is_selector(selector_or_value=value):
        raise ValueError(f"`{field_name}` field can only contain selector values")


def validate_field_is_in_range_zero_one_or_empty_or_selector(
    value: Any, field_name: str = "confidence"
) -> None:
    if is_selector(selector_or_value=value) or value is None:
        return None
    validate_value_is_empty_or_number_in_range_zero_one(
        value=value, field_name=field_name
    )


def validate_value_is_empty_or_number_in_range_zero_one(
    value: Any, field_name: str = "confidence", error: Type[Exception] = ValueError
) -> None:
    validate_field_has_given_type(
        field_name=field_name,
        allowed_types=[type(None), int, float],
        value=value,
        error=error,
    )
    if value is None:
        return None
    if not (0 <= value <= 1):
        raise error(f"Parameter `{field_name}` must be in range [0.0, 1.0]")


def validate_value_is_empty_or_selector_or_positive_number(
    value: Any, field_name: str
) -> None:
    if is_selector(selector_or_value=value):
        return None
    validate_value_is_empty_or_positive_number(value=value, field_name=field_name)


def validate_value_is_empty_or_positive_number(
    value: Any, field_name: str, error: Type[Exception] = ValueError
) -> None:
    validate_field_has_given_type(
        field_name=field_name,
        allowed_types=[type(None), int, float],
        value=value,
        error=error,
    )
    if value is None:
        return None
    if value <= 0:
        raise error(f"Parameter `{field_name}` must be positive (> 0)")


def validate_field_is_list_of_selectors(
    value: Any, field_name: str, error: Type[Exception] = ValueError
) -> None:
    if not issubclass(type(value), list):
        raise error(f"`{field_name}` field must be list")
    if any(not is_selector(selector_or_value=e) for e in value):
        raise error(f"Parameter `{field_name}` must be a list of selectors")


def validate_field_is_empty_or_selector_or_list_of_string(
    value: Any, field_name: str
) -> None:
    if is_selector(selector_or_value=value) or value is None:
        return value
    validate_field_is_list_of_string(value=value, field_name=field_name)


def validate_field_is_list_of_string(
    value: Any, field_name: str, error: Type[Exception] = ValueError
) -> None:
    if not issubclass(type(value), list):
        raise error(f"`{field_name}` field must be list")
    if any(not issubclass(type(e), str) for e in value):
        raise error(f"Parameter `{field_name}` must be a list of string")


def validate_field_is_selector_or_one_of_values(
    value: Any, field_name: str, selected_values: set
) -> None:
    if is_selector(selector_or_value=value) or value is None:
        return value
    validate_field_is_one_of_selected_values(
        value=value, field_name=field_name, selected_values=selected_values
    )


def validate_field_is_one_of_selected_values(
    value: Any,
    field_name: str,
    selected_values: set,
    error: Type[Exception] = ValueError,
) -> None:
    try:
        if value not in selected_values:
            raise error(
                f"Value of field `{field_name}` must be in {selected_values}. Found: {value}"
            )
    except TypeError as check_error:
        raise error(
            f"Value of field `{field_name}` must be in {selected_values}. Found: {value} which is not comparable"
        ) from check_error


def validate_field_is_selector_or_has_given_type(
    value: Any, field_name: str, allowed_types: List[type]
) -> None:
    if is_selector(selector_or_value=value):
        return None
    validate_field_has_given_type(
        field_name=field_name, allowed_types=allowed_types, value=value
    )
    return None


def validate_field_has_given_type(
    value: Any,
    field_name: str,
    allowed_types: List[type],
    error: Type[Exception] = ValueError,
) -> None:
    if all(not issubclass(type(value), allowed_type) for allowed_type in allowed_types):
        raise error(
            f"`{field_name}` field type must be one of {allowed_types}. Detected: {value}"
        )


def validate_field_is_dict_of_strings(
    value: Any,
    field_name: str,
    error: Type[Exception] = ValueError,
) -> None:
    if not issubclass(type(value), dict):
        raise error(f"`{field_name}` field is expected to be dict.")
    for key, key_value in value.items():
        if not issubclass(type(key), str):
            raise error(
                f"`{field_name}` field holds dict which has key={key} that is not string."
            )
        if not issubclass(type(key_value), str):
            raise error(
                f"`{field_name}` field holds dict which has key={key} that holds non-string value: {key_value}."
            )


def validate_image_biding(value: Any, field_name: str = "image") -> None:
    try:
        if not issubclass(type(value), list):
            value = [value]
        if len(value) == 0:
            raise VariableTypeError(f"Parameter `{field_name}` must not be empty.")
        for e in value:
            InferenceRequestImage.model_validate(e)
    except (ValueError, ValidationError) as error:
        raise VariableTypeError(
            f"Parameter `{field_name}` must be compatible with `InferenceRequestImage`"
        ) from error


def validate_selector_is_inference_parameter(
    step_type: str,
    field_name: str,
    input_step: GraphNone,
    applicable_fields: Set[str],
) -> None:
    if field_name not in applicable_fields:
        return None
    input_step_type = input_step.get_type()
    if input_step_type not in {"InferenceParameter"}:
        raise InvalidStepInputDetected(
            f"Field {field_name} of step {step_type} comes from invalid input type: {input_step_type}. "
            f"Expected: `InferenceParameter`"
        )


def validate_selector_holds_image(
    step_type: str,
    field_name: str,
    input_step: GraphNone,
    applicable_fields: Optional[Set[str]] = None,
) -> None:
    if applicable_fields is None:
        applicable_fields = {"image"}
    if field_name not in applicable_fields:
        return None
    if input_step.get_type() not in STEPS_WITH_IMAGE:
        raise InvalidStepInputDetected(
            f"Field {field_name} of step {step_type} comes from invalid input type: {input_step.get_type()}. "
            f"Expected: {STEPS_WITH_IMAGE}"
        )


def validate_selector_holds_detections(
    step_name: str,
    image_selector: Optional[str],
    detections_selector: str,
    field_name: str,
    input_step: GraphNone,
    applicable_fields: Optional[Set[str]] = None,
) -> None:
    if applicable_fields is None:
        applicable_fields = {"detections"}
    if field_name not in applicable_fields:
        return None
    if input_step.get_type() not in {
        "ObjectDetectionModel",
        "KeypointsDetectionModel",
        "InstanceSegmentationModel",
        "DetectionFilter",
        "DetectionsConsensus",
        "DetectionOffset",
        "YoloWorld",
    }:
        raise InvalidStepInputDetected(
            f"Step step with name {step_name} cannot take as an input predictions from {input_step.get_type()}. "
            f"Step requires detection-based output."
        )
    if get_last_selector_chunk(detections_selector) != "predictions":
        raise InvalidStepInputDetected(
            f"Step with name {step_name} must take as input step output of name `predictions`"
        )
    if not hasattr(input_step, "image") or image_selector is None:
        # Here, filter do not hold the reference to image, we skip the check in this case
        return None
    input_step_image_reference = input_step.image
    if image_selector != input_step_image_reference:
        raise InvalidStepInputDetected(
            f"Step step with name {step_name} was given detections reference that is bound to different image: "
            f"step.image: {image_selector}, detections step image: {input_step_image_reference}"
        )


def is_selector(selector_or_value: Any) -> bool:
    if not issubclass(type(selector_or_value), str):
        return False
    return selector_or_value.startswith("$")


def get_last_selector_chunk(selector: str) -> str:
    return selector.split(".")[-1]
