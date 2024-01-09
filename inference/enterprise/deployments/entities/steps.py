from abc import ABCMeta
from enum import Enum
from typing import Any, List, Literal, Optional, Set, Union

from pydantic import BaseModel, Field, validator

from inference.enterprise.deployments.errors import (
    InvalidStepInputDetected,
    VariableTypeError,
)


class RoboflowModel(BaseModel, metaclass=ABCMeta):
    type: Literal["RoboflowModel"]
    name: str
    image: Union[str, List[str]]
    model_id: str
    disable_active_learning: Union[Optional[bool], str] = Field(default=False)

    @validator("image")
    @classmethod
    def image_must_only_hold_selectors(cls, value: Any) -> None:
        if issubclass(type(value), list):
            if any(not is_selector(selector_or_value=e) for e in value):
                raise ValueError("`image` field can only contain selector values")
        if not is_selector(selector_or_value=value):
            raise ValueError("`image` field can only contain selector values")

    @validator("model_id")
    @classmethod
    def model_id_must_be_selector_or_str(cls, value: Any) -> None:
        if is_selector(selector_or_value=value):
            return None
        if not issubclass(type(value), str):
            raise ValueError("`model_id` field must be string")
        if len(value.split("/")) != 2:
            raise ValueError(
                "`model_id` field must be a valid Roboflow model identifier"
            )

    @validator("disable_active_learning")
    @classmethod
    def disable_active_learning_must_be_selector_or_bool(cls, value: Any) -> None:
        if is_selector(selector_or_value=value):
            return None
        if not issubclass(type(value), bool):
            raise ValueError("`disable_active_learning` field must be bool")

    def get_input_names(self) -> Set[str]:
        return {"image", "model_id", "disable_active_learning"}

    def get_output_names(self) -> Set[str]:
        return set()

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        if field_name == "model_id":
            if not issubclass(type(value), str):
                raise VariableTypeError("Parameter `model_id` must be string")
            chunks = value.split("/")
            if len(chunks) != 2:
                raise VariableTypeError(
                    "Parameter `model_id` must be a valid model id, example: some/3"
                )
        if field_name == "disable_active_learning":
            if not issubclass(type(value), bool):
                raise VariableTypeError(
                    "Parameter `disable_active_learning` must be bool"
                )


class ClassificationModel(RoboflowModel):
    type: Literal["ClassificationModel"]
    confidence: Union[Optional[float], str] = Field(default=0.0)

    @validator("confidence")
    @classmethod
    def confidence_must_be_selector_or_number(cls, value: Any) -> None:
        if is_selector(selector_or_value=value):
            return None
        if not issubclass(type(value), float) or not issubclass(type(value), int):
            raise ValueError("`confidence` field must be number")
        if not 0 <= value <= 1:
            raise ValueError("Parameter `confidence` must be in range [0.0, 1.0]")

    def get_input_names(self) -> Set[str]:
        inputs = super().get_input_names()
        inputs.add("confidence")
        return inputs

    def get_output_names(self) -> Set[str]:
        outputs = super().get_output_names()
        outputs.update(["predictions", "top", "confidence"])
        return outputs

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        super().validate_field_binding(field_name=field_name, value=value)
        if field_name == "confidence":
            if not issubclass(type(value), float) or not issubclass(type(value), int):
                raise VariableTypeError("Parameter `confidence` must be a number")
            if not 0 <= value <= 1:
                raise VariableTypeError(
                    "Parameter `confidence` must be in range [0.0, 1.0]"
                )


class MultiLabelClassificationModel(RoboflowModel):
    type: Literal["MultiLabelClassificationModel"]
    confidence: Union[Optional[float], str] = Field(default=0.0)

    @validator("confidence")
    @classmethod
    def confidence_must_be_selector_or_number(cls, value: Any) -> None:
        if is_selector(selector_or_value=value):
            return None
        if not issubclass(type(value), float) or not issubclass(type(value), int):
            raise ValueError("`confidence` field must be number")
        if not 0 <= value <= 1:
            raise ValueError("Parameter `confidence` must be in range [0.0, 1.0]")

    def get_input_names(self) -> Set[str]:
        inputs = super().get_input_names()
        inputs.add("confidence")
        return inputs

    def get_output_names(self) -> Set[str]:
        outputs = super().get_output_names()
        outputs.update(["predictions", "predicted_classes"])
        return outputs

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        super().validate_field_binding(field_name=field_name, value=value)
        if field_name == "confidence":
            if not issubclass(type(value), float) or not issubclass(type(value), int):
                raise VariableTypeError("Parameter `confidence` must be a number")
            if not 0 <= value <= 1:
                raise VariableTypeError(
                    "Parameter `confidence` must be in range [0.0, 1.0]"
                )


class ObjectDetectionModel(RoboflowModel):
    type: Literal["ObjectDetectionModel"]
    class_agnostic_nms: Union[Optional[bool], str] = Field(default=False)
    class_filter: Union[Optional[List[str]], str] = Field(default=None)
    confidence: Union[Optional[float], str] = Field(default=0.0)
    iou_threshold: Union[Optional[float], str] = Field(default=1.0)
    max_detections: Union[Optional[int], str] = Field(default=300)
    max_candidates: Union[Optional[int], str] = Field(default=3000)

    def get_input_names(self) -> Set[str]:
        inputs = super().get_input_names()
        inputs.update(
            [
                "class_agnostic_nms",
                "class_filter",
                "confidence",
                "iou_threshold",
                "max_detections",
                "max_candidates",
            ]
        )
        return inputs

    def get_output_names(self) -> Set[str]:
        outputs = super().get_output_names()
        outputs.add("predictions")
        return outputs

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        super().validate_field_binding(field_name=field_name, value=value)
        if field_name == "class_agnostic_nms":
            if not issubclass(type(value), bool):
                raise VariableTypeError("Parameter `class_agnostic_nms` must be a bool")
        if field_name == "class_filter":
            if not issubclass(type(value), list):
                raise VariableTypeError("Parameter `class_filter` must be a list")
            if any(not issubclass(type(e), str) for e in value):
                raise VariableTypeError(
                    "Parameter `class_filter` must be a list of string"
                )
        if field_name == "confidence" or field_name == "iou_threshold":
            if not issubclass(type(value), float) or not issubclass(type(value), int):
                raise VariableTypeError(f"Parameter `{field_name}` must be a number")
            if not 0 <= value <= 1:
                raise VariableTypeError(
                    f"Parameter `{field_name}` must be in range [0.0, 1.0]"
                )
        if field_name == "max_detections" or field_name == "max_candidates":
            if not issubclass(type(value), int):
                raise VariableTypeError(f"Parameter `{field_name}` must be a integer")
            if value <= 0:
                raise VariableTypeError(
                    f"Parameter `{field_name}` must be greater than zero"
                )


class KeypointsDetectionModel(ObjectDetectionModel):
    type: Literal["KeypointsDetectionModel"]
    keypoint_confidence: Union[Optional[float], str] = Field(default=0.0)

    def get_input_names(self) -> Set[str]:
        inputs = super().get_input_names()
        inputs.add("keypoint_confidence")
        return inputs

    def get_output_names(self) -> Set[str]:
        outputs = super().get_output_names()
        outputs.add("predictions")
        return outputs

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        super().validate_field_binding(field_name=field_name, value=value)
        if field_name == "keypoint_confidence":
            if not issubclass(type(value), float) or not issubclass(type(value), int):
                raise VariableTypeError(f"Parameter `{field_name}` must be a number")
            if not 0 <= value <= 1:
                raise VariableTypeError(
                    f"Parameter `{field_name}` must be in range [0.0, 1.0]"
                )


class InstanceSegmentationModel(ObjectDetectionModel):
    type: Literal["InstanceSegmentationModel"]
    mask_decode_mode: Optional[str] = Field(default="accurate")
    tradeoff_factor: Union[Optional[float], str] = Field(default=0.0)

    def get_input_names(self) -> Set[str]:
        inputs = super().get_input_names()
        inputs.update(["mask_decode_mode", "tradeoff_factor"])
        return inputs

    def get_output_names(self) -> Set[str]:
        outputs = super().get_output_names()
        outputs.add("predictions")
        return outputs

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        super().validate_field_binding(field_name=field_name, value=value)
        if field_name == "mask_decode_mode":
            if not issubclass(type(value), str):
                raise VariableTypeError(f"Parameter `{field_name}` must be a string")
            if value not in {"accurate", "tradeoff", "fast"}:
                raise VariableTypeError(
                    f"Parameter `{field_name}` must be in 'accurate', 'tradeoff', 'fast'"
                )
        if field_name == "tradeoff_factor":
            if not issubclass(type(value), float) or not issubclass(type(value), int):
                raise VariableTypeError(f"Parameter `{field_name}` must be a number")
            if not 0 <= value <= 1:
                raise VariableTypeError(
                    f"Parameter `{field_name}` must be in range [0.0, 1.0]"
                )


class OCRModel(BaseModel):
    type: Literal["OCRModel"]
    name: str
    image: Union[str, List[str]]

    def get_input_names(self) -> Set[str]:
        return {"image"}

    def get_output_names(self) -> Set[str]:
        return {"result"}


class Crop(BaseModel):
    type: Literal["Crop"]
    name: str
    image: Union[str, List[str]]
    detections: str

    def get_input_names(self) -> Set[str]:
        return {"image", "detections"}

    def get_output_names(self) -> Set[str]:
        return {"crops"}

    def validate_detections(self, input_step_type: str) -> None:
        if input_step_type not in {
            "ObjectDetectionModel",
            "KeypointsDetectionModel",
            "InstanceSegmentationModel",
        }:
            raise InvalidStepInputDetected(
                f"Crop step with name {self.name} cannot take as an input predictions from {input_step_type}. "
                f"Cropping requires detection-based output."
            )


class Operator(Enum):
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    LOWER_THAN = "lower_than"
    GREATER_THAN = "greater_than"
    LOWER_OR_EQUAL_THAN = "lower_or_equal_than"
    GREATER_OR_EQUAL_THAN = "greater_or_equal_than"
    IN = "in"


class Condition(BaseModel):
    type: Literal["Condition"]
    name: str
    left: Union[int, float, bool, str, list, set]
    operator: Operator
    right: Union[int, float, bool, str, list, set]
    step_if_true: str
    step_if_false: str

    def get_input_names(self) -> Set[str]:
        return {"left", "right"}

    def get_output_names(self) -> Set[str]:
        return set()


def is_selector(selector_or_value: Any) -> bool:
    if not issubclass(type(selector_or_value), str):
        return False
    return selector_or_value.startswith("$")
