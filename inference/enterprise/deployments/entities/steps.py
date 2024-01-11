from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Annotated, Any, List, Literal, Optional, Set, Union

from pydantic import BaseModel, Field, validator

from inference.core.entities.requests.inference import InferenceRequestImage
from inference.enterprise.deployments.entities.base import GraphNone
from inference.enterprise.deployments.errors import (
    ExecutionGraphError,
    InvalidStepInputDetected,
    VariableTypeError,
)


class StepInterface(GraphNone, metaclass=ABCMeta):
    @abstractmethod
    def get_input_names(self) -> Set[str]:
        """
        Supposed to give the name of all fields expected to represent inputs
        """
        pass

    @abstractmethod
    def get_output_names(self) -> Set[str]:
        """
        Supposed to give the name of all fields expected to represent outputs to be referred by other steps
        """

    @abstractmethod
    def validate_field_selector(self, field_name: str, input_step: GraphNone) -> None:
        """
        Supposed to validate the type of input is referred
        """
        pass

    @abstractmethod
    def validate_field_binding(self, field_name: str, value: Any) -> None:
        """
        Supposed to validate the type of value that is to be bounded with field as a result of graph
        execution (values passed by client to invocation, as well as constructed during graph execution)
        """
        pass


class RoboflowModel(BaseModel, StepInterface, metaclass=ABCMeta):
    type: Literal["RoboflowModel"]
    name: str
    image: Union[str, List[str]]
    model_id: str
    disable_active_learning: Union[Optional[bool], str] = Field(default=False)

    @validator("image")
    @classmethod
    def image_must_only_hold_selectors(cls, value: Any) -> Union[str, List[str]]:
        if issubclass(type(value), list):
            if any(not is_selector(selector_or_value=e) for e in value):
                raise ValueError("`image` field can only contain selector values")
        if not is_selector(selector_or_value=value):
            raise ValueError("`image` field can only contain selector values")
        return value

    @validator("model_id")
    @classmethod
    def model_id_must_be_selector_or_str(cls, value: Any) -> str:
        if is_selector(selector_or_value=value):
            return value
        if not issubclass(type(value), str):
            raise ValueError("`model_id` field must be string")
        return value

    @validator("disable_active_learning")
    @classmethod
    def disable_active_learning_must_be_selector_or_bool(
        cls, value: Any
    ) -> Union[Optional[bool], str]:
        if is_selector(selector_or_value=value) or value is None:
            return value
        if not issubclass(type(value), bool):
            raise ValueError("`disable_active_learning` field must be bool")
        return value

    def get_type(self) -> str:
        return self.type

    def get_input_names(self) -> Set[str]:
        return {"image", "model_id", "disable_active_learning"}

    def get_output_names(self) -> Set[str]:
        return set()

    def validate_field_selector(self, field_name: str, input_step: GraphNone) -> None:
        if not is_selector(selector_or_value=getattr(self, field_name)):
            raise ExecutionGraphError(
                f"Attempted to validate selector value for field {field_name}, but field is not selector."
            )
        if field_name == "image":
            if input_step.get_type() not in {
                "InferenceImage",
                "Crop",
                "AbsoluteStaticCrop",
                "RelativeStaticCrop",
            }:
                raise InvalidStepInputDetected(
                    f"Field {field_name} of step {self.type} comes from invalid input type: {input_step.get_type()}. "
                    f"Expected: `InferenceImage`, `Crop`, `AbsoluteStaticCrop`, `RelativeStaticCrop`"
                )
        if field_name in {"model_id", "disable_active_learning"}:
            if input_step.get_type() not in {"InferenceParameter"}:
                raise InvalidStepInputDetected(
                    f"Field {field_name} of step {self.type} comes from invalid input type: {input_step.get_type()}. "
                    f"Expected: `InferenceParameter`"
                )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        if field_name == "image":
            try:
                if not issubclass(type(value), list):
                    value = [value]
                for e in value:
                    InferenceRequestImage.validate(e)
            except ValueError as error:
                raise VariableTypeError(
                    "Parameter `image` must be compatible with `InferenceRequestImage`"
                ) from error
        if field_name == "model_id":
            if not issubclass(type(value), str):
                raise VariableTypeError("Parameter `model_id` must be string")
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
    def confidence_must_be_selector_or_number(
        cls, value: Any
    ) -> Union[Optional[float], str]:
        if is_selector(selector_or_value=value) or value is None:
            return value
        if not issubclass(type(value), float) and not issubclass(type(value), int):
            raise ValueError("`confidence` field must be number")
        if not 0 <= value <= 1:
            raise ValueError("Parameter `confidence` must be in range [0.0, 1.0]")
        return value

    def get_input_names(self) -> Set[str]:
        inputs = super().get_input_names()
        inputs.add("confidence")
        return inputs

    def get_output_names(self) -> Set[str]:
        outputs = super().get_output_names()
        outputs.update(["predictions", "top", "confidence", "parent_id"])
        return outputs

    def validate_field_selector(self, field_name: str, input_step: GraphNone) -> None:
        super().validate_field_selector(field_name=field_name, input_step=input_step)
        if field_name in {"confidence"}:
            if input_step.get_type() not in {"InferenceParameter"}:
                raise InvalidStepInputDetected(
                    f"Field {field_name} of step {self.type} comes from invalid input type: {input_step.get_type()}. "
                    f"Expected: `InferenceParameter`"
                )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        super().validate_field_binding(field_name=field_name, value=value)
        if field_name == "confidence":
            if not issubclass(type(value), float) and not issubclass(type(value), int):
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
    def confidence_must_be_selector_or_number(
        cls, value: Any
    ) -> Union[Optional[float], str]:
        if is_selector(selector_or_value=value) or value is None:
            return value
        if not issubclass(type(value), float) and not issubclass(type(value), int):
            raise ValueError("`confidence` field must be number")
        if not 0 <= value <= 1:
            raise ValueError("Parameter `confidence` must be in range [0.0, 1.0]")
        return value

    def get_input_names(self) -> Set[str]:
        inputs = super().get_input_names()
        inputs.add("confidence")
        return inputs

    def get_output_names(self) -> Set[str]:
        outputs = super().get_output_names()
        outputs.update(["predictions", "predicted_classes", "parent_id"])
        return outputs

    def validate_field_selector(self, field_name: str, input_step: GraphNone) -> None:
        super().validate_field_selector(field_name=field_name, input_step=input_step)
        if field_name in {"confidence"}:
            if input_step.get_type() not in {"InferenceParameter"}:
                raise InvalidStepInputDetected(
                    f"Field {field_name} of step {self.type} comes from invalid input type: {input_step.get_type()}. "
                    f"Expected: `InferenceParameter`"
                )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        super().validate_field_binding(field_name=field_name, value=value)
        if field_name == "confidence":
            if not issubclass(type(value), float) and not issubclass(type(value), int):
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

    @validator("class_agnostic_nms")
    @classmethod
    def class_agnostic_nms_must_be_selector_or_bool(
        cls, value: Any
    ) -> Union[Optional[bool], str]:
        if is_selector(selector_or_value=value) or value is None:
            return value
        if not issubclass(type(value), bool):
            raise ValueError("`class_agnostic_nms` field must be bool")
        return value

    @validator("class_filter")
    @classmethod
    def class_filter_must_be_selector_or_list_of_string(
        cls, value: Any
    ) -> Union[Optional[List[str]], str]:
        if is_selector(selector_or_value=value) or value is None:
            return value
        if not issubclass(type(value), list):
            raise ValueError("`class_filter` field must be list")
        if any(not issubclass(type(e), str) for e in value):
            raise ValueError("Parameter `class_filter` must be a list of string")
        return value

    @validator("confidence", "iou_threshold")
    @classmethod
    def field_must_be_selector_or_number_from_zero_to_one(
        cls, value: Any
    ) -> Union[Optional[float], str]:
        if is_selector(selector_or_value=value) or value is None:
            return value
        if not issubclass(type(value), float) and not issubclass(type(value), int):
            raise ValueError("field must be number")
        if not 0 <= value <= 1:
            raise ValueError("Parameter must be in range [0.0, 1.0]")
        return value

    @validator("max_detections", "max_candidates")
    @classmethod
    def field_must_be_selector_or_positive_number(
        cls, value: Any
    ) -> Union[Optional[int], str]:
        if is_selector(selector_or_value=value) or value is None:
            return value
        if not issubclass(type(value), float) and not issubclass(type(value), int):
            raise ValueError("field must be number")
        if value <= 0:
            raise ValueError("Parameter must be positive")
        return value

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
        outputs.update(["predictions", "parent_id"])
        return outputs

    def validate_field_selector(self, field_name: str, input_step: GraphNone) -> None:
        super().validate_field_selector(field_name=field_name, input_step=input_step)
        if field_name in {
            "class_agnostic_nms",
            "class_filter",
            "confidence",
            "iou_threshold",
            "max_detections",
            "max_candidates",
        }:
            if input_step.get_type() not in {"InferenceParameter"}:
                raise InvalidStepInputDetected(
                    f"Field {field_name} of step {self.type} comes from invalid input type: {input_step.get_type()}. "
                    f"Expected: `InferenceParameter`"
                )

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
            if not issubclass(type(value), float) and not issubclass(type(value), int):
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

    @validator("keypoint_confidence")
    @classmethod
    def field_must_be_selector_or_number_from_zero_to_one(
        cls, value: Any
    ) -> Union[Optional[float], str]:
        if is_selector(selector_or_value=value) or value is None:
            return value
        if not issubclass(type(value), float) and not issubclass(type(value), int):
            raise ValueError("field must be number")
        if not 0 <= value <= 1:
            raise ValueError("Parameter must be in range [0.0, 1.0]")
        return value

    def get_input_names(self) -> Set[str]:
        inputs = super().get_input_names()
        inputs.add("keypoint_confidence")
        return inputs

    def validate_field_selector(self, field_name: str, input_step: GraphNone) -> None:
        super().validate_field_selector(field_name=field_name, input_step=input_step)
        if field_name in {"keypoint_confidence"}:
            if input_step.get_type() not in {"InferenceParameter"}:
                raise InvalidStepInputDetected(
                    f"Field {field_name} of step {self.type} comes from invalid input type: {input_step.get_type()}. "
                    f"Expected: `InferenceParameter`"
                )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        super().validate_field_binding(field_name=field_name, value=value)
        if field_name == "keypoint_confidence":
            if not issubclass(type(value), float) and not issubclass(type(value), int):
                raise VariableTypeError(f"Parameter `{field_name}` must be a number")
            if not 0 <= value <= 1:
                raise VariableTypeError(
                    f"Parameter `{field_name}` must be in range [0.0, 1.0]"
                )


class InstanceSegmentationModel(ObjectDetectionModel):
    type: Literal["InstanceSegmentationModel"]
    mask_decode_mode: Optional[str] = Field(default="accurate")
    tradeoff_factor: Union[Optional[float], str] = Field(default=0.0)

    @validator("mask_decode_mode")
    @classmethod
    def mask_decode_mode_must_be_selector_or_one_of_allowed_values(
        cls, value: Any
    ) -> Optional[str]:
        if is_selector(selector_or_value=value) or value is None:
            return value
        if not issubclass(type(value), str):
            raise ValueError("Field `mask_decode_mode` must be string")
        if value not in {"accurate", "tradeoff", "fast"}:
            raise ValueError(
                "Field `mask_decode_mode` must be in 'accurate', 'tradeoff', 'fast'"
            )
        return value

    @validator("tradeoff_factor")
    @classmethod
    def field_must_be_selector_or_number_from_zero_to_one(
        cls, value: Any
    ) -> Union[Optional[float], str]:
        if is_selector(selector_or_value=value) or value is None:
            return value
        if not issubclass(type(value), float) and not issubclass(type(value), int):
            raise ValueError("field must be number")
        if not 0 <= value <= 1:
            raise ValueError("Parameter must be in range [0.0, 1.0]")
        return value

    def get_input_names(self) -> Set[str]:
        inputs = super().get_input_names()
        inputs.update(["mask_decode_mode", "tradeoff_factor"])
        return inputs

    def validate_field_selector(self, field_name: str, input_step: GraphNone) -> None:
        super().validate_field_selector(field_name=field_name, input_step=input_step)
        if field_name in {"mask_decode_mode", "tradeoff_factor"}:
            if input_step.get_type() not in {"InferenceParameter"}:
                raise InvalidStepInputDetected(
                    f"Field {field_name} of step {self.type} comes from invalid input type: {input_step.get_type()}. "
                    f"Expected: `InferenceParameter`"
                )

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
            if not issubclass(type(value), float) and not issubclass(type(value), int):
                raise VariableTypeError(f"Parameter `{field_name}` must be a number")
            if not 0 <= value <= 1:
                raise VariableTypeError(
                    f"Parameter `{field_name}` must be in range [0.0, 1.0]"
                )


class OCRModel(BaseModel, StepInterface):
    type: Literal["OCRModel"]
    name: str
    image: Union[str, List[str]]

    @validator("image")
    @classmethod
    def image_must_only_hold_selectors(cls, value: Any) -> Union[str, List[str]]:
        if issubclass(type(value), list):
            if any(not is_selector(selector_or_value=e) for e in value):
                raise ValueError("`image` field can only contain selector values")
        if not is_selector(selector_or_value=value):
            raise ValueError("`image` field can only contain selector values")
        return value

    def validate_field_selector(self, field_name: str, input_step: GraphNone) -> None:
        if not is_selector(selector_or_value=getattr(self, field_name)):
            raise ExecutionGraphError(
                f"Attempted to validate selector value for field {field_name}, but field is not selector."
            )
        if field_name == "image":
            if input_step.get_type() not in {
                "InferenceImage",
                "Crop",
                "RelativeStaticCrop",
                "AbsoluteStaticCrop",
            }:
                raise InvalidStepInputDetected(
                    f"Field {field_name} of step {self.type} comes from invalid input type: {input_step.get_type()}. "
                    f"Expected: `InferenceImage`, `Crop`, `RelativeStaticCrop`, `AbsoluteStaticCrop`"
                )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        if field_name == "image":
            try:
                if not issubclass(type(value), list):
                    value = [value]
                for e in value:
                    InferenceRequestImage.validate(e)
            except ValueError as error:
                raise VariableTypeError(
                    "Parameter `image` must be compatible with `InferenceRequestImage`"
                ) from error

    def get_type(self) -> str:
        return self.type

    def get_input_names(self) -> Set[str]:
        return {"image"}

    def get_output_names(self) -> Set[str]:
        return {"result", "parent_id"}


class Crop(BaseModel, StepInterface):
    type: Literal["Crop"]
    name: str
    image: Union[str, List[str]]
    detections: str

    @validator("image")
    @classmethod
    def image_must_only_hold_selectors(cls, value: Any) -> Union[str, List[str]]:
        if issubclass(type(value), list):
            if any(not is_selector(selector_or_value=e) for e in value):
                raise ValueError("`image` field can only contain selector values")
        if not is_selector(selector_or_value=value):
            raise ValueError("`image` field can only contain selector values")
        return value

    @validator("detections")
    @classmethod
    def detections_must_hold_selector(cls, value: Any) -> str:
        if not is_selector(selector_or_value=value):
            raise ValueError("`image` field can only contain selector values")
        return value

    def get_type(self) -> str:
        return self.type

    def get_input_names(self) -> Set[str]:
        return {"image", "detections"}

    def get_output_names(self) -> Set[str]:
        return {"crops", "parent_id"}

    def validate_field_selector(self, field_name: str, input_step: GraphNone) -> None:
        if not is_selector(selector_or_value=getattr(self, field_name)):
            raise ExecutionGraphError(
                f"Attempted to validate selector value for field {field_name}, but field is not selector."
            )
        if field_name == "image":
            if input_step.get_type() not in {
                "InferenceImage",
                "Crop",
                "RelativeStaticCrop",
                "AbsoluteStaticCrop",
            }:
                raise InvalidStepInputDetected(
                    f"Field {field_name} of step {self.type} comes from invalid input type: {input_step.get_type()}. "
                    f"Expected: `InferenceImage`, `Crop`, `RelativeStaticCrop`, `AbsoluteStaticCrop`"
                )
        if field_name == "detections":
            if input_step.get_type() not in {
                "ObjectDetectionModel",
                "KeypointsDetectionModel",
                "InstanceSegmentationModel",
                "DetectionFilter",
            }:
                raise InvalidStepInputDetected(
                    f"Crop step with name {self.name} cannot take as an input predictions from {input_step.get_type()}. "
                    f"Cropping requires detection-based output."
                )
            crop_step_image_reference = self.image
            if input_step.get_type() == "DetectionFilter":
                # Here, filter do not hold the reference to image, we skip the check in this case
                return None
            input_step_image_reference = input_step.image
            if crop_step_image_reference != input_step_image_reference:
                raise InvalidStepInputDetected(
                    f"Crop step with name {self.name} was given detections reference that is bound to different image: "
                    f"crop.image: {crop_step_image_reference}, detections step image: {input_step_image_reference}"
                )
            if get_last_selector_chunk(self.detections) != "predictions":
                raise InvalidStepInputDetected(
                    f"Crop step with name {self.name} must take as input step output of name `predictions`"
                )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        if field_name == "image":
            try:
                if not issubclass(type(value), list):
                    value = [value]
                for e in value:
                    InferenceRequestImage.validate(e)
            except ValueError as error:
                raise VariableTypeError(
                    "Parameter `image` must be compatible with `InferenceRequestImage`"
                ) from error


class Operator(Enum):
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    LOWER_THAN = "lower_than"
    GREATER_THAN = "greater_than"
    LOWER_OR_EQUAL_THAN = "lower_or_equal_than"
    GREATER_OR_EQUAL_THAN = "greater_or_equal_than"
    IN = "in"


class Condition(BaseModel, StepInterface):
    type: Literal["Condition"]
    name: str
    left: Union[float, int, bool, str, list, set]
    operator: Operator
    right: Union[float, int, bool, str, list, set]
    step_if_true: str
    step_if_false: str

    def get_type(self) -> str:
        return self.type

    def get_input_names(self) -> Set[str]:
        return {"left", "right"}

    def get_output_names(self) -> Set[str]:
        return set()

    def validate_field_selector(self, field_name: str, input_step: GraphNone) -> None:
        if not is_selector(selector_or_value=getattr(self, field_name)):
            raise ExecutionGraphError(
                f"Attempted to validate selector value for field {field_name}, but field is not selector."
            )
        input_type = input_step.get_type()
        if field_name in {"left", "right"}:
            if input_type == "InferenceImage":
                raise InvalidStepInputDetected(
                    f"Field {field_name} of step {self.type} comes from invalid input type: {input_type}. "
                    f"Expected: anything else than `InferenceImage`"
                )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        pass


class BinaryOperator(Enum):
    OR = "or"
    AND = "and"


class DetectionFilterDefinition(BaseModel):
    type: Literal["DetectionFilterDefinition"]
    field_name: str
    operator: Operator
    reference_value: Union[float, int, bool, str, list, set]


class CompoundDetectionFilterDefinition(BaseModel):
    type: Literal["CompoundDetectionFilterDefinition"]
    left: DetectionFilterDefinition
    operator: BinaryOperator
    right: DetectionFilterDefinition


class DetectionFilter(BaseModel, StepInterface):
    type: Literal["DetectionFilter"]
    name: str
    predictions: str
    filter_definition: Annotated[
        Union[DetectionFilterDefinition, CompoundDetectionFilterDefinition],
        Field(discriminator="type"),
    ]

    def get_input_names(self) -> Set[str]:
        return {"predictions"}

    def get_output_names(self) -> Set[str]:
        return {"predictions", "parent_id"}

    def validate_field_selector(self, field_name: str, input_step: GraphNone) -> None:
        if not is_selector(selector_or_value=getattr(self, field_name)):
            raise ExecutionGraphError(
                f"Attempted to validate selector value for field {field_name}, but field is not selector."
            )
        if field_name == "predictions":
            if input_step.get_type() not in {
                "ObjectDetectionModel",
                "KeypointsDetectionModel",
                "InstanceSegmentationModel",
            }:
                raise InvalidStepInputDetected(
                    f"DetectionFilter step with name {self.name} cannot take as an input predictions from {input_step.get_type()}. "
                    f"Cropping requires detection-based output."
                )
            if get_last_selector_chunk(self.predictions) != "predictions":
                raise InvalidStepInputDetected(
                    f"DetectionFilter step with name {self.name} must take as input step output of name `predictions`"
                )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        pass

    def get_type(self) -> str:
        return self.type


class DetectionOffset(BaseModel, StepInterface):
    type: Literal["DetectionOffset"]
    name: str
    predictions: str
    offset_x: Union[int, str]
    offset_y: Union[int, str]

    def get_input_names(self) -> Set[str]:
        return {"predictions", "offset_x", "offset_y"}

    def get_output_names(self) -> Set[str]:
        return {"predictions", "parent_id"}

    def validate_field_selector(self, field_name: str, input_step: GraphNone) -> None:
        if not is_selector(selector_or_value=getattr(self, field_name)):
            raise ExecutionGraphError(
                f"Attempted to validate selector value for field {field_name}, but field is not selector."
            )
        if field_name == "predictions":
            if input_step.get_type() not in {
                "ObjectDetectionModel",
                "KeypointsDetectionModel",
                "InstanceSegmentationModel",
            }:
                raise InvalidStepInputDetected(
                    f"DetectionOffset step with name {self.name} cannot take as an input predictions from {input_step.get_type()}. "
                    f"Cropping requires detection-based output."
                )
            if get_last_selector_chunk(self.predictions) != "predictions":
                raise InvalidStepInputDetected(
                    f"DetectionOffset step with name {self.name} must take as input step output of name `predictions`"
                )
        if field_name in {"offset_x", "offset_y"}:
            if input_step.get_type() not in {"InferenceParameter"}:
                raise InvalidStepInputDetected(
                    f"DetectionOffset step with name {self.name} must take `InferenceParameter` node as "
                    f"{field_name} selector"
                )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        if field_name in {"offset_x", "offset_y"}:
            if not issubclass(type(value), int):
                raise VariableTypeError(f"Parameter `{field_name}` must be integer'")

    def get_type(self) -> str:
        return self.type


class AbsoluteStaticCrop(BaseModel, StepInterface):
    type: Literal["AbsoluteStaticCrop"]
    name: str
    image: Union[str, List[str]]
    x_center: Union[int, str]
    y_center: Union[int, str]
    width: Union[int, str]
    height: Union[int, str]

    @validator("image")
    @classmethod
    def image_must_only_hold_selectors(cls, value: Any) -> Union[str, List[str]]:
        if issubclass(type(value), list):
            if any(not is_selector(selector_or_value=e) for e in value):
                raise ValueError("`image` field can only contain selector values")
        if not is_selector(selector_or_value=value):
            raise ValueError("`image` field can only contain selector values")
        return value

    @validator("x_center", "y_center", "width", "height")
    @classmethod
    def detections_must_hold_selector(cls, value: Any) -> str:
        if issubclass(type(value), str):
            if not is_selector(selector_or_value=value):
                raise ValueError("Field must be either integer of valid selector")
        elif not issubclass(type(value), int):
            raise ValueError("Field must be either integer of valid selector")
        return value

    def get_type(self) -> str:
        return self.type

    def get_input_names(self) -> Set[str]:
        return {"image", "x_center", "y_center", "width", "height"}

    def get_output_names(self) -> Set[str]:
        return {"crops", "parent_id"}

    def validate_field_selector(self, field_name: str, input_step: GraphNone) -> None:
        if not is_selector(selector_or_value=getattr(self, field_name)):
            raise ExecutionGraphError(
                f"Attempted to validate selector value for field {field_name}, but field is not selector."
            )
        if field_name == "image":
            if input_step.get_type() not in {
                "InferenceImage",
                "Crop",
                "RelativeStaticCrop",
                "AbsoluteStaticCrop",
            }:
                raise InvalidStepInputDetected(
                    f"Field {field_name} of step {self.type} comes from invalid input type: {input_step.get_type()}. "
                    f"Expected: `InferenceImage`, `Crop`, `RelativeStaticCrop`, `AbsoluteStaticCrop`"
                )
        if field_name in {"x_center", "y_center", "width", "height"}:
            if input_step.get_type() not in {"InferenceParameter"}:
                raise InvalidStepInputDetected(
                    f"DetectionOffset step with name {self.name} must take `InferenceParameter` node as "
                    f"{field_name} selector"
                )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        if field_name == "image":
            try:
                if not issubclass(type(value), list):
                    value = [value]
                for e in value:
                    InferenceRequestImage.validate(e)
            except ValueError as error:
                raise VariableTypeError(
                    "Parameter `image` must be compatible with `InferenceRequestImage`"
                ) from error
        if field_name in {"x_center", "y_center", "width", "height"}:
            if (
                not issubclass(type(value), int) and not issubclass(type(value), float)
            ) or value != round(value):
                raise VariableTypeError(
                    f"Field {field_name} of step {self.type} must be integer"
                )


class RelativeStaticCrop(BaseModel, StepInterface):
    type: Literal["RelativeStaticCrop"]
    name: str
    image: Union[str, List[str]]
    x_center: Union[float, str]
    y_center: Union[float, str]
    width: Union[float, str]
    height: Union[float, str]

    @validator("image")
    @classmethod
    def image_must_only_hold_selectors(cls, value: Any) -> Union[str, List[str]]:
        if issubclass(type(value), list):
            if any(not is_selector(selector_or_value=e) for e in value):
                raise ValueError("`image` field can only contain selector values")
        if not is_selector(selector_or_value=value):
            raise ValueError("`image` field can only contain selector values")
        return value

    @validator("x_center", "y_center", "width", "height")
    @classmethod
    def detections_must_hold_selector(cls, value: Any) -> str:
        if issubclass(type(value), str):
            if not is_selector(selector_or_value=value):
                raise ValueError("Field must be either float of valid selector")
        elif not issubclass(type(value), float):
            raise ValueError("Field must be either float of valid selector")
        return value

    def get_type(self) -> str:
        return self.type

    def get_input_names(self) -> Set[str]:
        return {"image", "x_center", "y_center", "width", "height"}

    def get_output_names(self) -> Set[str]:
        return {"crops", "parent_id"}

    def validate_field_selector(self, field_name: str, input_step: GraphNone) -> None:
        if not is_selector(selector_or_value=getattr(self, field_name)):
            raise ExecutionGraphError(
                f"Attempted to validate selector value for field {field_name}, but field is not selector."
            )
        if field_name == "image":
            if input_step.get_type() not in {
                "InferenceImage",
                "Crop",
                "RelativeStaticCrop",
                "AbsoluteStaticCrop",
            }:
                raise InvalidStepInputDetected(
                    f"Field {field_name} of step {self.type} comes from invalid input type: {input_step.get_type()}. "
                    f"Expected: `InferenceImage`, `Crop`, `RelativeStaticCrop`, `AbsoluteStaticCrop`"
                )
        if field_name in {"x_center", "y_center", "width", "height"}:
            if input_step.get_type() not in {"InferenceParameter"}:
                raise InvalidStepInputDetected(
                    f"DetectionOffset step with name {self.name} must take `InferenceParameter` node as "
                    f"{field_name} selector"
                )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        if field_name == "image":
            try:
                if not issubclass(type(value), list):
                    value = [value]
                for e in value:
                    InferenceRequestImage.validate(e)
            except ValueError as error:
                raise VariableTypeError(
                    "Parameter `image` must be compatible with `InferenceRequestImage`"
                ) from error
        if field_name in {"x_center", "y_center", "width", "height"}:
            if not issubclass(type(value), float):
                raise VariableTypeError(
                    f"Field {field_name} of step {self.type} must be float"
                )


class ClipComparison(BaseModel, StepInterface):
    type: Literal["ClipComparison"]
    name: str
    image: Union[str, List[str]]
    text: Union[str, List[str]]

    @validator("image")
    @classmethod
    def image_must_only_hold_selectors(cls, value: Any) -> Union[str, List[str]]:
        if issubclass(type(value), list):
            if any(not is_selector(selector_or_value=e) for e in value):
                raise ValueError("`image` field can only contain selector values")
        if not is_selector(selector_or_value=value):
            raise ValueError("`image` field can only contain selector values")
        return value

    @validator("text")
    @classmethod
    def text_must_be_valid(cls, value: Any) -> Union[str, List[str]]:
        if issubclass(type(value), list):
            if any(not issubclass(type(e), str) for e in value):
                raise ValueError(
                    "`text` field given as list must only contain string elements"
                )
        elif not issubclass(type(value), str):
            raise ValueError("`text` field given must be string or list of strings")
        return value

    def validate_field_selector(self, field_name: str, input_step: GraphNone) -> None:
        if not is_selector(selector_or_value=getattr(self, field_name)):
            raise ExecutionGraphError(
                f"Attempted to validate selector value for field {field_name}, but field is not selector."
            )
        if field_name == "image":
            if input_step.get_type() not in {
                "InferenceImage",
                "Crop",
                "RelativeStaticCrop",
                "AbsoluteStaticCrop",
            }:
                raise InvalidStepInputDetected(
                    f"Field {field_name} of step {self.type} comes from invalid input type: {input_step.get_type()}. "
                    f"Expected: `InferenceImage`, `Crop`, `RelativeStaticCrop`, `AbsoluteStaticCrop`"
                )
        if field_name == "text":
            if input_step.get_type() not in {"InferenceParameter"}:
                raise InvalidStepInputDetected(
                    f"Field {field_name} of step {self.type} must be 'InferenceParameter'"
                )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        if field_name == "image":
            try:
                if not issubclass(type(value), list):
                    value = [value]
                for e in value:
                    InferenceRequestImage.validate(e)
            except ValueError as error:
                raise VariableTypeError(
                    "Parameter `image` must be compatible with `InferenceRequestImage`"
                ) from error
        if field_name == "text":
            if issubclass(type(value), list):
                if any(not issubclass(type(e), str) for e in value):
                    raise VariableTypeError(
                        "`text` field given as list must only contain string elements"
                    )
            elif not issubclass(type(value), str):
                raise VariableTypeError(
                    "`text` field given must be string or list of strings"
                )

    def get_type(self) -> str:
        return self.type

    def get_input_names(self) -> Set[str]:
        return {"image", "text"}

    def get_output_names(self) -> Set[str]:
        return {"similarity", "parent_id"}


def is_selector(selector_or_value: Any) -> bool:
    if not issubclass(type(selector_or_value), str):
        return False
    return selector_or_value.startswith("$")


def get_last_selector_chunk(selector: str) -> str:
    return selector.split(".")[-1]
