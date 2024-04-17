from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    confloat,
    field_validator,
)
from typing_extensions import Annotated

from inference.enterprise.workflows.entities.base import GraphNone
from inference.enterprise.workflows.entities.types import (
    BAR_CODE_DETECTION_KIND,
    BOOLEAN_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    DICTIONARY_KIND,
    FLOAT_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    IMAGE_METADATA_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    PARENT_ID_KIND,
    PREDICTION_TYPE_KIND,
    QR_CODE_DETECTION_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    ROBOFLOW_PROJECT,
    STRING_KIND,
    WILDCARD_KIND,
    FloatZeroToOne,
    InferenceImageSelector,
    InferenceParameterSelector,
    Kind,
    OutputStepImageSelector,
    StepOutputSelector,
    StepSelector,
)
from inference.enterprise.workflows.entities.validators import (
    get_last_selector_chunk,
    is_selector,
    validate_field_has_given_type,
    validate_field_is_dict_of_strings,
    validate_field_is_list_of_string,
    validate_field_is_one_of_selected_values,
    validate_field_is_selector_or_has_given_type,
    validate_image_biding,
    validate_selector_holds_detection_predictions,
    validate_selector_holds_image,
    validate_selector_is_inference_parameter,
    validate_value_is_empty_or_number_in_range_zero_one,
    validate_value_is_empty_or_positive_number,
)
from inference.enterprise.workflows.errors import (
    ExecutionGraphError,
    InvalidStepInputDetected,
    VariableTypeError,
)


class OutputDefinition(BaseModel):
    name: str
    kind: List[Kind]


class StepInterface(GraphNone, metaclass=ABCMeta):
    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []

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
    def validate_field_selector(
        self, field_name: str, input_step: GraphNone, index: Optional[int] = None
    ) -> None:
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
    model_config = ConfigDict(protected_namespaces=())
    type: Literal["RoboflowModel"]
    name: str = Field(description="Unique name of step in workflows")
    image: Union[InferenceImageSelector, OutputStepImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    model_id: Union[InferenceParameterSelector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = (
        Field(
            description="Roboflow model identifier",
            examples=["my_project/3", "$inputs.model"],
        )
    )
    disable_active_learning: Union[
        Optional[bool], InferenceParameterSelector(kind=[BOOLEAN_KIND])
    ] = Field(
        default=False,
        description="Parameter to decide if Active Learning data sampling is disabled for the model",
        examples=[True, "$inputs.disable_active_learning"],
    )
    active_learning_target_dataset: Union[
        Optional[str], InferenceParameterSelector(kind=[ROBOFLOW_PROJECT])
    ] = Field(
        default=None,
        description="Target dataset for Active Learning data sampling - see Roboflow Active Learning "
        "docs for more information",
        examples=["my_project", "$inputs.al_target_project"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return super(RoboflowModel, cls).describe_outputs() + [
            OutputDefinition(name="prediction_type", kind=[PREDICTION_TYPE_KIND])
        ]

    def get_type(self) -> str:
        return self.type

    def get_input_names(self) -> Set[str]:
        return {
            "image",
            "model_id",
            "disable_active_learning",
            "active_learning_target_dataset",
        }

    def get_output_names(self) -> Set[str]:
        return {"prediction_type"}

    def validate_field_selector(
        self, field_name: str, input_step: GraphNone, index: Optional[int] = None
    ) -> None:
        if not is_selector(selector_or_value=getattr(self, field_name)):
            raise ExecutionGraphError(
                f"Attempted to validate selector value for field {field_name}, but field is not selector."
            )
        validate_selector_holds_image(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
        )
        validate_selector_is_inference_parameter(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
            applicable_fields={
                "model_id",
                "disable_active_learning",
                "active_learning_target_dataset",
            },
        )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        if field_name == "image":
            validate_image_biding(value=value)
        elif field_name == "model_id":
            validate_field_has_given_type(
                field_name=field_name,
                allowed_types=[str],
                value=value,
                error=VariableTypeError,
            )
        elif field_name == "disable_active_learning":
            validate_field_has_given_type(
                field_name=field_name,
                allowed_types=[bool],
                value=value,
                error=VariableTypeError,
            )
        elif field_name in {"active_learning_target_dataset"}:
            validate_field_has_given_type(
                field_name=field_name,
                allowed_types=[type(None), str],
                value=value,
                error=VariableTypeError,
            )


class ClassificationModel(RoboflowModel):
    model_config = ConfigDict(
        json_schema_extra={
            "block_type": "model",
            "description": "Run a classification model.",
            "docs": "https://inference.roboflow.com/workflows/classify_objects",
        }
    )
    type: Literal["ClassificationModel"]
    confidence: Union[
        Optional[FloatZeroToOne],
        InferenceParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.4,
        description="Confidence threshold for predictions",
        examples=[0.3, "$inputs.confidence_threshold"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return super(ClassificationModel, cls).describe_outputs() + [
            OutputDefinition(name="predictions", kind=[CLASSIFICATION_PREDICTION_KIND]),
            OutputDefinition(name="top", kind=[STRING_KIND]),
            OutputDefinition(name="confidence", kind=[FLOAT_ZERO_TO_ONE_KIND]),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
        ]

    def get_input_names(self) -> Set[str]:
        inputs = super().get_input_names()
        inputs.add("confidence")
        return inputs

    def get_output_names(self) -> Set[str]:
        outputs = super().get_output_names()
        outputs.update(["predictions", "top", "confidence", "parent_id"])
        return outputs

    def validate_field_selector(
        self, field_name: str, input_step: GraphNone, index: Optional[int] = None
    ) -> None:
        super().validate_field_selector(field_name=field_name, input_step=input_step)
        validate_selector_is_inference_parameter(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
            applicable_fields={"confidence"},
        )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        super().validate_field_binding(field_name=field_name, value=value)
        if field_name == "confidence":
            if value is None:
                raise VariableTypeError("Parameter `confidence` cannot be None")
            validate_value_is_empty_or_number_in_range_zero_one(
                value=value, error=VariableTypeError
            )


class MultiLabelClassificationModel(RoboflowModel):
    model_config = ConfigDict(
        json_schema_extra={
            "block_type": "model",
            "description": "Run a multi-label classification model.",
            "docs": "https://inference.roboflow.com/workflows/classify_objects_multi",
        }
    )
    type: Literal["MultiLabelClassificationModel"]
    confidence: Union[
        Optional[FloatZeroToOne],
        InferenceParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.4,
        description="Confidence threshold for predictions",
        examples=[0.3, "$inputs.confidence_threshold"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return super(MultiLabelClassificationModel, cls).describe_outputs() + [
            OutputDefinition(name="predictions", kind=[CLASSIFICATION_PREDICTION_KIND]),
            OutputDefinition(name="predicted_classes", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
        ]

    def get_input_names(self) -> Set[str]:
        inputs = super().get_input_names()
        inputs.add("confidence")
        return inputs

    def get_output_names(self) -> Set[str]:
        outputs = super().get_output_names()
        outputs.update(["predictions", "predicted_classes", "parent_id"])
        return outputs

    def validate_field_selector(
        self, field_name: str, input_step: GraphNone, index: Optional[int] = None
    ) -> None:
        super().validate_field_selector(field_name=field_name, input_step=input_step)
        validate_selector_is_inference_parameter(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
            applicable_fields={"confidence"},
        )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        super().validate_field_binding(field_name=field_name, value=value)
        if field_name == "confidence":
            if value is None:
                raise VariableTypeError("Parameter `confidence` cannot be None")
            validate_value_is_empty_or_number_in_range_zero_one(
                value=value, error=VariableTypeError
            )


class ObjectDetectionModel(RoboflowModel):
    model_config = ConfigDict(
        json_schema_extra={
            "block_type": "model",
            "description": "Detect objects using an object detection model.",
            "docs": "https://inference.roboflow.com/workflows/detect_objects",
        }
    )
    type: Literal["ObjectDetectionModel"]
    class_agnostic_nms: Union[
        Optional[bool], InferenceParameterSelector(kind=[BOOLEAN_KIND])
    ] = Field(
        default=False,
        description="Value to decide if NMS is to be used in class-agnostic mode.",
        examples=[True, "$inputs.class_agnostic_nms"],
    )
    class_filter: Union[
        Optional[List[str]], InferenceParameterSelector(kind=[LIST_OF_VALUES_KIND])
    ] = Field(
        default=None,
        description="List of classes to retrieve from predictions (to define subset of those which was used while model training)",
        examples=[["a", "b", "c"], "$inputs.class_filter"],
    )
    confidence: Union[
        Optional[FloatZeroToOne],
        InferenceParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.4,
        description="Confidence threshold for predictions",
        examples=[0.3, "$inputs.confidence_threshold"],
    )
    iou_threshold: Union[
        Optional[FloatZeroToOne],
        InferenceParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.3,
        description="Parameter of NMS, to decide on minimum box intersection over union to merge boxes",
        examples=[0.4, "$inputs.iou_threshold"],
    )
    max_detections: Union[
        Optional[PositiveInt], InferenceParameterSelector(kind=[INTEGER_KIND])
    ] = Field(
        default=300,
        description="Maximum number of detections to return",
        examples=[300, "$inputs.max_detections"],
    )
    max_candidates: Union[
        Optional[PositiveInt], InferenceParameterSelector(kind=[INTEGER_KIND])
    ] = Field(
        default=3000,
        description="Maximum number of candidates as NMS input to be taken into account.",
        examples=[3000, "$inputs.max_candidates"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return super(ObjectDetectionModel, cls).describe_outputs() + [
            OutputDefinition(
                name="predictions", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            ),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="image", kind=[IMAGE_METADATA_KIND]),
        ]

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
        outputs.update(["predictions", "parent_id", "image"])
        return outputs

    def validate_field_selector(
        self, field_name: str, input_step: GraphNone, index: Optional[int] = None
    ) -> None:
        super().validate_field_selector(field_name=field_name, input_step=input_step)
        validate_selector_is_inference_parameter(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
            applicable_fields={
                "class_agnostic_nms",
                "class_filter",
                "confidence",
                "iou_threshold",
                "max_detections",
                "max_candidates",
            },
        )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        super().validate_field_binding(field_name=field_name, value=value)
        if value is None:
            raise VariableTypeError(f"Parameter `{field_name}` cannot be None")
        if field_name == "class_agnostic_nms":
            validate_field_has_given_type(
                field_name=field_name,
                allowed_types=[bool],
                value=value,
                error=VariableTypeError,
            )
        elif field_name == "class_filter":
            if value is None:
                return None
            validate_field_is_list_of_string(
                value=value, field_name=field_name, error=VariableTypeError
            )
        elif field_name == "confidence" or field_name == "iou_threshold":
            validate_value_is_empty_or_number_in_range_zero_one(
                value=value,
                field_name=field_name,
                error=VariableTypeError,
            )
        elif field_name == "max_detections" or field_name == "max_candidates":
            validate_value_is_empty_or_positive_number(
                value=value,
                field_name=field_name,
                error=VariableTypeError,
            )


class KeypointsDetectionModel(ObjectDetectionModel):
    model_config = ConfigDict(
        json_schema_extra={
            "block_type": "model",
            "description": "Run inference on a keypoint detection model.",
            "docs": "https://inference.roboflow.com/workflows/detect_keypoints",
        }
    )
    type: Literal["KeypointsDetectionModel"]
    keypoint_confidence: Union[
        Optional[FloatZeroToOne],
        InferenceParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.0,
        description="Confidence threshold to predict keypoint as visible.",
        examples=[0.3, "$inputs.keypoint_confidence"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        outputs = super(KeypointsDetectionModel, cls).describe_outputs()
        result = []
        for o in outputs:
            if o.name != "predictions":
                result.append(o)
            else:
                result.append(
                    OutputDefinition(
                        name="predictions", kind=[KEYPOINT_DETECTION_PREDICTION_KIND]
                    ),
                )
        return result

    def get_input_names(self) -> Set[str]:
        inputs = super().get_input_names()
        inputs.add("keypoint_confidence")
        return inputs

    def validate_field_selector(
        self, field_name: str, input_step: GraphNone, index: Optional[int] = None
    ) -> None:
        super().validate_field_selector(field_name=field_name, input_step=input_step)
        validate_selector_is_inference_parameter(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
            applicable_fields={"keypoint_confidence"},
        )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        super().validate_field_binding(field_name=field_name, value=value)
        if field_name == "keypoint_confidence":
            validate_value_is_empty_or_number_in_range_zero_one(
                value=value,
                field_name=field_name,
                error=VariableTypeError,
            )


DECODE_MODES = {"accurate", "tradeoff", "fast"}


class InstanceSegmentationModel(ObjectDetectionModel):
    model_config = ConfigDict(
        json_schema_extra={
            "block_type": "model",
            "description": "Run an instance segmentation model.",
            "docs": "https://inference.roboflow.com/workflows/segment_objects",
        }
    )
    type: Literal["InstanceSegmentationModel"]
    mask_decode_mode: Union[
        Literal["accurate", "tradeoff", "fast"],
        InferenceParameterSelector(kind=[STRING_KIND]),
    ] = Field(
        default="accurate",
        description="Parameter of mask decoding in prediction post-processing.",
        examples=["accurate", "$inputs.mask_decode_mode"],
    )
    tradeoff_factor: Union[
        Optional[FloatZeroToOne],
        InferenceParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.0,
        description="Post-processing parameter to dictate tradeoff between fast and accurate",
        examples=[0.3, "$inputs.tradeoff_factor"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        outputs = super(InstanceSegmentationModel, cls).describe_outputs()
        result = []
        for o in outputs:
            if o.name != "predictions":
                result.append(o)
            else:
                result.append(
                    OutputDefinition(
                        name="predictions", kind=[INSTANCE_SEGMENTATION_PREDICTION_KIND]
                    )
                )
        return result

    def get_input_names(self) -> Set[str]:
        inputs = super().get_input_names()
        inputs.update(["mask_decode_mode", "tradeoff_factor"])
        return inputs

    def validate_field_selector(
        self, field_name: str, input_step: GraphNone, index: Optional[int] = None
    ) -> None:
        super().validate_field_selector(field_name=field_name, input_step=input_step)
        validate_selector_is_inference_parameter(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
            applicable_fields={"mask_decode_mode", "tradeoff_factor"},
        )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        super().validate_field_binding(field_name=field_name, value=value)
        if field_name == "mask_decode_mode":
            validate_field_is_one_of_selected_values(
                value=value,
                field_name=field_name,
                selected_values=DECODE_MODES,
                error=VariableTypeError,
            )
        elif field_name == "tradeoff_factor":
            validate_value_is_empty_or_number_in_range_zero_one(
                value=value,
                field_name=field_name,
                error=VariableTypeError,
            )


class OCRModel(BaseModel, StepInterface):
    model_config = ConfigDict(
        json_schema_extra={
            "block_type": "model",
            "description": "Run Optical Character Recognition on a model.",
            "docs": "https://inference.roboflow.com/workflows/ocr",
        }
    )
    type: Literal["OCRModel"]
    name: str = Field(description="Unique name of step in workflows")
    image: Union[InferenceImageSelector, OutputStepImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return super(OCRModel, cls).describe_outputs() + [
            OutputDefinition(name="result", kind=[STRING_KIND]),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="prediction_type", kind=[PREDICTION_TYPE_KIND]),
        ]

    def validate_field_selector(
        self, field_name: str, input_step: GraphNone, index: Optional[int] = None
    ) -> None:
        if not is_selector(selector_or_value=getattr(self, field_name)):
            raise ExecutionGraphError(
                f"Attempted to validate selector value for field {field_name}, but field is not selector."
            )
        validate_selector_holds_image(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
        )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        if field_name == "image":
            validate_image_biding(value=value)

    def get_type(self) -> str:
        return self.type

    def get_input_names(self) -> Set[str]:
        return {"image"}

    def get_output_names(self) -> Set[str]:
        return {"result", "parent_id", "prediction_type"}


class Crop(BaseModel, StepInterface):
    model_config = ConfigDict(
        json_schema_extra={
            "block_type": "transformation",
            "description": "Create dynamic crops based on detections from a detections-based model.",
            "docs": "https://inference.roboflow.com/workflows/crop",
        }
    )
    type: Literal["Crop"]
    name: str = Field(description="Unique name of step in workflows")
    image: Union[InferenceImageSelector, OutputStepImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    predictions: StepOutputSelector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Reference to predictions of detection-like model, that can be based of cropping (detection must define RoI - eg: bounding box)",
        examples=["$steps.my_object_detection_model.predictions"],
        validation_alias=AliasChoices("predictions", "detections"),
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return super(Crop, cls).describe_outputs() + [
            OutputDefinition(name="crops", kind=[IMAGE_KIND]),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
        ]

    def get_type(self) -> str:
        return self.type

    def get_input_names(self) -> Set[str]:
        return {"image", "predictions"}

    def get_output_names(self) -> Set[str]:
        return {"crops", "parent_id"}

    def validate_field_selector(
        self, field_name: str, input_step: GraphNone, index: Optional[int] = None
    ) -> None:
        if not is_selector(selector_or_value=getattr(self, field_name)):
            raise ExecutionGraphError(
                f"Attempted to validate selector value for field {field_name}, but field is not selector."
            )
        validate_selector_holds_image(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
        )
        validate_selector_holds_detection_predictions(
            step_name=self.name,
            image_selector=self.image,
            detections_selector=self.predictions,
            field_name=field_name,
            input_step=input_step,
        )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        if field_name == "image":
            validate_image_biding(value=value)


class Operator(Enum):
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    LOWER_THAN = "lower_than"
    GREATER_THAN = "greater_than"
    LOWER_OR_EQUAL_THAN = "lower_or_equal_than"
    GREATER_OR_EQUAL_THAN = "greater_or_equal_than"
    IN = "in"
    STR_STARTS_WITH = "str_starts_with"
    STR_ENDS_WITH = "str_ends_with"
    STR_CONTAINS = "str_contains"


class Condition(BaseModel, StepInterface):
    model_config = ConfigDict(
        json_schema_extra={
            "block_type": "flow_control",
            "description": "Control the flow of a workflow based on the result of a step.",
            "docs": "https://inference.roboflow.com/workflows/condition",
        }
    )
    type: Literal["Condition"]
    name: str = Field(description="Unique name of step in workflows")
    left: Union[
        float,
        int,
        bool,
        StepOutputSelector(
            kind=[
                FLOAT_KIND,
                INTEGER_KIND,
                BOOLEAN_KIND,
                STRING_KIND,
                LIST_OF_VALUES_KIND,
            ]
        ),
        str,
        list,
        set,
    ] = Field(
        description="Left operand of expression `left operator right` to evaluate boolean value of condition statement",
        examples=["$steps.classification.top", 3, "some"],
    )
    operator: Operator = Field(
        description="Operator in expression `left operator right` to evaluate boolean value of condition statement",
        examples=["equal", "in"],
    )
    right: Union[
        float,
        int,
        bool,
        StepOutputSelector(
            kind=[
                FLOAT_KIND,
                INTEGER_KIND,
                BOOLEAN_KIND,
                STRING_KIND,
                LIST_OF_VALUES_KIND,
            ]
        ),
        str,
        list,
        set,
    ] = Field(
        description="Right operand of expression `left operator right` to evaluate boolean value of condition statement",
        examples=["$steps.classification.top", 3, "some"],
    )
    step_if_true: StepSelector = Field(
        description="Reference to step which shall be executed if expression evaluates to true",
        examples=["$steps.on_true"],
    )
    step_if_false: StepSelector = Field(
        description="Reference to step which shall be executed if expression evaluates to false",
        examples=["$steps.on_false"],
    )

    def get_type(self) -> str:
        return self.type

    def get_input_names(self) -> Set[str]:
        return {"left", "right"}

    def get_output_names(self) -> Set[str]:
        return set()

    def validate_field_selector(
        self, field_name: str, input_step: GraphNone, index: Optional[int] = None
    ) -> None:
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


class QRCodeDetector(BaseModel, StepInterface):
    model_config = ConfigDict(
        json_schema_extra={
            "block_type": "model",
            "description": "Detect the location of QR codes in an image.",
            "docs": "https://inference.roboflow.com/workflows/detect_qr_codes",
        }
    )
    type: Literal["QRCodeDetector", "QRCodeDetection"]
    name: str = Field(description="Unique name of step in workflows")
    image: Union[InferenceImageSelector, OutputStepImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return super(QRCodeDetector, cls).describe_outputs() + [
            OutputDefinition(name="predictions", kind=[QR_CODE_DETECTION_KIND]),
            OutputDefinition(name="image", kind=[IMAGE_METADATA_KIND]),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
        ]

    def get_type(self) -> str:
        return self.type

    def get_input_names(self) -> Set[str]:
        return {"image"}

    def get_output_names(self) -> Set[str]:
        return {"predictions", "image", "parent_id"}

    def validate_field_selector(
        self, field_name: str, input_step: GraphNone, index: Optional[int] = None
    ) -> None:
        if not is_selector(selector_or_value=getattr(self, field_name)):
            raise ExecutionGraphError(
                f"Attempted to validate selector value for field {field_name}, but field is not selector."
            )
        validate_selector_holds_image(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
        )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        if field_name == "image":
            validate_image_biding(value=value)


class BarcodeDetector(BaseModel, StepInterface):
    model_config = ConfigDict(
        json_schema_extra={
            "block_type": "model",
            "description": "Detect the location of barcodes in an image.",
            "docs": None,
        }
    )
    type: Literal["BarcodeDetector", "BarcodeDetection"]
    name: str = Field(description="Unique name of step in workflows")
    image: Union[InferenceImageSelector, OutputStepImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return super(BarcodeDetector, cls).describe_outputs() + [
            OutputDefinition(name="predictions", kind=[BAR_CODE_DETECTION_KIND]),
            OutputDefinition(name="image", kind=[IMAGE_METADATA_KIND]),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
        ]

    def get_type(self) -> str:
        return self.type

    def get_input_names(self) -> Set[str]:
        return {"image"}

    def get_output_names(self) -> Set[str]:
        return {"predictions", "image", "parent_id"}

    def validate_field_selector(
        self, field_name: str, input_step: GraphNone, index: Optional[int] = None
    ) -> None:
        if not is_selector(selector_or_value=getattr(self, field_name)):
            raise ExecutionGraphError(
                f"Attempted to validate selector value for field {field_name}, but field is not selector."
            )
        validate_selector_holds_image(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
        )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        if field_name == "image":
            validate_image_biding(value=value)


class DetectionFilterDefinition(BaseModel):
    type: Literal["DetectionFilterDefinition"]
    field_name: str = Field(
        description="Name of detection-like prediction element field to take into filtering expression evaluation: `predictions[<idx>][<field_name>] operator reference_value`",
        examples=[
            "x",
            "y",
            "width",
            "height",
            "confidence",
            "class",
            "class_id",
            "points",
            "keypoints",
        ],
    )
    operator: Operator = Field(
        description="Operator in filtering expression: `predictions[<idx>][<field_name>] operator reference_value`",
        examples=["equal", "in"],
    )
    reference_value: Union[float, int, bool, str, list, set] = Field(
        description="Reference value to take into filtering expression evaluation: `predictions[<idx>][<field_name>] operator reference_value`",
        examples=[0.3, 300],
    )


class CompoundDetectionFilterDefinition(BaseModel):
    type: Literal["CompoundDetectionFilterDefinition"]
    left: Annotated[
        Union[DetectionFilterDefinition, "CompoundDetectionFilterDefinition"],
        Field(
            discriminator="type",
            description="Left operand (potentially nested expression) in expression `left bin_operator right`",
        ),
    ]
    operator: BinaryOperator = Field(
        description="Binary operator in expression `left bin_operator right`",
        examples=["and", "or"],
    )
    right: Annotated[
        Union[DetectionFilterDefinition, "CompoundDetectionFilterDefinition"],
        Field(
            discriminator="type",
            description="Right operand (potentially nested expression) in expression `left bin_operator right`",
        ),
    ]


class DetectionFilter(BaseModel, StepInterface):
    model_config = ConfigDict(
        json_schema_extra={
            "block_type": "transformation",
            "description": "Filter predictions from detection models based on defined conditions.",
            "docs": "https://inference.roboflow.com/workflows/filter_detections",
        }
    )
    type: Literal["DetectionFilter"]
    name: str = Field(description="Unique name of step in workflows")
    predictions: StepOutputSelector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Reference to detection-like predictions",
        examples=["$steps.object_detection_model.predictions"],
    )
    filter_definition: Annotated[
        Union[DetectionFilterDefinition, CompoundDetectionFilterDefinition],
        Field(
            discriminator="type",
            description="Definition of a filter expression to be applied for each element of detection-like predictions to decide if element should persist in the output. Can be used for instance to filter-out bounding boxes based on coordinates.",
        ),
    ]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return super(DetectionFilter, cls).describe_outputs() + [
            OutputDefinition(
                name="predictions",
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                    KEYPOINT_DETECTION_PREDICTION_KIND,
                ],
            ),
            OutputDefinition(name="image", kind=[IMAGE_METADATA_KIND]),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="prediction_type", kind=[PREDICTION_TYPE_KIND]),
        ]

    def get_input_names(self) -> Set[str]:
        return {"predictions"}

    def get_output_names(self) -> Set[str]:
        return {"predictions", "parent_id", "image", "prediction_type"}

    def validate_field_selector(
        self, field_name: str, input_step: GraphNone, index: Optional[int] = None
    ) -> None:
        if not is_selector(selector_or_value=getattr(self, field_name)):
            raise ExecutionGraphError(
                f"Attempted to validate selector value for field {field_name}, but field is not selector."
            )
        validate_selector_holds_detection_predictions(
            step_name=self.name,
            image_selector=None,
            detections_selector=self.predictions,
            field_name=field_name,
            input_step=input_step,
            applicable_fields={"predictions"},
        )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        pass

    def get_type(self) -> str:
        return self.type


class DetectionOffset(BaseModel, StepInterface):
    model_config = ConfigDict(
        json_schema_extra={
            "block_type": "transformation",
            "description": "Apply a fixed offset on the width and height of detections.",
            "docs": "https://inference.roboflow.com/workflows/offset_detections",
        }
    )
    type: Literal["DetectionOffset"]
    name: str = Field(description="Unique name of step in workflows")
    predictions: StepOutputSelector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Reference to detection-like predictions",
        examples=["$steps.object_detection_model.predictions"],
    )
    offset_x: Union[PositiveInt, InferenceParameterSelector(kind=[INTEGER_KIND])] = (
        Field(description="Offset for boxes width", examples=[10, "$inputs.offset_x"])
    )
    offset_y: Union[PositiveInt, InferenceParameterSelector(kind=[INTEGER_KIND])] = (
        Field(description="Offset for boxes height", examples=[10, "$inputs.offset_y"])
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return super(DetectionOffset, cls).describe_outputs() + [
            OutputDefinition(
                name="predictions",
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                    KEYPOINT_DETECTION_PREDICTION_KIND,
                ],
            ),
            OutputDefinition(name="image", kind=[IMAGE_METADATA_KIND]),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="prediction_type", kind=[PREDICTION_TYPE_KIND]),
        ]

    def get_input_names(self) -> Set[str]:
        return {"predictions", "offset_x", "offset_y"}

    def get_output_names(self) -> Set[str]:
        return {"predictions", "parent_id", "image", "prediction_type"}

    def validate_field_selector(
        self, field_name: str, input_step: GraphNone, index: Optional[int] = None
    ) -> None:
        if not is_selector(selector_or_value=getattr(self, field_name)):
            raise ExecutionGraphError(
                f"Attempted to validate selector value for field {field_name}, but field is not selector."
            )
        validate_selector_holds_detection_predictions(
            step_name=self.name,
            image_selector=None,
            detections_selector=self.predictions,
            field_name=field_name,
            input_step=input_step,
            applicable_fields={"predictions"},
        )
        validate_selector_is_inference_parameter(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
            applicable_fields={"offset_x", "offset_y"},
        )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        if field_name in {"offset_x", "offset_y"}:
            validate_field_has_given_type(
                field_name=field_name,
                value=value,
                allowed_types=[int],
                error=VariableTypeError,
            )

    def get_type(self) -> str:
        return self.type


class AbsoluteStaticCrop(BaseModel, StepInterface):
    model_config = ConfigDict(
        json_schema_extra={
            "block_type": "transformation",
            "description": "Use absolute coordinates for cropping.",
            "docs": "https://inference.roboflow.com/workflows/absolute_static_crop",
        }
    )
    type: Literal["AbsoluteStaticCrop"]
    name: str = Field(description="Unique name of step in workflows")
    image: Union[InferenceImageSelector, OutputStepImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    x_center: Union[PositiveInt, InferenceParameterSelector(kind=[INTEGER_KIND])] = (
        Field(
            description="Center X of static crop (absolute coordinate)",
            examples=[40, "$inputs.center_x"],
        )
    )
    y_center: Union[PositiveInt, InferenceParameterSelector(kind=[INTEGER_KIND])] = (
        Field(
            description="Center Y of static crop (absolute coordinate)",
            examples=[40, "$inputs.center_y"],
        )
    )
    width: Union[PositiveInt, InferenceParameterSelector(kind=[INTEGER_KIND])] = Field(
        description="Width of static crop (absolute value)",
        examples=[40, "$inputs.width"],
    )
    height: Union[PositiveInt, InferenceParameterSelector(kind=[INTEGER_KIND])] = Field(
        description="Height of static crop (absolute value)",
        examples=[40, "$inputs.height"],
    )

    def get_type(self) -> str:
        return self.type

    def get_input_names(self) -> Set[str]:
        return {"image", "x_center", "y_center", "width", "height"}

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return super(AbsoluteStaticCrop, cls).describe_outputs() + [
            OutputDefinition(name="crops", kind=[IMAGE_KIND]),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
        ]

    def get_output_names(self) -> Set[str]:
        return {"crops", "parent_id"}

    def validate_field_selector(
        self, field_name: str, input_step: GraphNone, index: Optional[int] = None
    ) -> None:
        if not is_selector(selector_or_value=getattr(self, field_name)):
            raise ExecutionGraphError(
                f"Attempted to validate selector value for field {field_name}, but field is not selector."
            )
        validate_selector_holds_image(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
        )
        validate_selector_is_inference_parameter(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
            applicable_fields={"x_center", "y_center", "width", "height"},
        )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        if field_name == "image":
            validate_image_biding(value=value)
        if field_name in {"x_center", "y_center", "width", "height"}:
            if (
                not issubclass(type(value), int) and not issubclass(type(value), float)
            ) or value != round(value):
                raise VariableTypeError(
                    f"Field {field_name} of step {self.type} must be integer"
                )


class RelativeStaticCrop(BaseModel, StepInterface):
    model_config = ConfigDict(
        json_schema_extra={
            "block_type": "transformation",
            "description": "Use relative coordinates for cropping.",
            "docs": "https://inference.roboflow.com/workflows/absolute_static_crop",
        }
    )
    type: Literal["RelativeStaticCrop"]
    name: str = Field(description="Unique name of step in workflows")
    image: Union[InferenceImageSelector, OutputStepImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    x_center: Union[
        FloatZeroToOne, InferenceParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        description="Center X of static crop (relative coordinate 0.0-1.0)",
        examples=[0.3, "$inputs.center_x"],
    )
    y_center: Union[
        FloatZeroToOne, InferenceParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        description="Center Y of static crop (relative coordinate 0.0-1.0)",
        examples=[0.3, "$inputs.center_y"],
    )
    width: Union[
        FloatZeroToOne, InferenceParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        description="Width of static crop (relative value 0.0-1.0)",
        examples=[0.3, "$inputs.width"],
    )
    height: Union[
        FloatZeroToOne, InferenceParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        description="Height of static crop (relative value 0.0-1.0)",
        examples=[0.3, "$inputs.height"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return super(RelativeStaticCrop, cls).describe_outputs() + [
            OutputDefinition(name="crops", kind=[IMAGE_KIND]),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
        ]

    def get_type(self) -> str:
        return self.type

    def get_input_names(self) -> Set[str]:
        return {"image", "x_center", "y_center", "width", "height"}

    def get_output_names(self) -> Set[str]:
        return {"crops", "parent_id"}

    def validate_field_selector(
        self, field_name: str, input_step: GraphNone, index: Optional[int] = None
    ) -> None:
        if not is_selector(selector_or_value=getattr(self, field_name)):
            raise ExecutionGraphError(
                f"Attempted to validate selector value for field {field_name}, but field is not selector."
            )
        validate_selector_holds_image(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
        )
        validate_selector_is_inference_parameter(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
            applicable_fields={"x_center", "y_center", "width", "height"},
        )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        if field_name == "image":
            validate_image_biding(value=value)
        if field_name in {"x_center", "y_center", "width", "height"}:
            validate_field_has_given_type(
                field_name=field_name,
                value=value,
                allowed_types=[float],
                error=VariableTypeError,
            )


class ClipComparison(BaseModel, StepInterface):
    model_config = ConfigDict(
        json_schema_extra={
            "block_type": "model",
            "description": "Compare CLIP image and text embeddings.",
            "docs": "https://inference.roboflow.com/workflows/compare_clip_vectors",
        }
    )
    type: Literal["ClipComparison"]
    name: str = Field(description="Unique name of step in workflows")
    image: Union[InferenceImageSelector, OutputStepImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    text: Union[InferenceParameterSelector(kind=[LIST_OF_VALUES_KIND]), List[str]] = (
        Field(
            description="List of texts to calculate similarity against each input image",
            examples=[["a", "b", "c"], "$inputs.texts"],
        )
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return super(ClipComparison, cls).describe_outputs() + [
            OutputDefinition(name="similarity", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="predictions_type", kind=[PREDICTION_TYPE_KIND]),
        ]

    def validate_field_selector(
        self, field_name: str, input_step: GraphNone, index: Optional[int] = None
    ) -> None:
        if not is_selector(selector_or_value=getattr(self, field_name)):
            raise ExecutionGraphError(
                f"Attempted to validate selector value for field {field_name}, but field is not selector."
            )
        validate_selector_holds_image(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
        )
        validate_selector_is_inference_parameter(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
            applicable_fields={"text"},
        )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        if field_name == "image":
            validate_image_biding(value=value)
        if field_name == "text":
            if issubclass(type(value), list):
                validate_field_is_list_of_string(
                    value=value, field_name=field_name, error=VariableTypeError
                )
            elif not issubclass(type(value), str):
                validate_field_has_given_type(
                    value=value,
                    field_name=field_name,
                    allowed_types=[str],
                    error=VariableTypeError,
                )

    def get_type(self) -> str:
        return self.type

    def get_input_names(self) -> Set[str]:
        return {"image", "text"}

    def get_output_names(self) -> Set[str]:
        return {"similarity", "parent_id", "predictions_type"}


class AggregationMode(Enum):
    AVERAGE = "average"
    MAX = "max"
    MIN = "min"


class DetectionsConsensus(BaseModel, StepInterface):
    model_config = ConfigDict(
        json_schema_extra={
            "block_type": "fusion",
            "description": "Combine predictions from multiple detection models to make a decision about object presence.",
            "docs": "https://inference.roboflow.com/workflows/reach_consensus",
        }
    )
    type: Literal["DetectionsConsensus"]
    name: str = Field(description="Unique name of step in workflows")
    predictions: List[
        StepOutputSelector(
            kind=[
                OBJECT_DETECTION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_PREDICTION_KIND,
                KEYPOINT_DETECTION_PREDICTION_KIND,
            ]
        ),
    ] = Field(
        min_items=1,
        description="Reference to detection-like model predictions made against single image to agree on model consensus",
        examples=[["$steps.a.predictions", "$steps.b.predictions"]],
    )
    required_votes: Union[
        PositiveInt, InferenceParameterSelector(kind=[INTEGER_KIND])
    ] = Field(
        description="Required number of votes for single detection from different models to accept detection as output detection",
        examples=[2, "$inputs.required_votes"],
    )
    class_aware: Union[bool, InferenceParameterSelector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Flag to decide if margin detections is class-aware or only bounding boxes aware",
        examples=[True, "$inputs.class_aware"],
    )
    iou_threshold: Union[
        FloatZeroToOne, InferenceParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        default=0.3,
        description="IoU threshold to consider detections from different models as matching (increasing votes for region)",
        examples=[0.3, "$inputs.iou_threshold"],
    )
    confidence: Union[
        FloatZeroToOne, InferenceParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        default=0.0,
        description="Confidence threshold for merged detections",
        examples=[0.1, "$inputs.confidence"],
    )
    classes_to_consider: Optional[
        Union[List[str], InferenceParameterSelector(kind=[LIST_OF_VALUES_KIND])]
    ] = Field(
        default=None,
        description="Optional list of classes to consider in consensus procedure.",
        examples=[["a", "b"], "$inputs.classes_to_consider"],
    )
    required_objects: Optional[
        Union[
            PositiveInt,
            Dict[str, PositiveInt],
            InferenceParameterSelector(kind=[INTEGER_KIND, DICTIONARY_KIND]),
        ]
    ] = Field(
        default=None,
        description="If given, it holds the number of objects that must be present in merged results, to assume that object presence is reached. Can be selector to `InferenceParameter`, integer value or dictionary with mapping of class name into minimal number of merged detections of given class to assume consensus.",
        examples=[3, {"a": 7, "b": 2}, "$inputs.required_objects"],
    )
    presence_confidence_aggregation: AggregationMode = Field(
        default=AggregationMode.MAX,
        description="Mode dictating aggregation of confidence scores and classes both in case of object presence deduction procedure.",
        examples=["max", "min"],
    )
    detections_merge_confidence_aggregation: AggregationMode = Field(
        default=AggregationMode.AVERAGE,
        description="Mode dictating aggregation of confidence scores and classes both in case of boxes consensus procedure. One of `average`, `max`, `min`. Default: `average`. While using for merging overlapping boxes, against classes - `average` equals to majority vote, `max` - for the class of detection with max confidence, `min` - for the class of detection with min confidence.",
        examples=["min", "max"],
    )
    detections_merge_coordinates_aggregation: AggregationMode = Field(
        default=AggregationMode.AVERAGE,
        description="Mode dictating aggregation of bounding boxes. One of `average`, `max`, `min`. Default: `average`. `average` means taking mean from all boxes coordinates, `min` - taking smallest box, `max` - taking largest box.",
        examples=["min", "max"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return super(DetectionsConsensus, cls).describe_outputs() + [
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(
                name="predictions",
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                    KEYPOINT_DETECTION_PREDICTION_KIND,
                ],
            ),
            OutputDefinition(name="image", kind=[IMAGE_METADATA_KIND]),
            OutputDefinition(
                name="object_present", kind=[BOOLEAN_KIND, DICTIONARY_KIND]
            ),
            OutputDefinition(
                name="presence_confidence",
                kind=[FLOAT_ZERO_TO_ONE_KIND, DICTIONARY_KIND],
            ),
            OutputDefinition(name="predictions_type", kind=[PREDICTION_TYPE_KIND]),
        ]

    def get_input_names(self) -> Set[str]:
        return {
            "predictions",
            "required_votes",
            "class_aware",
            "iou_threshold",
            "confidence",
            "classes_to_consider",
            "required_objects",
        }

    def get_output_names(self) -> Set[str]:
        return {
            "parent_id",
            "predictions",
            "image",
            "object_present",
            "presence_confidence",
            "predictions_type",
        }

    def validate_field_selector(
        self, field_name: str, input_step: GraphNone, index: Optional[int] = None
    ) -> None:
        if field_name != "predictions" and not is_selector(
            selector_or_value=getattr(self, field_name)
        ):
            raise ExecutionGraphError(
                f"Attempted to validate selector value for field {field_name}, but field is not selector."
            )
        if field_name == "predictions":
            if index is None or index > len(self.predictions):
                raise ExecutionGraphError(
                    f"Attempted to validate selector value for field {field_name}, which requires multiple inputs, "
                    f"but `index` not provided."
                )
            if not is_selector(
                selector_or_value=self.predictions[index],
            ):
                raise ExecutionGraphError(
                    f"Attempted to validate selector value for field {field_name}[{index}], but field is not selector."
                )
            validate_selector_holds_detection_predictions(
                step_name=self.name,
                image_selector=None,
                detections_selector=self.predictions[index],
                field_name=field_name,
                input_step=input_step,
                applicable_fields={"predictions"},
            )
            return None
        validate_selector_is_inference_parameter(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
            applicable_fields={
                "required_votes",
                "class_aware",
                "iou_threshold",
                "confidence",
                "classes_to_consider",
                "required_objects",
            },
        )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        if field_name == "required_votes":
            if value is None:
                raise VariableTypeError("Field `required_votes` cannot be None.")
            validate_value_is_empty_or_positive_number(
                value=value, field_name="required_votes", error=VariableTypeError
            )
        elif field_name == "class_aware":
            validate_field_has_given_type(
                field_name=field_name,
                allowed_types=[bool],
                value=value,
                error=VariableTypeError,
            )
        elif field_name in {"iou_threshold", "confidence"}:
            if value is None:
                raise VariableTypeError(f"Fields `{field_name}` cannot be None.")
            validate_value_is_empty_or_number_in_range_zero_one(
                value=value,
                field_name=field_name,
                error=VariableTypeError,
            )
        elif field_name == "classes_to_consider":
            if value is None:
                return None
            validate_field_is_list_of_string(
                value=value,
                field_name=field_name,
                error=VariableTypeError,
            )
        elif field_name == "required_objects":
            self._validate_required_objects_binding(value=value)
            return None

    def get_type(self) -> str:
        return self.type

    def _validate_required_objects_binding(self, value: Any) -> None:
        if value is None:
            return value
        validate_field_has_given_type(
            value=value,
            field_name="required_objects",
            allowed_types=[int, dict],
            error=VariableTypeError,
        )
        if issubclass(type(value), int):
            validate_value_is_empty_or_positive_number(
                value=value,
                field_name="required_objects",
                error=VariableTypeError,
            )
            return None
        for k, v in value.items():
            if v is None:
                raise VariableTypeError(
                    f"Field `required_objects[{k}]` must not be None."
                )
            validate_value_is_empty_or_positive_number(
                value=v,
                field_name=f"required_objects[{k}]",
                error=VariableTypeError,
            )


ACTIVE_LEARNING_DATA_COLLECTOR_ELIGIBLE_SELECTORS = {
    "ObjectDetectionModel": "predictions",
    "KeypointsDetectionModel": "predictions",
    "InstanceSegmentationModel": "predictions",
    "DetectionFilter": "predictions",
    "DetectionsConsensus": "predictions",
    "DetectionOffset": "predictions",
    "ClassificationModel": "top",
    "LMMForClassification": "top",
}


class DisabledActiveLearningConfiguration(BaseModel):
    enabled: Literal[False]


class LimitDefinition(BaseModel):
    type: Literal["minutely", "hourly", "daily"]
    value: PositiveInt


class RandomSamplingConfig(BaseModel):
    type: Literal["random"]
    name: str
    traffic_percentage: confloat(ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=lambda: [])
    limits: List[LimitDefinition] = Field(default_factory=lambda: [])


class CloseToThresholdSampling(BaseModel):
    type: Literal["close_to_threshold"]
    name: str
    probability: confloat(ge=0.0, le=1.0)
    threshold: confloat(ge=0.0, le=1.0)
    epsilon: confloat(ge=0.0, le=1.0)
    max_batch_images: Optional[int] = Field(default=None)
    only_top_classes: bool = Field(default=True)
    minimum_objects_close_to_threshold: int = Field(default=1)
    selected_class_names: Optional[List[str]] = Field(default=None)
    tags: List[str] = Field(default_factory=lambda: [])
    limits: List[LimitDefinition] = Field(default_factory=lambda: [])


class ClassesBasedSampling(BaseModel):
    type: Literal["classes_based"]
    name: str
    probability: confloat(ge=0.0, le=1.0)
    selected_class_names: List[str]
    tags: List[str] = Field(default_factory=lambda: [])
    limits: List[LimitDefinition] = Field(default_factory=lambda: [])


class DetectionsBasedSampling(BaseModel):
    type: Literal["detections_number_based"]
    name: str
    probability: confloat(ge=0.0, le=1.0)
    more_than: Optional[NonNegativeInt]
    less_than: Optional[NonNegativeInt]
    selected_class_names: Optional[List[str]] = Field(default=None)
    tags: List[str] = Field(default_factory=lambda: [])
    limits: List[LimitDefinition] = Field(default_factory=lambda: [])


class ActiveLearningBatchingStrategy(BaseModel):
    batches_name_prefix: str
    recreation_interval: Literal["never", "daily", "weekly", "monthly"]
    max_batch_images: Optional[int] = Field(default=None)


ActiveLearningStrategyType = Annotated[
    Union[
        RandomSamplingConfig,
        CloseToThresholdSampling,
        ClassesBasedSampling,
        DetectionsBasedSampling,
    ],
    Field(discriminator="type"),
]


class EnabledActiveLearningConfiguration(BaseModel):
    enabled: bool
    persist_predictions: bool
    sampling_strategies: List[ActiveLearningStrategyType]
    batching_strategy: ActiveLearningBatchingStrategy
    tags: List[str] = Field(default_factory=lambda: [])
    max_image_size: Optional[Tuple[PositiveInt, PositiveInt]] = Field(default=None)
    jpeg_compression_level: int = Field(default=95, gt=0, le=100)


class ActiveLearningDataCollector(BaseModel, StepInterface):
    model_config = ConfigDict(
        json_schema_extra={
            "block_type": "sink",
            "description": "Collect data and predictions that flow through workflows for use in active learning.",
            "docs": "https://inference.roboflow.com/workflows/active_learning",
        }
    )
    type: Literal["ActiveLearningDataCollector"]
    name: str = Field(description="Unique name of step in workflows")
    image: Union[InferenceImageSelector, OutputStepImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    predictions: StepOutputSelector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
            CLASSIFICATION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Reference to detection-like predictions",
        examples=["$steps.object_detection_model.predictions"],
    )
    target_dataset: Union[InferenceParameterSelector(kind=[ROBOFLOW_PROJECT]), str] = (
        Field(
            description="name of Roboflow dataset / project to be used as target for collected data",
            examples=["my_dataset", "$inputs.target_al_dataset"],
        )
    )
    target_dataset_api_key: Union[
        InferenceParameterSelector(kind=[STRING_KIND]), Optional[str]
    ] = Field(
        default=None,
        description="API key to be used for data registration. This may help in a scenario when data applicable for Universe models predictions to be saved in private workspaces and for models that were trained in the same workspace (not necessarily within the same project))",
    )
    disable_active_learning: Union[
        bool, InferenceParameterSelector(kind=[BOOLEAN_KIND])
    ] = Field(
        default=False,
        description="boolean flag that can be also reference to input - to arbitrarily disable data collection for specific request - overrides all AL config",
        examples=[True, "$inputs.disable_active_learning"],
    )
    active_learning_configuration: Optional[
        Union[EnabledActiveLearningConfiguration, DisabledActiveLearningConfiguration]
    ] = Field(
        default=None,
        description="Optional configuration of Active Learning data sampling in the exact format explained in Active Learning docs.",
    )

    def get_type(self) -> str:
        return self.type

    def get_input_names(self) -> Set[str]:
        return {
            "image",
            "predictions",
            "target_dataset",
            "target_dataset_api_key",
            "disable_active_learning",
        }

    def get_output_names(self) -> Set[str]:
        return set()

    def validate_field_selector(
        self, field_name: str, input_step: GraphNone, index: Optional[int] = None
    ) -> None:
        selector = getattr(self, field_name)
        if not is_selector(selector_or_value=selector):
            raise ExecutionGraphError(
                f"Attempted to validate selector value for field {field_name}, but field is not selector."
            )
        if field_name == "predictions":
            input_step_type = input_step.get_type()
            expected_last_selector_chunk = (
                ACTIVE_LEARNING_DATA_COLLECTOR_ELIGIBLE_SELECTORS.get(input_step_type)
            )
            if expected_last_selector_chunk is None:
                raise ExecutionGraphError(
                    f"Attempted to validate predictions selector of {self.name} step, but input step of type: "
                    f"{input_step_type} does match by type."
                )
            if get_last_selector_chunk(selector) != expected_last_selector_chunk:
                raise ExecutionGraphError(
                    f"It is only allowed to refer to {input_step_type} step output named {expected_last_selector_chunk}. "
                    f"Reference that was found: {selector}"
                )
            input_step_image = getattr(input_step, "image", self.image)
            if input_step_image != self.image:
                raise ExecutionGraphError(
                    f"ActiveLearningDataCollector step refers to input step that uses reference to different image. "
                    f"ActiveLearningDataCollector step image: {self.image}. Input step (of type {input_step_image}) "
                    f"uses {input_step_image}."
                )
        validate_selector_holds_image(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
        )
        validate_selector_is_inference_parameter(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
            applicable_fields={
                "target_dataset",
                "target_dataset_api_key",
                "disable_active_learning",
            },
        )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        if field_name == "image":
            validate_image_biding(value=value)
        elif field_name in {"disable_active_learning"}:
            validate_field_has_given_type(
                field_name=field_name,
                allowed_types=[bool],
                value=value,
                error=VariableTypeError,
            )
        elif field_name in {"target_dataset"}:
            validate_field_has_given_type(
                field_name=field_name,
                allowed_types=[str],
                value=value,
                error=VariableTypeError,
            )
        elif field_name in {"target_dataset_api_key"}:
            validate_field_has_given_type(
                field_name=field_name,
                allowed_types=[str],
                value=value,
                error=VariableTypeError,
            )


class YoloWorldModel(BaseModel, StepInterface):
    model_config = ConfigDict(
        json_schema_extra={
            "block_type": "model",
            "description": "Run a zero-shot object detection model.",
            "docs": "https://inference.roboflow.com/workflows/yolo_world",
        }
    )
    type: Literal["YoloWorldModel", "YoloWorld"]
    name: str = Field(description="Unique name of step in workflows")
    image: Union[InferenceImageSelector, OutputStepImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    class_names: Union[
        InferenceParameterSelector(kind=[LIST_OF_VALUES_KIND]), List[str]
    ] = Field(
        description="List of classes to use YoloWorld model against",
        examples=[["a", "b", "c"], "$inputs.class_names"],
    )
    version: Union[
        Literal["s", "m", "l", "x", "v2-s", "v2-m", "v2-l", "v2-x"],
        InferenceParameterSelector(kind=[STRING_KIND]),
    ] = Field(
        default="l",
        description="Variant of YoloWorld model",
        examples=["l", "$inputs.variant"],
    )
    confidence: Union[
        Optional[FloatZeroToOne],
        InferenceParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.4,
        description="Confidence threshold for detections",
        examples=[0.3, "$inputs.confidence"],
    )

    def get_input_names(self) -> Set[str]:
        return {"image", "class_names", "version", "confidence"}

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return super(YoloWorldModel, cls).describe_outputs() + [
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(
                name="predictions", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            ),
            OutputDefinition(name="image", kind=[IMAGE_METADATA_KIND]),
            OutputDefinition(name="predictions_type", kind=[PREDICTION_TYPE_KIND]),
        ]

    def get_output_names(self) -> Set[str]:
        return {"predictions", "parent_id", "image", "prediction_type"}

    def validate_field_selector(
        self, field_name: str, input_step: GraphNone, index: Optional[int] = None
    ) -> None:
        selector = getattr(self, field_name)
        if not is_selector(selector_or_value=selector):
            raise ExecutionGraphError(
                f"Attempted to validate selector value for field {field_name}, but field is not selector."
            )
        validate_selector_holds_image(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
        )
        validate_selector_is_inference_parameter(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
            applicable_fields={"class_names", "version", "confidence"},
        )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        if field_name == "image":
            validate_image_biding(value=value)
        elif field_name == "class_names":
            validate_field_is_list_of_string(
                value=value,
                field_name=field_name,
                error=VariableTypeError,
            )
        elif field_name == "version":
            validate_field_is_one_of_selected_values(
                value=value,
                field_name=field_name,
                selected_values={None, "s", "m", "l"},
                error=VariableTypeError,
            )
        elif field_name == "confidence":
            validate_value_is_empty_or_number_in_range_zero_one(
                value=value,
                field_name=field_name,
                error=VariableTypeError,
            )

    def get_type(self) -> str:
        return self.type


GPT_4V_MODEL_TYPE = "gpt_4v"
COG_VLM_MODEL_TYPE = "cog_vlm"
SUPPORTED_LMMS = {GPT_4V_MODEL_TYPE, COG_VLM_MODEL_TYPE}


class LMMConfig(BaseModel):
    max_tokens: int = Field(default=450)
    gpt_image_detail: Literal["low", "high", "auto"] = Field(
        default="auto",
        description="To be used for GPT-4V only.",
    )
    gpt_model_version: str = Field(default="gpt-4-vision-preview")


class LMM(BaseModel, StepInterface):
    model_config = ConfigDict(
        json_schema_extra={
            "block_type": "model",
            "description": "Run a large language model.",
            "docs": "https://inference.roboflow.com/workflows/use_lmm",
        }
    )
    type: Literal["LMM"]
    name: str = Field(description="Unique name of step in workflows")
    image: Union[InferenceImageSelector, OutputStepImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    prompt: Union[InferenceParameterSelector(kind=[STRING_KIND]), str] = Field(
        description="Holds unconstrained text prompt to LMM mode",
        examples=["my prompt", "$inputs.prompt"],
    )
    lmm_type: Union[
        InferenceParameterSelector(kind=[STRING_KIND]), Literal["gpt_4v", "cog_vlm"]
    ] = Field(
        description="Type of LMM to be used", examples=["gpt_4v", "$inputs.lmm_type"]
    )
    lmm_config: LMMConfig = Field(
        default_factory=lambda: LMMConfig(), description="Configuration of LMM"
    )
    remote_api_key: Union[
        InferenceParameterSelector(kind=[STRING_KIND]), Optional[str]
    ] = Field(
        default=None,
        description="Holds API key required to call LMM model - in current state of development, we require OpenAI key when `lmm_type=gpt_4v` and do not require additional API key for CogVLM calls.",
        examples=["xxx-xxx", "$inputs.api_key"],
    )
    json_output: Optional[
        Union[InferenceParameterSelector(kind=[DICTIONARY_KIND]), Dict[str, str]]
    ] = Field(
        default=None,
        description="Holds dictionary that maps name of requested output field into its description",
        examples=[{"count": "number of cats in the picture"}, "$inputs.json_output"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return super(LMM, cls).describe_outputs() + [
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="image", kind=[IMAGE_METADATA_KIND]),
            OutputDefinition(name="structured_output", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="raw_output", kind=[STRING_KIND]),
            OutputDefinition(name="*", kind=[WILDCARD_KIND]),
        ]

    @field_validator("json_output")
    @classmethod
    def validate_json_output(cls, value: Any) -> str:
        validate_field_is_selector_or_has_given_type(
            value=value, field_name="json_output", allowed_types=[type(None), dict]
        )
        if not issubclass(type(value), dict):
            return value
        validate_field_is_dict_of_strings(
            value=value,
            field_name="json_output",
        )
        output_names = {"raw_output", "structured_output", "image", "parent_id"}
        for key in value.keys():
            if key in output_names:
                raise ValueError(
                    f"`json_output` specified for `LMM` step defines field (`{key}`) that collide with step "
                    f"output names: {output_names} which is forbidden."
                )
        return value

    def get_input_names(self) -> Set[str]:
        return {
            "image",
            "prompt",
            "lmm_type",
            "remote_api_key",
            "json_output",
        }

    def get_output_names(self) -> Set[str]:
        outputs = {"raw_output", "structured_output", "image", "parent_id"}
        if issubclass(type(self.json_output), dict):
            outputs.update(self.json_output.keys())
        return outputs

    def validate_field_selector(
        self, field_name: str, input_step: GraphNone, index: Optional[int] = None
    ) -> None:
        selector = getattr(self, field_name)
        if not is_selector(selector_or_value=selector):
            raise ExecutionGraphError(
                f"Attempted to validate selector value for field {field_name}, but field is not selector."
            )
        validate_selector_holds_image(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
        )
        validate_selector_is_inference_parameter(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
            applicable_fields={
                "prompt",
                "lmm_type",
                "remote_api_key",
                "json_output",
            },
        )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        if field_name == "image":
            validate_image_biding(value=value)
        elif field_name == "prompt":
            validate_field_has_given_type(
                field_name=field_name,
                allowed_types=[str],
                value=value,
                error=VariableTypeError,
            )
        elif field_name == "lmm_type":
            validate_field_is_one_of_selected_values(
                field_name=field_name,
                selected_values=SUPPORTED_LMMS,
                value=value,
                error=VariableTypeError,
            )
        elif field_name == "remote_api_key":
            validate_field_has_given_type(
                field_name=field_name,
                allowed_types=[str, type(None)],
                value=value,
                error=VariableTypeError,
            )
        elif field_name == "json_output":
            validate_field_has_given_type(
                field_name=field_name,
                allowed_types=[type(None), dict],
                value=value,
                error=VariableTypeError,
            )
            if value is None:
                return None
            validate_field_is_dict_of_strings(
                value=value,
                field_name="json_output",
                error=VariableTypeError,
            )
            output_names = {"raw_output", "structured_output", "image", "parent_id"}
            for key in value.keys():
                if key in output_names:
                    raise VariableTypeError(
                        f"`json_output` injected for `LMM` step {self.name} defines field (`{key}`) that collide "
                        f"with step output names: {output_names} which is forbidden."
                    )

    def get_type(self) -> str:
        return self.type


class LMMForClassification(BaseModel, StepInterface):
    model_config = ConfigDict(
        json_schema_extra={
            "block_type": "model",
            "description": "Run a large language model for classification.",
            "docs": "https://inference.roboflow.com/workflows/use_lmm_classification",
        }
    )
    type: Literal["LMMForClassification"]
    name: str = Field(description="Unique name of step in workflows")
    image: Union[InferenceImageSelector, OutputStepImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    lmm_type: Union[
        InferenceParameterSelector(kind=[STRING_KIND]), Literal["gpt_4v", "cog_vlm"]
    ] = Field(
        description="Type of LMM to be used", examples=["gpt_4v", "$inputs.lmm_type"]
    )
    classes: Union[
        List[str], InferenceParameterSelector(kind=[LIST_OF_VALUES_KIND])
    ] = Field(
        description="List of classes that LMM shall classify against",
        examples=[["a", "b"], "$inputs.classes"],
    )
    lmm_config: LMMConfig = Field(
        default_factory=lambda: LMMConfig(), description="Configuration of LMM"
    )
    remote_api_key: Union[
        InferenceParameterSelector(kind=[STRING_KIND]), Optional[str]
    ] = Field(
        default=None,
        description="Holds API key required to call LMM model - in current state of development, we require OpenAI key when `lmm_type=gpt_4v` and do not require additional API key for CogVLM calls.",
        examples=["xxx-xxx", "$inputs.api_key"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return super(LMMForClassification, cls).describe_outputs() + [
            OutputDefinition(name="raw_output", kind=[STRING_KIND]),
            OutputDefinition(name="top", kind=[STRING_KIND]),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="image", kind=[IMAGE_METADATA_KIND]),
            OutputDefinition(name="prediction_type", kind=[PREDICTION_TYPE_KIND]),
        ]

    def get_input_names(self) -> Set[str]:
        return {
            "image",
            "lmm_type",
            "classes",
            "remote_api_key",
        }

    def get_output_names(self) -> Set[str]:
        return {"raw_output", "top", "parent_id", "image", "prediction_type"}

    def validate_field_selector(
        self, field_name: str, input_step: GraphNone, index: Optional[int] = None
    ) -> None:
        selector = getattr(self, field_name)
        if not is_selector(selector_or_value=selector):
            raise ExecutionGraphError(
                f"Attempted to validate selector value for field {field_name}, but field is not selector."
            )
        validate_selector_holds_image(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
        )
        validate_selector_is_inference_parameter(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
            applicable_fields={
                "lmm_type",
                "classes",
                "remote_api_key",
            },
        )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        if field_name == "image":
            validate_image_biding(value=value)
        elif field_name == "lmm_type":
            validate_field_is_one_of_selected_values(
                field_name=field_name,
                selected_values=SUPPORTED_LMMS,
                value=value,
                error=VariableTypeError,
            )
        elif field_name == "remote_api_key":
            validate_field_has_given_type(
                field_name=field_name,
                allowed_types=[str, type(None)],
                value=value,
                error=VariableTypeError,
            )
        elif field_name == "classes":
            validate_field_is_list_of_string(
                field_name=field_name,
                value=value,
                error=VariableTypeError,
            )
            if len(value) == 0:
                raise VariableTypeError(
                    f"Cannot bind empty list of classes to `classes` field of {self.name} step."
                )

    def get_type(self) -> str:
        return self.type
