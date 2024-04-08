from copy import deepcopy
from typing import Annotated, Any, Callable, Dict, List, Literal, Tuple, Type, Union

from pydantic import BaseModel, ConfigDict, Field

from inference.enterprise.workflows.complier.steps_executors.constants import (
    PARENT_ID_KEY,
)
from inference.enterprise.workflows.core_steps.common.operators import (
    BINARY_OPERATORS_FUNCTIONS,
    OPERATORS_FUNCTIONS,
    BinaryOperator,
    Operator,
)
from inference.enterprise.workflows.entities.steps import OutputDefinition
from inference.enterprise.workflows.entities.types import (
    IMAGE_METADATA_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    PARENT_ID_KIND,
    PREDICTION_TYPE_KIND,
    FlowControl,
    StepOutputSelector,
)
from inference.enterprise.workflows.errors import ExecutionGraphError
from inference.enterprise.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)


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


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "This block is responsible for filtering detections-based predictions based on conditions defined.",
            "docs": "https://inference.roboflow.com/workflows/filter_detections",
            "block_type": "transformation",
        }
    )
    type: Literal["DetectionFilter"]
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
    image_metadata: Annotated[
        StepOutputSelector(kind=[IMAGE_METADATA_KIND]),
        Field(
            description="Metadata of image used to create `predictions`. Must be output from the step referred in `predictions` field",
            examples=["$steps.detection.image"],
        ),
    ]
    prediction_type: Annotated[
        Union[StepOutputSelector(kind=[PREDICTION_TYPE_KIND])],
        Field(
            description="Type of `predictions`. Must be output from the step referred in `predictions` field",
            examples=["$steps.detection.prediction_type"],
        ),
    ]


class DetectionFilterBlock(WorkflowBlock):

    @classmethod
    def get_input_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
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

    async def run_locally(
        self,
        predictions: List[List[dict]],
        filter_definition: Union[
            DetectionFilterDefinition, CompoundDetectionFilterDefinition
        ],
        image_metadata: List[dict],
        prediction_type: List[str],
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        filter_callable = build_filter_callable(definition=filter_definition)
        result_detections, result_parent_id = [], []
        for prediction in predictions:
            filtered_predictions = [
                deepcopy(p) for p in prediction if filter_callable(p)
            ]
            result_detections.append(filtered_predictions)
            result_parent_id.append([p[PARENT_ID_KEY] for p in filtered_predictions])
        return [
            {
                "predictions": d,
                PARENT_ID_KEY: p,
                "image": i,
                "prediction_type": pt,
            }
            for d, p, i, pt in zip(
                result_detections, result_parent_id, image_metadata, prediction_type
            )
        ]


def build_filter_callable(
    definition: Union[DetectionFilterDefinition, CompoundDetectionFilterDefinition],
) -> Callable[[dict], bool]:
    if definition.type == "CompoundDetectionFilterDefinition":
        left_callable = build_filter_callable(definition=definition.left)
        right_callable = build_filter_callable(definition=definition.right)
        binary_operator = BINARY_OPERATORS_FUNCTIONS[definition.operator]
        return lambda e: binary_operator(left_callable(e), right_callable(e))
    if definition.type == "DetectionFilterDefinition":
        operator = OPERATORS_FUNCTIONS[definition.operator]
        return lambda e: operator(e[definition.field_name], definition.reference_value)
    raise ExecutionGraphError(
        f"Detected filter definition of type {definition.type} which is unknown"
    )
