from copy import deepcopy
from typing import Any, Dict, List, Literal, Tuple, Type, Union
from uuid import uuid4

from pydantic import AliasChoices, ConfigDict, Field, PositiveInt
from typing_extensions import Annotated

from inference.core.workflows.constants import (
    DETECTION_ID_KEY,
    HEIGHT_KEY,
    PARENT_ID_KEY,
    WIDTH_KEY,
)
from inference.core.workflows.entities.base import OutputDefinition
from inference.core.workflows.entities.types import (
    BATCH_OF_IMAGE_METADATA_KIND,
    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
    BATCH_OF_PARENT_ID_KIND,
    BATCH_OF_PREDICTION_TYPE_KIND,
    INTEGER_KIND,
    FlowControl,
    StepOutputSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Apply a fixed offset to the width and height of a detection.

You can use this block to add padding around the result of a detection. This is useful 
to ensure that you can analyze bounding boxes that may be within the region of an 
object instead of being around an object.
"""

SHORT_DESCRIPTION = "Apply a fixed offset on the width and height of detections."


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
        }
    )
    type: Literal["DetectionOffset"]
    name: str = Field(description="Unique name of step in workflows")
    predictions: StepOutputSelector(
        kind=[
            BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
            BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
            BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Reference to detection-like predictions",
        examples=["$steps.object_detection_model.predictions"],
    )
    offset_width: Union[PositiveInt, WorkflowParameterSelector(kind=[INTEGER_KIND])] = (
        Field(
            description="Offset for boxes width",
            examples=[10, "$inputs.offset_x"],
            validation_alias=AliasChoices("offset_width", "offset_x"),
        )
    )
    offset_height: Union[
        PositiveInt, WorkflowParameterSelector(kind=[INTEGER_KIND])
    ] = Field(
        description="Offset for boxes height",
        examples=[10, "$inputs.offset_y"],
        validation_alias=AliasChoices("offset_height", "offset_y"),
    )
    image_metadata: Annotated[
        StepOutputSelector(kind=[BATCH_OF_IMAGE_METADATA_KIND]),
        Field(
            description="Metadata of image used to create `predictions`. Must be output from the step referred in `predictions` field",
            examples=["$steps.detection.image"],
        ),
    ]
    prediction_type: Annotated[
        StepOutputSelector(kind=[BATCH_OF_PREDICTION_TYPE_KIND]),
        Field(
            description="Type of `predictions`. Must be output from the step referred in `predictions` field",
            examples=["$steps.detection.prediction_type"],
        ),
    ]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions",
                kind=[
                    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
                    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
                    BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
                ],
            ),
            OutputDefinition(name="image", kind=[BATCH_OF_IMAGE_METADATA_KIND]),
            OutputDefinition(name="parent_id", kind=[BATCH_OF_PARENT_ID_KIND]),
            OutputDefinition(
                name="prediction_type", kind=[BATCH_OF_PREDICTION_TYPE_KIND]
            ),
        ]


class DetectionOffsetBlock(WorkflowBlock):
    # TODO: This block breaks parent coordinates.
    # Issue report: https://github.com/roboflow/inference/issues/380

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    async def run_locally(
        self,
        predictions: List[List[dict]],
        offset_width: int,
        offset_height: int,
        image_metadata: List[dict],
        prediction_type: List[str],
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        result_predictions, result_parent_id = [], []
        for detections in predictions:
            offset_detections = [
                offset_detection(
                    detection=detection,
                    offset_width=offset_width,
                    offset_height=offset_height,
                )
                for detection in detections
            ]
            result_predictions.append(offset_detections)
            result_parent_id.append(
                [detection[PARENT_ID_KEY] for detection in offset_detections]
            )
        return [
            {
                "predictions": prediction,
                PARENT_ID_KEY: parent_id,
                "image": image,
                "prediction_type": single_prediction_type,
            }
            for prediction, parent_id, image, single_prediction_type in zip(
                result_predictions, result_parent_id, image_metadata, prediction_type
            )
        ]


def offset_detection(
    detection: Dict[str, Any], offset_width: int, offset_height: int
) -> Dict[str, Any]:
    detection_copy = deepcopy(detection)
    detection_copy[WIDTH_KEY] += round(offset_width)
    detection_copy[HEIGHT_KEY] += round(offset_height)
    detection_copy[PARENT_ID_KEY] = detection_copy[DETECTION_ID_KEY]
    detection_copy[DETECTION_ID_KEY] = str(uuid4())
    return detection_copy
