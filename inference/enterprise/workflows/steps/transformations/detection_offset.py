from copy import deepcopy
from typing import Annotated, Any, Callable, Dict, List, Literal, Tuple, Type, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from inference.enterprise.workflows.complier.steps_executors.constants import (
    DETECTION_ID_KEY,
    HEIGHT_KEY,
    PARENT_ID_KEY,
    WIDTH_KEY,
)
from inference.enterprise.workflows.entities.steps import OutputDefinition
from inference.enterprise.workflows.entities.types import (
    IMAGE_METADATA_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    PARENT_ID_KIND,
    PREDICTION_TYPE_KIND,
    FlowControl,
    InferenceParameterSelector,
    StepOutputSelector,
)


class BlockManifest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "This block is responsible for applying fixed offset on width and height of detections.",
            "docs": "https://inference.roboflow.com/workflows/offset_detections",
            "block_type": "transformation",
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
    image_metadata: Annotated[
        Union[StepOutputSelector(kind=IMAGE_METADATA_KIND),],
        Field(
            description="Metadata of image used to create `predictions`. Must be output from the step referred in `predictions` field",
            examples=["$steps.detection.image"],
        ),
    ]
    prediction_type: Annotated[
        Union[StepOutputSelector(kind=PREDICTION_TYPE_KIND),],
        Field(
            description="Type of `predictions`. Must be output from the step referred in `predictions` field",
            examples=["$steps.detection.prediction_type"],
        ),
    ]


class DetectionOffsetBlock:

    @classmethod
    def get_input_manifest(cls) -> Type[BaseModel]:
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
        predictions: List[dict],
        offset_x: int,
        offset_y: int,
        image_metadata: List[dict],
        prediction_type: List[str],
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        result_detections, result_parent_id = [], []
        for detection in predictions:
            offset_detections = [
                offset_detection(detection=d, offset_x=offset_x, offset_y=offset_y)
                for d in detection
            ]
            result_detections.append(offset_detections)
            result_parent_id.append([d[PARENT_ID_KEY] for d in offset_detections])
        return [
            {"predictions": d, PARENT_ID_KEY: p, "image": i, "prediction_type": pt}
            for d, p, i, pt in zip(
                result_detections, result_parent_id, image_metadata, prediction_type
            )
        ]


def offset_detection(
    detection: Dict[str, Any], offset_x: int, offset_y: int
) -> Dict[str, Any]:
    detection_copy = deepcopy(detection)
    detection_copy[WIDTH_KEY] += round(offset_x)
    detection_copy[HEIGHT_KEY] += round(offset_y)
    detection_copy[PARENT_ID_KEY] = detection_copy[DETECTION_ID_KEY]
    detection_copy[DETECTION_ID_KEY] = str(uuid4())
    return detection_copy
