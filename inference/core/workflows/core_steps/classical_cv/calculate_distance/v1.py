from dataclasses import replace
from typing import Dict, List, Literal, Optional, Type, Union
import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.workflows.execution_engine.constants import DETECTION_ID_KEY
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    OriginCoordinatesSystem,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    DICTIONARY_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    StepOutputImageSelector,
    StepOutputSelector,
    WorkflowImageSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION = "Calculate the distance between two objects"

LONG_DESCRIPTION = """
Calculate the distance between two objects in an image using a pixel-to-milimeter ratio."""

class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Calculate Distance",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
        }
    )
    type: Literal["roboflow_core/calculate_distance@v1", "CalculateDistance", "Distance"]
    # images: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
    #     title="Image to Measure",
    #     description="The input image for this step.",
    #     examples=["$inputs.image", "$steps.cropping.crops"],
    #     validation_alias=AliasChoices("images", "image"),
    # )
    predictions: StepOutputSelector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        title="Object Detections",
        description="The output of a detection model describing the bounding boxes that will be used to measure the objects.",
        examples=["$steps.my_object_detection_model.predictions"],
        validation_alias=AliasChoices("predictions", "detections"),
    )
    reference_milimeters: float = Field(
        title="Reference Pixel-to-Militimeter Ratio",
        description="The pixel-to-milimeter ratio of the input image, i.e. 10 pixels = 1 milimeter.",
        examples=[10.0],
    )
    object_1_class_name: str = Field(
        title="First Object Class Name",
        description="The class name of the first object.",
        examples=["car"],
    )
    
    object_2_class_name: str = Field(
        title="Second Class Object 2",
        description="The class name of the second object.",
        examples=["person"],
    )
    
    
    reference_axis: Literal["horizontal", "vertical"] = Field(
        title="Reference Axis",
        description="The axis along which the distance will be measured.",
        examples=["vertical", "horizontal"],
    )

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def get_output_dimensionality_offset(cls) -> int:
        return 1

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="measurements", kind=[DICTIONARY_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class CalculateDistanceBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        #images: Batch[WorkflowImageData],
        predictions: Batch[sv.Detections],
        reference_milimeters: float,
        object_1_class_name: str,
        object_2_class_name: str,
        reference_axis: Literal["horizontal", "vertical"],
    ) -> BlockResult:
        results = []
        for detections in predictions:
            measurements = calculate_distance(
                #image=image,
                detections=detections,
                reference_milimeters=reference_milimeters,
                object_1_class_name=object_1_class_name,
                object_2_class_name=object_2_class_name,
                reference_axis=reference_axis,
            )
            results.append([{"measurements": measurements}])
        return results

def calculate_distance(
    #image: WorkflowImageData,
    detections: sv.Detections,
    reference_milimeters: float,
    object_1_class_name: str,
    object_2_class_name: str,
    reference_axis: Literal["horizontal", "vertical"],
) -> List[Dict[str, Union[str, float]]]:
    reference_bbox_1 = None
    reference_bbox_2 = None
    for (x_min, y_min, x_max, y_max), class_name in zip(
        detections.xyxy.round().astype(dtype=int), detections.data['class_name']
    ):
        print(class_name)
        if class_name == object_1_class_name:
            reference_bbox_1 = (x_min, y_min, x_max, y_max)
        elif class_name == object_2_class_name:
            reference_bbox_2 = (x_min, y_min, x_max, y_max)

        if reference_bbox_1 and reference_bbox_2:
            break

    if not reference_bbox_1 or not reference_bbox_2:
        raise ValueError(f"Reference class '{object_1_class_name}' or '{object_2_class_name}' not found in predictions.")

    if reference_axis == "vertical":
        distance_pixels = max(0, reference_bbox_2[1] - reference_bbox_1[3])
    else:
        distance_pixels = max(0, reference_bbox_2[0] - reference_bbox_1[2])

    print(f"Distance in pixels: {distance_pixels}")
    
    distance_mm = distance_pixels / reference_milimeters
    
    return {"distance_mm": distance_mm, "distance_pixels": distance_pixels}
