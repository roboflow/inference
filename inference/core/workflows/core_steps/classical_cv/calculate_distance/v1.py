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
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    StepOutputImageSelector,
    StepOutputSelector,
    WorkflowImageSelector,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    WorkflowParameterSelector,
    FLOAT_KIND,
    STRING_KIND

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

    
    predictions: StepOutputSelector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(
        title="Object Detections",
        description="The output of a detection model describing the bounding boxes that will be used to measure the objects.",
        examples=["$steps.model.predictions"],
        validation_alias=AliasChoices("predictions", "detections"),
    )
    
    object_1_class_name: str = Field(
        title="First Object Class Name",
        description="The class name of the first object.",
        examples=["car"],
    )
    
    object_2_class_name: str = Field(
        title="Second Object Class Name",
        description="The class name of the second object.",
        examples=["person"],
    )
    
    reference_axis: Literal["horizontal", "vertical"] = Field(
        title="Reference Axis",
        description="The axis along which the distance will be measured.",
        examples=["vertical", "horizontal"],
    )
    
    calibration_type: Literal["reference object", "pixel-ratio"] = Field(
        title="Calibration Method",
        description="Select how to calibrate the measurement of distance between objects.",
    )
    
    reference_predictions: StepOutputSelector(
        kind=[
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            OBJECT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        title="Reference Object",
        description="Predictions from the reference object model",
        examples=["$steps.model.reference_predictions"],
        # json_schema_extra={
        #     # "relevant_for": {
        #     #     "calibration_type": {
        #     #         "values": ["reference object"],
        #     #         "required": True,
        #     #     },
        #     # },
        # },
    )
    
    # reference_object_class_name: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = Field( 
    #     title="Reference Object Class Name",
    #     description="The class name of the reference object.",
    #     examples=["marker", "$inputs.reference_object_class_name"],
    #     json_schema_extra={
    #         "relevant_for": {
    #             "calibration_type": {
    #                 "values": ["reference object"],
    #                 "required": True,
    #             },
    #         },
    #     },
    # )
    
    reference_width: Union[int, WorkflowParameterSelector(kind=[FLOAT_KIND])] = Field( 
        title="Width",
        default=2.5,
        description="Width of the reference object in centimeters",
        examples=[2.5, "$inputs.reference_width"],
        gt=0,
        json_schema_extra={
            "relevant_for": {
                "calibration_type": {
                    "values": ["reference object"],
                    "required": True,
                },
            },
        },
    )
    
    reference_height: Union[int, WorkflowParameterSelector(kind=[FLOAT_KIND])] = Field(  # type: ignore
        title="Height",
        default=2.5,
        description="Height of the reference object in centimeters",
        examples=[2.5, "$inputs.reference_height"],
        gt=0,
        json_schema_extra={
            "relevant_for": {
                "calibration_type": {
                    "values": ["reference object"],
                    "required": True,
                },
            },
        },
    )
    
    pixel_ratio: Union[float, WorkflowParameterSelector(kind=[FLOAT_KIND])]  = Field(
        title="Reference Pixel-to-Centimeter Ratio",
        description="The pixel-to-centimeter ratio of the input image, i.e. 100 pixels = 1 centimeter.",
        examples=[100, "$inputs.resize_height"],
        gt=0,
        json_schema_extra={
            "relevant_for": {
                "calibration_type": {
                    "values": ["pixel-ratio"],
                    "required": True,
                },
            },
        },
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
        object_1_class_name: str,
        object_2_class_name: str,
        reference_axis: Literal["horizontal", "vertical"],
        #calibration_type: Literal["reference object", "pixel-ratio"],
        #reference_predictions: Batch[sv.Detections],
        #reference_width: float,
        #reference_height: float,
        pixel_ratio: float,
    ) -> BlockResult:
        results = []
        for detections in predictions:
            measurements = calculate_distance_pixel_ratio(
                #image=image,
                detections=detections,
                pixel_ratio=pixel_ratio,
                object_1_class_name=object_1_class_name,
                object_2_class_name=object_2_class_name,
                reference_axis=reference_axis,
            )
            results.append([{"measurements": measurements}])
        return results

def calculate_distance_pixel_ratio(
    #image: WorkflowImageData,
    detections: sv.Detections,
    pixel_ratio: float,
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
    
    distance_cm = distance_pixels / pixel_ratio
    
    return {"distance_cm": distance_cm, "distance_pixels": distance_pixels}
