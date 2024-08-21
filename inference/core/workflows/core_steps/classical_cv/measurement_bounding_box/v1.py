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
    BATCH_OF_DICTIONARY_KIND,
    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
    StepOutputImageSelector,
    StepOutputSelector,
    WorkflowImageSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Calculate measurements of detected objects in an image using a reference bounding box to establish a pixels-to-inch ratio.
"""

class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Measurement Bounding Box",
            "version": "v1",
            "short_description": "Measure objects in an image using a reference bounding box.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
        }
    )
    type: Literal["roboflow_core/measurement_bounding_box@v1", "MeasurementBoundingBox", "Measure"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        title="Image to Measure",
        description="The input image for this step.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("images", "image"),
    )
    predictions: StepOutputSelector(
        kind=[
            BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        title="Regions of Interest",
        description="The output of a detection model describing the bounding boxes that will be used to measure the objects.",
        examples=["$steps.my_object_detection_model.predictions"],
        validation_alias=AliasChoices("predictions", "detections"),
    )
    reference_class: str = Field(
        title="Reference Class",
        description="The class name of the reference object used to calculate the pixels-to-inch ratio.",
        examples=["reference_object"],
    )
    reference_inches: float = Field(
        title="Reference Inches",
        description="The actual size in inches of the reference object.",
        examples=[1.0],
    )
    reference_dimension: Literal["width", "height"] = Field(
        title="Reference Dimension",
        description="The dimension of the reference object to use for calculating the pixels-to-inch ratio.",
        examples=["width", "height"],
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
            OutputDefinition(name="measurements", kind=[BATCH_OF_DICTIONARY_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class MeasurementBoundingBoxBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        predictions: Batch[sv.Detections],
        reference_class: str,
        reference_inches: float,
        reference_dimension: Literal["width", "height"],
    ) -> BlockResult:
        results = []
        for image, detections in zip(images, predictions):
            measurements = measure_objects(
                image=image,
                detections=detections,
                reference_class=reference_class,
                reference_inches=reference_inches,
                reference_dimension=reference_dimension
            )
            results.append([{"measurements": measurements}])
        return results

def measure_objects(
    image: WorkflowImageData,
    detections: sv.Detections,
    reference_class: str,
    reference_inches: float,
    reference_dimension: Literal["width", "height"],
) -> List[Dict[str, Union[str, float]]]:
    reference_bbox = None
    for (x_min, y_min, x_max, y_max), class_name in zip(
        detections.xyxy.round().astype(dtype=int), detections.data['class_name']
    ):
        if class_name == reference_class:
            reference_bbox = (x_min, y_min, x_max, y_max)
            break

    if not reference_bbox:
        raise ValueError(f"Reference class '{reference_class}' not found in predictions.")

    if reference_dimension == "width":
        ref_size = reference_bbox[2] - reference_bbox[0]
    else:
        ref_size = reference_bbox[3] - reference_bbox[1]

    pixels_per_inch = ref_size / reference_inches

    measurements = []
    for (x_min, y_min, x_max, y_max) in detections.xyxy.round().astype(dtype=int):
        width = (x_max - x_min) / pixels_per_inch
        height = (y_max - y_min) / pixels_per_inch
        measurements.append({
            "width_inches": width,
            "height_inches": height,
        })
    
    return measurements

