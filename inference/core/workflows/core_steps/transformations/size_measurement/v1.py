from typing import List, Literal, Optional, Type, Union, Tuple
import cv2 as cv
import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field, validator
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
)
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    LIST_OF_VALUES_KIND,
    StepOutputSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY: str = "dimensions"
TYPE: str = "roboflow_core/size_measurement@v1"
SHORT_DESCRIPTION = "Measure the dimensions of objects in relation to a reference object."
LONG_DESCRIPTION = """
The `SizeMeasurementBlock` is a transformer block designed to measure the dimensions of objects
in relation to a reference object. The reference object is detected using one model,
and the object to be measured is detected using another model. The block outputs the dimensions of the
objects to be measured in terms of the reference object.
"""

class SizeMeasurementManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Size Measurement",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
        }
    )
    type: Literal[f"{TYPE}", "SizeMeasurement"]
    reference_predictions: StepOutputSelector(
        kind=[
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            OBJECT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Predictions from the reference object model",
        examples=["$segmentation.reference_predictions"],
    )
    object_predictions: StepOutputSelector(
        kind=[
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            OBJECT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Predictions from the model that detects the object to measure",
        examples=["$segmentation.object_predictions"],
    )
    reference_dimensions: Union[str, Tuple[float, float]] = Field(
        description="Dimensions of the reference object (width, height) in desired units (e.g., inches) as a string in the format 'width,height' or as a tuple (width, height)",
        examples=["5.0,5.0", (5.0, 5.0)],
    )

    @validator("reference_dimensions", pre=True)
    def parse_reference_dimensions(cls, value: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
        if isinstance(value, str):
            try:
                width, height = map(float, value.split(","))
                return width, height
            except ValueError:
                raise ValueError("reference_dimensions must be a string in the format 'width,height'")
        elif isinstance(value, tuple) and len(value) == 2:
            return value
        else:
            raise ValueError("reference_dimensions must be a string in the format 'width,height' or a tuple (width, height)")

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=OUTPUT_KEY, kind=[LIST_OF_VALUES_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


def get_detection_dimensions(detection: sv.Detections, index: int) -> Tuple[float, float]:
    if detection.mask is not None:
        mask = detection.mask[index].astype(np.uint8)
        x, y, w, h = cv.boundingRect(mask)
    else:
        bbox = detection.xyxy[index]
        x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
    return w, h

class SizeMeasurementBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return SizeMeasurementManifest
    def run(
        self,
        reference_predictions: Batch[sv.Detections],
        object_predictions: Batch[sv.Detections],
        reference_dimensions: Union[str, Tuple[float, float]],
    ) -> BlockResult:
        result = []
        ref_width_actual, ref_height_actual = SizeMeasurementManifest.parse_reference_dimensions(reference_dimensions)
        
        for ref_detections, obj_detections in zip(reference_predictions, object_predictions):
            if ref_detections is None or obj_detections is None:
                result.append({OUTPUT_KEY: None})
                continue

            if len(ref_detections) == 0:
                result.append({OUTPUT_KEY: None})
                continue

            ref_index = np.argmax(ref_detections.confidence)
            ref_width_pixels, ref_height_pixels = get_detection_dimensions(ref_detections, ref_index)

            if ref_width_pixels == 0 or ref_height_pixels == 0:
                result.append({OUTPUT_KEY: None})
                continue

            dimensions = []
            for i in range(len(obj_detections)):
                obj_width_pixels, obj_height_pixels = get_detection_dimensions(obj_detections, i)
                obj_width_actual = (obj_width_pixels / ref_width_pixels) * ref_width_actual
                obj_height_actual = (obj_height_pixels / ref_height_pixels) * ref_height_actual
                dimensions.append({"width": obj_width_actual, "height": obj_height_actual})

            result.append({OUTPUT_KEY: dimensions})
        
        return result
