from typing import List, Literal, Optional, Tuple, Type, Union

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
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    StepOutputSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY: str = "dimensions"
SHORT_DESCRIPTION = (
    "Measure the dimensions of objects in relation to a reference object."
)
LONG_DESCRIPTION = """
The `SizeMeasurementBlock` is a transformer block designed to measure the dimensions of objects
in relation to a reference object. The reference object is detected using one model,
and the object to be measured is detected using another model. The block outputs the dimensions of the
objects to be measured in terms of the reference object.
Note: if reference_predictions provides multiple boxes, the most confident one will be selected.
In order to achieve different behavior you can use Detection Transformation block with custom filter
and also continue_if block if no reference detection meets expectations.
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
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-ruler",
                "opencv": True,
            },
        }
    )
    type: Literal[f"roboflow_core/size_measurement@v1"]
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
    reference_dimensions: Union[
        str,
        Tuple[float, float],
        List[float],
        WorkflowParameterSelector(
            kind=[STRING_KIND, LIST_OF_VALUES_KIND],
        ),
    ] = Field(  # type: ignore
        description="Dimensions of the reference object (width, height) in desired units (e.g., inches) as a string in the format 'width,height' or as a tuple (width, height)",
        examples=["5.0,5.0", (5.0, 5.0), "$inputs.reference_dimensions"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=OUTPUT_KEY, kind=[LIST_OF_VALUES_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


def get_detection_dimensions(
    detection: sv.Detections, index: int
) -> Tuple[float, float]:
    if detection.mask is not None:
        mask = detection.mask[index].astype(np.uint8)
        *_, w, h = cv.boundingRect(mask)
    else:
        bbox = detection.xyxy[index]
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return w, h


class SizeMeasurementBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return SizeMeasurementManifest

    def run(
        self,
        reference_predictions: sv.Detections,
        object_predictions: sv.Detections,
        reference_dimensions: Union[str, Tuple[float, float]],
    ) -> BlockResult:
        if isinstance(reference_dimensions, str):
            try:
                reference_dimensions = [
                    float(d) for d in reference_dimensions.split(",")
                ]
            except ValueError:
                raise ValueError(
                    "reference_dimensions must be a string in the format 'width,height'"
                )
        if (
            not isinstance(reference_dimensions, (tuple, list))
            or len(reference_dimensions) != 2
        ):
            raise ValueError(
                "reference_dimensions must be a string in the format 'width,height' or a tuple (width, height)"
            )

        ref_width_actual, ref_height_actual = reference_dimensions

        ref_index = np.argmax(reference_predictions.confidence)
        ref_width_pixels, ref_height_pixels = get_detection_dimensions(
            reference_predictions, ref_index
        )

        if ref_width_pixels == 0 or ref_height_pixels == 0:
            return {OUTPUT_KEY: None}

        dimensions = []
        for i in range(len(object_predictions)):
            obj_width_pixels, obj_height_pixels = get_detection_dimensions(
                object_predictions, i
            )
            obj_width_actual = (obj_width_pixels / ref_width_pixels) * ref_width_actual
            obj_height_actual = (
                obj_height_pixels / ref_height_pixels
            ) * ref_height_actual
            dimensions.append({"width": obj_width_actual, "height": obj_height_actual})

        return {OUTPUT_KEY: dimensions}
