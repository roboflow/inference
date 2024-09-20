from typing import List, Literal, Optional, Tuple, Type, Union

import cv2 as cv
import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    LIST_OF_VALUES_KIND,
    StepOutputSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY_RECT: str = "rect"
OUTPUT_KEY_WIDTH: float = "width"
OUTPUT_KEY_HEIGHT: float = "height"
OUTPUT_KEY_ANGLE: float = "angle"
SHORT_DESCRIPTION = "Find minimal bounding rectangle surrounding detection contour"
LONG_DESCRIPTION = """
The `BoundingRect` is a transformer block designed to simplify polygon
to the minimum boundig rectangle.
This block is best suited when Zone needs to be created based on shape of detected object
(i.e. basketball field, road segment, zebra crossing etc.)
Input detections should be filtered beforehand and contain only desired classes of interest.
"""


class BoundingRectManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Bounding Rect",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
        }
    )
    type: Literal[f"roboflow_core/min_rect@v1"]
    predictions: StepOutputSelector(
        kind=[
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="",
        examples=["$segmentation.predictions"],
    )

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return False

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=OUTPUT_KEY_RECT, kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name=OUTPUT_KEY_WIDTH, kind=[FLOAT_KIND]),
            OutputDefinition(name=OUTPUT_KEY_HEIGHT, kind=[FLOAT_KIND]),
            OutputDefinition(name=OUTPUT_KEY_ANGLE, kind=[FLOAT_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


def calculate_minimum_bounding_rectangle(
    mask: np.ndarray,
) -> Tuple[np.array, float, float, float]:
    contours = sv.mask_to_polygons(mask)
    largest_contour = max(contours, key=len)

    rect = cv.minAreaRect(largest_contour)
    box = cv.boxPoints(rect)
    box = np.int0(box)

    width, height = rect[1]
    angle = rect[2]
    return box, width, height, angle


class BoundingRectBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BoundingRectManifest

    def run(
        self,
        predictions: sv.Detections,
    ) -> BlockResult:
        result = []
        if predictions.mask is None:
            raise ValueError(
                "Mask missing. This block operates on output from segmentation model."
            )
        for mask in predictions.mask:
            polygon, width, height, angle = calculate_minimum_bounding_rectangle(mask)
            result.append(
                {
                    OUTPUT_KEY_RECT: polygon,
                    OUTPUT_KEY_WIDTH: width,
                    OUTPUT_KEY_HEIGHT: height,
                    OUTPUT_KEY_ANGLE: angle,
                }
            )
        return result
