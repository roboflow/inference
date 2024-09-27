from typing import List, Literal, Optional, Tuple, Type

import cv2 as cv
import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.constants import (
    BOUNDING_RECT_ANGLE_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_HEIGHT_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_RECT_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_WIDTH_KEY_IN_SV_DETECTIONS,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    StepOutputSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY: str = "detections_with_rect"

SHORT_DESCRIPTION = "Find minimal bounding rectangle surrounding detection contour"
LONG_DESCRIPTION = """
The `BoundingRect` is a transformer block designed to simplify polygon
to the minimum boundig rectangle.
This block is best suited when Zone needs to be created based on shape of detected object
(i.e. basketball field, road segment, zebra crossing etc.)
Input detections should be filtered beforehand and contain only desired classes of interest.
Resulsts are stored in sv.Detections.data
"""


class BoundingRectManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Bounding Rectangle",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
        }
    )
    type: Literal[f"roboflow_core/bounding_rect@v1"]
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
            OutputDefinition(
                name=OUTPUT_KEY, kind=[INSTANCE_SEGMENTATION_PREDICTION_KIND]
            ),
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
        if predictions.mask is None:
            raise ValueError(
                "Mask missing. This block operates on output from segmentation model."
            )
        to_be_merged = []
        for i in range(len(predictions)):
            # copy
            det = predictions[i]

            rect, width, height, angle = calculate_minimum_bounding_rectangle(
                det.mask[0]
            )

            det[BOUNDING_RECT_RECT_KEY_IN_SV_DETECTIONS] = np.array(
                [rect], dtype=np.float16
            )
            det[BOUNDING_RECT_WIDTH_KEY_IN_SV_DETECTIONS] = np.array(
                [width], dtype=np.float16
            )
            det[BOUNDING_RECT_HEIGHT_KEY_IN_SV_DETECTIONS] = np.array(
                [height], dtype=np.float16
            )
            det[BOUNDING_RECT_ANGLE_KEY_IN_SV_DETECTIONS] = np.array(
                [angle], dtype=np.float16
            )

            to_be_merged.append(det)

        return {OUTPUT_KEY: sv.Detections.merge(to_be_merged)}
