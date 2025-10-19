from functools import lru_cache
from typing import List, Optional, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field
from supervision.detection.utils.internal import get_data_item
from typing_extensions import Literal, Type

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY: str = "overlaps"
SHORT_DESCRIPTION = "Filter objects overlapping some other class"
LONG_DESCRIPTION = """
The `OverlapFilter` is an analytics block that filters out objects overlapping instances of some other class

For instance, for filtering people on bicycles, "bicycle" could be used as the overlap class.

Examples applications: people in a car, items on a pallet

The filter will remove the overlap class from the results, and only return the objects that overlap it. So
in the case above, bicycle will also be removed from the results.
"""


class OverlapManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Overlap Filter",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "analytics",
            "ui_manifest": {
                "section": "flow_control",
                "icon": "far fa-square-o",
                "blockPriority": 1.5,
            },
        }
    )
    type: Literal["roboflow_core/overlap@v1"]
    predictions: Selector(
        kind=[OBJECT_DETECTION_PREDICTION_KIND, INSTANCE_SEGMENTATION_PREDICTION_KIND]
    ) = Field(  # type: ignore
        description="Object predictions",
        examples=["$steps.object_detection_model.predictions"],
    )
    overlap_type: Literal["Center Overlap", "Any Overlap"] = Field(
        default="Center Overlap",
        description="Select center for centerpoint overlap, any for any overlap",
        examples=["Center Overlap", "Any Overlap"],
    )
    overlap_class_name: Union[str] = Field(
        description="Overlap Class Name",
        json_schema_extra={
            "hide_description": True,
        },
    )

    @classmethod
    @lru_cache(maxsize=None)
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_KEY,
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class OverlapBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return OverlapManifest

    @classmethod
    def coords_overlap(
        cls,
        overlap: list[int],
        other: list[int],
        overlap_type: Literal["Center Overlap", "Any Overlap"],
    ):

        # coords are [x1, y1, x2, y2]
        if overlap_type == "Center Overlap":
            size = [other[2] - other[0], other[3] - other[1]]
            (x, y) = [other[0] + size[0] / 2, other[1] + size[1] / 2]
            return (
                x > overlap[0] and x < overlap[2] and y > overlap[1] and y < overlap[3]
            )
        else:
            return not (
                other[2] < overlap[0]
                or other[0] > overlap[2]
                or other[3] < overlap[1]
                or other[1] > overlap[3]
            )

    def run(
        self,
        predictions: sv.Detections,
        overlap_type: Literal["Center Overlap", "Any Overlap"],
        overlap_class_name: str,
    ) -> BlockResult:

        overlaps = []
        others = {}
        for i in range(len(predictions.xyxy)):
            data = get_data_item(predictions.data, i)
            if data["class_name"] == overlap_class_name:
                overlaps.append(predictions.xyxy[i])
            else:
                others[i] = predictions.xyxy[i]

        # set of indices representing the overlapped objects
        idx = set()
        for overlap in overlaps:
            if not others:
                break
            overlapped = {
                k
                for k in others
                if OverlapBlockV1.coords_overlap(overlap, others[k], overlap_type)
            }
            # once it's overlapped we don't need to check again
            for k in overlapped:
                del others[k]

            idx = idx.union(overlapped)

        return {OUTPUT_KEY: predictions[list(idx)]}
