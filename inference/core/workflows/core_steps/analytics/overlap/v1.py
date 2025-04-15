from typing import List, Optional, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field
from typing_extensions import Literal, Type
from supervision.detection.utils import get_data_item


from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
)
from inference.core.workflows.execution_engine.entities.types import (
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
"""


# trims a numpy array or list to specific indices
# after checking for None or empty
def trim_array(arr, idx):
    idx = list(idx)
    if arr is None:
        return arr

    if not arr.any():
        return arr

    if isinstance(arr, np.ndarray):
        return arr[idx]
    else:
        return [arr[i] for i in idx]


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
                "icon": "far fa-square-o",
                "blockPriority": 1,
            },
        }
    )
    type: Literal["roboflow_core/overlap@v1", "Overlap"]
    predictions: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Object predictions",
        examples=["$steps.object_detection_model.predictions"],
    )
    overlap_type: Literal["Center Overlap", "Any Overlap"] = Field(
        default="Center Overlap",
        description="Overlap Type.",
        examples=["Center Overlap", "Any Overlap"],
    )
    overlap_class_name: Union[str] = Field(
        description="Overlap Class Name",
        json_schema_extra={
            "hide_description": True,
        },
    )

    @classmethod
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
    def __init__(self):
        pass

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return OverlapManifest

    @classmethod
    def coords_overlap(cls, overlap, other, overlap_type):

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
        overlap_type: str,
        overlap_class_name: str,
    ) -> BlockResult:

        overlaps = []
        others = []
        for i in range(len(predictions.xyxy)):
            data = get_data_item(predictions.data, i)
            if data["class_name"] == overlap_class_name:
                overlaps.append(predictions.xyxy[i])
            else:
                others.append((predictions.xyxy[i], i))

        idx = set()
        for overlap in overlaps:
            if not others:
                break
            idx = idx.union(
                {
                    other[1]
                    for other in others
                    if OverlapBlockV1.coords_overlap(overlap, other[0], overlap_type)
                }
            )
            # once it's overlapped we don't need to check again
            others = [o for o in others if o[1] not in idx]

        # trim Detections to the filtered indices
        # this should be faster than iterating since each collection is copied once
        filtered = sv.Detections.empty()
        filtered.xyxy = trim_array(predictions.xyxy, idx)
        filtered.mask = trim_array(predictions.mask, idx)
        filtered.confidence = trim_array(predictions.confidence, idx)
        filtered.tracker_id = trim_array(predictions.tracker_id, idx)
        filtered.class_id = trim_array(predictions.class_id, idx)
        filtered.data = {
            k: trim_array(predictions.data[k], idx) for k in predictions.data.keys()
        }

        return {OUTPUT_KEY: filtered}
