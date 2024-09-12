from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field
from typing_extensions import Literal, Type

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    VideoMetadata,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    StepOutputImageSelector,
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
    WorkflowVideoMetadataSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY: str = "time_in_zone"
TYPE: str = "PerspectiveCorrection"
SHORT_DESCRIPTION = "Track duration of time spent by objects in zone"
LONG_DESCRIPTION = """
The `TimeInZoneBlock` is an analytics block designed to measure time spent by objects in a zone.
The block requires detections to be tracked (i.e. each object must have unique tracker_id assigned,
which persists between frames)
"""


class TimeInZoneManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Time in zone",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "analytics",
        }
    )
    type: Literal["roboflow_core/time_in_zone@v1", "TimeInZone"]
    image: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        title="Image",
        description="The input image for this step.",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    metadata: WorkflowVideoMetadataSelector
    detections: StepOutputSelector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Predictions",
        default=None,
        examples=["$steps.object_detection_model.predictions"],
    )
    zone: Union[list, StepOutputSelector(kind=[LIST_OF_VALUES_KIND]), WorkflowParameterSelector(kind=[LIST_OF_VALUES_KIND])] = Field(  # type: ignore
        description="Zones (one for each batch) in a format [(x1, y1), (x2, y2), (x3, y3), ...]",
        examples=["$inputs.zones"],
    )
    detections_anchor: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = Field(  # type: ignore
        description=f"Detection anchor point. Allowed values: {', '.join(sv.Position.list())}",
        default="CENTER",
        examples=["CENTER"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_KEY,
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class TimeInZoneBlockV1(WorkflowBlock):
    def __init__(self):
        self._batch_of_tracked_ids_in_zone: Dict[str, Dict[Union[int, str], float]] = {}
        self._batch_of_masks: Dict[str, np.ndarray] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return TimeInZoneManifest

    def run(
        self,
        image: WorkflowImageData,
        detections: sv.Detections,
        metadata: VideoMetadata,
        zone: List[Tuple[int, int]],
        detections_anchor: str = "CENTER",
    ) -> BlockResult:
        if metadata.video_identifier not in self._batch_of_masks:
            polygon = np.array(zone)
            h, w, *_ = image.numpy_image.shape
            mask = sv.polygon_to_mask(polygon=polygon, resolution_wh=(w, h))
            self._batch_of_masks[metadata.video_identifier] = mask
        mask = self._batch_of_masks[metadata.video_identifier]

        tracked_ids_in_zone = self._batch_of_tracked_ids_in_zone.setdefault(
            metadata.video_identifier, {}
        )
        points = detections.get_anchors_coordinates(
            anchor=sv.Position(detections_anchor)
        )
        result_detections = []
        if metadata.comes_from_video_file and metadata.fps != 0:
            ts_end = metadata.frame_number / metadata.fps
        else:
            ts_end = metadata.frame_timestamp.timestamp()
        for i, (x, y), tracker_id in zip(
            range(len(detections)), points, detections.tracker_id
        ):
            # copy
            detection = detections[i]

            detection[OUTPUT_KEY] = np.array([0], dtype=np.float64)
            if mask[int(round(y)), int(round(x))] == 1:
                ts_start = tracked_ids_in_zone.setdefault(tracker_id, ts_end)
                detection[OUTPUT_KEY] = np.array([ts_end - ts_start], dtype=np.float64)
            elif tracker_id in tracked_ids_in_zone:
                del tracked_ids_in_zone[tracker_id]
            result_detections.append(detection)
        return {OUTPUT_KEY: sv.Detections.merge(result_detections)}
