from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field
from typing_extensions import Literal, Type

from inference.core.workflows.execution_engine.constants import (
    TIME_IN_ZONE_KEY_IN_SV_DETECTIONS,
)
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    Selector,
    WorkflowImageSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY: str = "timed_detections"
SHORT_DESCRIPTION = "Track object time in zone."
LONG_DESCRIPTION = """
The `TimeInZoneBlock` is an analytics block designed to measure time spent by objects in a zone.
The block requires detections to be tracked (i.e. each object must have unique tracker_id assigned,
which persists between frames)
"""


class TimeInZoneManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Time in Zone",
            "version": "v2",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "analytics",
            "ui_manifest": {
                "section": "video",
                "icon": "far fa-timer",
                "blockPriority": 1,
            },
        }
    )
    type: Literal["roboflow_core/time_in_zone@v2"]
    image: Union[WorkflowImageSelector] = Field(
        title="Image",
        description="The input image for this step.",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    detections: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Model predictions to calculate the time spent in zone for.",
        examples=["$steps.object_detection_model.predictions"],
    )
    zone: Union[list, Selector(kind=[LIST_OF_VALUES_KIND]), Selector(kind=[LIST_OF_VALUES_KIND])] = Field(  # type: ignore
        description="Coordinates of the target zone.",
        examples=[[(100, 100), (100, 200), (300, 200), (300, 100)], "$inputs.zones"],
    )
    triggering_anchor: Union[str, Selector(kind=[STRING_KIND]), Literal[tuple(sv.Position.list())]] = Field(  # type: ignore
        description=f"The point on the detection that must be inside the zone.",
        default="CENTER",
        examples=["CENTER"],
    )
    remove_out_of_zone_detections: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(  # type: ignore
        description=f"If true, detections found outside of zone will be filtered out.",
        default=True,
        examples=[True, False],
    )
    reset_out_of_zone_detections: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(  # type: ignore
        description=f"If true, detections found outside of zone will have time reset.",
        default=True,
        examples=[True, False],
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
        return ">=1.3.0,<2.0.0"


class TimeInZoneBlockV2(WorkflowBlock):
    def __init__(self):
        self._batch_of_tracked_ids_in_zone: Dict[str, Dict[Union[int, str], float]] = {}
        self._batch_of_polygon_zones: Dict[str, sv.PolygonZone] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return TimeInZoneManifest

    def run(
        self,
        image: WorkflowImageData,
        detections: sv.Detections,
        zone: List[Tuple[int, int]],
        triggering_anchor: str,
        remove_out_of_zone_detections: bool,
        reset_out_of_zone_detections: bool,
    ) -> BlockResult:
        if detections.tracker_id is None:
            raise ValueError(
                f"tracker_id not initialized, {self.__class__.__name__} requires detections to be tracked"
            )
        metadata = image.video_metadata
        if metadata.video_identifier not in self._batch_of_polygon_zones:
            if not isinstance(zone, list) or len(zone) < 3:
                raise ValueError(
                    f"{self.__class__.__name__} requires zone to be a list containing more than 2 points"
                )
            if any(
                (not isinstance(e, list) and not isinstance(e, tuple)) or len(e) != 2
                for e in zone
            ):
                raise ValueError(
                    f"{self.__class__.__name__} requires each point of zone to be a list containing exactly 2 coordinates"
                )
            if any(
                not isinstance(e[0], (int, float)) or not isinstance(e[1], (int, float))
                for e in zone
            ):
                raise ValueError(
                    f"{self.__class__.__name__} requires each coordinate of zone to be a number"
                )
            self._batch_of_polygon_zones[metadata.video_identifier] = sv.PolygonZone(
                polygon=np.array(zone),
                triggering_anchors=(sv.Position(triggering_anchor),),
            )
        polygon_zone = self._batch_of_polygon_zones[metadata.video_identifier]
        tracked_ids_in_zone = self._batch_of_tracked_ids_in_zone.setdefault(
            metadata.video_identifier, {}
        )
        result_detections = []
        if metadata.comes_from_video_file and metadata.fps != 0:
            ts_end = metadata.frame_number / metadata.fps
        else:
            ts_end = metadata.frame_timestamp.timestamp()
        for i, is_in_zone, tracker_id in zip(
            range(len(detections)),
            polygon_zone.trigger(detections),
            detections.tracker_id,
        ):
            if (
                not is_in_zone
                and tracker_id in tracked_ids_in_zone
                and reset_out_of_zone_detections
            ):
                del tracked_ids_in_zone[tracker_id]
            if not is_in_zone and remove_out_of_zone_detections:
                continue

            # copy
            detection = detections[i]

            detection[TIME_IN_ZONE_KEY_IN_SV_DETECTIONS] = np.array(
                [0], dtype=np.float64
            )
            if is_in_zone:
                ts_start = tracked_ids_in_zone.setdefault(tracker_id, ts_end)
                detection[TIME_IN_ZONE_KEY_IN_SV_DETECTIONS] = np.array(
                    [ts_end - ts_start], dtype=np.float64
                )
            elif tracker_id in tracked_ids_in_zone:
                del tracked_ids_in_zone[tracker_id]
            result_detections.append(detection)
        return {OUTPUT_KEY: sv.Detections.merge(result_detections)}
