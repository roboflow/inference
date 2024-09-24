from typing import Dict, List, Optional, Tuple, Union

import supervision as sv
from pydantic import ConfigDict, Field
from typing_extensions import Literal, Type

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    VideoMetadata,
)
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    StepOutputSelector,
    WorkflowParameterSelector,
    WorkflowVideoMetadataSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY_COUNT_IN: str = "count_in"
OUTPUT_KEY_COUNT_OUT: str = "count_out"
IN: str = "in"
OUT: str = "out"
DETECTIONS_IN_OUT_PARAM: str = "in_out"
SHORT_DESCRIPTION = "Count detections passing line"
LONG_DESCRIPTION = """
The `LineCounter` is an analytics block designed to count objects passing the line.
The block requires detections to be tracked (i.e. each object must have unique tracker_id assigned,
which persists between frames)
"""


class LineCounterManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Line Counter",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "analytics",
        }
    )
    type: Literal["roboflow_core/line_counter@v1"]
    metadata: WorkflowVideoMetadataSelector
    detections: StepOutputSelector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Predictions",
        examples=["$steps.object_detection_model.predictions"],
    )

    line_segment: Union[list, StepOutputSelector(kind=[LIST_OF_VALUES_KIND]), WorkflowParameterSelector(kind=[LIST_OF_VALUES_KIND])] = Field(  # type: ignore
        description="Line in the format [[x1, y1], [x2, y2]] consisting of exactly two points. For line [[0, 100], [100, 100]] line will count objects entering from the bottom as IN",
        examples=[[[0, 50], [500, 50]], "$inputs.zones"],
    )
    triggering_anchor: Union[str, WorkflowParameterSelector(kind=[STRING_KIND]), Literal[tuple(sv.Position.list())]] = Field(  # type: ignore
        description=f"Point from the detection for triggering line crossing.",
        default="CENTER",
        examples=["CENTER"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_KEY_COUNT_IN,
                kind=[INTEGER_KIND],
            ),
            OutputDefinition(
                name=OUTPUT_KEY_COUNT_OUT,
                kind=[INTEGER_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class LineCounterBlockV1(WorkflowBlock):
    def __init__(self):
        self._batch_of_line_zones: Dict[str, sv.LineZone] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return LineCounterManifest

    def run(
        self,
        detections: sv.Detections,
        metadata: VideoMetadata,
        line_segment: List[Tuple[int, int]],
        triggering_anchor: str = "CENTER",
    ) -> BlockResult:
        if detections.tracker_id is None:
            raise ValueError(
                f"tracker_id not initialized, {self.__class__.__name__} requires detections to be tracked"
            )
        if metadata.video_identifier not in self._batch_of_line_zones:
            if not isinstance(line_segment, list) or len(line_segment) != 2:
                raise ValueError(
                    f"{self.__class__.__name__} requires line zone to be a list containing exactly 2 points"
                )
            if any(not isinstance(e, list) or len(e) != 2 for e in line_segment):
                raise ValueError(
                    f"{self.__class__.__name__} requires each point of line zone to be a list containing exactly 2 coordinates"
                )
            if any(
                not isinstance(e[0], (int, float)) or not isinstance(e[1], (int, float))
                for e in line_segment
            ):
                raise ValueError(
                    f"{self.__class__.__name__} requires each coordinate of line zone to be a number"
                )
            self._batch_of_line_zones[metadata.video_identifier] = sv.LineZone(
                start=sv.Point(*line_segment[0]),
                end=sv.Point(*line_segment[1]),
                triggering_anchors=[sv.Position(triggering_anchor)],
            )
        line_zone = self._batch_of_line_zones[metadata.video_identifier]

        line_zone.trigger(detections=detections)

        return {
            OUTPUT_KEY_COUNT_IN: line_zone.in_count,
            OUTPUT_KEY_COUNT_OUT: line_zone.out_count,
        }
