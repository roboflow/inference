from typing import Dict, List, Optional, Tuple, Union

import supervision as sv
from pydantic import ConfigDict, Field
from typing_extensions import Literal, Type

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
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

OUTPUT_KEY_COUNT_IN: str = "count_in"
OUTPUT_KEY_COUNT_OUT: str = "count_out"
OUTPUT_KEY_DETECTIONS_IN: str = "detections_in"
OUTPUT_KEY_DETECTIONS_OUT: str = "detections_out"
IN: str = "in"
OUT: str = "out"
DETECTIONS_IN_OUT_PARAM: str = "in_out"
SHORT_DESCRIPTION = "Count detections passing a line."
LONG_DESCRIPTION = """
The `LineCounter` is an analytics block designed to count objects passing the line.
The block requires detections to be tracked (i.e. each object must have unique tracker_id assigned,
which persists between frames)
"""


class LineCounterManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Line Counter",
            "version": "v2",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "analytics",
            "ui_manifest": {
                "section": "video",
                "icon": "far fa-arrow-down-up-across-line",
                "blockPriority": 2,
            },
        }
    )
    type: Literal["roboflow_core/line_counter@v2"]
    image: WorkflowImageSelector
    detections: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Model predictions to count line crossings for.",
        examples=["$steps.object_detection_model.predictions"],
    )

    line_segment: Union[list, Selector(kind=[LIST_OF_VALUES_KIND]), Selector(kind=[LIST_OF_VALUES_KIND])] = Field(  # type: ignore
        description="Line consisting of exactly two points. For line [[0, 100], [100, 100]], objects entering from the bottom will count as IN.",
        examples=[[[0, 50], [500, 50]], "$inputs.zones"],
    )
    triggering_anchor: Optional[Union[str, Selector(kind=[STRING_KIND]), Literal[tuple(sv.Position.list())]]] = Field(  # type: ignore
        description=f"The point on the detection that must cross the line to be counted.",
        default=None,
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
            OutputDefinition(
                name=OUTPUT_KEY_DETECTIONS_IN,
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                ],
            ),
            OutputDefinition(
                name=OUTPUT_KEY_DETECTIONS_OUT,
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class LineCounterBlockV2(WorkflowBlock):
    def __init__(self):
        self._batch_of_line_zones: Dict[str, sv.LineZone] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return LineCounterManifest

    def run(
        self,
        detections: sv.Detections,
        image: WorkflowImageData,
        line_segment: List[Tuple[int, int]],
        triggering_anchor: Optional[str] = None,
    ) -> BlockResult:
        if detections.tracker_id is None:
            raise ValueError(
                f"tracker_id not initialized, {self.__class__.__name__} requires detections to be tracked"
            )
        metadata = image.video_metadata
        vid_id = metadata.video_identifier
        line_zones = self._batch_of_line_zones

        if vid_id not in line_zones:
            # Perform all checks in one loop to avoid redundant passes
            if not (isinstance(line_segment, list) and len(line_segment) == 2):
                raise ValueError(
                    f"{self.__class__.__name__} requires line zone to be a list containing exactly 2 points"
                )
            for e in line_segment:
                if not (isinstance(e, list) and len(e) == 2):
                    raise ValueError(
                        f"{self.__class__.__name__} requires each point of line zone to be a list containing exactly 2 coordinates"
                    )
                x, y = e
                if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
                    raise ValueError(
                        f"{self.__class__.__name__} requires each coordinate of line zone to be a number"
                    )

            start, end = line_segment
            if triggering_anchor is not None:
                line_zone = sv.LineZone(
                    start=sv.Point(*start),
                    end=sv.Point(*end),
                    triggering_anchors=[sv.Position(triggering_anchor)],
                )
            else:
                line_zone = sv.LineZone(
                    start=sv.Point(*start),
                    end=sv.Point(*end),
                )
            line_zones[vid_id] = line_zone
        else:
            line_zone = line_zones[vid_id]

        mask_in, mask_out = line_zone.trigger(detections=detections)
        # sv.Detections likely supports indexing with a boolean array: do both in one step without repeated calls
        detections_in = detections[mask_in]
        detections_out = detections[mask_out]

        return {
            OUTPUT_KEY_COUNT_IN: line_zone.in_count,
            OUTPUT_KEY_COUNT_OUT: line_zone.out_count,
            OUTPUT_KEY_DETECTIONS_IN: detections_in,
            OUTPUT_KEY_DETECTIONS_OUT: detections_out,
        }
