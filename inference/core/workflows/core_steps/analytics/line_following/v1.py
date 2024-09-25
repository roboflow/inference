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
    BOOLEAN_KIND,
    FLOAT_KIND,
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

OUTPUT_KEY: str = "deviation_threshold"
DETECTIONS_TIME_IN_ZONE_PARAM: str = "time_in_zone"
SHORT_DESCRIPTION = "xxx"
LONG_DESCRIPTION = """
xxx
"""


class LineFollowingManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Line following",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "analytics",
        }
    )
    type: Literal["roboflow_core/line_following_analytics@v1"]
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
    triggering_anchor: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = Field(  # type: ignore
        description=f"Triggering anchor. Allowed values: {', '.join(sv.Position.list())}",
        default="CENTER",
        examples=["CENTER"],
    )
    reference_path: Union[list, StepOutputSelector(kind=[LIST_OF_VALUES_KIND]), WorkflowParameterSelector(kind=[LIST_OF_VALUES_KIND])] = Field(  # type: ignore
        description="Line segments (one for each batch) in a format [[(x1, y1), (x2, y2), (x3, y3), ...], ...];",
        examples=["$inputs.expected_path"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_KEY,
                kind=[
                    FLOAT_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class LineFollowingAnalyticsBlockV1(WorkflowBlock):
    def __init__(self):
        self._paths = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return PathAnalysisManifest

    def run(
        self,
        detections: sv.Detections,
        metadata: VideoMetadata,
        triggering_anchor: str,
        reference_path: List[Tuple[int, int]],
    ) -> BlockResult:
        return {OUTPUT_KEY: 0}

