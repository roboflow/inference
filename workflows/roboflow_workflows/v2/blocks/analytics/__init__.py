"""Analytics blocks that measure tracked objects over time."""

from roboflow_workflows.v2.blocks.analytics.velocity import VelocityBlock
from roboflow_workflows.v2.blocks.analytics.background_subtraction import (
    BackgroundSubtractionBlock,
)
from roboflow_workflows.v2.blocks.analytics.event_log import (
    DetectionEventLogBlock,
)
from roboflow_workflows.v2.blocks.analytics.line_counter import (
    LineCounterBlock,
)
from roboflow_workflows.v2.blocks.analytics.motion_detection import (
    MotionDetectionBlock,
)
from roboflow_workflows.v2.blocks.analytics.overlap_filter import (
    OverlapFilterBlock,
)
from roboflow_workflows.v2.blocks.analytics.path_deviation import (
    PathDeviationBlock,
)
from roboflow_workflows.v2.blocks.analytics.time_in_zone import (
    TimeInZoneBlock,
)

from roboflow_workflows.v2.blocks.analytics.data_aggregator import (
    DataAggregatorBlock,
)

BLOCKS = (
    VelocityBlock,
    LineCounterBlock,
    TimeInZoneBlock,
    PathDeviationBlock,
    DetectionEventLogBlock,
    OverlapFilterBlock,
    BackgroundSubtractionBlock,
    MotionDetectionBlock,
    DataAggregatorBlock,
)

__all__ = [
    "BLOCKS",
    "VelocityBlock",
    "LineCounterBlock",
    "TimeInZoneBlock",
    "PathDeviationBlock",
    "DetectionEventLogBlock",
    "OverlapFilterBlock",
    "BackgroundSubtractionBlock",
    "MotionDetectionBlock",
    "DataAggregatorBlock",
]
