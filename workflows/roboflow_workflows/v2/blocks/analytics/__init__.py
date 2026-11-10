"""Analytics blocks that measure tracked objects over time."""

from roboflow_workflows.v2.blocks.analytics.velocity.v1 import VelocityBlock
from roboflow_workflows.v2.blocks.analytics.background_subtraction.v1 import BackgroundSubtractionBlock
from roboflow_workflows.v2.blocks.analytics.detection_event_log.v1 import DetectionEventLogBlock
from roboflow_workflows.v2.blocks.analytics.line_counter.v2 import LineCounterBlock
from roboflow_workflows.v2.blocks.analytics.motion_detection.v1 import MotionDetectionBlock
from roboflow_workflows.v2.blocks.analytics.overlap.v1 import OverlapFilterBlock
from roboflow_workflows.v2.blocks.analytics.path_deviation_analytics.v2 import PathDeviationBlock
from roboflow_workflows.v2.blocks.analytics.time_in_zone.v3 import TimeInZoneBlock

from roboflow_workflows.v2.blocks.analytics.data_aggregator.v1 import DataAggregatorBlock

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
