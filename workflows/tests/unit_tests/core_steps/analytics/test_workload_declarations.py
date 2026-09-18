"""Workload declarations of the analytics blocks.

Analytics blocks turn detections into counts, times and rates. The distinction
this module pins is between a block that only post-processes the detections of
the CURRENT frame and one that carries history across frames - only the latter
declares ``TEMPORAL_BUFFERING`` and the stateful-video caveat.
"""

from roboflow_workflows.core_steps.analytics.data_aggregator.v1 import (
    BlockManifest as DataAggregatorManifest,
)
from roboflow_workflows.core_steps.analytics.line_counter.v2 import LineCounterManifest
from roboflow_workflows.core_steps.analytics.overlap.v1 import OverlapManifest
from roboflow_workflows.execution_engine.entities.workload import WorkOperation

STATEFUL_CODES = [
    "stateful_video_state_resets_on_stateless_http",
    "temporal_block_no_benefit_on_still_image",
]


def test_line_counter_declares_detection_work_and_cross_frame_state() -> None:
    manifest = LineCounterManifest(
        type="roboflow_core/line_counter@v2",
        name="counter",
        image="$inputs.image",
        detections="$steps.tracker.tracked_detections",
        line_segment=[[0, 0], [100, 100]],
    )
    assert manifest.discover_work_operations() == [
        WorkOperation.DETECTION_PROCESSING,
        WorkOperation.TEMPORAL_BUFFERING,
    ]
    assert [
        restriction.code for restriction in manifest.discover_portable_restrictions()
    ] == STATEFUL_CODES


def test_data_aggregator_declares_aggregation_rather_than_detection_work() -> None:
    manifest = DataAggregatorManifest(
        type="roboflow_core/data_aggregator@v1",
        name="aggregator",
        data={"count": "$steps.counter.count_in"},
        aggregation_mode={"count": ["sum"]},
        interval=60,
    )
    assert manifest.discover_work_operations() == [
        WorkOperation.DATA_AGGREGATION,
        WorkOperation.TEMPORAL_BUFFERING,
    ]
    assert [
        restriction.code for restriction in manifest.discover_portable_restrictions()
    ] == STATEFUL_CODES


def test_a_stateless_analytics_block_declares_no_buffering_and_no_caveat() -> None:
    manifest = OverlapManifest(
        type="roboflow_core/overlap@v1",
        name="overlap",
        predictions="$steps.model.predictions",
        overlap_class_name="person",
    )
    operations = manifest.discover_work_operations()
    assert operations == [WorkOperation.DETECTION_PROCESSING]
    assert WorkOperation.TEMPORAL_BUFFERING not in operations
    assert manifest.discover_portable_restrictions() == []
