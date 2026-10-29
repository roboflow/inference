"""
Embedded workflows whose ``image`` input is bound to a parent step output.

Inlining replaces the child's ``$inputs.image`` with the parent binding, so a child
that compiles on its own must still compile once its image comes from a step such as
an image preprocessing block. Video-state blocks (trackers, analytics, cache, flow
control) used to accept only ``$inputs.*`` image selectors, which made the merged
definition fail to parse.
"""

import datetime
from typing import Any, Dict

import numpy as np
from roboflow_workflows.core_steps.cache.memory_cache import WorkflowMemoryCache
from roboflow_workflows.execution_engine.core import ExecutionEngine
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)

ZONE = [[0, 0], [100, 0], [100, 100], [0, 100]]

VIDEO_STATE_CHILD_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {
            "type": "WorkflowBatchInput",
            "name": "predictions",
            "kind": ["object_detection_prediction"],
        },
    ],
    "steps": [
        {
            "type": "roboflow_core/byte_tracker@v2",
            "name": "tracker",
            "image": "$inputs.image",
            "detections": "$inputs.predictions",
        },
        {
            "type": "roboflow_core/time_in_zone@v2",
            "name": "time_in_zone_v2",
            "image": "$inputs.image",
            "detections": "$steps.tracker.tracked_detections",
            "zone": ZONE,
        },
        {
            "type": "roboflow_core/time_in_zone@v3",
            "name": "time_in_zone_v3",
            "image": "$inputs.image",
            "detections": "$steps.tracker.tracked_detections",
            "zone": ZONE,
        },
        {
            "type": "roboflow_core/velocity@v1",
            "name": "velocity",
            "image": "$inputs.image",
            "detections": "$steps.tracker.tracked_detections",
        },
        {
            "type": "roboflow_core/line_counter@v2",
            "name": "line_counter",
            "image": "$inputs.image",
            "detections": "$steps.tracker.tracked_detections",
            "line_segment": [[0, 50], [100, 50]],
        },
        {
            "type": "roboflow_core/path_deviation_analytics@v2",
            "name": "path_deviation",
            "image": "$inputs.image",
            "detections": "$steps.tracker.tracked_detections",
            "reference_path": [[0, 0], [100, 100]],
        },
        {
            "type": "roboflow_core/detection_event_log@v1",
            "name": "event_log",
            "image": "$inputs.image",
            "detections": "$steps.tracker.tracked_detections",
        },
        {
            "type": "roboflow_core/cache_get@v1",
            "name": "before",
            "image": "$inputs.image",
            "key": "k",
        },
        {
            "type": "roboflow_core/rate_limiter@v1",
            "name": "rate_limiter",
            "depends_on": "$steps.before.output",
            "next_steps": ["$steps.as_text"],
            "video_reference_image": "$inputs.image",
        },
        {
            "type": "roboflow_core/property_definition@v1",
            "name": "as_text",
            "data": "$steps.before.output",
            "operations": [{"type": "ToString"}],
        },
        {
            "type": "roboflow_core/delta_filter@v1",
            "name": "delta_filter",
            "image": "$inputs.image",
            "value": "$steps.as_text.output",
            "next_steps": ["$steps.store"],
        },
        {
            "type": "roboflow_core/cache_set@v1",
            "name": "store",
            "image": "$inputs.image",
            "key": "k",
            "value": "$steps.as_text.output",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "time_in_zone",
            "selector": "$steps.time_in_zone_v3.timed_detections",
        },
        {
            "type": "JsonField",
            "name": "velocity",
            "selector": "$steps.velocity.velocity_detections",
        },
    ],
}

CACHE_CHILD_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/cache_get@v1",
            "name": "before",
            "image": "$inputs.image",
            "key": "k",
        },
        {
            "type": "roboflow_core/property_definition@v1",
            "name": "as_text",
            "data": "$steps.before.output",
            "operations": [{"type": "ToString"}],
        },
        {
            "type": "roboflow_core/cache_set@v1",
            "name": "store",
            "image": "$inputs.image",
            "key": "k",
            "value": "$steps.as_text.output",
        },
    ],
    "outputs": [
        {"type": "JsonField", "name": "before", "selector": "$steps.before.output"},
        {"type": "JsonField", "name": "stored", "selector": "$steps.store.output"},
    ],
}


def _parent_workflow(
    child: Dict[str, Any],
    *,
    extra_inputs: list,
    extra_bindings: Dict[str, str],
    outputs: list,
) -> Dict[str, Any]:
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}, *extra_inputs],
        "steps": [
            {
                "type": "roboflow_core/convert_grayscale@v1",
                "name": "standard_image",
                "image": "$inputs.image",
            },
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "nested",
                "workflow_definition": child,
                "parameter_bindings": {
                    "image": "$steps.standard_image.image",
                    **extra_bindings,
                },
            },
        ],
        "outputs": outputs,
    }


def _video_frame(*, video_id: str, frame_number: int) -> WorkflowImageData:
    video_metadata = VideoMetadata(
        video_identifier=video_id,
        frame_number=frame_number,
        frame_timestamp=datetime.datetime.fromtimestamp(
            1726570875 + frame_number, tz=datetime.timezone.utc
        ),
        fps=1,
    )
    frame = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id=video_id),
        numpy_image=np.zeros((32, 32, 3), dtype=np.uint8),
        video_metadata=video_metadata,
    )

    return frame


def test_video_state_blocks_compile_when_embedded_image_comes_from_parent_step() -> (
    None
):
    # given
    workflow_definition = _parent_workflow(
        VIDEO_STATE_CHILD_WORKFLOW,
        extra_inputs=[
            {
                "type": "WorkflowBatchInput",
                "name": "predictions",
                "kind": ["object_detection_prediction"],
            }
        ],
        extra_bindings={"predictions": "$inputs.predictions"},
        outputs=[
            {
                "type": "JsonField",
                "name": "time_in_zone",
                "selector": "$steps.nested.time_in_zone",
            },
            {
                "type": "JsonField",
                "name": "velocity",
                "selector": "$steps.nested.velocity",
            },
        ],
    )

    # when
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
        init_parameters={},
        max_concurrent_steps=1,
    )

    # then
    assert execution_engine is not None


def test_embedded_cache_keeps_video_namespace_when_image_comes_from_parent_step() -> (
    None
):
    # given
    WorkflowMemoryCache.cache.clear()
    WorkflowMemoryCache._retain_counts.clear()
    workflow_definition = _parent_workflow(
        CACHE_CHILD_WORKFLOW,
        extra_inputs=[],
        extra_bindings={},
        outputs=[
            {
                "type": "JsonField",
                "name": "before",
                "selector": "$steps.nested.before",
            },
            {
                "type": "JsonField",
                "name": "stored",
                "selector": "$steps.nested.stored",
            },
        ],
    )
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
        init_parameters={},
        max_concurrent_steps=1,
    )

    # when
    first = execution_engine.run(
        runtime_parameters={"image": _video_frame(video_id="cam-a", frame_number=1)}
    )
    second = execution_engine.run(
        runtime_parameters={"image": _video_frame(video_id="cam-a", frame_number=2)}
    )
    other_video = execution_engine.run(
        runtime_parameters={"image": _video_frame(video_id="cam-b", frame_number=1)}
    )

    # then
    assert first[0] == {"before": False, "stored": "False"}
    assert second[0] == {"before": "False", "stored": "False"}
    assert other_video[0] == {"before": False, "stored": "False"}
    assert "cam-a" in WorkflowMemoryCache.cache
    assert "cam-b" in WorkflowMemoryCache.cache
