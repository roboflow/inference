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
import pytest
from roboflow_workflows.core_steps.cache.memory_cache import WorkflowMemoryCache
from roboflow_workflows.execution_engine.core import ExecutionEngine
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)

ZONE = [[0, 0], [100, 0], [100, 100], [0, 100]]

GRAYSCALE_STEP = {
    "type": "roboflow_core/convert_grayscale@v1",
    "name": "standard_image",
    "image": "$inputs.image",
}
GRAYSCALE_IMAGE = "$steps.standard_image.image"

ABSOLUTE_STATIC_CROP_STEP = {
    "type": "roboflow_core/absolute_static_crop@v1",
    "name": "standard_image",
    "images": "$inputs.image",
    "x_center": 16,
    "y_center": 16,
    "width": 16,
    "height": 16,
}
RELATIVE_STATIC_CROP_STEP = {
    "type": "roboflow_core/relative_statoic_crop@v1",
    "name": "standard_image",
    "images": "$inputs.image",
    "x_center": 0.5,
    "y_center": 0.5,
    "width": 0.5,
    "height": 0.5,
}
CROP_IMAGE = "$steps.standard_image.crops"

CAMERA_CALIBRATION_STEP = {
    "type": "roboflow_core/camera-calibration@v1",
    "name": "standard_image",
    "image": "$inputs.image",
    "fx": 32.0,
    "fy": 32.0,
    "cx": 16.0,
    "cy": 16.0,
    "k1": 0.0,
    "k2": 0.0,
    "k3": 0.0,
    "p1": 0.0,
    "p2": 0.0,
}
CALIBRATED_IMAGE = "$steps.standard_image.calibrated_image"

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
    outputs: list,
    image_step: Dict[str, Any] = GRAYSCALE_STEP,
    image_selector: str = GRAYSCALE_IMAGE,
    extra_inputs: list = (),
    extra_bindings: Dict[str, str] = {},
) -> Dict[str, Any]:
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}, *extra_inputs],
        "steps": [
            image_step,
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "nested",
                "workflow_definition": child,
                "parameter_bindings": {"image": image_selector, **extra_bindings},
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
        comes_from_video_file=True,
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


def _reset_memory_cache() -> None:
    WorkflowMemoryCache.cache.clear()
    WorkflowMemoryCache._retain_counts.clear()


@pytest.fixture
def clean_memory_cache():
    _reset_memory_cache()
    yield
    _reset_memory_cache()


@pytest.mark.parametrize(
    "image_step, image_selector",
    [
        (GRAYSCALE_STEP, GRAYSCALE_IMAGE),
        (ABSOLUTE_STATIC_CROP_STEP, CROP_IMAGE),
        (RELATIVE_STATIC_CROP_STEP, CROP_IMAGE),
    ],
    ids=["grayscale", "absolute_static_crop", "relative_static_crop"],
)
def test_embedded_cache_keeps_one_namespace_per_video_across_frames(
    clean_memory_cache: None,
    image_step: Dict[str, Any],
    image_selector: str,
) -> None:
    # given
    workflow_definition = _parent_workflow(
        CACHE_CHILD_WORKFLOW,
        image_step=image_step,
        image_selector=image_selector,
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
    camera_a_results = [
        execution_engine.run(
            runtime_parameters={
                "image": _video_frame(video_id="cam-a", frame_number=frame_number)
            }
        )[0]
        for frame_number in (1, 2, 3)
    ]
    camera_b_result = execution_engine.run(
        runtime_parameters={"image": _video_frame(video_id="cam-b", frame_number=1)}
    )[0]

    # then
    assert camera_a_results == [
        {"before": False, "stored": "False"},
        {"before": "False", "stored": "False"},
        {"before": "False", "stored": "False"},
    ]
    assert camera_b_result == {"before": False, "stored": "False"}
    namespaces = sorted(WorkflowMemoryCache.cache.keys())
    assert len(namespaces) == 2
    assert namespaces[0].startswith("cam-a")
    assert namespaces[1].startswith("cam-b")


RATE_LIMITED_CHILD_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/rate_limiter@v1",
            "name": "rate_limiter",
            "cooldown_seconds": 1.0,
            "depends_on": "$inputs.image",
            "next_steps": ["$steps.passed"],
            "video_reference_image": "$inputs.image",
        },
        {
            "type": "roboflow_core/property_definition@v1",
            "name": "passed",
            "data": "$inputs.image",
            "operations": [{"type": "ExtractImageProperty", "property_name": "width"}],
        },
    ],
    "outputs": [
        {"type": "JsonField", "name": "passed", "selector": "$steps.passed.output"},
    ],
}


@pytest.mark.parametrize(
    "image_step, image_selector",
    [
        (GRAYSCALE_STEP, GRAYSCALE_IMAGE),
        (CAMERA_CALIBRATION_STEP, CALIBRATED_IMAGE),
    ],
    ids=["grayscale", "camera_calibration"],
)
def test_embedded_rate_limiter_follows_video_time_of_step_output_image(
    image_step: Dict[str, Any],
    image_selector: str,
) -> None:
    # given
    workflow_definition = _parent_workflow(
        RATE_LIMITED_CHILD_WORKFLOW,
        image_step=image_step,
        image_selector=image_selector,
        outputs=[
            {
                "type": "JsonField",
                "name": "passed",
                "selector": "$steps.nested.passed",
            },
        ],
    )
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
        init_parameters={},
        max_concurrent_steps=1,
    )

    # when
    results = [
        execution_engine.run(
            runtime_parameters={
                "image": _video_frame(video_id="cam-a", frame_number=frame_number)
            }
        )[0]
        for frame_number in (1, 2, 3, 5)
    ]

    # then
    assert [result["passed"] for result in results] == [32, 32, 32, 32]
