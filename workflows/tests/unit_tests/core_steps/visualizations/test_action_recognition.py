"""Temporal presentation contracts, including ownership and bounded memory."""

from datetime import datetime

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from roboflow_workflows.core_steps.visualizations.action_recognition.v1 import (
    TYPE,
    ActionRecognitionVisualizationBlockV1,
    BlockManifest,
)
from roboflow_workflows.execution_engine.entities.base import (
    ActionRecognitionPrediction,
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)


def _image(number, *, video="demo", fps=30, size=(90, 160), tensor=False):
    pixels = np.full((*size, 3), number % 256, dtype=np.uint8)
    representation = (
        {"tensor_image": torch.from_numpy(pixels).permute(2, 0, 1)}
        if tensor
        else {"numpy_image": pixels}
    )
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id=f"{video}-{number}"),
        video_metadata=VideoMetadata(
            video_identifier=video,
            frame_number=number,
            fps=fps,
            frame_timestamp=datetime(2026, 1, 1),
        ),
        **representation,
    )


def _action(name, start, end):
    return ActionRecognitionPrediction(
        class_name=name,
        class_id=0,
        start_frame_idx=start,
        end_frame_idx=end,
    )


def test_ten_second_delay_uses_new_timeline_for_old_pixels_and_inclusive_bounds():
    block = ActionRecognitionVisualizationBlockV1()
    for number in range(1, 301):
        result = block.run(_image(number), [])
        assert result["status"] == "buffering"
        assert result["displayed_frame_number"] is None
        assert result["actions"] == []

    # Evidence arrives after the image. Actions at both inclusive endpoints
    # belong to frame 1; classes from elsewhere in the window must not leak in.
    timeline = [
        _action("starts", 1, 10),
        _action("ends", 0, 1),
        _action("future", 2, 5),
        _action("past", 0, 0),
    ]
    result = block.run(_image(301), timeline)
    assert result["status"] == "ready"
    assert result["displayed_frame_number"] == 1
    assert result["actions"] == ["ends", "starts"]
    output = result["image"]
    assert output.video_metadata.frame_number == 1
    assert output.parent_metadata.parent_id == "demo-1"
    assert output.numpy_image[-1, -1, 0] == 1

    # Neither missing predictions nor model failure can be reported as proof
    # that analysis completed. 'ready' describes image availability only.
    result = block.run(_image(302), [])
    assert result["status"] == "ready"
    assert result["actions"] == []


@pytest.mark.parametrize("tensor", [False, True])
def test_buffer_owns_host_pixels_and_output_cannot_mutate_history(tensor):
    block = ActionRecognitionVisualizationBlockV1()
    original = _image(1, tensor=tensor)
    block.run(original, [], delay_seconds=1)
    original.numpy_image[:] = 255
    result = block.run(_image(31), [_action("a", 1, 1)], delay_seconds=1)
    assert result["image"].numpy_image[-1, -1, 0] == 1
    assert not result["image"].is_tensor_materialised()
    assert all(not f.is_tensor_materialised() for f in block._streams["demo"].frames)
    result["image"].numpy_image[:] = 200
    repeated = block.run(_image(31), [], delay_seconds=1)
    assert repeated["image"].numpy_image[-1, -1, 0] == 1
    assert repeated["actions"] == []
    assert len(block._streams["demo"].frames) == 2


def test_gaps_select_predecessor_without_reversing_presentation():
    block = ActionRecognitionVisualizationBlockV1()
    displayed = []
    for number in [1, 3, 6, 31, 32, 33, 34, 36, 90]:
        result = block.run(_image(number), [], delay_seconds=1)
        if result["status"] == "ready":
            displayed.append(result["displayed_frame_number"])
    assert displayed == [1, 1, 3, 3, 6, 36]


def test_streams_reset_independently_and_duplicate_input_does_not_restart():
    block = ActionRecognitionVisualizationBlockV1()
    for video in ["a", "b"]:
        block.run(_image(1, video=video), [], delay_seconds=1)
        assert (
            block.run(_image(31, video=video), [], delay_seconds=1)[
                "displayed_frame_number"
            ]
            == 1
        )
    restarted = block.run(_image(1, video="a"), [], delay_seconds=1)
    assert restarted["status"] == "buffering"
    assert (
        block.run(_image(31, video="b"), [], delay_seconds=1)["displayed_frame_number"]
        == 1
    )
    assert (
        block.run(_image(31, video="b"), [], delay_seconds=2)["status"] == "buffering"
    )


def test_resize_is_before_buffering_and_source_is_unchanged():
    block = ActionRecognitionVisualizationBlockV1()
    image = _image(1, size=(360, 640))
    result = block.run(image, [], delay_seconds=0, max_frame_side=320)
    assert result["image"].numpy_image.shape == (180, 320, 3)
    assert block._streams["demo"].frames[0].numpy_image.nbytes == 180 * 320 * 3
    assert np.all(image.numpy_image == 1)
    assert image.numpy_image.shape == (360, 640, 3)


def test_memory_budget_rejects_full_delay_upfront_and_is_shared_across_streams():
    block = ActionRecognitionVisualizationBlockV1()
    with pytest.raises(MemoryError, match="max_buffer_mb"):
        block.run(_image(1, size=(1080, 1920)), [], max_frame_side=1920)
    assert not block._streams
    block.run(_image(1), [], delay_seconds=1, max_buffer_mb=2)
    with pytest.raises(MemoryError, match="max_buffer_mb"):
        block.run(_image(1, video="other"), [], delay_seconds=1, max_buffer_mb=2)
    assert len(block._streams) == 1
    for number in range(2, 1000):
        block.run(_image(number), [], delay_seconds=1, max_buffer_mb=2)
    assert len(block._streams["demo"].frames) <= 32
    assert (
        sum(f.numpy_image.nbytes for f in block._streams["demo"].frames) <= 2 * 1024**2
    )


@pytest.mark.parametrize("fps", [None, 0, -1, float("nan"), float("inf")])
def test_invalid_source_fps_is_not_replaced_by_processing_fps(fps):
    image = _image(1, fps=fps)
    image.video_metadata.measured_fps = 30
    with pytest.raises(ValueError, match="source FPS"):
        ActionRecognitionVisualizationBlockV1().run(image, [])


def test_source_fps_is_pinned_like_the_model():
    block = ActionRecognitionVisualizationBlockV1()
    block.run(_image(1, fps=30), [], delay_seconds=1)
    result = block.run(_image(31, fps=60), [], delay_seconds=1)
    assert result["displayed_frame_number"] == 1


@pytest.mark.parametrize("delay", [-1, float("nan"), float("inf"), 301])
def test_manifest_rejects_invalid_delay(delay):
    with pytest.raises(ValidationError):
        BlockManifest(
            type=TYPE,
            name="display",
            image="$inputs.image",
            timeline="$steps.model.timeline",
            delay_seconds=delay,
        )


def test_manifest_discloses_temporal_state_and_outputs():
    manifest = BlockManifest(
        type=TYPE,
        name="display",
        image="$inputs.image",
        timeline="$steps.model.timeline",
    )
    restrictions = manifest.get_actual_restrictions(
        ignore_environment_restrictions=True
    )
    assert restrictions.complete
    assert any("stateful" in item.code for item in restrictions.items)
    assert {output.name for output in manifest.describe_outputs()} == {
        "image",
        "displayed_frame_number",
        "actions",
        "status",
    }


def test_real_engine_routes_late_timeline_and_serializes_displayed_identity():
    from roboflow_workflows.execution_engine.core import ExecutionEngine

    engine = ExecutionEngine.init(
        workflow_definition={
            "version": "1.0",
            "inputs": [
                {"type": "WorkflowImage", "name": "image"},
                {"type": "WorkflowBatchInput", "name": "timeline"},
            ],
            "steps": [
                {
                    "type": TYPE,
                    "name": "display",
                    "image": "$inputs.image",
                    "timeline": "$inputs.timeline",
                    "delay_seconds": 1,
                }
            ],
            "outputs": [
                {
                    "type": "JsonField",
                    "name": field,
                    "selector": f"$steps.display.{field}",
                }
                for field in ["image", "status", "actions", "displayed_frame_number"]
            ],
        }
    )

    def run(number, timeline, serialize=False):
        image = _image(number)
        return engine.run(
            runtime_parameters={
                "image": [
                    {
                        "type": "numpy_object",
                        "value": image.numpy_image,
                        "video_metadata": image.video_metadata,
                    }
                ],
                "timeline": [timeline],
            },
            serialize_results=serialize,
        )[0]

    warmup = run(1, [])
    assert warmup["status"] == "buffering"
    assert isinstance(warmup["image"], WorkflowImageData)
    timeline = [_action("wave", 1, 5)]
    ready = run(31, timeline)
    assert ready["actions"] == ["wave"]
    assert ready["displayed_frame_number"] == 1
    assert ready["image"].video_metadata.frame_number == 1
    assert ready["image"].numpy_image[-1, -1, 0] == 1
    serialized = run(31, timeline, serialize=True)
    assert serialized["actions"] == ["wave"]
    assert serialized["displayed_frame_number"] == 1


def test_resolution_change_rewarms_and_releases_old_reservation():
    block = ActionRecognitionVisualizationBlockV1()
    block.run(_image(1, size=(180, 320)), [], delay_seconds=1)
    reservation = block._streams["demo"].reserved_bytes
    result = block.run(_image(31), [], delay_seconds=1)
    assert result["status"] == "buffering"
    assert block._streams["demo"].reserved_bytes < reservation
    result = block.run(_image(61), [], delay_seconds=1)
    assert result["displayed_frame_number"] == 31
    assert result["image"].numpy_image.shape == (90, 160, 3)


def test_stream_limit_does_not_silently_evict_an_active_stream():
    block = ActionRecognitionVisualizationBlockV1()
    for index in range(16):
        block.run(_image(1, video=str(index)), [], delay_seconds=1)
    with pytest.raises(MemoryError, match="Too many video streams"):
        block.run(_image(1, video="extra"), [], delay_seconds=1)
    result = block.run(_image(31, video="0"), [], delay_seconds=1)
    assert result["displayed_frame_number"] == 1
