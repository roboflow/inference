"""Rendering and connection contracts for action-recognition overlays."""

from datetime import datetime

import cv2
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


def _image(tensor=False, size=(720, 1280), video="demo", frame=900):
    pixels = np.full((*size, 3), 90, dtype=np.uint8)
    data = (
        {"tensor_image": torch.from_numpy(pixels).permute(2, 0, 1)}
        if tensor
        else {"numpy_image": pixels}
    )
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="frame"),
        video_metadata=VideoMetadata(
            video_identifier=video,
            frame_number=frame,
            frame_timestamp=datetime(2026, 1, 1),
            fps=30,
        ),
        **data,
    )


def _window(**changes):
    return (
        dict(
            status="ready",
            classes=["punch", "hook"],
            start_frame=360,
            end_frame=840,
            fps=30,
            video_identifier="demo",
        )
        | changes
    )


def _segment(name, start, end):
    return ActionRecognitionPrediction(
        class_name=name, class_id=0, start_frame_idx=start, end_frame_idx=end
    )


def test_manifest_defaults_and_requires_history_only_in_timeline_mode():
    inputs = dict(
        type=TYPE,
        name="visualize",
        image="$inputs.image",
        window="$steps.actions.window",
    )
    assert BlockManifest(**inputs).mode == "compact"
    with pytest.raises(ValidationError, match="requires"):
        BlockManifest(**inputs, mode="timeline")
    assert (
        BlockManifest(
            **inputs, mode="timeline", timeline="$steps.actions.timeline"
        ).timeline
        == "$steps.actions.timeline"
    )


@pytest.mark.parametrize(
    "tensor,mode,size",
    [
        (False, "compact", (720, 1280)),
        (True, "compact", (180, 320)),
        (False, "timeline", (180, 320)),
        (True, "timeline", (640, 360)),
    ],
)
def test_render_preserves_size_metadata_and_input_and_needs_no_history_state(
    tensor, mode, size
):
    image = _image(tensor=tensor, size=size)
    original = image.numpy_image.copy()
    history = [_segment("punch", 390, 510), _segment("hook", 540, 630)]
    block = ActionRecognitionVisualizationBlockV1()
    output = block.run(image, _window(), mode=mode, timeline=history)["image"]
    assert output.numpy_image.shape == original.shape
    assert output.video_metadata == image.video_metadata
    assert output.parent_metadata == image.parent_metadata
    assert np.any(output.numpy_image != original)
    np.testing.assert_array_equal(image.numpy_image, original)
    # Interleaving another stream cannot contaminate the first stream's output.
    block.run(
        _image(video="other"), _window(video_identifier="other", classes=["other"])
    )
    again = block.run(image, _window(), mode=mode, timeline=history)["image"]
    np.testing.assert_array_equal(output.numpy_image, again.numpy_image)


@pytest.mark.parametrize(
    "status,classes,expected",
    [
        ("collecting", [], "Collecting context"),
        ("error", [], "Last analysis failed"),
        ("ready", [], "No actions reported"),
        ("ready", ["punch", "hook"], "punch"),
    ],
)
@pytest.mark.parametrize("mode", ["compact", "timeline"])
def test_states_do_not_show_stale_history_or_confidence(
    monkeypatch, status, classes, expected, mode
):
    labels = []
    original = cv2.putText

    def record(img, text, *args, **kwargs):
        labels.append(text)
        return original(img, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", record)
    ActionRecognitionVisualizationBlockV1().run(
        _image(),
        _window(status=status, classes=classes),
        mode=mode,
        timeline=[_segment("old", 370, 400)],
    )
    assert any(expected in text for text in labels)
    assert not any("%" in text or text == "1.0" for text in labels)
    if not classes:
        assert not any(text == "old" for text in labels)


def test_timeline_pixels_clip_history_and_leave_the_unanalyzed_tail_clear():
    history = [
        _segment("punch", 100, 400),
        _segment("hook", 390, 600),
        _segment("punch", 840, 900),
        _segment("future", 841, 850),
    ]
    block = ActionRecognitionVisualizationBlockV1()
    image = _image()
    actual = block.run(image, _window(), mode="timeline", timeline=history)[
        "image"
    ].numpy_image
    # Crossing ranges and entirely future history must render exactly as
    # already-clipped history, including one frame at the inclusive window end.
    expected = block.run(
        image,
        _window(),
        mode="timeline",
        timeline=[
            _segment("punch", 360, 400),
            _segment("hook", 390, 600),
            _segment("punch", 840, 840),
        ],
    )["image"].numpy_image
    np.testing.assert_array_equal(actual, expected)
    assert history[-2].end_frame_idx == 900
    without_last_frame = block.run(
        image, _window(), mode="timeline", timeline=history[:2]
    )["image"].numpy_image
    assert np.any(actual != without_last_frame)
    # Both rows have simultaneous evidence, but neither class's bar can reach
    # the far-right tail representing frames after the analyzed window.
    hook_color, punch_color = (255, 201, 142), (193, 240, 126)
    hook_x = np.where(np.all(actual[570] == hook_color, axis=1))[0]
    punch_x = np.where(np.all(actual[610] == punch_color, axis=1))[0]
    assert len(set(hook_x) & set(punch_x)) > 0
    assert hook_x.max() < 1117
    assert punch_x.max() <= 1117
    assert not np.any(np.all(actual[550:630, 1130:1220] == punch_color, axis=2))


def test_frames_fallback_and_overflow_are_explicit(monkeypatch):
    labels = []
    monkeypatch.setattr(cv2, "putText", lambda img, text, *a, **k: labels.append(text))
    ActionRecognitionVisualizationBlockV1().run(
        _image(),
        _window(
            fps=None,
            classes=["a very long action label " * 40]
            + [f"class-{i}" for i in range(10)],
        ),
        max_classes=2,
    )
    assert "f360 - f840" in labels
    assert any("more" in text for text in labels)
    assert all(len(text) < 100 for text in labels)
    assert not any("~" in text for text in labels)


def test_rejects_different_stream_or_image_before_window():
    block = ActionRecognitionVisualizationBlockV1()
    with pytest.raises(ValueError, match="same video"):
        block.run(_image(video="other"), _window())
    with pytest.raises(ValueError, match="precedes"):
        block.run(_image(frame=20), _window())


def test_in_place_render_invalidates_the_tensor_cache():
    image = _image(tensor=True)
    before = image.tensor_image.clone()
    output = ActionRecognitionVisualizationBlockV1().run(
        image, _window(), copy_image=False
    )["image"]
    assert not torch.equal(image.tensor_image, before)
    np.testing.assert_array_equal(image.numpy_image, output.numpy_image)


@pytest.mark.parametrize(
    "frame,label", [(1799, "~01:00.0"), (1798, "~00:59.9"), (1800, "~01:00.0")]
)
def test_displayed_time_rounds_across_minutes(monkeypatch, frame, label):
    labels = []
    original = cv2.putText

    def record(img, text, *args, **kwargs):
        labels.append(text)
        return original(img, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", record)
    ActionRecognitionVisualizationBlockV1().run(_image(frame=frame), _window())
    assert any(text.endswith("Frame " + label) for text in labels)
