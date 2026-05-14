from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.foundation.action_recognition.v2 import (
    ActionRecognitionBlockV2,
    BlockManifest,
    _parse_letter,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)


def _make_image(
    video_id: str = "cam-1",
    frame_number: int = 0,
    fps: int = 30,
) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id=video_id),
        numpy_image=np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
        video_metadata=VideoMetadata(
            video_identifier=video_id,
            frame_number=frame_number,
            frame_timestamp=datetime.now(),
            fps=fps,
            comes_from_video_file=None,
        ),
    )


# ── Manifest validation ─────────────────────────────────────────────


def test_manifest_defaults() -> None:
    raw = {
        "type": "roboflow_core/action_recognition@v2",
        "name": "ar",
        "image": "$inputs.image",
        "prompt": "A: dunk. B: no dunk.",
    }
    result = BlockManifest.model_validate(raw)
    assert result.window_seconds == 1.0
    assert result.stride_seconds is None  # defaults to window/2 at runtime
    assert result.sample_fps == 5.0
    assert result.choices == ["A", "B"]
    assert result.max_tokens == 500


def test_manifest_rejects_window_too_small() -> None:
    with pytest.raises(ValidationError):
        BlockManifest.model_validate({
            "type": "roboflow_core/action_recognition@v2",
            "name": "ar",
            "image": "$inputs.image",
            "prompt": "x",
            "window_seconds": 0.0,
        })


def test_manifest_rejects_stride_too_small() -> None:
    with pytest.raises(ValidationError):
        BlockManifest.model_validate({
            "type": "roboflow_core/action_recognition@v2",
            "name": "ar",
            "image": "$inputs.image",
            "prompt": "x",
            "stride_seconds": 0.0,
        })


def test_manifest_rejects_invalid_sample_fps() -> None:
    with pytest.raises(ValidationError):
        BlockManifest.model_validate({
            "type": "roboflow_core/action_recognition@v2",
            "name": "ar",
            "image": "$inputs.image",
            "prompt": "x",
            "sample_fps": 0.0,
        })


def test_manifest_rejects_empty_choices() -> None:
    with pytest.raises(ValidationError):
        BlockManifest.model_validate({
            "type": "roboflow_core/action_recognition@v2",
            "name": "ar",
            "image": "$inputs.image",
            "prompt": "x",
            "choices": [],
        })


# ── _parse_letter ───────────────────────────────────────────────────


def test_parse_letter_json() -> None:
    assert _parse_letter('{"letter": "A"}', ["A", "B"]) == "A"


def test_parse_letter_bare() -> None:
    assert _parse_letter("B", ["A", "B"]) == "B"


def test_parse_letter_out_of_set() -> None:
    assert _parse_letter('{"letter": "Z"}', ["A", "B"]) is None


def test_parse_letter_empty() -> None:
    assert _parse_letter("", ["A", "B"]) is None
    assert _parse_letter(None, ["A", "B"]) is None


# ── Block behavior ───────────────────────────────────────────────────


def _stub_response(content: str):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _run(
    block,
    image,
    *,
    prompt="A: dunk. B: no dunk.",
    choices=("A", "B"),
    window_seconds=1.0,
    stride_seconds=0.5,
    sample_fps=5.0,
    base_url="http://stub/v1",
    model_name="m",
    api_key=None,
    resolution=128,
    max_tokens=500,
):
    return block.run(
        image=image,
        prompt=prompt,
        choices=list(choices),
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        sample_fps=sample_fps,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        resolution=resolution,
        max_tokens=max_tokens,
    )


@patch("inference.core.workflows.core_steps.models.foundation.action_recognition.v2.OpenAI")
def test_first_call_fires_immediately(mock_openai_cls: MagicMock) -> None:
    """Very first frame triggers a fire (no prior fire time)."""
    client = MagicMock()
    client.chat.completions.create.return_value = _stub_response('{"letter": "A"}')
    mock_openai_cls.return_value = client

    block = ActionRecognitionBlockV2()
    out = _run(block, _make_image(frame_number=0))
    assert out["letter"] == "A"
    assert client.chat.completions.create.call_count == 1


@patch("inference.core.workflows.core_steps.models.foundation.action_recognition.v2.OpenAI")
def test_within_stride_does_not_refire(mock_openai_cls: MagicMock) -> None:
    """Frames within stride_seconds of the last fire don't trigger new LLM calls."""
    client = MagicMock()
    client.chat.completions.create.return_value = _stub_response('{"letter": "A"}')
    mock_openai_cls.return_value = client

    block = ActionRecognitionBlockV2()
    # frame 0 at fps 30 → video_time 0
    _run(block, _make_image(frame_number=0), stride_seconds=0.5)
    # frame 3 at fps 30 → video_time 0.1, well under 0.5s stride
    out = _run(block, _make_image(frame_number=3), stride_seconds=0.5)
    assert client.chat.completions.create.call_count == 1
    # letter is still cached
    assert out["letter"] == "A"


@patch("inference.core.workflows.core_steps.models.foundation.action_recognition.v2.OpenAI")
def test_fires_again_after_stride(mock_openai_cls: MagicMock) -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _stub_response('{"letter": "A"}'),
        _stub_response('{"letter": "B"}'),
    ]
    mock_openai_cls.return_value = client

    block = ActionRecognitionBlockV2()
    _run(block, _make_image(frame_number=0), stride_seconds=0.5)
    # frame 16 at fps 30 → video_time ≈ 0.533, past 0.5s stride
    out = _run(block, _make_image(frame_number=16), stride_seconds=0.5)
    assert client.chat.completions.create.call_count == 2
    assert out["letter"] == "B"


@patch("inference.core.workflows.core_steps.models.foundation.action_recognition.v2.OpenAI")
def test_default_stride_is_half_window(mock_openai_cls: MagicMock) -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _stub_response('{"letter": "A"}')
    mock_openai_cls.return_value = client

    block = ActionRecognitionBlockV2()
    # window=1.0, stride=None → effective stride=0.5
    _run(block, _make_image(frame_number=0), window_seconds=1.0, stride_seconds=None)
    # frame 10 (video_time 0.333) — under 0.5s, no refire
    _run(block, _make_image(frame_number=10), window_seconds=1.0, stride_seconds=None)
    assert client.chat.completions.create.call_count == 1
    # frame 16 (video_time 0.533) — past 0.5s, refires
    _run(block, _make_image(frame_number=16), window_seconds=1.0, stride_seconds=None)
    assert client.chat.completions.create.call_count == 2


@patch("inference.core.workflows.core_steps.models.foundation.action_recognition.v2.OpenAI")
def test_subsample_fps_caps_buffer_growth(mock_openai_cls: MagicMock) -> None:
    """At source 30 fps, sample_fps=5, only every 6th frame is encoded."""
    client = MagicMock()
    client.chat.completions.create.return_value = _stub_response('{"letter": "A"}')
    mock_openai_cls.return_value = client

    block = ActionRecognitionBlockV2()
    state_id = "cam-1"
    sampled = []
    prev_t = None
    for i in range(30):  # 1 second of source frames
        _run(
            block,
            _make_image(video_id=state_id, frame_number=i, fps=30),
            window_seconds=1.0,
            stride_seconds=0.5,
            sample_fps=5.0,
        )
        t = block._states[state_id].last_subsample_video_time
        if t != prev_t:
            sampled.append(i)
            prev_t = t

    # 5 samples per second target; with 30 source frames we expect ~5 samples.
    # Allow some tolerance for the 1/5 = 0.2s interval landing at frame
    # boundaries (33ms each).
    assert 4 <= len(sampled) <= 6, sampled


@patch("inference.core.workflows.core_steps.models.foundation.action_recognition.v2.OpenAI")
def test_window_evicts_old_frames(mock_openai_cls: MagicMock) -> None:
    """Frames older than window_seconds are pruned from the buffer."""
    client = MagicMock()
    client.chat.completions.create.return_value = _stub_response('{"letter": "A"}')
    mock_openai_cls.return_value = client

    block = ActionRecognitionBlockV2()
    state_id = "cam-1"
    # Feed 60 frames at fps 30 = 2 seconds of video.
    # window_seconds=0.5 so buffer should hold ~0.5 × sample_fps = ~3 frames at steady state.
    for i in range(60):
        _run(
            block,
            _make_image(video_id=state_id, frame_number=i, fps=30),
            window_seconds=0.5,
            stride_seconds=0.5,
            sample_fps=5.0,
        )
    buffer_size = len(block._states[state_id].buffer)
    # ceil(0.5 * 5) = 3; allow ±1 for boundary effects
    assert 2 <= buffer_size <= 4, buffer_size


@patch("inference.core.workflows.core_steps.models.foundation.action_recognition.v2.OpenAI")
def test_isolated_state_per_video(mock_openai_cls: MagicMock) -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _stub_response('{"letter": "A"}')
    mock_openai_cls.return_value = client

    block = ActionRecognitionBlockV2()
    _run(block, _make_image(video_id="A", frame_number=0))
    _run(block, _make_image(video_id="B", frame_number=0))
    assert "A" in block._states and "B" in block._states
    assert block._states["A"] is not block._states["B"]


@patch("inference.core.workflows.core_steps.models.foundation.action_recognition.v2.OpenAI")
def test_extra_body_carries_enum_and_thinking_off(mock_openai_cls: MagicMock) -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _stub_response('{"letter": "B"}')
    mock_openai_cls.return_value = client

    block = ActionRecognitionBlockV2()
    _run(block, _make_image(frame_number=0), choices=["A", "B", "C"])

    extra = client.chat.completions.create.call_args.kwargs["extra_body"]
    assert extra["chat_template_kwargs"] == {"enable_thinking": False}
    enum_set = extra["response_format"]["json_schema"]["schema"]["properties"]["letter"]["enum"]
    assert enum_set == ["A", "B", "C"]


@patch("inference.core.workflows.core_steps.models.foundation.action_recognition.v2.OpenAI")
def test_llm_error_returns_last_good_letter(mock_openai_cls: MagicMock) -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _stub_response('{"letter": "A"}'),
        RuntimeError("boom"),
    ]
    mock_openai_cls.return_value = client

    block = ActionRecognitionBlockV2()
    _run(block, _make_image(frame_number=0), stride_seconds=0.0)
    # 16 frames later, force a refire; LLM raises; we should still see the prior 'A'.
    out = _run(block, _make_image(frame_number=16), stride_seconds=0.0)
    assert out["letter"] == "A"
    assert "boom" in out["error_status"]


@patch("inference.core.workflows.core_steps.models.foundation.action_recognition.v2.OpenAI")
def test_call_has_max_tokens(mock_openai_cls: MagicMock) -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _stub_response('{"letter": "A"}')
    mock_openai_cls.return_value = client

    block = ActionRecognitionBlockV2()
    _run(block, _make_image(frame_number=0), max_tokens=750)
    assert client.chat.completions.create.call_args.kwargs["max_tokens"] == 750
