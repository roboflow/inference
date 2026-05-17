from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.foundation.action_recognition.v1 import (
    ActionRecognitionBlockV1,
    BlockManifest,
    _parse_letter,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)


def _make_image(
    width: int = 320,
    height: int = 240,
    video_id: str = "cam-1",
    frame_number: int = 0,
    fps: int = 30,
) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id=video_id),
        numpy_image=np.random.randint(0, 255, (height, width, 3), dtype=np.uint8),
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
        "type": "roboflow_core/action_recognition@v1",
        "name": "ar",
        "image": "$inputs.image",
        "prompt": "A: stroke. B: no stroke.",
    }
    result = BlockManifest.model_validate(raw)
    assert result.choices == ["A", "B"]
    assert result.timeout_seconds == 0.5
    assert result.max_frames == 5
    assert result.base_url == "https://api.together.xyz/v1"
    assert result.model_name == "Qwen/Qwen3.5-9B"


def test_manifest_rejects_zero_max_frames() -> None:
    raw = {
        "type": "roboflow_core/action_recognition@v1",
        "name": "ar",
        "image": "$inputs.image",
        "prompt": "x",
        "max_frames": 0,
    }
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(raw)


def test_manifest_rejects_empty_choices() -> None:
    raw = {
        "type": "roboflow_core/action_recognition@v1",
        "name": "ar",
        "image": "$inputs.image",
        "prompt": "x",
        "choices": [],
    }
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(raw)


def test_manifest_rejects_negative_timeout() -> None:
    raw = {
        "type": "roboflow_core/action_recognition@v1",
        "name": "ar",
        "image": "$inputs.image",
        "prompt": "x",
        "timeout_seconds": -0.5,
    }
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(raw)


# ── _parse_letter ───────────────────────────────────────────────────


def test_parse_letter_json() -> None:
    assert _parse_letter('{"letter": "A"}', ["A", "B"]) == "A"


def test_parse_letter_json_with_whitespace() -> None:
    assert _parse_letter('{\n  "letter": "B"\n}', ["A", "B"]) == "B"


def test_parse_letter_bare_letter_fallback() -> None:
    assert _parse_letter("A", ["A", "B"]) == "A"


def test_parse_letter_rejects_out_of_set() -> None:
    assert _parse_letter('{"letter": "Z"}', ["A", "B"]) is None


def test_parse_letter_handles_empty() -> None:
    assert _parse_letter("", ["A", "B"]) is None
    assert _parse_letter(None, ["A", "B"]) is None


# ── Block behavior ───────────────────────────────────────────────────


def _run(
    block,
    image,
    *,
    prompt="A: stroke. B: no stroke.",
    choices=("A", "B"),
    timeout_seconds=0.5,
    max_frames=5,
    base_url="http://stub/v1",
    model_name="m",
    api_key=None,
    resolution=128,
):
    return block.run(
        image=image,
        prompt=prompt,
        choices=list(choices),
        timeout_seconds=timeout_seconds,
        max_frames=max_frames,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        resolution=resolution,
    )


def _stub_openai_response(content_str):
    msg = MagicMock()
    msg.content = content_str
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@patch("inference.core.workflows.core_steps.models.foundation.action_recognition.v1.OpenAI")
def test_first_call_fires_llm(mock_openai_cls: MagicMock) -> None:
    """The first frame should always trigger an LLM call."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _stub_openai_response('{"letter": "A"}')
    mock_openai_cls.return_value = mock_client

    block = ActionRecognitionBlockV1()
    out = _run(block, _make_image(frame_number=0))
    assert out["letter"] == "A"
    assert out["error_status"] == ""
    assert mock_client.chat.completions.create.call_count == 1


@patch("inference.core.workflows.core_steps.models.foundation.action_recognition.v1.OpenAI")
def test_throttled_frame_returns_cached_letter_without_calling_llm(
    mock_openai_cls: MagicMock,
) -> None:
    """Within the cooldown window, no new LLM call. Letter persists."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _stub_openai_response('{"letter": "A"}')
    mock_openai_cls.return_value = mock_client

    block = ActionRecognitionBlockV1()
    # Frame 0 fires.
    _run(block, _make_image(frame_number=0))
    # Frame 1 (33ms later at 30 fps) should be throttled by default 0.5s cooldown.
    out = _run(block, _make_image(frame_number=1))
    assert out["letter"] == "A"
    assert mock_client.chat.completions.create.call_count == 1


@patch("inference.core.workflows.core_steps.models.foundation.action_recognition.v1.OpenAI")
def test_fires_again_after_cooldown_elapsed(mock_openai_cls: MagicMock) -> None:
    """After enough video time has passed, a new call fires."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _stub_openai_response('{"letter": "A"}'),
        _stub_openai_response('{"letter": "B"}'),
    ]
    mock_openai_cls.return_value = mock_client

    block = ActionRecognitionBlockV1()
    # frame_number 0 at fps 30 => video time 0.0
    _run(block, _make_image(frame_number=0), timeout_seconds=0.1)
    # frame_number 6 at fps 30 => video time 0.2 — past 0.1s cooldown
    out = _run(block, _make_image(frame_number=6), timeout_seconds=0.1)
    assert mock_client.chat.completions.create.call_count == 2
    assert out["letter"] == "B"


@patch("inference.core.workflows.core_steps.models.foundation.action_recognition.v1.OpenAI")
def test_buffer_capped_at_max_frames(mock_openai_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _stub_openai_response('{"letter": "A"}')
    mock_openai_cls.return_value = mock_client

    block = ActionRecognitionBlockV1()
    for i in range(20):
        _run(block, _make_image(frame_number=i), max_frames=3, timeout_seconds=0.0)

    state = block._states["cam-1"]
    assert len(state.buffer) == 3


@patch("inference.core.workflows.core_steps.models.foundation.action_recognition.v1.OpenAI")
def test_isolated_state_per_video(mock_openai_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _stub_openai_response('{"letter": "A"}')
    mock_openai_cls.return_value = mock_client

    block = ActionRecognitionBlockV1()
    _run(block, _make_image(video_id="cam-A", frame_number=0))
    _run(block, _make_image(video_id="cam-B", frame_number=0))
    assert "cam-A" in block._states
    assert "cam-B" in block._states
    assert block._states["cam-A"] is not block._states["cam-B"]


@patch("inference.core.workflows.core_steps.models.foundation.action_recognition.v1.OpenAI")
def test_extra_body_includes_response_format_with_correct_enum(
    mock_openai_cls: MagicMock,
) -> None:
    """The forwarded extra_body must contain a json_schema response_format
    whose enum exactly matches the user-supplied `choices`."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _stub_openai_response('{"letter": "C"}')
    mock_openai_cls.return_value = mock_client

    block = ActionRecognitionBlockV1()
    _run(block, _make_image(frame_number=0), choices=["A", "B", "C", "D"])

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    extra = call_kwargs["extra_body"]
    enum_in_schema = (
        extra["response_format"]["json_schema"]["schema"]["properties"]["letter"][
            "enum"
        ]
    )
    assert enum_in_schema == ["A", "B", "C", "D"]
    assert extra["chat_template_kwargs"] == {"enable_thinking": False}


@patch("inference.core.workflows.core_steps.models.foundation.action_recognition.v1.OpenAI")
def test_thinking_is_always_disabled(mock_openai_cls: MagicMock) -> None:
    """Thinking-disable is hardcoded — every LLM call should carry it."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _stub_openai_response('{"letter": "A"}')
    mock_openai_cls.return_value = mock_client

    block = ActionRecognitionBlockV1()
    _run(block, _make_image(frame_number=0))
    extra = mock_client.chat.completions.create.call_args.kwargs["extra_body"]
    assert extra["chat_template_kwargs"] == {"enable_thinking": False}


@patch("inference.core.workflows.core_steps.models.foundation.action_recognition.v1.OpenAI")
def test_llm_error_returns_last_letter_and_error_status(
    mock_openai_cls: MagicMock,
) -> None:
    """If a fresh fire fails, we still return the most recent good letter and
    surface the error_status."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _stub_openai_response('{"letter": "A"}'),
        RuntimeError("boom"),
    ]
    mock_openai_cls.return_value = mock_client

    block = ActionRecognitionBlockV1()
    _run(block, _make_image(frame_number=0), timeout_seconds=0.0)
    out = _run(block, _make_image(frame_number=10), timeout_seconds=0.0)
    assert out["letter"] == "A"
    assert "boom" in out["error_status"]


@patch("inference.core.workflows.core_steps.models.foundation.action_recognition.v1.OpenAI")
def test_message_carries_one_image_url_per_buffered_frame(
    mock_openai_cls: MagicMock,
) -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _stub_openai_response('{"letter": "A"}')
    mock_openai_cls.return_value = mock_client

    block = ActionRecognitionBlockV1()
    # Push 5 frames to fill the buffer with timeout=0 to fire on every call.
    for i in range(5):
        _run(block, _make_image(frame_number=i), max_frames=5, timeout_seconds=0.0)

    last_call = mock_client.chat.completions.create.call_args
    messages = last_call.kwargs["messages"]
    content = messages[0]["content"]
    image_parts = [c for c in content if c["type"] == "image_url"]
    text_parts = [c for c in content if c["type"] == "text"]
    assert len(image_parts) == 5
    assert len(text_parts) == 1
    for ip in image_parts:
        assert ip["image_url"]["url"].startswith("data:image/jpeg;base64,")
