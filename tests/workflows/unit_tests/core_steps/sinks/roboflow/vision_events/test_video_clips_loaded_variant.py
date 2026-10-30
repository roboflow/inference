"""Video clips on the Vision Events block variant the server loader selects.

Runs under both `ENABLE_TENSOR_DATA_REPRESENTATION` values: the loader picks
`v1` or `v1_tensor`, and the clip path must behave the same in both. The
package suite (`workflows/tests/unit_tests/core_steps/sinks/
test_roboflow_vision_events_video_clips.py`) covers every status path.
"""

from concurrent.futures import Future
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest import mock

import pytest

from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION
from inference.core.workflows.core_steps.loader import load_blocks

API_BASE_URL = "https://api.roboflow.test"
EVENTS_URL = f"{API_BASE_URL}/vision-events"
PREPARE_URL = f"{API_BASE_URL}/vision-events/videos/prepare-upload"
BLOCK_TYPE = "roboflow_core/roboflow_vision_events@v1"


def _loaded_block_class() -> Any:
    for block in load_blocks():
        manifest = block.get_manifest()
        if manifest.model_fields["type"].annotation.__args__[0] == BLOCK_TYPE:
            return block
    raise AssertionError(f"{BLOCK_TYPE} is not loaded")


class _Ticket:
    def release(self) -> None:
        pass


class _Decision:
    admitted = True
    reason = None

    def __init__(self) -> None:
        self.ticket = _Ticket()


class _Budget:
    def admit(self, capture: Any) -> _Decision:
        return _Decision()


class _PendingCapture:
    def __init__(self, start: datetime, end: datetime) -> None:
        self.future: Future = Future()
        self.requested_window = (start, end)
        self.available_from = start
        self.released = 0

    def release(self) -> None:
        self.released += 1


class _Anchor:
    availability = "ok"
    pre_roll_seconds = 5.0
    post_roll_seconds = 5.0

    def __init__(self, label: str) -> None:
        self.label = label
        self.frame_timestamp = datetime(2026, 10, 30, 12, 0, 0)
        self.captures: List[_PendingCapture] = []

    def capture(self, pre: float, post: float) -> _PendingCapture:
        capture = _PendingCapture(
            self.frame_timestamp - timedelta(seconds=pre),
            self.frame_timestamp + timedelta(seconds=post),
        )
        self.captures.append(capture)
        return capture


class _Uploader:
    def __init__(self) -> None:
        self.submissions: List[Dict[str, Any]] = []

    def submit(self, capture: Any, **kwargs: Any) -> Future:
        self.submissions.append(kwargs)
        return Future()

    def shutdown(self, deadline_seconds: float) -> None:
        pass


def _platform_client() -> mock.MagicMock:
    client = mock.MagicMock()
    client.build_api_headers.side_effect = lambda explicit_headers=None: dict(
        explicit_headers or {}
    )
    client.wrap_url.side_effect = lambda url: url
    return client


def _run(
    block_class: Any, *, disable_sinks: bool = False, **run_parameters: Any
) -> Dict[str, Any]:
    block = block_class(
        api_key="my-api-key",
        background_tasks=None,
        thread_pool_executor=None,
        disable_sinks=disable_sinks,
        platform_client=_platform_client(),
    )
    parameters = {
        "input_image": None,
        "output_image": None,
        "predictions": None,
        "event_type": "custom",
        "solution": "my-use-case",
        "custom_metadata": {},
        "fire_and_forget": False,
        "disable_sink": False,
        "cooldown_seconds": 0,
    }
    parameters.update(run_parameters)
    module = __import__(block_class.__module__, fromlist=["API_BASE_URL"])
    with mock.patch.object(module, "API_BASE_URL", API_BASE_URL):
        result = block.run(**parameters)
    return result


@pytest.fixture
def clip_runtime(monkeypatch: pytest.MonkeyPatch) -> _Uploader:
    from roboflow_workflows.core_steps.sinks.roboflow.vision_events import (
        clip_attachments,
    )

    uploader = _Uploader()
    monkeypatch.setattr(clip_attachments, "default_clip_budget", lambda: _Budget())
    monkeypatch.setattr(clip_attachments, "create_clip_uploader", lambda: uploader)
    return uploader


def test_loader_selects_the_variant_of_the_tensor_mode() -> None:
    # when
    module_leaf = _loaded_block_class().__module__.rsplit(".", 1)[-1]

    # then
    assert module_leaf == ("v1_tensor" if ENABLE_TENSOR_DATA_REPRESENTATION else "v1")


def test_loaded_variant_attaches_clips_in_order(
    requests_mock: Any, clip_runtime: _Uploader
) -> None:
    # given
    requests_mock.post(EVENTS_URL, json={"success": True}, status_code=201)
    requests_mock.post(
        PREPARE_URL,
        [
            {
                "status_code": 201,
                "json": {
                    "videoId": f"vid-{index}",
                    "signedUrl": f"https://storage.test/{index}?sig=secret",
                    "requiredHeaders": {"x-goog-if-generation-match": "0"},
                },
            }
            for index in (1, 2)
        ],
    )
    anchors = [_Anchor("Raw"), _Anchor("")]

    # when
    result = _run(
        _loaded_block_class(), video_clips=anchors, video_label="Fallback"
    )

    # then
    event = [r for r in requests_mock.request_history if r.url == EVENTS_URL][0]
    assert event.json()["videos"] == [
        {"sourceId": "vid-1", "label": "Raw"},
        {"sourceId": "vid-2", "label": "Fallback"},
    ]
    assert [video["video_status"] for video in result["videos"]] == [
        "pending",
        "pending",
    ]
    assert [submission["video_id"] for submission in clip_runtime.submissions] == [
        "vid-1",
        "vid-2",
    ]


@pytest.mark.parametrize("disable_sinks", [True, False])
def test_loaded_variant_disabled_sink_captures_nothing(
    requests_mock: Any, clip_runtime: _Uploader, disable_sinks: bool
) -> None:
    # given
    anchor = _Anchor("Raw")

    # when
    result = _run(
        _loaded_block_class(),
        disable_sinks=disable_sinks,
        video_clips=[anchor],
        disable_sink=not disable_sinks,
    )

    # then
    assert anchor.captures == []
    assert result["videos"][0]["video_status"] == "disabled"
    assert requests_mock.request_history == []


def test_loaded_variant_without_clips_sends_no_videos_field(
    requests_mock: Any, clip_runtime: _Uploader
) -> None:
    # given
    requests_mock.post(EVENTS_URL, json={"success": True}, status_code=201)

    # when
    result = _run(_loaded_block_class(), video_clips=None)

    # then
    assert result["videos"] == []
    assert "videos" not in requests_mock.request_history[0].json()
    assert clip_runtime.submissions == []
