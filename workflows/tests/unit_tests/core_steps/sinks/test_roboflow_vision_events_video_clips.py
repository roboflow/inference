"""Video clips on the Roboflow Vision Events sink, both variants.

The recorder, its anchors, the admission budget and the clip uploader are
fakes shaped like the shared `video_clips` interface, so these tests cover the
sink's side only: capture on the caller's thread, admission, prepare-upload,
the event's `videos` field, the hand-off and the immediate outputs.
"""

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest import mock
from urllib.parse import quote

import numpy as np
import pytest
import urllib3.filepost
from roboflow_workflows.core_steps.sinks.roboflow.vision_events import clip_attachments
from roboflow_workflows.core_steps.sinks.roboflow.vision_events import (
    v1 as vision_events_v1,
)
from roboflow_workflows.core_steps.sinks.roboflow.vision_events import (
    v1_tensor as vision_events_v1_tensor,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)

VISION_EVENTS_MODULES = [vision_events_v1, vision_events_v1_tensor]
API_BASE_URL = "https://api.roboflow.test"
EVENTS_URL = f"{API_BASE_URL}/vision-events"
IMAGE_UPLOAD_URL = f"{API_BASE_URL}/vision-events/upload"
PREPARE_URL = f"{API_BASE_URL}/vision-events/videos/prepare-upload"
SIGNATURE = "X-Goog-Signature=very-secret-signature"
ANCHOR_TIME = datetime(2026, 10, 30, 12, 0, 0)


class _Ticket:
    def __init__(self) -> None:
        self.released = 0

    def release(self) -> None:
        self.released += 1


class _Decision:
    def __init__(self, admitted: bool, reason: Optional[str] = None) -> None:
        self.admitted = admitted
        self.reason = reason
        self.ticket = _Ticket() if admitted else None


class FakeBudget:
    def __init__(self, reject_reason: Optional[str] = None) -> None:
        self.reject_reason = reject_reason
        self.decisions: List[_Decision] = []

    def admit(self, capture: "FakePendingCapture") -> _Decision:
        if self.reject_reason:
            decision = _Decision(admitted=False, reason=self.reject_reason)
            # The real budget releases a rejected capture itself.
            capture.release()
        else:
            decision = _Decision(admitted=True)
        self.decisions.append(decision)
        return decision


class FakePendingCapture:
    def __init__(self, anchor: "FakeAnchor", pre: float, post: float) -> None:
        self.anchor = anchor
        self.future: Future = Future()
        start = anchor.frame_timestamp - timedelta(seconds=pre)
        end = anchor.frame_timestamp + timedelta(seconds=post)
        self.requested_window = (start, end)
        self.available_from = start + timedelta(seconds=1)
        self.released = 0

    def requested_window_dict(self) -> Dict[str, str]:
        start, end = self.requested_window
        return {"start": start.isoformat(), "end": end.isoformat()}

    def release(self) -> None:
        self.released += 1


class FakeClipUnavailableError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class FakeAnchor:
    def __init__(
        self,
        label: str = "Raw",
        *,
        availability: str = "ok",
        pre_roll_seconds: float = 5.0,
        post_roll_seconds: float = 3.0,
        capture_error: Optional[Exception] = None,
    ) -> None:
        self.recorder_step = f"recorder_{label}"
        self.video_identifier = "0"
        self.frame_timestamp = ANCHOR_TIME
        self.frame_number = 42
        self.availability = availability
        self.label = label
        self.pre_roll_seconds = pre_roll_seconds
        self.post_roll_seconds = post_roll_seconds
        self.capture_error = capture_error
        self.captures: List[FakePendingCapture] = []

    def capture(self, pre: float, post: float) -> FakePendingCapture:
        if self.capture_error is not None:
            raise self.capture_error
        pending_capture = FakePendingCapture(self, pre, post)
        self.captures.append(pending_capture)
        return pending_capture


class FakeUploader:
    def __init__(self, submit_error: Optional[Exception] = None) -> None:
        self.submissions: List[Dict[str, Any]] = []
        self.shutdowns: List[float] = []
        self.submit_error = submit_error

    def submit(self, capture: FakePendingCapture, **kwargs: Any) -> Future:
        if self.submit_error is not None:
            raise self.submit_error
        self.submissions.append({"capture": capture, **kwargs})
        return Future()

    def shutdown(self, deadline_seconds: float) -> None:
        self.shutdowns.append(deadline_seconds)


def _platform_client() -> mock.MagicMock:
    client = mock.MagicMock()
    client.build_api_headers.side_effect = lambda explicit_headers=None: dict(
        explicit_headers or {}
    )
    client.wrap_url.side_effect = lambda url: url
    return client


def _image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((8, 12, 3), dtype=np.uint8),
    )


@pytest.fixture
def budget(monkeypatch: pytest.MonkeyPatch) -> FakeBudget:
    fake_budget = FakeBudget()
    monkeypatch.setattr(clip_attachments, "default_clip_budget", lambda: fake_budget)
    return fake_budget


@pytest.fixture
def uploader(monkeypatch: pytest.MonkeyPatch) -> FakeUploader:
    fake_uploader = FakeUploader()
    monkeypatch.setattr(clip_attachments, "create_clip_uploader", lambda: fake_uploader)
    return fake_uploader


@pytest.fixture
def platform(requests_mock: Any) -> Any:
    requests_mock.post(IMAGE_UPLOAD_URL, json={"sourceId": "img-1", "url": ""})
    requests_mock.post(EVENTS_URL, json={"success": True}, status_code=201)
    counter = {"n": 0}

    def _prepare(request: Any, context: Any) -> Dict[str, Any]:
        counter["n"] += 1
        context.status_code = 201
        return {
            "success": True,
            "videoId": f"vid-{counter['n']}",
            "signedUrl": f"https://storage.test/upload/{counter['n']}?{SIGNATURE}",
            "requiredHeaders": {
                "x-goog-content-length-range": "1,3221225472",
                "x-goog-if-generation-match": "0",
            },
        }

    requests_mock.post(PREPARE_URL, json=_prepare)
    return requests_mock


def _block(module: Any, **init_parameters: Any) -> Any:
    parameters = {
        "api_key": "my-api-key",
        "background_tasks": None,
        "thread_pool_executor": None,
        "platform_client": _platform_client(),
    }
    parameters.update(init_parameters)
    return module.RoboflowVisionEventsBlockV1(**parameters)


def _run(block: Any, module: Any, **run_parameters: Any) -> Dict[str, Any]:
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
        "custom_value": "clip",
    }
    parameters.update(run_parameters)
    with mock.patch.object(module, "API_BASE_URL", API_BASE_URL):
        result = block.run(**parameters)
    return result


def _requests_to(requests_mock: Any, url: str) -> List[Any]:
    return [request for request in requests_mock.request_history if request.url == url]


def _event_payload(requests_mock: Any) -> Dict[str, Any]:
    events = _requests_to(requests_mock, EVENTS_URL)
    assert len(events) == 1
    return events[0].json()


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz: Any = None) -> datetime:  # type: ignore[override]
        fixed = datetime(2026, 10, 30, 12, 0, 0, tzinfo=timezone.utc)
        return fixed if tz is not None else fixed.replace(tzinfo=None)


# === Compatibility: no-video requests are byte-identical ===


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
@pytest.mark.parametrize(
    "new_fields",
    [
        {"video_clips": None},
        {"video_clips": []},
        {
            "video_clips": None,
            "video_label": "Other",
            "confirm_ingestion": True,
            "clip_upload_timeout_seconds": 120.0,
        },
    ],
)
def test_no_video_requests_are_byte_identical(
    module: Any, new_fields: Dict[str, Any], platform: Any, budget: FakeBudget
) -> None:
    # given
    def _send(**extra: Any) -> List[tuple]:
        platform.reset_mock()
        with mock.patch.object(
            module, "uuid4", return_value="event-1"
        ), mock.patch.object(module, "datetime", _FixedDatetime), mock.patch.object(
            urllib3.filepost, "choose_boundary", return_value="fixed-boundary"
        ):
            result = _run(_block(module), module, input_image=_image(), **extra)
        assert result["error_status"] is False
        return [
            (request.method, request.url, sorted(request.headers.items()), request.body)
            for request in platform.request_history
        ]

    # when
    without_fields = _send()
    with_fields = _send(**new_fields)

    # then
    assert without_fields == with_fields
    assert len(without_fields) == 2
    assert b'"videos"' not in without_fields[-1][3]
    assert budget.decisions == []


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
def test_no_video_run_reports_empty_videos_output(module: Any, platform: Any) -> None:
    # when
    result = _run(_block(module), module)

    # then
    assert result["videos"] == []
    assert result["error_status"] is False


# === Pending path and the event request shape ===


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
def test_single_clip_is_captured_prepared_posted_and_handed_off(
    module: Any, platform: Any, budget: FakeBudget, uploader: FakeUploader
) -> None:
    # given
    anchor = FakeAnchor("Raw", pre_roll_seconds=7.5, post_roll_seconds=2.0)

    # when
    result = _run(
        _block(module), module, video_clips=anchor, clip_upload_timeout_seconds=600
    )

    # then
    assert result["error_status"] is False
    assert len(anchor.captures) == 1
    capture = anchor.captures[0]
    assert capture.requested_window == (
        ANCHOR_TIME - timedelta(seconds=7.5),
        ANCHOR_TIME + timedelta(seconds=2.0),
    )
    payload = _event_payload(platform)
    assert payload["videos"] == [{"sourceId": "vid-1", "label": "Raw"}]
    prepare = _requests_to(platform, PREPARE_URL)
    assert len(prepare) == 1
    assert prepare[0].json() == {
        "name": f"{payload['eventId']}-Raw.mp4",
        "contentType": "video/mp4",
    }
    assert prepare[0].headers["Authorization"] == "Bearer my-api-key"
    assert result["event_id"] == payload["eventId"]
    assert result["videos"] == [
        {
            "video_id": "vid-1",
            "label": "Raw",
            "video_status": "pending",
            "requested_window": capture.requested_window_dict(),
            "available_from": capture.available_from.replace(
                tzinfo=timezone.utc
            ).isoformat(),
        }
    ]
    assert len(uploader.submissions) == 1
    submission = uploader.submissions[0]
    assert submission["capture"] is capture
    assert submission["video_id"] == "vid-1"
    assert submission["signed_url"] == f"https://storage.test/upload/1?{SIGNATURE}"
    assert submission["required_headers"] == {
        "x-goog-content-length-range": "1,3221225472",
        "x-goog-if-generation-match": "0",
    }
    assert submission["event_id"] == payload["eventId"]
    assert submission["label"] == "Raw"
    assert 590 < submission["upload_timeout_seconds"] <= 600
    assert submission["admission_ticket"] is budget.decisions[0].ticket
    assert submission["confirm_ingestion"] is False
    assert submission["ingestion_probe"] is None
    assert capture.released == 0


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
def test_prepare_upload_goes_through_the_secure_gateway_wrapper(
    module: Any, platform: Any, budget: FakeBudget, uploader: FakeUploader
) -> None:
    # given
    platform_client = _platform_client()

    # when
    _run(
        _block(module, platform_client=platform_client),
        module,
        video_clips=FakeAnchor(),
    )

    # then
    wrapped = [call.args[0] for call in platform_client.wrap_url.call_args_list]
    assert PREPARE_URL in wrapped
    assert EVENTS_URL in wrapped


GATEWAY_PREFIX = "https://gateway.test/proxy?url="


def _gateway_wrap(url: str) -> str:
    # Wraps only the storage host so the mocked platform URLs still match.
    if not url.startswith("https://storage.test/"):
        return url
    return GATEWAY_PREFIX + quote(url, safe="~()*!'")


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
def test_signed_upload_url_goes_through_the_secure_gateway_wrapper(
    module: Any,
    platform: Any,
    budget: FakeBudget,
    uploader: FakeUploader,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # given
    platform_client = _platform_client()
    platform_client.wrap_url.side_effect = _gateway_wrap

    # when
    with caplog.at_level(logging.DEBUG):
        result = _run(
            _block(module, platform_client=platform_client),
            module,
            video_clips=FakeAnchor(),
        )

    # then: the uploader PUTs to the gateway, and the signature stays redacted
    signed_url = f"https://storage.test/upload/1?{SIGNATURE}"
    upload_url = uploader.submissions[0]["signed_url"]
    assert upload_url == _gateway_wrap(signed_url)
    assert upload_url.startswith(GATEWAY_PREFIX)
    assert signed_url in [
        call.args[0] for call in platform_client.wrap_url.call_args_list
    ]
    assert "very-secret" not in clip_attachments.redact_signed_urls(upload_url)
    assert "very-secret" not in str(result)
    assert "very-secret" not in caplog.text


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
@pytest.mark.parametrize("anchors_count", [1, 2, 4, 5])
def test_event_videos_keep_request_order_and_cap_at_four(
    module: Any,
    anchors_count: int,
    platform: Any,
    budget: FakeBudget,
    uploader: FakeUploader,
) -> None:
    # given
    labels = ["Raw", "", "Annotated", "   ", "Fifth"][:anchors_count]
    anchors = [FakeAnchor(label) for label in labels]

    # when
    result = _run(_block(module), module, video_clips=anchors, video_label="Fallback")

    # then
    kept = min(anchors_count, 4)
    expected_labels = ["Raw", "Fallback", "Annotated", "Fallback"][:kept]
    payload = _event_payload(platform)
    assert payload["videos"] == [
        {"sourceId": f"vid-{index + 1}", "label": label}
        for index, label in enumerate(expected_labels)
    ]
    assert [video["label"] for video in result["videos"]][:kept] == expected_labels
    assert [video["video_status"] for video in result["videos"]] == (
        ["pending"] * kept + ["rejected:too_many_videos"] * (anchors_count - kept)
    )
    assert [len(anchor.captures) for anchor in anchors] == (
        [1] * kept + [0] * (anchors_count - kept)
    )
    assert [submission["label"] for submission in uploader.submissions] == (
        expected_labels
    )
    assert len(_requests_to(platform, PREPARE_URL)) == kept


# === Disabled and throttled sinks capture nothing ===


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
@pytest.mark.parametrize("by_policy", [True, False])
def test_disabled_sink_captures_nothing(
    module: Any, by_policy: bool, platform: Any, budget: FakeBudget
) -> None:
    # given
    anchors = [FakeAnchor("Raw"), FakeAnchor("Annotated")]
    block = _block(module, disable_sinks=by_policy)

    # when
    result = _run(block, module, video_clips=anchors, disable_sink=not by_policy)

    # then
    assert [anchor.captures for anchor in anchors] == [[], []]
    assert [video["video_status"] for video in result["videos"]] == [
        "disabled",
        "disabled",
    ]
    assert [video["label"] for video in result["videos"]] == ["Raw", "Annotated"]
    assert platform.request_history == []
    assert budget.decisions == []


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
def test_throttled_sink_captures_nothing(
    module: Any, platform: Any, budget: FakeBudget, uploader: FakeUploader
) -> None:
    # given
    block = _block(module)
    first_anchor, second_anchor = FakeAnchor("Raw"), FakeAnchor("Raw")
    _run(block, module, video_clips=first_anchor, cooldown_seconds=60)

    # when
    result = _run(block, module, video_clips=second_anchor, cooldown_seconds=60)

    # then
    assert result["throttling_status"] is True
    assert len(first_anchor.captures) == 1
    assert second_anchor.captures == []
    assert result["videos"][0]["video_status"] == "throttled"
    assert result["videos"][0]["video_id"] == ""
    assert len(uploader.submissions) == 1


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
def test_event_store_mode_reports_clips_unsupported(
    module: Any, requests_mock: Any, budget: FakeBudget
) -> None:
    # given
    requests_mock.post("http://edge:8001/v2/events", json={"id": "e1"}, status_code=201)
    anchor = FakeAnchor()

    # when
    result = _run(
        _block(module),
        module,
        video_clips=anchor,
        write_to_event_store=True,
        event_store_url="http://edge:8001",
    )

    # then
    assert anchor.captures == []
    assert result["videos"][0]["video_status"] == "unavailable:unsupported_sink_mode"
    assert "videos" not in requests_mock.request_history[0].json()


# === Unavailable anchors ===


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
@pytest.mark.parametrize(
    "anchor, expected_status",
    [
        (FakeAnchor(availability="unavailable:no_encoder"), "unavailable:no_encoder"),
        (
            FakeAnchor(availability="unavailable:sinks_disabled"),
            "unavailable:sinks_disabled",
        ),
        (
            FakeAnchor(capture_error=FakeClipUnavailableError("unavailable:closing")),
            "unavailable:closing",
        ),
        (FakeAnchor(capture_error=RuntimeError("boom")), "unavailable:capture_failed"),
        (None, "unavailable:no_anchor"),
        ({"kind": "video_clip_anchor"}, "unavailable:invalid_anchor"),
    ],
)
def test_unavailable_anchor_is_reported_and_event_still_sent(
    module: Any,
    anchor: Any,
    expected_status: str,
    platform: Any,
    budget: FakeBudget,
    uploader: FakeUploader,
) -> None:
    # when
    result = _run(_block(module), module, video_clips=[anchor])

    # then
    assert result["error_status"] is False
    assert result["videos"][0]["video_status"] == expected_status
    assert result["videos"][0]["requested_window"] is None
    assert _requests_to(platform, PREPARE_URL) == []
    assert "videos" not in _event_payload(platform)
    assert uploader.submissions == []


# === Rejections ===


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
@pytest.mark.parametrize(
    "reason", ["budget_exceeded", "too_many_pending_captures", "clip_too_large"]
)
def test_admission_rejection_releases_capture_and_sends_event_without_video(
    module: Any, reason: str, platform: Any, budget: FakeBudget, uploader: FakeUploader
) -> None:
    # given
    budget.reject_reason = reason
    anchor = FakeAnchor()

    # when
    result = _run(_block(module), module, video_clips=anchor)

    # then
    assert result["videos"][0]["video_status"] == f"rejected:{reason}"
    assert anchor.captures[0].released >= 1
    assert _requests_to(platform, PREPARE_URL) == []
    assert "videos" not in _event_payload(platform)
    assert uploader.submissions == []


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
def test_prepare_failure_releases_capture_and_keeps_other_videos(
    module: Any, platform: Any, budget: FakeBudget, uploader: FakeUploader
) -> None:
    # given
    responses = [
        {"status_code": 500, "json": {"error": "boom"}},
        {
            "status_code": 201,
            "json": {
                "videoId": "vid-ok",
                "signedUrl": f"https://storage.test/ok?{SIGNATURE}",
                "requiredHeaders": {},
            },
        },
    ]
    platform.post(PREPARE_URL, responses)
    anchors = [FakeAnchor("Raw"), FakeAnchor("Annotated")]

    # when
    result = _run(_block(module), module, video_clips=anchors)

    # then
    assert [video["video_status"] for video in result["videos"]] == [
        "prepare_failed",
        "pending",
    ]
    assert anchors[0].captures[0].released == 1
    assert budget.decisions[0].ticket.released == 1
    assert _event_payload(platform)["videos"] == [
        {"sourceId": "vid-ok", "label": "Annotated"}
    ]
    assert [submission["video_id"] for submission in uploader.submissions] == ["vid-ok"]


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
def test_failed_event_releases_clips_and_logs_orphaned_ids(
    module: Any,
    platform: Any,
    budget: FakeBudget,
    uploader: FakeUploader,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # given
    platform.post(EVENTS_URL, status_code=500, text="server error")
    anchor = FakeAnchor()

    # when
    with caplog.at_level(logging.WARNING):
        result = _run(_block(module), module, video_clips=anchor)

    # then
    assert result["error_status"] is True
    assert result["videos"][0]["video_status"] == "rejected:event_failed"
    assert result["videos"][0]["video_id"] == "vid-1"
    assert anchor.captures[0].released == 1
    assert budget.decisions[0].ticket.released == 1
    assert uploader.submissions == []
    assert "orphaned video ids: vid-1" in caplog.text


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
def test_image_upload_failure_releases_clips(
    module: Any, platform: Any, budget: FakeBudget, uploader: FakeUploader
) -> None:
    # given
    platform.post(IMAGE_UPLOAD_URL, status_code=500, text="no")
    anchor = FakeAnchor()

    # when
    result = _run(_block(module), module, video_clips=anchor, input_image=_image())

    # then
    assert result["error_status"] is True
    assert result["videos"][0]["video_status"] == "rejected:event_failed"
    assert anchor.captures[0].released == 1
    assert uploader.submissions == []


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
def test_uploader_refusal_is_reported_and_releases_clip(
    module: Any,
    platform: Any,
    budget: FakeBudget,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # given
    refusing = FakeUploader(
        submit_error=RuntimeError(f"cannot PUT https://storage.test/x?{SIGNATURE}")
    )
    monkeypatch.setattr(clip_attachments, "create_clip_uploader", lambda: refusing)
    anchor = FakeAnchor()

    # when
    with caplog.at_level(logging.WARNING):
        result = _run(_block(module), module, video_clips=anchor)

    # then
    assert result["videos"][0]["video_status"] == "rejected:uploader_unavailable"
    assert anchor.captures[0].released == 1
    assert "very-secret-signature" not in caplog.text
    assert "https://storage.test/x?<redacted>" in caplog.text


# === Signed URL redaction ===


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
def test_signed_urls_never_reach_outputs_or_logs(
    module: Any,
    platform: Any,
    budget: FakeBudget,
    uploader: FakeUploader,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # given: the event POST fails with a body that echoes a signed URL
    platform.post(
        EVENTS_URL,
        status_code=400,
        text=f"bad video https://storage.test/upload/1?{SIGNATURE}",
    )

    # when
    with caplog.at_level(logging.DEBUG):
        result = _run(_block(module), module, video_clips=FakeAnchor())

    # then
    assert "very-secret-signature" not in str(result["videos"])
    assert "very-secret-signature" not in result["message"]
    assert "https://storage.test/upload/1?<redacted>" in result["message"]
    orphan_logs = [
        record.getMessage()
        for record in caplog.records
        if "orphaned video ids" in record.getMessage()
    ]
    assert orphan_logs
    assert all("very-secret-signature" not in line for line in orphan_logs)


def test_redact_signed_urls_strips_every_query_string() -> None:
    # given
    text = (
        f"PUT https://storage.googleapis.com/b/o?{SIGNATURE}&X-Goog-Date=1 failed; "
        "retry https://storage.test/p?token=abc"
    )

    # when
    redacted = clip_attachments.redact_signed_urls(text)

    # then
    assert "very-secret" not in redacted
    assert "token=abc" not in redacted
    assert "https://storage.googleapis.com/b/o?<redacted>" in redacted


# === Fire and forget ===


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
def test_fire_and_forget_captures_on_the_caller_thread(
    module: Any, platform: Any, budget: FakeBudget, uploader: FakeUploader
) -> None:
    # given
    thread_pool = mock.MagicMock()
    anchor = FakeAnchor()
    block = _block(module, thread_pool_executor=thread_pool)

    # when
    result = _run(block, module, video_clips=anchor, fire_and_forget=True)

    # then: captured and admitted now, prepared and posted by the task
    assert len(anchor.captures) == 1
    assert len(budget.decisions) == 1
    assert result["event_id"] == ""
    assert result["videos"][0]["video_status"] == "pending"
    assert result["videos"][0]["video_id"] == ""
    assert platform.request_history == []
    task = thread_pool.submit.call_args.args[0]
    with mock.patch.object(module, "API_BASE_URL", API_BASE_URL):
        task()
    assert _event_payload(platform)["videos"] == [{"sourceId": "vid-1", "label": "Raw"}]
    assert len(uploader.submissions) == 1


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
def test_close_releases_clips_of_an_event_task_that_never_ran(
    module: Any, platform: Any, budget: FakeBudget, uploader: FakeUploader
) -> None:
    # given: the pool never runs the queued task (cancelled at pipeline end)
    thread_pool = mock.MagicMock()
    anchor = FakeAnchor()
    block = _block(module, thread_pool_executor=thread_pool)
    _run(block, module, video_clips=anchor, fire_and_forget=True)
    assert block._clip_uploads.unsettled_count() == 1

    # when
    block.prepare_close(5.0)
    block.close()

    # then: the admission ticket and the capture (and its artifact) are released
    assert budget.decisions[0].ticket.released == 1
    assert anchor.captures[0].released == 1
    assert block._clip_uploads.unsettled_count() == 0
    assert uploader.submissions == []
    assert platform.request_history == []


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
def test_cancelled_event_task_releases_its_clips_at_once(
    module: Any, platform: Any, budget: FakeBudget, uploader: FakeUploader
) -> None:
    # given: a one-worker pool busy with another task
    thread_pool = ThreadPoolExecutor(max_workers=1)
    blocker = threading.Event()
    thread_pool.submit(blocker.wait)
    anchor = FakeAnchor()
    block = _block(module, thread_pool_executor=thread_pool)
    _run(block, module, video_clips=anchor, fire_and_forget=True)

    # when: the host cancels queued tasks
    thread_pool.shutdown(wait=False, cancel_futures=True)

    # then
    assert budget.decisions[0].ticket.released == 1
    assert anchor.captures[0].released == 1
    assert block._clip_uploads.unsettled_count() == 0
    blocker.set()
    thread_pool.shutdown(wait=True)
    block.close()
    assert anchor.captures[0].released == 1


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
def test_close_leaves_handed_off_clips_to_the_uploader(
    module: Any, platform: Any, budget: FakeBudget, uploader: FakeUploader
) -> None:
    # given
    thread_pool = mock.MagicMock()
    anchor = FakeAnchor()
    block = _block(module, thread_pool_executor=thread_pool)
    _run(block, module, video_clips=anchor, fire_and_forget=True)
    task = thread_pool.submit.call_args.args[0]
    with mock.patch.object(module, "API_BASE_URL", API_BASE_URL):
        task()

    # when
    block.close()

    # then: the uploader owns the capture and the ticket
    assert block._clip_uploads.unsettled_count() == 0
    assert len(uploader.submissions) == 1
    assert budget.decisions[0].ticket.released == 0
    assert anchor.captures[0].released == 0


# === Ingestion confirmation ===


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
@pytest.mark.parametrize(
    "platform_status, http_status, expected",
    [
        ("uploaded", 200, "confirmed"),
        ("failed", 200, "failed"),
        ("pending", 200, "pending"),
        (None, 404, "pending"),
    ],
)
def test_confirm_ingestion_hands_a_probe_to_the_uploader(
    module: Any,
    platform_status: Optional[str],
    http_status: int,
    expected: str,
    platform: Any,
    budget: FakeBudget,
    uploader: FakeUploader,
) -> None:
    # given
    platform.get(
        f"{API_BASE_URL}/vision-events/videos/vid-1",
        json={"status": platform_status},
        status_code=http_status,
    )
    _run(_block(module), module, video_clips=FakeAnchor(), confirm_ingestion=True)
    submission = uploader.submissions[0]

    # when
    ingestion = submission["ingestion_probe"]("vid-1")

    # then
    assert submission["confirm_ingestion"] is True
    assert ingestion == expected
    status_request = platform.request_history[-1]
    assert status_request.method == "GET"
    assert status_request.headers["Authorization"] == "Bearer my-api-key"


# === Lifecycle ===


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
def test_close_shuts_the_uploader_down_within_the_host_deadline(
    module: Any, platform: Any, budget: FakeBudget, uploader: FakeUploader
) -> None:
    # given
    block = _block(module)
    _run(block, module, video_clips=FakeAnchor())

    # when
    block.prepare_close(4.0)
    block.prepare_close(100.0)
    block.close()
    block.close()

    # then
    assert len(uploader.shutdowns) == 1
    assert 0 <= uploader.shutdowns[0] <= 4.0


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
def test_close_without_clips_creates_no_uploader(
    module: Any, platform: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # given
    factory = mock.MagicMock()
    monkeypatch.setattr(clip_attachments, "create_clip_uploader", factory)
    block = _block(module)
    _run(block, module)

    # when
    block.close()

    # then
    factory.assert_not_called()


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
def test_clip_after_prepare_close_is_rejected_and_released(
    module: Any, platform: Any, budget: FakeBudget, uploader: FakeUploader
) -> None:
    # given
    block = _block(module)
    block.prepare_close(5.0)
    anchor = FakeAnchor()

    # when
    result = _run(block, module, video_clips=anchor)

    # then
    assert result["videos"][0]["video_status"] == "rejected:uploader_unavailable"
    assert anchor.captures[0].released == 1
    assert _requests_to(platform, PREPARE_URL) == []
    assert uploader.submissions == []


# === Manifest ===


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
def test_manifest_accepts_one_anchor_or_a_list(module: Any) -> None:
    # given
    base = {
        "type": "roboflow_core/roboflow_vision_events@v1",
        "name": "sink",
        "event_type": "custom",
        "solution": "my-use-case",
    }

    # when
    single = module.BlockManifest.model_validate(
        {**base, "video_clips": "$steps.recorder.anchor"}
    )
    many = module.BlockManifest.model_validate(
        {**base, "video_clips": ["$steps.raw.anchor", "$steps.annotated.anchor"]}
    )
    defaults = module.BlockManifest.model_validate(base)

    # then
    assert single.video_clips == "$steps.recorder.anchor"
    assert many.video_clips == ["$steps.raw.anchor", "$steps.annotated.anchor"]
    assert defaults.video_clips is None
    assert defaults.video_label == "Evidence"
    assert defaults.confirm_ingestion is False
    assert defaults.clip_upload_timeout_seconds == 55 * 60
    assert "videos" in {
        output.name for output in module.BlockManifest.describe_outputs()
    }


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
@pytest.mark.parametrize(
    "field, value",
    [
        ("video_label", ""),
        ("clip_upload_timeout_seconds", 0),
        ("clip_upload_timeout_seconds", 3601),
    ],
)
def test_manifest_rejects_invalid_clip_settings(
    module: Any, field: str, value: Any
) -> None:
    # given
    raw = {
        "type": "roboflow_core/roboflow_vision_events@v1",
        "name": "sink",
        "event_type": "custom",
        "solution": "my-use-case",
        field: value,
    }

    # when / then
    with pytest.raises(Exception):
        module.BlockManifest.model_validate(raw)


# === With the real clip core (runs once `common.video_clips` is present) ===


def _real_clip_core() -> Any:
    return pytest.importorskip(
        "roboflow_workflows.core_steps.common.video_clips.uploader",
        reason="the video clips core (admission, capture, uploader) is not merged",
    )


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
@pytest.mark.parametrize(
    "put_response, clip_error, expected_terminal_status",
    [
        ({"status_code": 200}, False, "put_accepted"),
        ({"status_code": 412, "text": "PreconditionFailed"}, False, "put_accepted"),
        ({"status_code": 400, "text": "<Code>ExpiredToken</Code>"}, False, "expired"),
        (
            {"status_code": 400, "text": "MalformedSecurityHeader"},
            False,
            "upload_failed",
        ),
        ({"status_code": 200}, True, "clip_failed"),
    ],
)
def test_completion_record_per_terminal_status_with_the_real_uploader(
    module: Any,
    put_response: Dict[str, Any],
    clip_error: bool,
    expected_terminal_status: str,
    platform: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # given
    _real_clip_core()
    from roboflow_workflows.core_steps.common.video_clips.artifact import ClipArtifact
    from roboflow_workflows.core_steps.common.video_clips.capture import (
        ClipAnchor,
        ClipEncodeError,
        PendingCapture,
    )

    captures: List[Any] = []

    def _ring_capture(pre: float, post: float) -> Any:
        pending_capture = PendingCapture(
            capture_id="capture-1",
            video_identifier="0",
            backend="x264",
            requested_window=(
                ANCHOR_TIME - timedelta(seconds=pre),
                ANCHOR_TIME + timedelta(seconds=post),
            ),
            available_from=ANCHOR_TIME - timedelta(seconds=pre),
            estimated_size_bytes=1000,
        )
        captures.append(pending_capture)
        return pending_capture

    anchor = ClipAnchor(
        "recorder", "0", ANCHOR_TIME, 1, "ok", "Raw", 5.0, 2.0, _ring_capture
    )
    platform.put("https://storage.test/upload/1", **put_response)
    block = _block(module)

    # when
    result = _run(block, module, video_clips=anchor)
    report = mock.MagicMock()
    report.to_dict.return_value = {"frames": 10, "partial": None}
    with caplog.at_level(logging.INFO):
        if clip_error:
            captures[0].future.set_exception(ClipEncodeError("encode failed"))
        else:
            captures[0].future.set_result(
                ClipArtifact(
                    b"\x00" * 64, size_bytes=64, duration_seconds=7.0, report=report
                )
            )
        block.prepare_close(10.0)
        block.close()

    # then
    assert result["videos"][0]["video_status"] == "pending"
    records = [
        getattr(record, "vision_event_clip_completed")
        for record in caplog.records
        if hasattr(record, "vision_event_clip_completed")
    ]
    assert len(records) == 1
    assert records[0]["video_id"] == "vid-1"
    assert records[0]["event_id"] == result["event_id"]
    assert records[0]["label"] == "Raw"
    assert records[0]["terminal_status"] == expected_terminal_status
    assert records[0]["ingestion"] is None
    assert "very-secret-signature" not in caplog.text
    puts = [request for request in platform.request_history if request.method == "PUT"]
    if not clip_error:
        assert puts[0].headers["x-goog-if-generation-match"] == "0"
        assert puts[0].headers["x-goog-content-length-range"] == "1,3221225472"
