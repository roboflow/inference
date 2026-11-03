import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote

import pytest
import requests
from roboflow_workflows.core_steps.common.video_clips.admission import Budget
from roboflow_workflows.core_steps.common.video_clips.artifact import (
    ClipArtifact,
    allocate_clip_path,
)
from roboflow_workflows.core_steps.common.video_clips.capture import (
    ClipEncodeError,
    ClipShutdownError,
    PendingCapture,
)
from roboflow_workflows.core_steps.common.video_clips.report import ClipReport
from roboflow_workflows.core_steps.common.video_clips.uploader import (
    COMPLETION_EVENT,
    STATUS_CANCELLED,
    STATUS_CLIP_FAILED,
    STATUS_EXPIRED,
    STATUS_PUT_ACCEPTED,
    STATUS_SHUTDOWN_DEADLINE,
    STATUS_UPLOAD_FAILED,
    ClipUploader,
    redact_text,
    redact_url,
)

START = datetime(2026, 1, 1, tzinfo=timezone.utc)
SECRET = "X-Goog-Signature=deadbeefsecret&X-Goog-Credential=svc%40proj"
SIGNED_URL = f"https://storage.googleapis.com/bucket/videos/abc.mp4?{SECRET}"
REQUIRED_HEADERS = {"Content-Type": "video/mp4", "x-goog-content-length-range": "0,1"}
CLIP_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"\x01" * 100


class _Response:
    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text


class _FakePut:
    def __init__(self, outcomes: List[object], delay: float = 0.0):
        self.outcomes = list(outcomes)
        self.calls: List[dict] = []
        self.delay = delay
        self.active = 0
        self.peak = 0
        self._lock = threading.Lock()

    def __call__(self, url, *, data, headers, timeout):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.calls.append({"url": url, "body": data.read(), "headers": headers})
            outcome = (
                self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
            )
        try:
            if self.delay:
                time.sleep(self.delay)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        finally:
            with self._lock:
                self.active -= 1


class _FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps: List[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _report() -> ClipReport:
    return ClipReport(requested_window=(START, START + timedelta(seconds=5)))


def _capture(artifact: Optional[ClipArtifact] = None) -> PendingCapture:
    capture = PendingCapture(
        capture_id="capture-1",
        video_identifier="camera-1",
        backend="x264",
        requested_window=(START, START + timedelta(seconds=5)),
        available_from=START,
        estimated_size_bytes=len(CLIP_BYTES),
    )
    if artifact is not None:
        capture.future.set_result(artifact)

    return capture


def _artifact(released: Optional[list] = None) -> ClipArtifact:
    return ClipArtifact(
        CLIP_BYTES,
        size_bytes=len(CLIP_BYTES),
        duration_seconds=5.0,
        report=_report(),
        on_release=(lambda: released.append(True)) if released is not None else None,
    )


def _uploader(put: _FakePut, clock: _FakeClock, **kwargs) -> ClipUploader:
    settings = {
        "budget": Budget(),
        "http_put": put,
        "sleep": clock.sleep,
        "clock": clock,
    }
    settings.update(kwargs)

    return ClipUploader(**settings)


def _submit(uploader: ClipUploader, capture: PendingCapture, **kwargs):
    settings = {
        "video_id": "video-1",
        "signed_url": SIGNED_URL,
        "required_headers": REQUIRED_HEADERS,
        "event_id": "event-1",
        "label": "Evidence",
        "upload_timeout_seconds": 3300.0,
    }
    settings.update(kwargs)

    return uploader.submit(capture, **settings)


def test_successful_put_sends_exactly_the_required_headers_and_releases(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # given
    put = _FakePut([_Response(200)])
    clock = _FakeClock()
    budget = Budget()
    released: list = []
    capture = _capture(_artifact(released))
    ticket = budget.admit(capture).ticket
    uploader = _uploader(put, clock, budget=budget)
    completions = []

    # when
    with caplog.at_level(logging.INFO):
        completion = _submit(
            uploader,
            capture,
            admission_ticket=ticket,
            on_complete=completions.append,
        ).result(timeout=5)

    # then
    assert completion.terminal_status == STATUS_PUT_ACCEPTED
    assert put.calls[0]["headers"] == REQUIRED_HEADERS
    assert put.calls[0]["body"] == CLIP_BYTES
    assert put.calls[0]["url"] == SIGNED_URL
    assert ticket.released is True
    assert budget.usage().pending_captures == 0
    assert released == [True]
    assert completions == [completion]
    record = completion.to_record()
    assert record["event_id"] == "event-1"
    assert record["video_id"] == "video-1"
    assert record["ingestion"] is None
    assert record["timings"]["uploaded_at"] is not None
    assert record["report"]["requested_window"]["start"] == START.isoformat()
    logged = [r for r in caplog.records if hasattr(r, COMPLETION_EVENT)]
    assert len(logged) == 1
    assert getattr(logged[0], COMPLETION_EVENT)["terminal_status"] == "put_accepted"
    assert "deadbeefsecret" not in caplog.text


@pytest.mark.parametrize(
    "response, expected_status",
    [
        (_Response(412, "<Code>PreconditionFailed</Code>"), STATUS_PUT_ACCEPTED),
        (_Response(400, "<Code>ExpiredToken</Code>"), STATUS_EXPIRED),
        (_Response(400, "<Code>MalformedSecurityHeader</Code>"), STATUS_UPLOAD_FAILED),
        (_Response(403, "<Code>SignatureDoesNotMatch</Code>"), STATUS_UPLOAD_FAILED),
    ],
)
def test_terminal_http_statuses(response: _Response, expected_status: str) -> None:
    # given
    put = _FakePut([response])
    uploader = _uploader(put, _FakeClock())

    # when
    completion = _submit(uploader, _capture(_artifact())).result(timeout=5)

    # then
    assert completion.terminal_status == expected_status
    assert len(put.calls) == 1


def test_transient_errors_back_off_then_succeed_with_redacted_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # given
    put = _FakePut(
        [
            _Response(503, f"retry {SIGNED_URL}"),
            requests.ConnectionError(f"connection reset for url: {SIGNED_URL}"),
            _Response(429),
            _Response(200),
        ]
    )
    clock = _FakeClock()
    uploader = _uploader(put, clock)

    # when
    with caplog.at_level(logging.INFO):
        completion = _submit(uploader, _capture(_artifact())).result(timeout=5)

    # then
    assert completion.terminal_status == STATUS_PUT_ACCEPTED
    assert clock.sleeps == [1.0, 2.0, 4.0]
    assert len(put.calls) == 4
    assert "deadbeefsecret" not in caplog.text
    assert "X-Goog-Credential" not in caplog.text
    assert "storage.googleapis.com/bucket/videos/abc.mp4?REDACTED" in caplog.text


def test_transient_errors_until_deadline_end_as_shutdown_deadline(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # given
    put = _FakePut([_Response(500, f"oops {SIGNED_URL}")])
    clock = _FakeClock()
    uploader = _uploader(put, clock)

    # when
    with caplog.at_level(logging.INFO):
        completion = _submit(
            uploader, _capture(_artifact()), upload_timeout_seconds=10.0
        ).result(timeout=5)

    # then
    assert completion.terminal_status == STATUS_SHUTDOWN_DEADLINE
    assert clock.sleeps == [1.0, 2.0, 4.0]
    assert "deadbeefsecret" not in (completion.error or "")
    assert "deadbeefsecret" not in caplog.text


def test_encode_failure_is_clip_failed_with_partial_report() -> None:
    # given
    capture = _capture()
    capture.future.set_exception(ClipEncodeError("no frames", report=_report()))
    put = _FakePut([_Response(200)])
    uploader = _uploader(put, _FakeClock())

    # when
    completion = _submit(uploader, capture).result(timeout=5)

    # then
    assert completion.terminal_status == STATUS_CLIP_FAILED
    assert completion.report is not None
    assert completion.error == "no frames"
    assert put.calls == []


def test_ring_shutdown_error_is_shutdown_deadline() -> None:
    # given
    capture = _capture()
    capture.future.set_exception(ClipShutdownError("closed"))
    uploader = _uploader(_FakePut([_Response(200)]), _FakeClock())

    # when
    completion = _submit(uploader, capture).result(timeout=5)

    # then
    assert completion.terminal_status == STATUS_SHUTDOWN_DEADLINE


def test_cancelled_capture_is_cancelled() -> None:
    # given
    capture = _capture()
    capture.future.cancel()
    uploader = _uploader(_FakePut([_Response(200)]), _FakeClock())

    # when
    completion = _submit(uploader, capture).result(timeout=5)

    # then
    assert completion.terminal_status == STATUS_CANCELLED


def test_clip_resolved_later_is_uploaded_when_ready() -> None:
    # given
    put = _FakePut([_Response(200)])
    capture = _capture()
    uploader = _uploader(put, _FakeClock())
    future = _submit(uploader, capture)
    time.sleep(0.05)
    assert not future.done()

    # when
    capture.future.set_result(_artifact())

    # then
    assert future.result(timeout=5).terminal_status == STATUS_PUT_ACCEPTED
    assert future.result().timings["ready_at"] is not None


def test_clip_not_ready_before_deadline_is_shutdown_deadline() -> None:
    # given
    clock = _FakeClock()
    capture = _capture()
    uploader = _uploader(_FakePut([_Response(200)]), clock)
    future = _submit(uploader, capture, upload_timeout_seconds=60.0)

    # when
    clock.now += 61.0

    # then
    completion = future.result(timeout=5)
    assert completion.terminal_status == STATUS_SHUTDOWN_DEADLINE
    assert capture.future.cancelled()


def test_shutdown_finishes_waiting_jobs_and_refuses_new_ones() -> None:
    # given
    uploader = _uploader(_FakePut([_Response(200)]), _FakeClock())
    waiting = _submit(uploader, _capture())

    # when
    started = time.monotonic()
    uploader.shutdown(0.2)
    late = _submit(uploader, _capture(_artifact()))

    # then
    assert time.monotonic() - started < 2.0
    assert waiting.result(timeout=1).terminal_status == STATUS_SHUTDOWN_DEADLINE
    assert late.result(timeout=1).terminal_status == STATUS_SHUTDOWN_DEADLINE
    assert uploader.pending_jobs() == 0


class _BlockingPut:
    """A PUT that runs until the test lets it return."""

    def __init__(self, outcome: object):
        self.outcome = outcome
        self.started = threading.Event()
        self.proceed = threading.Event()

    def __call__(self, url, *, data, headers, timeout):
        self.started.set()
        self.proceed.wait(timeout=10)
        data.read()
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


@pytest.mark.parametrize(
    "outcome, expected_status",
    [
        (_Response(200), STATUS_PUT_ACCEPTED),
        (requests.ConnectionError("reset"), STATUS_SHUTDOWN_DEADLINE),
    ],
)
def test_shutdown_lets_an_in_flight_put_record_its_real_outcome(
    outcome: object, expected_status: str
) -> None:
    # given: a PUT is running when the shutdown deadline passes
    put = _BlockingPut(outcome)
    uploader = _uploader(put, _FakeClock())
    released: list = []
    completion = _submit(uploader, _capture(_artifact(released)))
    assert put.started.wait(timeout=5)

    # when
    uploader.shutdown(0.1)

    # then: the clip is neither released nor reported while the PUT runs
    assert not completion.done()
    assert released == []
    put.proceed.set()
    result = completion.result(timeout=5)
    assert result.terminal_status == expected_status
    assert released == [True]
    assert uploader.pending_jobs() == 0


def test_oversized_clip_is_upload_failed() -> None:
    # given
    put = _FakePut([_Response(200)])
    uploader = _uploader(put, _FakeClock(), budget=Budget(max_clip_bytes=10))

    # when
    completion = _submit(uploader, _capture(_artifact())).result(timeout=5)

    # then
    assert completion.terminal_status == STATUS_UPLOAD_FAILED
    assert "clip_too_large" in completion.error
    assert put.calls == []


def test_file_backed_artifact_is_streamed(tmp_path: Path) -> None:
    # given
    path = allocate_clip_path(tmp_path)
    path.write_bytes(CLIP_BYTES)
    artifact = ClipArtifact.from_muxed_file(
        path, duration_seconds=1.0, report=_report(), inmemory_max_bytes=1
    )
    put = _FakePut([_Response(200)])
    uploader = _uploader(put, _FakeClock())

    # when
    completion = _submit(uploader, _capture(artifact)).result(timeout=5)

    # then
    assert completion.terminal_status == STATUS_PUT_ACCEPTED
    assert put.calls[0]["body"] == CLIP_BYTES
    assert not path.exists()


def test_confirm_ingestion_polls_the_probe_until_confirmed() -> None:
    # given
    states = ["pending", "pending", "confirmed"]
    probed = []

    def _probe(video_id: str) -> str:
        probed.append(video_id)
        return states.pop(0)

    clock = _FakeClock()
    uploader = _uploader(_FakePut([_Response(200)]), clock)

    # when
    completion = _submit(
        uploader, _capture(_artifact()), confirm_ingestion=True, ingestion_probe=_probe
    ).result(timeout=5)

    # then
    assert completion.terminal_status == STATUS_PUT_ACCEPTED
    assert completion.ingestion == "confirmed"
    assert probed == ["video-1"] * 3


def test_confirm_ingestion_is_bounded_and_reports_pending() -> None:
    # given
    def _probe(video_id: str) -> str:
        raise requests.ConnectionError(f"boom {SIGNED_URL}")

    clock = _FakeClock()
    uploader = _uploader(_FakePut([_Response(200)]), clock)

    # when
    completion = _submit(
        uploader, _capture(_artifact()), confirm_ingestion=True, ingestion_probe=_probe
    ).result(timeout=5)

    # then
    assert completion.ingestion == "pending"
    assert sum(clock.sleeps) <= 300.0
    assert sum(clock.sleeps) >= 290.0


def test_upload_concurrency_is_bounded_by_the_budget() -> None:
    # given
    put = _FakePut([_Response(200)], delay=0.05)
    uploader = ClipUploader(
        budget=Budget(max_upload_concurrency=2), http_put=put, max_workers=4
    )

    # when
    futures = [_submit(uploader, _capture(_artifact())) for _ in range(6)]
    results = [future.result(timeout=10) for future in futures]

    # then
    assert all(r.terminal_status == STATUS_PUT_ACCEPTED for r in results)
    assert put.peak == 2


def test_redaction_helpers() -> None:
    assert (
        redact_url(SIGNED_URL)
        == "https://storage.googleapis.com/bucket/videos/abc.mp4?REDACTED"
    )
    assert redact_url("https://example.com/a") == "https://example.com/a"
    assert "deadbeefsecret" not in redact_text(f"error at {SIGNED_URL} (x)")


def test_redaction_hides_the_signature_of_a_gateway_wrapped_url() -> None:
    # given: the sink wraps the signed URL for the secure gateway
    wrapped = "https://gateway.example/proxy?url=" + quote(SIGNED_URL, safe="")

    # then
    assert redact_url(wrapped) == "https://gateway.example/proxy?REDACTED"
    assert "deadbeefsecret" not in redact_text(f"PUT {wrapped} failed")
