"""`ClipUploader`: PUT finished clips to their signed URLs and report the outcome.

For every admitted capture the sink hands over the capture, the platform's
``videoId``, the signed URL and its ``requiredHeaders``. The uploader waits for
the capture's future (resolved on the ring's own thread), then PUTs the clip
with exactly the required headers on one of its daemon worker threads, holding
one of the budget's process-wide upload slots.

Terminal statuses:

* ``put_accepted`` - 2xx, or 412 (the object already exists: an earlier attempt
  of this PUT succeeded);
* ``expired`` - 400 whose body names ``ExpiredToken``;
* ``upload_failed`` - any other 4xx except 408/429 (for example
  ``MalformedSecurityHeader``) or a clip larger than the budget allows;
* ``clip_failed`` - the capture produced no clip;
* ``cancelled`` - the capture was released before it finished;
* ``shutdown_deadline`` - the clip was not uploaded by
  ``issue_time + upload_timeout_seconds``, or `shutdown()` ran out of time.

Transient failures (connection errors, 5xx, 408, 429) are retried with
exponential backoff until the deadline. After ``put_accepted`` and with
``confirm_ingestion`` the sink-provided probe is polled for at most five
minutes. Every outcome is logged once as the structured
``vision_event_clip_completed`` record. Signed-URL query strings never appear in
logs or errors.
"""

import json
import logging
import re
import threading
import time
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass, field
from datetime import datetime, timezone
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit

import requests
from roboflow_workflows.core_steps.common.video_clips.admission import (
    AdmissionTicket,
    Budget,
    default_budget,
)
from roboflow_workflows.core_steps.common.video_clips.capture import (
    ClipEncodeError,
    ClipShutdownError,
    PendingCapture,
)
from roboflow_workflows.core_steps.common.video_clips.report import ClipReport

logger = logging.getLogger(__name__)

STATUS_PUT_ACCEPTED = "put_accepted"
STATUS_UPLOAD_FAILED = "upload_failed"
STATUS_EXPIRED = "expired"
STATUS_CLIP_FAILED = "clip_failed"
STATUS_CANCELLED = "cancelled"
STATUS_SHUTDOWN_DEADLINE = "shutdown_deadline"
TERMINAL_STATUSES = (
    STATUS_PUT_ACCEPTED,
    STATUS_UPLOAD_FAILED,
    STATUS_EXPIRED,
    STATUS_CLIP_FAILED,
    STATUS_CANCELLED,
    STATUS_SHUTDOWN_DEADLINE,
)

INGESTION_CONFIRMED = "confirmed"
INGESTION_PENDING = "pending"
INGESTION_FAILED = "failed"

COMPLETION_EVENT = "vision_event_clip_completed"
INGESTION_POLL_BOUND_SECONDS = 300.0
INGESTION_POLL_INTERVAL_SECONDS = 2.0
BACKOFF_INITIAL_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 30.0
PUT_TIMEOUT_SECONDS = (10.0, 300.0)
WORKER_POLL_SECONDS = 0.5
TRANSIENT_4XX = (408, 429)
ERROR_BODY_CHARACTERS = 300
REDACTED = "REDACTED"

_URL_QUERY_PATTERN = re.compile(r"(https?://[^\s?#'\"<>]+)\?[^\s'\"<>]*")

IngestionProbe = Callable[[str], str]
HttpPut = Callable[..., Any]


def redact_url(url: str) -> str:
    """Remove the query string and fragment from a URL.

    Args:
        url: A URL, typically a signed upload URL.

    Returns:
        The URL with its query replaced by ``REDACTED``.
    """
    parts = urlsplit(url)
    query = REDACTED if parts.query else ""
    redacted = urlunsplit((parts.scheme, parts.netloc, parts.path, query, ""))

    return redacted


def redact_text(text: str) -> str:
    """Redact the query string of every URL inside free text.

    Args:
        text: An error message or response body.

    Returns:
        The text with URL query strings replaced by ``REDACTED``.
    """
    redacted = _URL_QUERY_PATTERN.sub(rf"\1?{REDACTED}", text)

    return redacted


@dataclass
class ClipCompletion:
    """Final outcome of one clip upload.

    Attributes:
        event_id: Vision event the video belongs to.
        video_id: Platform video id.
        label: Video label.
        terminal_status: One of `TERMINAL_STATUSES`.
        report: The clip report, when a clip (or a partial report) exists.
        timings: ISO timestamps ``captured_at``, ``ready_at`` and
            ``uploaded_at`` (``None`` when not reached).
        ingestion: ``None`` unless ingestion was confirmed on request:
            ``"confirmed"``, ``"pending"`` or ``"failed"``.
        error: Redacted error description for failed statuses.
    """

    event_id: Optional[str]
    video_id: str
    label: str
    terminal_status: str
    report: Optional[ClipReport] = None
    timings: Dict[str, Optional[str]] = field(default_factory=dict)
    ingestion: Optional[str] = None
    error: Optional[str] = None

    def to_record(self) -> Dict[str, Any]:
        """Return the `vision_event_clip_completed` record.

        Returns:
            JSON-compatible dictionary.
        """
        record = {
            "event_id": self.event_id,
            "video_id": self.video_id,
            "label": self.label,
            "terminal_status": self.terminal_status,
            "report": self.report.to_dict() if self.report is not None else None,
            "timings": dict(self.timings),
            "ingestion": self.ingestion,
            "error": self.error,
        }

        return record


class _UploadJob:
    """State of one submitted clip (job lock guards completion)."""

    def __init__(
        self,
        *,
        capture: PendingCapture,
        video_id: str,
        signed_url: str,
        required_headers: Dict[str, str],
        event_id: Optional[str],
        label: str,
        deadline: float,
        admission_ticket: Optional[AdmissionTicket],
        confirm_ingestion: bool,
        ingestion_probe: Optional[IngestionProbe],
        on_complete: Optional[Callable[[ClipCompletion], None]],
    ):
        self.capture = capture
        self.video_id = video_id
        self.signed_url = signed_url
        self.required_headers = required_headers
        self.event_id = event_id
        self.label = label
        self.deadline = deadline
        self.admission_ticket = admission_ticket
        self.confirm_ingestion = confirm_ingestion
        self.ingestion_probe = ingestion_probe
        self.on_complete = on_complete
        self.future: "Future[ClipCompletion]" = Future()
        self.lock = threading.Lock()
        self.finished = False
        self.ready_at: Optional[str] = None
        self.uploaded_at: Optional[str] = None
        self.report: Optional[ClipReport] = None


class ClipUploader:
    """Upload clips to signed URLs on a small pool of daemon threads.

    One uploader per sink instance is typical; the concurrency cap is enforced
    process-wide through the budget's upload slots.

    Args:
        budget: Budget providing upload slots and caps; `default_budget()`
            when ``None``.
        max_workers: Worker threads; the budget's upload concurrency when
            ``None``.
        http_put: ``requests.put``-compatible callable.
        sleep: Backoff sleep ``f(seconds)``; interruptible by `shutdown()` when
            ``None``.
        clock: Monotonic clock in seconds.
        ingestion_poll_bound_seconds: Longest ingestion confirmation poll.
        ingestion_poll_interval_seconds: Delay between ingestion probes.
        backoff_initial_seconds: First retry delay.
        backoff_max_seconds: Largest retry delay.
    """

    def __init__(
        self,
        *,
        budget: Optional[Budget] = None,
        max_workers: Optional[int] = None,
        http_put: Optional[HttpPut] = None,
        sleep: Optional[Callable[[float], Any]] = None,
        clock: Callable[[], float] = time.monotonic,
        ingestion_poll_bound_seconds: float = INGESTION_POLL_BOUND_SECONDS,
        ingestion_poll_interval_seconds: float = INGESTION_POLL_INTERVAL_SECONDS,
        backoff_initial_seconds: float = BACKOFF_INITIAL_SECONDS,
        backoff_max_seconds: float = BACKOFF_MAX_SECONDS,
    ):
        self._budget = budget or default_budget()
        self._max_workers = max_workers or self._budget.max_upload_concurrency
        self._http_put = http_put or requests.put
        self._stop_event = threading.Event()
        self._sleep = sleep or self._stop_event.wait
        self._clock = clock
        self._ingestion_poll_bound_seconds = ingestion_poll_bound_seconds
        self._ingestion_poll_interval_seconds = ingestion_poll_interval_seconds
        self._backoff_initial_seconds = backoff_initial_seconds
        self._backoff_max_seconds = backoff_max_seconds
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._work: "Queue[_UploadJob]" = Queue()
        self._jobs: List[_UploadJob] = []
        self._workers: List[threading.Thread] = []
        self._shutting_down = False

    def submit(
        self,
        capture: PendingCapture,
        *,
        video_id: str,
        signed_url: str,
        required_headers: Mapping[str, str],
        event_id: Optional[str],
        label: str,
        upload_timeout_seconds: float,
        admission_ticket: Optional[AdmissionTicket] = None,
        confirm_ingestion: bool = False,
        ingestion_probe: Optional[IngestionProbe] = None,
        on_complete: Optional[Callable[[ClipCompletion], None]] = None,
    ) -> "Future[ClipCompletion]":
        """Upload a capture's clip once it is ready.

        The uploader owns ``capture`` and ``admission_ticket`` from here on and
        releases both at the terminal status.

        Args:
            capture: The admitted pending capture.
            video_id: Platform video id from prepare-upload.
            signed_url: Signed PUT URL from prepare-upload.
            required_headers: Headers the PUT must carry, sent exactly.
            event_id: Vision event id, for the completion record.
            label: Video label, for the completion record.
            upload_timeout_seconds: Seconds from now (the URL's issue time) by
                which the PUT must have succeeded.
            admission_ticket: The capture's budget ticket.
            confirm_ingestion: Poll ``ingestion_probe`` after a successful PUT.
            ingestion_probe: ``f(video_id) -> "confirmed" | "pending" |
                "failed"``, provided by the sink (it owns the HTTP client).
            on_complete: Called with the `ClipCompletion` after the record is
                logged.

        Returns:
            Future resolved with the `ClipCompletion`.
        """
        job = _UploadJob(
            capture=capture,
            video_id=video_id,
            signed_url=signed_url,
            required_headers=dict(required_headers),
            event_id=event_id,
            label=label,
            deadline=self._clock() + upload_timeout_seconds,
            admission_ticket=admission_ticket,
            confirm_ingestion=confirm_ingestion,
            ingestion_probe=ingestion_probe,
            on_complete=on_complete,
        )
        with self._lock:
            shutting_down = self._shutting_down
            if not shutting_down:
                self._jobs.append(job)
                self._ensure_workers()
        if shutting_down:
            self._finish(
                job, status=STATUS_SHUTDOWN_DEADLINE, error="uploader shut down"
            )
            return job.future

        capture.future.add_done_callback(lambda _: self._on_clip_ready(job))

        return job.future

    def shutdown(self, deadline_seconds: Optional[float] = None) -> None:
        """Stop accepting clips and wait for in-flight ones up to a deadline.

        Clips not uploaded when the deadline passes end as
        ``shutdown_deadline``. Idempotent.

        Args:
            deadline_seconds: Seconds to wait; ``None`` waits until every job
                reached its own upload deadline.
        """
        with self._lock:
            self._shutting_down = True
            end = (
                None
                if deadline_seconds is None
                else time.monotonic() + deadline_seconds
            )
            while self._jobs:
                remaining = None if end is None else end - time.monotonic()
                if remaining is not None and remaining <= 0:
                    break
                self._idle.wait(
                    timeout=(
                        WORKER_POLL_SECONDS
                        if remaining is None
                        else min(remaining, WORKER_POLL_SECONDS)
                    )
                )
            leftover = list(self._jobs)
        self._stop_event.set()
        for job in leftover:
            self._finish(
                job,
                status=STATUS_SHUTDOWN_DEADLINE,
                error="upload not finished before the shutdown deadline",
            )

    def pending_jobs(self) -> int:
        """Number of clips not yet at a terminal status.

        Returns:
            The count.
        """
        with self._lock:
            count = len(self._jobs)

        return count

    def _ensure_workers(self) -> None:
        # Uploader lock held.
        self._workers = [worker for worker in self._workers if worker.is_alive()]
        while len(self._workers) < self._max_workers:
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"video-clip-uploader-{len(self._workers)}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

    def _on_clip_ready(self, job: _UploadJob) -> None:
        # Runs on the ring's finalizer thread (or the submitting thread when the
        # clip was already done): never block here.
        job.ready_at = _now_iso()
        self._work.put(job)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._work.get(timeout=WORKER_POLL_SECONDS)
            except Empty:
                self._expire_waiting_jobs()
                continue
            try:
                self._process(job)
            except Exception as error:
                self._finish(
                    job,
                    status=STATUS_UPLOAD_FAILED,
                    error=redact_text(f"unexpected uploader error: {error!r}"),
                )
            self._expire_waiting_jobs()

    def _expire_waiting_jobs(self) -> None:
        now = self._clock()
        with self._lock:
            expired = [
                job
                for job in self._jobs
                if now >= job.deadline and not job.capture.future.done()
            ]
        for job in expired:
            self._finish(
                job,
                status=STATUS_SHUTDOWN_DEADLINE,
                error="clip not ready before the upload deadline",
            )

    def _process(self, job: _UploadJob) -> None:
        if job.finished:
            return None

        capture_future = job.capture.future
        if capture_future.cancelled():
            self._finish(job, status=STATUS_CANCELLED, error="capture released")
            return None

        error = capture_future.exception()
        if error is not None:
            job.report = getattr(error, "report", None)
            status = (
                STATUS_SHUTDOWN_DEADLINE
                if isinstance(error, ClipShutdownError)
                else STATUS_CLIP_FAILED
            )
            message = str(error) if isinstance(error, ClipEncodeError) else repr(error)
            self._finish(job, status=status, error=redact_text(message))
            return None

        artifact = capture_future.result()
        job.report = artifact.report
        if artifact.size_bytes > self._budget.max_clip_bytes:
            self._finish(
                job,
                status=STATUS_UPLOAD_FAILED,
                error=(
                    f"clip_too_large: {artifact.size_bytes} bytes exceeds "
                    f"{self._budget.max_clip_bytes}"
                ),
            )
            return None

        if job.admission_ticket is not None:
            job.admission_ticket.set_bytes(artifact.size_bytes)
        status, message = self._put_with_retries(job, artifact)
        if status == STATUS_PUT_ACCEPTED:
            job.uploaded_at = _now_iso()
        self._finish(job, status=status, error=message)

    def _put_with_retries(
        self, job: _UploadJob, artifact: Any
    ) -> "tuple[str, Optional[str]]":
        attempt = 0
        redacted_url = redact_url(job.signed_url)
        while True:
            if self._stop_event.is_set():
                return STATUS_SHUTDOWN_DEADLINE, "uploader shut down during the upload"
            if self._clock() >= job.deadline:
                return STATUS_SHUTDOWN_DEADLINE, "upload deadline passed"

            with self._budget.upload_slot():
                status, message, transient = self._put_once(job, artifact)
            if not transient:
                return status, message

            delay = min(
                self._backoff_max_seconds,
                self._backoff_initial_seconds * (2**attempt),
            )
            attempt += 1
            logger.warning(
                "Video clip upload of %s to %s failed transiently (%s); "
                "retrying in %.1f s.",
                job.video_id,
                redacted_url,
                message,
                delay,
            )
            if self._clock() + delay >= job.deadline:
                return STATUS_SHUTDOWN_DEADLINE, f"upload deadline passed: {message}"
            self._sleep(delay)

    def _put_once(
        self, job: _UploadJob, artifact: Any
    ) -> "tuple[str, Optional[str], bool]":
        try:
            with artifact.open_stream() as body:
                response = self._http_put(
                    job.signed_url,
                    data=body,
                    headers=dict(job.required_headers),
                    timeout=PUT_TIMEOUT_SECONDS,
                )
        except (requests.RequestException, OSError) as error:
            return STATUS_UPLOAD_FAILED, redact_text(repr(error)), True

        status_code = int(response.status_code)
        if 200 <= status_code < 300 or status_code == 412:
            return STATUS_PUT_ACCEPTED, None, False

        body_text = redact_text(_response_text(response)[:ERROR_BODY_CHARACTERS])
        message = f"HTTP {status_code}: {body_text}"
        if status_code >= 500 or status_code in TRANSIENT_4XX:
            return STATUS_UPLOAD_FAILED, message, True
        if "ExpiredToken" in body_text:
            return STATUS_EXPIRED, message, False

        return STATUS_UPLOAD_FAILED, message, False

    def _finish(self, job: _UploadJob, *, status: str, error: Optional[str]) -> None:
        with job.lock:
            if job.finished:
                return None
            job.finished = True

        if job.admission_ticket is not None:
            job.admission_ticket.release()
        job.capture.release()
        completion = ClipCompletion(
            event_id=job.event_id,
            video_id=job.video_id,
            label=job.label,
            terminal_status=status,
            report=job.report,
            timings={
                "captured_at": job.capture.captured_at.isoformat(),
                "ready_at": job.ready_at,
                "uploaded_at": job.uploaded_at,
            },
            error=error,
        )
        if status == STATUS_PUT_ACCEPTED and job.confirm_ingestion:
            threading.Thread(
                target=self._confirm_and_complete,
                args=(job, completion),
                name="video-clip-ingestion-probe",
                daemon=True,
            ).start()
            return None

        self._complete(job, completion)

    def _confirm_and_complete(
        self, job: _UploadJob, completion: ClipCompletion
    ) -> None:
        completion.ingestion = self._poll_ingestion(job)
        self._complete(job, completion)

    def _poll_ingestion(self, job: _UploadJob) -> str:
        if job.ingestion_probe is None:
            return INGESTION_PENDING

        end = self._clock() + self._ingestion_poll_bound_seconds
        while True:
            try:
                state = job.ingestion_probe(job.video_id)
            except Exception as error:
                logger.debug(
                    "Ingestion probe for video %s failed: %s",
                    job.video_id,
                    redact_text(repr(error)),
                )
                state = INGESTION_PENDING
            if state in (INGESTION_CONFIRMED, INGESTION_FAILED):
                return state
            if (
                self._stop_event.is_set()
                or self._clock() + self._ingestion_poll_interval_seconds > end
            ):
                return INGESTION_PENDING
            self._sleep(self._ingestion_poll_interval_seconds)

    def _complete(self, job: _UploadJob, completion: ClipCompletion) -> None:
        record = completion.to_record()
        logger.info(
            "%s %s",
            COMPLETION_EVENT,
            json.dumps(record, default=str),
            extra={COMPLETION_EVENT: record},
        )
        try:
            if job.on_complete is not None:
                job.on_complete(completion)
        except Exception as error:
            logger.warning(
                "Video clip completion callback failed: %s", redact_text(repr(error))
            )
        finally:
            with self._lock:
                if job in self._jobs:
                    self._jobs.remove(job)
                self._idle.notify_all()
            try:
                job.future.set_result(completion)
            except InvalidStateError:
                pass


def _response_text(response: Any) -> str:
    try:
        text = response.text
    except Exception:
        return ""

    return text if isinstance(text, str) else str(text)


def _now_iso() -> str:
    timestamp = datetime.now(timezone.utc).isoformat()

    return timestamp
