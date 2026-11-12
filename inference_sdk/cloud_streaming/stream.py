"""One Workflow run on Cloud Streaming: its leases, results, output video and stop."""

import threading
from typing import Any, Callable, Dict, Iterator, Optional, Set

import numpy as np

from inference_sdk.cloud_streaming.api import CloudStreamingAPI, CloudStreamingError
from inference_sdk.cloud_streaming.results import ResultGap, ResultsFeed, StreamResult
from inference_sdk.cloud_streaming.sources import Publisher, require_av
from inference_sdk.cloud_streaming.tunnel import verified_rtsps
from inference_sdk.utils.logging import get_logger

logger = get_logger("cloud_streaming")

# Owner and watch leases last two minutes; the platform asks for a renewal every 30 seconds.
RENEW_INTERVAL_S = 30.0
VIDEO_RETRY_S = 2.0
DRAIN_S = 3.0
TERMINAL_STATES = {"completed", "cancelled", "error"}


class CloudStream:
    """A Workflow running on a Cloud Streaming source.

    Iterate `results()` for per-frame Workflow outputs and `video()` for the annotated output.
    `stop()` (or leaving a `with` block) ends what this stream started: it cancels the job and
    ends the temporary source it publishes to. A stream from `attach_stream()` only disconnects.
    """

    def __init__(
        self,
        api: CloudStreamingAPI,
        job: Dict[str, Any],
        owns_job: bool,
        session_id: Optional[str] = None,
        publisher: Optional[Publisher] = None,
    ):
        self._api = api
        self._job_id = job["id"]
        self.source_id: str = job["sourceId"]
        self._preview = job.get("purpose") == "preview"
        # A preview on a temporary source lives on its owner lease; any other preview on its
        # watch lease.
        self._on_temporary_source = bool(job.get("temporarySessionId"))
        self._owns_job = owns_job
        self._session_id = session_id
        self._publisher = publisher
        self._stopped = threading.Event()
        self._stop_lock = threading.Lock()
        self._feeds: Set[ResultsFeed] = set()
        self._viewers: Set[object] = set()
        self._watched_output: Optional[str] = None
        self._failure: Optional[str] = None
        self._keeper = threading.Thread(
            target=self._keep_alive, name="cloud-streaming-leases", daemon=True
        )
        self._keeper.start()
        if publisher:
            threading.Thread(
                target=self._end_with_publisher,
                name="cloud-streaming-publisher-watch",
                daemon=True,
            ).start()

    @property
    def job_id(self) -> str:
        """The job this stream follows; a successor job (after a resize or retry) replaces it."""
        return self._job_id

    def job(self) -> Dict[str, Any]:
        """The job's current public record."""
        return self._api.get_job(self._job_id)

    def results(
        self, on_gap: Optional[Callable[[ResultGap], None]] = None
    ) -> Iterator[StreamResult]:
        """Workflow results, one per processed frame, until the stream ends.

        Missed results (a slow reader, a reconnect) are reported to `on_gap`, or logged.
        Raises `CloudStreamingError` at the end when the job or its publisher failed.
        """
        feed = ResultsFeed(self._api, self._stopped)
        self._feeds.add(feed)
        try:
            for item in feed.follow(self._job_id, self._switch_job):
                if isinstance(item, ResultGap):
                    (on_gap or _log_gap)(item)
                else:
                    yield item
        finally:
            self._feeds.discard(feed)
        self._raise_if_failed()

    def video(self, output: Optional[str] = None) -> Iterator[np.ndarray]:
        """Annotated BGR frames of the Workflow image `output` (the job's default when None).

        A job publishes one image output at a time, so this changes it for every viewer.
        Needs PyAV (`pip install 'inference-sdk[streaming]'`).
        """
        av = require_av()
        self._watched_output = output
        viewer = object()
        self._viewers.add(viewer)
        try:
            while not self._stopped.is_set():
                url = self._watch_url(output)
                if url is None:
                    break
                with verified_rtsps(url) as tunnel:
                    yield from self._decode(av, tunnel.url)
                self._stopped.wait(VIDEO_RETRY_S)
        finally:
            self._viewers.discard(viewer)
        self._raise_if_failed()

    def _decode(self, av: Any, url: str) -> Iterator[np.ndarray]:
        # An error means the output is not published yet (it starts with the watch lease) or the
        # connection dropped; either way the caller asks for a fresh grant. FFmpeg's messages
        # name the url, credentials included, so they are never passed on.
        try:
            with av.open(
                url, options={"rtsp_transport": "tcp", "timeout": "10000000"}
            ) as container:
                for frame in container.decode(video=0):
                    if self._stopped.is_set():
                        return
                    yield frame.to_ndarray(format="bgr24")
        except av.FFmpegError:
            return

    def stop(self) -> None:
        """End what this stream started and release its connections. Safe to call twice."""
        with self._stop_lock:
            if self._stopped.is_set():
                return
            self._stopped.set()
        for feed in list(self._feeds):
            feed.close()
        if self._owns_job:
            _best_effort(self._api.cancel_job, self._job_id)
        if self._session_id:
            _best_effort(self._api.stop_session, self._session_id)
        if self._publisher:
            self._publisher.stop()
        self._keeper.join()

    def __enter__(self) -> "CloudStream":
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    def _watch_url(self, output: Optional[str]) -> Optional[str]:
        while not self._stopped.is_set():
            try:
                grant = self._api.watch(self._job_id, output)
            except CloudStreamingError as error:
                if self._job_ended():
                    return None
                # The platform answers 400 until the job runs.
                if error.status != 400:
                    raise
                self._stopped.wait(VIDEO_RETRY_S)
                continue
            if not grant.get("rtspsUrl"):
                raise CloudStreamingError(
                    "this job's cell does not offer RTSPS output video"
                )
            return grant["rtspsUrl"]
        return None

    def _job_ended(self) -> bool:
        job = self._api.get_job(self._job_id)
        if job["state"] not in TERMINAL_STATES:
            return False
        if job.get("successorJobId"):
            self._switch_job(job["successorJobId"])
            return False
        return True

    def _switch_job(self, job_id: str) -> None:
        logger.info("Job %s continues as %s", self._job_id, job_id)
        self._job_id = job_id

    def _end_with_publisher(self) -> None:
        # A temporary source ends when its publisher leaves, and its jobs with it.
        self._publisher.done.wait()
        if self._stopped.is_set():
            return
        if self._publisher.error:
            self._failure = f"publishing stopped: {self._publisher.error}"
        else:
            # The video ended: let the results of its last frames arrive first.
            self._stopped.wait(DRAIN_S)
        self.stop()

    def _keep_alive(self) -> None:
        owner_lease = self._owns_job and self._on_temporary_source
        preview_watch = (
            self._owns_job and self._preview and not self._on_temporary_source
        )
        while not self._stopped.is_set():
            if (
                owner_lease
                and self._renew(self._api.hold_preview, self._job_id) is False
            ):
                if self._renew(self._job_ended):
                    return
            if self._viewers or preview_watch:
                self._renew(self._api.watch, self._job_id, self._watched_output)
            self._stopped.wait(RENEW_INTERVAL_S)

    def _renew(self, call: Callable[..., Any], *args: Any) -> Any:
        try:
            return call(*args)
        except CloudStreamingError as error:
            # The platform answers 400 until the job runs.
            if error.status != 400:
                logger.warning("Renewing the stream's leases failed: %s", error)
            return None

    def _raise_if_failed(self) -> None:
        if self._failure:
            raise CloudStreamingError(self._failure)
        if self._stopped.is_set():
            return
        job = self._api.get_job(self._job_id)
        if job["state"] == "error":
            error = job.get("lastError") or {}
            raise CloudStreamingError(
                f"the job failed: {error.get('message', 'unknown error')}"
            )
        if job["state"] == "cancelled" or job.get("cancelRequested"):
            raise CloudStreamingError(
                f"the job was cancelled: {job.get('stopReason') or 'by request'}"
            )


def _log_gap(gap: ResultGap) -> None:
    logger.warning(
        "Missed results %d to %d of job %s", gap.from_seq, gap.to_seq, gap.job_id
    )


def _best_effort(call: Callable[[str], Any], resource_id: str) -> None:
    try:
        call(resource_id)
    except CloudStreamingError as error:
        logger.warning(
            "%s(%s) failed while stopping: %s", call.__name__, resource_id, error
        )
