"""Video clip attachments for the Roboflow Vision Events sink (both variants).

The numpy (`v1`) and tensor-native (`v1_tensor`) sinks share this module so the
clip logic cannot drift between them. Per event, after the sink's disable and
cooldown checks:

1. `plan_attachments` normalises `video_clips` to at most
   `MAX_VIDEOS_PER_EVENT` anchors (extras become `rejected:too_many_videos`),
   calls each available anchor's `capture()` with the rolls of its recorder,
   and asks the process-wide admission budget whether the capture may proceed.
   This runs on the caller's thread and is cheap.
2. `ClipAttachments.prepare_uploads` asks the platform for one upload slot per
   admitted capture (`POST /vision-events/videos/prepare-upload`).
3. `ClipAttachments.videos_field` gives the event's `videos` list, in request
   order.
4. After the event POST, `ClipAttachments.hand_off` gives every capture to the
   sink's clip uploader, or `ClipAttachments.abandon` releases them when the
   event failed.

The uploader PUTs each clip when its post-roll completes and logs the
`vision_event_clip_completed` record; with `confirm_ingestion` it polls the
`ingestion_probe` built here. Signed upload URLs never reach a log line, an
error message or an output: their query strings are redacted.
"""

import dataclasses
import logging
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from roboflow_workflows.prototypes.platform_client import (
    RoboflowPlatformClient,
    ensure_roboflow_api_access,
)

logger = logging.getLogger(__name__)

# The platform keeps the first 4 videos of an event and drops the rest with a
# warning, so the sink rejects the extras visibly instead of orphaning uploads.
MAX_VIDEOS_PER_EVENT = 4
MAX_LABEL_LENGTH = 256
DEFAULT_VIDEO_LABEL = "Evidence"
VIDEO_CONTENT_TYPE = "video/mp4"
DEFAULT_CLIP_UPLOAD_TIMEOUT_SECONDS = 55 * 60
# Signed upload URLs expire one hour after prepare-upload.
MAX_CLIP_UPLOAD_TIMEOUT_SECONDS = 60 * 60
DEFAULT_CLOSE_DEADLINE_SECONDS = 30.0
PLATFORM_REQUEST_TIMEOUT_SECONDS = 30

AVAILABLE = "ok"
STATUS_PENDING = "pending"
STATUS_DISABLED = "disabled"
STATUS_THROTTLED = "throttled"
STATUS_PREPARE_FAILED = "prepare_failed"
STATUS_TOO_MANY_VIDEOS = "rejected:too_many_videos"
STATUS_EVENT_FAILED = "rejected:event_failed"
STATUS_UPLOADER_UNAVAILABLE = "rejected:uploader_unavailable"
STATUS_NO_ANCHOR = "unavailable:no_anchor"
STATUS_INVALID_ANCHOR = "unavailable:invalid_anchor"
STATUS_CAPTURE_FAILED = "unavailable:capture_failed"
STATUS_UNSUPPORTED_SINK_MODE = "unavailable:unsupported_sink_mode"
INGESTION_CONFIRMED = "confirmed"
INGESTION_PENDING = "pending"
INGESTION_FAILED = "failed"

_SIGNED_URL_QUERY_PATTERN = re.compile(r"(https?://[^\s\"'?]+)\?[^\s\"']*")
_UNSAFE_NAME_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]+")


def redact_signed_urls(text: str) -> str:
    """Strip the query string (the signature) from every URL in `text`.

    Args:
        text (str): Any text that may contain a signed URL.

    Returns:
        str: `text` with each `http(s)://host/path?query` reduced to
        `http(s)://host/path?<redacted>`.
    """
    redacted = _SIGNED_URL_QUERY_PATTERN.sub(r"\1?<redacted>", text)

    return redacted


@dataclasses.dataclass
class ClipAttachment:
    """One `video_clips` entry of an event and its progress.

    Attributes:
        label (str): Caption sent with the video.
        video_status (str): Immediate status reported in the block output.
        anchor (Any): The `ClipAnchor` the entry came from, if any.
        pending_capture (Any): The `PendingCapture` returned by
            `anchor.capture()` once admitted.
        admission_ticket (Any): Ticket of the admitted capture; the uploader
            releases it at the terminal status.
        video_id (str): Platform video id from prepare-upload.
        signed_url (Optional[str]): Upload URL from prepare-upload. Never logged.
        required_headers (Dict[str, str]): Headers the PUT must carry.
        requested_window (Optional[Dict[str, Optional[str]]]): `{start, end}`
            requested from the recorder, ISO 8601.
        available_from (Optional[str]): Earliest buffered frame time at
            capture, ISO 8601.
    """

    label: str
    video_status: str
    anchor: Any = None
    pending_capture: Any = None
    admission_ticket: Any = None
    video_id: str = ""
    signed_url: Optional[str] = None
    required_headers: Dict[str, str] = dataclasses.field(default_factory=dict)
    requested_window: Optional[Dict[str, Optional[str]]] = None
    available_from: Optional[str] = None

    def to_output(self) -> Dict[str, Any]:
        """Serialise the entry for the block's `videos` output.

        Returns:
            Dict[str, Any]: `video_id`, `label`, `video_status`,
            `requested_window` and `available_from`.
        """
        output = {
            "video_id": self.video_id,
            "label": self.label,
            "video_status": self.video_status,
            "requested_window": (
                dict(self.requested_window) if self.requested_window else None
            ),
            "available_from": self.available_from,
        }

        return output


class ClipAttachments:
    """The clips attached to one event, from capture to uploader hand-off.

    Args:
        attachments (List[ClipAttachment]): Entries in request order.
    """

    def __init__(self, attachments: List[ClipAttachment]):
        self.attachments = attachments
        self._prepared_at: Optional[float] = None
        self._ingestion_probe: Optional[Callable[[str], str]] = None
        self._uploader: Any = None
        self._clip_upload_timeout_seconds = DEFAULT_CLIP_UPLOAD_TIMEOUT_SECONDS
        self._confirm_ingestion = False
        self._handed_off = False

    def has_pending(self) -> bool:
        """Tell whether any entry still waits for upload preparation.

        Returns:
            bool: True when at least one entry is `pending`.
        """
        pending = any(
            attachment.video_status == STATUS_PENDING for attachment in self.attachments
        )

        return pending

    def bind_uploader(
        self,
        uploader_factory: Callable[[], Any],
        *,
        clip_upload_timeout_seconds: float,
        confirm_ingestion: bool,
    ) -> None:
        """Choose the uploader and upload settings used by `hand_off`.

        Called on the sink's thread before the event is sent. When the
        uploader cannot be created, pending entries become
        `rejected:uploader_unavailable` and their captures are released.

        Args:
            uploader_factory (Callable[[], Any]): Returns the sink's uploader.
            clip_upload_timeout_seconds (float): Upload deadline, measured from
                the prepare-upload call.
            confirm_ingestion (bool): Poll the platform after an accepted upload.
        """
        self._clip_upload_timeout_seconds = clip_upload_timeout_seconds
        self._confirm_ingestion = confirm_ingestion
        if not self.has_pending():
            return None

        try:
            self._uploader = uploader_factory()
        except Exception as error:
            logger.warning(
                "Vision Events clip uploader is unavailable: %s",
                f"{type(error).__name__}: {error}",
            )
            for attachment in self.attachments:
                if attachment.video_status == STATUS_PENDING:
                    attachment.video_status = STATUS_UPLOADER_UNAVAILABLE
                    _release(attachment)

    def outputs(self) -> List[Dict[str, Any]]:
        """Return the `videos` block output for every entry, in request order.

        Returns:
            List[Dict[str, Any]]: One dict per entry, see `ClipAttachment.to_output`.
        """
        outputs = [attachment.to_output() for attachment in self.attachments]

        return outputs

    def prepare_uploads(
        self,
        *,
        event_id: str,
        api_base_url: str,
        platform_client: RoboflowPlatformClient,
        request_headers: Dict[str, Any],
        raise_for_status: Callable[[requests.Response], None],
    ) -> None:
        """Ask the platform for one upload slot per pending capture.

        A failed prepare marks the entry `prepare_failed` and releases its
        capture; the event is still sent, without that video.

        Args:
            event_id (str): Event id, used in the uploaded file name.
            api_base_url (str): Roboflow API base URL.
            platform_client (RoboflowPlatformClient): Resolved platform client;
                it wraps the URLs for the secure gateway.
            request_headers (Dict[str, Any]): Headers of a JSON Vision Events
                API request, credential included.
            raise_for_status (Callable[[requests.Response], None]): The sink's
                credential-redacting `raise_for_status`.
        """
        self._prepared_at = time.monotonic()
        self._ingestion_probe = _IngestionProbe(
            api_base_url=api_base_url,
            platform_client=platform_client,
            request_headers=request_headers,
        )
        for attachment in self.attachments:
            if attachment.video_status != STATUS_PENDING:
                continue

            try:
                video_id, signed_url, required_headers = _prepare_video_upload(
                    name=_upload_file_name(event_id, label=attachment.label),
                    api_base_url=api_base_url,
                    platform_client=platform_client,
                    request_headers=request_headers,
                    raise_for_status=raise_for_status,
                )
            except Exception as error:
                logger.warning(
                    "Vision Events video prepare-upload failed for label %r: %s",
                    attachment.label,
                    redact_signed_urls(f"{type(error).__name__}: {error}"),
                )
                attachment.video_status = STATUS_PREPARE_FAILED
                _release(attachment)
                continue

            attachment.video_id = video_id
            attachment.signed_url = signed_url
            attachment.required_headers = required_headers

    def videos_field(self) -> List[Dict[str, str]]:
        """Return the event's `videos` list for the prepared entries, in order.

        Returns:
            List[Dict[str, str]]: `{"sourceId": <video id>, "label": <label>}`
            per prepared entry; empty when nothing was prepared.
        """
        videos = [
            {"sourceId": attachment.video_id, "label": attachment.label}
            for attachment in self.attachments
            if attachment.video_status == STATUS_PENDING and attachment.video_id
        ]

        return videos

    def hand_off(self, *, event_id: str) -> None:
        """Give every prepared capture to the uploader chosen by `bind_uploader`.

        The uploader owns the capture and the admission ticket from here on,
        PUTs the clip when its post-roll completes and logs the
        `vision_event_clip_completed` record.

        Args:
            event_id (str): Id of the posted event.
        """
        self._handed_off = True
        uploader = self._uploader
        confirm_ingestion = self._confirm_ingestion
        upload_timeout_seconds = self._clip_upload_timeout_seconds
        if self._prepared_at is not None:
            elapsed = time.monotonic() - self._prepared_at
            upload_timeout_seconds = max(0.0, upload_timeout_seconds - elapsed)
        for attachment in self.attachments:
            if attachment.video_status != STATUS_PENDING or not attachment.video_id:
                continue

            try:
                if uploader is None:
                    raise RuntimeError("no clip uploader is bound")

                uploader.submit(
                    attachment.pending_capture,
                    video_id=attachment.video_id,
                    signed_url=attachment.signed_url,
                    required_headers=dict(attachment.required_headers),
                    event_id=event_id,
                    label=attachment.label,
                    upload_timeout_seconds=upload_timeout_seconds,
                    admission_ticket=attachment.admission_ticket,
                    confirm_ingestion=confirm_ingestion,
                    ingestion_probe=(
                        self._ingestion_probe if confirm_ingestion else None
                    ),
                )
            except Exception as error:
                logger.warning(
                    "Vision Events clip uploader refused video %s of event %s: %s",
                    attachment.video_id,
                    event_id,
                    redact_signed_urls(f"{type(error).__name__}: {error}"),
                )
                attachment.video_status = STATUS_UPLOADER_UNAVAILABLE
                _release(attachment)
            # The uploader holds the URL from here on; the sink drops it.
            attachment.signed_url = None
            attachment.pending_capture = None
            attachment.admission_ticket = None

    def abandon(self, *, event_id: str, reason: str) -> None:
        """Release every pending capture after the event was not created.

        Prepared videos stay on the platform without an upload (the platform
        has no video delete); their ids are logged as orphaned.

        Args:
            event_id (str): Id the event would have had.
            reason (str): Why the event failed, for the log.
        """
        if self._handed_off:
            return None

        orphaned_video_ids = []
        for attachment in self.attachments:
            if attachment.video_status != STATUS_PENDING:
                continue

            attachment.video_status = STATUS_EVENT_FAILED
            attachment.signed_url = None
            _release(attachment)
            if attachment.video_id:
                orphaned_video_ids.append(attachment.video_id)
        if orphaned_video_ids:
            logger.warning(
                "Vision event %s was not created (%s); orphaned video ids: %s",
                event_id,
                redact_signed_urls(reason),
                ", ".join(orphaned_video_ids),
            )


class _IngestionProbe:
    """One platform lookup of a video's ingestion status (the uploader polls it)."""

    def __init__(
        self,
        *,
        api_base_url: str,
        platform_client: RoboflowPlatformClient,
        request_headers: Dict[str, Any],
    ):
        self._api_base_url = api_base_url
        self._platform_client = platform_client
        self._request_headers = dict(request_headers)

    def __call__(self, video_id: str) -> str:
        try:
            ensure_roboflow_api_access(self._platform_client)
            response = requests.get(
                self._platform_client.wrap_url(
                    f"{self._api_base_url}/vision-events/videos/{video_id}"
                ),
                headers=self._request_headers,
                timeout=PLATFORM_REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code >= 400:
                return INGESTION_PENDING

            status = (response.json() or {}).get("status")
        except Exception as error:
            logger.debug(
                "Vision Events video %s status lookup failed: %s",
                video_id,
                redact_signed_urls(f"{type(error).__name__}: {error}"),
            )
            return INGESTION_PENDING

        if status == "uploaded":
            return INGESTION_CONFIRMED

        if status == "failed":
            return INGESTION_FAILED

        return INGESTION_PENDING


class SinkClipUploads:
    """The clip uploader of one sink instance and its shutdown.

    The uploader is created on the first event with a clip. `prepare_close`
    records the host's close deadline; `close` shuts the uploader down within
    what is left of it (unfinished uploads end as `shutdown_deadline`). Both
    are idempotent and safe to call concurrently.

    Args:
        uploader_factory (Optional[Callable[[], Any]]): Creates the uploader;
            defaults to `create_clip_uploader`.
    """

    def __init__(self, uploader_factory: Optional[Callable[[], Any]] = None):
        self._uploader_factory = uploader_factory
        self._lock = threading.Lock()
        self._uploader: Any = None
        self._close_deadline: Optional[float] = None
        self._closed = False

    def get(self) -> Any:
        """Return the sink's uploader, creating it on first use.

        Returns:
            Any: The `ClipUploader`.

        Raises:
            RuntimeError: If the sink is closing.
        """
        with self._lock:
            if self._closed or self._close_deadline is not None:
                raise RuntimeError("the sink is closing")

            if self._uploader is None:
                self._uploader = (self._uploader_factory or create_clip_uploader)()
            uploader = self._uploader

        return uploader

    def prepare_close(self, deadline_seconds: Optional[float] = None) -> None:
        """Stop taking new clips and remember the host's close deadline.

        Uploads already handed over keep running: they start when their
        post-roll completes, never at close.

        Args:
            deadline_seconds (Optional[float]): Seconds the host allows for the
                whole close; None uses `DEFAULT_CLOSE_DEADLINE_SECONDS`.
        """
        with self._lock:
            if self._close_deadline is not None:
                return None

            if deadline_seconds is None:
                deadline_seconds = DEFAULT_CLOSE_DEADLINE_SECONDS
            self._close_deadline = time.monotonic() + max(0.0, deadline_seconds)

    def close(self) -> None:
        """Shut the uploader down within the remaining close deadline."""
        self.prepare_close()
        with self._lock:
            if self._closed:
                return None

            self._closed = True
            uploader = self._uploader
            remaining = max(0.0, self._close_deadline - time.monotonic())
        if uploader is None:
            return None

        try:
            uploader.shutdown(remaining)
        except Exception as error:
            logger.warning(
                "Vision Events clip uploader shutdown failed: %s",
                redact_signed_urls(f"{type(error).__name__}: {error}"),
            )


def plan_attachments(
    video_clips: Any,
    *,
    video_label: str,
    budget_factory: Optional[Callable[[], Any]] = None,
) -> Optional[ClipAttachments]:
    """Capture and admit the clips of one event.

    Runs on the caller's thread, after the sink's disable and cooldown checks.
    `anchor.capture()` only snapshots ring packets and registers the capture,
    so the post-roll keeps filling in while the event is being posted.

    Args:
        video_clips (Any): The block's `video_clips` input: None, one anchor
            or a list of anchors.
        video_label (str): Label used when an anchor's label is empty.
        budget_factory (Optional[Callable[[], Any]]): Returns the admission
            budget; defaults to the process-wide one.

    Returns:
        Optional[ClipAttachments]: None when `video_clips` is unset (the event
        request stays exactly as without clips), otherwise the entries.
    """
    if video_clips is None:
        return None

    attachments = [
        ClipAttachment(
            label=resolve_label(anchor, fallback=video_label),
            video_status=STATUS_PENDING,
            anchor=anchor,
        )
        for anchor in normalize_video_clips(video_clips)
    ]
    for attachment in attachments[MAX_VIDEOS_PER_EVENT:]:
        attachment.video_status = STATUS_TOO_MANY_VIDEOS
    budget = None
    for attachment in attachments[:MAX_VIDEOS_PER_EVENT]:
        unavailable_status = _anchor_unavailable_status(attachment.anchor)
        if unavailable_status is not None:
            attachment.video_status = unavailable_status
            continue

        if budget is None:
            budget = (budget_factory or default_clip_budget)()
        _capture_and_admit(attachment, budget=budget)
    clip_attachments = ClipAttachments(attachments)

    return clip_attachments


def skipped_attachments(
    video_clips: Any,
    *,
    video_label: str,
    video_status: str,
) -> List[Dict[str, Any]]:
    """Report every anchor with one status and capture nothing.

    Used when the sink is disabled, throttled or in a mode without video
    support.

    Args:
        video_clips (Any): The block's `video_clips` input.
        video_label (str): Label used when an anchor's label is empty.
        video_status (str): Status reported for every anchor.

    Returns:
        List[Dict[str, Any]]: The `videos` block output.
    """
    outputs = [
        ClipAttachment(
            label=resolve_label(anchor, fallback=video_label),
            video_status=video_status,
        ).to_output()
        for anchor in normalize_video_clips(video_clips)
    ]

    return outputs


def normalize_video_clips(video_clips: Any) -> List[Any]:
    """Return `video_clips` as a list of anchors (entries may be None).

    Args:
        video_clips (Any): None, one anchor, or a list or tuple of anchors.

    Returns:
        List[Any]: The anchors in request order.
    """
    if video_clips is None:
        return []

    if isinstance(video_clips, (list, tuple)):
        return list(video_clips)

    return [video_clips]


def resolve_label(anchor: Any, *, fallback: str) -> str:
    """Pick the caption of a clip: the recorder's label, else the sink's.

    Args:
        anchor (Any): The clip anchor (or None).
        fallback (str): The sink's `video_label`.

    Returns:
        str: A non-empty label of at most `MAX_LABEL_LENGTH` characters.
    """
    label = getattr(anchor, "label", None)
    if not isinstance(label, str) or not label.strip():
        label = fallback
    if not isinstance(label, str) or not label.strip():
        label = DEFAULT_VIDEO_LABEL
    resolved_label = label.strip()[:MAX_LABEL_LENGTH]

    return resolved_label


def default_clip_budget() -> Any:
    """Return the process-wide clip admission budget.

    Imported lazily: a sink without `video_clips` never loads the clip
    machinery.

    Returns:
        Any: `admission.default_budget()`.
    """
    from roboflow_workflows.core_steps.common.video_clips.admission import (
        default_budget,
    )

    budget = default_budget()

    return budget


def create_clip_uploader() -> Any:
    """Create the clip uploader of one sink instance.

    Returns:
        Any: A `ClipUploader` on the process-wide budget.
    """
    from roboflow_workflows.core_steps.common.video_clips.uploader import ClipUploader

    uploader = ClipUploader()

    return uploader


def _capture_and_admit(attachment: ClipAttachment, *, budget: Any) -> None:
    anchor = attachment.anchor
    try:
        pending_capture = anchor.capture(
            anchor.pre_roll_seconds, anchor.post_roll_seconds
        )
    except Exception as error:
        reason = getattr(error, "reason", None)
        if isinstance(reason, str) and reason.startswith("unavailable:"):
            attachment.video_status = reason
            return None

        logger.warning(
            "Video clip capture failed for recorder %r: %s",
            getattr(anchor, "recorder_step", None),
            f"{type(error).__name__}: {error}",
        )
        attachment.video_status = STATUS_CAPTURE_FAILED
        return None

    attachment.requested_window = _window_to_output(pending_capture)
    attachment.available_from = _timestamp_to_output(
        getattr(pending_capture, "available_from", None)
    )
    try:
        decision = budget.admit(pending_capture)
    except Exception as error:
        logger.warning(
            "Video clip admission failed: %s", f"{type(error).__name__}: {error}"
        )
        attachment.video_status = "rejected:admission_error"
        _release_quietly(pending_capture)
        return None

    if not getattr(decision, "admitted", False):
        # A rejecting budget has already released the capture.
        attachment.video_status = _rejected_status(getattr(decision, "reason", None))
        return None

    attachment.pending_capture = pending_capture
    attachment.admission_ticket = getattr(decision, "ticket", None)


def _anchor_unavailable_status(anchor: Any) -> Optional[str]:
    if anchor is None:
        return STATUS_NO_ANCHOR

    if not callable(getattr(anchor, "capture", None)):
        return STATUS_INVALID_ANCHOR

    availability = getattr(anchor, "availability", None)
    if availability == AVAILABLE:
        return None

    if isinstance(availability, str) and availability.startswith("unavailable:"):
        return availability

    return STATUS_INVALID_ANCHOR


def _rejected_status(reason: Optional[str]) -> str:
    reason = str(reason or "admission")
    if reason.startswith("rejected:"):
        return reason

    return f"rejected:{reason}"


def _release(attachment: ClipAttachment) -> None:
    ticket = attachment.admission_ticket
    pending_capture = attachment.pending_capture
    attachment.admission_ticket = None
    attachment.pending_capture = None
    if ticket is not None:
        _release_quietly(ticket)
    if pending_capture is not None:
        _release_quietly(pending_capture)


def _release_quietly(resource: Any) -> None:
    try:
        resource.release()
    except Exception as error:
        logger.warning(
            "Video clip resource release failed: %s",
            f"{type(error).__name__}: {error}",
        )


def _prepare_video_upload(
    *,
    name: str,
    api_base_url: str,
    platform_client: RoboflowPlatformClient,
    request_headers: Dict[str, Any],
    raise_for_status: Callable[[requests.Response], None],
) -> Tuple[str, str, Dict[str, str]]:
    ensure_roboflow_api_access(platform_client)
    response = requests.post(
        platform_client.wrap_url(f"{api_base_url}/vision-events/videos/prepare-upload"),
        headers=request_headers,
        json={"name": name, "contentType": VIDEO_CONTENT_TYPE},
        timeout=PLATFORM_REQUEST_TIMEOUT_SECONDS,
    )
    raise_for_status(response)
    body = response.json() or {}
    video_id = body.get("videoId")
    signed_url = body.get("signedUrl")
    if not video_id or not signed_url:
        raise ValueError("prepare-upload response lacks `videoId` or `signedUrl`")

    required_headers = {
        str(key): str(value)
        for key, value in (body.get("requiredHeaders") or {}).items()
    }
    prepared = (str(video_id), str(signed_url), required_headers)

    return prepared


def _upload_file_name(event_id: str, *, label: str) -> str:
    safe_label = _UNSAFE_NAME_CHARACTERS.sub("_", label).strip("_")[:64] or "clip"
    file_name = f"{event_id}-{safe_label}.mp4"

    return file_name


def _window_to_output(pending_capture: Any) -> Optional[Dict[str, Optional[str]]]:
    window_dict = getattr(pending_capture, "requested_window_dict", None)
    if callable(window_dict):
        window = window_dict()
    else:
        window = getattr(pending_capture, "requested_window", None)
    if window is None:
        return None

    if isinstance(window, dict):
        start, end = window.get("start"), window.get("end")
    elif isinstance(window, (list, tuple)) and len(window) == 2:
        start, end = window
    else:
        start, end = getattr(window, "start", None), getattr(window, "end", None)
    output = {"start": _timestamp_to_output(start), "end": _timestamp_to_output(end)}

    return output


def _timestamp_to_output(value: Any) -> Optional[str]:
    if value is None:
        return None

    if isinstance(value, datetime):
        if value.tzinfo is None:
            # Recorder timestamps are naive UTC.
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()

    return str(value)
