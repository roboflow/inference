"""`ClipReport`: the bounds, timing and loss accounting of one video clip.

The sink serialises `ClipReport.to_dict()` into the `vision_event_clip_completed`
record. Field semantics follow the Phase 1 interface, section 4.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class LossCounts:
    """Frames missing from a clip's window, by cause.

    Attributes:
        decimated: In-window frames dropped by the output grid (`clip_fps`).
        ring_drops: In-window frames evicted by the ring byte cap.
        queue_drops: In-window frames dropped because the ingest queue was full
            or was dropped at shutdown.
        stale_dropped: In-window frames the recorder skipped because their
            processing lag exceeded `max_processing_lag_seconds`.
        inference_gaps: Intervals between written frames much longer than the
            nominal frame interval.
        pts_clamped: Written frames whose PTS was repaired (duplicate or
            backward timestamps).
    """

    decimated: int = 0
    ring_drops: int = 0
    queue_drops: int = 0
    stale_dropped: int = 0
    inference_gaps: int = 0
    pts_clamped: int = 0

    def to_dict(self) -> Dict[str, int]:
        """Return the counts as a plain dictionary.

        Returns:
            Mapping of loss category to count.
        """
        counts = {
            "decimated": self.decimated,
            "ring_drops": self.ring_drops,
            "queue_drops": self.queue_drops,
            "stale_dropped": self.stale_dropped,
            "inference_gaps": self.inference_gaps,
            "pts_clamped": self.pts_clamped,
        }

        return counts


@dataclass(frozen=True)
class ClipReport:
    """What a clip contains and how it differs from the requested window.

    Attributes:
        requested_window: Requested ``(start, end)``; the end is inclusive.
        actual_start: Timestamp of the first written frame (may lead the window
            by up to one output second because clips start on a keyframe).
        actual_end: Moment the last written frame stops showing, on the source
            axis.
        lead_seconds: Source seconds between ``actual_start`` and the window
            start when the clip starts early, else 0.
        frames: Number of written frames.
        frame_density_fps: Written frames per source second of the clip.
        source_fps_declared: Fps declared by the source, if any.
        playback_speed: Playback speed of the clip.
        clip_fps: Output frame-rate cap, if any.
        width: Output width in pixels.
        height: Output height in pixels.
        backend: Encoder backend, ``"x264"`` or ``"nvenc"``.
        partial: Why the clip ends or starts early: ``None``, ``"stalled"``
            (no frames for the grace period), ``"reset"`` (the source restarted:
            its frame number went backwards), ``"resolution_change"``,
            ``"shutdown"`` or ``"ring_evicted"``.
        loss: Missing frames by cause.
        wall_clock_jump_flagged: True if the timestamp axis stepped backwards
            by more than a second inside the clip.
        encode_ms_p50: Median encoder time per frame on the ring's thread.
        encode_ms_p95: 95th percentile of the same.
        ring_cost_ms_p50: Median `submit()` time per frame on the caller's
            (pipeline) thread.
        ring_cost_ms_p95: 95th percentile of the same.
        finalize_ms: Time spent muxing the clip.
    """

    requested_window: Tuple[datetime, datetime]
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    lead_seconds: float = 0.0
    frames: int = 0
    frame_density_fps: float = 0.0
    source_fps_declared: Optional[float] = None
    playback_speed: float = 1.0
    clip_fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    backend: str = "x264"
    partial: Optional[str] = None
    loss: LossCounts = field(default_factory=LossCounts)
    wall_clock_jump_flagged: bool = False
    encode_ms_p50: Optional[float] = None
    encode_ms_p95: Optional[float] = None
    ring_cost_ms_p50: Optional[float] = None
    ring_cost_ms_p95: Optional[float] = None
    finalize_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the report with ISO-8601 timestamps.

        Returns:
            JSON-compatible dictionary.
        """
        start, end = self.requested_window
        serialized = {
            "requested_window": {"start": _iso(start), "end": _iso(end)},
            "actual_start": _iso(self.actual_start),
            "actual_end": _iso(self.actual_end),
            "lead_seconds": self.lead_seconds,
            "frames": self.frames,
            "frame_density_fps": self.frame_density_fps,
            "source_fps_declared": self.source_fps_declared,
            "playback_speed": self.playback_speed,
            "clip_fps": self.clip_fps,
            "width": self.width,
            "height": self.height,
            "backend": self.backend,
            "partial": self.partial,
            "loss": self.loss.to_dict(),
            "wall_clock_jump_flagged": self.wall_clock_jump_flagged,
            "encode_ms_p50": self.encode_ms_p50,
            "encode_ms_p95": self.encode_ms_p95,
            "ring_cost_ms_p50": self.ring_cost_ms_p50,
            "ring_cost_ms_p95": self.ring_cost_ms_p95,
            "finalize_ms": self.finalize_ms,
        }

        return serialized


def _iso(timestamp: Optional[datetime]) -> Optional[str]:
    if timestamp is None:
        return None

    return timestamp.isoformat()
