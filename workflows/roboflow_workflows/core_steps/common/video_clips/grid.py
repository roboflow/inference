"""Pure grid, PTS and timeline planning for video clips (no I/O, no `av`).

Product contract (plan rev 8, section 1):

* Times are source seconds on the `frame_timestamp` axis (naive timestamps are
  read as UTC).
* With an output cap ``f`` (`clip_fps`) and playback speed ``s`` the grid keeps
  the first frame of each ``s / f`` second slot, slots anchored at the first
  frame the ring saw (subscription start). Without a cap every frame is kept.
* Output PTS is ``round((t - actual_start) / s * 90000)``; the first PTS is 0.
  Duplicate or backward PTS are repaired to the previous PTS + 1 tick and
  counted; nothing else moves a frame: large forward gaps stay gaps (counted
  as inference gaps), so no frame is ever moved forward into a window.
* Final-frame duration: ``1 / clip_fps`` if set, else the median of the last
  (up to) five source intervals divided by ``s``, else ``1 / source_fps``,
  else 1/30 s.
"""

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Sequence

PTS_CLOCK_RATE = 90000
DEFAULT_FALLBACK_FPS = 30.0
FINAL_DURATION_INTERVALS = 5
SLOT_EPSILON = 1e-9
INFERENCE_GAP_FACTOR = 2.0
WALL_CLOCK_JUMP_SECONDS = 1.0


def to_seconds(timestamp: datetime) -> float:
    """Convert a frame timestamp to POSIX seconds.

    Args:
        timestamp: Aware timestamp, or naive timestamp in UTC.

    Returns:
        Seconds since the epoch.
    """
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    seconds = timestamp.timestamp()

    return seconds


def grid_slot(
    t: float,
    *,
    grid_origin: float,
    playback_speed: float,
    clip_fps: float,
) -> int:
    """Return the output slot a source time falls in.

    Args:
        t: Source time in seconds.
        grid_origin: Source time the grid is anchored at.
        playback_speed: Playback speed ``s``.
        clip_fps: Output frame-rate cap ``f``.

    Returns:
        Index of the ``s / f`` second slot containing ``t``.
    """
    slot_length = playback_speed / clip_fps
    slot = math.floor((t - grid_origin) / slot_length + SLOT_EPSILON)

    return slot


class GridSelector:
    """Decide, frame by frame, which source frames the output grid keeps.

    Stateful: the grid is anchored at the first frame offered. A frame whose
    slot is not after the last kept frame's slot is decimated, which also drops
    frames with backward timestamps when a cap is set.

    Args:
        playback_speed: Playback speed ``s`` (> 0).
        clip_fps: Output frame-rate cap ``f``; ``None`` keeps every frame.

    Raises:
        ValueError: If ``playback_speed`` or ``clip_fps`` is not positive.
    """

    def __init__(self, *, playback_speed: float, clip_fps: Optional[float]):
        if playback_speed <= 0:
            raise ValueError(f"playback_speed must be > 0, got {playback_speed}")
        if clip_fps is not None and clip_fps <= 0:
            raise ValueError(f"clip_fps must be > 0, got {clip_fps}")

        self._playback_speed = playback_speed
        self._clip_fps = clip_fps
        self._origin: Optional[float] = None
        self._last_slot: Optional[int] = None

    @property
    def origin(self) -> Optional[float]:
        """Source time the grid is anchored at, once a frame was offered."""
        return self._origin

    def accept(self, t: float) -> bool:
        """Offer a frame to the grid.

        Args:
            t: Source time of the frame in seconds.

        Returns:
            True when the frame is kept, False when it is decimated.
        """
        if self._origin is None:
            self._origin = t
        if self._clip_fps is None:
            return True

        slot = grid_slot(
            t,
            grid_origin=self._origin,
            playback_speed=self._playback_speed,
            clip_fps=self._clip_fps,
        )
        if self._last_slot is not None and slot <= self._last_slot:
            return False

        self._last_slot = slot

        return True


@dataclass
class TimelinePlan:
    """Output timing of a clip, derived from its frames' source times.

    Attributes:
        pts: Output PTS of each frame in 90 kHz ticks; the first is 0.
        final_duration_ticks: Display duration of the last frame.
        duration_ticks: Total output duration (last PTS + final duration).
        actual_end_offset_seconds: Source seconds from the first frame to the
            moment the last frame stops showing.
        pts_clamped: Frames whose PTS was repaired (duplicate or backward).
        inference_gaps: Source intervals longer than the gap threshold.
        wall_clock_jump_flagged: True if a backward step larger than
            `WALL_CLOCK_JUMP_SECONDS` was seen on the timestamp axis.
    """

    pts: List[int] = field(default_factory=list)
    final_duration_ticks: int = 0
    duration_ticks: int = 0
    actual_end_offset_seconds: float = 0.0
    pts_clamped: int = 0
    inference_gaps: int = 0
    wall_clock_jump_flagged: bool = False


def plan_pts(
    times: Sequence[float],
    *,
    actual_start: float,
    playback_speed: float,
) -> List[int]:
    """Return strictly increasing output PTS for source times.

    Args:
        times: Source times in output order.
        actual_start: Source time mapped to PTS 0 (the first frame's time).
        playback_speed: Playback speed ``s``.

    Returns:
        PTS in 90 kHz ticks; duplicates and backward steps become the previous
        PTS + 1. The first PTS is clamped to 0 when ``times[0]`` is not later
        than ``actual_start``.
    """
    pts_values: List[int] = []
    previous_pts = -1
    for t in times:
        pts = round((t - actual_start) / playback_speed * PTS_CLOCK_RATE)
        if pts <= previous_pts:
            pts = previous_pts + 1
        pts_values.append(pts)
        previous_pts = pts

    return pts_values


def final_frame_duration_ticks(
    times: Sequence[float],
    *,
    playback_speed: float,
    clip_fps: Optional[float],
    declared_fps: Optional[float],
) -> int:
    """Return the display duration of the last output frame.

    Args:
        times: Source times of the written frames.
        playback_speed: Playback speed ``s``.
        clip_fps: Output frame-rate cap, if any.
        declared_fps: Declared source fps, if known (non-positive is ignored).

    Returns:
        Duration in 90 kHz ticks, at least one tick.
    """
    intervals = _positive_intervals(times[-(FINAL_DURATION_INTERVALS + 1) :])
    if clip_fps:
        seconds = 1.0 / clip_fps
    elif intervals:
        seconds = statistics.median(intervals) / playback_speed
    elif declared_fps and declared_fps > 0:
        seconds = 1.0 / declared_fps
    else:
        seconds = 1.0 / DEFAULT_FALLBACK_FPS
    ticks = max(1, round(seconds * PTS_CLOCK_RATE))

    return ticks


def count_inference_gaps(
    times: Sequence[float],
    *,
    playback_speed: float,
    clip_fps: Optional[float],
    declared_fps: Optional[float],
) -> int:
    """Count source intervals that are much longer than the nominal interval.

    The nominal interval is the declared frame interval (or the median measured
    interval when the fps is unknown), raised to the grid slot length when a
    cap is set. An interval counts when it exceeds `INFERENCE_GAP_FACTOR` times
    the nominal interval.

    Args:
        times: Source times of the written frames.
        playback_speed: Playback speed ``s``.
        clip_fps: Output frame-rate cap, if any.
        declared_fps: Declared source fps, if known.

    Returns:
        Number of gaps.
    """
    intervals = [b - a for a, b in zip(times, times[1:])]
    if not intervals:
        return 0

    positive_intervals = [interval for interval in intervals if interval > 0]
    if declared_fps and declared_fps > 0:
        nominal = 1.0 / declared_fps
    elif positive_intervals:
        nominal = statistics.median(positive_intervals)
    else:
        return 0

    if clip_fps:
        nominal = max(nominal, playback_speed / clip_fps)
    threshold = INFERENCE_GAP_FACTOR * nominal
    gaps = sum(1 for interval in intervals if interval > threshold)

    return gaps


def plan_timeline(
    times: Sequence[float],
    *,
    playback_speed: float,
    clip_fps: Optional[float],
    declared_fps: Optional[float],
) -> TimelinePlan:
    """Plan the output timing of a clip.

    Args:
        times: Source times of the frames in output (decode) order.
        playback_speed: Playback speed ``s`` (> 0).
        clip_fps: Output frame-rate cap, if any.
        declared_fps: Declared source fps, if known.

    Returns:
        The timeline plan; empty when ``times`` is empty.

    Raises:
        ValueError: If ``playback_speed`` is not positive.
    """
    if playback_speed <= 0:
        raise ValueError(f"playback_speed must be > 0, got {playback_speed}")
    if not times:
        return TimelinePlan()

    actual_start = times[0]
    pts_values = plan_pts(
        times, actual_start=actual_start, playback_speed=playback_speed
    )
    unrepaired = [
        round((t - actual_start) / playback_speed * PTS_CLOCK_RATE) for t in times
    ]
    pts_clamped = sum(
        1 for repaired, raw in zip(pts_values, unrepaired) if repaired != raw
    )
    final_duration = final_frame_duration_ticks(
        times,
        playback_speed=playback_speed,
        clip_fps=clip_fps,
        declared_fps=declared_fps,
    )
    duration_ticks = pts_values[-1] + final_duration

    plan = TimelinePlan(
        pts=pts_values,
        final_duration_ticks=final_duration,
        duration_ticks=duration_ticks,
        actual_end_offset_seconds=duration_ticks / PTS_CLOCK_RATE * playback_speed,
        pts_clamped=pts_clamped,
        inference_gaps=count_inference_gaps(
            times,
            playback_speed=playback_speed,
            clip_fps=clip_fps,
            declared_fps=declared_fps,
        ),
        wall_clock_jump_flagged=any(
            b - a < -WALL_CLOCK_JUMP_SECONDS for a, b in zip(times, times[1:])
        ),
    )

    return plan


def _positive_intervals(times: Sequence[float]) -> List[float]:
    return [b - a for a, b in zip(times, times[1:]) if b > a]
