"""Continue to ``next_steps`` at most once per ``cooldown_seconds`` per stream.

Canonical type: ``roboflow_core/rate_limiter@v1``.
"""

import threading
from fractions import Fraction
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import Field, StrictFloat, StrictInt
from roboflow_workflows.v2.blocks.control._shared.routing import (
    DEFAULT_MAX_SOURCES,
    Lane,
    _check_capacity,
    _check_clock,
    _lane_of,
    _video_time_of,
    _within_cooldown,
)
from roboflow_workflows.v2.data.batch import TimeSpan, Timestamp
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    Ref,
    Select,
    StepRef,
    Stop,
)
from roboflow_workflows.v2.engine.sources import engine_observation


class RateLimiterBlock(Block):
    """Continue to ``next_steps`` at most once per ``cooldown_seconds`` per stream.

    The lane field is ``video_reference_image``, or ``depends_on`` when that
    is unset. Its ``video_identifier`` (an image with video metadata), else
    its source, else its batch position names the stream (lane).

    Example: two cameras at 10 fps with ``cooldown_seconds=1`` and
    ``video_reference_image`` each continue on frames 0, 10, 20 of their own
    video. With ``scope="global"`` all cameras share one cooldown, as in V1.
    Then leave ``video_reference_image`` unset, so the cooldown runs on
    observation time: video times of two videos cannot be compared, and a
    second video fails with ``ClockMismatchError``.

    Observation time is the end of the lane field's observed time (set by
    stream sources and timestamped inputs; a field carrying no context takes
    the run's origin, e.g. its source pulse). Without one, as in a plain
    ``session.run``, every call of one run shares the engine time at which the
    run first reached the block, so cooldowns are real seconds between runs.
    """

    type = "roboflow_core/rate_limiter@v1"
    metadata = {
        "name": "Rate Limiter",
        "port_of": "roboflow_core/rate_limiter@v1",
        "section": "flow_control",
    }

    class Params(BlockParams):
        cooldown_seconds: StrictFloat = Field(
            default=1.0,
            ge=0.0,
            description=(
                "Least time between two passes of one lane. A call exactly "
                "cooldown_seconds after the last pass continues."
            ),
            examples=[1.0],
        )
        depends_on: Ref(allow_none=True) = Field(
            description=(
                "Output of the step this limiter follows; it orders the step, "
                "its value is not read. Without video_reference_image, its "
                "source names the lane and its observation time is the clock."
            ),
            examples=["$steps.model.predictions"],
        )
        next_steps: List[StepRef] = Field(
            description="Steps that run when the cooldown allows.",
            examples=[["$steps.upload"]],
        )
        video_reference_image: Optional[Ref(IMAGE_KIND)] = Field(
            default=None,
            description=(
                "Frame whose video time (frame_number / fps) is the clock, and "
                "whose video_identifier, else source, names the lane. Without a "
                "finite positive fps, this frame's observation time is used. "
                "Leave it unset with scope='global' over several videos."
            ),
            examples=["$inputs.image"],
        )
        scope: Literal["source", "global"] = Field(
            default="source",
            description=(
                "'source': each video stream, source or batch position has its "
                "own cooldown. 'global': one cooldown for every call (V1). With "
                "'global', video time works for one video only; several videos "
                "need observation time (no video_reference_image)."
            ),
            examples=["source"],
        )
        max_sources: StrictInt = Field(
            default=DEFAULT_MAX_SOURCES,
            ge=1,
            description=(
                "Most lanes the block keeps a last-pass time for. One more "
                "fails with SourceCapacityError until a processing reset."
            ),
            examples=[DEFAULT_MAX_SOURCES],
        )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_passes: Dict[Lane, Tuple[str, Fraction]] = {}
        self._run_moment: Optional[Tuple[Optional[str], Timestamp]] = None

    def run(
        self,
        *,
        cooldown_seconds: float,
        depends_on: Any,
        next_steps: List[str],
        video_reference_image: Optional[ImageData],
        scope: str,
        max_sources: int,
    ) -> Select:
        """Select ``next_steps`` when the lane's cooldown has elapsed.

        Args:
            cooldown_seconds: Least seconds between two passes of a lane.
            depends_on: Value ignored; binds the step after its predecessor.
                Without ``video_reference_image`` its context gives lane
                and time.
            next_steps: Step selectors as written in the step parameters.
            video_reference_image: Frame giving video time and lane, if any.
            scope: ``"source"`` for a cooldown per lane, ``"global"`` for one.
            max_sources: Most lanes with a last-pass time.

        Returns:
            ``Select(next_steps)`` when allowed, else ``Stop()``.

        Raises:
            ClockMismatchError: When the lane's last pass used another clock.
            SourceCapacityError: When a new lane would exceed ``max_sources``.
        """
        reference = (
            "depends_on" if video_reference_image is None else "video_reference_image"
        )
        if scope == "global":
            lane: Lane = ("global",)
        else:
            lane = _lane_of(video_reference_image, block=self, field=reference)

        with self._lock:
            clock_id, now = self._time_of(video_reference_image, field=reference)
            last_pass = self._last_passes.get(lane)
            if last_pass is not None:
                _check_clock(last_pass[0], clock_id=clock_id, lane=lane)
                if _within_cooldown(now - last_pass[1], cooldown_seconds):
                    return Stop()
            else:
                _check_capacity(
                    self, records=self._last_passes, lane=lane, limit=max_sources
                )
            self._last_passes[lane] = (clock_id, now)

        return Select(next_steps)

    def _time_of(
        self, image: Optional[ImageData], *, field: str
    ) -> Tuple[str, Fraction]:
        # Called under self._lock (the per-run moment is shared state).
        video_time = _video_time_of(image)
        if video_time is not None:
            return video_time

        context = self.execution_context
        temporal = context.temporal_at(context.indices[0], field=field)
        if temporal is None:
            stamp = self._moment_of_run(context.run_id)
        else:
            coverage = temporal.observed_coverage
            stamp = coverage.end if isinstance(coverage, TimeSpan) else coverage

        return stamp.clock_id, stamp.seconds

    def _moment_of_run(self, run_id: Optional[str]) -> Timestamp:
        # Every call of one run without temporal context shares one time.
        cached = self._run_moment
        if run_id is None or cached is None or cached[0] != run_id:
            cached = (run_id, engine_observation())
            self._run_moment = cached

        return cached[1]
