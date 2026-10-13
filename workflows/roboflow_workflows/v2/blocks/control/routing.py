"""Value routing and throttling: V2 ports of three V1 flow-control blocks.

    v2/switch_case@v1    route to the first case whose key equals str(value)
    v2/delta_filter@v1   continue only when a lane's value changed
    v2/rate_limiter@v1   continue at most once per cooldown per lane

The two stateful blocks keep one record per lane::

    lane = ("video", video_metadata.video_identifier)  image with video metadata
           ("source", SampleContext.source_id)          else, lane field's source
           ("position", index[:1])                      else (passive fallback)

* The lane field is ``image`` for delta_filter, and ``video_reference_image``
  (else ``depends_on``) for rate_limiter. Only that field's source and time
  count, not the whole call's: a context-free value next to an image with a
  source still uses the image's source.
* Crops keep their frame's video metadata, so crops of one frame share the
  frame's lane. A plain ``session.run`` without metadata gives each top-level
  batch position its own lane across runs; a one-image run behaves like V1.
* Order comes from the engine, not from the lock. One run calls a step's
  invocations one after another in index order; pipelined runs keep pulse
  order within a domain. Calls of different domains may overlap: one lock per
  block keeps the records consistent, and a lane fed by two domains sees
  arrival order.
* Capacity: ``max_sources`` bounds the records. One lane too many fails with
  ``SourceCapacityError`` and changes nothing. Records are never evicted by
  time; a processing reset replaces the block and its records.

``rate_limiter`` differs from V1 on purpose: each lane has its own cooldown
(``scope="global"`` keeps V1's single one), and without a usable video time
it measures the observation time of the lane field's sample instead of the
wall clock when the block runs.
"""

import dataclasses
import math
import threading
from fractions import Fraction
from typing import Any, Dict, Hashable, List, Literal, Optional, Tuple

import numpy as np
import torch
from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
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

DEFAULT_MAX_SOURCES = 256
DEFAULT_COOLDOWN_SECONDS = 1.0

Lane = Tuple[Hashable, ...]


class SourceCapacityError(RuntimeError):
    """A new lane's record would exceed the block's ``max_sources``."""


class ClockMismatchError(ValueError):
    """A rate limiter lane received times from two different clocks."""


class SwitchCaseBlock(Block):
    """Route to the step of the case whose key equals ``str(value)``.

    Example: ``cases={"dog": "$steps.dogs", "car": "$steps.cars"}`` with
    ``default_next_steps=["$steps.review"]`` sends a ``"dog"`` top class to
    ``dogs``, ``"car"`` to ``cars`` and anything else to ``review``. Bound to
    per-crop values, each crop is routed alone.
    """

    type = "v2/switch_case@v1"
    metadata = {
        "name": "Switch Case",
        "port_of": "roboflow_core/switch_case@v1",
        "section": "flow_control",
    }

    class Params(BlockParams):
        value: (
            StrictStr | StrictBool | StrictInt | StrictFloat | Ref(allow_none=True)
        ) = Field(
            description=(
                "Value matched against the case keys after str(): True matches "
                "'True', 1.0 matches '1.0' (not '1'), None matches 'None'. A 0-d "
                "tensor is read as its Python scalar first. Bind one value such "
                "as top_class, not a predictions object: its str() matches no "
                "key, so only default_next_steps run."
            ),
            examples=["$steps.top_class.output", "$inputs.mode", "red"],
        )
        cases: Dict[str, StepRef] = Field(
            default_factory=dict,
            description=(
                "Case key to the step that runs when value matches it. A step "
                "may be targeted once across cases and default_next_steps."
            ),
            examples=[{"dog": "$steps.dogs", "car": "$steps.cars"}],
        )
        case_insensitive: StrictBool = Field(
            default=False,
            description="Match keys ignoring letter case ('RED' matches 'red').",
            examples=[False],
        )
        default_next_steps: List[StepRef] = Field(
            default_factory=list,
            description="Steps that run when no case matches; empty stops.",
            examples=[["$steps.review"]],
        )

        @model_validator(mode="after")
        def _targets_and_keys_are_unique(self) -> "SwitchCaseBlock.Params":
            targets = [*self.cases.values(), *self.default_next_steps]
            duplicated = sorted(
                {target for target in targets if targets.count(target) > 1}
            )
            if duplicated:
                raise ValueError(
                    "Each step may be targeted at most once across `cases` and "
                    f"`default_next_steps`. Duplicated targets: {duplicated}"
                )

            lowered = [key.lower() for key in self.cases]
            if self.case_insensitive and len(set(lowered)) != len(lowered):
                raise ValueError(
                    "`cases` contains keys that collide when case_insensitive "
                    "is enabled"
                )

            return self

    def run(
        self,
        *,
        value: Any,
        cases: Dict[str, str],
        case_insensitive: bool,
        default_next_steps: List[str],
    ) -> Select:
        """Select the matching case's step, else the default steps.

        Args:
            value: Value to match; ``None`` matches the key ``"None"``.
            cases: Case key to step selector.
            case_insensitive: Whether keys match ignoring letter case.
            default_next_steps: Steps selected when no case matches.

        Returns:
            ``Select`` of the matched step, of ``default_next_steps`` when
            nothing matches, or ``Stop()`` when both are empty.

        Raises:
            TypeError: When ``value`` is a tensor with one or more dimensions.
        """
        key = str(_scalar(value))
        if case_insensitive:
            key = key.lower()
            matched = next(
                (target for case, target in cases.items() if case.lower() == key),
                None,
            )
        else:
            matched = cases.get(key)

        if matched is not None:
            return Select(matched)

        decision = Select(default_next_steps) if default_next_steps else Stop()

        return decision


class DeltaFilterBlock(Block):
    """Continue to ``next_steps`` only when the stream's value changed.

    A stream (lane) is the image's ``video_identifier``, else its source, else
    its batch position; crops use their frame's lane.

    Example: bound to a per-frame object count of one camera, ``3, 3, 4, 4``
    continues on the first and third frame. Equality is Python ``==``, so
    ``1``, ``1.0`` and ``True`` are one value. Tensors, arrays, lists, dicts
    and prediction dataclasses compare element by element.

    The lane's last ``value`` is kept by reference, not copied
    (``retains_inputs``). A workflow step that mutates that payload in place
    changes the remembered value; the compiler adds a warning to
    ``CompiledWorkflow.warnings``, and
    ``CompileOptions(mutation_conflicts="error")`` rejects the workflow
    instead. The engine does not copy caller inputs either: a caller that
    changes an object in place and passes it again compares it with itself,
    which is no change. Pass a new object per run instead.
    """

    type = "v2/delta_filter@v1"
    metadata = {
        "name": "Delta Filter",
        "port_of": "roboflow_core/delta_filter@v1",
        "section": "flow_control",
    }
    retains_inputs = ("value",)

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description=(
                "Frame whose video_metadata.video_identifier names the lane. "
                "Without video metadata: this frame's source, else its batch "
                "position. The value's own source is not read."
            ),
            examples=["$inputs.image"],
        )
        value: Ref(allow_none=True) = Field(
            description=(
                "Value watched for changes. The first value of a lane passes, "
                "unless it is None. After a value, None counts as a change."
            ),
            examples=["$steps.count.output"],
        )
        next_steps: List[StepRef] = Field(
            description="Steps that run when the value changed.",
            examples=[["$steps.notify"]],
        )
        max_sources: StrictInt = Field(
            default=DEFAULT_MAX_SOURCES,
            ge=1,
            description=(
                "Most lanes (video streams, sources or batch positions) the "
                "block remembers a value for. One more fails with "
                "SourceCapacityError until a processing reset."
            ),
            examples=[DEFAULT_MAX_SOURCES],
        )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: Dict[Lane, Any] = {}

    def run(
        self,
        *,
        image: ImageData,
        value: Any,
        next_steps: List[str],
        max_sources: int,
    ) -> Select:
        """Select ``next_steps`` when ``value`` differs from the lane's last one.

        Args:
            image: Frame that names the lane.
            value: Current value; ``None`` is a value too.
            next_steps: Step selectors as written in the step parameters.
            max_sources: Most lanes with a remembered value.

        Returns:
            ``Select(next_steps)`` on a change, else ``Stop()``.

        Raises:
            TypeError: When ``value`` and the remembered value cannot be
                compared to a single bool.
            SourceCapacityError: When a new lane would exceed ``max_sources``.
        """
        lane = _lane_of(image, block=self, field="image")
        with self._lock:
            if lane in self._values:
                unchanged = _values_equal(self._values[lane], value)
            else:
                # V1 compares with dict.get's None: only a first None is no change.
                unchanged = value is None
            if unchanged:
                return Stop()

            if lane not in self._values:
                _check_capacity(
                    self, records=self._values, lane=lane, limit=max_sources
                )
            self._values[lane] = value

        return Select(next_steps)


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

    type = "v2/rate_limiter@v1"
    metadata = {
        "name": "Rate Limiter",
        "port_of": "roboflow_core/rate_limiter@v1",
        "section": "flow_control",
    }

    class Params(BlockParams):
        cooldown_seconds: StrictFloat = Field(
            default=DEFAULT_COOLDOWN_SECONDS,
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


def _values_equal(first: Any, second: Any) -> bool:
    # Python == for scalars and plain objects, so 1 == 1.0 == True. Containers,
    # tensors, arrays and dataclasses compare element by element; a 0-d tensor
    # equals the Python scalar it holds. Raises TypeError when == is no bool.
    if first is second:
        return True

    first_tensor = isinstance(first, torch.Tensor)
    second_tensor = isinstance(second, torch.Tensor)
    if first_tensor and second_tensor:
        same = first.shape == second.shape and torch.equal(
            first, second.to(first.device)
        )
        return same
    if first_tensor or second_tensor:
        tensor, other = (first, second) if first_tensor else (second, first)
        same = tensor.dim() == 0 and _is_scalar(other) and tensor.item() == other
        return same

    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        same = (
            isinstance(first, np.ndarray)
            and isinstance(second, np.ndarray)
            and first.shape == second.shape
            and bool(np.array_equal(first, second))
        )
        return same

    if isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
        same = (
            type(first) is type(second)
            and len(first) == len(second)
            and all(_values_equal(a, b) for a, b in zip(first, second))
        )
        return same

    if isinstance(first, dict) or isinstance(second, dict):
        same = (
            isinstance(first, dict)
            and isinstance(second, dict)
            and first.keys() == second.keys()
            and all(_values_equal(first[key], second[key]) for key in first)
        )
        return same

    if _is_dataclass_instance(first) or _is_dataclass_instance(second):
        same = type(first) is type(second) and all(
            _values_equal(getattr(first, field.name), getattr(second, field.name))
            for field in dataclasses.fields(first)
        )
        return same

    result = first == second
    if not isinstance(result, (bool, np.bool_)):
        raise TypeError(
            f"cannot decide whether {type(first).__name__} and "
            f"{type(second).__name__} values are equal: == returned "
            f"{type(result).__name__}, not a bool"
        )

    return bool(result)


def _within_cooldown(elapsed: Fraction, cooldown_seconds: float) -> bool:
    if math.isinf(cooldown_seconds):
        return True

    # str() keeps the written value: 0.1 is 1/10, not its binary float.
    within = elapsed < Fraction(str(cooldown_seconds))

    return within


def _scalar(value: Any) -> Any:
    if not isinstance(value, torch.Tensor):
        return value

    if value.dim() != 0:
        raise TypeError(
            "v2/switch_case@v1 matches one value; got a tensor of shape "
            f"{tuple(value.shape)}. Extract one element first."
        )

    scalar = value.item()

    return scalar


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (bool, int, float, complex, np.generic))


def _is_dataclass_instance(value: Any) -> bool:
    return dataclasses.is_dataclass(value) and not isinstance(value, type)


def _lane_of(image: Optional[ImageData], *, block: Block, field: str) -> Lane:
    metadata = image.video_metadata if image is not None else None
    if metadata is not None:
        return ("video", metadata.video_identifier)

    # The field's own source: the whole call's merge is None when another
    # argument carries no context, which would collapse sources to positions.
    context = block.execution_context
    index = context.indices[0]
    sample = context.sample_at(index, field=field)
    if sample is not None:
        return ("source", sample.source_id)

    lane = ("position", index[:1])

    return lane


def _video_time_of(image: Optional[ImageData]) -> Optional[Tuple[str, Fraction]]:
    metadata = image.video_metadata if image is not None else None
    if metadata is None:
        return None

    fps = metadata.fps
    if fps is None or isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
        return None

    # str(fps) keeps the written rate: 29.97 is 2997/100, not its binary float.
    seconds = Fraction(metadata.frame_number) / Fraction(str(fps))
    clock_id = f"video-time:{metadata.video_identifier}"

    return clock_id, seconds


def _check_clock(last_clock_id: str, *, clock_id: str, lane: Lane) -> None:
    if clock_id != last_clock_id:
        raise ClockMismatchError(
            f"v2/rate_limiter@v1: lane {lane!r} last passed on clock "
            f"{last_clock_id!r} and is now timed on {clock_id!r}; times of two "
            "clocks cannot be compared. Give every call of a lane the same "
            "video_reference_image (with a finite positive fps) or none; with "
            "scope='global', video time works only for one video."
        )


def _check_capacity(
    block: Block, *, records: Dict[Lane, Any], lane: Lane, limit: int
) -> None:
    if len(records) >= limit:
        raise SourceCapacityError(
            f"{block.type}: lane {lane!r} would be lane {len(records) + 1} of "
            f"max_sources={limit}. The block keeps one record per video "
            "stream, source or batch position for the step's lifetime; none "
            "is dropped by time. Raise max_sources, or reset processing to "
            "drop the records."
        )
