"""Query-condition gate: the V2 port of V1 ``roboflow_core/continue_if@v1``.

``roboflow_core/continue_if_boolean@v1`` takes a ready boolean. This block evaluates a V1 UQL
``StatementGroup`` instead, compiled once when the workflow compiles::

    evaluation_parameters ── condition_statement ── passed?
      passed                            -> Select(next_steps)
      failed, stop_delay == 0           -> Stop()
      failed, a true of this lane at most stop_delay seconds earlier
                                        -> Select(next_steps)   (hold)
      failed otherwise                  -> Stop()

Lanes, observation time and their limits:

    lane = ("source", SampleContext.source_id)  if the call's index has a source
           ("position", index[:1])               otherwise (passive fallback)
    time = TemporalContext.observed_coverage     (a span: its end)
           one engine_observation() per run      without temporal context

    lane record (stop_delay > 0 only), kept for the step's lifetime:
        clock_id, last observation time         ordering guard
        latest true, latest earlier true        hold history

* Observation time, not processing time. V1 measures ``stop_delay`` with
  ``time.time()`` when the gate runs. V2 measures it between the times the
  source observed the samples, so a slow upstream step neither shortens nor
  lengthens a hold. Active runs stamp every pulse with its source and its
  observation time on the engine's monotonic clock, so each source has its
  own hold. V1 kept one timer per block; one camera's true kept all open.
* Passive fallback: a plain ``session.run`` carries neither. Each top-level
  batch position is then one lane across runs, and a run is one moment.
  Images of one batch never hold each other open; a single-image run behaves
  like V1. Inputs given ``InputValue`` sample/temporal metadata use them.
* Order: every call with ``stop_delay > 0``, true or false, must use its
  lane's clock and a time no earlier than the lane's last one; otherwise it
  fails with ``ObservationOrderError`` and changes nothing. Engine-stamped
  pulses always satisfy this. Members a source collects out of order must be
  ordered before this gate. This is not event-time reordering support.
* A false call is held only by a true of an EARLIER time. Several calls of
  one frame (per-crop gates) share the frame's time, so their order does not
  change the result; one crop's true holds its lane's other crops open only
  from the next frame on.
* Each call measures the hold with its own ``stop_delay``, as V1 does, so a
  selector may change it per run. A false call that finds its lane's hold
  expired clears the hold history but keeps the ordering guard; a later,
  longer ``stop_delay`` does not revive the hold.
* ``stop_delay == 0`` bypasses the records: it neither reads nor writes them
  and accepts any time. A later nonzero call is checked against the last
  nonzero-delay observation; zero-delay calls are not part of the history.
* Capacity: every lane seen with ``stop_delay > 0`` keeps its record, also
  after its hold expires and for lanes that were only ever false. Record
  ``max_sources + 1`` fails with ``HoldCapacityError``; nothing is reclaimed
  by time, because one source's time cannot prove another source's hold is
  over. A processing reset replaces the block and its records.
* Concurrency: the engine serializes a step's calls only within one domain
  (pulse order), so calls of different sources may overlap. One lock per
  block instance guards the records and the run moment; the condition itself
  is evaluated outside it.
"""

import math
import threading
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, Any, Dict, Hashable, List, Mapping, Optional, Tuple
from pydantic import (
    Field,
    StrictFloat,
    StrictInt,
    ValidationInfo,
    field_validator,
    model_validator,
)
from roboflow_workflows.v2.blocks.query import (
    CompiledCondition,
    StatementGroup,
    compile_condition,
)
from roboflow_workflows.v2.data.batch import TimeSpan, Timestamp
from roboflow_workflows.v2.data.kinds import FLOAT_KIND
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

StopDelay = Annotated[StrictFloat, Field(ge=0, allow_inf_nan=False)]

CONDITION_STATEMENT_EXAMPLE = {
    "type": "StatementGroup",
    "statements": [
        {
            "type": "BinaryStatement",
            "left_operand": {
                "type": "DynamicOperand",
                "operand_name": "prediction",
                "operations": [
                    {
                        "type": "ClassificationPropertyExtract",
                        "property_name": "top_class_confidence",
                    }
                ],
            },
            "comparator": {"type": "(Number) >="},
            "right_operand": {"type": "DynamicOperand", "operand_name": "minimum"},
        }
    ],
}

class HoldCapacityError(RuntimeError):
    """A new lane's record would exceed the block's ``max_sources``."""

class ObservationOrderError(ValueError):
    """A ``stop_delay`` call went back in time or changed its lane's clock."""

class ContinueIfQueryBlock(Block):
    """Gate on a V1 query; for a ready boolean use ``roboflow_core/continue_if_boolean@v1``.

    This ports the query condition of ``roboflow_core/continue_if@v1``.

    Example: with ``CONDITION_STATEMENT_EXAMPLE`` and
    ``evaluation_parameters={"prediction": "$steps.classifier.predictions",
    "minimum": 0.7}``, an image whose top class has confidence 0.92 continues
    and one with 0.50 stops. Bound to per-crop values, each crop decides alone.
    """

    type = "roboflow_core/continue_if@v1"
    metadata = {
        "name": "Continue If (condition)",
        "port_of": "roboflow_core/continue_if@v1",
        "section": "flow_control",
    }

    class Params(BlockParams):
        @model_validator(mode="before")
        @classmethod
        def _reject_boolean_condition(cls, values: Any) -> Any:
            if isinstance(values, Mapping) and "condition" in values:
                raise ValueError(
                    "condition is a field of roboflow_core/continue_if_boolean@v1 (ready boolean). "
                    "roboflow_core/continue_if@v1 takes a V1 query condition_statement "
                    "and evaluation_parameters."
                )

            return values

        # Before condition_statement: its validator reads these names.
        evaluation_parameters: Dict[str, Any | Ref()] = Field(
            default_factory=dict,
            description=(
                "Values the condition names, e.g. {'prediction': "
                "'$steps.classifier.predictions', 'minimum': '$inputs.minimum'}."
            ),
            examples=[{"prediction": "$steps.classifier.predictions", "minimum": 0.7}],
        )
        condition_statement: StatementGroup = Field(
            description=(
                "V1 UQL StatementGroup. Dynamic operands name keys of "
                "evaluation_parameters."
            ),
            examples=[CONDITION_STATEMENT_EXAMPLE],
        )
        next_steps: List[StepRef] = Field(
            description="Steps that run while the gate is open; empty always stops.",
            examples=[["$steps.detector"]],
        )
        stop_delay: StopDelay | Ref(FLOAT_KIND) = Field(
            default=0,
            description=(
                "Seconds a source's gate stays open after its last true "
                "condition, measured between observation times (V1 measures "
                "processing time). 0 closes at once and keeps no state. Above "
                "0, each source's times must not go backward. Without source "
                "metadata each batch position has its own hold."
            ),
            examples=[0, 5],
        )
        max_sources: StrictInt = Field(
            default=DEFAULT_MAX_SOURCES,
            ge=1,
            description=(
                "Most sources (or batch positions) the gate tracks with a "
                "nonzero stop_delay. Each keeps a record for the step's "
                "lifetime, even after its hold expires; one more fails with "
                "HoldCapacityError until a processing reset."
            ),
            examples=[DEFAULT_MAX_SOURCES],
        )

        # Not a model validator: the engine reruns those per invocation.
        # This literal-only field is validated once per definition, and
        # selectors replace evaluation_parameters values, never keys.
        @field_validator("condition_statement")
        @classmethod
        def _condition_compiles(
            cls, value: StatementGroup, info: ValidationInfo
        ) -> StatementGroup:
            if "evaluation_parameters" in info.data:
                compile_gate_condition(
                    value, parameter_names=tuple(info.data["evaluation_parameters"])
                )

            return value

    def __init__(self) -> None:
        self._compiled: Optional[
            Tuple[StatementGroup, Tuple[str, ...], CompiledCondition]
        ] = None
        self._lock = threading.Lock()
        self._lanes = _LaneRecords()
        self._run_moment: Optional[Tuple[str, Timestamp]] = None

    def run(
        self,
        *,
        condition_statement: StatementGroup,
        evaluation_parameters: Dict[str, Any],
        next_steps: List[str],
        stop_delay: float,
        max_sources: int,
    ) -> Select:
        """Evaluate the condition and select ``next_steps`` or stop.

        Args:
            condition_statement: Validated UQL condition.
            evaluation_parameters: Resolved values by operand name.
            next_steps: Step selectors as written in the step parameters.
            stop_delay: Hold time in seconds, ``>= 0``.
            max_sources: Most lanes with a record.

        Returns:
            ``Select(next_steps)`` when the condition holds or a hold is
            active, else ``Stop()``.

        Raises:
            QueryRuntimeError: When the condition cannot evaluate its values.
            ValueError: When a selector-bound ``stop_delay`` is negative or
                not finite.
            ObservationOrderError: When a nonzero-delay call is earlier than
                its lane's last one or uses another clock.
            HoldCapacityError: When a new lane would exceed ``max_sources``.
        """
        if not next_steps:
            return Stop()

        condition = self._condition(condition_statement, tuple(evaluation_parameters))
        passed = bool(condition.evaluate(evaluation_parameters))
        if stop_delay == 0:
            return Select(next_steps) if passed else Stop()

        delay = _checked_delay(stop_delay)
        with self._lock:
            lane, now = self._lane_and_time()
            held = self._lanes.observe(
                lane, now=now, passed=passed, delay=delay, max_sources=max_sources
            )
        decision = Select(next_steps) if held else Stop()

        return decision

    def _condition(
        self, definition: StatementGroup, parameter_names: Tuple[str, ...]
    ) -> CompiledCondition:
        # Compiled once per step; recompiled only if a graph update changes it.
        with self._lock:
            cached = self._compiled
            if (
                cached is None
                or cached[0] != definition
                or cached[1] != parameter_names
            ):
                condition = compile_gate_condition(
                    definition, parameter_names=parameter_names
                )
                cached = (definition, parameter_names, condition)
                self._compiled = cached

        return cached[2]

    def _lane_and_time(self) -> Tuple[Hashable, Timestamp]:
        context = self.execution_context
        index = context.indices[0]
        sample = context.sample_at(index)
        temporal = context.temporal_at(index)
        if sample is not None:
            lane = ("source", sample.source_id)
        else:
            lane = ("position", index[:1])
        if temporal is None:
            return lane, self._moment_of_run(context.run_id)

        coverage = temporal.observed_coverage
        now = coverage.end if isinstance(coverage, TimeSpan) else coverage

        return lane, now

    def _moment_of_run(self, run_id: Optional[str]) -> Timestamp:
        # Every call of one run without temporal context shares one time.
        # Called under self._lock.
        cached = self._run_moment
        if run_id is None or cached is None or cached[0] != run_id:
            cached = (run_id, engine_observation())
            self._run_moment = cached

        return cached[1]

def compile_gate_condition(
    definition: StatementGroup, *, parameter_names: Tuple[str, ...]
) -> CompiledCondition:
    """Compile a gate condition whose operands are the declared parameters.

    Args:
        definition: Validated UQL condition.
        parameter_names: Keys of ``evaluation_parameters``.

    Returns:
        The compiled condition.

    Raises:
        QueryDefinitionError: When an operand or nested filter names a value
            that is not in ``parameter_names``.
    """
    condition = compile_condition(
        definition,
        path="$.condition_statement",
        declared_names=set(parameter_names),
    )

    return condition

def _checked_delay(stop_delay: float) -> Fraction:
    # Literals are validated by Params; a selector-bound value is not.
    if isinstance(stop_delay, bool) or not math.isfinite(stop_delay) or stop_delay < 0:
        raise ValueError(
            f"stop_delay must be a finite number >= 0 seconds, got {stop_delay!r}"
        )

    delay = Fraction(stop_delay)

    return delay

@dataclass
class _Lane:
    """Ordering guard and hold history of one lane, on one clock.

    ``previous_true`` is the latest true strictly before ``latest_true``; it
    holds the other crops of the frame that set ``latest_true``.
    """

    clock_id: str
    last_seen: Fraction
    latest_true: Optional[Fraction] = None
    previous_true: Optional[Fraction] = None

    def record_true(self, moment: Fraction) -> None:
        if self.latest_true is None or moment > self.latest_true:
            self.previous_true = self.latest_true
            self.latest_true = moment

    def is_held(self, moment: Fraction, *, delay: Fraction) -> bool:
        if self.latest_true is None:
            return False

        # Times never go backward, so moment >= latest_true here.
        last_true = (
            self.latest_true if self.latest_true < moment else self.previous_true
        )
        held = last_true is not None and moment - last_true <= delay
        if moment - self.latest_true > delay:
            self.latest_true = None
            self.previous_true = None

        return held

class _LaneRecords:
    """Lane records of one gate step. Callers hold the block's lock."""

    def __init__(self) -> None:
        self._lanes: Dict[Hashable, _Lane] = {}

    def observe(
        self,
        lane: Hashable,
        *,
        now: Timestamp,
        passed: bool,
        delay: Fraction,
        max_sources: int,
    ) -> bool:
        """Record one nonzero-delay call and say whether its lane is open.

        Checks run before any change, so a failing call leaves records as
        they were.
        """
        record = self._lanes.get(lane)
        if record is None:
            record = self._new_record(lane, now=now, max_sources=max_sources)
        else:
            _check_order(record, now=now, lane=lane)
            record.last_seen = now.seconds
        if passed:
            record.record_true(now.seconds)
            return True

        held = record.is_held(now.seconds, delay=delay)

        return held

    def _new_record(self, lane: Hashable, *, now: Timestamp, max_sources: int) -> _Lane:
        if len(self._lanes) >= max_sources:
            raise HoldCapacityError(
                f"roboflow_core/continue_if@v1: lane {lane!r} would be lane "
                f"{len(self._lanes) + 1} of max_sources={max_sources}. With "
                "stop_delay > 0 the gate keeps one record (clock and last "
                "observation time) per source, or per batch position without "
                "a source, for the step's lifetime, even after its hold "
                "expires. Raise max_sources, or reset processing to drop the "
                "records."
            )

        record = _Lane(clock_id=now.clock_id, last_seen=now.seconds)
        self._lanes[lane] = record

        return record

def _check_order(record: _Lane, *, now: Timestamp, lane: Hashable) -> None:
    if now.clock_id != record.clock_id:
        raise ObservationOrderError(
            f"roboflow_core/continue_if@v1: lane {lane!r} changed clocks from "
            f"{record.clock_id!r} to {now.clock_id!r}; stop_delay needs one clock"
        )
    if now.seconds < record.last_seen:
        raise ObservationOrderError(
            f"roboflow_core/continue_if@v1: lane {lane!r} observed time "
            f"{float(now.seconds):g} s after {float(record.last_seen):g} s. "
            "With stop_delay > 0 a lane's observation times must not go "
            "backward; equal times (calls of one frame) are fine. Order "
            "collected or out-of-order members before this gate, or use "
            "stop_delay=0."
        )
