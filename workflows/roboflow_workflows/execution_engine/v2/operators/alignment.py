"""``v2/align@v1``: pair each leader sample with the nearest follower samples.

Alignment is a declared correspondence, never inferred from equal clocks or
arrival order. Every input promises strictly increasing timestamps on the
declared clock; a leader sample is decided once every follower has reached
``leader + tolerance`` or ended::

    leader     L0 ........ L1 ............. L2
    follower   F0 . F1 . F2 . F3 . F4 ... (reached L1 + tolerance?)
                                            yes -> decide L1: nearest F within
                                            [L1 - tol, L1 + tol], ties -> earlier

Because timestamps strictly increase, a follower value at exactly
``leader + tolerance`` proves no better candidate can still arrive, so the
outcome does not depend on how the readers interleave, as long as the
declared ``max_pending`` holds what has to wait. Nothing ever blocks: an
undecided leader stays retained and the operator returns.

Retention per input is bounded by ``max_pending``. Values no pending or
future leader can match are released first (evicted); if an input still
retains more, the run fails naming the operator, the input and the inputs it
waits for, instead of silently forcing a decision. A filtered arrival
carries absence and no timestamp, so it never advances a timeline; the
input's end (EOF) resolves what waits for it.
"""

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Deque, List, Literal, Mapping, Optional, Sequence

from pydantic import Field
from roboflow_workflows.execution_engine.v2.data import (
    AXIS_KIND_SAMPLE,
    Axis,
    EntryLayout,
    EntryMetadata,
    Timestamp,
)
from roboflow_workflows.execution_engine.v2.errors import (
    OperatorError,
    OperatorInputError,
    WorkflowCompileError,
)
from roboflow_workflows.execution_engine.v2.execution.entries import Entry
from roboflow_workflows.execution_engine.v2.operators.contract import (
    Arrival,
    Operator,
    OperatorInput,
    OperatorParams,
    OperatorPort,
    OperatorPulse,
    TerminationReason,
    operator_step_path,
)

BATCH_PORT = "samples"
"""The one port of ``layout='batch'``: members in declared input order."""


class AlignParams(OperatorParams):
    """Parameters of ``v2/align@v1``."""

    clock: str = Field(
        description="Clock id every compared timestamp must be on.",
        examples=["demo-media"],
    )
    coverage: Literal["media", "capture"] = Field(
        default="media",
        description="Which coverage of each value's temporal context is compared.",
    )
    leader: Optional[str] = Field(
        default=None,
        description="Input whose samples are paired; the first input when omitted.",
    )
    tolerance_ms: int = Field(
        default=0,
        ge=0,
        strict=True,
        description="Largest accepted distance between leader and follower, in ms.",
    )
    missing: Literal["drop", "partial"] = Field(
        default="drop",
        description="Leader without a match on some follower: drop it, or emit "
        "with that follower absent.",
    )
    consume: Literal["reuse", "once"] = Field(
        default="reuse",
        description="Whether one follower value may pair with several leaders.",
    )
    max_pending: int = Field(
        default=32,
        ge=1,
        strict=True,
        description="Values one input may retain while waiting; exceeding it fails.",
    )
    layout: Literal["fields", "batch"] = Field(
        default="fields",
        description="One port per input, or one 'samples' port with a stationary "
        "member axis in declared input order.",
    )


@dataclass(frozen=True, eq=False)
class _Sample:
    """One timestamped value; compared by identity, never by payload."""

    time: Fraction
    arrival: Arrival


class _Timeline:
    """Retained samples of one input and how far that input has advanced."""

    def __init__(self) -> None:
        self.samples: Deque[_Sample] = deque()
        self.reached: Optional[Fraction] = None
        self.ended = False

    def has_reached(self, time: Fraction) -> bool:
        return self.ended or (self.reached is not None and self.reached >= time)


class Align(Operator):
    """Pair each leader sample with the nearest follower samples in tolerance.

    Emits one pulse per decided leader, without adding a time axis. With
    ``layout='fields'`` every input keeps its name, payload and context;
    with ``layout='batch'`` the members form one stationary ``samples`` axis
    in declared input order, whose root carries no sample or temporal
    context. A missing member stays absent (``fields``) or filtered at its
    position (``batch``).
    """

    type = "v2/align@v1"
    Params = AlignParams
    input_roles = ("input",)

    @classmethod
    def plan_ports(
        cls, name: str, params: AlignParams, inputs: Sequence[OperatorInput]
    ) -> Mapping[str, OperatorPort]:
        """Plan one port per input, or one ``samples`` port for ``batch``.

        Args:
            name: Declared operator name.
            params: Validated parameters.
            inputs: The ``input`` role inputs.

        Returns:
            Port name to planned port.

        Raises:
            WorkflowCompileError: With fewer than two inputs, an unknown
                leader or, for ``batch``, inputs of different kinds.
            OperatorInputError: For a grouped input; ``field_path`` is
                ``("input", name)``.
        """

        def fail(message: str) -> WorkflowCompileError:
            return WorkflowCompileError(
                f"$operators.{name} ({cls.type}): {message}",
                step_path=operator_step_path(name),
            )

        names = [item.name for item in inputs]
        if len(inputs) < 2:
            raise fail(f"alignment needs at least two inputs, got {names}")
        for item in inputs:
            if item.layout.depth:
                raise OperatorInputError(
                    f"$operators.{name} ({cls.type}) input {item.name!r} has axes "
                    f"{list(item.layout.axis_ids)}; this version aligns ungrouped "
                    "values only",
                    step_path=operator_step_path(name),
                    role=item.role,
                    input=item.name,
                    axis_id=item.layout.axis_ids[0],
                )
        if params.leader is not None and params.leader not in names:
            raise fail(f"leader {params.leader!r} is not one of the inputs {names}")

        if params.layout == "fields":
            ports = {item.name: OperatorPort(*item.kinds) for item in inputs}
            return ports

        kind_sets = {
            item.name: sorted(kind.name for kind in item.kinds) for item in inputs
        }
        if len({tuple(kinds) for kinds in kind_sets.values()}) > 1:
            raise fail(
                f"layout 'batch' puts every input on one axis and needs the same "
                f"kinds for all of them, got {kind_sets}"
            )
        ports = {BATCH_PORT: OperatorPort(*inputs[0].kinds, layout=_batch_layout(name))}

        return ports

    def __init__(
        self, *, name: str, params: AlignParams, inputs: Sequence[OperatorInput]
    ):
        super().__init__(name=name, params=params, inputs=inputs)
        self._names = [item.name for item in self.inputs]
        self._leader = params.leader if params.leader is not None else self._names[0]
        self._followers = [item for item in self._names if item != self._leader]
        self._tolerance = Fraction(params.tolerance_ms, 1000)
        self._timelines = {item: _Timeline() for item in self._names}
        self._last_leader: Optional[Fraction] = None
        self._layout = _batch_layout(name)

    def push(self, arrivals: Sequence[Arrival]) -> List[OperatorPulse]:
        """Ingest the inputs of one upstream pulse together, then decide.

        Args:
            arrivals: Arrivals of one upstream pulse.

        Returns:
            One pulse per leader decided now.

        Raises:
            OperatorError: When a present value has no usable timestamp, or
                an input retains more than ``max_pending`` values.
        """
        for arrival in arrivals:
            self._ingest(arrival)
        emitted = self._decide()
        self._evict()
        self._check_bounds()

        return emitted

    def end_input(self, name: str) -> List[OperatorPulse]:
        """Treat input ``name`` as advanced to infinity and decide again.

        Args:
            name: Input whose domain terminated.

        Returns:
            Pulses of leaders decidable now.
        """
        self._timelines[name].ended = True
        emitted = self._decide()
        self._evict()

        return emitted

    def finish(self, reason: TerminationReason) -> List[OperatorPulse]:
        """Decide every retained leader with what is retained, then release.

        On ``stop`` the inputs are treated as ended, so ``missing`` decides
        leaders that still wait for a follower.

        Args:
            reason: ``eof`` or ``stop``.

        Returns:
            Pulses of the remaining leaders.
        """
        for timeline in self._timelines.values():
            timeline.ended = True
        emitted = self._decide()
        self._evict()
        self._release()

        return emitted

    def close(self) -> None:
        """Release every retained value."""
        self._release()

    def _ingest(self, arrival: Arrival) -> None:
        if arrival.entry.is_effectively_filtered():
            self.counters.filtered += 1
            return

        time = self._time_of(arrival)
        timeline = self._timelines[arrival.input]
        if timeline.reached is not None and time <= timeline.reached:
            self.counters.late += 1
            return

        timeline.reached = time
        timeline.samples.append(_Sample(time=time, arrival=arrival))

    def _time_of(self, arrival: Arrival) -> Fraction:
        """Exact seconds of the arrival's coverage on the declared clock."""
        coverage_name = f"{self.params.coverage}_coverage"
        context = arrival.entry.metadata.temporal_at(())
        coverage = getattr(context, coverage_name) if context is not None else None
        origin = f"value of pulse {arrival.pulse.source}#{arrival.pulse.sequence}"
        if coverage is None:
            raise OperatorError(
                f"{origin} has no {coverage_name}; alignment needs a Timestamp on "
                f"clock {self.params.clock!r}",
                operator=self.name,
                input=arrival.input,
            )
        if not isinstance(coverage, Timestamp):
            raise OperatorError(
                f"{origin} has a {type(coverage).__name__} {coverage_name}; "
                "alignment compares Timestamp points",
                operator=self.name,
                input=arrival.input,
            )
        if coverage.clock_id != self.params.clock:
            raise OperatorError(
                f"{origin} is on clock {coverage.clock_id!r}, not the declared clock "
                f"{self.params.clock!r}; incompatible clocks are never compared",
                operator=self.name,
                input=arrival.input,
            )

        return coverage.seconds

    def _decide(self) -> List[OperatorPulse]:
        emitted = []
        leaders = self._timelines[self._leader].samples
        while leaders:
            leader = leaders[0]
            horizon = leader.time + self._tolerance
            if not all(
                self._timelines[name].has_reached(horizon) for name in self._followers
            ):
                break

            leaders.popleft()
            self._last_leader = leader.time
            matches = {
                name: self._nearest(name, leader.time) for name in self._followers
            }
            pulse = self._pair(leader, matches)
            if pulse is not None:
                emitted.append(pulse)

        return emitted

    def _nearest(self, name: str, time: Fraction) -> Optional[_Sample]:
        """The follower value closest to ``time`` within tolerance; earlier on ties."""
        candidates = [
            sample
            for sample in self._timelines[name].samples
            if abs(sample.time - time) <= self._tolerance
        ]
        nearest = min(
            candidates,
            key=lambda sample: (abs(sample.time - time), sample.time),
            default=None,
        )

        return nearest

    def _pair(
        self, leader: _Sample, matches: Mapping[str, Optional[_Sample]]
    ) -> Optional[OperatorPulse]:
        if self.params.missing == "drop" and any(
            sample is None for sample in matches.values()
        ):
            self.counters.dropped += 1
            return None

        if self.params.consume == "once":
            for name, sample in matches.items():
                if sample is not None:
                    self._timelines[name].samples.remove(sample)

        members = {**matches, self._leader: leader}
        present = {
            name: members[name].arrival
            for name in self._names
            if members[name] is not None
        }
        if self.params.layout == "fields":
            ports = {name: arrival.entry for name, arrival in present.items()}
        else:
            ports = {BATCH_PORT: self._batch_entry(present)}
        causes = dict.fromkeys(arrival.pulse for arrival in present.values())
        pulse = OperatorPulse(ports=ports, causes=tuple(causes))

        return pulse

    def _batch_entry(self, present: Mapping[str, Arrival]) -> Entry:
        """Members at their declared positions; a missing one stays filtered."""
        positions = [(position,) for position in range(len(self._names))]
        values, sample, temporal = {}, {(): None}, {(): None}
        for index, name in zip(positions, self._names):
            if name not in present:
                continue
            member = present[name].entry
            values[index] = member.values[()]
            sample[index] = member.metadata.sample_at(())
            temporal[index] = member.metadata.temporal_at(())
        entry = Entry(
            layout=self._layout,
            metadata=EntryMetadata(sample=sample, temporal=temporal),
            children={(): tuple(positions)},
            values=values,
            filtered=frozenset(index for index in positions if index not in values),
        )

        return entry

    def _evict(self) -> None:
        """Release follower values no pending or future leader can match."""
        leader = self._timelines[self._leader]
        for name in self._followers:
            samples = self._timelines[name].samples
            before = len(samples)
            if leader.samples:
                oldest = leader.samples[0].time - self._tolerance
                while samples and samples[0].time < oldest:
                    samples.popleft()
            elif leader.ended:
                samples.clear()
            elif self._last_leader is not None:
                # Future leaders are later than the last one decided.
                newest_useless = self._last_leader - self._tolerance
                while samples and samples[0].time <= newest_useless:
                    samples.popleft()
            self.counters.evicted += before - len(samples)

    def _check_bounds(self) -> None:
        limit = self.params.max_pending
        for name, timeline in self._timelines.items():
            retained = len(timeline.samples)
            self.counters.peak_retained = max(self.counters.peak_retained, retained)
            if retained <= limit:
                continue

            raise OperatorError(
                f"retains {retained} values, above max_pending={limit}. "
                f"{self._waiting_reason(name)} Raise max_pending, or make the "
                "slower inputs emit more often",
                operator=self.name,
                input=name,
            )

    def _waiting_reason(self, name: str) -> str:
        leaders = self._timelines[self._leader].samples
        if name != self._leader:
            return (
                f"Its values are ahead of leader {self._leader!r} and wait for "
                "leaders that have not arrived."
            )

        horizon = leaders[0].time + self._tolerance
        waiting_for = [
            follower
            for follower in self._followers
            if not self._timelines[follower].has_reached(horizon)
        ]

        return (
            f"The oldest leader at {float(leaders[0].time)}s waits for {waiting_for} "
            f"to reach {float(horizon)}s."
        )

    def _release(self) -> None:
        for timeline in self._timelines.values():
            timeline.samples.clear()


def _batch_layout(operator_name: str) -> EntryLayout:
    axis = Axis(f"operators.{operator_name}:samples", AXIS_KIND_SAMPLE, stationary=True)
    layout = EntryLayout((axis,))

    return layout
