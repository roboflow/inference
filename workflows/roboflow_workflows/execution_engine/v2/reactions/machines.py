"""State machine records and transitions of one reaction runtime.

Each machine instance (one per source for ``scope="source"``, one for
``scope="global"``) is a single managed-state record under a reserved key::

    {"state": "reviewing", "version": 3}

A record is initialized lazily at the initial state, version 0, on the first
inspection or transition of that machine instance.

Every transition is one compare-and-set on that record. ``version`` grows by
one per applied transition, so a handler decision stamped with an older
version is ``stale`` even if the state name is the same again (A -> B -> A)::

    event / setter        read record ──► pick transition ──► CAS(old -> new)
                                │ state not in "from"      │ miss: re-read (fixed)
                                ▼                          │       stale (stamped)
                             ignored                       ▼
                                                    emit machine event
                                                    (no lock held)

The controller owns the records, not handler scheduling: applied transitions
with an ``emit`` are handed to the ``emit`` callback of the enclosing reaction
runtime, which builds the event cause and dispatches it. Backend failures are
never retried; an unknown CAS outcome propagates without an emitted event.
No transaction or exactly-once delivery is promised.
"""

import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
)

from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    EventEmissionError,
    StepPath,
)
from roboflow_workflows.execution_engine.v2.state import ManagedState
from roboflow_workflows.execution_engine.v2.state.codec import (
    INT64_MAX,
    decode_value,
    encode_value,
    machine_storage_key,
    validate_name,
)
from roboflow_workflows.execution_engine.v2.state.errors import (
    StateError,
    StateScopeError,
)

if TYPE_CHECKING:
    from roboflow_workflows.execution_engine.v2.reactions.dispatch import EventCause
    from roboflow_workflows.execution_engine.v2.reactions.plan import (
        EventOrigin,
        PlannedMachine,
        PlannedTransition,
        ReactionPlan,
    )

__all__ = [
    "MachineRuntime",
    "MachineStamp",
    "MachineStateError",
    "TransitionCounters",
    "TransitionResult",
]

TransitionOutcome = Literal["applied", "ignored", "stale"]

_MAX_CAS_ATTEMPTS = 64


class MachineStateError(StateError):
    """A machine record is corrupt, missing or cannot be updated.

    Raised for a stored record that is not ``{"state": <declared state>,
    "version": <int>}``, a record deleted after initialization, a version
    that would leave the signed 64-bit range and a transition that lost the
    compare-and-set race too many times in a row.
    """


@dataclass(frozen=True)
class MachineStamp:
    """Identity of one applied transition, carried by the event it emits.

    Args:
        machine: Path of the machine.
        source_id: Source of the machine instance; ``None`` for a global one.
        transition: Applied transition name.
        from_state: State before the transition.
        to_state: State after the transition.
        version: Record version after the transition.
    """

    machine: StepPath
    source_id: Optional[str]
    transition: str
    from_state: str
    to_state: str
    version: int


@dataclass(frozen=True)
class TransitionResult:
    """Outcome of one transition attempt.

    Args:
        outcome: ``applied``; ``ignored`` when the current state is not a
            ``from`` state of the transition; ``stale`` when a stamped
            decision no longer matches the record.
        state: State after an applied transition, else the state found.
        stamp: Stamp of the applied transition; ``None`` otherwise.
    """

    outcome: TransitionOutcome
    state: str
    stamp: Optional[MachineStamp] = None


@dataclass(frozen=True)
class TransitionCounters:
    """Attempt counts of one transition since the runtime started.

    Args:
        applied: Applied transitions.
        ignored: Attempts whose current state was not a ``from`` state.
        stale: Stamped decisions rejected because the record had moved on.
    """

    applied: int = 0
    ignored: int = 0
    stale: int = 0


EmitMachineEvent = Callable[..., None]
"""``emit(origin, fields, *, parent, stamp) -> None`` of the reaction runtime."""


class MachineRuntime:
    """Applies the transitions of a reaction plan to managed-state records.

    Args:
        plan: Reaction plan with ``machines``, ``transitions_for`` and
            ``setter``.
        managed_state: State service holding the records.
        emit: Callback receiving every applied transition that emits an
            event, called without any controller lock held.
    """

    def __init__(
        self,
        plan: "ReactionPlan",
        managed_state: ManagedState,
        *,
        emit: EmitMachineEvent,
    ) -> None:
        self._plan = plan
        self._state = managed_state
        self._emit = emit
        self._machines: Dict[StepPath, "PlannedMachine"] = {
            machine.path: machine for machine in plan.machines
        }
        self._counts: Dict[Tuple[StepPath, str], List[int]] = {
            (machine.path, transition.name): [0, 0, 0]
            for machine in plan.machines
            for transition in machine.transitions
        }
        self._counts_lock = threading.Lock()

    def on_event(
        self, origin: "EventOrigin", cause: "EventCause", fields: Mapping[str, Any]
    ) -> None:
        """Apply the fixed transitions triggered by one event.

        Each machine applies at most one transition per event: the one whose
        ``from`` states hold its current state. Machines are visited in the
        order their first triggered transition was declared; each applied
        transition is emitted before the next machine is visited.

        Args:
            origin: Origin of the incoming event.
            cause: Cause of the incoming event; its sample gives the source.
            fields: Incoming event fields; must hold every ``$event`` field
                the emitted events use.

        Raises:
            EventEmissionError: When a source machine is triggered by an
                event without a source.
            MachineStateError: On a corrupt record or CAS contention.
            StateBackendError: When the backend fails; nothing is emitted.
        """
        grouped: Dict[StepPath, List["PlannedTransition"]] = {}
        for transition in self._plan.transitions_for(origin):
            grouped.setdefault(transition.machine, []).append(transition)

        for path, candidates in grouped.items():
            machine = self._machines[path]
            source_id = self._event_source(machine, cause=cause, origin=origin)
            _, applied = self._apply_current(
                machine, candidates=tuple(candidates), source_id=source_id, target=None
            )
            if applied is None:
                continue

            self._emit_applied(applied, cause=cause, event_fields=fields)

    def set_state(
        self,
        *,
        handler: StepPath,
        cause: "EventCause",
        machine_ref: str,
        transition: str,
        next_state: str,
    ) -> TransitionResult:
        """Apply a handler-selected transition.

        The record is the one of the handler cause's source (none for a
        global machine). The nearest stamp in the cause chain of that machine
        and source makes the decision stamped: it applies only while the
        record still equals the stamped state and version, else the result is
        ``stale``. Stamps of other sources are skipped.

        Args:
            handler: Path of the handler run calling the setter.
            cause: Cause of that handler run.
            machine_ref: Machine name relative to the handler's scope.
            transition: Transition name.
            next_state: Requested target state.

        Returns:
            The attempt's outcome.

        Raises:
            ContractError: When the handler may not set this transition or
                ``next_state`` is not one of its targets.
            StateScopeError: When a source machine has no source in context.
            MachineStateError: On a corrupt record or CAS contention.
            StateBackendError: When the backend fails; nothing is emitted.
        """
        planned = self._plan.setter(handler, machine_ref, transition)
        if planned.handler != handler:
            raise ContractError(
                f"Handler {'/'.join(handler)} may not set transition "
                f"{planned.name!r} of machine {machine_ref!r}"
            )
        if next_state not in planned.targets:
            raise ContractError(
                f"State {next_state!r} is not a target of transition "
                f"{planned.name!r}; allowed: {list(planned.targets)}"
            )

        machine = self._machines[planned.machine]
        # The handler cause picks the record; a stamp only counts when it was
        # applied to that same record.
        source_id = self._cause_source(machine, cause=cause)
        stamp = _nearest_stamp(cause, machine=machine.path, source_id=source_id)
        if stamp is not None:
            result, applied = self._apply_stamped(
                machine, transition=planned, target=next_state, stamp=stamp
            )
        else:
            result, applied = self._apply_current(
                machine, candidates=(planned,), source_id=source_id, target=next_state
            )
        if applied is not None:
            self._emit_applied(applied, cause=cause, event_fields={})

        return result

    def current(
        self, machine: StepPath, *, source_id: Optional[str] = None
    ) -> Tuple[str, int]:
        """Read the state and version of one machine instance.

        Args:
            machine: Machine path.
            source_id: Source of a source machine; must be ``None`` for a
                global one.

        Returns:
            ``(state, version)``; the initial state with version 0 when no
            transition has been applied.

        Raises:
            ContractError: On an unknown machine.
            StateScopeError: On a missing or unexpected ``source_id``.
            StateValueError: When ``source_id`` is not a non-empty ``str``;
                no record is created.
        """
        planned = self._machines.get(tuple(machine))
        if planned is None:
            raise ContractError(f"Unknown state machine {'/'.join(machine)!r}")
        if planned.scope == "source" and source_id is None:
            raise StateScopeError(
                f"State machine {'/'.join(machine)!r} is per source; pass source_id"
            )
        if planned.scope == "global" and source_id is not None:
            raise StateScopeError(
                f"State machine {'/'.join(machine)!r} is global; source_id must be None"
            )

        key = self._ensure_record(planned, source_id=source_id)
        state, version, _ = self._read(planned, key=key)

        return state, version

    def counters(self) -> Mapping[str, TransitionCounters]:
        """Snapshot the per-transition counters.

        Returns:
            Read-only mapping ``"<machine path>.<transition>"`` → counters;
            later attempts do not change a returned snapshot.
        """
        with self._counts_lock:
            snapshot = {
                f"{'/'.join(machine)}.{name}": TransitionCounters(*counts)
                for (machine, name), counts in self._counts.items()
            }

        return MappingProxyType(snapshot)

    def _apply_current(
        self,
        machine: "PlannedMachine",
        *,
        candidates: Tuple["PlannedTransition", ...],
        source_id: Optional[str],
        target: Optional[str],
    ) -> Tuple[TransitionResult, Optional[MachineStamp]]:
        # Current-state mode: pick the transition matching the state read now;
        # a lost CAS re-reads and picks again.
        key = self._ensure_record(machine, source_id=source_id)
        for _ in range(_MAX_CAS_ATTEMPTS):
            state, version, raw = self._read(machine, key=key)
            chosen = next((t for t in candidates if state in t.sources), None)
            if chosen is None:
                self._count(candidates, outcome="ignored")
                return TransitionResult("ignored", state=state), None

            to_state = target if target is not None else chosen.targets[0]
            stamp = self._swap(
                machine,
                key=key,
                raw=raw,
                transition=chosen,
                source_id=source_id,
                from_state=state,
                to_state=to_state,
                version=version,
            )
            if stamp is not None:
                self._count((chosen,), outcome="applied")
                others = tuple(t for t in candidates if t is not chosen)
                self._count(others, outcome="ignored")
                return TransitionResult("applied", state=to_state, stamp=stamp), stamp

        raise MachineStateError(
            f"State machine {'/'.join(machine.path)!r} lost {_MAX_CAS_ATTEMPTS} "
            "compare-and-set attempts in a row"
        )

    def _apply_stamped(
        self,
        machine: "PlannedMachine",
        *,
        transition: "PlannedTransition",
        target: str,
        stamp: MachineStamp,
    ) -> Tuple[TransitionResult, Optional[MachineStamp]]:
        # Stamped mode: only the exact stamped record may change; no retry.
        key = self._ensure_record(machine, source_id=stamp.source_id)
        state, version, raw = self._read(machine, key=key)
        if (state, version) != (stamp.to_state, stamp.version):
            self._count((transition,), outcome="stale")
            return TransitionResult("stale", state=state), None
        if state not in transition.sources:
            self._count((transition,), outcome="ignored")
            return TransitionResult("ignored", state=state), None

        applied = self._swap(
            machine,
            key=key,
            raw=raw,
            transition=transition,
            source_id=stamp.source_id,
            from_state=state,
            to_state=target,
            version=version,
        )
        if applied is None:
            self._count((transition,), outcome="stale")
            current_state, _, _ = self._read(machine, key=key)
            return TransitionResult("stale", state=current_state), None

        self._count((transition,), outcome="applied")

        return TransitionResult("applied", state=target, stamp=applied), applied

    def _swap(
        self,
        machine: "PlannedMachine",
        *,
        key: str,
        raw: str,
        transition: "PlannedTransition",
        source_id: Optional[str],
        from_state: str,
        to_state: str,
        version: int,
    ) -> Optional[MachineStamp]:
        if version >= INT64_MAX:
            raise MachineStateError(
                f"State machine {'/'.join(machine.path)!r} version {version} "
                "cannot grow past the signed 64-bit range"
            )

        new_version = version + 1
        new_raw = encode_value({"state": to_state, "version": new_version})
        # An unknown outcome raises here and is never retried.
        if not self._state.backend.compare_and_set(key, raw, new_raw):
            return None

        stamp = MachineStamp(
            machine=machine.path,
            source_id=source_id,
            transition=transition.name,
            from_state=from_state,
            to_state=to_state,
            version=new_version,
        )

        return stamp

    def _read(self, machine: "PlannedMachine", *, key: str) -> Tuple[str, int, str]:
        raw = self._state.backend.get(key)
        if raw is None:
            raise MachineStateError(
                f"State machine {'/'.join(machine.path)!r} record is missing; it "
                "was deleted after initialization"
            )

        try:
            record = decode_value(raw)
        except ValueError:
            record = None
        if (
            type(record) is not dict
            or set(record) != {"state", "version"}
            or record["state"] not in machine.states
            or type(record["version"]) is not int
            or not 0 <= record["version"] <= INT64_MAX
        ):
            raise MachineStateError(
                f"State machine {'/'.join(machine.path)!r} record is corrupt: {raw!r}"
            )

        return record["state"], record["version"], raw

    def _ensure_record(
        self, machine: "PlannedMachine", *, source_id: Optional[str]
    ) -> str:
        # Checked before the key exists, so a bad id never creates a record.
        if source_id is not None:
            validate_name(source_id, what="source id")
        key = machine_storage_key(
            self._state.namespace, source_id, "/".join(machine.path)
        )
        self._state.initialize_once(key, {"state": machine.initial, "version": 0})

        return key

    def _event_source(
        self, machine: "PlannedMachine", *, cause: "EventCause", origin: "EventOrigin"
    ) -> Optional[str]:
        if machine.scope == "global":
            return None

        source_id = _cause_source_id(cause)
        if source_id is None:
            raise EventEmissionError(
                f"Event {origin.event!r} has no source, but state machine "
                f"{'/'.join(machine.path)!r} is per source"
            )

        return source_id

    def _cause_source(
        self, machine: "PlannedMachine", *, cause: "EventCause"
    ) -> Optional[str]:
        if machine.scope == "global":
            return None

        source_id = _cause_source_id(cause)
        if source_id is None:
            raise StateScopeError(
                f"Handler run has no source, but state machine "
                f"{'/'.join(machine.path)!r} is per source"
            )

        return source_id

    def _emit_applied(
        self,
        stamp: MachineStamp,
        *,
        cause: "EventCause",
        event_fields: Mapping[str, Any],
    ) -> None:
        transition = _transition_named(self._machines[stamp.machine], stamp.transition)
        if transition.emits is None:
            return

        fields = _emitted_fields(transition, stamp=stamp, event_fields=event_fields)
        self._emit(transition.emits, fields, parent=cause, stamp=stamp)

    def _count(
        self, transitions: Iterable["PlannedTransition"], *, outcome: TransitionOutcome
    ) -> None:
        position = ("applied", "ignored", "stale").index(outcome)
        with self._counts_lock:
            for transition in transitions:
                self._counts[(transition.machine, transition.name)][position] += 1


def _nearest_stamp(
    cause: Optional["EventCause"], *, machine: StepPath, source_id: Optional[str]
) -> Optional[MachineStamp]:
    while cause is not None:
        stamp = cause.stamp
        if (
            stamp is not None
            and stamp.machine == machine
            and stamp.source_id == source_id
        ):
            return stamp
        cause = cause.parent

    return None


def _cause_source_id(cause: "EventCause") -> Optional[str]:
    sample = cause.sample
    source_id = sample.source_id if sample is not None else None

    return source_id


def _transition_named(machine: "PlannedMachine", name: str) -> "PlannedTransition":
    transition = next(t for t in machine.transitions if t.name == name)

    return transition


def _emitted_fields(
    transition: "PlannedTransition",
    *,
    stamp: MachineStamp,
    event_fields: Mapping[str, Any],
) -> Dict[str, Any]:
    transition_values = {
        "from": stamp.from_state,
        "to": stamp.to_state,
        "name": stamp.transition,
    }
    fields: Dict[str, Any] = {}
    for name, (kind, value) in transition.fields.items():
        if kind == "literal":
            fields[name] = value
        elif kind == "transition":
            fields[name] = transition_values[value]
        elif value in event_fields:
            fields[name] = event_fields[value]
        else:
            raise ContractError(
                f"Transition {transition.name!r} emits $event.{value}, but the "
                "incoming event does not carry that field"
            )

    return fields
