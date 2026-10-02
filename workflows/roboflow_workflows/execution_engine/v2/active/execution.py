"""One pulse executed as one ``RunState`` over the session's instances.

A pulse is one emission of one domain: a source, or an operator. It runs that
domain's route (``plan.route(domain)``: the steps whose domain it is plus the
static steps, in plan order) with a fresh ``RunState`` whose only dynamic
entries are the emitted ports::

    run = begin_pulse(session, pulse=PulseKey(run_id, "camera", 3),
                      emission=emission, inputs=static_entries)
    execute_pulse(run)                                # plan.route("camera")
    result = build_group_result(run, group=group)     # a group anchored on "camera"
    arrivals = operator_arrivals(run, plan.operator("pair"))
    abandon_pulse(run)                                # drop the pulse's entries

    emitted = operator.push(arrivals)                 # zero or more OperatorPulse
    run = begin_operator_pulse(session, pulse=PulseKey(run_id, "pair", 0),
                               emission=emitted[0], inputs=static_entries)

Every present source port becomes an entry with the port's compiled layout,
carrying the source's sample context and the pulse's temporal context at index
``()`` (an ``InputValue`` payload overlays its own indexed metadata verbatim).
An operator pulse carries the operator's entries as they are, with the
upstream pulses that caused it in ``RunState.causes``. Every declared port an
emission omits is a terminally absent entry, filtered at ``()``: consumers
skip it and group fields selecting it come out ``filtered``.

These primitives are what the active runtime uses per pulse and what the
same-session two-pulse experiment drives by hand. They do not make pulses
concurrent: a block instance is shared by every pulse of its session. A
pipelined run passes its ``coordination``; each pulse then carries the ticket
``(domain, sequence)`` that orders it at every stage it visits.
"""

import time
from fractions import Fraction
from typing import Dict, List, Mapping, Optional

from roboflow_workflows.execution_engine.v2.data import (
    EntryLayout,
    EntryMetadata,
    InputValue,
    SampleContext,
    TemporalContext,
    Timestamp,
    validate_entry,
)
from roboflow_workflows.execution_engine.v2.errors import ContractError
from roboflow_workflows.execution_engine.v2.execution.entries import (
    Entry,
    entry_from_tree,
)
from roboflow_workflows.execution_engine.v2.execution.inputs import (
    check_payload,
    check_shared_axes,
    kinds_named,
)
from roboflow_workflows.execution_engine.v2.execution.outputs import (
    GroupResult,
    build_group_result,
)
from roboflow_workflows.execution_engine.v2.execution.steps import (
    RunState,
    execute_step,
)
from roboflow_workflows.execution_engine.v2.operators import Arrival, OperatorPulse
from roboflow_workflows.execution_engine.v2.pipelining.stages import (
    SERIAL,
    Coordination,
    Ticket,
)
from roboflow_workflows.execution_engine.v2.plan import (
    CompiledWorkflow,
    ExecutionSession,
    PlannedOperator,
    PlannedOutputGroup,
    PlannedSource,
    PulseKey,
    SourcePort,
)
from roboflow_workflows.execution_engine.v2.sources import Emission

ENGINE_CLOCK_ID = "engine.monotonic"
"""Clock of the observation timestamps the runtime stamps itself."""


def engine_observation() -> Timestamp:
    """Return the current time on the engine's monotonic clock.

    Returns:
        A nanosecond timestamp on ``ENGINE_CLOCK_ID``. It is never comparable
        to a media clock and never derived from media ticks.
    """
    stamp = Timestamp(
        ticks=time.monotonic_ns(),
        time_base=Fraction(1, 10**9),
        clock_id=ENGINE_CLOCK_ID,
    )

    return stamp


def begin_pulse(
    session: ExecutionSession,
    *,
    pulse: PulseKey,
    emission: Emission,
    inputs: Mapping[str, Entry],
    observed: Optional[Timestamp] = None,
    coordination: Coordination = SERIAL,
) -> RunState:
    """Create the run state of one pulse from an emission.

    Args:
        session: Session whose block instances run the pulse.
        pulse: Identity of the pulse.
        emission: What the source emitted for this pulse.
        inputs: Static input entries prepared once per active run.
        observed: When the runtime received the emission; the engine clock
            now when omitted.
        coordination: The run's coordination; ``SERIAL`` gates nothing.

    Returns:
        A fresh run state holding the static inputs and the emitted ports;
        its ``run_id`` is ``pulse.run_id``.

    Raises:
        ContractError: When the emission names an undeclared port, a payload
            violates its port's kinds or layout, or metadata addresses a
            position the payload does not have.
    """
    ports = port_entries(
        session.plan,
        pulse.source,
        emission=emission,
        observed=observed if observed is not None else engine_observation(),
    )
    run = RunState(
        session=session,
        run_id=pulse.run_id,
        inputs=dict(inputs),
        pulse=pulse,
        ports=ports,
        coordination=coordination,
        ticket=pulse_ticket(pulse),
    )
    run.record(
        "pulse_started",
        run_id=run.run_id,
        source=pulse.source,
        pulse=pulse.sequence,
        ports=sorted(emission.data),
    )

    return run


def pulse_ticket(pulse: PulseKey) -> Ticket:
    """Return the pulse's place in its domain's order at every stage.

    Args:
        pulse: Identity of the pulse.

    Returns:
        ``Ticket(domain=pulse.source, ordinal=pulse.sequence)``; sequences
        of a domain have no gaps, so the ticket order has none either.
    """
    ticket = Ticket(domain=pulse.source, ordinal=pulse.sequence)

    return ticket


def port_entries(
    plan: CompiledWorkflow, source: str, *, emission: Emission, observed: Timestamp
) -> Dict[SourcePort, Entry]:
    """Turn one emission into one entry per declared port of the source.

    Args:
        plan: Compiled plan declaring the source and owning the kinds.
        source: Declared name of the emitting source.
        emission: The emission; ``data`` maps present ports to payloads.
            A grouped port takes a ``Batch`` tree with full indices under the
            port's compiled layout. A payload may be an ``InputValue`` whose
            indexed metadata overlays the pulse context verbatim, explicit
            ``None`` included.
        observed: Observation time of the pulse on the engine clock.

    Returns:
        Present ports as entries with the pulse's context, omitted ports as
        terminally absent entries.

    Raises:
        ContractError: On an undeclared port name, a payload rejected by the
            port's kinds, a tree not matching the port's layout or metadata
            addressing a missing position.
    """
    planned = plan.source(source)
    unknown = sorted(set(emission.data) - set(planned.outputs))
    if unknown:
        raise ContractError(
            f"Source {planned.name!r} emitted undeclared ports {unknown}; declared: "
            f"{sorted(planned.outputs)}"
        )

    context = _pulse_metadata(planned, emission=emission, observed=observed)
    entries: Dict[SourcePort, Entry] = {}
    for name, port in planned.outputs.items():
        key = SourcePort(source=planned.name, output=name)
        if name not in emission.data:
            entries[key] = absent_entry(port.layout)
            continue

        supplied = emission.data[name]
        if isinstance(supplied, InputValue):
            data, metadata = supplied.data, _overlay(context, supplied.metadata)
        else:
            data, metadata = supplied, context
        location = f"Source port {key.describe()}"
        try:
            validate_entry(data, layout=port.layout, metadata=metadata)
        except ContractError as error:
            raise ContractError(f"{location}: {error}") from error
        entry = entry_from_tree(data, layout=port.layout, metadata=metadata)
        kinds = kinds_named(plan, port.kinds)
        for index in sorted(entry.values):
            where = f"{location} at index {list(index)}" if index else location
            check_payload(entry.values[index], kinds=kinds, location=where)
        entries[key] = entry

    # Ports declared over the same source-local axes assert native
    # correspondence; the emitted groups must agree along those axes.
    present = {
        key.output: entry for key, entry in entries.items() if not entry.is_filtered(())
    }
    check_shared_axes(present, what=f"Source $sources.{source} ports")

    return entries


def begin_operator_pulse(
    session: ExecutionSession,
    *,
    pulse: PulseKey,
    emission: OperatorPulse,
    inputs: Mapping[str, Entry],
    coordination: Coordination = SERIAL,
) -> RunState:
    """Create the run state of one pulse an operator emitted.

    Args:
        session: Session whose block instances run the pulse.
        pulse: Identity of the pulse; ``pulse.source`` names the operator.
        emission: What the operator emitted.
        inputs: Static input entries prepared once per active run.
        coordination: The run's coordination; ``SERIAL`` gates nothing.

    Returns:
        A fresh run state holding the static inputs and the operator's
        ports, with ``causes`` from the emission; nothing of an upstream
        pulse's state is shared.

    Raises:
        ContractError: When the emission names an undeclared port or an
            entry's layout differs from the port's planned layout.
    """
    ports = operator_port_entries(session.plan, pulse.source, emission=emission)
    run = RunState(
        session=session,
        run_id=pulse.run_id,
        inputs=dict(inputs),
        pulse=pulse,
        ports=ports,
        causes=emission.causes,
        coordination=coordination,
        ticket=pulse_ticket(pulse),
    )
    run.record(
        "pulse_started",
        run_id=run.run_id,
        source=pulse.source,
        pulse=pulse.sequence,
        ports=sorted(emission.ports),
        causes=[f"{cause.source}#{cause.sequence}" for cause in emission.causes],
    )

    return run


def operator_port_entries(
    plan: CompiledWorkflow, operator: str, *, emission: OperatorPulse
) -> Dict[SourcePort, Entry]:
    """Check an operator's emitted entries against its planned ports.

    Args:
        plan: Compiled plan declaring the operator.
        operator: Declared operator name.
        emission: The operator's emission.

    Returns:
        Present ports as emitted, omitted ports as terminally absent entries.

    Raises:
        ContractError: On an undeclared port or a layout mismatch.
    """
    planned = plan.operator(operator)
    unknown = sorted(set(emission.ports) - set(planned.outputs))
    if unknown:
        raise ContractError(
            f"Operator $operators.{operator} emitted undeclared ports {unknown}; "
            f"planned: {sorted(planned.outputs)}"
        )

    entries: Dict[SourcePort, Entry] = {}
    for name, port in planned.outputs.items():
        key = planned.port(name)
        if name not in emission.ports:
            entries[key] = absent_entry(port.layout)
            continue

        entry = emission.ports[name]
        if entry.layout != port.layout:
            raise ContractError(
                f"Operator port {key.describe()} emitted axes "
                f"{list(entry.layout.axis_ids)}, planned {list(port.layout.axis_ids)}"
            )
        entries[key] = entry

    return entries


def operator_arrivals(run: RunState, operator: PlannedOperator) -> List[Arrival]:
    """Read the operator's inputs of the pulse's domain, after its route ran.

    All inputs of one domain arrive together, filtered ones included, so
    the operator sees the pulse atomically.

    Args:
        run: State of a pulse after its route ran.
        operator: An operator consuming the pulse's domain.

    Returns:
        One arrival per input of that domain, in declaration order.
    """
    arrivals = [
        Arrival(input=item.name, entry=run.entry_for(item.source), pulse=run.pulse)
        for item in operator.inputs_from(run.pulse.source)
    ]

    return arrivals


def absent_entry(layout: EntryLayout) -> Entry:
    """Return the entry of a port the emission omitted: filtered as a whole.

    Args:
        layout: Compiled layout of the port.

    Returns:
        An entry without values whose root is filtered.
    """
    entry = Entry(
        layout=layout,
        metadata=EntryMetadata(),
        children={},
        values={},
        filtered=frozenset({()}),
    )

    return entry


def _pulse_metadata(
    planned: PlannedSource, *, emission: Emission, observed: Timestamp
) -> EntryMetadata:
    """Context every present port of the pulse carries at index ``()``."""
    sample = SampleContext(
        source_id=planned.name,
        source_type=planned.spec.type,
        source_metadata=emission.source_metadata,
    )
    temporal = TemporalContext(
        observed_coverage=observed,
        media_coverage=emission.media,
        capture_coverage=emission.capture,
    )
    metadata = EntryMetadata(sample={(): sample}, temporal={(): temporal})

    return metadata


def _overlay(context: EntryMetadata, supplied: EntryMetadata) -> EntryMetadata:
    """The pulse context with an ``InputValue``'s maps written over it, key by key."""
    overlaid = EntryMetadata(
        sample={**context.sample, **supplied.sample},
        temporal={**context.temporal, **supplied.temporal},
    )

    return overlaid


def execute_pulse(run: RunState) -> None:
    """Run every step of the pulse's domain route, in plan order.

    Args:
        run: State created by ``begin_pulse``.

    Raises:
        StepExecutionError: When a step fails.
    """
    for step in run.plan.route(run.pulse.source):
        execute_step(run, step)


def group_result(
    run: RunState, group: PlannedOutputGroup, *, filtered: bool = False
) -> GroupResult:
    """Build the result of one group anchored on the pulse's domain.

    Args:
        run: State of the pulse after its route ran.
        group: A group returned by ``plan.groups_of(run.pulse.source)``.
        filtered: Deliver every field as filtered without reading entries;
            used for an explicit ``Emission({})``.

    Returns:
        The group's result for this pulse.
    """
    result = build_group_result(run, group=group, filtered=filtered)

    return result


def abandon_pulse(run: RunState) -> None:
    """Drop every entry of a finished or cancelled pulse.

    The session's block instances and any result already built are
    untouched; only the pulse's own maps are cleared, so its payloads and
    decisions are not retained by the run state.

    Args:
        run: State of the pulse.
    """
    run.record("pulse_abandoned", source=run.pulse.source, pulse=run.pulse.sequence)
    run.ports.clear()
    run.outputs.clear()
    run.decisions.clear()
    run.constants.clear()
    run.child_inputs.clear()
    run.child_outputs.clear()
