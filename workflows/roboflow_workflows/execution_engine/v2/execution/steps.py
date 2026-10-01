"""Sequential execution of one planned step.

    domain    every index of P known to a varying element/group binding's
              source (filtered ones included); without such bindings, every
              index known to the gating controllers at P; () at P == ().
              A position only an absent group would supply is never invoked
    gates     at i, every controller must have selected this step's target at
              i[:len(controller axes)] (conjunction; empty selection denies)
    leaves    one value per binding (arguments.resolve_leaf); default blocks
              skip unavailable or None values, accepts_empty blocks do not
    validate  each admitted logical invocation with
              BlockSpec.validate_resolved_arguments (decision 018), before
              batch packaging and before any call of the step
    call      per invocation, or once for a batch-delivering step, inside an
              ExecutionContext with the run id and the call's indices: the
              selected implementation's run(), or for execution "phases"
              run_phases over its graph with the same arguments; a phase
              failure names the phase in both modes
    ready     futures in the result, including Selected/Selection payloads,
              are resolved in one pass right after each call, in the same
              context (decision 023)
    record    one Entry per output (or the controller's decisions), keeping
              skipped and denied indices as filtered positions; a
              ``selected`` output resolves Selected/Selection to the chosen
              members' payloads, and member policies take their contexts

Skipped invocations, denied ones included, run zero times and produce no
placeholder values: their positions are filtered in every output.
"""

import copy
import dataclasses
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Mapping,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
)

from roboflow_workflows.execution_engine.v2.context import (
    ExecutionContext,
    use_execution_context,
)
from roboflow_workflows.execution_engine.v2.data import Batch, EntryMetadata, Index
from roboflow_workflows.execution_engine.v2.declaration import (
    MEMBER_POLICIES,
    SAME_PAYLOAD,
    Select,
    Selected,
    Selection,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    StepExecutionError,
    StepPath,
    format_step_path,
)
from roboflow_workflows.execution_engine.v2.execution.arguments import (
    Leaf,
    arguments_for,
    batch_values,
    has_group,
    resolve_leaf,
    skip_reason,
    validation_arguments,
)
from roboflow_workflows.execution_engine.v2.execution.entries import (
    Entry,
    common_or_none,
    scalar_entry,
)
from roboflow_workflows.execution_engine.v2.execution.inputs import (
    check_kinds,
    check_payload,
    decode_payload,
    kinds_named,
)
from roboflow_workflows.execution_engine.v2.phases import PhaseFailure, run_phases
from roboflow_workflows.execution_engine.v2.plan import (
    ChildInputPort,
    ChildOutputPort,
    CompiledWorkflow,
    Constant,
    ExecutionSession,
    Gate,
    InputPort,
    PlannedOutput,
    PlannedStep,
    PulseKey,
    Source,
    SourcePort,
)
from roboflow_workflows.execution_engine.v2.readiness import resolve_futures


@dataclass
class RunState:
    """Mutable state of one run: entries by source, decisions and trace.

    A passive run holds the workflow inputs; a pulse of an active run holds
    the static inputs and the emitted source ports of that pulse.

    Args:
        session: Session whose block instances run.
        run_id: Identity of the run; for a pulse, ``PulseKey.run_id``.
        inputs: Workflow input entries by name.
        pulse: Identity of the pulse; ``None`` for a passive run.
        ports: Entries of every declared port of the pulse's source, the
            omitted ones terminally absent; empty for a passive run.
        causes: Upstream pulses whose arrivals produced this operator
            emission, in contribution order; empty for source pulses and
            passive runs.
    """

    session: ExecutionSession
    run_id: str
    inputs: Dict[str, Entry]
    pulse: Optional[PulseKey] = None
    ports: Dict[SourcePort, Entry] = field(default_factory=dict)
    causes: Tuple[PulseKey, ...] = ()
    outputs: Dict[Tuple[StepPath, str], Entry] = field(default_factory=dict)
    decisions: Dict[StepPath, Entry] = field(default_factory=dict)
    constants: Dict[int, Tuple[Constant, Entry]] = field(default_factory=dict)
    child_inputs: Dict[ChildInputPort, Entry] = field(default_factory=dict)
    child_outputs: Dict[ChildOutputPort, Entry] = field(default_factory=dict)
    trace: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def plan(self) -> CompiledWorkflow:
        """The executed plan."""
        return self.session.plan

    def entry_for(self, source: Source) -> Entry:
        """Return the entry of a bound or output source.

        A ``Constant`` (for example a literal cast into a group) is
        materialized once per run as a private copy that every consumer of
        that constant shares, so in-place changes reach neither the plan nor
        later runs and sessions. A nested workflow input is prepared once per
        run by ``_child_input_entry``, and a gated child output by
        ``_child_output_entry``.

        Args:
            source: Input port, source port, step port, constant or child
                boundary port.

        Returns:
            The entry holding the source's value.

        Raises:
            ContractError: For a wildcard port, which has no single entry,
                or a source port read outside a pulse of its source.
            WorkflowInputError: When a nested workflow input rejects its value.
        """
        if isinstance(source, SourcePort):
            if source not in self.ports:
                raise ContractError(
                    f"{source.describe()} is read outside a pulse of "
                    f"$sources.{source.source}; active plans run with start()"
                )
            return self.ports[source]
        if isinstance(source, Constant):
            if id(source) not in self.constants:
                value = copy.deepcopy(source.value)
                entry = scalar_entry(value, metadata=EntryMetadata())
                # The Constant is kept alive with its copy, so its id stays unique.
                self.constants[id(source)] = (source, entry)
            return self.constants[id(source)][1]
        if isinstance(source, ChildInputPort):
            return self._child_input_entry(source)
        if isinstance(source, ChildOutputPort):
            return self._child_output_entry(source)
        if isinstance(source, InputPort):
            return self.inputs[source.name]
        if source.output == "*":
            raise ContractError(
                f"{source.describe()} selects several outputs and cannot be bound "
                "to one parameter"
            )

        return self.outputs[(source.step, source.output)]

    def _child_input_entry(self, port: ChildInputPort) -> Entry:
        """Prepare a nested workflow input once per run (decision 021).

        A literal or default (``Constant`` source) is privately copied,
        decoded by the first successful declared kind decoder and checked. Any
        other source is an already prepared payload: every present leaf is
        checked against the child's kinds, and the source entry itself is
        returned, keeping identity, layout, metadata and filtered positions.
        An outer child input is prepared first, so deep chains decode once.
        """
        if port in self.child_inputs:
            return self.child_inputs[port]

        planned = self.plan.child_input(port)
        kinds = kinds_named(self.plan, planned.kinds)
        location = f"Nested workflow input {port.describe()}"
        if isinstance(planned.source, Constant):
            value = copy.deepcopy(planned.source.value)
            payload = decode_payload(value, kinds=kinds, location=location)
            entry = scalar_entry(payload, metadata=EntryMetadata())
        else:
            entry = self.entry_for(planned.source)
            for index in sorted(entry.values):
                if entry.is_filtered(index):
                    continue
                where = f"{location} at index {list(index)}" if index else location
                check_payload(entry.values[index], kinds=kinds, location=where)
        self.child_inputs[port] = entry

        return entry

    def _child_output_entry(self, port: ChildOutputPort) -> Entry:
        """Apply the whole-child gates to a forwarded child output (decision 026).

        The entry shares the source's payloads and metadata. Its structure is
        the source's, or, for a source shallower than the effective layout
        (a forwarded scalar under a batch gate), what the controllers at that
        layout know together; a value is broadcast from its source prefix. A
        position is filtered when its source position is unavailable or any
        gate denies it; a gate denies whole nodes at its own depth, and a
        broadcast keeps the source's unavailable prefixes, so a denied or
        filtered group is absent rather than a group of filtered children.
        Admitted genuine empty groups stay empty. Without gates the source entry itself
        is returned.
        """
        if port in self.child_outputs:
            return self.child_outputs[port]

        planned = self.plan.child_output(port)
        source = self.entry_for(planned.source)
        if not planned.gates:
            self.child_outputs[port] = source
            return source

        depth = planned.layout.depth
        if depth == source.depth:
            children, filtered, domain = source.children, source.filtered, None
        else:
            controllers = [
                self.decisions[gate.controller]
                for gate in planned.gates
                if gate.controller_layout.depth == depth
            ]
            children, filtered, domain = _merged_structure(controllers, depth=depth)
        values: Dict[Index, Any] = {}
        denied = set()
        for index in domain if domain is not None else sorted(source.values):
            prefix = index[: source.depth]
            if not source.has_value(prefix) or _denied(self, planned.gates, index):
                denied.add(index)
                continue
            values[index] = source.values[prefix]
        view = Entry(
            layout=planned.layout,
            metadata=source.metadata,
            children=children,
            values=values,
            filtered=frozenset(filtered) | frozenset(denied),
        )
        # A gate denies the node at its own depth, and a broadcast keeps the
        # source's unavailable prefixes, so a denied or filtered group is
        # filtered as a whole, even a genuinely empty one (decision 003).
        denied_nodes = {
            node
            for gate in planned.gates
            for node in view.nodes_at(gate.controller_layout.depth)
            if _denied(self, (gate,), node)
        }
        if depth > source.depth:
            denied_nodes |= source.filtered
            denied_nodes |= {
                node
                for node in view.nodes_at(source.depth)
                if not source.has_value(node)
            }
        entry = dataclasses.replace(view, filtered=view.filtered | denied_nodes)
        self.child_outputs[port] = entry

        return entry

    def record(self, event: str, **details: Any) -> None:
        """Append a JSON-friendly trace event.

        Args:
            event: Event name.
            **details: JSON-friendly event details.
        """
        self.trace.append({"event": event, **details})


@dataclass(frozen=True)
class _Structure:
    """Known index structure of a step's invocation layout ``P``."""

    children: Mapping[Index, Tuple[Index, ...]]
    filtered: FrozenSet[Index]
    domain: Tuple[Index, ...]
    invocable: FrozenSet[Index]


@dataclass(frozen=True)
class _Call:
    """One block call covering one invocation, or all of a batch step."""

    indices: Tuple[Index, ...]
    arguments: Mapping[str, Any]
    batched: bool


def execute_step(run: RunState, step: PlannedStep) -> None:
    """Run one step for every admitted invocation and record its outputs.

    Args:
        run: State of the current run.
        step: The planned step.

    Raises:
        StepExecutionError: When an argument violates the declaration, the
            block raises, a future fails or the result violates the outputs.
    """
    observer = run.session.observer
    location = list(step.path)
    observer.on_step_started(step=step.path, block_type=step.block_type)
    run.record("step_started", step=location, block_type=step.block_type)

    structure = _invocation_structure(run, step)
    entries = [run.entry_for(binding.source) for binding in step.bindings]
    admitted: List[Index] = []
    leaves: Dict[Index, List[Leaf]] = {}
    skipped: Dict[Index, str] = {}
    for index in structure.domain:
        resolved: List[Leaf] = []
        reason = _gate_denial(run, step, index)
        if reason is None and index not in structure.invocable:
            reason = "filtered_input"
        if reason is None:
            resolved = [
                resolve_leaf(binding, index, entry=entry)
                for binding, entry in zip(step.bindings, entries)
            ]
            reason = skip_reason(step, resolved)
        if reason is not None:
            skipped[index] = reason
            observer.on_invocation_skipped(step=step.path, index=index, reason=reason)
            run.record(
                "invocation_skipped", step=location, index=list(index), reason=reason
            )
            continue
        admitted.append(index)
        leaves[index] = resolved

    _validate_invocations(run, step, leaves=leaves)
    calls = _prepare_calls(step, admitted=admitted, leaves=leaves, entries=entries)
    results: Dict[Index, Any] = {}
    for call in calls:
        results.update(_invoke(run, step, call))

    _record_outputs(
        run,
        step,
        structure=structure,
        results=results,
        skipped=skipped,
        leaves=leaves,
        entries=entries,
    )
    observer.on_step_finished(
        step=step.path, invocations=len(calls), skipped=len(skipped)
    )
    run.record(
        "step_finished", step=location, invocations=len(calls), skipped=len(skipped)
    )


def _invocation_structure(run: RunState, step: PlannedStep) -> _Structure:
    depth = step.invocation_layout.depth
    items = [
        run.entry_for(binding.source)
        for binding in step.bindings
        if binding.mode == "element"
    ]
    groups = [
        run.entry_for(binding.source)
        for binding in step.bindings
        if binding.mode == "group"
    ]
    controllers = [
        run.decisions[gate.controller]
        for gate in step.gates
        if gate.controller_layout.depth == depth
    ]
    providers = items + groups or controllers
    if not providers and depth:
        raise ContractError(
            f"{format_step_path(step.path)} runs over "
            f"{list(step.invocation_layout.axis_ids)}, but no varying binding or "
            "gate at that layout supplies its indices"
        )
    if not providers:
        structure = _Structure(
            children={}, filtered=frozenset(), domain=((),), invocable=frozenset({()})
        )
        return structure

    children, filtered, level = _merged_structure(providers, depth=depth)

    # Decision 016: an item position exists even when filtered, but a group
    # whose parent was filtered before expansion never existed.
    item_positions = {index for item in items for index in item.nodes_at(depth)}
    invocable = {
        index
        for index in level
        if not (items or groups)
        or index in item_positions
        or any(has_group(group, index) for group in groups)
    }
    structure = _Structure(
        children=children,
        filtered=frozenset(filtered),
        domain=tuple(level),
        invocable=frozenset(invocable),
    )

    return structure


def _merged_structure(
    providers: Sequence[Entry], *, depth: int
) -> Tuple[Dict[Index, Tuple[Index, ...]], FrozenSet[Index], List[Index]]:
    """Merge the index structure several entries know over shared axes.

    Providers share the axes but may know different parts of them: a
    position filtered in one branch can be present in another. A node is
    filtered only when no provider knows it.

    Returns:
        Known children per node, filtered nodes, and the known nodes at
        ``depth`` in logical order.
    """
    children: Dict[Index, Tuple[Index, ...]] = {}
    filtered = set()
    level: List[Index] = [()]
    for _ in range(depth):
        deeper = set()
        for node in level:
            knowing = [
                provider
                for provider in providers
                if node in provider.children and not provider.is_filtered(node)
            ]
            if not knowing:
                filtered.add(node)
                continue
            known_children = sorted(
                {child for provider in knowing for child in provider.children[node]}
            )
            children[node] = tuple(known_children)
            deeper.update(known_children)
        level = sorted(deeper)

    return children, frozenset(filtered), level


def _gate_denial(run: RunState, step: PlannedStep, index: Index) -> Optional[str]:
    reason = "denied_by_gate" if _denied(run, step.gates, index) else None

    return reason


def _denied(run: RunState, gates: Sequence[Gate], index: Index) -> bool:
    """Whether a gate denies ``index``: every gate must admit its prefix."""
    for gate in gates:
        decisions = run.decisions[gate.controller]
        prefix = index[: gate.controller_layout.depth]
        if (
            not decisions.has_value(prefix)
            or gate.target not in decisions.values[prefix]
        ):
            return True

    return False


def _validate_invocations(
    run: RunState, step: PlannedStep, *, leaves: Mapping[Index, Sequence[Leaf]]
) -> None:
    """Check every admitted logical invocation before any call (decision 018).

    Runs before batch packaging, so a batch-delivering step is checked per
    logical invocation exactly like a per-invocation step.
    """
    for index, invocation in leaves.items():
        arguments = validation_arguments(step, [leaf.value for leaf in invocation])
        try:
            step.spec.validate_resolved_arguments(step.params, arguments)
        except ContractError as error:
            _fail(run, step, str(error), index=index, cause=error)


def _prepare_calls(
    step: PlannedStep,
    *,
    admitted: Sequence[Index],
    leaves: Mapping[Index, Sequence[Leaf]],
    entries: Sequence[Entry],
) -> List[_Call]:
    if not step.delivers_batches:
        calls = [
            _Call(
                indices=(index,),
                arguments=arguments_for(step, [leaf.value for leaf in leaves[index]]),
                batched=False,
            )
            for index in admitted
        ]
        return calls

    if not admitted:
        return []

    values = batch_values(
        step,
        indices=admitted,
        leaves=[leaves[index] for index in admitted],
        entries=entries,
    )
    batch_call = _Call(
        indices=tuple(admitted), arguments=arguments_for(step, values), batched=True
    )

    return [batch_call]


class _BlockFailure(Exception):
    """A block call or one of its futures failed; the cause is chained."""

    def __init__(self, message: str, *, phase: Optional[str] = None):
        super().__init__(message)
        self.phase = phase


def _invoke(run: RunState, step: PlannedStep, call: _Call) -> Dict[Index, Any]:
    """Call the block once, wait for its futures and split the result per index."""
    index = None if call.batched else call.indices[0]
    run.record(
        "invocation",
        step=list(step.path),
        index=list(index) if index is not None else None,
        indices=[list(item) for item in call.indices],
    )
    context = ExecutionContext(
        step_path=step.path,
        block_type=step.block_type,
        session_id=run.session.session_id,
        run_id=run.run_id,
        indices=call.indices,
    )
    try:
        result = _ready_result(run, step, call, context=context)
    except _BlockFailure as failure:
        _fail(
            run,
            step,
            str(failure),
            index=index,
            cause=failure.__cause__,
            phase=failure.phase,
        )
    run.session.observer.on_invocation(
        step=step.path, index=index, arguments=call.arguments, result=result
    )

    if not call.batched:
        return {index: result}

    if not isinstance(result, (list, tuple)) or len(result) != len(call.indices):
        _fail(
            run,
            step,
            f"a batch-delivering block must return a list with one result per "
            f"invocation ({len(call.indices)}), got {type(result).__name__}",
            index=None,
        )
    split = dict(zip(call.indices, result))

    return split


def _ready_result(
    run: RunState, step: PlannedStep, call: _Call, *, context: ExecutionContext
) -> Any:
    """Call the block and wait for its futures, inside the call's context."""
    with use_execution_context(context):
        try:
            raw = _call_block(run, step, call)
        except PhaseFailure as failure:
            original = failure.__cause__
            raise _BlockFailure(str(failure), phase=failure.phase) from original
        except Exception as error:
            raise _BlockFailure(
                f"block raised {type(error).__name__}: {error}"
            ) from error
        try:
            result = resolve_futures(raw)
        except Exception as error:
            # The failure's traceback keeps this frame; it must not keep the result.
            raw = None
            raise _BlockFailure(
                f"a future returned by the block failed with "
                f"{type(error).__name__}: {error}"
            ) from error

    return result


def _call_block(run: RunState, step: PlannedStep, call: _Call) -> Any:
    """The one place where execution modes differ: ``run()`` or the phase graph."""
    instance = run.session.instances[step.path]
    if step.execution == "run":
        raw = instance.run(**call.arguments)
        return raw

    def record_phase(name: str) -> None:
        run.record(
            "phase",
            step=list(step.path),
            indices=[list(index) for index in call.indices],
            phase=name,
        )

    raw = run_phases(
        instance, step.selected.phases, call.arguments, on_phase=record_phase
    )

    return raw


def _record_outputs(
    run: RunState,
    step: PlannedStep,
    *,
    structure: _Structure,
    results: Mapping[Index, Any],
    skipped: Mapping[Index, str],
    leaves: Mapping[Index, Sequence[Leaf]],
    entries: Sequence[Entry],
) -> None:
    filtered = structure.filtered | frozenset(skipped)
    if step.spec.is_control:
        decisions = {
            index: _decision(run, step, index=index, result=result)
            for index, result in results.items()
        }
        run.decisions[step.path] = Entry(
            layout=step.invocation_layout,
            metadata=EntryMetadata(),
            children=structure.children,
            values=decisions,
            filtered=filtered,
        )
        return

    produced = {
        index: _output_values(run, step, index=index, result=result)
        for index, result in results.items()
    }
    built = {
        output.name: _output_entry(
            run,
            step,
            output,
            structure=structure,
            filtered=filtered,
            produced=produced,
            leaves=leaves,
            entries=entries,
        )
        for output in step.outputs.values()
    }
    _check_shared_expand_axes(run, step, built=built, invoked=list(produced))
    for name, entry in built.items():
        run.outputs[(step.path, name)] = entry


def _decision(
    run: RunState, step: PlannedStep, *, index: Index, result: Any
) -> FrozenSet[str]:
    if not isinstance(result, Select):
        _fail(
            run,
            step,
            f"a control block must return Select(...) or Stop(), got "
            f"{type(result).__name__}",
            index=index,
        )
    unknown = sorted(set(result.targets) - set(step.control_targets))
    if unknown:
        _fail(
            run,
            step,
            f"selected {unknown}, which are not targets of this step; targets: "
            f"{sorted(step.control_targets)}",
            index=index,
        )

    return frozenset(result.targets)


def _output_values(
    run: RunState, step: PlannedStep, *, index: Index, result: Any
) -> Mapping[str, Any]:
    if result is None and not step.outputs:
        return {}
    if isinstance(result, Batch) or not isinstance(result, Mapping):
        _fail(
            run,
            step,
            f"blocks return a mapping of output names to values, got "
            f"{type(result).__name__}; declared outputs: {sorted(step.outputs)}",
            index=index,
        )

    omitted = sorted(set(step.outputs) - set(result))
    unknown = sorted(set(result) - set(step.outputs))
    if omitted or unknown:
        _fail(
            run,
            step,
            f"result keys do not match the declared outputs. Omitted: {omitted}; "
            f"unknown: {unknown}; declared: {sorted(step.outputs)}. Return None "
            "as the value of an output that has no payload",
            index=index,
        )

    return result


def _output_entry(
    run: RunState,
    step: PlannedStep,
    output: PlannedOutput,
    *,
    structure: _Structure,
    filtered: FrozenSet[Index],
    produced: Mapping[Index, Mapping[str, Any]],
    leaves: Mapping[Index, Sequence[Leaf]],
    entries: Sequence[Entry],
) -> Entry:
    kinds = kinds_named(run.plan, output.kinds)
    children = dict(structure.children)
    values: Dict[Index, Any] = {}
    chosen: Dict[Index, Optional[Index]] = {}
    output_filtered = set(filtered)
    for index, values_at in produced.items():
        value = values_at[output.name]
        if output.context_policy == "selected":
            group = _source_group(step, output, leaves[index])
            placed = _chosen_members(run, step, output, value, index=index, group=group)
            for placed_index, payload, member in placed:
                _check_output_value(
                    run, step, output, kinds, payload, index=placed_index
                )
                values[placed_index] = payload
                chosen[placed_index] = member
            if output.transform == "expand":
                children[index] = tuple(placed_index for placed_index, _, _ in placed)
            continue

        if isinstance(value, (Selected, Selection)):
            _fail(
                run,
                step,
                f"output {output.name!r} returned {type(value).__name__}, but "
                f"declares context_policy={output.context_policy!r}; declare "
                "context_policy='selected' with source=<Group field> to return "
                "chosen members",
                index=index,
            )
        if output.transform == "same":
            _check_output_value(run, step, output, kinds, value, index=index)
            values[index] = value
            continue

        if output.transform == "expand":
            pairs = _expanded_children(run, step, output, value, index=index)
            known = tuple(child_index for child_index, _ in pairs)
        else:
            pairs, known = _preserved_children(
                run,
                step,
                output,
                value,
                index=index,
                leaves=leaves[index],
                entries=entries,
            )
        if known is None:
            output_filtered.add(index)
            continue

        for child_index, child in pairs:
            _check_output_value(run, step, output, kinds, child, index=child_index)
            values[child_index] = child
        children[index] = known
        output_filtered.update(child for child in known if child not in values)

    entry = Entry(
        layout=output.layout,
        metadata=_output_metadata(
            step, output, leaves=leaves, entries=entries, chosen=chosen
        ),
        children=children,
        values=values,
        filtered=frozenset(output_filtered),
    )

    return entry


def _chosen_members(
    run: RunState,
    step: PlannedStep,
    output: PlannedOutput,
    value: Any,
    *,
    index: Index,
    group: Optional[Batch],
) -> List[Tuple[Index, Any, Optional[Index]]]:
    """Resolve a ``selected`` output's ``Selected`` / ``Selection`` result.

    Returns ``(output index, payload, chosen member)`` triples: one at
    ``index`` for ``Selected`` (``None`` chooses no member and emits a
    ``None`` payload), and one child ``index + (k,)`` per member, in the
    requested order, for ``Selection``. A member must be one of the group's
    delivered full logical indices.
    """
    wrapper = Selection if output.transform == "expand" else Selected
    if value is None and wrapper is Selected:
        return [(index, None, None)]
    if not isinstance(value, wrapper):
        _fail(
            run,
            step,
            f"output {output.name!r} declares context_policy='selected'; return "
            f"{wrapper.__name__}(...) with indices from "
            f"{output.source_field!r}.indices, got {type(value).__name__}",
            index=index,
        )

    members = dict(group.iter_with_indices()) if group is not None else {}
    selections = value.chosen() if isinstance(value, Selection) else [value]
    placed = []
    for position, selected in enumerate(selections):
        if selected.index not in members:
            _fail(
                run,
                step,
                f"output {output.name!r} selected {list(selected.index)}, which "
                f"{output.source_field!r} did not deliver here; delivered "
                f"{[list(member) for member in members]}",
                index=index,
            )
        payload = (
            members[selected.index]
            if selected.value is SAME_PAYLOAD
            else selected.value
        )
        placed_index = index + (position,) if isinstance(value, Selection) else index
        placed.append((placed_index, payload, selected.index))

    return placed


def _source_group(
    step: PlannedStep, output: PlannedOutput, invocation: Sequence[Leaf]
) -> Optional[Batch]:
    """Group the output's ``source_field`` delivered to one invocation, if any."""
    for position, binding in enumerate(step.bindings):
        if binding.field == output.source_field:
            return invocation[position].value

    return None


def _binding_position(step: PlannedStep, field_name: Optional[str]) -> int:
    """Position in ``step.bindings`` of the binding of a whole field."""
    position = next(
        position
        for position, binding in enumerate(step.bindings)
        if binding.field == field_name
    )

    return position


def _check_output_value(
    run: RunState,
    step: PlannedStep,
    output: PlannedOutput,
    kinds: Sequence[Any],
    value: Any,
    *,
    index: Index,
) -> None:
    if isinstance(value, Batch):
        _fail(
            run,
            step,
            f"output {output.name!r} holds a Batch where one payload is expected; "
            "only expand/preserve outputs return a Batch, and it holds payloads",
            index=index,
        )
    try:
        check_kinds(kinds, value)
    except ContractError as error:
        _fail(
            run,
            step,
            f"output {output.name!r} is not a valid {list(output.kinds)}: {error}",
            index=index,
            cause=error,
        )


def _expanded_children(
    run: RunState, step: PlannedStep, output: PlannedOutput, value: Any, *, index: Index
) -> List[Tuple[Index, Any]]:
    """Place the children of an expand output under the invocation index."""
    if not isinstance(value, Batch):
        _fail(
            run,
            step,
            f"expand output {output.name!r} must return a Batch of children, e.g. "
            f"Batch.of(children), got {type(value).__name__}; a list is one payload",
            index=index,
        )

    pairs = []
    for child_index, child in value.iter_with_indices():
        if len(child_index) == 1 and value.parent_index == ():
            full_index = index + child_index
        elif len(child_index) == len(index) + 1 and child_index[:-1] == index:
            full_index = child_index
        else:
            _fail(
                run,
                step,
                f"expand output {output.name!r} returned child index "
                f"{list(child_index)}; use local indices like (0,) or full indices "
                f"under {list(index)}",
                index=index,
            )
        pairs.append((full_index, child))

    ordered = sorted(pairs, key=lambda pair: pair[0])

    return ordered


def _preserved_children(
    run: RunState,
    step: PlannedStep,
    output: PlannedOutput,
    value: Any,
    *,
    index: Index,
    leaves: Sequence[Leaf],
    entries: Sequence[Entry],
) -> Tuple[List[Tuple[Index, Any]], Optional[Tuple[Index, ...]]]:
    """Align a preserve output with the group the block received.

    Returns the output children and the known child domain of ``index`` in
    the source group, which keeps children filtered upstream as filtered
    positions. The domain is ``None`` when the whole source group was
    unavailable, so the output position stays filtered too.
    """
    position = _binding_position(step, output.group_field)
    leaf = leaves[position]
    group = leaf.value
    if group is None:
        # An absent group (decision 016): nothing to preserve at this index.
        if value is not None and not (isinstance(value, Batch) and not len(value)):
            _fail(
                run,
                step,
                f"preserve output {output.name!r} must be None or an empty Batch "
                f"when {output.group_field!r} is absent, got {type(value).__name__}",
                index=index,
            )
        return [], None

    if not isinstance(value, Batch):
        _fail(
            run,
            step,
            f"preserve output {output.name!r} must return a Batch aligned to "
            f"{output.group_field!r}, got {type(value).__name__}",
            index=index,
        )

    if len(value) != len(group):
        pairs = None
    elif set(value.indices) == set(group.indices):
        pairs = list(value.iter_with_indices())
    elif value.indices == Batch.of(value.content).indices:
        # Local positions of a Batch.of(...) result follow the delivered order.
        pairs = list(zip(group.indices, value.content))
    else:
        pairs = None
    if pairs is None:
        _fail(
            run,
            step,
            f"preserve output {output.name!r} returned indices "
            f"{[list(item) for item in value.indices]}, but {output.group_field!r} "
            f"delivered {[list(item) for item in group.indices]}",
            index=index,
        )

    entry = entries[position]
    if leaf.binding.mode == "constant_group":
        known = (index + (0,),) if leaf.unavailable is None else None
    elif entry.is_filtered(index):
        known = None
    else:
        known = entry.children.get(index)

    ordered = sorted(pairs, key=lambda pair: pair[0])

    return ordered, known


def _check_shared_expand_axes(
    run: RunState,
    step: PlannedStep,
    *,
    built: Mapping[str, Entry],
    invoked: Sequence[Index],
) -> None:
    """Expand outputs sharing one axis must produce corresponding children."""
    by_axis: Dict[str, List[str]] = {}
    for output in step.outputs.values():
        if output.transform == "expand":
            by_axis.setdefault(output.layout.axis_ids[-1], []).append(output.name)

    for axis_id, names in by_axis.items():
        for index in invoked:
            domains = {name: built[name].children.get(index) for name in names}
            if len(set(domains.values())) > 1:
                _fail(
                    run,
                    step,
                    f"outputs {names} share axis {axis_id!r} but returned different "
                    f"child indices: "
                    f"{ {name: [list(item) for item in domain] for name, domain in domains.items()} }",
                    index=index,
                )


def _output_metadata(
    step: PlannedStep,
    output: PlannedOutput,
    *,
    leaves: Mapping[Index, Sequence[Leaf]],
    entries: Sequence[Entry],
    chosen: Mapping[Index, Optional[Index]],
) -> EntryMetadata:
    """Context of one output, following its ``source_field`` and policy.

    ``preserve`` keeps the group's own contexts. Otherwise each invocation
    takes ``common_or_none`` of the contributing bindings' contexts: the
    ``source_field`` bindings, else every varying binding (every binding
    when none varies). Children of ``expand`` inherit the invocation context.

    Member policies then replace contexts taken from the source group::

        first / last       temporal at i  = first / last delivered member's
        selected           both at i      = chosen member's
        selected + expand  temporal at i  = None (a collection has no one time)
                           both at i+(k,) = k-th chosen member's, explicitly

    No member (an empty or absent group, or ``Selected`` returned as
    ``None``) means temporal ``None``, never a stale inherited timestamp.
    """
    if output.transform == "preserve":
        position = _binding_position(step, output.group_field)
        metadata = entries[position].metadata
        return metadata

    positions = [
        position
        for position, binding in enumerate(step.bindings)
        if output.source_field in (None, binding.field)
    ]
    if output.source_field is None:
        varying = [
            position for position in positions if step.bindings[position].is_varying
        ]
        positions = varying or positions

    contexts = {
        kind: {
            index: common_or_none(
                _binding_contexts(
                    index, invocation, entries, positions=positions, kind=kind
                )
            )
            for index, invocation in leaves.items()
        }
        for kind in ("sample", "temporal")
    }
    children: Dict[str, Dict[Index, Any]] = {"sample": {}, "temporal": {}}
    if output.context_policy in MEMBER_POLICIES and step.bindings_for(
        output.source_field
    ):
        source = entries[_binding_position(step, output.source_field)].metadata
        for index, invocation in leaves.items():
            if output.context_policy == "selected":
                # No member at i for a Selection: its children hold them.
                member = chosen.get(index)
            else:
                member = _end_member(output, _source_group(step, output, invocation))
            contexts["temporal"][index] = (
                source.temporal_at(member) if member is not None else None
            )
            if output.context_policy == "selected" and member is not None:
                contexts["sample"][index] = source.sample_at(member)
        if output.transform == "expand" and output.context_policy == "selected":
            for placed_index, member in chosen.items():
                children["sample"][placed_index] = source.sample_at(member)
                children["temporal"][placed_index] = source.temporal_at(member)

    metadata = EntryMetadata(
        sample={**_compact(contexts["sample"]), **children["sample"]},
        temporal={**_compact(contexts["temporal"]), **children["temporal"]},
    )

    return metadata


def _end_member(output: PlannedOutput, group: Optional[Batch]) -> Optional[Index]:
    """First or last delivered member of a group, per the output's policy."""
    if not group:
        return None

    member = group.indices[0] if output.context_policy == "first" else group.indices[-1]

    return member


def _binding_contexts(
    index: Index,
    invocation: Sequence[Leaf],
    entries: Sequence[Entry],
    *,
    positions: Sequence[int],
    kind: str,
) -> List[Any]:
    """Context each available contributing leaf brings to invocation ``index``."""
    contexts = []
    for position in positions:
        leaf, entry = invocation[position], entries[position]
        if leaf.unavailable is not None:
            continue
        if not leaf.binding.is_varying and entry.metadata.is_empty:
            # A constant, literal or context-free parameter has no context.
            continue

        lookup = (
            entry.metadata.sample_at if kind == "sample" else entry.metadata.temporal_at
        )
        if leaf.binding.mode == "group" and len(leaf.value):
            contexts.append(
                common_or_none(lookup(child) for child in leaf.value.indices)
            )
        elif leaf.binding.mode == "group" and kind == "sample":
            # An empty group still belongs to its parent's source.
            contexts.append(lookup(index))
        elif leaf.binding.mode == "group":
            # No member was observed, so no timestamp describes the result;
            # the parent's inherited one (e.g. a root PTS) would be invented.
            contexts.append(None)
        else:
            contexts.append(lookup(index[: leaf.binding.source_layout.depth]))

    return contexts


def _compact(contexts: Mapping[Index, Any]) -> Dict[Index, Any]:
    """One key at the root when every invocation shares a context."""
    values = list(contexts.values())
    if values and all(value == values[0] for value in values[1:]):
        compact = {(): values[0]} if values[0] is not None else {}
        return compact

    return dict(contexts)


def _fail(
    run: RunState,
    step: PlannedStep,
    message: str,
    *,
    index: Optional[Index],
    cause: Optional[BaseException] = None,
    phase: Optional[str] = None,
) -> NoReturn:
    """Report a step failure to the host hooks, then raise it."""
    error = StepExecutionError(
        message,
        step_path=step.path,
        block_type=step.block_type,
        index=index,
        phase=phase,
    )
    error.__cause__ = cause
    run.record(
        "step_failed",
        step=list(step.path),
        index=list(index) if index is not None else None,
        phase=phase,
        error=str(error),
    )
    if run.session.error_handler is not None:
        run.session.error_handler(error)
    run.session.observer.on_error(error=error)

    raise error
