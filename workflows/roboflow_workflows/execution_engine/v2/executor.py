"""Serial reference executor for the passive Workflows V2 path.

One ``execute_plan`` call is one passive invocation: it binds the caller's
inputs as EE-owned entries, executes every compiled step in order with a
fresh block instance, assembles item and one-trailing-group views per
invocation prefix, applies gate decisions, normalises ordinary block results
into per-output entries with derived layouts and metadata, and resolves the
declared workflow outputs.

Completion protocol (M1): every node emits exactly one aggregate buffer per
invocation, carrying all of its declared outputs. Before that emission the
trace shows the node ``pending``; afterwards ``complete`` or ``filtered`` (the
whole emission was gated out). A failure raises ``WorkflowExecutionError``
whose trace ends with ``run_failed``. A missing declared output after
``run()`` returns is an error, never pending work.

Filtering model: entries keep the surviving payload tree plus a set of
minimal filtered logical index paths. A group that still exists in the tree
with no children is a genuine empty group and reaches reducers; a group
whose candidate children were all gated out is recorded as filtered and its
dependent invocations are skipped without renumbering surviving siblings.
"""

import uuid
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
)

from roboflow_workflows.execution_engine.v2.compiler import (
    APPEND,
    BATCH_VIEW,
    COLLAPSE,
    ITEM_VIEW,
    PRESERVE,
    CompiledInput,
    CompiledOutput,
    CompiledStep,
    CompiledWorkflow,
    instantiate_block,
)
from roboflow_workflows.execution_engine.v2.data import (
    Batch,
    EntryLayout,
    EntryMetadata,
    Index,
    InputValue,
    WorkflowsBuffer,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    WorkflowCompileError,
    WorkflowExecutionError,
)

STATUS_PENDING = "pending"
STATUS_COMPLETE = "complete"
STATUS_FILTERED = "filtered"

REASON_UPSTREAM = "upstream_filtered"
REASON_GATE_FILTERED = "gate_filtered"
REASON_GATE_FALSE = "gate_false"
REASON_ALL_CHILDREN_FILTERED = "all_children_filtered"


@dataclass(frozen=True)
class RunResult:
    """Outcome of one passive invocation of a compiled plan.

    Args:
        outputs: Buffer keyed by declared workflow output names. Filtered
            outputs are absent from its dictionaries.
        statuses: Mapping of every declared output name to ``"complete"`` or
            ``"filtered"``.
        trace: Ordered JSON-friendly executor events including pending and
            terminal node states, filtered invocations and output resolution.
        invocation_id: Identifier distinct for every invocation.
        filtered_paths: Mapping of output name to the minimal logical index
            paths that were filtered inside that entry; ``()`` means the whole
            entry was filtered.
    """

    outputs: WorkflowsBuffer
    statuses: Mapping[str, str]
    trace: Tuple[Mapping[str, Any], ...]
    invocation_id: str
    filtered_paths: Mapping[str, Tuple[Index, ...]]


@dataclass(frozen=True)
class _Entry:
    node: str
    selector: str
    layout: EntryLayout
    metadata: EntryMetadata
    data: Any
    present: bool
    filtered: FrozenSet[Index]

    @property
    def depth(self) -> int:
        return len(self.layout.axes)


@dataclass(frozen=True)
class _InvocationResult:
    """Normalised result of one block call at one invocation prefix.

    ``outputs`` maps each output name to one payload (item preserve, batch
    collapse) or to a ``Batch`` of children carrying full logical indices
    (append, batch preserve). A payload is never a ``Batch``: normalisation
    rejects it. ``input_child_indices`` are the batch-input children this
    call consumed, used to resolve collapsed contexts; ``None`` for a call
    with item inputs only.
    """

    outputs: Mapping[str, Any]
    input_child_indices: Optional[Tuple[Index, ...]]


@dataclass
class _Invocation:
    plan: CompiledWorkflow
    invocation_id: str
    pulse_id: int
    lineage_id: str
    entries: Dict[str, _Entry] = field(default_factory=dict)
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, event: str, **details: Any) -> None:
        self.trace.append({"event": event, **details})


def execute_plan(plan: CompiledWorkflow, *, inputs: Mapping[str, Any]) -> RunResult:
    """Run a compiled plan once with fresh invocation state.

    Args:
        plan: Compiled workflow produced by ``compile_workflow``.
        inputs: Mapping of workflow input name to ``InputValue`` or plain value.

    Returns:
        ``RunResult`` describing outputs, statuses and the executor trace.

    Raises:
        WorkflowExecutionError: If binding, a block call, a block result or a
            buffer violates the compiled contract. The error carries the
            partial trace as its ``trace`` attribute.
    """
    invocation = _Invocation(
        plan=plan,
        invocation_id=uuid.uuid4().hex,
        pulse_id=plan.next_pulse_id(),
        lineage_id=f"plan:{plan.plan_id}",
    )
    invocation.record(
        "run_started",
        invocation_id=invocation.invocation_id,
        pulse_id=invocation.pulse_id,
        lineage_id=invocation.lineage_id,
        execution_order=[step.name for step in plan.steps],
    )
    try:
        _bind_inputs(invocation, inputs=inputs)
        for step in plan.steps:
            _execute_step(invocation, step=step)
        result = _resolve_outputs(invocation)
    except WorkflowExecutionError as error:
        invocation.record("run_failed", error=str(error))
        error.trace = tuple(invocation.trace)
        raise
    invocation.record("run_completed", statuses=dict(result.statuses))

    return RunResult(
        outputs=result.outputs,
        statuses=result.statuses,
        trace=tuple(invocation.trace),
        invocation_id=result.invocation_id,
        filtered_paths=result.filtered_paths,
    )


def _bind_inputs(invocation: _Invocation, *, inputs: Mapping[str, Any]) -> None:
    plan = invocation.plan
    if not isinstance(inputs, Mapping):
        raise WorkflowExecutionError(
            f"Workflow inputs must be a mapping, got {type(inputs).__name__}."
        )
    unknown = sorted(set(inputs) - set(plan.inputs))
    missing = sorted(set(plan.inputs) - set(inputs))
    if unknown or missing:
        raise WorkflowExecutionError(
            f"Workflow inputs do not match the compiled definition. Missing: "
            f"{missing}; unknown: {unknown}; declared: {sorted(plan.inputs)}."
        )
    for name, compiled in plan.inputs.items():
        value = inputs[name]
        if not isinstance(value, InputValue):
            value = InputValue(value)
        metadata = value.metadata
        tree = _adopt_input_tree(
            invocation, compiled=compiled, data=value.data, metadata=metadata
        )
        entry = _Entry(
            node=compiled.node,
            selector=f"$inputs.{name}",
            layout=compiled.layout,
            metadata=metadata,
            data=tree,
            present=True,
            filtered=frozenset(),
        )
        _validate_emission(invocation, node=compiled.node, entries=[entry])
        invocation.entries[entry.selector] = entry
        invocation.record(
            "input_bound",
            node=compiled.node,
            kind=compiled.kind,
            axes=_axis_ids(compiled.layout),
            status=STATUS_COMPLETE,
        )


def _adopt_input_tree(
    invocation: _Invocation,
    *,
    compiled: CompiledInput,
    data: Any,
    metadata: EntryMetadata,
) -> Any:
    depth = len(compiled.layout.axes)
    context = compiled.node

    def adopt(node: Any, index: Index, level: int) -> Any:
        if level == depth:
            if isinstance(node, Batch):
                raise WorkflowExecutionError(
                    f"{context} declares {depth} grouping axes "
                    f"{_axis_ids(compiled.layout)} but received a Batch at index "
                    f"{list(index)}; there is no automatic payload-to-Batch casting."
                )
            _validate_kind(
                invocation,
                kind=compiled.kind,
                payload=node,
                context=context,
                index=index,
            )
            return node
        if not isinstance(node, Batch):
            raise WorkflowExecutionError(
                f"{context} declares {depth} grouping axes "
                f"{_axis_ids(compiled.layout)} but received {type(node).__name__} "
                f"at index {list(index)} where a Batch was expected. Wrap grouped "
                "payloads with Batch.of(...)."
            )
        content: List[Any] = []
        indices: List[Index] = []
        seen: Set[Index] = set()
        for child_index, child in node.iter_with_indices():
            _validate_child_index(child_index, parent=index, context=context)
            if child_index in seen:
                raise WorkflowExecutionError(
                    f"{context} has duplicate logical index {list(child_index)}."
                )
            seen.add(child_index)
            content.append(adopt(child, child_index, level + 1))
            indices.append(child_index)
        return Batch(
            content,
            indices=indices,
            layout=compiled.layout,
            metadata=metadata,
            parent_index=index,
        )

    tree = adopt(data, (), 0)

    return tree


def _validate_child_index(child_index: Any, *, parent: Index, context: str) -> None:
    valid = (
        isinstance(child_index, tuple)
        and len(child_index) == len(parent) + 1
        and all(
            isinstance(item, int) and not isinstance(item, bool) for item in child_index
        )
        and child_index[: len(parent)] == parent
    )
    if not valid:
        raise WorkflowExecutionError(
            f"{context} child of group {list(parent)} has logical index "
            f"{child_index!r}; expected a full path of {len(parent) + 1} "
            f"non-negative integers extending the parent index."
        )


def _validate_kind(
    invocation: _Invocation, *, kind: str, payload: Any, context: str, index: Index
) -> None:
    try:
        invocation.plan.registry.validate(kind, payload)
    except ContractError as error:
        raise WorkflowExecutionError(
            f"{context} payload at index {list(index)} does not satisfy kind "
            f"{kind!r}: {error}"
        ) from error


def _execute_step(invocation: _Invocation, *, step: CompiledStep) -> None:
    node = step.node
    invocation.record(
        "step_started",
        node=node,
        block=step.block_type,
        prefix_axes=[axis.id for axis in step.prefix],
        status=STATUS_PENDING,
    )
    entries = {
        port: invocation.entries[selector] for port, selector in step.inputs.items()
    }
    depth = step.prefix_depth
    domains = {
        port: (_nodes_at_depth(entry.data, depth) if entry.present else {})
        for port, entry in entries.items()
    }
    union: Set[Index] = set()
    for domain in domains.values():
        union.update(domain)
    _check_prefix_domains(step=step, entries=entries)
    filtered_now: Dict[Index, str] = {}
    gated_paths = _apply_gate(
        invocation, step=step, entries=entries, filtered=filtered_now
    )
    try:
        block = instantiate_block(step)
    except WorkflowCompileError as error:
        raise WorkflowExecutionError(str(error)) from error

    results: Dict[Index, _InvocationResult] = {}
    batch_ports = [port for port in entries if step.views[port] == BATCH_VIEW]
    item_ports = [port for port in entries if step.views[port] == ITEM_VIEW]
    for index in sorted(union):
        upstream = [
            port
            for port, entry in entries.items()
            if _is_filtered(entry.filtered, index)
        ]
        if upstream:
            _filter_invocation(
                invocation,
                step=step,
                index=index,
                reason=REASON_UPSTREAM,
                filtered=filtered_now,
                detail={"ports": upstream},
            )
            continue
        if _is_filtered(gated_paths, index):
            continue
        kwargs: Dict[str, Any] = {}
        input_child_indices: Optional[Tuple[Index, ...]] = None
        if batch_ports:
            input_child_indices, all_gated = _assemble_group_views(
                step=step,
                index=index,
                entries=entries,
                domains=domains,
                batch_ports=batch_ports,
                kwargs=kwargs,
            )
            if all_gated:
                _filter_invocation(
                    invocation,
                    step=step,
                    index=index,
                    reason=REASON_ALL_CHILDREN_FILTERED,
                    filtered=filtered_now,
                    detail={"ports": batch_ports},
                )
                continue
        for port in item_ports:
            kwargs[port] = domains[port][index]
        try:
            raw = block.run(**kwargs)
        except Exception as error:
            raise WorkflowExecutionError(
                f"{node} block {step.block_type!r} failed at index {list(index)}: "
                f"{type(error).__name__}: {error}"
            ) from error
        results[index] = _normalize_result(
            invocation,
            step=step,
            index=index,
            raw=raw,
            input_child_indices=input_child_indices,
        )
        invocation.record("invocation_completed", node=node, index=list(index))

    # A filtered path belongs to the axes it addresses, not to its numeric
    # position: keep each path with the axis ids of its own entry so outputs
    # only inherit masks on the same lineage.
    inherited: Set[Tuple[Index, Tuple[str, ...]]] = set()
    for entry in entries.values():
        axis_ids = entry.layout.axis_ids
        inherited.update((path, axis_ids[: len(path)]) for path in entry.filtered)
    produced: List[_Entry] = []
    for output in step.outputs.values():
        entry = _build_output_entry(
            step=step,
            output=output,
            source=entries[output.source],
            results=results,
            filtered_now=filtered_now,
            inherited=inherited,
        )
        produced.append(entry)
        invocation.entries[entry.selector] = entry
    _validate_emission(invocation, node=node, entries=produced)
    status = (
        STATUS_COMPLETE if any(entry.present for entry in produced) else STATUS_FILTERED
    )
    invocation.record(
        "step_completed",
        node=node,
        status=status,
        invocations=len(results),
        filtered_invocations=len(filtered_now),
        outputs={
            entry.selector: {
                "axes": _axis_ids(entry.layout),
                "present": entry.present,
                "filtered_paths": [list(path) for path in sorted(entry.filtered)],
            }
            for entry in produced
        },
    )


def _apply_gate(
    invocation: _Invocation,
    *,
    step: CompiledStep,
    entries: Mapping[str, _Entry],
    filtered: Dict[Index, str],
) -> Set[Index]:
    """Evaluate the step's gate once per addressed path; return gated paths.

    The gate is looked up for every path that exists at the gate's depth in
    the step's inputs, including paths whose descendants are genuine empty
    groups. A ``false`` or filtered decision filters that whole path, so an
    ancestor gate also removes empty groups beneath it. Paths already filtered
    upstream are left to the upstream rule. A path with no descendants at the
    gate's depth needs no decision, so none is fabricated.
    """
    if step.gate is None:
        return set()

    node = step.node
    gate_entry = invocation.entries[step.gate]
    gate_values: Dict[Index, Any] = {}
    if gate_entry.present:
        gate_values = _nodes_at_depth(gate_entry.data, step.gate_depth)
    addressed: Set[Index] = set()
    for entry in entries.values():
        if entry.present:
            addressed.update(_nodes_at_depth(entry.data, step.gate_depth))
    gated: Set[Index] = set()
    for path in sorted(addressed):
        if any(_is_filtered(entry.filtered, path) for entry in entries.values()):
            continue
        if _is_filtered(gate_entry.filtered, path):
            reason = REASON_GATE_FILTERED
        elif path not in gate_values:
            raise WorkflowExecutionError(
                f"{node} gate {step.gate!r} has no value at index {list(path)}, "
                "which the step's inputs contain (possibly as an empty group). "
                "A gate must decide every path it addresses."
            )
        else:
            decision = gate_values[path]
            if not isinstance(decision, bool):
                raise WorkflowExecutionError(
                    f"{node} gate {step.gate!r} produced {decision!r} at index "
                    f"{list(path)}; gate values must be actual bools."
                )
            if decision:
                continue
            reason = REASON_GATE_FALSE
        gated.add(path)
        _filter_invocation(
            invocation,
            step=step,
            index=path,
            reason=reason,
            filtered=filtered,
            detail={"gate": step.gate},
        )

    return gated


def _check_prefix_domains(
    *,
    step: CompiledStep,
    entries: Mapping[str, _Entry],
) -> None:
    """Require every input to hold the same groups down to the prefix depth.

    Groups are compared at every level of the invocation prefix, not only at
    its leaves, so a genuine empty parent present on one input must exist on
    the others too. A path explicitly filtered on an input may be absent there.
    """
    depth = step.prefix_depth
    skeletons = {
        port: (_prefix_skeleton(entry.data, depth) if entry.present else set())
        for port, entry in entries.items()
    }
    union: Set[Index] = set()
    for skeleton in skeletons.values():
        union.update(skeleton)
    for index in sorted(union):
        missing = [
            port
            for port, entry in entries.items()
            if index not in skeletons[port] and not _is_filtered(entry.filtered, index)
        ]
        if missing:
            present = [port for port in entries if port not in missing]
            raise WorkflowExecutionError(
                f"{step.node} input domains disagree at index {list(index)}: present "
                f"on {present} but absent (and not filtered) on {missing}. Bound "
                f"selectors: {dict(step.inputs)}. Batched and item inputs must "
                "share one lineage and index domain, including empty groups."
            )


def _prefix_skeleton(tree: Any, depth: int) -> Set[Index]:
    """Return indices of all groups and items at levels 1..``depth``."""
    skeleton: Set[Index] = set()
    level: List[Any] = [tree]
    for _ in range(depth):
        deeper: List[Any] = []
        for node in level:
            for child_index, child in node.iter_with_indices():
                skeleton.add(child_index)
                deeper.append(child)
        level = deeper

    return skeleton


def _filter_invocation(
    invocation: _Invocation,
    *,
    step: CompiledStep,
    index: Index,
    reason: str,
    filtered: Dict[Index, str],
    detail: Mapping[str, Any],
) -> None:
    filtered[index] = reason
    invocation.record(
        "invocation_filtered",
        node=step.node,
        index=list(index),
        reason=reason,
        **detail,
    )


def _assemble_group_views(
    *,
    step: CompiledStep,
    index: Index,
    entries: Mapping[str, _Entry],
    domains: Mapping[str, Mapping[Index, Any]],
    batch_ports: List[str],
    kwargs: Dict[str, Any],
) -> Tuple[Tuple[Index, ...], bool]:
    """Fill ``kwargs`` with one-trailing-group views; report all-gated groups."""
    children_by_port: Dict[str, Dict[Index, Any]] = {}
    unfiltered_by_port: Dict[str, Set[Index]] = {}
    candidates_filtered = False
    for port in batch_ports:
        entry = entries[port]
        group = domains[port][index]
        children = {
            child_index: child for child_index, child in group.iter_with_indices()
        }
        children_by_port[port] = children
        unfiltered_by_port[port] = {
            child_index
            for child_index in children
            if not _is_filtered(entry.filtered, child_index)
        }
        candidates_filtered = candidates_filtered or _has_filtered_descendants(
            entry.filtered, index
        )
    union: Set[Index] = set()
    for unfiltered in unfiltered_by_port.values():
        union.update(unfiltered)
    for port in batch_ports:
        entry = entries[port]
        absent = sorted(
            child_index
            for child_index in union
            if child_index not in children_by_port[port]
            and not _is_filtered(entry.filtered, child_index)
        )
        if absent:
            raise WorkflowExecutionError(
                f"{step.node} batched inputs disagree on the child domain of group "
                f"{list(index)}: {port!r} ({step.inputs[port]!r}) lacks "
                f"{[list(item) for item in absent]} present on a sibling input."
            )
    surviving: Optional[Set[Index]] = None
    for unfiltered in unfiltered_by_port.values():
        surviving = set(unfiltered) if surviving is None else surviving & unfiltered
    child_indices = tuple(sorted(surviving or ()))
    if not child_indices and candidates_filtered:
        return child_indices, True
    # Every batch view is presented in one canonical order, sorted by logical
    # index, so position k of each Batch argument is the same logical child.
    # An existing group is reused only when its indices already are exactly
    # that sequence; an equal length alone does not prove it.
    for port in batch_ports:
        entry = entries[port]
        group = domains[port][index]
        if group.indices == child_indices:
            kwargs[port] = group
        else:
            children = children_by_port[port]
            kwargs[port] = Batch(
                [children[child_index] for child_index in child_indices],
                indices=list(child_indices),
                layout=entry.layout,
                metadata=entry.metadata,
                parent_index=index,
            )

    return child_indices, False


def _normalize_result(
    invocation: _Invocation,
    *,
    step: CompiledStep,
    index: Index,
    raw: Any,
    input_child_indices: Optional[Tuple[Index, ...]],
) -> _InvocationResult:
    node = step.node
    where = f"{node} at index {list(index)}"
    if isinstance(raw, Batch) or not isinstance(raw, Mapping):
        raise WorkflowExecutionError(
            f"{where} returned {type(raw).__name__}; blocks must return a mapping "
            f"of output names to payloads. Declared outputs: {sorted(step.outputs)}."
        )
    if len(raw) == 0:
        raise WorkflowExecutionError(
            f"{where} returned an empty mapping; this is a missing result, not a "
            "filtered emission. Filtering is decided by `when` gates, and every "
            f"declared output {sorted(step.outputs)} must be returned."
        )
    missing = sorted(set(step.outputs) - set(raw))
    unknown = sorted(set(raw) - set(step.outputs))
    if missing or unknown:
        raise WorkflowExecutionError(
            f"{where} result keys do not match declared outputs. Missing: "
            f"{missing}; unknown: {unknown}; declared: {sorted(step.outputs)}."
        )

    outputs: Dict[str, Any] = {}
    for output in step.outputs.values():
        value = raw[output.name]
        context = f"{node}.{output.name}"
        if output.transform == APPEND:
            outputs[output.name] = _normalize_appended(
                invocation, output=output, context=context, index=index, value=value
            )
        elif output.transform == PRESERVE and output.source_view == BATCH_VIEW:
            outputs[output.name] = _normalize_preserved_group(
                invocation,
                output=output,
                context=context,
                index=index,
                value=value,
                input_child_indices=input_child_indices or (),
            )
        else:
            outputs[output.name] = _normalize_payload(
                invocation, output=output, context=context, index=index, value=value
            )
    _check_shared_axis_domains(step=step, index=index, outputs=outputs)
    result = _InvocationResult(
        outputs=MappingProxyType(outputs), input_child_indices=input_child_indices
    )

    return result


def _normalize_payload(
    invocation: _Invocation,
    *,
    output: CompiledOutput,
    context: str,
    index: Index,
    value: Any,
) -> Any:
    """Check the one payload of an item preserve or batch collapse output."""
    if isinstance(value, Batch):
        raise WorkflowExecutionError(
            f"{context} at index {list(index)} returned a Batch but an "
            f"{output.source_view} {output.transform} output must return "
            "one payload."
        )
    _validate_kind(
        invocation,
        kind=output.kind,
        payload=value,
        context=context,
        index=index,
    )

    return value


def _normalize_appended(
    invocation: _Invocation,
    *,
    output: CompiledOutput,
    context: str,
    index: Index,
    value: Any,
) -> Batch:
    """Re-index an appended local-index Batch under the invocation prefix."""
    if not isinstance(value, Batch):
        raise WorkflowExecutionError(
            f"{context} at index {list(index)} returned {type(value).__name__}; an "
            "append output must return a Batch of children, e.g. Batch.of(children)."
        )
    children: Dict[Index, Any] = {}
    for local_index, child in value.iter_with_indices():
        valid = (
            isinstance(local_index, tuple)
            and len(local_index) == 1
            and isinstance(local_index[0], int)
            and not isinstance(local_index[0], bool)
            and local_index[0] >= 0
        )
        if not valid:
            raise WorkflowExecutionError(
                f"{context} at index {list(index)} returned child index "
                f"{local_index!r}; appended children need local one-component "
                "indices such as (0,), (1,)."
            )
        full_index = index + local_index
        if full_index in children:
            raise WorkflowExecutionError(
                f"{context} at index {list(index)} returned duplicate child index "
                f"{local_index!r}."
            )
        if isinstance(child, Batch):
            raise WorkflowExecutionError(
                f"{context} at index {list(index)} returned a nested Batch child at "
                f"{local_index!r}; an append output introduces exactly one axis."
            )
        _validate_kind(
            invocation,
            kind=output.kind,
            payload=child,
            context=context,
            index=full_index,
        )
        children[full_index] = child
    group = _sorted_group(children, parent_index=index)

    return group


def _normalize_preserved_group(
    invocation: _Invocation,
    *,
    output: CompiledOutput,
    context: str,
    index: Index,
    value: Any,
    input_child_indices: Tuple[Index, ...],
) -> Batch:
    """Check that a batch preserve output echoes the consumed child indices."""
    if not isinstance(value, Batch):
        raise WorkflowExecutionError(
            f"{context} at index {list(index)} returned {type(value).__name__}; a "
            "batch preserve output must return a Batch carrying the supplied full "
            "child indices."
        )
    returned_indices = list(value.indices)
    if sorted(returned_indices) != list(input_child_indices) or len(
        set(returned_indices)
    ) != len(returned_indices):
        raise WorkflowExecutionError(
            f"{context} at index {list(index)} returned child indices "
            f"{[list(item) for item in returned_indices]} but the supplied group "
            f"domain is {[list(item) for item in input_child_indices]}. A batch "
            "preserve output must echo the supplied full logical indices."
        )
    children: Dict[Index, Any] = {}
    for child_index, child in value.iter_with_indices():
        if isinstance(child, Batch):
            raise WorkflowExecutionError(
                f"{context} at index {list(index)} returned a nested Batch at "
                f"{list(child_index)}; only one trailing group axis is supported."
            )
        _validate_kind(
            invocation,
            kind=output.kind,
            payload=child,
            context=context,
            index=child_index,
        )
        children[child_index] = child
    group = _sorted_group(children, parent_index=index)

    return group


def _sorted_group(children: Mapping[Index, Any], *, parent_index: Index) -> Batch:
    ordered = sorted(children)
    group = Batch(
        [children[child_index] for child_index in ordered],
        indices=ordered,
        parent_index=parent_index,
    )

    return group


def _check_shared_axis_domains(
    *,
    step: CompiledStep,
    index: Index,
    outputs: Mapping[str, Any],
) -> None:
    by_key: Dict[str, List[str]] = {}
    for output in step.outputs.values():
        if output.axis_key is not None:
            by_key.setdefault(output.axis_key, []).append(output.name)
    for key, names in by_key.items():
        if len(names) < 2:
            continue
        domains = {
            name: [list(child_index) for child_index in outputs[name].indices]
            for name in names
        }
        reference = domains[names[0]]
        for name in names[1:]:
            if domains[name] != reference:
                raise WorkflowExecutionError(
                    f"{step.node} outputs {names} share axis key {key!r} and promise "
                    f"corresponding children, but at index {list(index)} their "
                    f"child domains differ: {domains}."
                )


def _build_output_entry(
    *,
    step: CompiledStep,
    output: CompiledOutput,
    source: _Entry,
    results: Mapping[Index, _InvocationResult],
    filtered_now: Mapping[Index, str],
    inherited: Set[Tuple[Index, Tuple[str, ...]]],
) -> _Entry:
    depth = step.prefix_depth
    output_axis_ids = output.layout.axis_ids
    filtered: Set[Index] = {
        path
        for path, axis_ids in inherited
        if len(path) <= output.depth and axis_ids == output_axis_ids[: len(path)]
    }
    filtered.update(filtered_now)
    present = source.present and _canonicalize_filtered(
        source.data, depth=depth, results=results, filtered=filtered
    )
    filtered = _minimal_paths(filtered)
    if not present:
        filtered.add(())
    metadata = _output_metadata(
        output=output, source=source, results=results, filtered=filtered, depth=depth
    )
    tree = None
    if present:
        tree = _build_tree(
            source.data,
            depth=depth,
            results=results,
            filtered=filtered,
            output=output,
            metadata=metadata,
        )
    entry = _Entry(
        node=step.node,
        selector=f"{step.node}.{output.name}",
        layout=output.layout,
        metadata=metadata,
        data=tree,
        present=present,
        filtered=frozenset(filtered),
    )

    return entry


def _canonicalize_filtered(
    source_tree: Any,
    *,
    depth: int,
    results: Mapping[Index, _InvocationResult],
    filtered: Set[Index],
) -> bool:
    """Mark groups whose candidates were all filtered; return root presence."""

    def visit(node: Any, index: Index, level: int) -> bool:
        if level == depth:
            return index in results
        surviving = 0
        for child_index, child in node.iter_with_indices():
            if _is_filtered(filtered, child_index):
                continue
            if visit(child, child_index, level + 1):
                surviving += 1
        if surviving == 0 and _has_filtered_descendants(filtered, index):
            filtered.add(index)
            return False
        return True

    # A gate on an ungrouped ancestor can filter the whole entry, even when
    # the entry has no invocations at all (e.g. an empty root group).
    root_present = () not in filtered and visit(source_tree, (), 0)
    if not root_present:
        filtered.add(())

    return root_present


def _build_tree(
    source_tree: Any,
    *,
    depth: int,
    results: Mapping[Index, _InvocationResult],
    filtered: Set[Index],
    output: CompiledOutput,
    metadata: EntryMetadata,
) -> Any:
    layout = output.layout

    def build(node: Any, index: Index, level: int) -> Any:
        if level == depth:
            value = results[index].outputs[output.name]
            if not isinstance(value, Batch):
                return value
            # A group emitted by this call: attach the output entry's views.
            return Batch(
                value.content,
                indices=value.indices,
                layout=layout,
                metadata=metadata,
                parent_index=index,
            )
        content: List[Any] = []
        indices: List[Index] = []
        for child_index, child in node.iter_with_indices():
            if _is_filtered(filtered, child_index):
                continue
            content.append(build(child, child_index, level + 1))
            indices.append(child_index)
        return Batch(
            content,
            indices=indices,
            layout=layout,
            metadata=metadata,
            parent_index=index,
        )

    tree = build(source_tree, (), 0)

    return tree


def _output_metadata(
    *,
    output: CompiledOutput,
    source: _Entry,
    results: Mapping[Index, _InvocationResult],
    filtered: Set[Index],
    depth: int,
) -> EntryMetadata:
    sample = _restrict_paths(
        source.metadata.sample, filtered=filtered, depth=output.depth
    )
    temporal = _restrict_paths(
        source.metadata.temporal, filtered=filtered, depth=output.depth
    )
    if output.transform != COLLAPSE:
        if sample is source.metadata.sample and temporal is source.metadata.temporal:
            return source.metadata
        return EntryMetadata(sample=sample, temporal=temporal)
    sample = dict(sample)
    temporal = dict(temporal)
    for index, result in results.items():
        if not result.input_child_indices:
            continue
        _apply_common_or_none(
            sample,
            index=index,
            children=result.input_child_indices,
            resolve=source.metadata.sample_at,
        )
        _apply_common_or_none(
            temporal,
            index=index,
            children=result.input_child_indices,
            resolve=source.metadata.temporal_at,
        )

    return EntryMetadata(sample=sample, temporal=temporal)


def _apply_common_or_none(
    target: Dict[Index, Any],
    *,
    index: Index,
    children: Iterable[Index],
    resolve: Any,
) -> None:
    resolved = [resolve(child_index) for child_index in children]
    common = resolved[0]
    agree = all(item == common for item in resolved[1:])
    value = common if agree else None
    inherited = _lookup(target, index)
    if value != inherited:
        target[index] = value


def _lookup(mapping: Mapping[Index, Any], index: Index) -> Any:
    for length in range(len(index), -1, -1):
        prefix = index[:length]
        if prefix in mapping:
            return mapping[prefix]

    return None


def _restrict_paths(
    mapping: Mapping[Index, Any], *, filtered: Set[Index], depth: int
) -> Mapping[Index, Any]:
    keep = {
        path: value
        for path, value in mapping.items()
        if len(path) <= depth and not _is_filtered(filtered, path)
    }
    if len(keep) == len(mapping):
        return mapping

    return keep


def _validate_emission(
    invocation: _Invocation, *, node: str, entries: List[_Entry]
) -> None:
    """Check a node's one aggregate emission against the buffer invariants.

    Consumers read ``invocation.entries``; building the node's
    ``WorkflowsBuffer`` here only validates every present entry's tree,
    full index paths, layout and metadata, so an inconsistent emission fails
    at its producer instead of in a downstream step.
    """
    present = [entry for entry in entries if entry.present]
    try:
        WorkflowsBuffer(
            lineage_id=invocation.lineage_id,
            pulse_id=invocation.pulse_id,
            data={entry.selector: entry.data for entry in present},
            layout={entry.selector: entry.layout for entry in present},
            metadata={entry.selector: entry.metadata for entry in present},
        )
    except (ValueError, TypeError) as error:
        raise WorkflowExecutionError(
            f"{node} produced an invalid buffer: {error}"
        ) from error


def _resolve_outputs(invocation: _Invocation) -> RunResult:
    plan = invocation.plan
    data: Dict[str, Any] = {}
    layout: Dict[str, EntryLayout] = {}
    metadata: Dict[str, EntryMetadata] = {}
    statuses: Dict[str, str] = {}
    filtered_paths: Dict[str, Tuple[Index, ...]] = {}
    for output in plan.outputs:
        entry = invocation.entries[output.selector]
        if entry.present:
            data[output.name] = entry.data
            layout[output.name] = entry.layout
            metadata[output.name] = entry.metadata
            statuses[output.name] = STATUS_COMPLETE
        else:
            statuses[output.name] = STATUS_FILTERED
        filtered_paths[output.name] = tuple(sorted(entry.filtered))
        invocation.record(
            "output_resolved",
            name=output.name,
            selector=output.selector,
            status=statuses[output.name],
            axes=_axis_ids(entry.layout),
            filtered_paths=[list(path) for path in filtered_paths[output.name]],
        )
    try:
        buffer = WorkflowsBuffer(
            lineage_id=invocation.lineage_id,
            pulse_id=invocation.pulse_id,
            data=data,
            layout=layout,
            metadata=metadata,
        )
    except (ValueError, TypeError) as error:
        raise WorkflowExecutionError(
            f"Workflow outputs produced an invalid buffer: {error}"
        ) from error

    return RunResult(
        outputs=buffer,
        statuses=MappingProxyType(statuses),
        trace=(),
        invocation_id=invocation.invocation_id,
        filtered_paths=MappingProxyType(filtered_paths),
    )


def _nodes_at_depth(tree: Any, depth: int) -> Dict[Index, Any]:
    found: Dict[Index, Any] = {(): tree}
    for _ in range(depth):
        deeper: Dict[Index, Any] = {}
        for node in found.values():
            for child_index, child in node.iter_with_indices():
                deeper[child_index] = child
        found = deeper

    return found


def _is_filtered(filtered: Iterable[Index], index: Index) -> bool:
    for path in filtered:
        if index[: len(path)] == path:
            return True

    return False


def _has_filtered_descendants(filtered: Iterable[Index], index: Index) -> bool:
    for path in filtered:
        if len(path) > len(index) and path[: len(index)] == index:
            return True

    return False


def _minimal_paths(filtered: Set[Index]) -> Set[Index]:
    minimal = {
        path
        for path in filtered
        if not any(other != path and path[: len(other)] == other for other in filtered)
    }

    return minimal


def _axis_ids(layout: EntryLayout) -> List[str]:
    ids = [axis.id for axis in layout.axes]

    return ids
