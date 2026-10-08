"""Workflow input preparation of the V2 executor.

Turns the caller's mapping into one ``Entry`` per declared input::

    parameter (layout ())   value as given, or the declared default
    grouped (layout [N,..]) Batch tree        explicit axes, kept as given
                            list [a, b, c]    top axis; nested lists for deeper axes
                            [a] or a          broadcast to the size of axis N

All inputs sharing a top axis must agree on its size; a plain scalar or
singleton list is broadcast to that size (V1 runtime input assembly). Inputs
sharing deeper axes must have identical structure. Futures anywhere in a
supplied value are resolved first. Each leaf then goes through the
declared kinds' deserializers (V1 order: the first that succeeds) and then
their validators. The caller's mapping and containers are never rewritten.
"""

import copy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from roboflow_workflows.execution_engine.v2.controls import ControlSnapshot
from roboflow_workflows.execution_engine.v2.data import (
    Batch,
    EntryLayout,
    EntryMetadata,
    Index,
    InputValue,
    validate_entry,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    WorkflowInputError,
)
from roboflow_workflows.execution_engine.v2.execution.entries import (
    Entry,
    entry_from_tree,
)
from roboflow_workflows.execution_engine.v2.kinds import BUILTIN_KINDS, Kind
from roboflow_workflows.execution_engine.v2.plan import (
    CompiledWorkflow,
    PlannedInput,
    resolve_futures,
)

_BUILTIN_KINDS_BY_NAME = {kind.name: kind for kind in BUILTIN_KINDS}


def kinds_named(plan: CompiledWorkflow, names: Sequence[str]) -> Tuple[Kind, ...]:
    """Return ``Kind`` objects for kind names of a plan.

    Args:
        plan: Plan whose catalogue owns the kinds.
        names: Kind names.

    Returns:
        Catalogue kinds; built-in kinds by name; otherwise hook-free kinds.
    """
    known = plan.catalogue.kinds
    kinds = tuple(
        known.get(name) or _BUILTIN_KINDS_BY_NAME.get(name) or Kind(name=name)
        for name in names
    )

    return kinds


def check_kinds(kinds: Sequence[Kind], payload: Any) -> None:
    """Raise unless one of ``kinds`` accepts ``payload``; ``None`` always passes.

    Args:
        kinds: Accepted kinds (a union).
        payload: Value to check.

    Raises:
        ContractError: When every kind rejects the payload.
    """
    if payload is None or not kinds:
        return

    problems = []
    for kind in kinds:
        try:
            kind.check(payload)
        except ContractError as error:
            problems.append(str(error))
            continue
        return

    raise ContractError("; ".join(problems))


def prepare_inputs(
    plan: CompiledWorkflow,
    inputs: Mapping[str, Any],
    *,
    controls: Optional[ControlSnapshot] = None,
) -> Dict[str, Entry]:
    """Validate the caller's inputs and build one entry per declared input.

    A root input an ``input`` control owns has exactly one writer, the
    session's ``ControlPanel``: its entry comes from ``controls`` and the
    caller may not supply it.

    Args:
        plan: Compiled plan declaring the inputs.
        inputs: Values by input name: payloads, plain lists, ``Batch`` trees
            or ``InputValue`` objects carrying metadata.
        controls: The run's control snapshot; required when the plan
            declares ``input`` controls.

    Returns:
        Entry per input name.

    Raises:
        WorkflowInputError: On unknown or missing inputs, a controlled input
            supplied by the caller, disagreeing sizes or structures, or
            values their kinds reject.
        ContractError: When the plan has controlled inputs and no snapshot.
    """
    if not isinstance(inputs, Mapping):
        raise WorkflowInputError(
            f"Workflow inputs must be a mapping, got {type(inputs).__name__}"
        )

    controlled = plan.controls.controlled_inputs
    if controlled and controls is None:
        raise ContractError(
            f"inputs {sorted(controlled)} are controlled; prepare them with the "
            "session's control snapshot (session.controls.current)"
        )
    supplied_controlled = sorted(set(inputs) & set(controlled))
    if supplied_controlled:
        owners = {name: controlled[name] for name in supplied_controlled}
        raise WorkflowInputError(
            f"Workflow inputs {supplied_controlled} are owned by controls {owners}; "
            "set them with session.controls.update(<control>=value) instead of "
            "passing them to run() or start()"
        )
    unknown = sorted(set(inputs) - set(plan.inputs))
    missing = sorted(
        name
        for name, planned in plan.inputs.items()
        if planned.required and name not in inputs and name not in controlled
    )
    if unknown or missing:
        raise WorkflowInputError(
            f"Workflow inputs do not match the definition. Missing: {missing}; "
            f"unknown: {unknown}; declared: {sorted(plan.inputs)}"
        )

    supplied = {
        name: _supplied(name, inputs[name]) if name in inputs else _default_of(planned)
        for name, planned in plan.inputs.items()
        if name not in controlled
    }
    if controls is not None:
        for name, control in controlled.items():
            supplied[name] = (controls.values[control], EntryMetadata())
    top_sizes = _top_axis_sizes(plan, supplied=supplied)

    entries: Dict[str, Entry] = {}
    for name, planned in plan.inputs.items():
        data, metadata = supplied[name]
        kinds = kinds_named(plan, planned.kinds)
        top_size = top_sizes.get(planned.layout.axis_ids[:1])
        tree = _build_tree(planned, data=data, kinds=kinds, top_size=top_size)
        try:
            validate_entry(tree, layout=planned.layout, metadata=metadata)
        except ContractError as error:
            raise WorkflowInputError(f"Workflow input {name!r}: {error}") from error
        entries[name] = entry_from_tree(tree, layout=planned.layout, metadata=metadata)

    _check_shared_axes(plan, entries=entries)

    return entries


def control_input_entries(
    plan: CompiledWorkflow, controls: ControlSnapshot
) -> Dict[str, Entry]:
    """Build the entries of every controlled input from a snapshot, for one run.

    Each pulse of an active run overlays these on the static entries prepared
    at ``start``, so the pulse reads the values of its own snapshot. Values
    are shared, not copied: control values are immutable built-in scalars.

    Args:
        plan: Compiled plan.
        controls: The run's control snapshot.

    Returns:
        Entry per controlled input name; empty without ``input`` controls.
    """
    entries: Dict[str, Entry] = {}
    for name, control in plan.controls.controlled_inputs.items():
        planned = plan.inputs[name]
        data = controls.values[control]
        kinds = kinds_named(plan, planned.kinds)
        tree = _build_tree(planned, data=data, kinds=kinds, top_size=None)
        metadata = EntryMetadata()
        validate_entry(tree, layout=planned.layout, metadata=metadata)
        entries[name] = entry_from_tree(tree, layout=planned.layout, metadata=metadata)

    return entries


def _supplied(name: str, value: Any) -> Tuple[Any, EntryMetadata]:
    """Unwrap ``InputValue`` and wait for futures anywhere in the value.

    Futures are resolved before any shape, codec or kind check; containers
    without futures, and ready payloads, keep their identity. Readiness does
    not look into an ``InputValue``, so only its data needs a second pass.
    """
    try:
        value = resolve_futures(value)
        if isinstance(value, InputValue):
            data, metadata = resolve_futures(value.data), value.metadata
        else:
            data, metadata = value, EntryMetadata()
    except Exception as error:
        raise WorkflowInputError(
            f"Workflow input {name!r}: a future in the supplied value failed with "
            f"{type(error).__name__}: {error}"
        ) from error

    return data, metadata


def _default_of(planned: PlannedInput) -> Tuple[Any, EntryMetadata]:
    # A private copy per run: blocks may mutate payloads in place, and the
    # default belongs to the plan.
    default = copy.deepcopy(planned.default)

    return default, EntryMetadata()


def _top_axis_sizes(
    plan: CompiledWorkflow, *, supplied: Mapping[str, Tuple[Any, EntryMetadata]]
) -> Dict[Tuple[str, ...], int]:
    fixed: Dict[Tuple[str, ...], Dict[str, int]] = {}
    for name, planned in plan.inputs.items():
        if not planned.layout.depth:
            continue

        data, _ = supplied[name]
        top_axis = planned.layout.axis_ids[:1]
        fixed.setdefault(top_axis, {})
        if isinstance(data, Batch) or (isinstance(data, list) and len(data) != 1):
            fixed[top_axis][name] = len(data)

    sizes: Dict[Tuple[str, ...], int] = {}
    for top_axis, by_input in fixed.items():
        distinct = set(by_input.values())
        if len(distinct) > 1:
            raise WorkflowInputError(
                f"Workflow inputs sharing axis {top_axis[0]!r} must have the same "
                f"length, or length 1 to be broadcast; got {by_input}"
            )
        sizes[top_axis] = distinct.pop() if distinct else 1

    return sizes


def _build_tree(
    planned: PlannedInput,
    *,
    data: Any,
    kinds: Sequence[Kind],
    top_size: Optional[int],
) -> Any:
    name = planned.name
    depth = planned.layout.depth
    if depth and data is None:
        raise WorkflowInputError(
            f"Workflow input {name!r} is grouped along {list(planned.layout.axis_ids)} "
            "but no value was provided"
        )

    if depth and not isinstance(data, Batch):
        if not isinstance(data, list):
            data = [data] * top_size
        elif len(data) == 1 and top_size != 1:
            data = data * top_size

    tree = _adopt(
        data,
        index=(),
        depth=depth,
        layout=planned.layout,
        kinds=kinds,
        name=name,
    )

    return tree


def _adopt(
    node: Any,
    *,
    index: Index,
    depth: int,
    layout: EntryLayout,
    kinds: Sequence[Kind],
    name: str,
) -> Any:
    """Convert one level of plain lists or ``Batch`` into EE-owned groups."""
    if len(index) == depth:
        if isinstance(node, Batch):
            raise WorkflowInputError(
                f"Workflow input {name!r} has a Batch at index {list(index)}, below "
                f"its declared axes {list(layout.axis_ids)}"
            )
        location = f"Workflow input {name!r}" + (
            f" at index {list(index)}" if index else ""
        )
        payload = decode_payload(node, kinds=kinds, location=location)
        return payload

    if isinstance(node, Batch):
        pairs = list(node.iter_with_indices())
        for child_index, _ in pairs:
            if len(child_index) != len(index) + 1 or child_index[:-1] != index:
                raise WorkflowInputError(
                    f"Workflow input {name!r}: group at index {list(index)} holds "
                    f"child index {list(child_index)}; nested groups need full "
                    "logical indices extending their parent"
                )
    elif isinstance(node, list):
        pairs = [(index + (position,), child) for position, child in enumerate(node)]
    else:
        raise WorkflowInputError(
            f"Workflow input {name!r} expects a list or Batch at index {list(index)} "
            f"for axis {layout.axis_ids[len(index)]!r}, got {type(node).__name__}"
        )

    children = [
        _adopt(
            child,
            index=child_index,
            depth=depth,
            layout=layout,
            kinds=kinds,
            name=name,
        )
        for child_index, child in pairs
    ]
    group = Batch(
        children,
        indices=[child_index for child_index, _ in pairs],
        layout=layout,
        parent_index=index,
    )

    return group


def decode_payload(value: Any, *, kinds: Sequence[Kind], location: str) -> Any:
    """Decode a supplied value with its declared kinds, then check it.

    The first declared kind whose decoder succeeds wins (V1 order); without a
    decoder the value is kept. Used for workflow inputs and for nested child
    inputs bound to literals or defaults.

    Args:
        value: Supplied value; ``None`` is kept as a ``None`` payload.
        kinds: Declared kinds (a union).
        location: Where the value belongs, for error messages.

    Returns:
        The payload.

    Raises:
        WorkflowInputError: When every decoder fails or no kind accepts it.
    """
    if value is None:
        return None

    decoders = [kind for kind in kinds if kind.deserialize is not None]
    payload = value
    if decoders:
        payload = _first_successful_decode(value, decoders=decoders, location=location)
    check_payload(payload, kinds=kinds, location=location)

    return payload


def check_payload(value: Any, *, kinds: Sequence[Kind], location: str) -> None:
    """Check an already prepared payload against declared kinds, never decoding it.

    Args:
        value: Payload; ``None`` passes.
        kinds: Declared kinds (a union).
        location: Where the value belongs, for error messages.

    Raises:
        WorkflowInputError: When no kind accepts the payload.
    """
    try:
        check_kinds(kinds, value)
    except ContractError as error:
        raise WorkflowInputError(
            f"{location} is not a valid {[kind.name for kind in kinds]}: {error}"
        ) from error


def _first_successful_decode(value: Any, *, decoders: List[Kind], location: str) -> Any:
    failures = []
    for kind in decoders:
        try:
            payload = kind.to_payload(value)
        except Exception as error:
            failures.append((kind.name, error))
            continue
        return payload

    details = "; ".join(
        f"{kind}: {type(error).__name__}: {error}" for kind, error in failures
    )
    error = WorkflowInputError(
        f"{location} could not be deserialized by any declared kind ({details})"
    )
    raise error from failures[-1][1]


def _check_shared_axes(plan: CompiledWorkflow, *, entries: Mapping[str, Entry]) -> None:
    try:
        check_shared_axes(entries, what="Workflow inputs")
    except ContractError as error:
        raise WorkflowInputError(str(error)) from error


def check_shared_axes(entries: Mapping[str, Entry], *, what: str) -> None:
    """Entries sharing a prefix of axes must share its group structure.

    Two axes with equal ids describe corresponding positions, so every
    group node the entries know along a shared prefix must be the same.
    Ungrouped entries take no part.

    Args:
        entries: Entries by name; each carries its layout.
        what: Noun for error messages, e.g. ``"Workflow inputs"``.

    Raises:
        ContractError: Naming the two entries, the shared axes and the depth
            at which their groups differ.
    """
    names = [name for name, entry in entries.items() if entry.depth]
    for position, first in enumerate(names):
        for second in names[position + 1 :]:
            left_ids = entries[first].layout.axis_ids
            shared = _shared_prefix(left_ids, entries[second].layout.axis_ids)
            for depth in range(shared):
                left = _groups_at(entries[first], depth=depth)
                right = _groups_at(entries[second], depth=depth)
                if left != right:
                    raise ContractError(
                        f"{what} {first!r} and {second!r} share axes "
                        f"{list(left_ids[: depth + 1])} but their groups at depth "
                        f"{depth} differ"
                    )


def _shared_prefix(left: Sequence[str], right: Sequence[str]) -> int:
    length = 0
    for left_id, right_id in zip(left, right):
        if left_id != right_id:
            break
        length += 1

    return length


def _groups_at(entry: Entry, *, depth: int) -> Dict[Index, Tuple[Index, ...]]:
    groups = {node: entry.children[node] for node in entry.nodes_at(depth)}

    return groups
