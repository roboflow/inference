"""Argument assembly for block calls.

Each selector leaf of a step is one ``Binding``. At invocation index ``i`` its
value comes from the bound entry:

    element         value at i
    ancestor        value at i[:len(S)]          (parent broadcast)
    constant        the one value of a parameter, Constant or scalar output
    group           Batch of the surviving children of i, full indices
    constant_group  Batch([value]) at i + (0,)   (scalar cast into a group)

A leaf is unavailable when its position was filtered or never existed; an
unavailable item or group arrives as ``None``. A group that exists but whose
children were all filtered is a present, empty survivor group. Default blocks
skip an invocation with an unavailable leaf, a ``None`` from a varying item
binding, or an all-filtered group. Blocks with ``accepts_empty`` are invoked
anyway. A genuinely empty group reaches every block. Static values (literals,
parameters, constants) never cause a skip, even when they are ``None``.

Every invocation is assembled twice from the same leaves:

    validation_arguments   parsed literals + selected payloads by identity;
                           checked by BlockSpec.validate_resolved_arguments
                           (decision 018) before any call of the step
    arguments_for          what the block receives: literals as private
                           copies, cast literals as their one-element group

A literal written at a ``Group`` or ``batch="always"`` position arrives as a
compiler-emitted binding with an empty selector (``is_literal_cast``).
"""

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel
from roboflow_workflows.execution_engine.v2.data import Batch, Index
from roboflow_workflows.execution_engine.v2.execution.entries import Entry
from roboflow_workflows.execution_engine.v2.plan import (
    Binding,
    PlannedStep,
    SkipReason,
)

GROUP_MODES = ("group", "constant_group")


@dataclass(frozen=True)
class Leaf:
    """Value of one binding at one invocation.

    Args:
        binding: The binding.
        value: What the block receives: ``None`` when unavailable, the
            surviving children (possibly none) for a present group.
        unavailable: Why the value is unavailable or empty, or ``None``.
    """

    binding: Binding
    value: Any
    unavailable: Optional[SkipReason] = None


def resolve_leaf(binding: Binding, index: Index, *, entry: Entry) -> Leaf:
    """Look up the value of one binding at one invocation index.

    Args:
        binding: Binding to resolve.
        index: Invocation index at the step's invocation layout.
        entry: Entry of the bound source.

    Returns:
        The leaf, marked unavailable when its position is filtered or missing.
    """
    if binding.mode == "group":
        leaf = _group_leaf(binding, index, entry=entry)
        return leaf

    position = index[: binding.source_layout.depth]
    if not entry.has_value(position):
        leaf = Leaf(binding=binding, value=None, unavailable="filtered_input")
        return leaf

    value = entry.values[position]

    if binding.mode == "constant_group":
        value = Batch(
            [value],
            indices=[index + (0,)],
            layout=binding.cast_layout,
            metadata=entry.metadata,
            parent_index=index,
        )

    leaf = Leaf(binding=binding, value=value)

    return leaf


def _group_leaf(binding: Binding, index: Index, *, entry: Entry) -> Leaf:
    if not has_group(entry, index):
        # The parent never produced this group (decision 016): absent, not empty.
        leaf = Leaf(binding=binding, value=None, unavailable="filtered_input")
        return leaf

    known = entry.children[index]
    survivors = [child for child in known if entry.has_value(child)]
    group = Batch(
        [entry.values[child] for child in survivors],
        indices=survivors,
        layout=binding.source_layout,
        metadata=entry.metadata,
        parent_index=index,
    )
    unavailable = "all_children_filtered" if known and not survivors else None
    leaf = Leaf(binding=binding, value=group, unavailable=unavailable)

    return leaf


def has_group(entry: Entry, index: Index) -> bool:
    """Return whether ``entry`` holds a materialized group at ``index``.

    Args:
        entry: Entry of a group source.
        index: Parent index of the group.

    Returns:
        ``True`` for an unfiltered known group, empty or not.
    """
    present = not entry.is_filtered(index) and index in entry.children

    return present


def skip_reason(step: PlannedStep, leaves: Sequence[Leaf]) -> Optional[SkipReason]:
    """Return why a default block must not run with these leaves.

    Args:
        step: The step.
        leaves: Leaves of one invocation, one per binding.

    Returns:
        A skip reason, or ``None`` when the block runs. Always ``None`` for
        blocks that accept empty values.
    """
    if step.spec.accepts_empty:
        return None

    for leaf in leaves:
        if leaf.unavailable is not None:
            return leaf.unavailable
        if (
            leaf.binding.is_varying
            and leaf.binding.mode not in GROUP_MODES
            and leaf.value is None
        ):
            return "empty_value"

    return None


def is_literal_cast(binding: Binding) -> bool:
    """Return whether a binding carries a literal the compiler cast.

    The compiler binds a literal written at a ``Group`` or ``batch="always"``
    position as a ``Constant`` with an empty selector, so it is delivered as a
    one-element group or batch. It stays a literal for validation.

    Args:
        binding: A binding of the step.

    Returns:
        ``True`` for such a cast literal.
    """
    cast = binding.selector == ""

    return cast


def validation_arguments(step: PlannedStep, values: Sequence[Any]) -> Dict[str, Any]:
    """Build one logical invocation as ``validate_resolved_arguments`` expects.

    Literals are the step's parsed parameters (never re-parsed or copied),
    selected payloads are the resolved values by identity, groups are
    ``Batch`` values and unavailable leaves ``None``. A cast literal stays the
    literal it was written as, so it is not checked against a selector kind.

    Args:
        step: The step.
        values: Value per binding at this invocation, aligned with
            ``step.bindings``.

    Returns:
        One value per ``Params`` field.
    """
    selected = [
        (binding, value)
        for binding, value in zip(step.bindings, values)
        if not is_literal_cast(binding)
    ]
    arguments = _assemble(step, selected, literal=_shell)

    return arguments


def arguments_for(step: PlannedStep, values: Sequence[Any]) -> Dict[str, Any]:
    """Build the keyword arguments of one block call.

    Literal parameters are private copies: the validated parameters belong to
    the plan and are shared by every invocation, run and session. Bound values,
    including cast literals, are delivered as they are.

    Args:
        step: The step.
        values: Delivered value per binding, aligned with ``step.bindings``.

    Returns:
        One argument per ``Params`` field.
    """
    arguments = _assemble(step, list(zip(step.bindings, values)), literal=_private_copy)

    return arguments


def _assemble(
    step: PlannedStep,
    bound: Sequence[Tuple[Binding, Any]],
    *,
    literal: Callable[[Any], Any],
) -> Dict[str, Any]:
    """Place bound values into the step's parameters, field by field."""
    by_field: Dict[str, Dict[Tuple[Any, ...], Any]] = {}
    for binding, value in bound:
        by_field.setdefault(binding.field, {})[binding.position] = value

    arguments: Dict[str, Any] = {}
    for name in step.spec.fields:
        leaves = by_field.get(name, {})
        if () in leaves:
            arguments[name] = leaves[()]
            continue

        argument = literal(getattr(step.params, name))
        for (position,), value in leaves.items():
            argument[position] = value
        arguments[name] = argument

    return arguments


def _shell(value: Any) -> Any:
    # A new container shell, so selected leaves never land in the plan's params.
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)

    return value


def _private_copy(value: Any) -> Any:
    if isinstance(value, (list, dict, set, BaseModel)):
        copied = copy.deepcopy(value)
        return copied

    return value


def batch_values(
    step: PlannedStep,
    *,
    indices: Sequence[Index],
    leaves: Sequence[Sequence[Leaf]],
    entries: Sequence[Entry],
) -> List[Any]:
    """Collect the values of one batch-delivering call.

    Leaves with ``Binding.delivers_batch`` become one ``Batch`` over every
    admitted invocation, across parents (indices ``(0,)`` when the step runs
    once at ``P == ()``); constants are repeated under each index. Other
    leaves are constants and arrive plain.

    Args:
        step: A step with ``delivers_batches``.
        indices: Admitted invocation indices, in order.
        leaves: Leaves per admitted invocation, aligned with ``indices``.
        entries: Bound entry per binding, whose metadata the delivered
            batches expose.

    Returns:
        Delivered value per binding, aligned with ``step.bindings``.
    """
    batch_indices = list(indices) if step.invocation_layout.depth else [(0,)]
    values: List[Any] = []
    for position, (binding, entry) in enumerate(zip(step.bindings, entries)):
        column = [invocation[position].value for invocation in leaves]
        if not binding.delivers_batch:
            values.append(column[0])
            continue

        item_view = binding.mode not in GROUP_MODES and step.invocation_layout.depth
        values.append(
            Batch(
                column,
                indices=batch_indices,
                layout=step.invocation_layout if item_view else None,
                metadata=entry.metadata if item_view else None,
            )
        )

    return values
