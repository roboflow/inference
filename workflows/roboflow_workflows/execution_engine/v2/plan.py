"""Compiled plan, execution session and run result of the V2 engine.

This module is the contract between the compiler, which builds a
``CompiledWorkflow``, and the executor, which runs it::

    plan = compile_workflow(definition, catalogue=catalogue)   # no block created
    session = plan.create_session(resources={"api_key": key})  # one instance per step
    result = session.run({"image": frame})                    # repeatable; state kept
    rows = result.rows()                                      # V1-shaped rows

A plan is pure data: steps in execution order, validated parameters, one
binding per selector leaf, invocation layouts, control gates and per-output
layouts and context sources. It never holds block instances or resources, so
describing it allocates nothing. ``create_session`` resolves constructor
resources and constructs every step once; ``ExecutionSession.run`` may be
called repeatedly and keeps block state. A new session has new instances.

Binding modes relate a bound value's layout ``S`` to the step's invocation
layout ``P`` (both ``EntryLayout``):

==================  ===========================  ======================================
Mode                Layout relation              Value per invocation at index ``i``
==================  ===========================  ======================================
``element``         ``S == P``                   element at ``i``
``ancestor``        ``S`` proper prefix of ``P``  element at ``i[:len(S)]``
``constant``        ``S == ()``                  the same value
``group``           ``S == P + (axis,)``         ``Batch`` of the children of ``i``
``constant_group``  ``S == ()``                  one-element ``Batch`` at ``i + (0,)``
==================  ===========================  ======================================

A ``constant_group`` binding keeps its true source layout ``()`` and declares
``cast_layout`` (``P`` plus a cast axis) as the layout of the delivered group.
V1 performs this cast for scalar leaves of dict/list group parameters at any
parent level (tasks/m1r-casting-check). For a whole-field scalar group beside a
batched parent, V1 fails in its compiler with a ``TypeError``; V2 applies the
same cast there as a disclosed correction.

Batch delivery is decided per step, not per class. ``PlannedStep.delivers_batches``
is true when a binding has ``batch="always"``, or ``batch="if_varying"`` with a
non-constant mode. Such a step is called once with every invocation: its
``always`` leaves arrive as ``Batch`` (constants cast), its varying
``if_varying`` leaves as ``Batch``, and all other leaves as plain values; it
returns a list of results in invocation order. Otherwise ``run`` is called per
invocation and returns one result. A CSV-shaped step whose ``if_varying``
leaves are all constant therefore returns a single mapping.

Readiness: futures in block results are resolved before dependent steps and
before a ``RunResult`` is returned (the accepted ready-boundary design). V1
additionally lets callers defer output future resolution
(``resolve_output_futures=False``). That option is deliberately not offered
here; it is a disclosed boundary difference pending human review.

Active plans: a definition with ``sources`` compiles to the same one plan with
``PlannedSource`` records, ``PlannedOutputGroup`` records and a causal
``domain`` on every step: the source whose pulses trigger it, or ``None`` for a
static step. ``route(source)`` lists the steps one pulse of that source runs
(its own domain plus every static step, in plan order); ``groups_of(source)``
lists the groups anchored on it. No step or group joins two sources until an
alignment operator exists. ``ExecutionSession.start`` drives such a plan; it
delegates to ``roboflow_workflows.execution_engine.v2.active.runtime``.

The executor, not this module, implements execution and row construction.
``run`` and ``rows`` delegate to ``roboflow_workflows.execution_engine.v2.execution``.
"""

import importlib
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.context import (
    ExecutionContext,
    use_execution_context,
)
from roboflow_workflows.execution_engine.v2.data import (
    Axis,
    Batch,
    EntryLayout,
    Index,
    WorkflowsBuffer,
)
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    BlockSpec,
    ContextPolicy,
    OutputTransform,
    is_selector_segment,
    parse_selector,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    ResourceError,
    StepExecutionError,
    StepPath,
    WorkflowInputError,
    format_step_path,
)
from roboflow_workflows.execution_engine.v2.kinds import kinds_compatible
from roboflow_workflows.execution_engine.v2.resources import (
    ResolvedResource,
    ResourceResolver,
)
from roboflow_workflows.execution_engine.v2.sources import (
    SourceParams,
    SourceSpec,
    source_step_path,
)

EXECUTION_MODULE = "roboflow_workflows.execution_engine.v2.execution"
ACTIVE_RUNTIME_MODULE = "roboflow_workflows.execution_engine.v2.active.runtime"

BindingMode = Literal["element", "ancestor", "constant", "group", "constant_group"]
BINDING_MODES: Tuple[str, ...] = (
    "element",
    "ancestor",
    "constant",
    "group",
    "constant_group",
)
CONSTANT_MODES: Tuple[str, ...] = ("constant", "constant_group")
SkipReason = Literal[
    "denied_by_gate",
    "filtered_input",
    "empty_value",
    "all_children_filtered",
]
OutputStatus = Literal["complete", "filtered"]


@dataclass(frozen=True)
class InputPort:
    """A root workflow input.

    Args:
        name: Declared workflow input name (a selector segment).
    """

    name: str

    def describe(self) -> str:
        """Return the selector text of this port."""
        return f"$inputs.{self.name}"


@dataclass(frozen=True)
class StepPort:
    """An output of a planned step.

    Args:
        step: Path of the producing step.
        output: Output name; ``"*"`` selects all outputs.
    """

    step: StepPath
    output: str

    def describe(self) -> str:
        """Return the selector text of this port."""
        return f"{format_step_path(self.step)}.{self.output}"


@dataclass(frozen=True)
class Constant:
    """A value fixed at compile time, e.g. a nested workflow input default.

    Args:
        value: The constant value; delivered without copying.
    """

    value: Any

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        return {"constant": repr(self.value)}


@dataclass(frozen=True)
class ChildInputPort:
    """An input of a nested workflow after composition.

    Every consumer and direct output of the child input shares this one port;
    its ``PlannedChildInput`` says where the value comes from and which kinds
    the child declared.

    Args:
        scope: Path of the nested workflow step, e.g. ``("child",)``.
        name: Input name declared by the child workflow.
    """

    scope: StepPath
    name: str

    def describe(self) -> str:
        """Return a readable location, e.g. ``$steps.child: $inputs.x``."""
        return f"{format_step_path(self.scope)}: $inputs.{self.name}"


@dataclass(frozen=True)
class ChildOutputPort:
    """A nested workflow output that does not come from a child step.

    Used for a child output that forwards a child input, a literal or a
    default, or a deeper child's such output (decision 026). Its
    ``PlannedChildOutput`` applies the whole-child gates. A child output read
    from a child step's output stays a ``StepPort``: that step is gated
    already.

    Args:
        scope: Path of the nested workflow step, e.g. ``("child",)``.
        name: Output name declared by the child workflow.
    """

    scope: StepPath
    name: str

    def describe(self) -> str:
        """Return the parent's selector text, e.g. ``$steps.child.out``."""
        return f"{format_step_path(self.scope)}.{self.name}"


@dataclass(frozen=True)
class SourcePort:
    """One port of a declared source, as consumers and groups address it.

    Args:
        source: Declared source name.
        output: Port name declared by the source class.
    """

    source: str
    output: str

    def describe(self) -> str:
        """Return the selector text of this port, e.g. ``$sources.camera.image``."""
        return f"$sources.{self.source}.{self.output}"


Source = Union[
    InputPort, StepPort, Constant, ChildInputPort, ChildOutputPort, SourcePort
]
BoundaryPort = Union[ChildInputPort, ChildOutputPort]


@dataclass(frozen=True)
class PlannedChildInput:
    """A nested workflow input kept as one boundary node (decision 021).

    It introduces no axis and runs no block. At run time its value is
    materialized once per run: a ``Constant`` source (literal or child
    default) is privately copied, decoded by the first successful declared
    kind decoder and checked; any other source is an already prepared payload
    that is only checked against ``kinds``, keeping identity, layout, metadata
    and filter structure.

    Args:
        scope: Path of the nested workflow step.
        name: Child input name.
        kinds: Kind names the child declared for the input.
        layout: Layout of the value; equal to the source's layout.
        source: Where the value comes from; an outer child input or a
            sibling's ``ChildOutputPort`` for deeper nesting.

    Raises:
        ContractError: On names that are not selector segments.
    """

    scope: StepPath
    name: str
    kinds: Tuple[str, ...]
    layout: EntryLayout
    source: Source

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", tuple(self.scope))
        object.__setattr__(self, "kinds", tuple(self.kinds))
        parts = self.scope + (self.name,)
        if not self.scope or not all(is_selector_segment(part) for part in parts):
            raise ContractError(
                "Child input scope and name must be selector segments, got "
                f"{self.scope!r} / {self.name!r}"
            )

    @property
    def port(self) -> ChildInputPort:
        """The port consumers use to address this input."""
        return ChildInputPort(scope=self.scope, name=self.name)

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {
            "port": self.port.describe(),
            "kinds": list(self.kinds),
            "axes": list(self.layout.axis_ids),
            "source": _describe_source(self.source),
        }

        return description


@dataclass(frozen=True)
class AxisOrigin:
    """Where an axis identity comes from.

    Args:
        kind: ``input`` (a workflow input's axis), ``source`` (a declared
            source's local axis), ``expand`` (created by an expanding step
            output) or ``cast`` (created by a scalar cast into a one-element
            group).
        name: Input name, source name, output name or parameter name,
            respectively.
        step: Producing step for ``expand`` and ``cast``; ``None`` otherwise.
    """

    kind: Literal["input", "source", "expand", "cast"]
    name: str
    step: Optional[StepPath] = None

    def describe(self) -> str:
        """Return a readable description."""
        if self.kind == "input":
            return f"$inputs.{self.name}"
        if self.kind == "source":
            return f"$sources.{self.name}"
        if self.kind == "expand":
            return f"{format_step_path(self.step)}.{self.name}"

        return f"{format_step_path(self.step)} cast of {self.name}"


@dataclass(frozen=True)
class Binding:
    """One selector leaf of a step and the value source replacing it.

    Args:
        field: Parameter holding the selector.
        position: ``()`` for the whole field, ``(i,)`` or ``(key,)`` for a
            list element or dict value. The executor rebuilds the same
            list/dict around resolved leaves.
        selector: Selector text as written in the definition.
        source: Where the value comes from.
        source_layout: Layout of the source value; empty for constants.
        mode: How the value maps onto invocations (module table).
        batch: Batch mode of the leaf declaration: ``never``, ``always`` or
            ``if_varying``.
        cast_layout: For ``constant_group`` only: layout of the delivered
            one-element group, ``P`` plus a cast axis owned by this step.
    """

    field: str
    position: Tuple[Any, ...]
    selector: str
    source: Source
    source_layout: EntryLayout
    mode: BindingMode
    batch: str = "never"
    cast_layout: Optional[EntryLayout] = None

    def __post_init__(self) -> None:
        if self.mode not in BINDING_MODES:
            raise ContractError(
                f"Binding {self.field!r} has unknown mode {self.mode!r}; "
                f"expected one of {list(BINDING_MODES)}"
            )
        if isinstance(self.source, Constant) and self.mode not in CONSTANT_MODES:
            raise ContractError(
                f"Binding {self.field!r} has a Constant source but mode {self.mode!r}"
            )
        if self.mode == "constant_group" and self.cast_layout is None:
            raise ContractError(
                f"Binding {self.field!r} in mode 'constant_group' needs cast_layout"
            )
        if self.mode != "constant_group" and self.cast_layout is not None:
            raise ContractError(
                f"Binding {self.field!r}: cast_layout is only allowed in mode "
                f"'constant_group', got mode {self.mode!r}"
            )
        object.__setattr__(self, "position", tuple(self.position))

    @property
    def field_path(self) -> Tuple[Any, ...]:
        """Field name followed by the position inside it."""
        return (self.field,) + self.position

    @property
    def is_varying(self) -> bool:
        """Whether the value can differ between invocations."""
        return self.mode not in CONSTANT_MODES

    @property
    def group_layout(self) -> Optional[EntryLayout]:
        """Layout of the delivered group for group modes, else ``None``."""
        if self.mode == "group":
            return self.source_layout

        return self.cast_layout

    @property
    def delivers_batch(self) -> bool:
        """Whether this leaf reaches the block as a ``Batch`` of invocations."""
        return self.batch == "always" or (
            self.batch == "if_varying" and self.is_varying
        )

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {
            "field": list(self.field_path),
            "selector": self.selector,
            "source": _describe_source(self.source),
            "source_axes": list(self.source_layout.axis_ids),
            "mode": self.mode,
            "batch": self.batch,
            "cast_axes": list(self.cast_layout.axis_ids) if self.cast_layout else None,
        }

        return description


@dataclass(frozen=True)
class Gate:
    """A control decision that must allow a step's invocation.

    A step runs at index ``i`` only if every gate allows ``i[:len(controller
    axes)]``. Gates combine by conjunction; a controller that did not run at
    that prefix, or selected no target there, denies it. An ancestor gate
    admits every deeper index under an allowed prefix.

    Args:
        controller: Path of the control step.
        target: Target selector key in the controller's ``control_targets``.
        controller_layout: Invocation layout of the controller.
    """

    controller: StepPath
    target: str
    controller_layout: EntryLayout

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {
            "controller": format_step_path(self.controller),
            "target": self.target,
            "controller_axes": list(self.controller_layout.axis_ids),
        }

        return description


@dataclass(frozen=True)
class PlannedChildOutput:
    """A child output governed by the gates on its whole child (decision 026).

    It introduces no axis and runs no block. At run time it is one entry per
    run: the source's values under the effective ``layout``, with every
    position filtered that any gate denies. Payloads are shared, never
    copied; kinds and metadata are the source's.

    The effective layout is the deepest of the source layout and the gate
    layouts, all of which are axis-id prefixes of it. A shallower source,
    such as a forwarded scalar under a batch gate, is broadcast over the
    original indices the deepest gate controller knows.

    Args:
        scope: Path of the nested workflow step.
        name: Child output name.
        layout: Effective layout of the value.
        source: What the child output reads inside the child.
        gates: Controllers targeting the whole child from its parent scope;
            the same gates its steps carry. May be empty.

    Raises:
        ContractError: On names that are not selector segments.
    """

    scope: StepPath
    name: str
    layout: EntryLayout
    source: Source
    gates: Tuple[Gate, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", tuple(self.scope))
        object.__setattr__(self, "gates", tuple(self.gates))
        parts = self.scope + (self.name,)
        if not self.scope or not all(is_selector_segment(part) for part in parts):
            raise ContractError(
                "Child output scope and name must be selector segments, got "
                f"{self.scope!r} / {self.name!r}"
            )

    @property
    def port(self) -> ChildOutputPort:
        """The port consumers use to address this output."""
        return ChildOutputPort(scope=self.scope, name=self.name)

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {
            "port": self.port.describe(),
            "axes": list(self.layout.axis_ids),
            "source": _describe_source(self.source),
            "gates": [gate.describe() for gate in self.gates],
        }

        return description


@dataclass(frozen=True)
class PlannedOutput:
    """Compiled layout and context source of one step output.

    Args:
        name: Output name.
        kinds: Declared kind names.
        layout: Layout of the produced entry.
        transform: ``same``, ``expand`` or ``preserve`` (see ``Output``).
        group_field: The ``Group`` field whose children ``preserve`` keeps.
        source_field: Field whose bindings provide context; ``None`` means
            every data binding of the step.
        context_policy: How several contributing contexts combine.
    """

    name: str
    kinds: Tuple[str, ...]
    layout: EntryLayout
    transform: OutputTransform = "same"
    group_field: Optional[str] = None
    source_field: Optional[str] = None
    context_policy: ContextPolicy = "common_or_none"

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {
            "kinds": list(self.kinds),
            "axes": list(self.layout.axis_ids),
            "transform": self.transform,
            "group_field": self.group_field,
            "source_field": self.source_field,
            "context_policy": self.context_policy,
        }

        return description


@dataclass(frozen=True)
class PlannedStep:
    """One step of a compiled plan.

    The invocation domain is every index of ``invocation_layout`` that exists
    in the structure of a varying binding's source (filtered positions
    included) or, for steps without varying bindings, every index at which
    the gating controllers ran. It stays known when all selected data at an
    index is missing, so empty-accepting blocks can still be invoked there.

    Args:
        path: Scope path, unique in the plan, e.g. ``("child", "scale")``.
        spec: Declaration of the block class.
        namespace: Catalogue namespace of the block, for resource keys.
        params: Validated parameters; selector positions still hold selectors.
        bindings: One binding per data selector leaf.
        invocation_layout: Layout ``P``: the step runs once per index of ``P``.
        outputs: Compiled outputs by name.
        gates: Control decisions governing this step.
        control_targets: For control blocks, target selector (as the block
            sees it) to every step it governs; a nested workflow target maps
            to all of that workflow's steps.
        dependencies: Paths of steps this step depends on (data and control).
        domain: Name of the source whose pulses trigger this step, reached
            through its data, gates or nested boundaries; ``None`` for a
            static step, which runs once per admitted pulse of every source.
            Always ``None`` in a plan without sources.

    Raises:
        ContractError: When names, bindings, outputs or control data
            contradict the declaration or the invocation layout.
    """

    path: StepPath
    spec: BlockSpec
    namespace: str
    params: BlockParams
    bindings: Tuple[Binding, ...]
    invocation_layout: EntryLayout
    outputs: Mapping[str, PlannedOutput]
    gates: Tuple[Gate, ...] = ()
    control_targets: Mapping[str, Tuple[StepPath, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    dependencies: Tuple[StepPath, ...] = ()
    domain: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.path or not all(is_selector_segment(part) for part in self.path):
            raise ContractError(
                f"Step path parts must be selector segments (letters, digits, _ or -), "
                f"got {self.path!r}"
            )

        object.__setattr__(self, "bindings", tuple(self.bindings))
        object.__setattr__(self, "gates", tuple(self.gates))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))
        object.__setattr__(
            self,
            "control_targets",
            MappingProxyType(
                {key: tuple(paths) for key, paths in self.control_targets.items()}
            ),
        )

        location = format_step_path(self.path)
        for binding in self.bindings:
            self._check_binding(binding, location=location)
        for output in self.outputs.values():
            self._check_output(output, location=location)
        if self.control_targets and not self.spec.is_control:
            raise ContractError(f"{location}: control_targets on a non-control block")

    @property
    def block_type(self) -> str:
        """Canonical block type."""
        return self.spec.type

    @property
    def delivers_batches(self) -> bool:
        """Whether this step is called once with ``Batch`` leaves (module doc)."""
        return any(binding.delivers_batch for binding in self.bindings)

    def bindings_for(self, field_name: str) -> Tuple[Binding, ...]:
        """Return every binding of one parameter, in position order.

        Args:
            field_name: Parameter name.

        Returns:
            The bindings; empty when the parameter holds no selector.
        """
        found = tuple(
            binding for binding in self.bindings if binding.field == field_name
        )

        return found

    def binding_for(self, field_name: str, position: Tuple[Any, ...] = ()) -> Binding:
        """Return the binding at one selector position.

        Args:
            field_name: Parameter name.
            position: Position inside the parameter.

        Returns:
            The binding.

        Raises:
            ContractError: When no binding exists there.
        """
        for binding in self.bindings_for(field_name):
            if binding.position == tuple(position):
                return binding

        raise ContractError(
            f"{format_step_path(self.path)} has no binding at "
            f"{(field_name,) + tuple(position)!r}"
        )

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {
            "path": format_step_path(self.path),
            "type": self.block_type,
            "namespace": self.namespace,
            "invocation_axes": list(self.invocation_layout.axis_ids),
            "delivers_batches": self.delivers_batches,
            "bindings": [binding.describe() for binding in self.bindings],
            "gates": [gate.describe() for gate in self.gates],
            "control_targets": {
                key: [format_step_path(path) for path in paths]
                for key, paths in self.control_targets.items()
            },
            "outputs": {
                name: output.describe() for name, output in self.outputs.items()
            },
            "dependencies": [format_step_path(path) for path in self.dependencies],
            "domain": self.domain,
            "accepts_empty": self.spec.accepts_empty,
            "mutates": list(self.spec.mutates),
        }

        return description

    def _check_binding(self, binding: Binding, *, location: str) -> None:
        field_spec = self.spec.fields.get(binding.field)
        marker = field_spec.marker_at(binding.position) if field_spec else None
        if marker is None:
            raise ContractError(
                f"{location}: binding at {binding.field_path!r} is not a declared "
                "selector position"
            )
        if marker.role == "step":
            raise ContractError(
                f"{location}: {binding.field_path!r} is a control target; targets "
                "belong in control_targets, not bindings"
            )
        if binding.batch != marker.batch:
            raise ContractError(
                f"{location}: binding {binding.field_path!r} copies batch mode "
                f"{binding.batch!r}, but the declaration says {marker.batch!r}"
            )

        expected_modes = {
            "item": ("element", "ancestor", "constant"),
            "group": ("group", "constant_group"),
        }[marker.role]
        if binding.mode not in expected_modes:
            raise ContractError(
                f"{location}: {marker.role} leaf {binding.field_path!r} cannot use "
                f"mode {binding.mode!r}"
            )

        invocation_ids = self.invocation_layout.axis_ids
        source_ids = binding.source_layout.axis_ids
        valid = {
            "element": source_ids == invocation_ids,
            "ancestor": 0 < len(source_ids) < len(invocation_ids)
            and invocation_ids[: len(source_ids)] == source_ids,
            "constant": not source_ids,
            "group": len(source_ids) == len(invocation_ids) + 1
            and source_ids[:-1] == invocation_ids,
            "constant_group": not source_ids
            and binding.cast_layout is not None
            and len(binding.cast_layout.axis_ids) == len(invocation_ids) + 1
            and binding.cast_layout.axis_ids[:-1] == invocation_ids,
        }[binding.mode]
        if not valid:
            raise ContractError(
                f"{location}: binding {binding.field_path!r} in mode "
                f"{binding.mode!r} has source axes {list(source_ids)} but the step "
                f"runs over {list(invocation_ids)}"
            )
        if (
            self.spec.accepts_batches
            and binding.batch == "never"
            and binding.is_varying
        ):
            raise ContractError(
                f"{location}: {binding.field_path!r} is not batch-delivered in a "
                "batch-accepting block, so its binding must be constant (V1 rule)"
            )

    def _check_output(self, output: PlannedOutput, *, location: str) -> None:
        invocation_ids = self.invocation_layout.axis_ids
        output_ids = output.layout.axis_ids
        if output.transform == "same":
            valid = output_ids == invocation_ids
        elif output.transform == "expand":
            valid = (
                len(output_ids) == len(invocation_ids) + 1
                and output_ids[:-1] == invocation_ids
            )
        else:
            group_layouts = {
                binding.group_layout.axis_ids
                for binding in self.bindings_for(output.group_field or "")
                if binding.group_layout is not None
            }
            if len(group_layouts) > 1:
                raise ContractError(
                    f"{location}: output {output.name!r} preserves "
                    f"{output.group_field!r}, whose group leaves have different "
                    f"layouts {sorted(map(list, group_layouts))}"
                )
            valid = group_layouts == {output_ids}
        if not valid:
            raise ContractError(
                f"{location}: output {output.name!r} ({output.transform}) has axes "
                f"{list(output_ids)}, inconsistent with invocation axes "
                f"{list(invocation_ids)}"
            )
        # A source field written as a literal contributes no context.
        source_spec = self.spec.fields.get(output.source_field or "")
        if output.source_field is not None and (
            source_spec is None or source_spec.role not in ("item", "group")
        ):
            raise ContractError(
                f"{location}: output {output.name!r} takes context from "
                f"{output.source_field!r}, which is not a data field of the block"
            )


@dataclass(frozen=True)
class PlannedInput:
    """A root workflow input.

    Args:
        name: Input name (a selector segment).
        kinds: Accepted kind names.
        layout: Layout of the supplied value; empty for parameters. Advanced
            definitions may declare independent axes per input.
        required: Whether the caller must supply a value.
        default: Value used when an optional input is omitted.
        declared_type: Definition input type, e.g. ``"WorkflowImage"``.

    Raises:
        ContractError: When the name is not a selector segment.
    """

    name: str
    kinds: Tuple[str, ...]
    layout: EntryLayout
    required: bool = True
    default: Any = None
    declared_type: str = ""

    def __post_init__(self) -> None:
        if not is_selector_segment(self.name):
            raise ContractError(
                f"Workflow input name must use letters, digits, _ or -, got {self.name!r}"
            )

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {
            "kinds": list(self.kinds),
            "axes": list(self.layout.axis_ids),
            "required": self.required,
            "declared_type": self.declared_type,
        }

        return description


@dataclass(frozen=True)
class PlannedWorkflowOutput:
    """A declared workflow output.

    Args:
        name: Output name in results.
        selector: Selector text as written.
        source: Producing port; ``StepPort(step, "*")`` selects all outputs,
            each kept as its own result entry with its own layout.
        options: Declaration options passed to ``Kind.to_output`` (for
            example a coordinate system).
    """

    name: str
    selector: str
    source: Source
    options: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {
            "selector": self.selector,
            "source": _describe_source(self.source),
            "options": dict(self.options),
        }

        return description


@dataclass(frozen=True)
class PlannedSourceOutput:
    """Compiled declaration of one source port.

    Args:
        name: Port name.
        kinds: Declared kind names.
        layout: Layout of one emission of the port, with the source's local
            axis ids scoped as ``sources.<source>:<local id>``; empty for one
            payload per pulse.
    """

    name: str
    kinds: Tuple[str, ...]
    layout: EntryLayout = field(default_factory=EntryLayout)

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {"kinds": list(self.kinds), "axes": list(self.layout.axis_ids)}

        return description


@dataclass(frozen=True)
class PlannedSource:
    """One declared source of an active plan.

    Args:
        name: Source name, unique among the plan's sources.
        spec: Declaration of the source class.
        namespace: Catalogue namespace of the source, for resource keys.
        params: Validated parameters; selector positions still hold selectors.
        bindings: One ``constant`` binding per ``$inputs.<name>`` leaf, read
            from an ungrouped input before the source opens.
        outputs: Compiled ports by name.

    Raises:
        ContractError: When the name is not a selector segment or a binding
            is not a constant read of a workflow input.
    """

    name: str
    spec: SourceSpec
    namespace: str
    params: SourceParams
    bindings: Tuple[Binding, ...]
    outputs: Mapping[str, PlannedSourceOutput]

    def __post_init__(self) -> None:
        if not is_selector_segment(self.name):
            raise ContractError(
                f"Source name must use letters, digits, _ or -, got {self.name!r}"
            )
        object.__setattr__(self, "bindings", tuple(self.bindings))
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))
        for binding in self.bindings:
            if binding.mode != "constant" or not isinstance(binding.source, InputPort):
                raise ContractError(
                    f"$sources.{self.name}: binding {binding.field_path!r} must read "
                    f"an ungrouped workflow input, got {binding.mode!r} of "
                    f"{_describe_source(binding.source)}"
                )

    @property
    def step_path(self) -> StepPath:
        """Structured location ``("$sources", name)`` used by errors and contexts."""
        return source_step_path(self.name)

    def port(self, output: str) -> SourcePort:
        """Return the port consumers use to address one output.

        Args:
            output: Port name.

        Returns:
            The port.

        Raises:
            ContractError: When the source declares no such port.
        """
        if output not in self.outputs:
            raise ContractError(
                f"$sources.{self.name} has no output {output!r}; its outputs are "
                f"{list(self.outputs)}"
            )

        return SourcePort(source=self.name, output=output)

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {
            "type": self.spec.type,
            "namespace": self.namespace,
            "bindings": [binding.describe() for binding in self.bindings],
            "outputs": {
                name: output.describe() for name, output in self.outputs.items()
            },
        }

        return description


@dataclass(frozen=True)
class PlannedOutputGroup:
    """A named output group delivered once per pulse of its anchor's source.

    Args:
        name: Group name; the host registers a handler under it.
        anchor: Source port whose emission defines the group's pulse. The
            group is delivered when that port is present in the emission, or
            as a fully filtered outcome for an explicitly filtered emission.
        outputs: The selected fields, each in the anchor source's domain or
            static.
        dependencies: Steps that must have run before every field is
            terminal, in plan order: each field's producing step and the
            controllers gating a forwarded child output. Fields read from
            source ports, inputs or constants need no step. The runtime
            delivers the group as soon as these steps completed; earlier
            steps a producer itself depends on precede it in plan order.
    """

    name: str
    anchor: SourcePort
    outputs: Tuple[PlannedWorkflowOutput, ...]
    dependencies: Tuple[StepPath, ...] = ()

    def __post_init__(self) -> None:
        if not is_selector_segment(self.name):
            raise ContractError(
                f"Output group name must use letters, digits, _ or -, got {self.name!r}"
            )
        if not isinstance(self.anchor, SourcePort):
            raise ContractError(
                f"Output group {self.name!r} must be anchored on a source port, got "
                f"{_describe_source(self.anchor)}"
            )
        object.__setattr__(self, "outputs", tuple(self.outputs))
        names = [output.name for output in self.outputs]
        if len(names) != len(set(names)):
            raise ContractError(
                f"Output group {self.name!r} repeats field names {names}"
            )
        object.__setattr__(
            self, "dependencies", tuple(tuple(path) for path in self.dependencies)
        )

    @property
    def source(self) -> str:
        """Name of the anchor's source."""
        return self.anchor.source

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {
            "anchor": self.anchor.describe(),
            "outputs": {output.name: output.describe() for output in self.outputs},
            "dependencies": [format_step_path(path) for path in self.dependencies],
        }

        return description


@dataclass(frozen=True)
class PulseKey:
    """Identity of one emission of one source within one active run.

    Args:
        active_run_id: Identity of the active run (one ``start`` call).
        source: Declared source name.
        sequence: Emission number of that source in that run, from 0.

    Raises:
        ContractError: On an empty run id or source name, or a negative
            sequence.
    """

    active_run_id: str
    source: str
    sequence: int

    def __post_init__(self) -> None:
        for label, value in (
            ("active_run_id", self.active_run_id),
            ("source", self.source),
        ):
            if not isinstance(value, str) or not value:
                raise ContractError(f"PulseKey {label} must be a non-empty string")
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ContractError(
                f"PulseKey sequence must be a non-negative int, got {self.sequence!r}"
            )

    @property
    def lineage_id(self) -> str:
        """Lineage carried by the pulse's buffers: ``run:<run>/source:<name>``."""
        return f"run:{self.active_run_id}/source:{self.source}"

    @property
    def pulse_id(self) -> int:
        """Pulse number within the lineage (the sequence)."""
        return self.sequence

    @property
    def run_id(self) -> str:
        """Unique per-pulse invocation identity, ``<run>:<source>:<sequence>``."""
        return f"{self.active_run_id}:{self.source}:{self.sequence}"


@dataclass(frozen=True)
class CompileOptions:
    """Caller choices that affect compilation.

    Args:
        mutation_conflicts: ``warn`` records conflicting in-place mutations in
            ``CompiledWorkflow.warnings``; ``error`` rejects them.
        max_nested_depth: Maximum nested workflow depth (V1 default 4).
        max_nested_count: Maximum number of nested workflow steps (V1 default 32).
        allow_local_code: Whether dynamic blocks may execute submitted Python
            when a session is created and run.

    Raises:
        ContractError: On an unknown policy or a negative limit.
    """

    mutation_conflicts: Literal["warn", "error"] = "warn"
    max_nested_depth: int = 4
    max_nested_count: int = 32
    allow_local_code: bool = False

    def __post_init__(self) -> None:
        if self.mutation_conflicts not in ("warn", "error"):
            raise ContractError(
                "mutation_conflicts must be 'warn' or 'error', "
                f"got {self.mutation_conflicts!r}"
            )
        for name in ("max_nested_depth", "max_nested_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractError(f"{name} must be a non-negative int, got {value!r}")


@dataclass(frozen=True)
class CompiledWorkflow:
    """A validated, inspectable plan. Holds no block instances.

    Args:
        inputs: Workflow inputs by name.
        steps: Steps in execution order.
        outputs: Declared workflow outputs.
        catalogue: Catalogue the plan was compiled against.
        options: Options used for compilation.
        warnings: Non-fatal compile findings, e.g. mutation conflicts.
        child_inputs: Nested workflow inputs used by the plan.
        child_outputs: Gated nested workflow outputs that do not come from a
            child step (decision 026).
        sources: Declared sources by name, in declaration order. A plan with
            sources is active: it has ``output_groups`` instead of flat
            ``outputs``, every input is ungrouped, and it runs through
            ``ExecutionSession.start``.
        output_groups: Output groups of an active plan, in declaration order.

    Raises:
        ContractError: On duplicate step paths or child inputs, references to
            steps, outputs, inputs, child inputs, source ports or controllers
            that do not exist earlier in the plan, a binding or child input
            whose declared layout differs from its source's layout, cyclic
            child boundaries, child output gates that do not govern the
            child, an expanded axis identity claimed by two producers, flat
            outputs or grouped inputs beside sources, a step whose recorded
            domain differs from its derived one, or a step, gate or group
            joining two sources.
    """

    inputs: Mapping[str, PlannedInput]
    steps: Tuple[PlannedStep, ...]
    outputs: Tuple[PlannedWorkflowOutput, ...]
    catalogue: Catalogue
    options: CompileOptions = field(default_factory=CompileOptions)
    warnings: Tuple[str, ...] = ()
    child_inputs: Tuple[PlannedChildInput, ...] = ()
    child_outputs: Tuple[PlannedChildOutput, ...] = ()
    sources: Mapping[str, PlannedSource] = field(
        default_factory=lambda: MappingProxyType({})
    )
    output_groups: Tuple[PlannedOutputGroup, ...] = ()
    _axis_origins: Mapping[str, AxisOrigin] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "child_inputs", tuple(self.child_inputs))
        object.__setattr__(self, "child_outputs", tuple(self.child_outputs))
        object.__setattr__(self, "sources", MappingProxyType(dict(self.sources)))
        object.__setattr__(self, "output_groups", tuple(self.output_groups))
        _check_plan_references(self)
        _check_active_shape(self)
        object.__setattr__(self, "_axis_origins", _collect_axis_origins(self))

    @property
    def is_active(self) -> bool:
        """Whether the plan declares sources and runs through ``start``."""
        return bool(self.sources)

    def source(self, name: str) -> PlannedSource:
        """Return the declared source named ``name``.

        Args:
            name: Source name.

        Returns:
            The planned source.

        Raises:
            ContractError: When the plan declares no such source.
        """
        if name not in self.sources:
            raise ContractError(
                f"Plan has no source {name!r}; declared sources: {list(self.sources)}"
            )

        return self.sources[name]

    def source_port(self, port: SourcePort) -> PlannedSourceOutput:
        """Return the compiled port addressed by ``port``.

        Args:
            port: Source port used by a binding, group or anchor.

        Returns:
            The planned port.

        Raises:
            ContractError: When the plan has no such source or port.
        """
        planned = self.source(port.source)
        if port.output not in planned.outputs:
            raise ContractError(
                f"{port.describe()}: $sources.{port.source} has no output "
                f"{port.output!r}; its outputs are {list(planned.outputs)}"
            )

        return planned.outputs[port.output]

    def route(self, source_name: str) -> Tuple[PlannedStep, ...]:
        """Return the steps one pulse of ``source_name`` executes, in plan order.

        These are the steps whose domain is the source plus every static
        step (domain ``None``), which runs once per admitted pulse of every
        source. Nothing is pruned for output groups.

        Args:
            source_name: Declared source name.

        Returns:
            The steps to execute for one pulse.

        Raises:
            ContractError: When the plan declares no such source.
        """
        self.source(source_name)
        steps = tuple(step for step in self.steps if step.domain in (source_name, None))

        return steps

    def groups_of(self, source_name: str) -> Tuple[PlannedOutputGroup, ...]:
        """Return the output groups anchored on ``source_name``, in order.

        Args:
            source_name: Declared source name.

        Returns:
            The groups delivered from that source's pulses.

        Raises:
            ContractError: When the plan declares no such source.
        """
        self.source(source_name)
        groups = tuple(
            group for group in self.output_groups if group.source == source_name
        )

        return groups

    def step(self, path: StepPath) -> PlannedStep:
        """Return the step with the given path.

        Args:
            path: Step path.

        Returns:
            The planned step.

        Raises:
            ContractError: When no such step exists.
        """
        for step in self.steps:
            if step.path == tuple(path):
                return step

        raise ContractError(f"Plan has no step {format_step_path(tuple(path))}")

    def child_input(self, port: ChildInputPort) -> PlannedChildInput:
        """Return the planned child input addressed by ``port``.

        Args:
            port: Child input port used by a binding, output or child input.

        Returns:
            The planned child input.

        Raises:
            ContractError: When the plan has no such child input.
        """
        for child_input in self.child_inputs:
            if child_input.scope == tuple(port.scope) and child_input.name == port.name:
                return child_input

        raise ContractError(f"Plan has no child input {port.describe()}")

    def child_output(self, port: ChildOutputPort) -> PlannedChildOutput:
        """Return the planned child output addressed by ``port``.

        Args:
            port: Child output port used by a binding, output or child input.

        Returns:
            The planned child output.

        Raises:
            ContractError: When the plan has no such child output.
        """
        for child_output in self.child_outputs:
            if (
                child_output.scope == tuple(port.scope)
                and child_output.name == port.name
            ):
                return child_output

        raise ContractError(f"Plan has no child output {port.describe()}")

    def origin(self, source: Source) -> Source:
        """Follow child input and output ports to the value behind them.

        Args:
            source: Any plan source.

        Returns:
            The workflow input, step output or constant holding the value.
        """
        while isinstance(source, (ChildInputPort, ChildOutputPort)):
            source = self._boundary(source).source

        return source

    def _boundary(
        self, port: BoundaryPort
    ) -> Union[PlannedChildInput, PlannedChildOutput]:
        if isinstance(port, ChildInputPort):
            return self.child_input(port)

        return self.child_output(port)

    def axis_origin(self, axis_id: str) -> AxisOrigin:
        """Return where an axis comes from.

        Output rows are built per element of workflow-input axes; outputs whose
        outermost axis was generated by a step (for example an input-free
        source) are copied whole into every row, as in V1. Independent
        generated axes never align by size.

        Args:
            axis_id: Axis identity.

        Returns:
            The input, expanding output or cast that introduces the axis.

        Raises:
            ContractError: When nothing in the plan introduces the axis.
        """
        if axis_id not in self._axis_origins:
            raise ContractError(f"Plan has no axis {axis_id!r}")

        return self._axis_origins[axis_id]

    def describe(self) -> Dict[str, Any]:
        """Describe the plan without constructing blocks.

        Returns:
            JSON-friendly structure with inputs, steps in order, outputs, axis
            origins and warnings.
        """
        description = {
            "inputs": {name: item.describe() for name, item in self.inputs.items()},
            "sources": {name: item.describe() for name, item in self.sources.items()},
            "steps": [step.describe() for step in self.steps],
            "outputs": {output.name: output.describe() for output in self.outputs},
            "output_groups": {
                group.name: group.describe() for group in self.output_groups
            },
            "child_inputs": [item.describe() for item in self.child_inputs],
            "child_outputs": [item.describe() for item in self.child_outputs],
            "axis_origins": {
                axis_id: origin.describe()
                for axis_id, origin in self._axis_origins.items()
            },
            "warnings": list(self.warnings),
        }

        return description

    def create_session(
        self,
        resources: Optional[Mapping[str, Any]] = None,
        *,
        observer: Optional["ExecutionObserver"] = None,
        error_handler: Optional["ErrorHandler"] = None,
    ) -> "ExecutionSession":
        """Resolve resources and construct every step once.

        Args:
            resources: Caller resources keyed by ``name`` or
                ``namespace.name``; values are passed without copying.
            observer: Receives workflow and step notifications.
            error_handler: Called with each ``StepExecutionError`` before it
                is raised.

        Returns:
            A session whose runs share these block instances.

        Raises:
            ResourceError: When a resource is missing or a constructor fails.
        """
        session = create_session(
            self,
            resources=resources,
            observer=observer,
            error_handler=error_handler,
        )

        return session


def _check_plan_references(plan: CompiledWorkflow) -> None:
    all_steps = {step.path: step for step in plan.steps}
    boundaries = _check_boundaries(plan, steps=all_steps)

    earlier: Dict[StepPath, PlannedStep] = {}
    for step in plan.steps:
        location = format_step_path(step.path)
        if step.path in earlier:
            raise ContractError(f"Plan contains {location} twice")
        for binding in step.bindings:
            _check_source(
                binding.source,
                layout=binding.source_layout,
                plan=plan,
                steps=earlier,
                boundaries=boundaries,
                location=f"{location} parameter {'.'.join(map(str, binding.field_path))}",
            )
        for gate in step.gates:
            _check_gate(gate, governed=step.path, steps=earlier, location=location)
        for paths in step.control_targets.values():
            for path in paths:
                if path not in all_steps or path in earlier or path == step.path:
                    raise ContractError(
                        f"{location}: control target {format_step_path(path)} must "
                        "be a later step of the plan"
                    )
        earlier[step.path] = step

    for output in plan.outputs:
        _check_source(
            output.source,
            layout=None,
            plan=plan,
            steps=earlier,
            boundaries=boundaries,
            location=f"output {output.name!r}",
        )
    for group in plan.output_groups:
        _check_source(
            group.anchor,
            layout=None,
            plan=plan,
            steps=earlier,
            boundaries=boundaries,
            location=f"output group {group.name!r} anchor",
        )
        for output in group.outputs:
            _check_source(
                output.source,
                layout=None,
                plan=plan,
                steps=earlier,
                boundaries=boundaries,
                location=f"output group {group.name!r} field {output.name!r}",
            )


def _check_active_shape(plan: CompiledWorkflow) -> None:
    """Check the rules of an active plan and every step's recorded domain."""
    names = [group.name for group in plan.output_groups]
    if len(names) != len(set(names)):
        raise ContractError(f"Plan repeats output group names {names}")
    if not plan.is_active:
        if plan.output_groups:
            raise ContractError("Output groups need declared sources")
        for step in plan.steps:
            if step.domain is not None:
                raise ContractError(
                    f"{format_step_path(step.path)} records domain {step.domain!r}, "
                    "but the plan declares no sources"
                )
        return

    if plan.outputs:
        raise ContractError(
            "A plan with sources uses output groups; flat outputs "
            f"{[output.name for output in plan.outputs]} cannot be delivered per pulse"
        )
    grouped = [name for name, item in plan.inputs.items() if item.layout.depth]
    if grouped:
        raise ContractError(
            f"A plan with sources accepts only ungrouped inputs, got {grouped}"
        )
    for name, planned_source in plan.sources.items():
        _check_source_record(name, planned_source, inputs=plan.inputs)

    steps = {step.path: step for step in plan.steps}
    boundaries = {item.port: item for item in plan.child_inputs + plan.child_outputs}
    for step in plan.steps:
        location = format_step_path(step.path)
        try:
            derived = derive_domain(
                [binding.source for binding in step.bindings],
                controllers=[gate.controller for gate in step.gates],
                steps=steps,
                boundaries=boundaries,
            )
        except ContractError as error:
            raise ContractError(f"{location}: {error}") from error
        if derived != step.domain:
            raise ContractError(
                f"{location} records domain {step.domain!r}, but its data, gates "
                f"and boundaries derive {derived!r}"
            )
    for group in plan.output_groups:
        for output in group.outputs:
            try:
                derived = derive_domain(
                    [output.source], controllers=(), steps=steps, boundaries=boundaries
                )
            except ContractError as error:
                raise ContractError(
                    f"output group {group.name!r} field {output.name!r}: {error}"
                ) from error
            if derived not in (None, group.source):
                raise ContractError(
                    f"output group {group.name!r} is anchored on "
                    f"{group.anchor.describe()}, but field {output.name!r} "
                    f"({output.selector}) comes from source {derived!r}"
                )
        expected = derive_dependencies(
            [output.source for output in group.outputs],
            steps=steps,
            boundaries=boundaries,
        )
        if group.dependencies != expected:
            raise ContractError(
                f"output group {group.name!r} records dependencies "
                f"{[format_step_path(path) for path in group.dependencies]}, but its "
                f"fields need {[format_step_path(path) for path in expected]}"
            )


def _check_source_record(
    name: str, planned: PlannedSource, *, inputs: Mapping[str, PlannedInput]
) -> None:
    """Check a planned source against its declaration and the plan's inputs."""
    location = f"$sources.{name}"
    if planned.name != name:
        raise ContractError(
            f"{location} is recorded under key {name!r} but names itself "
            f"{planned.name!r}"
        )

    expected_bindings = {
        (use.field, use.position): use
        for use in planned.spec.find_selectors(planned.params)
    }
    recorded = [(binding.field, binding.position) for binding in planned.bindings]
    if sorted(recorded) != sorted(expected_bindings):
        raise ContractError(
            f"{location} records bindings at {sorted(recorded)}, but its parameters "
            f"select exactly once at {sorted(expected_bindings)}"
        )
    for binding in planned.bindings:
        use = expected_bindings[(binding.field, binding.position)]
        where = f"{location} parameter {'.'.join(map(str, binding.field_path))}"
        if binding.selector != use.selector or binding.batch != use.marker.batch:
            raise ContractError(
                f"{where} records selector {binding.selector!r}, but the parameters "
                f"hold {use.selector!r}"
            )
        parsed = parse_selector(use.selector)
        if parsed.target != "input":
            raise ContractError(
                f"{where} must select a static workflow input, got {use.selector!r}"
            )
        selected = InputPort(name=parsed.name)
        if binding.source != selected:
            raise ContractError(
                f"{where} selects {use.selector!r} but reads "
                f"{_describe_source(binding.source)}"
            )
        planned_input = inputs.get(binding.source.name)
        if planned_input is None:
            raise ContractError(
                f"{where} reads unknown workflow input {binding.source.name!r}"
            )
        if planned_input.layout.depth or binding.source_layout.depth:
            raise ContractError(
                f"{where} must read an ungrouped input, but $inputs."
                f"{binding.source.name} has axes {list(planned_input.layout.axis_ids)}"
            )
        if not kinds_compatible(planned_input.kinds, use.marker.kind_names):
            raise ContractError(
                f"{where} accepts kinds {list(use.marker.kind_names)}, but "
                f"$inputs.{binding.source.name} provides {list(planned_input.kinds)}"
            )

    declared = planned.spec.outputs
    if list(planned.outputs) != list(declared):
        raise ContractError(
            f"{location} records outputs {list(planned.outputs)}, but "
            f"{planned.spec.type} declares {list(declared)}"
        )
    for port_name, port in planned.outputs.items():
        expected = PlannedSourceOutput(
            name=port_name,
            kinds=declared[port_name].kind_names,
            layout=scoped_layout(name, declared[port_name].layout),
        )
        if port != expected:
            raise ContractError(
                f"{location} output {port_name!r} records {port.describe()}, but the "
                f"declaration scoped to this source is {expected.describe()}"
            )


def scoped_layout(source_name: str, layout: EntryLayout) -> EntryLayout:
    """Scope a source-local layout to the plan: ``sources.<name>:<local id>``.

    Args:
        source_name: Declared source name.
        layout: Layout as declared by the source class.

    Returns:
        The same axes with plan-unique ids.
    """
    axes = tuple(
        Axis(
            id=f"sources.{source_name}:{axis.id}",
            kind=axis.kind,
            stationary=axis.stationary,
        )
        for axis in layout.axes
    )
    scoped = EntryLayout(axes=axes)

    return scoped


def derive_dependencies(
    sources: Iterable[Source],
    *,
    steps: Mapping[StepPath, PlannedStep],
    boundaries: Mapping[BoundaryPort, "Boundary"],
) -> Tuple[StepPath, ...]:
    """Derive the steps that must complete before every ``sources`` value is terminal.

    A step output (a wildcard included) needs its producing step; a child
    input needs whatever its source needs; a gated child output additionally
    needs every controller of its gates. Inputs, source ports and constants
    need no step. Steps a producer depends on itself precede it in plan
    order, so they are not repeated.

    Args:
        sources: Value sources of the selected fields.
        steps: Planned steps in plan order, by path.
        boundaries: Planned child inputs and outputs, by port.

    Returns:
        The required step paths in plan order, without repetition.
    """
    required = set()

    def visit(source: Source) -> None:
        if isinstance(source, StepPort):
            required.add(source.step)
        elif isinstance(source, (ChildInputPort, ChildOutputPort)):
            boundary = boundaries[source]
            visit(boundary.source)
            if isinstance(boundary, PlannedChildOutput):
                required.update(gate.controller for gate in boundary.gates)

    for source in sources:
        visit(source)
    ordered = tuple(path for path in steps if path in required)

    return ordered


def derive_domain(
    sources: Iterable[Source],
    *,
    controllers: Iterable[StepPath],
    steps: Mapping[StepPath, PlannedStep],
    boundaries: Mapping[BoundaryPort, "Boundary"],
) -> Optional[str]:
    """Derive the one source a consumer depends on, or ``None`` for static.

    A source port contributes its source; a workflow input or constant
    contributes nothing; a step output contributes the producing step's
    domain; a child input contributes its source's domain; a gated child
    output contributes its source's domain and its controllers' domains; a
    controller contributes its own domain.

    Args:
        sources: Value sources the consumer reads (bindings, an output).
        controllers: Control steps whose gates govern the consumer.
        steps: Planned steps that may be referenced, by path.
        boundaries: Planned child inputs and outputs, by port.

    Returns:
        The source name, or ``None`` when nothing source-derived is read.

    Raises:
        ContractError: When two different sources are reached; no alignment
            between independent sources exists in this engine.
    """
    reached: Dict[str, str] = {}

    def visit_step(path: StepPath, *, via: str) -> None:
        domain = steps[path].domain
        if domain is not None:
            reached.setdefault(domain, via)

    def visit(source: Source, *, via: str) -> None:
        if isinstance(source, SourcePort):
            reached.setdefault(source.source, via)
        elif isinstance(source, StepPort):
            visit_step(source.step, via=via)
        elif isinstance(source, (ChildInputPort, ChildOutputPort)):
            boundary = boundaries[source]
            visit(boundary.source, via=via)
            if isinstance(boundary, PlannedChildOutput):
                for gate in boundary.gates:
                    visit_step(gate.controller, via=via)

    for source in sources:
        visit(source, via=_describe_source(source))
    for controller in controllers:
        visit_step(controller, via=f"gate of {format_step_path(controller)}")

    if len(reached) > 1:
        described = "; ".join(
            f"{name!r} via {via}" for name, via in sorted(reached.items())
        )
        raise ContractError(
            f"joins independent sources {sorted(reached)} ({described}); sources "
            "correspond only through an explicit alignment, never by pulse "
            "order, timestamps or shape"
        )

    domain = next(iter(reached), None)

    return domain


Boundary = Union[PlannedChildInput, PlannedChildOutput]


def _check_boundaries(
    plan: CompiledWorkflow, *, steps: Mapping[StepPath, PlannedStep]
) -> Dict[BoundaryPort, Boundary]:
    """Check child inputs and outputs; they may read each other, never in a cycle."""
    boundaries: Dict[BoundaryPort, Boundary] = {}
    for item in plan.child_inputs + plan.child_outputs:
        if item.port in boundaries:
            raise ContractError(f"Plan contains {item.port.describe()} twice")
        boundaries[item.port] = item

    for port, item in boundaries.items():
        location = port.describe()
        seen = {port}
        source = item.source
        while isinstance(source, (ChildInputPort, ChildOutputPort)):
            if source in seen:
                raise ContractError(
                    f"{location}: child boundaries read each other in a cycle"
                )
            seen.add(source)
            source = boundaries[source].source if source in boundaries else None

        if isinstance(item, PlannedChildInput):
            listed = [other.port for other in plan.child_inputs]
            if isinstance(item.source, ChildInputPort) and item.source in listed:
                if listed.index(item.source) > listed.index(port):
                    raise ContractError(
                        f"{location}: reads the later-listed child input "
                        f"{item.source.describe()}; list outer child inputs first"
                    )
            _check_source(
                item.source,
                layout=item.layout,
                plan=plan,
                steps=steps,
                boundaries=boundaries,
                location=location,
            )
            continue

        _check_source(
            item.source,
            layout=None,
            plan=plan,
            steps=steps,
            boundaries=boundaries,
            location=location,
        )
        for gate in item.gates:
            _check_gate(gate, governed=item.scope, steps=steps, location=location)
        layouts = [
            _source_layout(item.source, plan=plan, steps=steps, boundaries=boundaries)
        ]
        layouts += [gate.controller_layout for gate in item.gates]
        deepest = max((layout for layout in layouts if layout is not None), key=_depth)
        if item.layout.axis_ids != deepest.axis_ids or not all(
            layout is None or deepest.axis_ids[: layout.depth] == layout.axis_ids
            for layout in layouts
        ):
            raise ContractError(
                f"{location}: layout {list(item.layout.axis_ids)} must be the deepest "
                f"of its source and gate layouts "
                f"{[list(layout.axis_ids) for layout in layouts if layout is not None]}, "
                "each a prefix of it"
            )

    return boundaries


def _check_gate(
    gate: Gate,
    *,
    governed: StepPath,
    steps: Mapping[StepPath, PlannedStep],
    location: str,
) -> None:
    """Check a gate on a step (its path) or on a whole child (its scope)."""
    controller = steps.get(gate.controller)
    if controller is None or not controller.spec.is_control:
        raise ContractError(
            f"{location}: gate controller {format_step_path(gate.controller)} "
            "is not an earlier control step"
        )
    targets = controller.control_targets.get(gate.target, ())
    if not any(path[: len(governed)] == governed for path in targets):
        raise ContractError(
            f"{location}: gate target {gate.target!r} of "
            f"{format_step_path(gate.controller)} does not govern it"
        )
    if gate.controller_layout != controller.invocation_layout:
        raise ContractError(
            f"{location}: gate layout differs from its controller's layout"
        )


def _check_source(
    source: Source,
    *,
    layout: Optional[EntryLayout],
    plan: CompiledWorkflow,
    steps: Mapping[StepPath, PlannedStep],
    boundaries: Mapping[BoundaryPort, Boundary],
    location: str,
) -> None:
    """Check that ``source`` exists, is ready in time and, when given, has ``layout``.

    A child port is ready once the step behind it and every controller of a
    child output on the way have run.
    """
    try:
        actual = _source_layout(source, plan=plan, steps=steps, boundaries=boundaries)
    except ContractError as error:
        raise ContractError(f"{location}: {error}") from error
    required = []
    reached = source
    while isinstance(reached, (ChildInputPort, ChildOutputPort)):
        item = boundaries[reached]
        if isinstance(item, PlannedChildOutput):
            required.extend(gate.controller for gate in item.gates)
        reached = item.source
    if isinstance(reached, StepPort):
        required.append(reached.step)
    for path in required:
        if path not in steps:
            raise ContractError(
                f"{location}: needs {format_step_path(path)}, which is not an "
                "earlier step"
            )

    if layout is not None and actual is not None and layout.axis_ids != actual.axis_ids:
        raise ContractError(
            f"{location}: declared source axes {list(layout.axis_ids)} differ from "
            f"the axes {list(actual.axis_ids)} of {_describe_source(source)}"
        )


def _source_layout(
    source: Source,
    *,
    plan: CompiledWorkflow,
    steps: Mapping[StepPath, PlannedStep],
    boundaries: Mapping[BoundaryPort, Boundary],
) -> Optional[EntryLayout]:
    """Layout of an existing source; ``None`` for a wildcard over a step's outputs."""
    if isinstance(source, Constant):
        return EntryLayout()
    if isinstance(source, InputPort):
        if source.name not in plan.inputs:
            raise ContractError(f"unknown workflow input {source.name!r}")
        return plan.inputs[source.name].layout
    if isinstance(source, SourcePort):
        return plan.source_port(source).layout
    if isinstance(source, (ChildInputPort, ChildOutputPort)):
        if source not in boundaries:
            raise ContractError(f"unknown {source.describe()}")
        return boundaries[source].layout

    producer = steps.get(source.step)
    if producer is None:
        raise ContractError(f"source {source.describe()} is not an earlier step")
    if source.output == "*":
        return None
    if source.output not in producer.outputs:
        raise ContractError(
            f"{format_step_path(source.step)} has no output {source.output!r}"
        )

    layout = producer.outputs[source.output].layout

    return layout


def _depth(layout: EntryLayout) -> int:
    return layout.depth


def _collect_axis_origins(plan: CompiledWorkflow) -> Mapping[str, AxisOrigin]:
    origins: Dict[str, AxisOrigin] = {}
    for item in plan.inputs.values():
        for axis_id in item.layout.axis_ids:
            origins.setdefault(axis_id, AxisOrigin(kind="input", name=item.name))
    for planned_source in plan.sources.values():
        for output in planned_source.outputs.values():
            for axis_id in output.layout.axis_ids:
                origins.setdefault(
                    axis_id, AxisOrigin(kind="source", name=planned_source.name)
                )

    for step in plan.steps:
        created = [
            (output.layout.axis_ids[-1], AxisOrigin("expand", output.name, step.path))
            for output in step.outputs.values()
            if output.transform == "expand"
        ]
        created.extend(
            (
                binding.cast_layout.axis_ids[-1],
                AxisOrigin("cast", binding.field, step.path),
            )
            for binding in step.bindings
            if binding.cast_layout is not None
        )
        for axis_id, origin in created:
            known = origins.setdefault(axis_id, origin)
            if known.step != step.path:
                raise ContractError(
                    f"{format_step_path(step.path)}: axis {axis_id!r} is already "
                    f"introduced by {known.describe()}; each producer needs its own "
                    "axis identity"
                )

    frozen_origins = MappingProxyType(origins)

    return frozen_origins


class ExecutionObserver:
    """No-op observer; subclass and override the callbacks you need.

    All callbacks are keyword-only. Block arguments are passed as one mapping,
    so a parameter named like a callback argument cannot collide with it.
    """

    def on_run_started(self, *, session_id: str, run_id: str) -> None:
        """A run of the session started."""

    def on_step_started(self, *, step: StepPath, block_type: str) -> None:
        """A step is about to be processed."""

    def on_invocation(
        self,
        *,
        step: StepPath,
        index: Optional[Index],
        arguments: Mapping[str, Any],
        result: Any,
    ) -> None:
        """A block call returned; ``index`` is ``None`` for a batch-delivering call."""

    def on_invocation_skipped(
        self, *, step: StepPath, index: Index, reason: SkipReason
    ) -> None:
        """An invocation was not executed, and why."""

    def on_step_finished(
        self, *, step: StepPath, invocations: int, skipped: int
    ) -> None:
        """A step finished all its invocations."""

    def on_error(self, *, error: StepExecutionError) -> None:
        """A step failed; the error is raised after the error handler runs."""

    def on_run_finished(
        self,
        *,
        run_id: str,
        result: Optional["RunResult"],
        error: Optional[BaseException],
    ) -> None:
        """A run ended with a result or an error."""

    def on_source_opened(self, *, source: str) -> None:
        """A source of an active run finished ``open``."""

    def on_source_closed(self, *, source: str, error: Optional[BaseException]) -> None:
        """A source of an active run was closed; ``error`` is what ``close`` raised."""

    def on_pulse_started(self, *, run_id: str, source: str, pulse: PulseKey) -> None:
        """One admitted emission starts executing; ``run_id`` is ``pulse.run_id``."""

    def on_pulse_finished(
        self,
        *,
        run_id: str,
        source: str,
        pulse: PulseKey,
        error: Optional[BaseException],
    ) -> None:
        """One pulse finished its steps and deliveries, or failed with ``error``."""

    def on_group_delivered(
        self, *, run_id: str, group: str, source: str, pulse: PulseKey
    ) -> None:
        """A registered handler returned for one group of one pulse."""


ErrorHandler = Callable[[StepExecutionError], None]

NULL_OBSERVER = ExecutionObserver()


class ExecutionSession:
    """Block instances of one plan, reused by every run of the session.

    Create sessions with ``CompiledWorkflow.create_session``.

    Args:
        plan: The compiled plan.
        instances: One constructed block per step path.
        resources: Resources chosen per step, for inspection.
        observer: Observer receiving run notifications.
        error_handler: Optional callback for step errors.
        session_id: Identity used while the instances were constructed; a new
            one is generated when omitted.
        source_resources: Constructor resources chosen per declared source.
            Source instances are not held here: every ``start`` constructs
            fresh ones from these values.
    """

    def __init__(
        self,
        *,
        plan: CompiledWorkflow,
        instances: Mapping[StepPath, Block],
        resources: Mapping[StepPath, Mapping[str, ResolvedResource]],
        observer: ExecutionObserver = NULL_OBSERVER,
        error_handler: Optional[ErrorHandler] = None,
        session_id: Optional[str] = None,
        source_resources: Optional[Mapping[str, Mapping[str, ResolvedResource]]] = None,
    ):
        self.plan = plan
        self.instances = MappingProxyType(dict(instances))
        self.resources = MappingProxyType(dict(resources))
        self.observer = observer
        self.error_handler = error_handler
        self.session_id = session_id if session_id is not None else uuid.uuid4().hex
        self.source_resources = MappingProxyType(dict(source_resources or {}))

    def run(self, inputs: Mapping[str, Any]) -> "RunResult":
        """Execute the plan once with this session's block instances.

        Args:
            inputs: Workflow input values by name.

        Returns:
            The run result.

        Raises:
            WorkflowInputError: When the plan declares sources; such a plan is
                driven by ``start``.
        """
        if self.plan.is_active:
            raise WorkflowInputError(
                f"The plan declares sources {list(self.plan.sources)}; run it with "
                "session.start(...) instead of session.run(...)"
            )

        execution = importlib.import_module(EXECUTION_MODULE)
        result = execution.run_session(self, inputs=inputs)

        return result

    def start(
        self,
        inputs: Optional[Mapping[str, Any]] = None,
        *,
        handlers: Optional[Mapping[str, Callable[[Any], None]]] = None,
        admission_bound: int = 2,
    ) -> Any:
        """Open the declared sources and process their pulses until they end.

        Static inputs and handlers are validated before any source opens.
        The returned run is driven by the active runtime module; see its
        ``ActiveRun`` for ``stop``, ``wait`` and failure attribution. A fast
        source may deliver a group before this call returns, so a handler
        that wants to stop the run calls ``session.stop()`` rather than a
        run handle it may not hold yet.

        Args:
            inputs: Values of the plan's ungrouped inputs (static
                configuration); omitted optional inputs take their defaults.
            handlers: Output group name to a synchronous callable receiving
                that group's ``GroupResult`` once per delivered pulse. Groups
                without a handler are not built.
            admission_bound: Emissions of one source admitted for processing
                at a time; a reader waits beyond it (bounded backpressure).

        Returns:
            The active run.

        Raises:
            WorkflowInputError: When the plan declares no sources, an input or
                handler is invalid, or a run of this session is still active.
        """
        if not self.plan.is_active:
            raise WorkflowInputError(
                "The plan declares no sources; run it with session.run(inputs) "
                "instead of session.start(...)"
            )

        runtime = importlib.import_module(ACTIVE_RUNTIME_MODULE)
        run = runtime.start_session(
            self,
            inputs=inputs if inputs is not None else {},
            handlers=handlers if handlers is not None else {},
            admission_bound=admission_bound,
        )

        return run

    def stop(self) -> None:
        """Request a graceful stop of this session's current active run.

        Targets the run registered at the time of the call: admission closes,
        admitted pulses and their handlers drain, sources close. The call
        never waits and is safe inside a group handler, also one that runs
        before ``start`` has returned. Without an unfinished run it is a
        no-op; it never cancels a later ``start``. A caller holding a
        particular ``ActiveRun`` uses that run's ``stop`` instead.

        Raises:
            WorkflowInputError: When the plan declares no sources.
        """
        if not self.plan.is_active:
            raise WorkflowInputError(
                "The plan declares no sources, so the session has no active run "
                "to stop"
            )

        runtime = importlib.import_module(ACTIVE_RUNTIME_MODULE)
        runtime.stop_session(self)


def create_session(
    plan: CompiledWorkflow,
    *,
    resources: Optional[Mapping[str, Any]] = None,
    observer: Optional[ExecutionObserver] = None,
    error_handler: Optional[ErrorHandler] = None,
) -> ExecutionSession:
    """Resolve resources and construct every step of ``plan`` once.

    Args:
        plan: Compiled plan.
        resources: Caller resources keyed by ``name`` or ``namespace.name``.
        observer: Receives run notifications; defaults to a no-op observer.
        error_handler: Called with each ``StepExecutionError`` before raising.

    Each constructor runs inside an ``ExecutionContext`` with the session id
    the session keeps and ``run_id=None``, so a block reading
    ``self.execution_context`` in ``__init__`` sees the same session as its
    later runs.

    The resources of every declared source are resolved here too, so a
    missing source resource fails before any run, but no source is
    constructed: ``start`` builds fresh source instances for every active run.

    Returns:
        The execution session.

    Raises:
        ResourceError: When a resource is missing, a factory fails or a
            constructor raises.
    """
    session_id = uuid.uuid4().hex
    resolver = ResourceResolver(provided=resources, providers=plan.catalogue.providers)
    source_resources: Dict[str, Mapping[str, ResolvedResource]] = {}
    for planned_source in plan.sources.values():
        resolved_for_source = resolver.resolve(
            planned_source.spec.resources,
            namespace=planned_source.namespace,
            step_path=planned_source.step_path,
            block_type=planned_source.spec.type,
        )
        source_resources[planned_source.name] = MappingProxyType(resolved_for_source)

    instances: Dict[StepPath, Block] = {}
    chosen: Dict[StepPath, Mapping[str, ResolvedResource]] = {}
    for step in plan.steps:
        resolved = resolver.resolve(
            step.spec.resources,
            namespace=step.namespace,
            step_path=step.path,
            block_type=step.block_type,
        )
        arguments = {name: item.value for name, item in resolved.items()}
        context = ExecutionContext(
            step_path=step.path, block_type=step.block_type, session_id=session_id
        )
        try:
            with use_execution_context(context):
                instance = step.spec.block_class(**arguments)
        except Exception as error:
            raise ResourceError(
                f"constructor of {step.spec.block_class.__qualname__} failed: "
                f"{type(error).__name__}: {error}",
                step_path=step.path,
                block_type=step.block_type,
            ) from error
        instances[step.path] = instance
        chosen[step.path] = MappingProxyType(resolved)

    session = ExecutionSession(
        plan=plan,
        instances=instances,
        resources=chosen,
        observer=observer if observer is not None else NULL_OBSERVER,
        error_handler=error_handler,
        session_id=session_id,
        source_resources=source_resources,
    )

    return session


@dataclass(frozen=True)
class RunResult:
    """Outcome of one run.

    Every selected port is one entry of ``outputs`` with its own layout and
    metadata. A plain workflow output selects one port; a wildcard output
    (``$steps.x.*``) selects one entry per step output, so ports with
    different layouts are never forced into one entry.

    Args:
        outputs: Buffer of selected port entries, keyed by entry key. Entries
            filtered as a whole are absent.
        selections: Workflow output name to ``{port selector: entry key}``.
        statuses: ``complete`` or ``filtered`` per entry key.
        filtered_paths: Minimal filtered logical index paths per entry key;
            ``()`` means the whole entry was filtered.
        plan: The plan that produced the result.
        session_id: Session that produced the result.
        run_id: Identity of this run.
        trace: Ordered JSON-friendly execution events, when recorded.
        input_row_count: Known row count of selected input-axis entries, kept
            even when all their payloads are filtered.

    Raises:
        ContractError: When selections, statuses and buffer entries disagree.
    """

    outputs: WorkflowsBuffer
    selections: Mapping[str, Mapping[str, str]]
    statuses: Mapping[str, OutputStatus]
    filtered_paths: Mapping[str, Tuple[Index, ...]]
    plan: CompiledWorkflow
    session_id: str
    run_id: str
    trace: Tuple[Mapping[str, Any], ...] = ()
    input_row_count: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.input_row_count, int)
            or isinstance(self.input_row_count, bool)
            or self.input_row_count < 0
        ):
            raise ContractError(
                "RunResult input_row_count must be a non-negative integer"
            )

        entry_keys = [
            key for ports in self.selections.values() for key in ports.values()
        ]
        if len(entry_keys) != len(set(entry_keys)):
            raise ContractError(f"RunResult entry keys repeat: {entry_keys}")
        if set(self.statuses) != set(entry_keys):
            raise ContractError(
                f"RunResult statuses {sorted(self.statuses)} must cover exactly the "
                f"selected entries {sorted(entry_keys)}"
            )
        complete = {
            key for key, status in self.statuses.items() if status == "complete"
        }
        if set(self.outputs.entry_names) != complete:
            raise ContractError(
                f"RunResult buffer entries {sorted(self.outputs.entry_names)} must be "
                f"exactly the complete entries {sorted(complete)}"
            )
        object.__setattr__(self, "trace", tuple(self.trace))

    def rows(self, *, serialize: bool = False) -> List[Dict[str, Any]]:
        """Build V1-shaped output rows from this result.

        Args:
            serialize: Pass values through their kinds' serializers.

        Returns:
            One mapping per element of the top-level workflow-input axis;
            filtered positions are ``None``.
        """
        execution = importlib.import_module(EXECUTION_MODULE)
        rows = execution.build_rows(self, serialize=serialize)

        return rows


def resolve_futures(value: Any) -> Any:
    """Wait for ``concurrent.futures.Future`` objects inside a block result.

    Looks at the value itself, mapping values, list and tuple items and
    ``Batch`` contents, recursively. Containers without futures are returned
    as the same objects; payloads are never copied.

    Args:
        value: A block result or part of one.

    Returns:
        ``value`` with every future replaced by its result.

    Raises:
        Exception: Whatever a future raised.
    """
    if isinstance(value, Future):
        resolved = resolve_futures(value.result())
        return resolved
    if isinstance(value, Batch):
        content = [resolve_futures(item) for item in value.content]
        if all(new is old for new, old in zip(content, value.content)):
            return value
        rebuilt = Batch(
            content,
            indices=value.indices,
            layout=value.layout,
            metadata=value.metadata,
            parent_index=value.parent_index,
        )
        return rebuilt
    if isinstance(value, Mapping):
        items = {key: resolve_futures(item) for key, item in value.items()}
        if all(items[key] is item for key, item in value.items()):
            return value
        rebuilt_mapping = type(value)(items) if isinstance(value, dict) else items
        return rebuilt_mapping
    if isinstance(value, (list, tuple)):
        items = [resolve_futures(item) for item in value]
        if all(new is old for new, old in zip(items, value)):
            return value
        rebuilt_sequence = type(value)(items)
        return rebuilt_sequence

    return value


def _describe_source(source: Source) -> Any:
    described = source.describe()

    return described
