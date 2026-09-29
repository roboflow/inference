"""Compiler for the passive Workflows V2 execution path.

The compiler turns a strict, versioned definition mapping into an immutable
``CompiledWorkflow``. It resolves selectors, checks kinds and grouping
lineage per port, derives per-output layouts from declared transformations,
includes gate dependencies, rejects cycles and unsupported features, and
validates block configuration through the registered factories. No block
``run()`` is invoked during compilation.

Definition shape accepted in M1::

    {
      "version": "2.0",
      "inputs": [{"name": ..., "kind": ..., "axes": [{"id", "kind", "stationary"}]}],
      "steps": [{"name": ..., "type": ..., "inputs": {port: selector},
                 "config": {...}, "when": "$steps.gate.keep"}],
      "outputs": [{"name": ..., "selector": ...}]
    }
"""

import copy
import re
import uuid
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from roboflow_workflows.execution_engine.v2.contracts import (
    CONTEXT_POLICY_COMMON_OR_NONE,
    TRANSFORM_APPEND,
    TRANSFORM_COLLAPSE,
    TRANSFORM_PRESERVE,
    VIEW_BATCH,
    VIEW_ITEM,
    BlockContract,
    InputSpec,
    OutputSpec,
    Registry,
)
from roboflow_workflows.execution_engine.v2.data import (
    AXIS_KIND_DYNAMIC_NESTING,
    AXIS_KIND_SAMPLE,
    AXIS_KIND_STATIC_NESTING,
    AXIS_KIND_TIME,
    Axis,
    EntryLayout,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    WorkflowCompileError,
)

SUPPORTED_VERSION = "2.0"
GATE_KIND = "boolean"

# Compiler spellings (also imported by the executor) of values owned by
# ``contracts`` and ``data``; they are the same values, not separate concepts.
ITEM_VIEW = VIEW_ITEM
BATCH_VIEW = VIEW_BATCH
PRESERVE = TRANSFORM_PRESERVE
APPEND = TRANSFORM_APPEND
COLLAPSE = TRANSFORM_COLLAPSE
COMMON_OR_NONE = CONTEXT_POLICY_COMMON_OR_NONE
SAMPLE_AXIS = AXIS_KIND_SAMPLE
STATIC_NESTING = AXIS_KIND_STATIC_NESTING
DYNAMIC_NESTING = AXIS_KIND_DYNAMIC_NESTING
TIME_AXIS = AXIS_KIND_TIME
SUPPORTED_INPUT_AXIS_KINDS = (SAMPLE_AXIS, STATIC_NESTING, DYNAMIC_NESTING)

# Patterns are applied with ``fullmatch`` so the whole string must match; a
# ``$`` anchor with ``match`` would also accept a trailing newline.
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_INPUT_SELECTOR = re.compile(r"\$inputs\.([A-Za-z_][A-Za-z0-9_]*)")
_STEP_SELECTOR = re.compile(
    r"\$steps\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)"
)
_DEFINITION_KEYS = frozenset({"version", "inputs", "steps", "outputs"})
_INPUT_KEYS = frozenset({"name", "kind", "axes"})
_AXIS_KEYS = frozenset({"id", "kind", "stationary"})
_STEP_KEYS = frozenset({"name", "type", "inputs", "config", "when"})
_OUTPUT_KEYS = frozenset({"name", "selector"})


@dataclass(frozen=True)
class CompiledPort:
    """Statically known description of one producer port.

    Args:
        selector: Selector addressing the port, e.g. ``$steps.crop.crops``.
        node: Producer node name, e.g. ``$inputs.images`` or ``$steps.crop``.
        kind: Declared payload kind identifier.
        layout: Logical grouping layout of the produced entry.
    """

    selector: str
    node: str
    kind: str
    layout: EntryLayout


@dataclass(frozen=True)
class CompiledInput:
    """Compiled workflow input declaration.

    Args:
        name: Workflow input name.
        kind: Declared payload kind identifier.
        layout: Declared grouping layout; empty for ungrouped payloads.
    """

    name: str
    kind: str
    layout: EntryLayout

    @property
    def node(self) -> str:
        """Return the node name used in traces and buffers."""
        return f"$inputs.{self.name}"


@dataclass(frozen=True)
class CompiledOutput:
    """Compiled per-output transformation of one step.

    Args:
        name: Declared output name.
        kind: Payload kind identifier of produced items.
        transform: ``preserve``, ``append`` or ``collapse``.
        source: Name of the step input whose grouping the output derives from.
        source_view: View of the source input, ``item`` or ``batch``.
        layout: Resulting entry layout.
        axis_key: Declared axis key for appended outputs, else ``None``.
    """

    name: str
    kind: str
    transform: str
    source: str
    source_view: str
    layout: EntryLayout
    axis_key: Optional[str]

    @property
    def depth(self) -> int:
        """Return the number of grouping axes of the output entry."""
        return len(self.layout.axes)


@dataclass(frozen=True)
class CompiledStep:
    """Immutable wiring of one step in execution order.

    Args:
        name: Step name.
        block_type: Registered block identifier.
        contract: Block contract taken from the registry snapshot.
        factory: Block factory producing a fresh instance per invocation.
        config: Deeply read-only view of the static configuration: mappings are
            read-only proxies, lists/tuples become tuples and sets become
            frozensets. Factories never receive this view; see
            ``materialize_config``.
        inputs: Mapping of input port name to bound selector.
        views: Mapping of input port name to ``item`` or ``batch``.
        prefix: Axes of the invocation prefix shared by all inputs.
        gate: Selector of the boolean gate, if declared with ``when``.
        gate_depth: Number of axes of the gate entry.
        outputs: Compiled per-output transformations.
        dependencies: Names of steps this step depends on (data and gate).
        _config_source: Private detached deep copy of the definition's
            configuration. Only ``materialize_config`` reads it.
    """

    name: str
    block_type: str
    contract: BlockContract
    factory: Callable[[Mapping[str, Any]], Any]
    config: Mapping[str, Any]
    inputs: Mapping[str, str]
    views: Mapping[str, str]
    prefix: Tuple[Axis, ...]
    gate: Optional[str]
    gate_depth: int
    outputs: Mapping[str, CompiledOutput]
    dependencies: Tuple[str, ...]
    _config_source: Mapping[str, Any] = field(repr=False, compare=False)

    def materialize_config(self) -> Dict[str, Any]:
        """Return an isolated copy of the configuration for one factory call.

        Every call returns a new deep copy with the definition's original
        container types, so a block may keep or mutate what it receives
        without affecting the compiled plan or any other instance.

        Returns:
            Fresh ``dict`` equal to the step's configuration.
        """
        materialized = copy.deepcopy(dict(self._config_source))

        return materialized

    @property
    def node(self) -> str:
        """Return the node name used in traces and buffers."""
        return f"$steps.{self.name}"

    @property
    def prefix_depth(self) -> int:
        """Return the number of axes in the invocation prefix."""
        return len(self.prefix)


@dataclass(frozen=True)
class CompiledWorkflowOutput:
    """Compiled workflow output binding.

    Args:
        name: Declared workflow output name.
        selector: Bound producer selector.
    """

    name: str
    selector: str


class CompiledWorkflow:
    """Immutable compiled plan for the serial reference executor.

    Instances are produced by ``compile_workflow``. A plan holds wiring,
    contracts, configuration and factories only; all invocation state lives in
    the executor for the duration of one ``run`` call.
    """

    def __init__(
        self,
        *,
        definition: Mapping[str, Any],
        registry: Registry,
        inputs: Mapping[str, CompiledInput],
        steps: Tuple[CompiledStep, ...],
        outputs: Tuple[CompiledWorkflowOutput, ...],
        ports: Mapping[str, CompiledPort],
    ):
        """Initialise the plan.

        Args:
            definition: Deep copy of the validated definition.
            registry: Detached registry snapshot used by this plan.
            inputs: Compiled workflow inputs keyed by name.
            steps: Steps in a valid execution order.
            outputs: Compiled workflow outputs.
            ports: Every producer port keyed by selector.
        """
        self._definition = definition
        self._registry = registry
        self._inputs = inputs
        self._steps = steps
        self._outputs = outputs
        self._ports = ports
        self._plan_id = uuid.uuid4().hex
        self._pulse_counter = 0

    @property
    def plan_id(self) -> str:
        """Return the identity of this compiled plan."""
        return self._plan_id

    @property
    def registry(self) -> Registry:
        """Return the registry snapshot used by the plan."""
        return self._registry

    @property
    def inputs(self) -> Mapping[str, CompiledInput]:
        """Return compiled workflow inputs keyed by name."""
        return self._inputs

    @property
    def steps(self) -> Tuple[CompiledStep, ...]:
        """Return steps in execution order."""
        return self._steps

    @property
    def outputs(self) -> Tuple[CompiledWorkflowOutput, ...]:
        """Return compiled workflow outputs."""
        return self._outputs

    @property
    def ports(self) -> Mapping[str, CompiledPort]:
        """Return all producer ports keyed by selector."""
        return self._ports

    def next_pulse_id(self) -> int:
        """Allocate the next pulse identifier for a run of this plan.

        Returns:
            Monotonically increasing integer starting at 0.
        """
        pulse_id = self._pulse_counter
        self._pulse_counter += 1

        return pulse_id

    def run(self, *, inputs: Mapping[str, Any]) -> Any:
        """Execute the plan once with fresh invocation state.

        Args:
            inputs: Mapping of workflow input name to ``InputValue`` or a plain
                payload/``Batch`` used as shorthand for a value without metadata.

        Returns:
            ``RunResult`` with outputs buffer, statuses, trace and invocation id.

        Raises:
            WorkflowExecutionError: If input binding, a block or a result
                violates the compiled contract.
        """
        from roboflow_workflows.execution_engine.v2.executor import execute_plan

        result = execute_plan(self, inputs=inputs)

        return result

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description of the compiled plan.

        Returns:
            Dictionary with execution order and per-port kinds and layouts.
        """
        description = {
            "plan_id": self._plan_id,
            "execution_order": [step.name for step in self._steps],
            "inputs": {
                name: {"kind": item.kind, "axes": list(_axes_ids(item.layout.axes))}
                for name, item in self._inputs.items()
            },
            "steps": {
                step.name: {
                    "type": step.block_type,
                    "prefix_axes": list(_axes_ids(step.prefix)),
                    "views": dict(step.views),
                    "gate": step.gate,
                    "dependencies": list(step.dependencies),
                    "outputs": {
                        name: {
                            "kind": output.kind,
                            "transform": output.transform,
                            "axes": list(_axes_ids(output.layout.axes)),
                        }
                        for name, output in step.outputs.items()
                    },
                }
                for step in self._steps
            },
            "outputs": {output.name: output.selector for output in self._outputs},
        }

        return description


def compile_workflow(
    definition: Mapping[str, Any],
    *,
    registry: Registry,
) -> CompiledWorkflow:
    """Compile a V2 workflow definition against an explicit registry.

    Args:
        definition: Workflow definition mapping using version ``"2.0"``.
        registry: Registry supplying kinds, contracts and factories. A snapshot
            is taken so later registry edits do not affect the plan.

    Returns:
        Immutable ``CompiledWorkflow`` ready for ``run``.

    Raises:
        WorkflowCompileError: On any structural, selector, kind, lineage,
            transformation, cycle, configuration or unsupported-feature error.
    """
    snapshot = registry.snapshot()
    raw = _validate_definition_shape(definition)
    inputs = _compile_inputs(raw["inputs"], registry=snapshot)
    raw_steps = _index_steps(raw["steps"])
    dependencies = _collect_dependencies(raw_steps)
    order = _topological_order(dependencies)

    ports: Dict[str, CompiledPort] = {}
    for item in inputs.values():
        ports[f"$inputs.{item.name}"] = CompiledPort(
            selector=f"$inputs.{item.name}",
            node=item.node,
            kind=item.kind,
            layout=item.layout,
        )
    steps: List[CompiledStep] = []
    for step_name in order:
        step = _compile_step(
            raw_steps[step_name],
            registry=snapshot,
            ports=ports,
            dependencies=dependencies[step_name],
        )
        for output in step.outputs.values():
            selector = f"$steps.{step.name}.{output.name}"
            ports[selector] = CompiledPort(
                selector=selector,
                node=step.node,
                kind=output.kind,
                layout=output.layout,
            )
        steps.append(step)
    outputs = _compile_outputs(raw["outputs"], ports=ports)
    _validate_factories(steps)

    plan = CompiledWorkflow(
        definition=copy.deepcopy(dict(raw)),
        registry=snapshot,
        inputs=MappingProxyType(inputs),
        steps=tuple(steps),
        outputs=outputs,
        ports=MappingProxyType(ports),
    )

    return plan


def _validate_definition_shape(definition: Any) -> Mapping[str, Any]:
    if not isinstance(definition, Mapping):
        raise WorkflowCompileError(
            f"Workflow definition must be a mapping, got {type(definition).__name__}."
        )
    _reject_unknown_keys(definition, allowed=_DEFINITION_KEYS, context="definition")
    missing = sorted(_DEFINITION_KEYS - set(definition))
    if missing:
        raise WorkflowCompileError(
            f"Workflow definition is missing required sections: {missing}."
        )
    version = definition["version"]
    if version != SUPPORTED_VERSION:
        raise WorkflowCompileError(
            f"Workflow definition `version` must be {SUPPORTED_VERSION!r} for the V2 "
            f"engine, got {version!r}. V1 definitions are compiled by "
            "`roboflow_workflows.execution_engine.core.ExecutionEngine`."
        )
    for section in ("inputs", "steps", "outputs"):
        if not isinstance(definition[section], list):
            raise WorkflowCompileError(
                f"Workflow definition section `{section}` must be a list, got "
                f"{type(definition[section]).__name__}."
            )

    return definition


def _reject_unknown_keys(item: Mapping[str, Any], *, allowed: frozenset, context: str):
    unknown = sorted(set(item) - allowed)
    if unknown:
        raise WorkflowCompileError(
            f"Unsupported keys {unknown} in {context}. Supported keys: "
            f"{sorted(allowed)}."
        )


def _require_name(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise WorkflowCompileError(
            f"{context} must be an identifier matching [A-Za-z_][A-Za-z0-9_]*, "
            f"got {value!r}."
        )

    return value


def _compile_inputs(
    raw_inputs: List[Any], *, registry: Registry
) -> Dict[str, CompiledInput]:
    inputs: Dict[str, CompiledInput] = {}
    axis_definitions: Dict[str, Tuple[Axis, str]] = {}
    for position, raw in enumerate(raw_inputs):
        context = f"inputs[{position}]"
        if not isinstance(raw, Mapping):
            raise WorkflowCompileError(f"{context} must be a mapping.")
        _reject_unknown_keys(raw, allowed=_INPUT_KEYS, context=context)
        name = _require_name(raw.get("name"), context=f"{context}.name")
        if name in inputs:
            raise WorkflowCompileError(f"Duplicate workflow input name {name!r}.")
        kind = _require_registered_kind(
            raw.get("kind"), registry=registry, context=f"$inputs.{name}"
        )
        axes = _compile_input_axes(raw.get("axes", []), input_name=name)
        for axis in axes:
            previous = axis_definitions.get(axis.id)
            if previous is not None and previous[0] != axis:
                raise WorkflowCompileError(
                    f"Axis {axis.id!r} is declared with different properties on "
                    f"$inputs.{previous[1]} and $inputs.{name}; a shared axis id "
                    "asserts one common lineage and must be declared identically."
                )
            axis_definitions[axis.id] = (axis, name)
        inputs[name] = CompiledInput(
            name=name, kind=kind, layout=_layout(axes, context=f"$inputs.{name}")
        )

    return inputs


def _compile_input_axes(raw_axes: Any, *, input_name: str) -> Tuple[Axis, ...]:
    context = f"$inputs.{input_name}.axes"
    if not isinstance(raw_axes, list):
        raise WorkflowCompileError(f"{context} must be a list of axis mappings.")
    axes: List[Axis] = []
    for position, raw in enumerate(raw_axes):
        axis_context = f"{context}[{position}]"
        if not isinstance(raw, Mapping):
            raise WorkflowCompileError(f"{axis_context} must be a mapping.")
        _reject_unknown_keys(raw, allowed=_AXIS_KEYS, context=axis_context)
        axis_id = _require_name(raw.get("id"), context=f"{axis_context}.id")
        kind = raw.get("kind")
        if kind == TIME_AXIS:
            raise WorkflowCompileError(
                f"{axis_context} declares a `time` axis. Temporal execution is not "
                "supported by the M1 passive V2 engine; only sample and nesting "
                "axes can be executed."
            )
        if kind not in SUPPORTED_INPUT_AXIS_KINDS:
            raise WorkflowCompileError(
                f"{axis_context}.kind must be one of "
                f"{list(SUPPORTED_INPUT_AXIS_KINDS)}, got {kind!r}."
            )
        stationary = raw.get("stationary", kind == STATIC_NESTING)
        if not isinstance(stationary, bool):
            raise WorkflowCompileError(
                f"{axis_context}.stationary must be a bool, got {stationary!r}."
            )
        if kind == SAMPLE_AXIS and position != 0:
            raise WorkflowCompileError(
                f"{axis_context} declares a sample axis at position {position}; a "
                "sample axis must be the first axis."
            )
        try:
            axis = Axis(id=axis_id, kind=kind, stationary=stationary)
        except ContractError as error:
            raise WorkflowCompileError(f"{axis_context} is invalid: {error}") from error
        axes.append(axis)

    return tuple(axes)


def _layout(axes: Tuple[Axis, ...], *, context: str) -> EntryLayout:
    try:
        layout = EntryLayout(axes=tuple(axes))
    except ContractError as error:
        raise WorkflowCompileError(f"Invalid layout for {context}: {error}") from error

    return layout


def _require_registered_kind(kind: Any, *, registry: Registry, context: str) -> str:
    if not isinstance(kind, str) or not kind:
        raise WorkflowCompileError(
            f"{context} must declare a non-empty string `kind`, got {kind!r}."
        )
    if kind not in registry.kind_names:
        raise WorkflowCompileError(
            f"{context} declares unknown kind {kind!r}. Registered kinds: "
            f"{list(registry.kind_names)}."
        )

    return kind


def _index_steps(raw_steps: List[Any]) -> Dict[str, Mapping[str, Any]]:
    steps: Dict[str, Mapping[str, Any]] = {}
    for position, raw in enumerate(raw_steps):
        context = f"steps[{position}]"
        if not isinstance(raw, Mapping):
            raise WorkflowCompileError(f"{context} must be a mapping.")
        _reject_unknown_keys(raw, allowed=_STEP_KEYS, context=context)
        name = _require_name(raw.get("name"), context=f"{context}.name")
        if name in steps:
            raise WorkflowCompileError(f"Duplicate step name {name!r}.")
        if not isinstance(raw.get("type"), str) or not raw["type"]:
            raise WorkflowCompileError(
                f"$steps.{name} must declare a non-empty string `type`."
            )
        if not isinstance(raw.get("inputs"), Mapping):
            raise WorkflowCompileError(
                f"$steps.{name}.inputs must be a mapping of port name to selector."
            )
        if "config" in raw and not isinstance(raw["config"], Mapping):
            raise WorkflowCompileError(f"$steps.{name}.config must be a mapping.")
        if "when" in raw and not isinstance(raw["when"], str):
            raise WorkflowCompileError(
                f"$steps.{name}.when must be a step selector string."
            )
        steps[name] = raw

    return steps


def _parse_selector(selector: Any, *, context: str) -> Tuple[str, Optional[str]]:
    """Return ``(node, output)`` where node is ``$inputs.x`` or ``$steps.s``."""
    if not isinstance(selector, str):
        raise WorkflowCompileError(
            f"{context} must be a selector string like `$inputs.name` or "
            f"`$steps.step.output`, got {selector!r}."
        )
    input_match = _INPUT_SELECTOR.fullmatch(selector)
    if input_match:
        return f"$inputs.{input_match.group(1)}", None
    step_match = _STEP_SELECTOR.fullmatch(selector)
    if step_match:
        return f"$steps.{step_match.group(1)}", step_match.group(2)
    raise WorkflowCompileError(
        f"{context} has invalid selector {selector!r}. Use `$inputs.<name>` or "
        "`$steps.<step>.<output>`."
    )


def _collect_dependencies(
    raw_steps: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Tuple[str, ...]]:
    dependencies: Dict[str, Tuple[str, ...]] = {}
    for name, raw in raw_steps.items():
        found: List[str] = []
        selectors = [
            (f"$steps.{name}.inputs[{port!r}]", selector)
            for port, selector in raw["inputs"].items()
        ]
        if "when" in raw:
            selectors.append((f"$steps.{name}.when", raw["when"]))
        for context, selector in selectors:
            node, output = _parse_selector(selector, context=context)
            if node.startswith("$steps."):
                producer = node[len("$steps.") :]
                if producer not in raw_steps:
                    raise WorkflowCompileError(
                        f"{context} references unknown step {producer!r} via "
                        f"{selector!r}. Known steps: {sorted(raw_steps)}."
                    )
                if producer not in found:
                    found.append(producer)
        dependencies[name] = tuple(found)

    return dependencies


def _topological_order(dependencies: Mapping[str, Tuple[str, ...]]) -> List[str]:
    """Order steps so producers precede consumers; raise on the first cycle."""
    order: List[str] = []
    state: Dict[str, int] = {}
    stack: List[str] = []

    def visit(name: str) -> None:
        marker = state.get(name, 0)
        if marker == 2:
            return
        if marker == 1:
            cycle = stack[stack.index(name) :] + [name]
            raise WorkflowCompileError(
                "Workflow definition contains a dependency cycle: "
                + " -> ".join(f"$steps.{item}" for item in cycle)
                + ". Gate (`when`) references count as dependencies."
            )
        state[name] = 1
        stack.append(name)
        for dependency in dependencies[name]:
            visit(dependency)
        stack.pop()
        state[name] = 2
        order.append(name)

    for name in dependencies:
        visit(name)

    return order


def _compile_step(
    raw: Mapping[str, Any],
    *,
    registry: Registry,
    ports: Mapping[str, CompiledPort],
    dependencies: Tuple[str, ...],
) -> CompiledStep:
    name = raw["name"]
    block_type = raw["type"]
    node = f"$steps.{name}"
    if block_type not in registry.block_names:
        raise WorkflowCompileError(
            f"{node} uses unknown block type {block_type!r}. Registered blocks: "
            f"{list(registry.block_names)}."
        )
    registration = registry.get_block(block_type)
    contract: BlockContract = registration.contract
    if contract.mutates_inputs:
        raise WorkflowCompileError(
            f"{node} uses block {block_type!r} which declares in-place mutation of "
            f"inputs {list(contract.mutates_inputs)}. Mutating block "
            "implementations are not supported by the M1 passive V2 engine."
        )
    bound_ports = _bind_inputs(raw["inputs"], node=node, contract=contract, ports=ports)
    views = {port_name: spec.view for port_name, spec in contract.inputs.items()}
    prefix = _resolve_prefix(node, contract=contract, bound=bound_ports)
    gate, gate_depth = _resolve_gate(
        raw.get("when"), node=node, prefix=prefix, ports=ports
    )
    outputs = _compile_step_outputs(
        node=node, step_name=name, contract=contract, bound=bound_ports, views=views
    )
    config_source = _detach_config(raw.get("config", {}), node=node)
    config_view = _freeze_config(copy.deepcopy(config_source))

    step = CompiledStep(
        name=name,
        block_type=block_type,
        contract=contract,
        factory=registration.factory,
        config=config_view,
        inputs=MappingProxyType(dict(raw["inputs"])),
        views=MappingProxyType(views),
        prefix=prefix,
        gate=gate,
        gate_depth=gate_depth,
        outputs=MappingProxyType(outputs),
        dependencies=dependencies,
        _config_source=config_source,
    )

    return step


def _detach_config(config: Mapping[str, Any], *, node: str) -> Dict[str, Any]:
    try:
        detached = copy.deepcopy(dict(config))
    except Exception as error:
        raise WorkflowCompileError(
            f"{node}.config cannot be copied into the compiled plan: {error}. "
            "Static configuration must be deep-copyable."
        ) from error

    return detached


def _freeze_config(value: Any) -> Any:
    """Recursively convert config containers to read-only equivalents."""
    if isinstance(value, Mapping):
        frozen_mapping = MappingProxyType(
            {key: _freeze_config(item) for key, item in value.items()}
        )
        return frozen_mapping
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_config(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(value)

    return value


def _bind_inputs(
    raw_inputs: Mapping[str, Any],
    *,
    node: str,
    contract: BlockContract,
    ports: Mapping[str, CompiledPort],
) -> Dict[str, CompiledPort]:
    declared = set(contract.inputs)
    supplied = set(raw_inputs)
    missing = sorted(declared - supplied)
    unknown = sorted(supplied - declared)
    if missing or unknown:
        raise WorkflowCompileError(
            f"{node} input binding does not match the block contract. Missing "
            f"ports: {missing}; unknown ports: {unknown}; declared ports: "
            f"{sorted(declared)}."
        )
    bound: Dict[str, CompiledPort] = {}
    for port_name, selector in raw_inputs.items():
        context = f"{node}.inputs[{port_name!r}]"
        _parse_selector(selector, context=context)
        port = ports.get(selector)
        if port is None:
            raise WorkflowCompileError(
                f"{context} references unknown selector {selector!r}. Available "
                f"selectors: {sorted(ports)}."
            )
        spec: InputSpec = contract.inputs[port_name]
        if spec.view not in (ITEM_VIEW, BATCH_VIEW):
            raise WorkflowCompileError(
                f"{context} declares unsupported view {spec.view!r}; supported "
                f"views are {ITEM_VIEW!r} and {BATCH_VIEW!r}."
            )
        if port.kind != spec.kind:
            raise WorkflowCompileError(
                f"{context} expects kind {spec.kind!r} but {selector!r} produces "
                f"kind {port.kind!r}."
            )
        if spec.view == BATCH_VIEW and not port.layout.axes:
            raise WorkflowCompileError(
                f"{context} declares a batch view but {selector!r} is ungrouped "
                "(no axes); a batch view consumes the trailing grouping axis."
            )
        bound[port_name] = port

    return bound


def _resolve_prefix(
    node: str, *, contract: BlockContract, bound: Mapping[str, CompiledPort]
) -> Tuple[Axis, ...]:
    prefixes: Dict[str, Tuple[Axis, ...]] = {}
    batched: Dict[str, Tuple[Axis, ...]] = {}
    for port_name, port in bound.items():
        axes = tuple(port.layout.axes)
        if contract.inputs[port_name].view == BATCH_VIEW:
            batched[port_name] = axes
            prefixes[port_name] = axes[:-1]
        else:
            prefixes[port_name] = axes
    if not prefixes:
        raise WorkflowCompileError(
            f"{node} block contract declares no inputs. Source blocks without "
            "flowing inputs are not supported by the M1 passive V2 engine."
        )
    reference = contract.reference
    if reference not in prefixes:
        raise WorkflowCompileError(
            f"{node} block contract names reference input {reference!r} which is "
            f"not a declared input {sorted(prefixes)}."
        )
    expected = _axes_ids(prefixes[reference])
    for port_name, axes in prefixes.items():
        if _axes_ids(axes) != expected:
            raise WorkflowCompileError(
                f"{node} inputs do not share one invocation prefix. Reference "
                f"input {reference!r} has prefix {expected} but {port_name!r} "
                f"({contract.inputs[port_name].view} view of {bound[port_name].selector!r}) "
                f"has prefix {_axes_ids(axes)}. Equal lengths never establish "
                "correspondence; a parent item beside a child batch is the only "
                "supported mixed binding."
            )
    batched_ids = {name: _axes_ids(axes) for name, axes in batched.items()}
    if len(set(batched_ids.values())) > 1:
        raise WorkflowCompileError(
            f"{node} batched inputs must share their full lineage, got "
            f"{batched_ids}."
        )

    return prefixes[reference]


def _resolve_gate(
    when: Any,
    *,
    node: str,
    prefix: Tuple[Axis, ...],
    ports: Mapping[str, CompiledPort],
) -> Tuple[Optional[str], int]:
    if when is None:
        return None, 0
    context = f"{node}.when"
    gate_node, output = _parse_selector(when, context=context)
    if output is None:
        raise WorkflowCompileError(
            f"{context} must reference a step output (`$steps.<step>.<output>`), "
            f"got {when!r}."
        )
    port = ports.get(when)
    if port is None:
        raise WorkflowCompileError(
            f"{context} references unknown selector {when!r}. Available selectors: "
            f"{sorted(ports)}."
        )
    if port.kind != GATE_KIND:
        raise WorkflowCompileError(
            f"{context} must reference an output of kind {GATE_KIND!r}, but "
            f"{when!r} produces kind {port.kind!r}."
        )
    gate_ids = _axes_ids(port.layout.axes)
    prefix_ids = _axes_ids(prefix)
    if gate_ids != prefix_ids[: len(gate_ids)]:
        raise WorkflowCompileError(
            f"{context} gate {when!r} has axes {gate_ids} which are not the "
            f"invocation prefix {prefix_ids} or an ancestor prefix of it."
        )

    return when, len(gate_ids)


def _compile_step_outputs(
    *,
    node: str,
    step_name: str,
    contract: BlockContract,
    bound: Mapping[str, CompiledPort],
    views: Mapping[str, str],
) -> Dict[str, CompiledOutput]:
    if not contract.outputs:
        raise WorkflowCompileError(f"{node} block contract declares no outputs.")
    outputs: Dict[str, CompiledOutput] = {}
    shared_axes: Dict[str, Axis] = {}
    for output_name, spec in contract.outputs.items():
        context = f"{node}.{output_name}"
        source = spec.source if spec.source is not None else contract.reference
        if source not in bound:
            raise WorkflowCompileError(
                f"{context} names source input {source!r} which is not a declared "
                f"input {sorted(bound)}."
            )
        if spec.context_policy != COMMON_OR_NONE:
            raise WorkflowCompileError(
                f"{context} declares context policy {spec.context_policy!r}; only "
                f"{COMMON_OR_NONE!r} is supported by the M1 passive V2 engine."
            )
        view = views[source]
        source_axes = tuple(bound[source].layout.axes)
        axis_key: Optional[str] = None
        if spec.transform == PRESERVE:
            axes = source_axes
        elif spec.transform == APPEND and view == ITEM_VIEW:
            axis_key = _require_axis_key(spec, context=context)
            axis = shared_axes.get(axis_key)
            if axis is None:
                try:
                    axis = spec.appended_axis(producer=step_name)
                except ContractError as error:
                    raise WorkflowCompileError(
                        f"{context} cannot derive its appended axis: {error}"
                    ) from error
                shared_axes[axis_key] = axis
            elif axis.stationary != bool(spec.stationary):
                raise WorkflowCompileError(
                    f"{context} shares axis key {axis_key!r} with another output "
                    "but declares different stationarity."
                )
            axes = source_axes + (axis,)
        elif spec.transform == COLLAPSE and view == BATCH_VIEW:
            axes = source_axes[:-1]
        else:
            raise WorkflowCompileError(
                f"{context} declares transform {spec.transform!r} on a {view} view "
                f"of input {source!r}. Supported combinations: item+preserve, "
                "item+append, batch+preserve, batch+collapse."
            )
        outputs[output_name] = CompiledOutput(
            name=output_name,
            kind=spec.kind,
            transform=spec.transform,
            source=source,
            source_view=view,
            layout=_layout(axes, context=context),
            axis_key=axis_key,
        )

    return outputs


def _require_axis_key(spec: OutputSpec, *, context: str) -> str:
    if not isinstance(spec.axis, str) or not _NAME.fullmatch(spec.axis):
        raise WorkflowCompileError(
            f"{context} declares an append transform without a valid `axis` key; "
            f"got {spec.axis!r}. The key identifies the appended nesting axis."
        )

    return spec.axis


def _compile_outputs(
    raw_outputs: List[Any], *, ports: Mapping[str, CompiledPort]
) -> Tuple[CompiledWorkflowOutput, ...]:
    outputs: List[CompiledWorkflowOutput] = []
    seen = set()
    for position, raw in enumerate(raw_outputs):
        context = f"outputs[{position}]"
        if not isinstance(raw, Mapping):
            raise WorkflowCompileError(f"{context} must be a mapping.")
        _reject_unknown_keys(raw, allowed=_OUTPUT_KEYS, context=context)
        name = _require_name(raw.get("name"), context=f"{context}.name")
        if name in seen:
            raise WorkflowCompileError(f"Duplicate workflow output name {name!r}.")
        seen.add(name)
        selector = raw.get("selector")
        _parse_selector(selector, context=f"{context}.selector")
        if selector not in ports:
            raise WorkflowCompileError(
                f"{context}.selector references unknown selector {selector!r}. "
                f"Available selectors: {sorted(ports)}."
            )
        outputs.append(CompiledWorkflowOutput(name=name, selector=selector))

    return tuple(outputs)


def _validate_factories(steps: List[CompiledStep]) -> None:
    for step in steps:
        instantiate_block(step)


def instantiate_block(step: CompiledStep) -> Any:
    """Create a fresh block instance from the step's factory and config.

    The factory receives its own ``materialize_config()`` copy, so neither
    the compiled plan nor other instances observe what a block keeps or
    mutates from its configuration.

    Args:
        step: Compiled step whose factory and configuration are used.

    Returns:
        Block instance exposing ``run``.

    Raises:
        WorkflowCompileError: If the factory rejects the configuration or the
            produced object has no callable ``run``.
    """
    config = step.materialize_config()
    try:
        block = step.factory(config)
    except Exception as error:
        raise WorkflowCompileError(
            f"{step.node} block {step.block_type!r} rejected its configuration "
            f"{step.materialize_config()!r}: {error}"
        ) from error
    if not callable(getattr(block, "run", None)):
        raise WorkflowCompileError(
            f"{step.node} block factory for {step.block_type!r} returned "
            f"{type(block).__name__} without a callable `run` method."
        )

    return block


def _axes_ids(axes: Tuple[Axis, ...]) -> Tuple[str, ...]:
    ids = tuple(axis.id for axis in axes)

    return ids
