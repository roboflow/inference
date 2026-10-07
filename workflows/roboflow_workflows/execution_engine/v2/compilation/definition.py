"""Parse V2 workflow definitions into immutable declarations.

A V2 definition keeps V1's JSON shape and selects this engine with
``"version": "2.0"``::

    {
      "version": "2.0",
      "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowBatchInput", "name": "values", "kind": ["float"]},
        {"type": "WorkflowParameter", "name": "factor", "default_value": 2}
      ],
      "steps": [
        {"type": "demo/scale@v1", "name": "scale", "value": "$inputs.values",
         "factor": "$inputs.factor"},
        {"type": "roboflow_core/inner_workflow@v1", "name": "child",
         "workflow_definition": {...}, "parameter_bindings": {"x": "$steps.scale.scaled"}}
      ],
      "outputs": [{"type": "JsonField", "name": "scaled", "selector": "$steps.scale.scaled"}],
      "dynamic_blocks_definitions": []
    }

Input declarations map onto explicit layouts:

======================================  ==========================================
Declaration                             Layout
======================================  ==========================================
``WorkflowImage``, ``WorkflowVideoMetadata``  ``(inputs,)``
``WorkflowBatchInput``, dimensionality d  ``(inputs, inputs.<name>:1 … :d-1)``
``WorkflowParameter``                   ``()``
``{"name", "kind", "axes": [...]}``     exactly the declared axes
======================================  ==========================================

Every batch input shares the root sample axis ``inputs``, like V1's single
``<workflow_input>`` lineage. Deeper levels of a nested batch input get axes of
their own, so two inputs never correspond merely because their sizes match.
The explicit ``axes`` form keeps the advanced independent-axis declaration; a
shared axis id there asserts correspondence. It may end in a ``time`` axis for
values a host collected already (at most one, after stationary axes only).

An active definition adds ``sources`` and delivers its outputs in groups::

    {
      "version": "2.0",
      "inputs": [{"type": "WorkflowParameter", "name": "path"}],
      "sources": [{"type": "demo/csv_temperature@v1", "name": "temp", "path": "$inputs.path"}],
      "steps": [{"type": "demo/to_fahrenheit@v1", "name": "convert",
                 "celsius": "$sources.temp.temperature"}],
      "outputs": [
        {"type": "OutputGroup", "name": "temperatures", "anchor": "$sources.temp.temperature",
         "outputs": [{"type": "JsonField", "name": "fahrenheit", "selector": "$steps.convert.value"}]}
      ]
    }

Operators relate pulses of sources explicitly. They are root declarations with
named selector maps (``inputs`` for alignment, ``collect``/``hold`` for a
window); every other key is a literal parameter of the operator class::

      "operators": [{"type": "v2/window@v1", "name": "clip", "size": 4,
                     "collect": {"frames": "$sources.camera.image"}}]

Their ports are addressed as ``$operators.<operator>.<port>``, also as an
``OutputGroup`` anchor.

Flat ``JsonField`` outputs and ``OutputGroup`` outputs do not mix in one list.

Reactions add ``state``, ``signals`` and ``handlers``. A handler subscribes to
an event of a step in its own scope (``$steps.<step>.events.<event>``; a step
of a nested workflow as ``$steps.<child>/<step>.events.<event>``) or to a root
signal (``$signals.<name>``), and runs its own passive workflow::

      "state": {"global": {"entries": 0}, "source": {"enabled": true}},
      "signals": [{"name": "ack", "fields": {"zone": ["string"]}}],
      "handlers": [{"name": "notify", "on": "$steps.zone.events.entered",
                    "execution": {"mode": "async",
                                  "queue": {"max_depth": 16, "overflow": "leaky"}},
                    "bindings": {"zone": "$event.zone_id", "channel": "ops"},
                    "workflow": {"inputs": [{"name": "zone", "kind": ["string"]}],
                                 "steps": [...], "outputs": [...]}}]

A root ``OutputGroup`` anchored at ``$handlers.<handler>`` (or one of its
outputs, ``$handlers.<handler>.<output>``) delivers that handler's results.

``state_machines`` move between declared states. A fixed transition fires on
an event (``on``); a handler-selected one names the handler whose
``v2/state_machine_set`` step picks one of its ``to`` states. A transition may
emit a machine event (``$state_machines.<machine>.events.<name>``) built from
``$event.<field>``, ``$transition.from|to|name`` and literals::

      "state_machines": [{"name": "gate", "scope": "source",
                          "initial_state": "idle", "states": ["idle", "open"],
                          "transitions": [
                              {"name": "open", "from": ["idle"], "to": "open",
                               "on": "$steps.zone.events.entered",
                               "emit": {"name": "opened",
                                        "fields": {"zone": "$event.zone_id"}}},
                              {"name": "close", "from": ["open"], "to": ["idle"],
                               "handler": "review"}]}]

Handlers and transitions may also subscribe to ``$system.events.started`` and
``$system.events.ended`` of an active run.

This module only checks structure. Composition and compilation resolve
selectors.
"""

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from roboflow_workflows.execution_engine.v2.data import (
    AXIS_KIND_DYNAMIC_NESTING,
    AXIS_KIND_SAMPLE,
    AXIS_KIND_STATIC_NESTING,
    AXIS_KIND_TIME,
    Axis,
    EntryLayout,
)
from roboflow_workflows.execution_engine.v2.declaration import (
    SELECTOR_SEGMENT,
    is_selector_segment,
    parse_selector,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    FieldPath,
    SelectorError,
    StepPath,
    WorkflowCompileError,
)
from roboflow_workflows.execution_engine.v2.events import EVENT_NAME
from roboflow_workflows.execution_engine.v2.kinds import WILDCARD_KIND_NAME
from roboflow_workflows.execution_engine.v2.operators.contract import INPUT_MAP_ROLES
from roboflow_workflows.execution_engine.v2.reactions.plan import (
    HANDLER_MODES,
    MACHINE_SCOPES,
    SYSTEM_EVENTS,
    TRANSITION_VALUES,
    QueuePolicy,
    StateDefaults,
)

SUPPORTED_VERSION = "2.0"
NESTED_WORKFLOW_TYPES: Tuple[str, ...] = (
    "roboflow_core/inner_workflow@v1",
    "inner_workflow",
)
JSON_FIELD_TYPE = "JsonField"
OUTPUT_GROUP_TYPE = "OutputGroup"

ROOT_AXIS = Axis(id="inputs", kind=AXIS_KIND_SAMPLE)
"""Sample axis shared by every batch-oriented workflow input."""

_BATCH_INPUT_KINDS: Mapping[str, Optional[Tuple[str, ...]]] = {
    "WorkflowImage": ("image",),
    "InferenceImage": ("image",),
    "WorkflowVideoMetadata": ("video_metadata",),
    "WorkflowBatchInput": None,
}
_PARAMETER_TYPES = ("WorkflowParameter", "InferenceParameter")
_TYPED_INPUT_KEYS = frozenset(
    {"type", "name", "kind", "dimensionality", "default_value"}
)
_EXPLICIT_INPUT_KEYS = frozenset({"name", "kind", "axes", "default_value"})
_AXIS_KEYS = frozenset({"id", "kind", "stationary"})
_INPUT_AXIS_KINDS = (
    AXIS_KIND_SAMPLE,
    AXIS_KIND_STATIC_NESTING,
    AXIS_KIND_DYNAMIC_NESTING,
    AXIS_KIND_TIME,
)
_DEFINITION_KEYS = frozenset(
    {
        "version",
        "inputs",
        "sources",
        "operators",
        "steps",
        "outputs",
        "dynamic_blocks_definitions",
        "state",
        "signals",
        "handlers",
        "state_machines",
        "recording",
        "retrospective",
        "execution",
    }
)
_ROOT_ONLY_KEYS = ("recording", "retrospective", "execution")
_EXECUTION_SETTINGS_KEYS = frozenset({"quality", "step_quality"})
_QUALITY_LABEL = re.compile(r"[A-Za-z0-9_\-]+")
_NESTED_STEP_KEYS = frozenset(
    {
        "type",
        "name",
        "workflow_definition",
        "workflow_workspace_id",
        "workflow_id",
        "workflow_version_id",
        "parameter_bindings",
        "execution_mode",
    }
)
_OUTPUT_KEYS = frozenset({"type", "name", "selector", "coordinates_system"})
_OUTPUT_GROUP_KEYS = frozenset({"type", "name", "anchor", "outputs"})
_OUTPUT_OPTION_KEYS = ("coordinates_system",)
_HANDLER_KEYS = frozenset({"name", "on", "execution", "bindings", "workflow"})
_EXECUTION_KEYS = frozenset({"mode", "queue"})
_QUEUE_KEYS = frozenset({"max_depth", "overflow"})
_SIGNAL_KEYS = frozenset({"name", "fields", "description"})
_STATE_KEYS = frozenset({"global", "source"})
_MACHINE_KEYS = frozenset(
    {"name", "scope", "initial_state", "states", "transitions", "description"}
)
_TRANSITION_KEYS = frozenset({"name", "from", "to", "on", "handler", "emit"})
_EMIT_KEYS = frozenset({"name", "fields"})
_SEGMENTS = rf"{SELECTOR_SEGMENT}(?:/{SELECTOR_SEGMENT})*"
_STEP_EVENT = re.compile(rf"\$steps\.({_SEGMENTS})\.events\.({EVENT_NAME.pattern})")
_SIGNAL_EVENT = re.compile(rf"\$signals\.({EVENT_NAME.pattern})")
_SYSTEM_EVENT = re.compile(rf"\$system\.events\.({EVENT_NAME.pattern})")
_MACHINE_EVENT = re.compile(
    rf"\$state_machines\.({_SEGMENTS})\.events\.({EVENT_NAME.pattern})"
)
_TRANSITION_VALUE = re.compile(rf"\$transition\.({SELECTOR_SEGMENT})")
_HANDLER_SELECTOR = re.compile(rf"\$handlers\.({_SEGMENTS})(?:\.({SELECTOR_SEGMENT}))?")
_EVENT_FIELD = re.compile(rf"\$event\.({SELECTOR_SEGMENT})")
HANDLERS_PREFIX = "$handlers."


@dataclass(frozen=True)
class WorkflowInputDeclaration:
    """A declared workflow input.

    Args:
        name: Input name.
        kinds: Accepted kind names.
        layout: Layout of the supplied value; empty for parameters.
        required: Whether a root caller must supply a value. A child input
            needs a binding unless its ``default`` is not ``None`` (V1 rule).
        default: Value used when the input is omitted.
        declared_type: Definition input type, e.g. ``"WorkflowImage"``; empty
            for the explicit ``axes`` form.
        location: Definition path for messages, e.g. ``"inputs[1]"``.
    """

    name: str
    kinds: Tuple[str, ...]
    layout: EntryLayout
    required: bool
    default: Any
    declared_type: str
    location: str


@dataclass(frozen=True)
class BlockStepDeclaration:
    """A step running a catalogue block.

    Args:
        name: Step name, unique in its workflow.
        type: Block type or alias.
        params: Step parameters without ``type`` and ``name``.
        location: Definition path for messages.
    """

    name: str
    type: str
    params: Mapping[str, Any]
    location: str


@dataclass(frozen=True)
class WorkflowReference:
    """Identity of a saved workflow, resolved by the caller's resolver.

    Args:
        workspace_id: Workspace of the saved workflow.
        workflow_id: Saved workflow id.
        version_id: Pinned version, or ``None`` for the resolver's default.
    """

    workspace_id: str
    workflow_id: str
    version_id: Optional[str] = None

    def describe(self) -> str:
        """Return ``workspace/workflow`` or ``workspace/workflow@version``."""
        text = f"{self.workspace_id}/{self.workflow_id}"
        if self.version_id is not None:
            text = f"{text}@{self.version_id}"

        return text


@dataclass(frozen=True)
class NestedStepDeclaration:
    """A step embedding another workflow.

    Exactly one of ``definition`` and ``reference`` is set.

    Args:
        name: Step name, unique in its workflow.
        definition: Inline child definition.
        reference: Saved child workflow identity.
        bindings: Child input name to a parent selector or a literal.
        location: Definition path for messages.
    """

    name: str
    definition: Optional[Mapping[str, Any]]
    reference: Optional[WorkflowReference]
    bindings: Mapping[str, Any]
    location: str


StepDeclaration = Union[BlockStepDeclaration, NestedStepDeclaration]


@dataclass(frozen=True)
class WorkflowOutputDeclaration:
    """A declared workflow output.

    Args:
        name: Output name.
        selector: Data selector, possibly ``$steps.<step>.*``.
        options: Output options such as ``coordinates_system``.
        location: Definition path for messages.
    """

    name: str
    selector: str
    options: Mapping[str, Any]
    location: str


@dataclass(frozen=True)
class OutputGroupDeclaration:
    """A declared output group.

    Args:
        name: Group name.
        anchor: ``$sources.<source>.<output>`` selector of the emission whose
            pulse the group follows.
        outputs: The group's fields, in declaration order.
        location: Definition path for messages.
    """

    name: str
    anchor: str
    outputs: Tuple[WorkflowOutputDeclaration, ...]
    location: str


@dataclass(frozen=True)
class SourceDeclaration:
    """A declared source instance.

    Args:
        name: Source name, unique among the workflow's sources.
        type: Source type or alias.
        params: Source parameters without ``type`` and ``name``.
        location: Definition path for messages.
    """

    name: str
    type: str
    params: Mapping[str, Any]
    location: str


@dataclass(frozen=True)
class OperatorInputDeclaration:
    """One ``name: selector`` entry of an operator's selector maps.

    Args:
        name: Input name, unique across the operator's maps.
        role: ``input``, ``collect`` or ``hold``, from the map holding it.
        selector: Data selector as written.
        location: Definition path for messages.
    """

    name: str
    role: str
    selector: str
    location: str


@dataclass(frozen=True)
class OperatorDeclaration:
    """A declared operator instance.

    Args:
        name: Operator name, unique among the workflow's sources and
            operators.
        type: Operator type or alias.
        params: Literal parameters without ``type``, ``name`` and the
            selector maps.
        inputs: Named selectors of every map, in declaration order.
        location: Definition path for messages.
    """

    name: str
    type: str
    params: Mapping[str, Any]
    inputs: Tuple[OperatorInputDeclaration, ...]
    location: str


@dataclass(frozen=True)
class EventSelector:
    """A parsed ``on`` selector of a handler.

    Args:
        kind: ``"step"`` for ``$steps.<path>.events.<event>``, ``"signal"``
            for ``$signals.<event>``, ``"system"`` for
            ``$system.events.<event>``, ``"machine"`` for
            ``$state_machines.<path>.events.<event>``.
        event: Event or signal name.
        path: Step or machine names from the declaring scope,
            ``("child", "zone")`` for ``$steps.child/zone``; ``()`` for a
            signal or a system event.
        text: The selector as written.
    """

    kind: str
    event: str
    path: StepPath
    text: str


@dataclass(frozen=True)
class HandlerDeclaration:
    """A declared event handler.

    Args:
        name: Handler name, unique in its workflow.
        on: Subscribed event.
        mode: ``"sync"`` or ``"async"``.
        queue: Queue policy of an async handler; ``None`` for sync.
        bindings: Handler workflow input to the event field it receives.
        constants: Handler workflow input to a literal value.
        workflow: The raw handler workflow definition; composed later.
        location: Definition path of the handler.
    """

    name: str
    on: EventSelector
    mode: str
    queue: Optional[QueuePolicy]
    bindings: Mapping[str, str]
    constants: Mapping[str, Any]
    workflow: Mapping[str, Any]
    location: str


@dataclass(frozen=True)
class TransitionDeclaration:
    """A declared state machine transition.

    Args:
        name: Transition name, unique in its machine.
        sources: ``from`` states.
        targets: ``to`` states; one for a fixed transition.
        on: Triggering event of a fixed transition, else ``None``.
        handler: Name of the selecting handler, else ``None``.
        emit: Name of the emitted machine event, or ``None``.
        fields: Emitted field to ``("event", field)``,
            ``("transition", "from" | "to" | "name")`` or ``("literal", value)``.
        location: Definition path of the transition.
    """

    name: str
    sources: Tuple[str, ...]
    targets: Tuple[str, ...]
    on: Optional[EventSelector]
    handler: Optional[str]
    emit: Optional[str]
    fields: Mapping[str, Tuple[str, Any]]
    location: str


@dataclass(frozen=True)
class MachineDeclaration:
    """A declared state machine.

    Args:
        name: Machine name, unique in its workflow.
        scope: ``"source"`` or ``"global"``.
        initial: Initial state.
        states: Declared states.
        transitions: Transitions in declaration order.
        location: Definition path of the machine.
    """

    name: str
    scope: str
    initial: str
    states: Tuple[str, ...]
    transitions: Tuple[TransitionDeclaration, ...]
    location: str


@dataclass(frozen=True)
class SignalDeclaration:
    """A declared external signal of the root workflow.

    Args:
        name: Signal name, used as ``$signals.<name>``.
        fields: Field name to accepted kind names.
        description: Human-readable meaning.
        location: Definition path of the signal.
    """

    name: str
    fields: Mapping[str, Tuple[str, ...]]
    description: str
    location: str


@dataclass(frozen=True)
class StateDeclaration:
    """Initial managed-state values declared by one workflow.

    Args:
        defaults: Validated global and per-source initial values.
        location: Definition path of the ``state`` key.
    """

    defaults: StateDefaults
    location: str


@dataclass(frozen=True)
class HandlerGroupDeclaration:
    """A root output group anchored at a handler.

    Args:
        name: Group name.
        handler: Handler path from the root, ``("child", "notify")`` for
            ``$handlers.child/notify``.
        anchor_output: Output named by the anchor, or ``None`` when the
            anchor names the handler only.
        fields: Group field name to the handler output it carries.
        location: Definition path of the group.
    """

    name: str
    handler: StepPath
    anchor_output: Optional[str]
    fields: Mapping[str, str]
    location: str


@dataclass(frozen=True)
class WorkflowDeclaration:
    """One parsed workflow definition, root or child.

    Args:
        inputs: Inputs by name, in declaration order.
        steps: Steps in declaration order.
        outputs: Flat outputs in declaration order; empty when the outputs
            are groups.
        dynamic_blocks: Raw ``dynamic_blocks_definitions`` entries.
        location: Definition path prefix; ``""`` for the root.
        sources: Sources in declaration order.
        output_groups: Output groups in declaration order; empty when the
            outputs are flat.
        operators: Operators in declaration order.
        handlers: Event handlers in declaration order.
        signals: External signals in declaration order.
        state: Initial managed-state values; ``None`` when not declared.
        handler_groups: Output groups anchored at handlers.
        machines: State machines in declaration order.
        recording: Raw root ``recording`` declaration; ``None`` when absent.
            Compiled by ``recording.compilation`` against the plan.
        retrospective: Raw root ``retrospective`` declaration; ``None`` when
            absent.
        execution: Root ``execution`` settings (quality labels); ``None``
            when absent.
    """

    inputs: Mapping[str, WorkflowInputDeclaration]
    steps: Tuple[StepDeclaration, ...]
    outputs: Tuple[WorkflowOutputDeclaration, ...]
    dynamic_blocks: Tuple[Any, ...]
    location: str
    sources: Tuple[SourceDeclaration, ...] = ()
    output_groups: Tuple[OutputGroupDeclaration, ...] = ()
    operators: Tuple[OperatorDeclaration, ...] = ()
    handlers: Tuple[HandlerDeclaration, ...] = ()
    signals: Tuple[SignalDeclaration, ...] = ()
    state: Optional[StateDeclaration] = None
    handler_groups: Tuple[HandlerGroupDeclaration, ...] = ()
    machines: Tuple[MachineDeclaration, ...] = ()
    recording: Optional[Mapping[str, Any]] = None
    retrospective: Optional[Mapping[str, Any]] = None
    execution: Optional["ExecutionSettingsDeclaration"] = None

    def step(self, name: str) -> Optional[StepDeclaration]:
        """Return the step named ``name``.

        Args:
            name: Step name.

        Returns:
            The step, or ``None`` when this workflow has no such step.
        """
        for step in self.steps:
            if step.name == name:
                return step

        return None


def parse_workflow(definition: Any, *, location: str = "") -> WorkflowDeclaration:
    """Check the structure of a V2 definition and parse its sections.

    Kind names are checked later, against the catalogue that includes the
    workflow's dynamic blocks.

    Args:
        definition: Workflow definition mapping; it is not modified.
        location: Definition path of this workflow for messages, for example
            ``"steps[1].workflow_definition."`` for a child.

    Returns:
        The parsed workflow.

    Raises:
        WorkflowCompileError: When the definition is malformed.
        SelectorError: When an output selector is malformed.
    """
    where = location or "definition"
    if not isinstance(definition, Mapping):
        raise WorkflowCompileError(
            f"{where} must be a mapping, got {type(definition).__name__}"
        )
    _reject_unknown_keys(definition, allowed=_DEFINITION_KEYS, location=where)
    misplaced = [key for key in _ROOT_ONLY_KEYS if key in definition]
    if location and misplaced:
        raise WorkflowCompileError(
            f"{location}{misplaced[0]} is root-only: declare recording, "
            "retrospective processing and execution settings in the root "
            "definition"
        )
    version = definition.get("version")
    if version != SUPPORTED_VERSION:
        raise WorkflowCompileError(
            f"{location}version must be {SUPPORTED_VERSION!r} for the V2 engine, got "
            f"{version!r}; V1 definitions run on the V1 ExecutionEngine"
        )

    sections: Dict[str, List[Any]] = {}
    for section in (
        "inputs",
        "sources",
        "operators",
        "steps",
        "outputs",
        "dynamic_blocks_definitions",
        "signals",
        "handlers",
        "state_machines",
    ):
        value = definition.get(section)
        value = [] if value is None else value
        if not isinstance(value, list):
            raise WorkflowCompileError(
                f"{location}{section} must be a list, got {type(value).__name__}"
            )
        sections[section] = value

    outputs, output_groups, handler_groups = _parse_outputs(
        sections["outputs"], location=location
    )
    sources = _parse_sources(sections["sources"], location=location)
    operators = _parse_operators(sections["operators"], location=location)
    clashing = sorted(
        {source.name for source in sources} & {item.name for item in operators}
    )
    if clashing:
        raise WorkflowCompileError(
            f"{location}operators reuse source names {clashing}; sources and "
            "operators share one namespace of pulse domains"
        )
    declaration = WorkflowDeclaration(
        inputs=MappingProxyType(_parse_inputs(sections["inputs"], location=location)),
        steps=_parse_steps(sections["steps"], location=location),
        outputs=outputs,
        dynamic_blocks=tuple(sections["dynamic_blocks_definitions"]),
        location=location,
        sources=sources,
        output_groups=output_groups,
        operators=operators,
        handlers=_parse_handlers(sections["handlers"], location=location),
        signals=_parse_signals(sections["signals"], location=location),
        state=_parse_state(definition.get("state"), location=location),
        handler_groups=handler_groups,
        machines=_parse_machines(sections["state_machines"], location=location),
        recording=_optional_section(definition, "recording"),
        retrospective=_optional_section(definition, "retrospective"),
        execution=_parse_execution_settings(_optional_section(definition, "execution")),
    )

    return declaration


@dataclass(frozen=True)
class ExecutionSettingsDeclaration:
    """The root ``execution`` section: compile-time quality labels.

    ::

        "execution": {
            "quality": "fast",
            "step_quality": {"$steps.segment": "accurate", "$steps.child/overlay": "fast"}
        }

    Labels are literals; a ``$inputs`` selector is rejected because a quality
    selects a compiled implementation and cannot change at run time.

    Args:
        quality: Workflow-level label, or ``None``.
        step_quality: Step-level labels by step selector, as written.
    """

    quality: Optional[str] = None
    step_quality: Mapping[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "step_quality", MappingProxyType(dict(self.step_quality or {}))
        )


def _parse_execution_settings(
    raw: Optional[Mapping[str, Any]],
) -> Optional[ExecutionSettingsDeclaration]:
    if raw is None:
        return None

    _reject_unknown_keys(raw, allowed=_EXECUTION_SETTINGS_KEYS, location="execution")
    quality = raw.get("quality")
    if quality is not None:
        quality = _quality_label(quality, location="execution.quality")
    step_quality: Dict[str, str] = {}
    raw_steps = raw.get("step_quality")
    if raw_steps is not None:
        if not isinstance(raw_steps, Mapping):
            raise WorkflowCompileError(
                "execution.step_quality must map step selectors ($steps.<name>) to "
                f"quality labels, got {type(raw_steps).__name__}"
            )
        for selector, label in raw_steps.items():
            if not isinstance(selector, str) or not selector.startswith("$steps."):
                raise WorkflowCompileError(
                    f"execution.step_quality keys must be step selectors such as "
                    f"'$steps.model' or '$steps.child/model', got {selector!r}"
                )
            step_quality[selector] = _quality_label(
                label, location=f"execution.step_quality[{selector!r}]"
            )
    declaration = ExecutionSettingsDeclaration(
        quality=quality, step_quality=step_quality
    )

    return declaration


def _quality_label(value: Any, *, location: str) -> str:
    if isinstance(value, str) and value.startswith("$"):
        raise WorkflowCompileError(
            f"{location} is the selector {value!r}; a quality label selects a "
            "compiled implementation, so it must be a literal, never a runtime "
            "value"
        )
    if not isinstance(value, str) or not _QUALITY_LABEL.fullmatch(value):
        raise WorkflowCompileError(
            f"{location} must be a quality label of letters, digits, _ or -, got "
            f"{value!r}"
        )

    return value


def _optional_section(definition: Mapping[str, Any], key: str) -> Optional[Any]:
    """A root section kept raw for its own compiler; ``None`` when absent."""
    value = definition.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise WorkflowCompileError(
            f"{key} must be a mapping, got {type(value).__name__}"
        )

    return MappingProxyType(dict(value))


def is_selector_text(value: Any) -> bool:
    """Return whether a binding value is written as a selector.

    Args:
        value: A value from ``parameter_bindings``.

    Returns:
        ``True`` for strings starting with ``$``. Such strings must be complete
        data selectors; every other value is a literal.
    """
    return isinstance(value, str) and value.startswith("$")


def require_data_selector(
    value: Any,
    *,
    location: str,
    step_path: StepPath = (),
    field_path: FieldPath = (),
) -> str:
    """Return ``value`` when it is a complete data selector.

    Args:
        value: Candidate selector.
        location: Definition path for the error message.
        step_path: Step holding the selector, reported by the error.
        field_path: Field path of the selector, reported by the error.

    Returns:
        The selector.

    Raises:
        SelectorError: When ``value`` is not ``$inputs.<name>``,
            ``$steps.<step>.<output>``, ``$steps.<step>.*``,
            ``$sources.<source>.<output>`` or ``$operators.<operator>.<output>``.
    """
    try:
        is_data_selector = parse_selector(value).target != "step"
    except SelectorError:
        is_data_selector = False
    if not is_data_selector:
        raise SelectorError(
            f"{location} holds malformed selector {value!r}; use $inputs.<name>, "
            "$steps.<step>.<output>, $steps.<step>.*, $sources.<source>.<output> "
            "or $operators.<operator>.<output>",
            step_path=step_path,
            field_path=field_path,
        )

    return value


def require_pulse_selector(value: Any, *, location: str) -> str:
    """Return ``value`` when it selects a port of a source or an operator.

    Args:
        value: Candidate selector.
        location: Definition path for the error message.

    Returns:
        The selector.

    Raises:
        SelectorError: When ``value`` selects anything but a source or
            operator port.
    """
    try:
        target = parse_selector(value).target
    except SelectorError:
        target = None
    if target not in ("source_output", "operator_output"):
        raise SelectorError(
            f"{location} must select a source port as $sources.<source>.<output> "
            f"or an operator port as $operators.<operator>.<output>, got {value!r}"
        )

    return value


def _parse_inputs(
    raw_inputs: List[Any], *, location: str
) -> Dict[str, WorkflowInputDeclaration]:
    inputs: Dict[str, WorkflowInputDeclaration] = {}
    declared_axes: Dict[str, Tuple[Axis, str]] = {}
    for position, raw in enumerate(raw_inputs):
        where = f"{location}inputs[{position}]"
        if not isinstance(raw, Mapping):
            raise WorkflowCompileError(f"{where} must be a mapping")

        name = _require_name(raw.get("name"), location=f"{where}.name")
        if name in inputs:
            raise WorkflowCompileError(f"{where}: duplicate workflow input {name!r}")

        if "type" in raw:
            declaration = _parse_typed_input(raw, name=name, location=where)
        else:
            declaration = _parse_explicit_input(raw, name=name, location=where)
        for axis in declaration.layout.axes:
            known = declared_axes.setdefault(axis.id, (axis, name))
            if known[0] != axis:
                raise WorkflowCompileError(
                    f"{where}: axis {axis.id!r} differs from its declaration on "
                    f"$inputs.{known[1]}; a shared axis id asserts one lineage and "
                    "must be declared identically"
                )
        inputs[name] = declaration

    return inputs


def _parse_typed_input(
    raw: Mapping[str, Any], *, name: str, location: str
) -> WorkflowInputDeclaration:
    _reject_unknown_keys(raw, allowed=_TYPED_INPUT_KEYS, location=location)
    input_type = raw["type"]
    if input_type in _PARAMETER_TYPES:
        if raw.get("dimensionality", 0) != 0:
            raise WorkflowCompileError(
                f"{location}: {input_type} is ungrouped; its dimensionality is 0"
            )
        # V1: an omitted root parameter takes default_value (None when absent).
        declaration = WorkflowInputDeclaration(
            name=name,
            kinds=_kind_names(raw.get("kind"), location=location),
            layout=EntryLayout(),
            required=False,
            default=raw.get("default_value"),
            declared_type=input_type,
            location=location,
        )
        return declaration

    if input_type not in _BATCH_INPUT_KINDS:
        raise WorkflowCompileError(
            f"{location}: unknown input type {input_type!r}; supported types: "
            f"{sorted([*_BATCH_INPUT_KINDS, *_PARAMETER_TYPES])}, or the explicit "
            "form with `axes`"
        )
    if "default_value" in raw:
        raise WorkflowCompileError(
            f"{location}: batch input {input_type} cannot declare default_value"
        )

    fixed_kinds = _BATCH_INPUT_KINDS[input_type]
    for key in ("kind", "dimensionality"):
        if fixed_kinds is not None and key in raw:
            raise WorkflowCompileError(
                f"{location}: {input_type} has a fixed {key}; use "
                "WorkflowBatchInput to choose it"
            )
    depth = raw.get("dimensionality", 1)
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise WorkflowCompileError(
            f"{location}: dimensionality must be an integer of at least 1, got "
            f"{depth!r}"
        )

    nested_axes = tuple(
        Axis(id=f"inputs.{name}:{level}", kind=AXIS_KIND_DYNAMIC_NESTING)
        for level in range(1, depth)
    )
    declaration = WorkflowInputDeclaration(
        name=name,
        kinds=fixed_kinds or _kind_names(raw.get("kind"), location=location),
        layout=EntryLayout(axes=(ROOT_AXIS,) + nested_axes),
        required=True,
        default=None,
        declared_type=input_type,
        location=location,
    )

    return declaration


def _parse_explicit_input(
    raw: Mapping[str, Any], *, name: str, location: str
) -> WorkflowInputDeclaration:
    _reject_unknown_keys(raw, allowed=_EXPLICIT_INPUT_KEYS, location=location)
    raw_axes = raw.get("axes", [])
    if not isinstance(raw_axes, list):
        raise WorkflowCompileError(f"{location}.axes must be a list of axis mappings")

    axes = tuple(
        _parse_axis(item, location=f"{location}.axes[{position}]")
        for position, item in enumerate(raw_axes)
    )
    try:
        layout = EntryLayout(axes=axes)
    except ContractError as error:
        raise WorkflowCompileError(f"{location}.axes are invalid: {error}") from error

    declaration = WorkflowInputDeclaration(
        name=name,
        kinds=_kind_names(raw.get("kind"), location=location),
        layout=layout,
        required="default_value" not in raw,
        default=raw.get("default_value"),
        declared_type="",
        location=location,
    )

    return declaration


def _parse_axis(raw: Any, *, location: str) -> Axis:
    if not isinstance(raw, Mapping):
        raise WorkflowCompileError(f"{location} must be a mapping")

    _reject_unknown_keys(raw, allowed=_AXIS_KEYS, location=location)
    axis_id = _require_name(raw.get("id"), location=f"{location}.id")
    kind = raw.get("kind")
    if kind not in _INPUT_AXIS_KINDS:
        raise WorkflowCompileError(
            f"{location}.kind must be one of {list(_INPUT_AXIS_KINDS)}, got {kind!r}"
        )

    try:
        axis = Axis(
            id=axis_id,
            kind=kind,
            stationary=raw.get("stationary", kind == AXIS_KIND_STATIC_NESTING),
        )
    except ContractError as error:
        raise WorkflowCompileError(f"{location} is invalid: {error}") from error

    return axis


def _kind_names(raw: Any, *, location: str) -> Tuple[str, ...]:
    if raw is None:
        return (WILDCARD_KIND_NAME,)

    names = [raw] if isinstance(raw, str) else raw
    if not isinstance(names, list) or not names:
        raise WorkflowCompileError(
            f"{location}.kind must be a kind name or a non-empty list of names"
        )
    for kind_name in names:
        if not isinstance(kind_name, str) or not kind_name:
            raise WorkflowCompileError(
                f"{location}.kind must contain kind names, got {kind_name!r}"
            )

    unique_names = tuple(dict.fromkeys(names))

    return unique_names


def _parse_steps(raw_steps: List[Any], *, location: str) -> Tuple[StepDeclaration, ...]:
    steps: List[StepDeclaration] = []
    names = set()
    for position, raw in enumerate(raw_steps):
        where = f"{location}steps[{position}]"
        if not isinstance(raw, Mapping):
            raise WorkflowCompileError(f"{where} must be a mapping")

        name = _require_name(raw.get("name"), location=f"{where}.name")
        if name in names:
            raise WorkflowCompileError(f"{where}: duplicate step name {name!r}")
        names.add(name)
        step_type = raw.get("type")
        if not isinstance(step_type, str) or not step_type:
            raise WorkflowCompileError(f"{where} ($steps.{name}) must declare a type")

        if step_type in NESTED_WORKFLOW_TYPES:
            steps.append(_parse_nested_step(raw, name=name, location=where))
            continue

        params = {
            key: value for key, value in raw.items() if key not in ("type", "name")
        }
        steps.append(
            BlockStepDeclaration(
                name=name,
                type=step_type,
                params=MappingProxyType(params),
                location=where,
            )
        )

    return tuple(steps)


def _parse_nested_step(
    raw: Mapping[str, Any], *, name: str, location: str
) -> NestedStepDeclaration:
    _reject_unknown_keys(raw, allowed=_NESTED_STEP_KEYS, location=location)
    mode = raw.get("execution_mode", "embedded")
    if mode != "embedded":
        raise WorkflowCompileError(
            f"{location} ($steps.{name}) uses execution_mode {mode!r}; the local "
            "sequential engine only embeds child workflows"
        )

    bindings = raw.get("parameter_bindings") or {}
    if not isinstance(bindings, Mapping):
        raise WorkflowCompileError(
            f"{location}.parameter_bindings must map child input names to selectors "
            "or literals"
        )

    inline = raw.get("workflow_definition")
    has_inline = isinstance(inline, Mapping) and len(inline) > 0
    workspace_id = _optional_text(raw.get("workflow_workspace_id"))
    workflow_id = _optional_text(raw.get("workflow_id"))
    if (workspace_id is None) != (workflow_id is None):
        raise WorkflowCompileError(
            f"{location} ($steps.{name}) must set both workflow_workspace_id and "
            "workflow_id to reference a saved workflow"
        )
    reference = None
    if workspace_id is not None:
        reference = WorkflowReference(
            workspace_id=workspace_id,
            workflow_id=workflow_id,
            version_id=_optional_text(raw.get("workflow_version_id")),
        )
    if has_inline == (reference is not None):
        raise WorkflowCompileError(
            f"{location} ($steps.{name}) needs exactly one of a non-empty "
            "workflow_definition or a saved workflow reference "
            "(workflow_workspace_id and workflow_id)"
        )

    declaration = NestedStepDeclaration(
        name=name,
        definition=inline if has_inline else None,
        reference=reference,
        bindings=MappingProxyType(dict(bindings)),
        location=location,
    )

    return declaration


def _parse_sources(
    raw_sources: List[Any], *, location: str
) -> Tuple[SourceDeclaration, ...]:
    sources: List[SourceDeclaration] = []
    names = set()
    for position, raw in enumerate(raw_sources):
        where = f"{location}sources[{position}]"
        if not isinstance(raw, Mapping):
            raise WorkflowCompileError(f"{where} must be a mapping")

        name = _require_name(raw.get("name"), location=f"{where}.name")
        if name in names:
            raise WorkflowCompileError(f"{where}: duplicate source name {name!r}")
        names.add(name)
        source_type = raw.get("type")
        if not isinstance(source_type, str) or not source_type:
            raise WorkflowCompileError(f"{where} ($sources.{name}) must declare a type")

        params = {
            key: value for key, value in raw.items() if key not in ("type", "name")
        }
        sources.append(
            SourceDeclaration(
                name=name,
                type=source_type,
                params=MappingProxyType(params),
                location=where,
            )
        )

    return tuple(sources)


def _parse_operators(
    raw_operators: List[Any], *, location: str
) -> Tuple[OperatorDeclaration, ...]:
    operators: List[OperatorDeclaration] = []
    names = set()
    for position, raw in enumerate(raw_operators):
        where = f"{location}operators[{position}]"
        if not isinstance(raw, Mapping):
            raise WorkflowCompileError(f"{where} must be a mapping")

        name = _require_name(raw.get("name"), location=f"{where}.name")
        if name in names:
            raise WorkflowCompileError(f"{where}: duplicate operator name {name!r}")
        names.add(name)
        operator_type = raw.get("type")
        if not isinstance(operator_type, str) or not operator_type:
            raise WorkflowCompileError(
                f"{where} ($operators.{name}) must declare a type"
            )

        inputs: List[OperatorInputDeclaration] = []
        for key, role in INPUT_MAP_ROLES.items():
            inputs.extend(
                _parse_operator_inputs(
                    raw.get(key), role=role, location=f"{where}.{key}"
                )
            )
        input_names = [item.name for item in inputs]
        repeated = sorted({item for item in input_names if input_names.count(item) > 1})
        if repeated:
            raise WorkflowCompileError(
                f"{where} ($operators.{name}) repeats input names {repeated} across "
                f"its {list(INPUT_MAP_ROLES)} maps"
            )
        if not inputs:
            raise WorkflowCompileError(
                f"{where} ($operators.{name}) declares no inputs; name the selectors "
                f"it consumes in one of {list(INPUT_MAP_ROLES)}"
            )

        params = {
            key: value
            for key, value in raw.items()
            if key not in ("type", "name") and key not in INPUT_MAP_ROLES
        }
        operators.append(
            OperatorDeclaration(
                name=name,
                type=operator_type,
                params=MappingProxyType(params),
                inputs=tuple(inputs),
                location=where,
            )
        )

    return tuple(operators)


def _parse_operator_inputs(
    raw: Any, *, role: str, location: str
) -> List[OperatorInputDeclaration]:
    if raw is None:
        return []
    if not isinstance(raw, Mapping):
        raise WorkflowCompileError(
            f"{location} must map input names to selectors, got {type(raw).__name__}"
        )

    inputs = [
        OperatorInputDeclaration(
            name=_require_name(name, location=f"{location} key {name!r}"),
            role=role,
            selector=require_data_selector(selector, location=f"{location}.{name}"),
            location=f"{location}.{name}",
        )
        for name, selector in raw.items()
    ]

    return inputs


def _parse_outputs(raw_outputs: List[Any], *, location: str) -> Tuple[
    Tuple[WorkflowOutputDeclaration, ...],
    Tuple[OutputGroupDeclaration, ...],
    Tuple[HandlerGroupDeclaration, ...],
]:
    """Parse flat ``JsonField`` outputs or ``OutputGroup`` outputs, never both.

    Groups anchored at ``$handlers.`` become handler groups.
    """
    types = {
        raw.get("type", JSON_FIELD_TYPE) if isinstance(raw, Mapping) else None
        for raw in raw_outputs
    }
    if types == {OUTPUT_GROUP_TYPE}:
        groups: List[OutputGroupDeclaration] = []
        handler_groups: List[HandlerGroupDeclaration] = []
        for position, raw in enumerate(raw_outputs):
            where = f"{location}outputs[{position}]"
            anchor = raw.get("anchor")
            if isinstance(anchor, str) and anchor.startswith(HANDLERS_PREFIX):
                handler_groups.append(_parse_handler_group(raw, location=where))
                continue
            groups.append(_parse_output_group(raw, location=where))
        names = [group.name for group in [*groups, *handler_groups]]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise WorkflowCompileError(
                f"{location}outputs: duplicate output group names {duplicates}"
            )
        return (), tuple(groups), tuple(handler_groups)
    if OUTPUT_GROUP_TYPE in types:
        raise WorkflowCompileError(
            f"{location}outputs mixes {OUTPUT_GROUP_TYPE} and {JSON_FIELD_TYPE} "
            "entries; declare either flat outputs or output groups"
        )

    outputs = _parse_json_fields(raw_outputs, location=f"{location}outputs")

    return outputs, (), ()


def _parse_output_group(raw: Any, *, location: str) -> OutputGroupDeclaration:
    if not isinstance(raw, Mapping):
        raise WorkflowCompileError(f"{location} must be a mapping")

    _reject_unknown_keys(raw, allowed=_OUTPUT_GROUP_KEYS, location=location)
    name = _require_name(raw.get("name"), location=f"{location}.name")
    fields = raw.get("outputs")
    if not isinstance(fields, list):
        raise WorkflowCompileError(
            f"{location}.outputs must be a list of {JSON_FIELD_TYPE} selections"
        )
    for position, field in enumerate(fields):
        field_type = (
            field.get("type", JSON_FIELD_TYPE) if isinstance(field, Mapping) else None
        )
        if field_type != JSON_FIELD_TYPE:
            raise WorkflowCompileError(
                f"{location}.outputs[{position}] must be a {JSON_FIELD_TYPE}; groups "
                "do not nest"
            )
        selector = field.get("selector")
        if isinstance(selector, str) and selector.startswith(HANDLERS_PREFIX):
            raise WorkflowCompileError(
                f"{location}.outputs[{position}] selects {selector!r} in a group "
                f"anchored at {raw.get('anchor')!r}; handler results arrive per "
                "handler run, so put them in a group anchored at the handler, "
                "e.g. $handlers.<handler>"
            )

    group = OutputGroupDeclaration(
        name=name,
        anchor=require_pulse_selector(raw.get("anchor"), location=f"{location}.anchor"),
        outputs=_parse_json_fields(fields, location=f"{location}.outputs"),
        location=location,
    )

    return group


def _parse_handler_group(raw: Any, *, location: str) -> HandlerGroupDeclaration:
    _reject_unknown_keys(raw, allowed=_OUTPUT_GROUP_KEYS, location=location)
    name = _require_name(raw.get("name"), location=f"{location}.name")
    anchor = raw["anchor"]
    handler, anchor_output = _parse_handler_selector(
        anchor, location=f"{location}.anchor"
    )
    fields = raw.get("outputs")
    if not isinstance(fields, list) or not fields:
        raise WorkflowCompileError(
            f"{location}.outputs must be a non-empty list of {JSON_FIELD_TYPE} "
            f"selections of {HANDLERS_PREFIX}{'/'.join(handler)}.<output>"
        )

    selected: Dict[str, str] = {}
    for position, item in enumerate(fields):
        where = f"{location}.outputs[{position}]"
        if not isinstance(item, Mapping):
            raise WorkflowCompileError(f"{where} must be a mapping")
        _reject_unknown_keys(item, allowed=_OUTPUT_KEYS, location=where)
        if item.get("type", JSON_FIELD_TYPE) != JSON_FIELD_TYPE:
            raise WorkflowCompileError(
                f"{where} must be a {JSON_FIELD_TYPE}; groups do not nest"
            )
        options = sorted(key for key in _OUTPUT_OPTION_KEYS if key in item)
        if options:
            raise WorkflowCompileError(
                f"{where} sets {options}; handler group fields carry handler "
                "outputs as the handler returns them"
            )
        field_name = _require_name(item.get("name"), location=f"{where}.name")
        if field_name in selected:
            raise WorkflowCompileError(f"{where}: duplicate group field {field_name!r}")
        selector = item.get("selector")
        path, output = _parse_handler_selector(selector, location=f"{where}.selector")
        if path != handler or output is None:
            raise WorkflowCompileError(
                f"{where}.selector is {selector!r}; a group anchored at {anchor!r} "
                f"carries outputs of that handler only, as "
                f"{HANDLERS_PREFIX}{'/'.join(handler)}.<output>"
            )
        selected[field_name] = output

    group = HandlerGroupDeclaration(
        name=name,
        handler=handler,
        anchor_output=anchor_output,
        fields=MappingProxyType(selected),
        location=location,
    )

    return group


def _parse_handler_selector(
    value: Any, *, location: str
) -> Tuple[StepPath, Optional[str]]:
    matched = _HANDLER_SELECTOR.fullmatch(value) if isinstance(value, str) else None
    if matched is None:
        raise SelectorError(
            f"{location} must be $handlers.<handler> or "
            f"$handlers.<handler>.<output> (a nested workflow's handler as "
            f"$handlers.<child>/<handler>), got {value!r}"
        )

    path = tuple(matched.group(1).split("/"))

    return path, matched.group(2)


def _parse_handlers(
    raw_handlers: List[Any], *, location: str
) -> Tuple[HandlerDeclaration, ...]:
    handlers: List[HandlerDeclaration] = []
    names = set()
    for position, raw in enumerate(raw_handlers):
        where = f"{location}handlers[{position}]"
        if not isinstance(raw, Mapping):
            raise WorkflowCompileError(f"{where} must be a mapping")

        _reject_unknown_keys(raw, allowed=_HANDLER_KEYS, location=where)
        name = _require_name(raw.get("name"), location=f"{where}.name")
        if name in names:
            raise WorkflowCompileError(f"{where}: duplicate handler name {name!r}")
        names.add(name)
        mode, queue = _parse_execution(raw.get("execution"), location=where)
        bindings, constants = _parse_handler_bindings(
            raw.get("bindings"), location=f"{where}.bindings"
        )
        workflow = raw.get("workflow")
        if not isinstance(workflow, Mapping) or not workflow:
            raise WorkflowCompileError(
                f"{where} ($handlers.{name}) needs a non-empty workflow definition"
            )
        handlers.append(
            HandlerDeclaration(
                name=name,
                on=_parse_event_selector(raw.get("on"), location=f"{where}.on"),
                mode=mode,
                queue=queue,
                bindings=MappingProxyType(bindings),
                constants=MappingProxyType(constants),
                workflow=workflow,
                location=where,
            )
        )

    return tuple(handlers)


def _parse_event_selector(value: Any, *, location: str) -> EventSelector:
    text = value if isinstance(value, str) else ""
    for kind, pattern in (("step", _STEP_EVENT), ("machine", _MACHINE_EVENT)):
        matched = pattern.fullmatch(text)
        if matched is not None:
            selector = EventSelector(
                kind=kind,
                event=matched.group(2),
                path=tuple(matched.group(1).split("/")),
                text=text,
            )
            return selector
    for kind, pattern in (("signal", _SIGNAL_EVENT), ("system", _SYSTEM_EVENT)):
        matched = pattern.fullmatch(text)
        if matched is not None:
            break
    else:
        raise SelectorError(
            f"{location} must be $steps.<step>.events.<event> (a nested workflow's "
            "step as $steps.<child>/<step>.events.<event>), $signals.<signal>, "
            "$state_machines.<machine>.events.<event> or "
            f"$system.events.<event>, got {value!r}"
        )
    if kind == "system" and matched.group(1) not in SYSTEM_EVENTS:
        raise SelectorError(
            f"{location} names unknown system event {matched.group(1)!r}; system "
            f"events are {sorted(SYSTEM_EVENTS)}"
        )

    selector = EventSelector(kind=kind, event=matched.group(1), path=(), text=text)

    return selector


def _parse_execution(raw: Any, *, location: str) -> Tuple[str, Optional[QueuePolicy]]:
    where = f"{location}.execution"
    if raw is None:
        return "sync", None
    if not isinstance(raw, Mapping):
        raise WorkflowCompileError(f"{where} must be a mapping")

    _reject_unknown_keys(raw, allowed=_EXECUTION_KEYS, location=where)
    mode = raw.get("mode", "sync")
    if mode not in HANDLER_MODES:
        raise WorkflowCompileError(
            f"{where}.mode must be one of {list(HANDLER_MODES)}, got {mode!r}"
        )
    raw_queue = raw.get("queue")
    if mode == "sync":
        if raw_queue is not None:
            raise WorkflowCompileError(
                f"{where} declares a queue for a sync handler; a sync handler "
                "runs before emit returns and queues nothing"
            )
        return mode, None

    raw_queue = {} if raw_queue is None else raw_queue
    if not isinstance(raw_queue, Mapping):
        raise WorkflowCompileError(f"{where}.queue must be a mapping")
    _reject_unknown_keys(raw_queue, allowed=_QUEUE_KEYS, location=f"{where}.queue")
    try:
        queue = QueuePolicy(**raw_queue)
    except ContractError as error:
        raise WorkflowCompileError(f"{where}.queue is invalid: {error}") from error

    return mode, queue


def _parse_handler_bindings(
    raw: Any, *, location: str
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    raw = {} if raw is None else raw
    if not isinstance(raw, Mapping):
        raise WorkflowCompileError(
            f"{location} must map handler workflow inputs to $event.<field> or "
            "literals"
        )

    bindings: Dict[str, str] = {}
    constants: Dict[str, Any] = {}
    for input_name, value in raw.items():
        where = f"{location}.{input_name}"
        _require_name(input_name, location=f"{location} key {input_name!r}")
        if not is_selector_text(value):
            constants[input_name] = value
            continue
        matched = _EVENT_FIELD.fullmatch(value)
        if matched is None:
            raise SelectorError(
                f"{where} is {value!r}; a handler binds only $event.<field> or a "
                "literal. Workflow inputs and step outputs are not available to "
                "handlers; emit the value as an event field instead"
            )
        bindings[input_name] = matched.group(1)

    return bindings, constants


def _parse_signals(
    raw_signals: List[Any], *, location: str
) -> Tuple[SignalDeclaration, ...]:
    signals: List[SignalDeclaration] = []
    names = set()
    for position, raw in enumerate(raw_signals):
        where = f"{location}signals[{position}]"
        if not isinstance(raw, Mapping):
            raise WorkflowCompileError(f"{where} must be a mapping")

        _reject_unknown_keys(raw, allowed=_SIGNAL_KEYS, location=where)
        name = raw.get("name")
        if not isinstance(name, str) or EVENT_NAME.fullmatch(name) is None:
            raise WorkflowCompileError(
                f"{where}.name must start with a letter or '_' and contain letters, "
                f"digits, '_' and '-', got {name!r}"
            )
        if name in names:
            raise WorkflowCompileError(f"{where}: duplicate signal name {name!r}")
        names.add(name)
        raw_fields = raw.get("fields", {})
        if not isinstance(raw_fields, Mapping):
            raise WorkflowCompileError(
                f"{where}.fields must map field names to kind names"
            )
        fields = {
            _require_name(field_name, location=f"{where}.fields key {field_name!r}"): (
                _kind_names(kinds or None, location=f"{where}.fields.{field_name}")
            )
            for field_name, kinds in raw_fields.items()
        }
        description = raw.get("description", "")
        if not isinstance(description, str):
            raise WorkflowCompileError(f"{where}.description must be text")
        signals.append(
            SignalDeclaration(
                name=name,
                fields=MappingProxyType(fields),
                description=description,
                location=where,
            )
        )

    return tuple(signals)


def _parse_machines(
    raw_machines: List[Any], *, location: str
) -> Tuple[MachineDeclaration, ...]:
    machines: List[MachineDeclaration] = []
    names = set()
    for position, raw in enumerate(raw_machines):
        where = f"{location}state_machines[{position}]"
        if not isinstance(raw, Mapping):
            raise WorkflowCompileError(f"{where} must be a mapping")

        _reject_unknown_keys(raw, allowed=_MACHINE_KEYS, location=where)
        name = _require_name(raw.get("name"), location=f"{where}.name")
        if name in names:
            raise WorkflowCompileError(f"{where}: duplicate state machine {name!r}")
        names.add(name)
        scope = raw.get("scope")
        if scope not in MACHINE_SCOPES:
            raise WorkflowCompileError(
                f"{where}.scope must be one of {list(MACHINE_SCOPES)}, got {scope!r}"
            )
        states = _state_names(raw.get("states"), location=f"{where}.states")
        initial = raw.get("initial_state")
        if initial not in states:
            raise WorkflowCompileError(
                f"{where}.initial_state {initial!r} is not one of the states {states}"
            )
        raw_transitions = raw.get("transitions")
        if not isinstance(raw_transitions, list):
            raise WorkflowCompileError(f"{where}.transitions must be a list")
        transitions = tuple(
            _parse_transition(item, states=states, location=f"{where}.transitions[{i}]")
            for i, item in enumerate(raw_transitions)
        )
        names_seen = [item.name for item in transitions]
        repeated = sorted({item for item in names_seen if names_seen.count(item) > 1})
        if repeated:
            raise WorkflowCompileError(f"{where}: duplicate transitions {repeated}")
        machines.append(
            MachineDeclaration(
                name=name,
                scope=scope,
                initial=initial,
                states=states,
                transitions=transitions,
                location=where,
            )
        )

    return tuple(machines)


def _parse_transition(
    raw: Any, *, states: Tuple[str, ...], location: str
) -> TransitionDeclaration:
    if not isinstance(raw, Mapping):
        raise WorkflowCompileError(f"{location} must be a mapping")

    _reject_unknown_keys(raw, allowed=_TRANSITION_KEYS, location=location)
    name = _require_name(raw.get("name"), location=f"{location}.name")
    where = f"{location} ({name})"
    if ("on" in raw) == ("handler" in raw):
        raise WorkflowCompileError(
            f"{where} needs exactly one of 'on' (an event fires it) or 'handler' "
            "(that handler's v2/state_machine_set step picks the target)"
        )
    sources = _state_names(raw.get("from"), location=f"{where}.from", states=states)
    raw_targets = raw.get("to")
    on = handler = None
    if "on" in raw:
        on = _parse_event_selector(raw["on"], location=f"{where}.on")
        if not isinstance(raw_targets, str):
            raise WorkflowCompileError(
                f"{where}.to must be one state for a transition fired by an event, "
                f"got {raw_targets!r}; list several states only for a "
                "handler-selected transition"
            )
    else:
        handler = _require_name(raw["handler"], location=f"{where}.handler")
    targets = _state_names(raw_targets, location=f"{where}.to", states=states)
    emit, fields = _parse_emit(raw.get("emit"), fixed=on is not None, location=where)

    transition = TransitionDeclaration(
        name=name,
        sources=sources,
        targets=targets,
        on=on,
        handler=handler,
        emit=emit,
        fields=MappingProxyType(fields),
        location=where,
    )

    return transition


def _state_names(
    raw: Any, *, location: str, states: Optional[Tuple[str, ...]] = None
) -> Tuple[str, ...]:
    names = [raw] if isinstance(raw, str) else raw
    if not isinstance(names, list) or not names:
        raise WorkflowCompileError(
            f"{location} must be a state name or a non-empty list of them, got {raw!r}"
        )
    for name in names:
        _require_name(name, location=f"{location} state {name!r}")
    if len(set(names)) != len(names):
        raise WorkflowCompileError(f"{location} repeats a state: {names}")
    unknown = [name for name in names if states is not None and name not in states]
    if unknown:
        raise WorkflowCompileError(
            f"{location} names unknown states {unknown}; states are {list(states)}"
        )

    return tuple(names)


def _parse_emit(
    raw: Any, *, fixed: bool, location: str
) -> Tuple[Optional[str], Dict[str, Tuple[str, Any]]]:
    where = f"{location}.emit"
    if raw is None:
        return None, {}
    if not isinstance(raw, Mapping):
        raise WorkflowCompileError(f"{where} must be a mapping with 'name', 'fields'")

    _reject_unknown_keys(raw, allowed=_EMIT_KEYS, location=where)
    name = raw.get("name")
    if not isinstance(name, str) or EVENT_NAME.fullmatch(name) is None:
        raise WorkflowCompileError(
            f"{where}.name must start with a letter or '_' and contain letters, "
            f"digits, '_' and '-', got {name!r}"
        )
    raw_fields = raw.get("fields", {})
    if not isinstance(raw_fields, Mapping):
        raise WorkflowCompileError(f"{where}.fields must map field names to values")

    fields: Dict[str, Tuple[str, Any]] = {}
    for field_name, value in raw_fields.items():
        field_where = f"{where}.fields.{field_name}"
        _require_name(field_name, location=f"{where}.fields key {field_name!r}")
        if not is_selector_text(value):
            fields[field_name] = ("literal", value)
            continue
        event_field = _EVENT_FIELD.fullmatch(value)
        transition_value = _TRANSITION_VALUE.fullmatch(value)
        if event_field is not None and fixed:
            fields[field_name] = ("event", event_field.group(1))
        elif event_field is not None:
            raise SelectorError(
                f"{field_where} is {value!r}, but a handler-selected transition does "
                "not keep the payload of the event that started its handler. Use "
                "$transition.from|to|name or a literal, or return the value from "
                "the handler workflow"
            )
        elif transition_value is not None:
            if transition_value.group(1) not in TRANSITION_VALUES:
                raise SelectorError(
                    f"{field_where} is {value!r}; a transition provides "
                    f"{[f'$transition.{item}' for item in TRANSITION_VALUES]}"
                )
            fields[field_name] = ("transition", transition_value.group(1))
        else:
            raise SelectorError(
                f"{field_where} is {value!r}; an emitted field is $event.<field>, "
                "$transition.from|to|name or a literal"
            )

    return name, fields


def _parse_state(raw: Any, *, location: str) -> Optional[StateDeclaration]:
    where = f"{location}state"
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise WorkflowCompileError(
            f"{where} must be a mapping with 'global' and/or 'source' initial values"
        )

    _reject_unknown_keys(raw, allowed=_STATE_KEYS, location=where)
    scopes = {}
    for scope in ("global", "source"):
        values = raw.get(scope, {})
        if not isinstance(values, Mapping):
            raise WorkflowCompileError(f"{where}.{scope} must map keys to values")
        scopes[scope] = values
    try:
        defaults = StateDefaults(global_=scopes["global"], source=scopes["source"])
    except ContractError as error:
        raise WorkflowCompileError(f"{where} is invalid: {error}") from error

    declaration = StateDeclaration(defaults=defaults, location=where)

    return declaration


def _parse_json_fields(
    raw_outputs: List[Any], *, location: str
) -> Tuple[WorkflowOutputDeclaration, ...]:
    outputs: List[WorkflowOutputDeclaration] = []
    names = set()
    for position, raw in enumerate(raw_outputs):
        where = f"{location}[{position}]"
        if not isinstance(raw, Mapping):
            raise WorkflowCompileError(f"{where} must be a mapping")

        _reject_unknown_keys(raw, allowed=_OUTPUT_KEYS, location=where)
        if raw.get("type", JSON_FIELD_TYPE) != JSON_FIELD_TYPE:
            raise WorkflowCompileError(
                f"{where}.type must be {JSON_FIELD_TYPE!r}, got {raw.get('type')!r}"
            )
        name = _require_name(raw.get("name"), location=f"{where}.name")
        if name in names:
            raise WorkflowCompileError(f"{where}: duplicate workflow output {name!r}")
        names.add(name)
        options = {key: raw[key] for key in _OUTPUT_OPTION_KEYS if key in raw}
        outputs.append(
            WorkflowOutputDeclaration(
                name=name,
                selector=require_data_selector(
                    raw.get("selector"), location=f"{where}.selector"
                ),
                options=MappingProxyType(options),
                location=where,
            )
        )

    return tuple(outputs)


def _require_name(value: Any, *, location: str) -> str:
    if not is_selector_segment(value):
        raise WorkflowCompileError(
            f"{location} must be a name of letters, digits, '_' and '-', got {value!r}"
        )

    return value


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()

    return text or None


def _reject_unknown_keys(
    item: Mapping[str, Any], *, allowed: frozenset, location: str
) -> None:
    unknown = sorted(set(item) - allowed)
    if unknown:
        raise WorkflowCompileError(
            f"{location} has unsupported keys {unknown}; supported keys: "
            f"{sorted(allowed)}"
        )
