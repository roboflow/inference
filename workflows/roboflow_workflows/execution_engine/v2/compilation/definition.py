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
shared axis id there asserts correspondence.

This module only checks structure. Composition and compilation resolve
selectors.
"""

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
from roboflow_workflows.execution_engine.v2.kinds import WILDCARD_KIND_NAME

SUPPORTED_VERSION = "2.0"
NESTED_WORKFLOW_TYPES: Tuple[str, ...] = (
    "roboflow_core/inner_workflow@v1",
    "inner_workflow",
)
JSON_FIELD_TYPE = "JsonField"

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
)
_DEFINITION_KEYS = frozenset(
    {"version", "inputs", "steps", "outputs", "dynamic_blocks_definitions"}
)
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
_OUTPUT_OPTION_KEYS = ("coordinates_system",)


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
class WorkflowDeclaration:
    """One parsed workflow definition, root or child.

    Args:
        inputs: Inputs by name, in declaration order.
        steps: Steps in declaration order.
        outputs: Outputs in declaration order.
        dynamic_blocks: Raw ``dynamic_blocks_definitions`` entries.
        location: Definition path prefix; ``""`` for the root.
    """

    inputs: Mapping[str, WorkflowInputDeclaration]
    steps: Tuple[StepDeclaration, ...]
    outputs: Tuple[WorkflowOutputDeclaration, ...]
    dynamic_blocks: Tuple[Any, ...]
    location: str

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
    version = definition.get("version")
    if version != SUPPORTED_VERSION:
        raise WorkflowCompileError(
            f"{location}version must be {SUPPORTED_VERSION!r} for the V2 engine, got "
            f"{version!r}; V1 definitions run on the V1 ExecutionEngine"
        )

    sections: Dict[str, List[Any]] = {}
    for section in ("inputs", "steps", "outputs", "dynamic_blocks_definitions"):
        value = definition.get(section)
        value = [] if value is None else value
        if not isinstance(value, list):
            raise WorkflowCompileError(
                f"{location}{section} must be a list, got {type(value).__name__}"
            )
        sections[section] = value

    declaration = WorkflowDeclaration(
        inputs=MappingProxyType(_parse_inputs(sections["inputs"], location=location)),
        steps=_parse_steps(sections["steps"], location=location),
        outputs=_parse_outputs(sections["outputs"], location=location),
        dynamic_blocks=tuple(sections["dynamic_blocks_definitions"]),
        location=location,
    )

    return declaration


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
            ``$steps.<step>.<output>`` or ``$steps.<step>.*``.
    """
    try:
        is_data_selector = parse_selector(value).target != "step"
    except SelectorError:
        is_data_selector = False
    if not is_data_selector:
        raise SelectorError(
            f"{location} holds malformed selector {value!r}; use $inputs.<name>, "
            "$steps.<step>.<output> or $steps.<step>.*",
            step_path=step_path,
            field_path=field_path,
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
    if kind == AXIS_KIND_TIME:
        raise WorkflowCompileError(
            f"{location} declares a time axis; temporal execution is not supported "
            "by the sequential V2 engine"
        )
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


def _parse_outputs(
    raw_outputs: List[Any], *, location: str
) -> Tuple[WorkflowOutputDeclaration, ...]:
    outputs: List[WorkflowOutputDeclaration] = []
    names = set()
    for position, raw in enumerate(raw_outputs):
        where = f"{location}outputs[{position}]"
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
