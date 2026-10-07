"""Compilation of the root ``recording`` declaration, and the capture it starts.

A definition names the output groups the engine records on every ``start``::

    "recording": {"type": "file", "directory": "$inputs.capture_dir",
                  "groups": ["analysis"]}

    compile_workflow ──▶ compile_stages
                           RecordingPlan: directory, groups, schema, digest
                           CompiledRetrospective (retrospective.py), optional
    session.start   ──▶ start_capture: create the recording (before any source
                           opens); one recorder per recorded group
    group delivered ──▶ recorder(result), in the group's turn, before handlers
    run concluded   ──▶ capture.finish(status), before ActiveRun.wait returns

The schema of each recorded group comes from the plan's selected ports, so a
retrospective stage is checked against what the plan delivers, not against
the definition text. The definition digest is provenance only.
"""

import dataclasses
import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.execution.entries import Entry
from roboflow_workflows.execution_engine.v2.execution.outputs import (
    GroupResult,
    selected_ports,
)
from roboflow_workflows.execution_engine.v2.plan import (
    CompiledWorkflow,
    CompileOptions,
    PlannedInput,
    PlannedOutputGroup,
)
from roboflow_workflows.execution_engine.v2.recording.codecs import CodecRegistry
from roboflow_workflows.execution_engine.v2.recording.errors import (
    RecordingDefinitionError,
    RecordingError,
)
from roboflow_workflows.execution_engine.v2.recording.schema import (
    RecordedEntrySchema,
    RecordedGroupSchema,
    RecordingSchema,
)
from roboflow_workflows.execution_engine.v2.recording.store import RecordingWriter

__all__ = [
    "Capture",
    "RecordingPlan",
    "StaticSetting",
    "compile_stages",
    "group_schema",
    "start_capture",
]

RECORDING_TYPES = ("file",)
RETROSPECTIVE_MODULE = "roboflow_workflows.execution_engine.v2.recording.retrospective"
_RECORDING_KEYS = frozenset({"type", "directory", "groups"})
_INPUT_PREFIX = "$inputs."


@dataclass(frozen=True)
class StaticSetting:
    """A declaration value: a literal, or an ungrouped root input by name.

    Args:
        value: The literal; ``None`` when ``input`` names the source.
        input: Root input whose value (or default) is used; ``None`` for a
            literal.
    """

    value: Any = None
    input: Optional[str] = None

    def describe(self) -> Any:
        """Return the declaration form: the literal or ``$inputs.<name>``."""
        if self.input is not None:
            return f"{_INPUT_PREFIX}{self.input}"

        return self.value

    def resolve(self, values: Mapping[str, Any]) -> Any:
        """Return the literal, or the named input's value from ``values``.

        Args:
            values: Root input values by name, defaults applied.

        Returns:
            The setting's value.
        """
        if self.input is None:
            return self.value

        value = values[self.input]

        return value


@dataclass(frozen=True)
class RecordingPlan:
    """What every ``start`` of the plan records, and where.

    Args:
        directory: Recording destination; a new directory per run.
        groups: Recorded output groups, in declaration order.
        schema: Recorded contract of every group, from the plan's ports.
        definition_digest: SHA-256 of the definition without
            ``retrospective``; provenance only, never a compatibility check.
    """

    directory: StaticSetting
    groups: Tuple[str, ...]
    schema: RecordingSchema
    definition_digest: str

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {
            "type": "file",
            "directory": self.directory.describe(),
            "groups": list(self.groups),
            "schema": {name: self.schema.group(name).to_json() for name in self.groups},
            "definition_digest": self.definition_digest,
        }

        return description


class Capture:
    """The open recording of one active run.

    Args:
        writer: Writer of the run's new recording.
        groups: Names of the recorded groups.
    """

    def __init__(self, writer: RecordingWriter, *, groups: Tuple[str, ...]):
        self._writer = writer
        self.recorders: Mapping[str, Callable[[GroupResult], None]] = {
            name: writer.write for name in groups
        }

    @property
    def directory(self) -> Path:
        """Directory of the recording."""
        return self._writer.directory

    @property
    def counters(self) -> Mapping[str, Mapping[str, int]]:
        """Chunks and bytes written per group."""
        return self._writer.counters

    def finish(self, status: str, *, error: Optional[str] = None) -> None:
        """Finalize the recording with the run's outcome.

        Args:
            status: ``complete``, ``stopped``, ``failed`` or ``cancelled``.
            error: The run's failure, for ``failed``.
        """
        self._writer.finish(status, error=error)


def compile_stages(
    plan: CompiledWorkflow,
    *,
    definition: Mapping[str, Any],
    recording: Optional[Mapping[str, Any]],
    retrospective: Optional[Mapping[str, Any]],
    catalogue: Catalogue,
    options: CompileOptions,
    reference_resolver: Any,
) -> CompiledWorkflow:
    """Compile the root ``recording`` and ``retrospective`` declarations.

    Nothing is constructed or opened: the retrospective workflow is compiled
    like any definition, beside the primary plan.

    Args:
        plan: The compiled primary plan.
        definition: The root definition, for the provenance digest.
        recording: Raw ``recording`` declaration, or ``None``.
        retrospective: Raw ``retrospective`` declaration, or ``None``.
        catalogue: The catalogue the definition was compiled against.
        options: Compile options, reused for the retrospective workflow.
        reference_resolver: Resolver of saved workflows, likewise.

    Returns:
        ``plan`` with ``recording`` and ``retrospective`` set.

    Raises:
        RecordingDefinitionError: When a declaration is malformed, names an
            unknown or unsupported group, or the plan is not active.
        WorkflowCompileError: When the retrospective workflow does not
            compile.
    """
    if recording is None:
        raise RecordingDefinitionError(
            "retrospective needs a recording declaration naming the group it reads"
        )

    recording_plan = compile_recording(recording, plan=plan, definition=definition)
    compiled_retrospective = None
    if retrospective is not None:
        retrospective_module = importlib.import_module(RETROSPECTIVE_MODULE)
        compiled_retrospective = retrospective_module.compile_retrospective(
            retrospective,
            recording=recording_plan,
            root=plan,
            catalogue=catalogue,
            options=options,
            reference_resolver=reference_resolver,
        )
    staged = dataclasses.replace(
        plan, recording=recording_plan, retrospective=compiled_retrospective
    )

    return staged


def compile_recording(
    raw: Mapping[str, Any],
    *,
    plan: CompiledWorkflow,
    definition: Mapping[str, Any],
) -> RecordingPlan:
    """Check a ``recording`` declaration against the compiled plan.

    Args:
        raw: The declaration.
        plan: The compiled primary plan.
        definition: The root definition, for the provenance digest.

    Returns:
        The recording plan.

    Raises:
        RecordingDefinitionError: On unknown keys, an unsupported type, a
            bad directory or group list, or a passive plan.
    """
    unknown = sorted(set(raw) - _RECORDING_KEYS)
    if unknown:
        raise RecordingDefinitionError(
            f"recording has unsupported keys {unknown}; supported keys: "
            f"{sorted(_RECORDING_KEYS)}"
        )
    if raw.get("type") not in RECORDING_TYPES:
        raise RecordingDefinitionError(
            f"recording.type must be one of {list(RECORDING_TYPES)}, got "
            f"{raw.get('type')!r}"
        )
    if not plan.is_active:
        raise RecordingDefinitionError(
            "recording needs an active definition with sources; recording "
            "passive session.run() results is not supported"
        )

    directory = parse_static_setting(
        raw.get("directory"), inputs=plan.inputs, location="recording.directory"
    )
    if directory.input is None and (
        not isinstance(directory.value, str) or not directory.value
    ):
        raise RecordingDefinitionError(
            "recording.directory must be a non-empty path or $inputs.<name>, got "
            f"{directory.value!r}"
        )
    groups = _recorded_groups(raw.get("groups"), plan=plan)
    schema = RecordingSchema(
        {name: group_schema(plan, _output_group(plan, name)) for name in groups}
    )
    recording_plan = RecordingPlan(
        directory=directory,
        groups=groups,
        schema=schema,
        definition_digest=definition_digest(definition),
    )

    return recording_plan


def parse_static_setting(
    raw: Any, *, inputs: Mapping[str, PlannedInput], location: str
) -> StaticSetting:
    """Parse a literal or a ``$inputs.<name>`` reference to a root input.

    Args:
        raw: Declared value.
        inputs: Root inputs of the primary plan; all are ungrouped there.
        location: Declaration path for messages.

    Returns:
        The setting.

    Raises:
        RecordingDefinitionError: When a selector is not ``$inputs.<name>``
            of a declared, ungrouped input.
    """
    if not isinstance(raw, str) or not raw.startswith("$"):
        return StaticSetting(value=raw)

    name = raw[len(_INPUT_PREFIX) :] if raw.startswith(_INPUT_PREFIX) else None
    if name is None or name not in inputs:
        raise RecordingDefinitionError(
            f"{location} selects {raw!r}; only $inputs.<name> of a root input is "
            f"allowed here; root inputs: {sorted(inputs)}"
        )
    if inputs[name].layout.depth:
        raise RecordingDefinitionError(
            f"{location} selects grouped input {name!r}; a static setting needs "
            "an ungrouped input"
        )

    setting = StaticSetting(input=name)

    return setting


def resolve_root_inputs(
    inputs: Mapping[str, PlannedInput],
    *,
    supplied: Optional[Mapping[str, Any]],
    needed: Tuple[str, ...],
) -> Dict[str, Any]:
    """Values of the root inputs a stage setting reads; defaults applied.

    Only the named inputs are resolved, so a stage does not need the primary
    run's other required inputs (a model id, a camera address).

    Args:
        inputs: Root inputs of the primary plan.
        supplied: Values the caller passed as ``root_inputs``.
        needed: Inputs the settings select.

    Returns:
        Value per needed input.

    Raises:
        RecordingDefinitionError: When ``supplied`` names an undeclared
            input or a needed required input is missing.
    """
    supplied = dict(supplied or {})
    unknown = sorted(set(supplied) - set(inputs))
    if unknown:
        raise RecordingDefinitionError(
            f"root_inputs {unknown} are not inputs of the root definition; "
            f"declared: {sorted(inputs)}"
        )

    values: Dict[str, Any] = {}
    for name in needed:
        if name in supplied:
            values[name] = supplied[name]
            continue
        if inputs[name].required:
            raise RecordingDefinitionError(
                f"root input {name!r} is required to locate the recording or "
                "results; pass it in root_inputs"
            )
        values[name] = inputs[name].default

    return values


def group_schema(
    plan: CompiledWorkflow, group: PlannedOutputGroup
) -> RecordedGroupSchema:
    """Recorded contract of one output group, from its selected ports.

    Args:
        plan: Compiled plan declaring the group.
        group: The output group.

    Returns:
        Key, field, wildcard port, kinds and plan-scoped layout per entry.
    """
    entries = tuple(
        RecordedEntrySchema(
            key=port.key,
            field=port.output.name,
            port=port.name,
            kinds=tuple(port.kind_names),
            layout=port.layout,
        )
        for port in selected_ports(plan, group.outputs)
    )
    schema = RecordedGroupSchema(
        name=group.name, anchor_domain=group.source, entries=entries
    )

    return schema


def definition_digest(definition: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of a definition without ``retrospective``.

    Args:
        definition: Root definition.

    Returns:
        Hex digest; recorded for provenance, never compared for compatibility.
    """
    primary = {
        key: value for key, value in definition.items() if key != "retrospective"
    }
    canonical = json.dumps(primary, sort_keys=True, separators=(",", ":"), default=repr)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return digest


def start_capture(plan: CompiledWorkflow, *, inputs: Mapping[str, Entry]) -> Capture:
    """Create the run's recording at the plan's resolved directory.

    Args:
        plan: Active plan with ``recording``.
        inputs: The run's prepared static input entries.

    Returns:
        The open capture, status ``recording``.

    Raises:
        RecordingError: When the directory is not a path or already holds a
            recording or other files (``RecordingExistsError``).
    """
    recording = plan.recording
    values = {name: entry.values[()] for name, entry in inputs.items()}
    directory = recording.directory.resolve(values)
    if not isinstance(directory, (str, Path)) or not str(directory):
        raise RecordingError(
            f"recording.directory {recording.directory.describe()!r} resolved to "
            f"{directory!r}; expected a non-empty path"
        )

    writer = RecordingWriter.create(
        Path(directory),
        schema=recording.schema,
        codecs=CodecRegistry(plan.catalogue.codecs.values()),
        definition_digest=recording.definition_digest,
    )
    capture = Capture(writer, groups=recording.groups)

    return capture


def recorded_group_names(
    recording: Mapping[str, Any], *, plan: CompiledWorkflow
) -> Tuple[str, ...]:
    """Return the groups a raw ``recording`` declaration records, before compiling it.

    The demand pass needs them before the plan is narrowed, so a recorded
    group is demanded whole even when the caller requests other outputs.

    Args:
        recording: Raw root ``recording`` declaration.
        plan: The compiled primary plan, not yet narrowed.

    Returns:
        The recorded group names in declaration order; empty for a malformed
        declaration or a passive plan, which ``compile_recording`` rejects
        afterwards with its own message.
    """
    if not isinstance(recording, Mapping) or not plan.is_active:
        return ()

    try:
        groups = _recorded_groups(recording.get("groups"), plan=plan)
    except RecordingDefinitionError:
        return ()

    return groups


def _recorded_groups(raw: Any, *, plan: CompiledWorkflow) -> Tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise RecordingDefinitionError(
            f"recording.groups must be a non-empty list of output group names, got "
            f"{raw!r}"
        )

    declared = [group.name for group in plan.output_groups]
    handler_groups = {group.name for group in plan.reactions.groups}
    seen = []
    for position, name in enumerate(raw):
        where = f"recording.groups[{position}]"
        if not isinstance(name, str):
            raise RecordingDefinitionError(
                f"{where} must be an output group name, got {name!r}"
            )
        if name in handler_groups:
            raise RecordingDefinitionError(
                f"{where} names handler output group {name!r}; recording groups "
                "anchored at $handlers is not supported, record a source or "
                "operator group"
            )
        if name not in declared:
            raise RecordingDefinitionError(
                f"{where} names unknown output group {name!r}; declared groups: "
                f"{declared}"
            )
        if name in seen:
            raise RecordingDefinitionError(f"{where} repeats group {name!r}")
        seen.append(name)

    return tuple(seen)


def _output_group(plan: CompiledWorkflow, name: str) -> PlannedOutputGroup:
    group = next(group for group in plan.output_groups if group.name == name)

    return group
