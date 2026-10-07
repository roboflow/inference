"""The root ``retrospective`` stage: analyse a recording of the plan, repeatedly.

One definition holds the primary workflow, its ``recording`` and one
retrospective stage reading exactly one recorded group::

    "retrospective": {"type": "workflow", "input_group": "analysis",
                      "workflow": {"inputs": [...], "steps": [...],
                                   "outputs": [<one OutputGroup>]}}
    "retrospective": {"type": "python", "input_group": "analysis",
                      "access": "all" | "chunks", "parameters": {...},
                      "result_directory": "$inputs.results_dir"}

    compile_workflow ──▶ the stage is compiled beside the primary plan: a
                         workflow stage is an active plan whose only source
                         is the engine's replay source, $sources.<input_group>
    plan.retrospective.start(recording=...)       fresh session per call:
        open + schema check ──▶ new block, operator and managed state
        ──▶ ActiveRun (one pulse per chunk); primary blocks, sources and
            resources are never constructed or called
    plan.retrospective.run_python(function, ...)  see ``results.py``

Stage ``inputs`` belong to the retrospective workflow; ``root_inputs`` resolve
the root definition's ``$inputs.<name>`` settings (recording directory,
parameters, result directory). They are never merged. An explicit
``recording=`` overrides ``recording.directory``; pass one or the other.
"""

import importlib
import inspect
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import (
    SUPPORTED_VERSION,
    compile_definition,
)
from roboflow_workflows.execution_engine.v2.errors import ContractError
from roboflow_workflows.execution_engine.v2.pipelining.options import PipelineOptions
from roboflow_workflows.execution_engine.v2.plan import (
    ACTIVE_RUNTIME_MODULE,
    MANAGED_STATE_RESOURCE,
    CompiledWorkflow,
    CompileOptions,
    ExecutionObserver,
    PlannedInput,
    requests_managed_state,
)
from roboflow_workflows.execution_engine.v2.recording.codecs import CodecRegistry
from roboflow_workflows.execution_engine.v2.recording.compilation import (
    RecordingPlan,
    StaticSetting,
    parse_static_setting,
    resolve_root_inputs,
)
from roboflow_workflows.execution_engine.v2.recording.errors import (
    RecordingDefinitionError,
)
from roboflow_workflows.execution_engine.v2.recording.replay import (
    RECORDING_RESOURCE,
    REPLAY_SOURCE_TYPE,
    make_replay_source,
)
from roboflow_workflows.execution_engine.v2.recording.results import (
    PYTHON_ACCESS_MODES,
    RetrospectiveOutcome,
    run_python_stage,
)
from roboflow_workflows.execution_engine.v2.recording.schema import (
    RecordedGroupSchema,
)
from roboflow_workflows.execution_engine.v2.recording.store import (
    RecordedGroup,
    Recording,
    open_recording,
)

__all__ = ["CompiledRetrospective", "compile_retrospective"]

STAGE_TYPES = ("workflow", "python")
WORKFLOW_LOCATION = "retrospective.workflow."
STATE_API_MODULE = "roboflow_workflows.execution_engine.v2.state.api"
_WORKFLOW_KEYS = frozenset({"type", "input_group", "workflow"})
_PYTHON_KEYS = frozenset(
    {"type", "input_group", "access", "parameters", "result_directory"}
)


@dataclass(frozen=True)
class CompiledRetrospective:
    """A compiled retrospective stage over one recorded group.

    Args:
        kind: ``workflow`` or ``python``.
        input_group: The recorded group the stage reads.
        group_schema: Contract a recording must match to be analysed.
        recording: The primary plan's recording declaration.
        root_inputs: Root inputs of the primary plan, for its settings.
        catalogue: The primary plan's catalogue; its codecs decode payloads.
        plan: Active plan of a workflow stage; ``None`` for Python.
        access: ``all`` or ``chunks`` for a Python stage; ``None`` otherwise.
        parameters: Python stage parameters: literals or root inputs.
        result_directory: Python stage result directory.
    """

    kind: str
    input_group: str
    group_schema: RecordedGroupSchema
    recording: RecordingPlan
    root_inputs: Mapping[str, PlannedInput] = field(repr=False)
    catalogue: Catalogue = field(repr=False, compare=False)
    plan: Optional[CompiledWorkflow] = field(default=None, repr=False)
    access: Optional[str] = None
    parameters: Mapping[str, StaticSetting] = field(
        default_factory=lambda: MappingProxyType({})
    )
    result_directory: Optional[StaticSetting] = None

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description: Dict[str, Any] = {
            "type": self.kind,
            "input_group": self.input_group,
        }
        if self.plan is not None:
            description["workflow"] = self.plan.describe()
        else:
            description["access"] = self.access
            description["parameters"] = {
                name: setting.describe() for name, setting in self.parameters.items()
            }
            description["result_directory"] = self.result_directory.describe()

        return description

    def start(
        self,
        *,
        recording: Optional[Union[str, Path]] = None,
        inputs: Optional[Mapping[str, Any]] = None,
        root_inputs: Optional[Mapping[str, Any]] = None,
        handlers: Optional[Mapping[str, Callable[[Any], None]]] = None,
        resources: Optional[Mapping[str, Any]] = None,
        pipeline: Optional[PipelineOptions] = None,
        admission_bound: int = 2,
        observer: Optional[ExecutionObserver] = None,
    ) -> Any:
        """Replay the recorded group through the retrospective workflow.

        Every call builds a new session: fresh block and operator instances
        and, when the workflow uses it, a fresh managed state, kept across
        the chunks of this replay and released when the run is done.

        Args:
            recording: Recording directory. It overrides the definition's
                ``recording.directory``; then ``root_inputs`` is needed only
                for other settings. ``None`` uses ``recording.directory``.
            inputs: Values of the retrospective workflow's own inputs.
            root_inputs: Values of root inputs the definition's settings
                select; only those are needed.
            handlers: Callback per output group name of the workflow.
            resources: Resources of the retrospective workflow's blocks.
                Managed state cannot be passed: a replay owns its own.
            pipeline: ``None`` replays serially; options pipeline the replay.
                The replay source must use the ``block`` overload policy:
                ``latest`` would drop recorded chunks.
            admission_bound: Chunks admitted but not yet processed.
            observer: Receives the replay's run notifications.

        Returns:
            The running ``ActiveRun``; ``wait()`` it like any active run.

        Raises:
            ContractError: For a Python stage, reserved resources or a
                ``latest`` overload policy for the replay source; nothing is
                constructed then.
            RecordingIncompleteError: When the recording was not finalized
                as ``complete`` or ``stopped``.
            RecordingSchemaError: When the recorded group differs from
                ``group_schema``; nothing is constructed then.
            RecordingError: When the recording cannot be opened.
        """
        if self.plan is None:
            raise ContractError(
                f"retrospective of group {self.input_group!r} is a Python stage; "
                "call run_python(function, ...) instead of start()"
            )

        if pipeline is not None and pipeline.overload_for(self.input_group) == "latest":
            raise ContractError(
                f"replay of recorded group {self.input_group!r} cannot use overload "
                "policy 'latest': it would drop recorded chunks. Use 'block' for "
                f"$sources.{self.input_group} (PipelineOptions overload or "
                "source_overload)"
            )
        provided = _replay_resources(resources)
        values = self._root_values(root_inputs, recording=recording)
        _, group = self._open(recording, values=values)
        release = _Release()
        if requests_managed_state(self.plan):
            state_api = importlib.import_module(STATE_API_MODULE)
            state = state_api.ManagedState(namespace=f"replay-{uuid.uuid4().hex}")
            release.add(state.close)
            provided[MANAGED_STATE_RESOURCE] = state
        provided[RECORDING_RESOURCE] = group
        try:
            session = self.plan.create_session(provided, observer=observer)
            release.add(session.close)
            runtime = importlib.import_module(ACTIVE_RUNTIME_MODULE)
            run = runtime.start_session(
                session,
                inputs=inputs if inputs is not None else {},
                handlers=handlers if handlers is not None else {},
                admission_bound=admission_bound,
                pipeline=pipeline,
                finalize=release,
            )
        except BaseException:
            release()
            raise

        return run

    def run_python(
        self,
        function: Callable[..., Any],
        *,
        recording: Optional[Union[str, Path]] = None,
        root_inputs: Optional[Mapping[str, Any]] = None,
    ) -> RetrospectiveOutcome:
        """Run a trusted host function over the recorded group.

        Args:
            function: Synchronous ``function(data, results, parameters)``;
                ``data`` is the recorded group (``access: all``) or one
                chunk (``access: chunks``).
            recording: Recording directory. It overrides the definition's
                ``recording.directory``; then ``root_inputs`` is needed only
                for other settings. ``None`` uses ``recording.directory``.
            root_inputs: Values of root inputs the definition's settings
                select; only those are needed.

        Returns:
            Registered result files, chunk count and recording status.

        Raises:
            ContractError: For a workflow stage or a coroutine function.
            RecordingIncompleteError: When the recording was not finalized
                as ``complete`` or ``stopped``.
            RecordingSchemaError: When the recorded group differs from
                ``group_schema``; ``function`` is not called then.
            RetrospectiveError: When ``function`` fails, with the chunk index
                in ``chunks`` access.
        """
        if self.plan is not None:
            raise ContractError(
                f"retrospective of group {self.input_group!r} is a workflow stage; "
                "call start(...) instead of run_python()"
            )
        if not callable(function) or _is_async_callable(function):
            raise ContractError(
                f"run_python needs a synchronous callable, got {function!r}"
            )

        needed = tuple(
            setting.input
            for setting in (*self.parameters.values(), self.result_directory)
            if setting.input is not None
        )
        values = self._root_values(root_inputs, recording=recording, extra=needed)
        recording_files, group = self._open(recording, values=values)
        parameters = MappingProxyType(
            {name: setting.resolve(values) for name, setting in self.parameters.items()}
        )
        directory = self.result_directory.resolve(values)
        if not isinstance(directory, (str, Path)) or not str(directory):
            raise ContractError(
                f"retrospective.result_directory "
                f"{self.result_directory.describe()!r} resolved to {directory!r}; "
                "expected a non-empty path"
            )

        outcome = run_python_stage(
            function,
            access=self.access,
            group=group,
            recording_status=recording_files.status,
            parameters=parameters,
            result_directory=Path(directory),
        )

        return outcome

    def _root_values(
        self,
        root_inputs: Optional[Mapping[str, Any]],
        *,
        recording: Optional[Union[str, Path]],
        extra: Tuple[str, ...] = (),
    ) -> Dict[str, Any]:
        directory_input = self.recording.directory.input
        needed = [*extra]
        if recording is None and directory_input is not None:
            needed.append(directory_input)
        values = resolve_root_inputs(
            self.root_inputs, supplied=root_inputs, needed=tuple(dict.fromkeys(needed))
        )

        return values

    def _open(
        self, recording: Optional[Union[str, Path]], *, values: Mapping[str, Any]
    ) -> Tuple[Recording, RecordedGroup]:
        """Open the recording and check the group before anything is built."""
        directory = recording
        if directory is None:
            directory = self.recording.directory.resolve(values)
        if not isinstance(directory, (str, Path)) or not str(directory):
            selected = self.recording.directory.input
            remedy = "pass recording=<dir>"
            if selected is not None:
                remedy += f" or root_inputs={{{selected!r}: <dir>}}"
            raise ContractError(
                f"recording directory resolved to {directory!r}; {remedy}"
            )

        opened = open_recording(
            Path(directory), codecs=CodecRegistry(self.catalogue.codecs.values())
        )
        group = opened.group(self.input_group)
        group.schema.require_compatible(self.group_schema)

        return opened, group


def compile_retrospective(
    raw: Mapping[str, Any],
    *,
    recording: RecordingPlan,
    root: CompiledWorkflow,
    catalogue: Catalogue,
    options: CompileOptions,
    reference_resolver: Any,
) -> CompiledRetrospective:
    """Compile the root ``retrospective`` declaration.

    Args:
        raw: The declaration.
        recording: The compiled ``recording`` declaration.
        root: The compiled primary plan.
        catalogue: The catalogue of the definition.
        options: Compile options of the definition.
        reference_resolver: Resolver of saved child workflows.

    Returns:
        The compiled stage.

    Raises:
        RecordingDefinitionError: On a malformed declaration, an unrecorded
            input group, a host-side ``entrypoint``, a workflow declaring
            sources or not exactly one output group, or a group a workflow
            cannot replay (wildcard fields).
        WorkflowCompileError: When the retrospective workflow does not
            compile. Definition-shape errors are located under
            ``retrospective.workflow.``; block type, selector and wiring
            errors name the retrospective step, as in any workflow (e.g.
            ``SelectorError``
            for ``$steps.<primary step>``: the workflow reads recorded fields
            as ``$sources.<input_group>.<field>``).
    """
    if "entrypoint" in raw:
        raise RecordingDefinitionError(
            "retrospective.entrypoint is not supported: the engine never imports "
            "code named by a definition; pass the function to "
            "plan.retrospective.run_python(function, ...)"
        )
    kind = raw.get("type")
    if kind not in STAGE_TYPES:
        raise RecordingDefinitionError(
            f"retrospective.type must be one of {list(STAGE_TYPES)}, got {kind!r}"
        )
    allowed = _WORKFLOW_KEYS if kind == "workflow" else _PYTHON_KEYS
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise RecordingDefinitionError(
            f"retrospective of type {kind!r} has unsupported keys {unknown}; "
            f"supported keys: {sorted(allowed)}"
        )
    input_group = raw.get("input_group")
    if not isinstance(input_group, str) or input_group not in recording.groups:
        raise RecordingDefinitionError(
            f"retrospective.input_group must name one recorded group "
            f"{list(recording.groups)}, got {input_group!r}"
        )

    common = dict(
        input_group=input_group,
        group_schema=recording.schema.group(input_group),
        recording=recording,
        root_inputs=root.inputs,
        catalogue=root.catalogue,
    )
    if kind == "workflow":
        plan = _compile_replay_workflow(
            raw.get("workflow"),
            schema=common["group_schema"],
            root=root,
            catalogue=catalogue,
            options=options,
            reference_resolver=reference_resolver,
        )
        compiled = CompiledRetrospective(kind=kind, plan=plan, **common)
        return compiled

    access = raw.get("access")
    if access not in PYTHON_ACCESS_MODES:
        raise RecordingDefinitionError(
            f"retrospective.access must be one of {list(PYTHON_ACCESS_MODES)}, got "
            f"{access!r}"
        )
    parameters = _python_parameters(raw.get("parameters"), inputs=root.inputs)
    result_directory = parse_static_setting(
        raw.get("result_directory"),
        inputs=root.inputs,
        location="retrospective.result_directory",
    )
    if result_directory.input is None and (
        not isinstance(result_directory.value, str) or not result_directory.value
    ):
        raise RecordingDefinitionError(
            "retrospective.result_directory must be a non-empty path or "
            f"$inputs.<name>, got {result_directory.value!r}"
        )
    compiled = CompiledRetrospective(
        kind=kind,
        access=access,
        parameters=parameters,
        result_directory=result_directory,
        **common,
    )

    return compiled


def _compile_replay_workflow(
    raw: Any,
    *,
    schema: RecordedGroupSchema,
    root: CompiledWorkflow,
    catalogue: Catalogue,
    options: CompileOptions,
    reference_resolver: Any,
) -> CompiledWorkflow:
    """The retrospective workflow, with the replay source as its only source."""
    if not isinstance(raw, Mapping):
        raise RecordingDefinitionError(
            f"retrospective.workflow must be a workflow definition mapping, got "
            f"{type(raw).__name__}"
        )
    if raw.get("sources"):
        raise RecordingDefinitionError(
            "retrospective.workflow declares sources; the engine supplies its only "
            f"source, $sources.{schema.name}, from the recording"
        )

    replay_source = make_replay_source(schema, catalogue=root.catalogue)
    try:
        replay_catalogue = Catalogue.merge(
            catalogue, Catalogue(sources=[replay_source])
        )
    except ContractError as error:
        raise RecordingDefinitionError(
            f"the replay source {REPLAY_SOURCE_TYPE!r} cannot join the catalogue: "
            f"{error}"
        ) from error
    definition = {
        "version": SUPPORTED_VERSION,
        **raw,
        "sources": [{"type": REPLAY_SOURCE_TYPE, "name": schema.name}],
    }
    plan = compile_definition(
        definition,
        catalogue=replay_catalogue,
        options=options,
        reference_resolver=reference_resolver,
        location=WORKFLOW_LOCATION,
    )
    groups = [group.name for group in plan.output_groups]
    handler_groups = [group.name for group in plan.reactions.groups]
    if len(groups) != 1 or handler_groups:
        raise RecordingDefinitionError(
            f"{WORKFLOW_LOCATION}outputs must declare exactly one OutputGroup and "
            f"no handler groups; found groups {groups} and handler groups "
            f"{handler_groups}"
        )

    return plan


def _python_parameters(
    raw: Any, *, inputs: Mapping[str, PlannedInput]
) -> Mapping[str, StaticSetting]:
    if raw is None:
        return MappingProxyType({})
    if not isinstance(raw, Mapping):
        raise RecordingDefinitionError(
            f"retrospective.parameters must be a mapping, got {type(raw).__name__}"
        )

    parameters = {
        name: parse_static_setting(
            value, inputs=inputs, location=f"retrospective.parameters.{name}"
        )
        for name, value in raw.items()
    }

    return MappingProxyType(parameters)


def _replay_resources(resources: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Caller resources, rejecting the reserved recording and managed state."""
    provided = dict(resources or {})
    reserved = sorted(
        key
        for key, value in provided.items()
        if key == RECORDING_RESOURCE
        or (
            value is not None
            and (
                key == MANAGED_STATE_RESOURCE
                or key.endswith(f".{MANAGED_STATE_RESOURCE}")
            )
        )
    )
    if reserved:
        raise ContractError(
            f"resources {reserved} cannot be passed to a replay: it reads the "
            "recording itself and owns a fresh managed state, so caller state is "
            "neither shared nor cleared"
        )

    return provided


def _is_async_callable(target: Any) -> bool:
    call = target if inspect.isroutine(target) else getattr(target, "__call__", None)
    is_async = call is not None and (
        inspect.iscoroutinefunction(call) or inspect.isasyncgenfunction(call)
    )

    return is_async


class _Release:
    """Release what a replay created for itself, once, in reverse order."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._steps = []

    def add(self, step: Callable[[], None]) -> None:
        self._steps.append(step)

    def __call__(self) -> None:
        with self._lock:
            steps, self._steps = self._steps, []
        for step in reversed(steps):
            step()
