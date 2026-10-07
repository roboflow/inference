"""Opt-in sequential Workflows V2 engine and class-owned block API.

Register block classes in a ``Catalogue``, compile a version ``2.0`` definition,
then create an execution session. Compilation is structural; the session owns
block instances and preserves their state across runs::

    plan = compile_workflow(definition, catalogue=catalogue)
    session = plan.create_session(resources=resources)
    rows = session.run(inputs).rows()

A definition that declares ``sources`` (class-owned ``Source`` plugins
registered in the same catalogue) is active: the session opens the sources and
delivers each ``OutputGroup`` to a registered handler once per pulse::

    run = session.start(static_inputs, handlers={"frames": on_frames})
    run.wait()

Root ``operators`` (class-owned ``Operator`` plugins, such as the built-in
``v2/align@v1`` and ``v2/window@v1``) turn pulses of sources into pulses of
their own domain; ``$operators.<name>.<port>`` selects their outputs.

A block declares the events it may emit (``events = {"entered": Event(...)}``)
and calls ``self.emit(...)``. Definition ``handlers`` run their own passive
workflows on those events or on declared ``signals``; ``state`` declares
initial values of the session's ``ManagedState`` (the reserved
``managed_state`` resource). ``state_machines`` move one managed record per
source (or one global record) on events and on ``v2/state_machine_set`` steps
of their handlers. The state package loads on first use.

A block may list alternative implementations; ``CompileOptions(target=...)``
selects one per step from caller-declared capabilities, and
``block_execution="phases"`` runs a selected implementation's phase graph
instead of its ``run``.

Serial execution is the default and the reference. ``PipelineOptions`` opts
into bounded pipelining: different pulses (or passive submissions) overlap at
different steps and phases, each stage taking one call at a time in order::

    run = session.start(inputs, handlers=handlers, pipeline=PipelineOptions())
    with session.pipeline(options=PipelineOptions(max_in_flight=2)) as pipeline:
        future = pipeline.submit({"image": image})

A root ``recording`` section makes the engine record named output groups of
every ``start`` into a local directory, with or without handlers. A root
``retrospective`` section analyses such a recording repeatedly without the
primary workflow: ``plan.retrospective.start(recording=...)`` replays it
through a workflow whose source is the recorded group, and
``plan.retrospective.run_python(function, ...)`` hands it to trusted host
code. Recording stages load on demand; the catalogue loads codecs, schema and store.

V1 defaults and discovery are unaffected. Native image blocks live in the
separately imported ``v2.blocks`` catalogue; this generic entry point loads no
image libraries, native block implementations or V1 engine. The lightweight
active runtime is imported to expose its public lifecycle and result types;
the passive pipeline (``PassivePipeline`` and its errors) loads on first use.
"""

import importlib
from typing import Any

from roboflow_workflows.execution_engine.v2.active.runtime import (
    ActiveRun,
    ActiveRunError,
    GroupHandler,
    GroupResult,
    SourceCounters,
)
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import (
    DemandPlan,
    WorkflowReference,
    compile_workflow,
)
from roboflow_workflows.execution_engine.v2.context import (
    ExecutionContext,
    NoExecutionContextError,
    current_pulse_run_id,
    get_execution_context,
)
from roboflow_workflows.execution_engine.v2.data import (
    Axis,
    Batch,
    EntryLayout,
    EntryMetadata,
    Index,
    InputValue,
    SampleContext,
    TemporalContext,
    TimeSpan,
    Timestamp,
    WorkflowsBuffer,
    validate_entry,
)
from roboflow_workflows.execution_engine.v2.declaration import (
    CONTEXT_POLICIES,
    SAME_PAYLOAD,
    Block,
    BlockParams,
    ContextPolicy,
    DependentResource,
    Group,
    Output,
    Ref,
    Select,
    Selected,
    Selection,
    StepRef,
    Stop,
    spec_of,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    DemandError,
    EventEmissionError,
    OperatorError,
    OperatorInputError,
    ReactionError,
    WorkflowCompileError,
    WorkflowExecutionError,
    WorkflowInputError,
)
from roboflow_workflows.execution_engine.v2.events import Event, EventPayloadError
from roboflow_workflows.execution_engine.v2.implementations import (
    Implementation,
    ImplementationSpec,
)
from roboflow_workflows.execution_engine.v2.kinds import Kind
from roboflow_workflows.execution_engine.v2.observer import ReactionObserver
from roboflow_workflows.execution_engine.v2.operators import (
    Arrival,
    Operator,
    OperatorCounters,
    OperatorDeclarationError,
    OperatorInput,
    OperatorParams,
    OperatorPort,
    OperatorPulse,
    OperatorSpec,
    spec_of_operator,
)
from roboflow_workflows.execution_engine.v2.phases import (
    PhaseFailure,
    PhaseGraph,
    PhaseSpec,
    phase,
    read_phase_graph,
    run_phases,
)
from roboflow_workflows.execution_engine.v2.pipelining.options import (
    OverloadPolicy,
    PipelineOptions,
)
from roboflow_workflows.execution_engine.v2.pipelining.stages import PipelineCounters
from roboflow_workflows.execution_engine.v2.plan import (
    MANAGED_STATE_RESOURCE,
    CompiledWorkflow,
    CompileOptions,
    ExecutionObserver,
    ExecutionSession,
    PlannedOperator,
    PlannedOperatorInput,
    PlannedOutputGroup,
    PlannedSource,
    PlannedSourceOutput,
    PulseKey,
    RunResult,
    SourcePort,
)
from roboflow_workflows.execution_engine.v2.reactions import (
    SYSTEM_EVENTS,
    EventOrigin,
    PlannedHandler,
    PlannedHandlerGroup,
    PlannedMachine,
    PlannedSignal,
    PlannedTransition,
    QueuePolicy,
    ReactionPlan,
    StateDefaults,
)
from roboflow_workflows.execution_engine.v2.resources import Factory
from roboflow_workflows.execution_engine.v2.sources import (
    Emission,
    Source,
    SourceDeclarationError,
    SourceOutput,
    SourceParams,
    SourceSpec,
    spec_of_source,
)
from roboflow_workflows.execution_engine.v2.targets import (
    ImplementationChoice,
    QualityRequest,
    QualitySettings,
    Target,
    UnsupportedQualityError,
    UnsupportedTargetError,
    select_implementation,
)

__all__ = [
    "CONTEXT_POLICIES",
    "MANAGED_STATE_RESOURCE",
    "MISSING",
    "SAME_PAYLOAD",
    "SYSTEM_EVENTS",
    "ActiveRun",
    "ActiveRunError",
    "Arrival",
    "Axis",
    "Batch",
    "Block",
    "BlockParams",
    "Catalogue",
    "CodecRegistry",
    "CompiledRetrospective",
    "CompiledWorkflow",
    "CompileOptions",
    "ContextPolicy",
    "ContractError",
    "DemandError",
    "DemandPlan",
    "DependentResource",
    "Emission",
    "EntryLayout",
    "EntryMetadata",
    "Event",
    "EventEmissionError",
    "EventOrigin",
    "EventPayloadError",
    "ExecutionContext",
    "ExecutionObserver",
    "ExecutionSession",
    "Factory",
    "Group",
    "GroupHandler",
    "GroupResult",
    "Implementation",
    "ImplementationChoice",
    "ImplementationSpec",
    "Index",
    "InputValue",
    "Kind",
    "MachineStamp",
    "ManagedState",
    "NoExecutionContextError",
    "Operator",
    "OperatorCounters",
    "OperatorDeclarationError",
    "OperatorError",
    "OperatorInputError",
    "OperatorInput",
    "OperatorParams",
    "OperatorPort",
    "OperatorPulse",
    "OperatorSpec",
    "Output",
    "OverloadPolicy",
    "PassivePipeline",
    "PayloadCodec",
    "PhaseFailure",
    "PhaseGraph",
    "PhaseSpec",
    "PipelineAbortedError",
    "PipelineCounters",
    "PipelineFullError",
    "PipelineOptions",
    "PlannedHandler",
    "PlannedHandlerGroup",
    "PlannedMachine",
    "PlannedOperator",
    "PlannedOperatorInput",
    "PlannedOutputGroup",
    "PlannedSignal",
    "PlannedSource",
    "PlannedSourceOutput",
    "PlannedTransition",
    "PulseKey",
    "QualityRequest",
    "QualitySettings",
    "QueuePolicy",
    "ReactionError",
    "ReactionObserver",
    "ReactionPlan",
    "RecordingCodecError",
    "RecordingCorruptError",
    "RecordingDefinitionError",
    "RecordingError",
    "RecordingExistsError",
    "RecordingIncompleteError",
    "RecordingPlan",
    "RecordingSchemaError",
    "Ref",
    "ResultFiles",
    "RetrospectiveError",
    "RetrospectiveOutcome",
    "RunResult",
    "SampleContext",
    "Select",
    "Selected",
    "Selection",
    "Source",
    "SourceCounters",
    "SourceDeclarationError",
    "SourceOutput",
    "SourceParams",
    "SourcePort",
    "SourceSpec",
    "StateDefaults",
    "StateError",
    "StateScope",
    "StepRef",
    "Stop",
    "Target",
    "TemporalContext",
    "TimeSpan",
    "Timestamp",
    "TransitionCounters",
    "TransitionResult",
    "UnsupportedQualityError",
    "UnsupportedTargetError",
    "WorkflowCompileError",
    "WorkflowExecutionError",
    "WorkflowInputError",
    "WorkflowReference",
    "WorkflowsBuffer",
    "compile_workflow",
    "current_pulse_run_id",
    "get_execution_context",
    "inspect_recording",
    "open_recording",
    "phase",
    "read_phase_graph",
    "run_phases",
    "select_implementation",
    "spec_of",
    "spec_of_operator",
    "spec_of_source",
    "validate_entry",
]

_STATE_MODULE = "roboflow_workflows.execution_engine.v2.state"
_RECORDING_MODULE = "roboflow_workflows.execution_engine.v2.recording"
_RECORDING_ERRORS = f"{_RECORDING_MODULE}.errors"
_MACHINES_MODULE = "roboflow_workflows.execution_engine.v2.reactions.machines"
_LAZY = {
    "MachineStamp": _MACHINES_MODULE,
    "TransitionCounters": _MACHINES_MODULE,
    "TransitionResult": _MACHINES_MODULE,
    "MISSING": _STATE_MODULE,
    "ManagedState": _STATE_MODULE,
    "StateError": _STATE_MODULE,
    "StateScope": _STATE_MODULE,
    "PassivePipeline": "roboflow_workflows.execution_engine.v2.pipelining.passive",
    "PipelineAbortedError": "roboflow_workflows.execution_engine.v2.pipelining.passive",
    "PipelineFullError": "roboflow_workflows.execution_engine.v2.pipelining.passive",
    "CodecRegistry": f"{_RECORDING_MODULE}.codecs",
    "PayloadCodec": f"{_RECORDING_MODULE}.codecs",
    "open_recording": f"{_RECORDING_MODULE}.store",
    "inspect_recording": f"{_RECORDING_MODULE}.store",
    "RecordingPlan": f"{_RECORDING_MODULE}.compilation",
    "CompiledRetrospective": f"{_RECORDING_MODULE}.retrospective",
    "ResultFiles": f"{_RECORDING_MODULE}.results",
    "RetrospectiveOutcome": f"{_RECORDING_MODULE}.results",
    "RecordingError": _RECORDING_ERRORS,
    "RecordingCodecError": _RECORDING_ERRORS,
    "RecordingCorruptError": _RECORDING_ERRORS,
    "RecordingDefinitionError": _RECORDING_ERRORS,
    "RecordingExistsError": _RECORDING_ERRORS,
    "RecordingIncompleteError": _RECORDING_ERRORS,
    "RecordingSchemaError": _RECORDING_ERRORS,
    "RetrospectiveError": _RECORDING_ERRORS,
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(importlib.import_module(module), name)

    return value
