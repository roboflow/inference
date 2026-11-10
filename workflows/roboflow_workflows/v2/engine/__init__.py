"""Opt-in sequential Workflows V2 engine and class-owned block API.

Register block classes in a ``Catalogue``, compile a version ``2.0`` definition,
then create an execution session. Compilation is structural; the session owns
block instances and preserves their state across runs::

    plan = compile_workflow(definition, catalogue=catalogue)
    with plan.create_session(resources=resources) as session:
        rows = session.run(inputs).rows()

A definition that declares ``sources`` (class-owned ``Source`` plugins
registered in the same catalogue) is active: the session opens the sources and
delivers each ``OutputGroup`` to a registered handler once per pulse::

    with plan.create_session(resources=resources) as session:
        with session.start(static_inputs, handlers={"frames": on_frames}) as run:
            run.wait()

Video sources can be selected explicitly with the same plugin mechanism::

    catalogue = Catalogue.from_modules([
        "roboflow_workflows.v2.blocks",
        "streamvision.workflows_v2",
    ])

This optional video module needs ``streamvision[workflows]``. The session context
closes session-owned state; finish active runs and passive pipelines before
leaving it. Caller-supplied model and service resources remain caller-owned.

Root ``operators`` (class-owned ``Operator`` plugins, such as the built-in
``roboflow_core/align@v1`` and ``roboflow_core/window@v1``) turn pulses of sources into pulses of
their own domain; ``$operators.<name>.<port>`` selects their outputs.

A block declares the events it may emit (``events = {"entered": Event(...)}``)
and calls ``self.emit(...)``. Definition ``handlers`` run their own passive
workflows on those events or on declared ``signals``; ``state`` declares
initial values of the session's ``ManagedState`` (the reserved
``managed_state`` resource). ``state_machines`` move one managed record per
source (or one global record) on events and on ``roboflow_core/state_machine_set@v1`` steps
of their handlers. The state package loads on first use.

A block may list alternative implementations; ``CompileOptions(target=...)``
selects one per step from caller-declared capabilities, and
``block_execution="phases"`` runs a selected implementation's phase graph
instead of its ``run``.

A host may add implementations to an existing block without changing it:
``catalogue.with_contributions([ImplementationContribution(...)])`` returns a
new catalogue whose block lists them first. The contribution names the block's
canonical type and ``contract_digest``; selection, plan pinning and update
comparison treat the added implementations like the block's own.

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

from roboflow_workflows.v2.data.batch import (
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
from roboflow_workflows.v2.data.kinds import Kind
from roboflow_workflows.v2.engine.active.runtime import (
    ActiveRun,
    ActiveRunError,
    GroupHandler,
    GroupResult,
    SourceCounters,
)
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import (
    DemandPlan,
    WorkflowReference,
    compile_workflow,
)
from roboflow_workflows.v2.engine.context import (
    ExecutionContext,
    NoExecutionContextError,
    current_pulse_run_id,
    get_execution_context,
)
from roboflow_workflows.v2.engine.contributions import (
    ImplementationContribution,
    contract_digest,
)
from roboflow_workflows.v2.engine.controls import (
    ControlPanel,
    ControlPlan,
    ControlReceipt,
    ControlSnapshot,
    ControlView,
    PlannedControl,
)
from roboflow_workflows.v2.engine.declaration import (
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
from roboflow_workflows.v2.engine.errors import (
    ContractError,
    ControlDefinitionError,
    ControlError,
    DemandError,
    EventEmissionError,
    GraphUpdateError,
    OperatorError,
    OperatorInputError,
    ReactionError,
    ResourceError,
    SessionClosedError,
    StepExecutionError,
    WorkflowCompileError,
    WorkflowExecutionError,
    WorkflowInputError,
)
from roboflow_workflows.v2.engine.events import Event, EventPayloadError
from roboflow_workflows.v2.engine.implementations import (
    Implementation,
    ImplementationOrigin,
    ImplementationSpec,
)
from roboflow_workflows.v2.engine.observer import ReactionObserver
from roboflow_workflows.v2.engine.operators import (
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
from roboflow_workflows.v2.engine.phases import (
    PhaseFailure,
    PhaseGraph,
    PhaseSpec,
    phase,
    read_phase_graph,
    run_phases,
)
from roboflow_workflows.v2.engine.pipelining.options import (
    OverloadPolicy,
    PipelineOptions,
)
from roboflow_workflows.v2.engine.pipelining.stages import PipelineCounters
from roboflow_workflows.v2.engine.plan import (
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
from roboflow_workflows.v2.engine.reactions import (
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
from roboflow_workflows.v2.engine.resources import Factory
from roboflow_workflows.v2.engine.sources import (
    Emission,
    Source,
    SourceDeclarationError,
    SourceOutput,
    SourceParams,
    SourceSpec,
    spec_of_source,
)
from roboflow_workflows.v2.engine.targets import (
    ImplementationChoice,
    QualityRequest,
    QualitySettings,
    Target,
    UnsupportedQualityError,
    UnsupportedTargetError,
    UnusedQualityHint,
    select_implementation,
)

__all__ = [
    "ActiveRun",
    "ActiveRunError",
    "Arrival",
    "Axis",
    "Batch",
    "Block",
    "BlockParams",
    "CONTEXT_POLICIES",
    "Catalogue",
    "CodecRegistry",
    "CompileOptions",
    "CompiledRetrospective",
    "CompiledWorkflow",
    "ContextPolicy",
    "ContractError",
    "ControlDefinitionError",
    "ControlError",
    "ControlPanel",
    "ControlPlan",
    "ControlReceipt",
    "ControlSnapshot",
    "ControlView",
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
    "GraphUpdateError",
    "Group",
    "GroupHandler",
    "GroupResult",
    "Implementation",
    "ImplementationChoice",
    "ImplementationContribution",
    "ImplementationOrigin",
    "ImplementationSpec",
    "Index",
    "InputValue",
    "Kind",
    "MANAGED_STATE_RESOURCE",
    "MISSING",
    "MachineStamp",
    "ManagedState",
    "NoExecutionContextError",
    "Operator",
    "OperatorCounters",
    "OperatorDeclarationError",
    "OperatorError",
    "OperatorInput",
    "OperatorInputError",
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
    "PlannedControl",
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
    "ResourceError",
    "ResultFiles",
    "RetrospectiveError",
    "RetrospectiveOutcome",
    "RunResult",
    "SAME_PAYLOAD",
    "SYSTEM_EVENTS",
    "SampleContext",
    "Select",
    "Selected",
    "Selection",
    "SessionClosedError",
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
    "StepExecutionError",
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
    "UnusedQualityHint",
    "UpdateReceipt",
    "WorkflowCompileError",
    "WorkflowExecutionError",
    "WorkflowInputError",
    "WorkflowReference",
    "WorkflowsBuffer",
    "compile_workflow",
    "contract_digest",
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

_STATE_MODULE = "roboflow_workflows.v2.engine.state"
_RECORDING_MODULE = "roboflow_workflows.v2.engine.recording"
_RECORDING_ERRORS = f"{_RECORDING_MODULE}.errors"
_MACHINES_MODULE = "roboflow_workflows.v2.engine.reactions.machines"
_LAZY = {
    "UpdateReceipt": "roboflow_workflows.v2.engine.updates.prepared",
    "MachineStamp": _MACHINES_MODULE,
    "TransitionCounters": _MACHINES_MODULE,
    "TransitionResult": _MACHINES_MODULE,
    "MISSING": _STATE_MODULE,
    "ManagedState": _STATE_MODULE,
    "StateError": _STATE_MODULE,
    "StateScope": _STATE_MODULE,
    "PassivePipeline": "roboflow_workflows.v2.engine.pipelining.passive",
    "PipelineAbortedError": "roboflow_workflows.v2.engine.pipelining.passive",
    "PipelineFullError": "roboflow_workflows.v2.engine.pipelining.passive",
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
