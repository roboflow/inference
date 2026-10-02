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
    OperatorError,
    OperatorInputError,
    WorkflowCompileError,
    WorkflowExecutionError,
    WorkflowInputError,
)
from roboflow_workflows.execution_engine.v2.implementations import (
    Implementation,
    ImplementationSpec,
)
from roboflow_workflows.execution_engine.v2.kinds import Kind
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
    Target,
    UnsupportedTargetError,
    select_implementation,
)

__all__ = [
    "CONTEXT_POLICIES",
    "SAME_PAYLOAD",
    "ActiveRun",
    "ActiveRunError",
    "Arrival",
    "Axis",
    "Batch",
    "Block",
    "BlockParams",
    "Catalogue",
    "CompiledWorkflow",
    "CompileOptions",
    "ContextPolicy",
    "ContractError",
    "DependentResource",
    "Emission",
    "EntryLayout",
    "EntryMetadata",
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
    "PhaseFailure",
    "PhaseGraph",
    "PhaseSpec",
    "PipelineAbortedError",
    "PipelineCounters",
    "PipelineFullError",
    "PipelineOptions",
    "PlannedOperator",
    "PlannedOperatorInput",
    "PlannedOutputGroup",
    "PlannedSource",
    "PlannedSourceOutput",
    "PulseKey",
    "Ref",
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
    "StepRef",
    "Stop",
    "Target",
    "TemporalContext",
    "TimeSpan",
    "Timestamp",
    "UnsupportedTargetError",
    "WorkflowCompileError",
    "WorkflowExecutionError",
    "WorkflowInputError",
    "WorkflowReference",
    "WorkflowsBuffer",
    "compile_workflow",
    "current_pulse_run_id",
    "get_execution_context",
    "phase",
    "read_phase_graph",
    "run_phases",
    "select_implementation",
    "spec_of",
    "spec_of_operator",
    "spec_of_source",
    "validate_entry",
]

_LAZY = {
    "PassivePipeline": "roboflow_workflows.execution_engine.v2.pipelining.passive",
    "PipelineAbortedError": "roboflow_workflows.execution_engine.v2.pipelining.passive",
    "PipelineFullError": "roboflow_workflows.execution_engine.v2.pipelining.passive",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(importlib.import_module(module), name)

    return value
