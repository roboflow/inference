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

V1 defaults and discovery are unaffected. Native image blocks live in the
separately imported ``v2.blocks`` catalogue; this generic entry point loads no
image libraries, native block implementations or V1 engine. The lightweight
active runtime is imported to expose its public lifecycle and result types.
"""

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
    "TemporalContext",
    "TimeSpan",
    "Timestamp",
    "WorkflowCompileError",
    "WorkflowExecutionError",
    "WorkflowInputError",
    "WorkflowReference",
    "WorkflowsBuffer",
    "compile_workflow",
    "get_execution_context",
    "spec_of",
    "spec_of_operator",
    "spec_of_source",
    "validate_entry",
]
