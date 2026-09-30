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
    Block,
    BlockParams,
    DependentResource,
    Group,
    Output,
    Ref,
    Select,
    StepRef,
    Stop,
    spec_of,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    WorkflowCompileError,
    WorkflowExecutionError,
    WorkflowInputError,
)
from roboflow_workflows.execution_engine.v2.kinds import Kind
from roboflow_workflows.execution_engine.v2.plan import (
    CompiledWorkflow,
    CompileOptions,
    ExecutionObserver,
    ExecutionSession,
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
    "ActiveRun",
    "ActiveRunError",
    "Axis",
    "Batch",
    "Block",
    "BlockParams",
    "Catalogue",
    "CompiledWorkflow",
    "CompileOptions",
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
    "Output",
    "PlannedOutputGroup",
    "PlannedSource",
    "PlannedSourceOutput",
    "PulseKey",
    "Ref",
    "RunResult",
    "SampleContext",
    "Select",
    "Source",
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
    "spec_of_source",
    "validate_entry",
]
