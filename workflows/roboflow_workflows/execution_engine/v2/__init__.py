"""Opt-in sequential Workflows V2 engine and class-owned block API.

Register block classes in a ``Catalogue``, compile a version ``2.0`` definition,
then create an execution session. Compilation is structural; the session owns
block instances and preserves their state across runs::

    plan = compile_workflow(definition, catalogue=catalogue)
    session = plan.create_session(resources=resources)
    rows = session.run(inputs).rows()

V1 defaults and discovery are unaffected. Native image blocks live in the
separately imported ``v2.blocks`` catalogue; this generic entry point loads no
image libraries, native block implementations or V1 engine.
"""

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
    RunResult,
)
from roboflow_workflows.execution_engine.v2.resources import Factory

__all__ = [
    "Axis",
    "Batch",
    "Block",
    "BlockParams",
    "Catalogue",
    "CompiledWorkflow",
    "CompileOptions",
    "ContractError",
    "DependentResource",
    "EntryLayout",
    "EntryMetadata",
    "ExecutionContext",
    "ExecutionObserver",
    "ExecutionSession",
    "Factory",
    "Group",
    "Index",
    "InputValue",
    "Kind",
    "NoExecutionContextError",
    "Output",
    "Ref",
    "RunResult",
    "SampleContext",
    "Select",
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
    "validate_entry",
]
