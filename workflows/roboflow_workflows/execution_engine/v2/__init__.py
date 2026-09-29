"""Explicit entry point of the passive Workflows V2 execution engine.

Importing this package does not register anything with the legacy V1 engine,
its block loader or ``REGISTERED_ENGINES``. Callers select V2 by importing
``compile_workflow`` from here and by passing an explicit ``Registry``. The
native image block catalogue lives in ``roboflow_workflows.execution_engine.v2.blocks``
and is never imported by this generic core.
"""

from roboflow_workflows.execution_engine.v2.compiler import (
    CompiledInput,
    CompiledOutput,
    CompiledPort,
    CompiledStep,
    CompiledWorkflow,
    CompiledWorkflowOutput,
    compile_workflow,
)
from roboflow_workflows.execution_engine.v2.contracts import (
    BlockContract,
    BlockRegistration,
    InputSpec,
    OutputSpec,
    Registry,
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
    Timestamp,
    TimeSpan,
    WorkflowsBuffer,
    validate_entry,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    WorkflowCompileError,
    WorkflowExecutionError,
)
from roboflow_workflows.execution_engine.v2.executor import RunResult, execute_plan

__all__ = [
    "Axis",
    "Batch",
    "BlockContract",
    "BlockRegistration",
    "CompiledInput",
    "CompiledOutput",
    "CompiledPort",
    "CompiledStep",
    "CompiledWorkflow",
    "CompiledWorkflowOutput",
    "ContractError",
    "EntryLayout",
    "EntryMetadata",
    "Index",
    "InputSpec",
    "InputValue",
    "OutputSpec",
    "Registry",
    "RunResult",
    "SampleContext",
    "TemporalContext",
    "TimeSpan",
    "Timestamp",
    "WorkflowCompileError",
    "WorkflowExecutionError",
    "WorkflowsBuffer",
    "compile_workflow",
    "execute_plan",
    "validate_entry",
]
