"""Public entry point of workload introspection.

`describe_workflow_workload` exposes compile-time workload facts of a workflow
definition (graph connectivity, per-step dimensionality and declarations,
model inventory, summaries) without initialising blocks, loading models,
evaluating custom Python or selecting an execution backend. Target services
own cost scores and hardware assumptions; this is an introspection extension,
not an estimator.
"""

from typing import Any, Callable, Dict, Optional, Union

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version
from roboflow_workflows.errors import (
    WorkflowDefinitionError,
    WorkflowExecutionEngineVersionError,
)
from roboflow_workflows.execution_engine.core import (
    _select_execution_engine,
    retrieve_requested_execution_engine_version,
)
from roboflow_workflows.execution_engine.entities.workload import ModelMetadataProvider
from roboflow_workflows.execution_engine.introspection.workload_entities import (
    WorkflowIntrospection,
)
from roboflow_workflows.execution_engine.profiling.core import WorkflowsProfiler
from roboflow_workflows.execution_engine.v1.compiler.core import (
    compile_workflow_structure,
)
from roboflow_workflows.execution_engine.v1.core import EXECUTION_ENGINE_V1_VERSION
from roboflow_workflows.execution_engine.v1.introspection.workload import (
    build_workflow_introspection,
)

EXECUTION_ENGINE_V1_SPECIFIER = SpecifierSet(">=1.0.0,<2.0.0")


def describe_workflow_workload(
    workflow_definition: dict,
    init_parameters: Optional[Dict[str, Union[Any, Callable[[None], Any]]]] = None,
    execution_engine_version: Optional[Union[str, Version]] = None,
    model_metadata_provider: Optional[ModelMetadataProvider] = None,
    profiler: Optional[WorkflowsProfiler] = None,
) -> WorkflowIntrospection:
    """Describe the compile-time workload of `workflow_definition`.

    The requested Execution Engine version is checked before anything else
    runs: no compilation, inner workflow resolution or model metadata lookup
    happens for a rejected request. Only v1 is supported, and the installed
    engine must satisfy the requested version as a minimum within its major
    version, exactly as `ExecutionEngine.init` requires. The engine and its
    blocks are never initialised. The request dict is never mutated and the
    result is built fresh per call.

    Args:
        workflow_definition: Workflow definition to describe.
        init_parameters: The usual `workflows_core.*` bindings (inner workflow
            resolver, api key, workspace resolver). Nothing in it triggers block
            initialisation, model loading or code execution.
        execution_engine_version: Requested engine version. Takes precedence
            over the definition's `version`; when omitted, the definition's
            `version` is used. The definition must still declare `version`:
            the workflow schema requires it, so a definition without it
            raises `WorkflowSyntaxError` during compilation.
        model_metadata_provider: Optional host enrichment hook. Without it,
            every model inventory entry is `unavailable`.
        profiler: Optional profiler passed to the compiler.

    Returns:
        Compile-time workload facts, reported for the installed v1 engine
        version.

    Raises:
        WorkflowExecutionEngineVersionError: If `execution_engine_version`
            cannot be parsed or does not belong to Execution Engine v1.
        WorkflowDefinitionError: If the definition's `version` cannot be
            parsed.
        WorkflowSyntaxError: If the definition does not match the workflow
            schema, e.g. it lacks the required `version`.
        NotSupportedExecutionEngineError: If the installed engine does not
            satisfy the requested version, e.g. a newer minor or patch.
    """
    requested_version = _resolve_requested_execution_engine_version(
        workflow_definition=workflow_definition,
        execution_engine_version=execution_engine_version,
    )
    if not EXECUTION_ENGINE_V1_SPECIFIER.contains(requested_version):
        raise WorkflowExecutionEngineVersionError(
            public_message="Describing workflow workload is only supported for Execution "
            f"Engine v1, requested `{requested_version}`.",
            context="describing_workflow_workload",
        )
    # same selection `ExecutionEngine.init` performs; the selected engine type
    # is discarded, never initialised
    _select_execution_engine(requested_engine_version=requested_version)

    compilation_result = compile_workflow_structure(
        workflow_definition=workflow_definition,
        init_parameters=init_parameters,
        execution_engine_version=EXECUTION_ENGINE_V1_VERSION,
        profiler=profiler,
    )
    return build_workflow_introspection(
        compilation_result=compilation_result,
        model_metadata_provider=model_metadata_provider,
    )


def _resolve_requested_execution_engine_version(
    workflow_definition: dict,
    execution_engine_version: Optional[Union[str, Version]],
) -> Version:
    if execution_engine_version is None:
        return retrieve_requested_execution_engine_version(
            workflow_definition=workflow_definition
        )
    if isinstance(execution_engine_version, Version):
        return execution_engine_version
    try:
        return Version(execution_engine_version)
    except (InvalidVersion, TypeError) as error:
        raise WorkflowExecutionEngineVersionError(
            public_message=f"Could not parse requested Execution Engine version "
            f"`{execution_engine_version}`.",
            context="describing_workflow_workload",
            inner_error=error,
        ) from error
