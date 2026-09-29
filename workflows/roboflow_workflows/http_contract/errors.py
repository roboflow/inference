from typing import Optional, Tuple

from roboflow_workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
    OperationTypeNotRecognisedError,
)
from roboflow_workflows.errors import (
    ClientCausedStepExecutionError,
    DynamicBlockCodeError,
    DynamicBlockError,
    ExecutionGraphStructureError,
    InvalidReferenceTargetError,
    NotSupportedExecutionEngineError,
    ReferenceTypeError,
    RuntimeInputError,
    RuntimeLimitsCausedStepExecutionError,
    StepExecutionError,
    StepInputDimensionalityError,
    WorkflowBlockError,
    WorkflowDefinitionError,
    WorkflowError,
    WorkflowExecutionEngineVersionError,
    WorkflowSyntaxError,
)
from roboflow_workflows.execution_engine.v1.inner_workflow.errors import (
    InnerWorkflowCompositionCycleError,
    InnerWorkflowInvalidStepEntryError,
    InnerWorkflowNestingDepthError,
    InnerWorkflowParameterBindingsError,
    InnerWorkflowTotalCountError,
)
from roboflow_workflows.http_contract.entities import WorkflowErrorResponse
from roboflow_workflows.prototypes.platform_errors import FeatureDeprecatedError

DEFINITION_ERRORS = (
    WorkflowSyntaxError,
    InvalidReferenceTargetError,
    ExecutionGraphStructureError,
    StepInputDimensionalityError,
)

CLIENT_ERRORS = (
    WorkflowDefinitionError,
    ReferenceTypeError,
    RuntimeInputError,
    InvalidInputTypeError,
    OperationTypeNotRecognisedError,
    DynamicBlockError,
    InnerWorkflowCompositionCycleError,
    InnerWorkflowInvalidStepEntryError,
    InnerWorkflowNestingDepthError,
    InnerWorkflowTotalCountError,
    InnerWorkflowParameterBindingsError,
    WorkflowExecutionEngineVersionError,
    NotSupportedExecutionEngineError,
)

STEP_EXECUTION_ERRORS = (
    ClientCausedStepExecutionError,
    RuntimeLimitsCausedStepExecutionError,
)


def _build_execution_error_response(
    error: "DynamicBlockCodeError | StepExecutionError",
) -> "WorkflowErrorResponse":
    """Build a WorkflowErrorResponse for execution errors."""
    if isinstance(error, DynamicBlockCodeError):
        block_id = error.block_type_name or "Dynamic Block"
        block_type = error.block_type_name
        property_name = "Python code"
        property_details = error.public_message
    elif isinstance(error, StepExecutionError):
        block_id = error.block_id
        block_type = error.block_type
        property_name = None
        property_details = str(error.inner_error)
    else:
        raise ValueError(f"Unsupported error type: {type(error)}")

    return WorkflowErrorResponse(
        message=error.public_message,
        error_type=error.__class__.__name__,
        context=error.context,
        inner_error_type=error.inner_error_type,
        inner_error_message=str(error.inner_error) if error.inner_error else None,
        blocks_errors=[
            WorkflowBlockError(
                block_id=block_id,
                block_type=block_type,
                property_name=property_name,
                property_details=property_details,
                block_traceback=error.block_traceback,
            ),
        ],
        # Attached by the workflow-run route when `debug=True` and the run
        # failed; carries logs of python blocks executed before the failure.
        python_blocks_output_streams=getattr(
            error, "python_blocks_output_streams", None
        ),
        python_blocks_debug_traces=getattr(error, "python_blocks_debug_traces", None),
    )


def workflow_error_payload(error: BaseException) -> Optional[Tuple[int, dict]]:
    if isinstance(error, DEFINITION_ERRORS):
        content = WorkflowErrorResponse(
            message=str(error.public_message),
            error_type=error.__class__.__name__,
            context=str(error.context),
            inner_error_type=str(error.inner_error_type),
            inner_error_message=str(error.inner_error),
            blocks_errors=error.blocks_errors,
        )
        return 400, content.model_dump()
    if isinstance(error, DynamicBlockCodeError):
        return 400, _build_execution_error_response(error).model_dump()
    if isinstance(error, CLIENT_ERRORS):
        return 400, {
            "message": error.public_message,
            "error_type": error.__class__.__name__,
            "context": error.context,
            "inner_error_type": error.inner_error_type,
            "inner_error_message": str(error.inner_error),
        }
    if isinstance(error, STEP_EXECUTION_ERRORS):
        content = WorkflowErrorResponse(
            message=str(error.public_message),
            error_type=error.__class__.__name__,
            context=str(error.context),
            inner_error_type=str(error.inner_error_type),
            inner_error_message=str(error.inner_error),
            blocks_errors=[
                WorkflowBlockError(
                    block_id=error.block_id,
                ),
            ],
            # Attached by the workflow-run route when `debug=True` and the run
            # failed; carries logs of python blocks executed before the failure.
            python_blocks_output_streams=getattr(
                error, "python_blocks_output_streams", None
            ),
            python_blocks_debug_traces=getattr(
                error, "python_blocks_debug_traces", None
            ),
        )
        return error.status_code, content.model_dump()
    if isinstance(error, StepExecutionError):
        return 500, _build_execution_error_response(error).model_dump()
    if isinstance(error, WorkflowError):
        return 500, {
            "message": error.public_message,
            "error_type": error.__class__.__name__,
            "context": error.context,
            "inner_error_type": error.inner_error_type,
            "inner_error_message": str(error.inner_error),
            # Attached by the workflow-run route when `debug=True` and the
            # run failed; carries logs of python blocks executed before the
            # failure.
            "python_blocks_output_streams": getattr(
                error, "python_blocks_output_streams", None
            ),
            "python_blocks_debug_traces": getattr(
                error, "python_blocks_debug_traces", None
            ),
        }
    if isinstance(error, FeatureDeprecatedError):
        return 410, {
            "message": str(error),
            "error_type": "FeatureDeprecatedError",
            **error.get_structured_public_error_details(),
        }
    return None
