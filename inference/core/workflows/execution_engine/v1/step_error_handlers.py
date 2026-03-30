from inference.core.exceptions import (
    CannotInitialiseModelDueToInputSizeError,
    InferenceModelNotFound,
    InvalidModelIDError,
    ModelManagerLockAcquisitionError,
    PaymentRequiredError,
    RoboflowAPIForbiddenError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
)
from inference.core.workflows.errors import (
    ClientCausedStepExecutionError,
    RuntimeLimitsCausedStepExecutionError,
)
from inference_models.errors import (
    ModelNotFoundError,
    ModelPackageAlternativesExhaustedError,
    ModelPackageRestrictedError,
    UnauthorizedModelAccessError,
)
from inference_sdk.http.errors import HTTPCallErrorError


def legacy_step_error_handler(step_name: str, error: Exception) -> None:
    if isinstance(error, (ModelManagerLockAcquisitionError, InferenceModelNotFound)):
        raise error
    return None


def extended_roboflow_errors_handler(step_name: str, error: Exception) -> None:
    if isinstance(
        error,
        (
            ModelManagerLockAcquisitionError,
            InferenceModelNotFound,
        ),
    ):
        raise error
    if isinstance(error, CannotInitialiseModelDueToInputSizeError):
        raise RuntimeLimitsCausedStepExecutionError(
            block_id=step_name,
            status_code=507,
            public_message=f"Could not complete workflow execution due to configured runtime constraints. "
            f"Details: model input size causes runtime memory requirements exceed the limit "
            f"configured for the environment.",
            context="workflow_execution | step_execution",
            inner_error=error,
        ) from error
    if isinstance(error, ModelPackageAlternativesExhaustedError) and any(
        isinstance(e, ModelPackageRestrictedError)
        for e in (error.alternatives_errors or [])
    ):
        raise RuntimeLimitsCausedStepExecutionError(
            block_id=step_name,
            status_code=507,
            public_message="Model loading failed due to restrictions of server configuration - "
            "usually due to excessive runtime memory requirement of the model (for instance "
            "caused by large input size).",
            context="workflow_execution | step_execution",
            inner_error=error,
        ) from error
    if isinstance(error, InvalidModelIDError):
        raise ClientCausedStepExecutionError(
            block_id=step_name,
            status_code=400,
            public_message=f"Problem with Workflow Block configuration - {error}",
            context="workflow_execution | step_execution",
            inner_error=error,
        ) from error
    if isinstance(error, (RoboflowAPINotAuthorizedError, UnauthorizedModelAccessError)):
        raise ClientCausedStepExecutionError(
            block_id=step_name,
            status_code=401,
            public_message=f"Unauthorized error occurred while execution of step {step_name} - "
            f"details of error: {error}. This error usually mean the problem with Roboflow API key.",
            context="workflow_execution | step_execution",
            inner_error=error,
        ) from error
    if isinstance(error, PaymentRequiredError):
        raise ClientCausedStepExecutionError(
            block_id=step_name,
            status_code=402,
            public_message=f"Not enough credits to execute step {step_name}. "
            f"Verify your workspace billing page. Details: {error}",
            context="workflow_execution | step_execution",
            inner_error=error,
        ) from error
    if isinstance(error, RoboflowAPIForbiddenError):
        raise ClientCausedStepExecutionError(
            block_id=step_name,
            status_code=403,
            public_message=f"Forbidden error occurred while execution of step {step_name} - "
            f"details of error: {error}. This error usually mean the problem with Roboflow API key.",
            context="workflow_execution | step_execution",
            inner_error=error,
        ) from error
    if isinstance(error, (RoboflowAPINotNotFoundError, ModelNotFoundError)):
        raise ClientCausedStepExecutionError(
            block_id=step_name,
            status_code=404,
            public_message=f"Could not find requested Roboflow resource while execution of step {step_name} - "
            f"details of error: {error}. This error usually mean the problem with not existing model.",
            context="workflow_execution | step_execution",
            inner_error=error,
        ) from error
    if isinstance(error, HTTPCallErrorError):
        if error.status_code == 400:
            raise ClientCausedStepExecutionError(
                block_id=step_name,
                status_code=400,
                public_message=f"Bad request error detected while remote execution of step {step_name} - "
                f"details of error: {error}. This error usually mean that the Workflow block configuration is faulty.",
                context="workflow_execution | step_execution",
                inner_error=error,
            ) from error
        if error.status_code == 401:
            raise ClientCausedStepExecutionError(
                block_id=step_name,
                status_code=401,
                public_message=f"Unauthorized error occurred while remote execution of step {step_name} - "
                f"details of error: {error}. This error usually mean the problem with Roboflow API key.",
                context="workflow_execution | step_execution",
                inner_error=error,
            ) from error
        if error.status_code == 402:
            raise ClientCausedStepExecutionError(
                block_id=step_name,
                status_code=402,
                public_message=f"Not enough credits to remote execute step {step_name}. "
                f"Verify your workspace billing page. Details: {error}",
                context="workflow_execution | step_execution",
                inner_error=error,
            ) from error
        if error.status_code == 403:
            raise ClientCausedStepExecutionError(
                block_id=step_name,
                status_code=403,
                public_message=f"Forbidden error occurred while remote execution of step {step_name} - "
                f"details of error: {error}. This error usually mean the problem with Roboflow API key.",
                context="workflow_execution | step_execution",
                inner_error=error,
            ) from error
        if error.status_code == 404:
            raise ClientCausedStepExecutionError(
                block_id=step_name,
                status_code=404,
                public_message=f"Could not find requested Roboflow resource while remote execution of step {step_name} - "
                f"details of error: {error}. This error usually mean the problem with not existing model.",
                context="workflow_execution | step_execution",
                inner_error=error,
            ) from error
        if error.status_code == 507:
            raise RuntimeLimitsCausedStepExecutionError(
                block_id=step_name,
                status_code=507,
                public_message=f"Could not complete workflow execution due to configured runtime constraints. "
                f"Details: {error.api_message}",
                context="workflow_execution | step_execution",
                inner_error=error,
            ) from error
    return None
