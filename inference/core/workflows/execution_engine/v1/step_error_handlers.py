from inference.core.exceptions import (
    InferenceModelNotFound,
    InvalidModelIDError,
    ModelManagerLockAcquisitionError,
    RoboflowAPIForbiddenError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
)
from inference.core.workflows.errors import ClientCausedStepExecutionError
from inference_models.errors import ModelNotFoundError, UnauthorizedModelAccessError
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
    return None
