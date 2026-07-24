from inference.core.env import OFFLINE_MODE
from inference.core.workflows.errors import WorkflowEnvironmentConfigurationError
from inference.core.workflows.prototypes.block import StepExecutionMode


def ensure_builtin_remote_execution_allowed(operation: str) -> None:
    """Reject built-in remote inference leaves when the process is offline."""
    if OFFLINE_MODE:
        raise RuntimeError(
            f"{operation} is not available while OFFLINE_MODE is enabled."
        )


def ensure_workflow_step_execution_mode_allowed(
    step_execution_mode: object,
) -> None:
    """Reject explicit remote-step overrides while the engine is offline."""
    if not OFFLINE_MODE:
        return
    is_remote = (
        step_execution_mode is StepExecutionMode.REMOTE
        or step_execution_mode == StepExecutionMode.REMOTE.value
        or getattr(step_execution_mode, "value", None) == StepExecutionMode.REMOTE.value
    )
    if is_remote:
        raise WorkflowEnvironmentConfigurationError(
            public_message=(
                "Remote Workflow step execution is not available while "
                "OFFLINE_MODE is enabled."
            ),
            context="workflow_compilation | steps_initialisation",
        )
