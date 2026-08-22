from inference.core.env import (
    OFFLINE_MODE,
    SECURE_GATEWAY,
    WORKFLOWS_REMOTE_API_TARGET,
)
from inference.core.workflows.errors import WorkflowEnvironmentConfigurationError
from inference.core.workflows.prototypes.block import StepExecutionMode


def ensure_builtin_remote_execution_allowed(operation: str) -> None:
    """Reject built-in remote inference leaves when the process is offline."""
    if OFFLINE_MODE:
        raise RuntimeError(
            f"{operation} is not available while OFFLINE_MODE is enabled."
        )


def _is_remote(step_execution_mode: object) -> bool:
    """True for the enum member, its value, or anything carrying that value."""
    return (
        step_execution_mode is StepExecutionMode.REMOTE
        or step_execution_mode == StepExecutionMode.REMOTE.value
        or getattr(step_execution_mode, "value", None) == StepExecutionMode.REMOTE.value
    )


def ensure_workflow_step_execution_mode_allowed(
    step_execution_mode: object,
) -> None:
    """Reject explicit remote-step overrides the environment cannot serve.

    ``env.py`` rewrites ``WORKFLOWS_STEP_EXECUTION_MODE`` for both of these
    conditions, but that only moves the *default*. A caller passing
    ``workflows_core.step_execution_mode`` through ``init_parameters`` never
    reads that value, so the same two conditions have to be enforced here, at
    the point the parameter is consumed.
    """
    gateway_blocks_hosted = bool(SECURE_GATEWAY) and (
        WORKFLOWS_REMOTE_API_TARGET == "hosted"
    )
    # Short-circuit on the environment before touching the argument: when no
    # guard applies the caller's object is never inspected, which keeps an
    # arbitrary step_execution_mode value out of this code path entirely.
    if not OFFLINE_MODE and not gateway_blocks_hosted:
        return
    if not _is_remote(step_execution_mode):
        return
    if OFFLINE_MODE:
        raise WorkflowEnvironmentConfigurationError(
            public_message=(
                "Remote Workflow step execution is not available while "
                "OFFLINE_MODE is enabled."
            ),
            context="workflow_compilation | steps_initialisation",
        )
    if gateway_blocks_hosted:
        # Hosted Roboflow inference endpoints are not reachable through the
        # gateway proxy: the SDK client path-joins onto api_url and cannot
        # compose with /proxy?url=<encoded>. Every remote step would dead-end.
        raise WorkflowEnvironmentConfigurationError(
            public_message=(
                "Remote Workflow step execution against hosted Roboflow inference "
                "endpoints is not supported behind SECURE_GATEWAY - they are not "
                "reachable through the gateway proxy. Use "
                "WORKFLOWS_REMOTE_API_TARGET=self-hosted with LOCAL_INFERENCE_API_URL "
                "pointing at a server inside the gateway perimeter, or run this "
                "Workflow with local step execution."
            ),
            context="workflow_compilation | steps_initialisation",
        )
