from inference.core.workflows.prototypes.block import BlockResult


def disabled_sink_message(disabled_by_execution_policy: bool) -> str:
    if disabled_by_execution_policy:
        return "Sink was disabled by workflow execution policy"
    return "Sink was disabled by parameter `disable_sink`"


def disabled_sink_response() -> BlockResult:
    return {
        "error_status": False,
        "message": disabled_sink_message(disabled_by_execution_policy=True),
    }
