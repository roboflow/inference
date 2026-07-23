from inference.core.workflows.prototypes.block import BlockResult


def disabled_sink_response() -> BlockResult:
    return {
        "error_status": False,
        "message": "Sink was disabled by workflow execution policy",
    }
