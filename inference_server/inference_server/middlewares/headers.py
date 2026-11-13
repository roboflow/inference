"""Response header names shared with the legacy inference server."""

from inference_server import configuration

PROCESSING_TIME_HEADER = "X-Processing-Time"
REMOTE_PROCESSING_TIME_HEADER = "X-Remote-Processing-Time"
REMOTE_PROCESSING_TIMES_HEADER = "X-Remote-Processing-Times"
MODEL_COLD_START_HEADER = "X-Model-Cold-Start"
MODEL_COLD_START_COUNT_HEADER = "X-Model-Cold-Start-Count"
MODEL_LOAD_TIME_HEADER = "X-Model-Load-Time"
MODEL_LOAD_DETAILS_HEADER = "X-Model-Load-Details"
MODEL_ID_HEADER = "X-Model-Id"
WORKFLOW_ID_HEADER = "X-Workflow-Id"
WORKSPACE_ID_HEADER = "X-Workspace-Id"
TRACE_ID_HEADER = "X-Trace-Id"
INFERENCE_ENGINE_HEADER = "x-inference-engine"
INFERENCE_ENGINE = "inference-models"

CORS_EXPOSE_HEADERS = [
    PROCESSING_TIME_HEADER,
    REMOTE_PROCESSING_TIME_HEADER,
    REMOTE_PROCESSING_TIMES_HEADER,
    MODEL_COLD_START_HEADER,
    MODEL_COLD_START_COUNT_HEADER,
    MODEL_LOAD_TIME_HEADER,
    MODEL_LOAD_DETAILS_HEADER,
    MODEL_ID_HEADER,
    WORKFLOW_ID_HEADER,
    WORKSPACE_ID_HEADER,
    TRACE_ID_HEADER,
]
if configuration.EXECUTION_ID_HEADER:
    CORS_EXPOSE_HEADERS.append(configuration.EXECUTION_ID_HEADER)
CORS_EXPOSE_HEADERS.extend(["traceparent", "tracestate"])
