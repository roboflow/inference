"""Pipeline management tools: start, stop, pause, resume, list, get results."""

from __future__ import annotations

from typing import Any, Optional

from inference_agent.core.protocols import InferenceBackend, ToolDefinition
from inference_agent.tools.registry import Tool
from inference_agent.tools.result_summarizer import ResultSummarizer


# ---------------------------------------------------------------------------
# start_pipeline
# ---------------------------------------------------------------------------

START_PIPELINE_SCHEMA = {
    "type": "object",
    "properties": {
        "video_reference": {
            "type": "string",
            "description": (
                "Video source: RTSP URL (rtsp://...), video file path, "
                "webcam index ('0', '1'), or MJPEG stream URL."
            ),
        },
        "pipeline_name": {
            "type": "string",
            "description": "Human-friendly name for this pipeline (e.g. 'loading-dock-monitor').",
        },
        "workflow_spec": {
            "type": "object",
            "description": "Inline workflow specification (JSON).",
        },
        "workspace_name": {
            "type": "string",
            "description": "Roboflow workspace name (used with workflow_id).",
        },
        "workflow_id": {
            "type": "string",
            "description": "Roboflow workflow ID (used with workspace_name).",
        },
        "model_id": {
            "type": "string",
            "description": "Simple model ID for basic detection (auto-wraps in a workflow).",
        },
        "max_fps": {
            "type": "number",
            "description": "Maximum frames per second to process.",
        },
        "parameters": {
            "type": "object",
            "description": "Additional workflow parameters.",
        },
    },
    "required": ["video_reference", "pipeline_name"],
}


async def _start_pipeline(
    backend: InferenceBackend,
    video_reference: str,
    pipeline_name: str,
    workflow_spec: Optional[dict] = None,
    workspace_name: Optional[str] = None,
    workflow_id: Optional[str] = None,
    model_id: Optional[str] = None,
    max_fps: Optional[float] = None,
    parameters: Optional[dict] = None,
) -> str:
    # If only model_id given, wrap it in a simple detect workflow
    if model_id and not workflow_spec and not workflow_id:
        workflow_spec = {
            "version": "1.0",
            "inputs": [
                {"type": "WorkflowImage", "name": "image"},
            ],
            "steps": [
                {
                    "type": "roboflow_core/roboflow_object_detection_model@v2",
                    "name": "detection",
                    "images": "$inputs.image",
                    "model_id": model_id,
                },
            ],
            "outputs": [
                {
                    "type": "JsonField",
                    "name": "predictions",
                    "selector": "$steps.detection.predictions",
                },
            ],
        }

    handle = await backend.start_pipeline(
        video_reference=video_reference,
        workflow_spec=workflow_spec,
        workspace_name=workspace_name,
        workflow_id=workflow_id,
        max_fps=max_fps,
        parameters=parameters,
    )

    fps_info = f" at {max_fps} FPS" if max_fps else ""
    return (
        f"Pipeline '{pipeline_name}' started (id: {handle.pipeline_id}). "
        f"Processing {video_reference}{fps_info}. "
        f"Use get_pipeline_results with pipeline_id='{handle.pipeline_id}' to see results."
    )


def create_start_pipeline_tool(backend: InferenceBackend) -> Tool:
    async def execute(**kwargs: Any) -> str:
        return await _start_pipeline(backend, **kwargs)

    return Tool(
        definition=ToolDefinition(
            name="start_pipeline",
            description=(
                "Start continuous video processing on a camera or video source. "
                "The pipeline runs a workflow on every frame. Accepts: "
                "(1) workspace_name + workflow_id for a Roboflow workflow, "
                "(2) an inline workflow_spec, or (3) a simple model_id. "
                "Video sources: RTSP URL, file path, webcam index (0,1,2)."
            ),
            input_schema=START_PIPELINE_SCHEMA,
        ),
        execute=execute,
        category="pipeline",
    )


# ---------------------------------------------------------------------------
# get_pipeline_results
# ---------------------------------------------------------------------------

GET_PIPELINE_RESULTS_SCHEMA = {
    "type": "object",
    "properties": {
        "pipeline_id": {
            "type": "string",
            "description": "Pipeline ID returned by start_pipeline.",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results to fetch (default 5).",
        },
        "include_frame": {
            "type": "boolean",
            "description": "Include an annotated frame snapshot (default false).",
        },
    },
    "required": ["pipeline_id"],
}


async def _get_pipeline_results(
    backend: InferenceBackend,
    summarizer: ResultSummarizer,
    pipeline_id: str,
    max_results: int = 5,
    include_frame: bool = False,
) -> Any:
    results = await backend.consume_results(pipeline_id, max_results=max_results)
    if not results:
        return "No results available yet. The pipeline may still be starting up."
    return summarizer.summarize_multiple(results, include_last_frame=include_frame)


def create_get_pipeline_results_tool(
    backend: InferenceBackend, summarizer: ResultSummarizer
) -> Tool:
    async def execute(**kwargs: Any) -> Any:
        return await _get_pipeline_results(backend, summarizer, **kwargs)

    return Tool(
        definition=ToolDefinition(
            name="get_pipeline_results",
            description=(
                "Get the latest results from a running pipeline. "
                "Returns detection summaries. Set include_frame=true "
                "to get an annotated snapshot."
            ),
            input_schema=GET_PIPELINE_RESULTS_SCHEMA,
        ),
        execute=execute,
        category="pipeline",
    )


# ---------------------------------------------------------------------------
# list_pipelines
# ---------------------------------------------------------------------------

def create_list_pipelines_tool(backend: InferenceBackend) -> Tool:
    async def execute() -> str:
        pipelines = await backend.list_pipelines()
        if not pipelines:
            return "No active pipelines."
        lines = ["Active pipelines:"]
        for p in pipelines:
            lines.append(
                f"  - {p.pipeline_id}: {p.workflow_description} "
                f"(status: {p.status}, source: {p.video_reference})"
            )
        return "\n".join(lines)

    return Tool(
        definition=ToolDefinition(
            name="list_pipelines",
            description="List all active video processing pipelines with their status.",
            input_schema={"type": "object", "properties": {}},
        ),
        execute=execute,
        category="pipeline",
    )


# ---------------------------------------------------------------------------
# stop_pipeline
# ---------------------------------------------------------------------------

STOP_PIPELINE_SCHEMA = {
    "type": "object",
    "properties": {
        "pipeline_id": {
            "type": "string",
            "description": "Pipeline ID to stop.",
        },
    },
    "required": ["pipeline_id"],
}


def create_stop_pipeline_tool(backend: InferenceBackend) -> Tool:
    async def execute(pipeline_id: str) -> str:
        await backend.stop_pipeline(pipeline_id)
        return f"Pipeline {pipeline_id} stopped."

    return Tool(
        definition=ToolDefinition(
            name="stop_pipeline",
            description="Stop a running video pipeline.",
            input_schema=STOP_PIPELINE_SCHEMA,
        ),
        execute=execute,
        category="pipeline",
    )


# ---------------------------------------------------------------------------
# pause_pipeline
# ---------------------------------------------------------------------------

PAUSE_PIPELINE_SCHEMA = {
    "type": "object",
    "properties": {
        "pipeline_id": {
            "type": "string",
            "description": "Pipeline ID to pause.",
        },
    },
    "required": ["pipeline_id"],
}


def create_pause_pipeline_tool(backend: InferenceBackend) -> Tool:
    async def execute(pipeline_id: str) -> str:
        await backend.pause_pipeline(pipeline_id)
        return f"Pipeline {pipeline_id} paused."

    return Tool(
        definition=ToolDefinition(
            name="pause_pipeline",
            description="Pause a running pipeline (can be resumed later).",
            input_schema=PAUSE_PIPELINE_SCHEMA,
        ),
        execute=execute,
        category="pipeline",
    )


# ---------------------------------------------------------------------------
# resume_pipeline
# ---------------------------------------------------------------------------

RESUME_PIPELINE_SCHEMA = {
    "type": "object",
    "properties": {
        "pipeline_id": {
            "type": "string",
            "description": "Pipeline ID to resume.",
        },
    },
    "required": ["pipeline_id"],
}


def create_resume_pipeline_tool(backend: InferenceBackend) -> Tool:
    async def execute(pipeline_id: str) -> str:
        await backend.resume_pipeline(pipeline_id)
        return f"Pipeline {pipeline_id} resumed."

    return Tool(
        definition=ToolDefinition(
            name="resume_pipeline",
            description="Resume a paused pipeline.",
            input_schema=RESUME_PIPELINE_SCHEMA,
        ),
        execute=execute,
        category="pipeline",
    )
