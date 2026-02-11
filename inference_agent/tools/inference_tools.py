"""Inference tools: run_inference, ask_about_image, detect_zero_shot."""

from __future__ import annotations

from typing import Any, Optional

from inference_agent.core.protocols import InferenceBackend, ToolDefinition
from inference_agent.tools.registry import Tool
from inference_agent.tools.result_summarizer import ResultSummarizer


# ---------------------------------------------------------------------------
# run_inference
# ---------------------------------------------------------------------------

RUN_INFERENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "Image source: file path, URL, or base64-encoded string.",
        },
        "model_id": {
            "type": "string",
            "description": (
                "Roboflow model ID, e.g. 'my-workspace/hard-hat-detection/3'. "
                "Provide this OR workspace_name+workflow_id OR workflow_spec."
            ),
        },
        "workspace_name": {
            "type": "string",
            "description": "Roboflow workspace name (used with workflow_id).",
        },
        "workflow_id": {
            "type": "string",
            "description": "Roboflow workflow ID (used with workspace_name).",
        },
        "workflow_spec": {
            "type": "object",
            "description": "Inline workflow specification (JSON). Use when composing a custom workflow.",
        },
        "parameters": {
            "type": "object",
            "description": "Additional workflow parameters.",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence threshold (default 0.4).",
        },
    },
    "required": ["image"],
}


async def _run_inference(
    backend: InferenceBackend,
    summarizer: ResultSummarizer,
    image: str,
    model_id: Optional[str] = None,
    workspace_name: Optional[str] = None,
    workflow_id: Optional[str] = None,
    workflow_spec: Optional[dict] = None,
    parameters: Optional[dict] = None,
    confidence: Optional[float] = None,
) -> Any:
    if confidence is not None:
        parameters = parameters or {}
        parameters["confidence"] = confidence

    result = await backend.run_single(
        image=image,
        model_id=model_id,
        workflow_spec=workflow_spec,
        workspace_name=workspace_name,
        workflow_id=workflow_id,
        parameters=parameters,
    )
    return summarizer.summarize(result)


def create_run_inference_tool(
    backend: InferenceBackend, summarizer: ResultSummarizer
) -> Tool:
    async def execute(**kwargs: Any) -> Any:
        return await _run_inference(backend, summarizer, **kwargs)

    return Tool(
        definition=ToolDefinition(
            name="run_inference",
            description=(
                "Run inference on a single image using a model or workflow. "
                "Accepts a Roboflow model_id (e.g. 'my-workspace/model/3') "
                "or a workflow (by workspace_name+workflow_id or inline spec)."
            ),
            input_schema=RUN_INFERENCE_SCHEMA,
        ),
        execute=execute,
        category="inference",
    )


# ---------------------------------------------------------------------------
# ask_about_image
# ---------------------------------------------------------------------------

ASK_ABOUT_IMAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "Image source: file path, URL, or base64-encoded string.",
        },
        "question": {
            "type": "string",
            "description": "Question to ask about the image.",
        },
        "model": {
            "type": "string",
            "enum": ["florence2", "qwen2.5-vl", "paligemma"],
            "description": "VLM model to use (default: florence2).",
        },
    },
    "required": ["image", "question"],
}


async def _ask_about_image(
    backend: InferenceBackend,
    image: str,
    question: str,
    model: str = "florence2",
) -> str:
    # Build a simple VLM workflow
    model_type_map = {
        "florence2": "roboflow_core/florence_2@v1",
        "qwen2.5-vl": "roboflow_core/qwen_2_5_vl@v1",
        "paligemma": "roboflow_core/pali_gemma@v1",
    }
    block_type = model_type_map.get(model, "roboflow_core/florence_2@v1")

    workflow_spec = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "prompt"},
        ],
        "steps": [
            {
                "type": block_type,
                "name": "vlm",
                "images": "$inputs.image",
                "task_type": "visual-question-answering",
                "prompt": "$inputs.prompt",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "answer",
                "selector": "$steps.vlm.output",
            },
        ],
    }

    result = await backend.run_single(
        image=image,
        workflow_spec=workflow_spec,
        parameters={"prompt": question},
    )
    # Extract the answer from workflow output
    predictions = result.predictions
    if "outputs" in predictions and isinstance(predictions["outputs"], list):
        for output in predictions["outputs"]:
            if isinstance(output, dict) and "answer" in output:
                return str(output["answer"])
    return str(predictions)


def create_ask_about_image_tool(backend: InferenceBackend) -> Tool:
    async def execute(**kwargs: Any) -> str:
        return await _ask_about_image(backend, **kwargs)

    return Tool(
        definition=ToolDefinition(
            name="ask_about_image",
            description=(
                "Ask a vision-language model a question about an image. "
                "Uses Florence-2, Qwen2.5-VL, or PaliGemma."
            ),
            input_schema=ASK_ABOUT_IMAGE_SCHEMA,
        ),
        execute=execute,
        category="inference",
    )


# ---------------------------------------------------------------------------
# detect_zero_shot
# ---------------------------------------------------------------------------

DETECT_ZERO_SHOT_SCHEMA = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "Image source: file path, URL, or base64-encoded string.",
        },
        "class_names": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Objects to detect, e.g. ['person', 'hard hat', 'safety vest'].",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence threshold (default 0.3).",
        },
    },
    "required": ["image", "class_names"],
}


async def _detect_zero_shot(
    backend: InferenceBackend,
    summarizer: ResultSummarizer,
    image: str,
    class_names: list[str],
    confidence: float = 0.3,
) -> str:
    # Build a YOLO-World workflow for zero-shot detection
    workflow_spec = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
        ],
        "steps": [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v2",
                "name": "yolo_world",
                "images": "$inputs.image",
                "model_id": "yolo_world/v2",
                "class_filter": class_names,
                "confidence": confidence,
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "detections",
                "selector": "$steps.yolo_world.predictions",
            },
        ],
    }

    result = await backend.run_single(image=image, workflow_spec=workflow_spec)
    return summarizer.summarize(result)


def create_detect_zero_shot_tool(
    backend: InferenceBackend, summarizer: ResultSummarizer
) -> Tool:
    async def execute(**kwargs: Any) -> str:
        return await _detect_zero_shot(backend, summarizer, **kwargs)

    return Tool(
        definition=ToolDefinition(
            name="detect_zero_shot",
            description=(
                "Detect objects by text description without a trained model. "
                "Uses YOLO-World for fast zero-shot detection by class name prompts."
            ),
            input_schema=DETECT_ZERO_SHOT_SCHEMA,
        ),
        execute=execute,
        category="inference",
    )
