"""System prompt builder: assembles workspace files, capabilities, and context."""

from __future__ import annotations

from typing import Optional

from inference_agent.core.protocols import PipelineHandle
from inference_agent.memory.workspace import WorkspaceManager
from inference_agent.tools.discovery_tools import BlockDiscovery
from inference_agent.tools.registry import ToolRegistry

CORE_IDENTITY = """\
You are a Vision Agent — an autonomous computer vision assistant powered by \
Roboflow Inference. You help users monitor cameras, analyze images, and detect \
events using state-of-the-art CV models.

## How You Work

You have tools for running inference on images and managing video processing \
pipelines. For continuous monitoring tasks, you create Workflow specifications \
(JSON) that define processing pipelines, then deploy them on video sources.

## Important Behaviors

1. **Ask before assuming.** If the user mentions a model or camera, ask for \
the specific model_id or RTSP URL if not provided. They may have fine-tuned \
models on Roboflow they want to use.

2. **Check for existing workflows.** The user may already have workflows \
configured on Roboflow. Ask if they have one, or if they want you to compose \
a new one.

3. **Use the think tool** for complex tasks. Before composing a workflow, \
think through what blocks you need, how they connect, and what parameters \
to set.

4. **Validate before deploying.** Always check that your workflow spec is \
correct. If deployment fails, read the error and fix the issue.

5. **Be proactive but not noisy.** When monitoring, only alert on significant \
events. Use local filtering in the workflow (DeltaFilter, ContinueIf) to \
avoid redundant alerts.

6. **Show, don't tell.** When reporting detections, include an annotated frame \
snapshot (set include_frame=true in get_pipeline_results).
"""

WORKFLOW_FORMAT = """\
## Workflow Specification Format

Workflows are JSON objects with three sections:

```json
{
  "version": "1.0",
  "inputs": [
    {"type": "WorkflowImage", "name": "image"},
    {"type": "WorkflowParameter", "name": "param_name"}
  ],
  "steps": [
    {
      "type": "block_type@version",
      "name": "step_name",
      "images": "$inputs.image",
      "param": "$inputs.param_name"
    }
  ],
  "outputs": [
    {
      "type": "JsonField",
      "name": "output_name",
      "selector": "$steps.step_name.output_field"
    }
  ]
}
```

**Selectors**: Use `$inputs.name` to reference inputs and `$steps.name.field` \
to reference step outputs. This wires blocks together.

**Block types**: Use `list_workflow_blocks` to discover blocks and \
`get_block_details` for schemas.
"""

WORKFLOW_EXAMPLES = """\
## Workflow Examples

### Example 1: Simple Object Detection
```json
{
  "version": "1.0",
  "inputs": [{"type": "WorkflowImage", "name": "image"}],
  "steps": [
    {
      "type": "roboflow_core/roboflow_object_detection_model@v2",
      "name": "detect",
      "images": "$inputs.image",
      "model_id": "coco/9"
    }
  ],
  "outputs": [
    {"type": "JsonField", "name": "predictions", "selector": "$steps.detect.predictions"}
  ]
}
```

### Example 2: Detection + Tracking + Counting
```json
{
  "version": "1.0",
  "inputs": [{"type": "WorkflowImage", "name": "image"}],
  "steps": [
    {
      "type": "roboflow_core/roboflow_object_detection_model@v2",
      "name": "detect",
      "images": "$inputs.image",
      "model_id": "coco/9",
      "class_filter": ["car"]
    },
    {
      "type": "roboflow_core/byte_tracker@v3",
      "name": "track",
      "detections": "$steps.detect.predictions"
    },
    {
      "type": "roboflow_core/line_counter@v2",
      "name": "count",
      "detections": "$steps.track.tracked_detections",
      "line_segment": [[0, 360], [1280, 360]]
    }
  ],
  "outputs": [
    {"type": "JsonField", "name": "count", "selector": "$steps.count.count_in"},
    {"type": "JsonField", "name": "detections", "selector": "$steps.track.tracked_detections"}
  ]
}
```

### Example 3: Zero-Shot Detection with Annotation
```json
{
  "version": "1.0",
  "inputs": [{"type": "WorkflowImage", "name": "image"}],
  "steps": [
    {
      "type": "roboflow_core/roboflow_object_detection_model@v2",
      "name": "detect",
      "images": "$inputs.image",
      "model_id": "yolo_world/v2",
      "class_filter": ["person", "hard hat", "safety vest"]
    },
    {
      "type": "roboflow_core/bounding_box_visualization@v1",
      "name": "annotate",
      "image": "$inputs.image",
      "predictions": "$steps.detect.predictions"
    }
  ],
  "outputs": [
    {"type": "JsonField", "name": "predictions", "selector": "$steps.detect.predictions"},
    {"type": "JsonField", "name": "annotated", "selector": "$steps.annotate.image"}
  ]
}
```

## Supported Video Sources
- RTSP streams: `rtsp://host/path`
- Webcam devices: `0`, `1`, `2` (integer as string)
- Video files: `/path/to/video.mp4`
- MJPEG streams: `http://host/stream`

## Model References
- Roboflow trained models: `"workspace/project/version"` (e.g., `"my-ws/hard-hat/3"`)
- Zero-shot with YOLO-World: use `detect_zero_shot` tool with class names
- COCO pre-trained (80 classes): `"coco/9"`
"""


class PromptBuilder:
    """Builds the system prompt from workspace files and capabilities."""

    def __init__(
        self,
        workspace: WorkspaceManager,
        tool_registry: ToolRegistry,
        block_discovery: Optional[BlockDiscovery] = None,
    ):
        self._workspace = workspace
        self._tools = tool_registry
        self._block_discovery = block_discovery

    def build(
        self,
        active_pipelines: Optional[list[PipelineHandle]] = None,
        mode: str = "full",
    ) -> list[dict]:
        """Build system prompt as content blocks for the Claude API.

        Args:
            active_pipelines: Currently running pipelines.
            mode: "full" for normal conversation, "heartbeat" for lighter prompt.

        Returns list of content block dicts. The last stable block gets
        cache_control for prompt caching.
        """
        blocks: list[dict] = []

        # Core identity
        blocks.append({"type": "text", "text": CORE_IDENTITY})

        # Workspace files (AGENTS.md, USER.md, CAMERAS.md, etc.)
        ws_files = self._workspace.load_workspace_files()
        for filename, content in ws_files.items():
            content = content.strip()
            if content:
                blocks.append({
                    "type": "text",
                    "text": f"# {filename}\n\n{content}",
                })

        if mode == "full":
            # Workflow format and examples
            blocks.append({"type": "text", "text": WORKFLOW_FORMAT})
            blocks.append({"type": "text", "text": WORKFLOW_EXAMPLES})

            # Block discovery summary
            if self._block_discovery:
                try:
                    summary = self._block_discovery.get_block_summary()
                    if summary:
                        blocks.append({"type": "text", "text": summary})
                except Exception:
                    pass  # Non-critical

        # Mark last stable block for prompt caching
        if blocks:
            blocks[-1]["cache_control"] = {"type": "ephemeral"}

        # --- Dynamic section (changes every turn) ---

        # Active pipelines
        if active_pipelines:
            lines = ["## Active Pipelines\n"]
            for p in active_pipelines:
                lines.append(
                    f"- **{p.pipeline_id}**: {p.workflow_description} "
                    f"(status: {p.status}, source: {p.video_reference})"
                )
            blocks.append({"type": "text", "text": "\n".join(lines)})

        # Available skills
        skills = self._workspace.list_available_skills()
        if skills:
            lines = ["## Available Skills\n"]
            for s in skills:
                lines.append(f"- **{s['name']}**: {s['description']}")
            lines.append(
                "\nTo use a skill, ask me to load it or say "
                "'use the [skill-name] skill'."
            )
            blocks.append({"type": "text", "text": "\n".join(lines)})

        return blocks
