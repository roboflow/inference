# Vision Agent: Implementation Plan (v2)

## Table of Contents
1. [Architecture — Revised](#1-architecture)
2. [Three Execution Backends](#2-execution-backends)
3. [Interface Contracts — Revised](#3-interface-contracts)
4. [OpenClaw-Inspired Workspace & Lifecycle](#4-workspace)
5. [Tool Inventory — Revised](#5-tool-inventory)
6. [System Prompt Design](#6-system-prompt)
7. [Claude API Integration Details](#7-claude-api)
8. [Work Streams (Parallelizable)](#8-work-streams)
9. [MVP — Minimum Testable Assumption](#9-mvp)
10. [Risk Register & Derisking](#10-risks)
11. [Dependency Graph & Sequencing](#11-sequencing)
12. [Detailed Implementation — Each Work Stream](#12-implementation)

---

## 1. Architecture — Revised

The Vision Agent is a long-running Python process modeled after OpenClaw's
**gateway architecture**: a single control plane that manages conversations,
video sources, inference backends, and proactive monitoring.

```
┌────────────────────────────────────────────────────────────────────────┐
│                           AGENT PROCESS                                │
│                                                                        │
│  ┌────────────┐  ┌──────────────────┐  ┌───────────────────────────┐  │
│  │ Interfaces  │  │   Session Mgr    │  │     Heartbeat / Cron      │  │
│  │ (CLI, Slack │──│  (conversation   │──│  (proactive monitoring    │  │
│  │  REST, ...)│  │   history,       │  │   periodic scene check,   │  │
│  └────────────┘  │   JSONL logs)    │  │   scheduled reports)      │  │
│                  └────────┬─────────┘  └─────────────┬─────────────┘  │
│                           │                          │                 │
│  ┌────────────────────────┴──────────────────────────┴──────────────┐ │
│  │                        Agent Core (agentic loop)                  │ │
│  │  1. Receive message / heartbeat tick                              │ │
│  │  2. Build system prompt (workspace files + capabilities + state)  │ │
│  │  3. Call Claude API with tools                                    │ │
│  │  4. Execute tool calls → feed results back → repeat              │ │
│  │  5. Emit response / alert to interface                           │ │
│  └───────────────────────────┬──────────────────────────────────────┘ │
│                              │                                        │
│  ┌───────────────────────────┴──────────────────────────────────────┐ │
│  │                       Tool Registry                               │ │
│  │  Inference │ Workflows │ Streams │ Memory │ Notifications │ Think │ │
│  └──────────────────────────┬───────────────────────────────────────┘ │
│                             │                                         │
│  ┌──────────────────────────┴───────────────────────────────────────┐ │
│  │                 Execution Backends (choose per task)              │ │
│  │                                                                   │ │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────────┐  │ │
│  │  │  Direct SDK      │ │  HTTP Client    │ │  WebRTC Client    │  │ │
│  │  │  (same machine,  │ │  (remote server │ │  (real-time       │  │ │
│  │  │   embedded       │ │   inference,    │ │   streaming to    │  │ │
│  │  │   InferencePipe- │ │   hosted pipe-  │ │   browser or      │  │ │
│  │  │   line)          │ │   line mgmt)    │ │   programmatic    │  │ │
│  │  └─────────────────┘ └─────────────────┘ │   frame access)   │  │ │
│  │                                           └───────────────────┘  │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
├────────────────────────────────────────────────────────────────────────┤
│                 INFERENCE (existing, untouched)                         │
│  Server (:9001)  │  Stream Manager (:7070)  │  Model Registry          │
│  Workflow Engine  │  InferencePipeline       │  WebRTC Worker           │
└────────────────────────────────────────────────────────────────────────┘
```

### Key Architectural Decisions (Revised)

1. **Raw `anthropic` Python SDK for the agent loop.** We need fine-grained
   control over image content blocks in tool results, prompt caching
   placement, extended thinking with interleaved beta, and the "think" tool
   pattern. The Claude Agent SDK is designed for code-editing agents.

2. **Three execution backends**, not one. Users have different deployment
   scenarios:
   - **Direct SDK**: Edge devices, Jetson, local dev — runs InferencePipeline
     in-process
   - **HTTP Client**: Cloud/remote inference server — uses
     `InferenceHTTPClient.start_inference_pipeline_with_workflow()` for hosted
     pipeline management and `run_workflow()` for single-shot inference
   - **WebRTC Client**: Real-time browser visualization or programmatic frame
     access — uses `WebRTCClient.stream()` with any source (RTSP, webcam,
     MJPEG, file, manual)

3. **User-centric, not auto-discovery-centric.** Rather than only
   auto-discovering models, the agent should:
   - **Ask** the user what model they want (they may have fine-tuned models on
     Roboflow: `my-workspace/hard-hat-detection/3`)
   - **Ask** if they already have a workflow configured on Roboflow (reference
     by `workspace_name` + `workflow_id`)
   - **Offer** zero-shot alternatives (YOLO-World, GroundingDINO) if they
     don't have a trained model
   - **Fall back** to discovery tools only when the user doesn't know what
     they need

4. **OpenClaw-inspired workspace with skills.** The agent has a workspace
   directory with Markdown files that shape its behavior — editable by both
   the user and the agent. Skills are Markdown instruction packs for
   domain-specific CV tasks (not executable code).

5. **JSONL session transcripts** for full auditability and crash recovery.
   Every tool call, result, and response is logged append-only.

6. **Heartbeat for proactive monitoring** — a periodic tick that checks active
   streams for events, runs scene analysis, and alerts the user. Cron for
   scheduled reports.

---

## 2. Three Execution Backends

The agent must support three ways to interact with Inference. Which backend is
used depends on the deployment and the task.

### 2.1 Direct SDK Backend

For running on the same machine as inference (edge devices, dev boxes).

```python
# Uses InferencePipeline directly
from inference import InferencePipeline

pipeline = InferencePipeline.init_with_workflow(
    video_reference="rtsp://192.168.1.50/stream1",
    workflow_specification=spec,         # Agent-composed OR
    # workspace_name="my-ws",           # User's pre-built workflow
    # workflow_id="my-workflow",
    on_prediction=result_handler,
    max_fps=10,
)
pipeline.start(use_main_thread=False)
```

**Capabilities**: Full InferencePipeline feature set — multi-source, all buffer
strategies, all sink types, on_prediction callbacks.

### 2.2 HTTP Client Backend

For connecting to a remote inference server. Uses the `InferenceHTTPClient`.

```python
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(api_url="http://server:9001", api_key=key)

# Single-shot inference
result = client.infer("image.jpg", model_id="my-workspace/model/3")

# Run a pre-built workflow by ID
result = client.run_workflow(
    workspace_name="my-workspace",
    workflow_id="my-workflow",
    images={"image": "frame.jpg"},
)

# Run an agent-composed workflow spec
result = client.run_workflow(
    specification=agent_composed_spec,
    images={"image": "frame.jpg"},
)

# Start a HOSTED pipeline (server manages it)
pipeline = client.start_inference_pipeline_with_workflow(
    video_reference="rtsp://192.168.1.50/stream1",
    workflow_id="my-workflow",
    workspace_name="my-workspace",
    max_fps=10,
)

# Poll for results
results = client.consume_inference_pipeline_result(pipeline["pipeline_id"])
```

**Capabilities**: All InferenceHTTPClient methods — model management, workflow
execution, hosted pipeline lifecycle (start/pause/resume/terminate/consume),
OCR, CLIP, SAM, depth estimation, gaze, VLMs.

### 2.3 WebRTC Client Backend

For real-time streaming with browser visualization or programmatic frame access.

```python
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc.sources import RTSPSource, WebcamSource

client = InferenceHTTPClient(api_url="http://server:9001", api_key=key)

session = client.webrtc.stream(
    source=RTSPSource("rtsp://192.168.1.50/stream1"),
    workflow="my-workflow-id",
    workspace="my-workspace",
    config=StreamConfig(
        data_output=["detections", "count"],  # What to send via data channel
        stream_output=["annotated_image"],     # What to stream as video
    ),
)

@session.on_data("detections")
def on_detections(detections, metadata):
    # Real-time detection results via data channel
    process_detections(detections, metadata.frame_id)

@session.on_frame
def on_frame(frame, metadata):
    # Annotated video frame (numpy array)
    display_or_save(frame)

session.run()  # Blocks until stream ends
```

**Capabilities**: Real-time bidirectional streaming, low latency, browser-
compatible via WebRTC, supports all source types (Webcam, RTSP, MJPEG, file
upload, manual frame injection), chunked binary data channel with ACK-based
flow control.

### Backend Selection Logic

```python
def select_backend(config: AgentConfig, task: str) -> str:
    """Select the best execution backend for a task."""
    if config.inference_mode == "direct":
        return "direct"        # Always use direct if configured
    if task == "realtime_visualization":
        return "webrtc"        # Browser viewing needs WebRTC
    if task == "continuous_monitoring":
        return "http_pipeline" # Hosted pipeline for long-running monitoring
    return "http_client"       # Default: HTTP for single-shot tasks
```

---

## 3. Interface Contracts — Revised

### 3.1 `AgentCore` — The Agentic Loop

```python
# inference_agent/core/protocols.py

from typing import Protocol, AsyncIterator, Any
from dataclasses import dataclass, field
from enum import Enum
import time

class EventType(Enum):
    THINKING = "thinking"          # Agent is reasoning (extended thinking)
    TOOL_CALL = "tool_call"        # Agent is calling a tool
    TOOL_RESULT = "tool_result"    # Tool returned a result
    RESPONSE = "response"          # Final text response to user
    ALERT = "alert"                # Proactive alert from monitoring
    STREAM_EVENT = "stream_event"  # Notable event on a video stream
    ERROR = "error"                # Error occurred
    ASK_USER = "ask_user"          # Agent needs user input

@dataclass
class AgentEvent:
    type: EventType
    data: Any
    timestamp: float = field(default_factory=time.time)

class AgentCore(Protocol):

    async def process_message(
        self,
        message: str,
        attachments: list[bytes] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Process a user message. Yields events as the agent works."""
        ...

    async def process_heartbeat(self) -> AsyncIterator[AgentEvent]:
        """Run a heartbeat check. Called periodically by the scheduler.

        Reads HEARTBEAT.md for checklist, checks active streams,
        yields alert events if anything significant is detected.
        """
        ...

    async def shutdown(self) -> None:
        """Graceful shutdown: stop streams, flush memory, save state."""
        ...
```

### 3.2 `LLMClient` — Claude API Wrapper

```python
# inference_agent/llm/protocols.py

@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: dict             # JSON Schema
    cache_control: dict | None = None  # For prompt caching on last tool

@dataclass
class LLMResponse:
    content_blocks: list[dict]     # Raw content blocks (text, tool_use, thinking)
    text: str | None               # Extracted text (convenience)
    tool_calls: list[ToolCall]     # Extracted tool calls (convenience)
    stop_reason: str               # "end_turn", "tool_use", "max_tokens"
    usage: dict                    # input_tokens, output_tokens, cache stats

class LLMClient(Protocol):

    async def chat(
        self,
        messages: list[dict],
        tools: list[ToolDefinition],
        system: list[dict],           # System prompt as content blocks
        max_tokens: int = 4096,
        thinking: dict | None = None, # {"type":"enabled","budget_tokens":N}
    ) -> LLMResponse:
        """Send messages to Claude. Handles caching, retries, thinking."""
        ...
```

### 3.3 `InferenceBackend` — Unified Backend Interface

```python
# inference_agent/backends/protocols.py

from typing import Protocol
from dataclasses import dataclass

@dataclass
class PipelineHandle:
    """Reference to a running pipeline, backend-agnostic."""
    pipeline_id: str
    backend: str            # "direct", "http", "webrtc"
    video_reference: str
    workflow_description: str  # Human-readable summary
    started_at: float
    status: str             # "running", "paused", "stopped", "error"

@dataclass
class InferenceResult:
    """Unified result from any backend."""
    predictions: dict       # Workflow output dict
    frame: bytes | None     # JPEG-encoded frame (optional)
    frame_id: int | None
    timestamp: float

class InferenceBackend(Protocol):
    """Unified interface for all three execution backends."""

    async def run_single(
        self,
        image: str | bytes,
        model_id: str | None = None,
        workflow_spec: dict | None = None,
        workspace_name: str | None = None,
        workflow_id: str | None = None,
        parameters: dict | None = None,
    ) -> InferenceResult:
        """Run inference on a single image.

        The caller provides EITHER:
        - model_id (for simple single-model inference), OR
        - workflow_spec (agent-composed workflow), OR
        - workspace_name + workflow_id (user's pre-built workflow on Roboflow)
        """
        ...

    async def start_pipeline(
        self,
        video_reference: str,
        workflow_spec: dict | None = None,
        workspace_name: str | None = None,
        workflow_id: str | None = None,
        max_fps: float | None = None,
        parameters: dict | None = None,
    ) -> PipelineHandle:
        """Start a continuous video processing pipeline.

        Same three modes: agent-composed spec, user's workflow ID, or both.
        """
        ...

    async def consume_results(
        self,
        pipeline_id: str,
        max_results: int = 5,
    ) -> list[InferenceResult]:
        """Consume latest results from a running pipeline."""
        ...

    async def stop_pipeline(self, pipeline_id: str) -> None: ...
    async def pause_pipeline(self, pipeline_id: str) -> None: ...
    async def resume_pipeline(self, pipeline_id: str) -> None: ...
    async def list_pipelines(self) -> list[PipelineHandle]: ...
    async def get_pipeline_status(self, pipeline_id: str) -> PipelineHandle: ...
```

### 3.4 `Memory` — Persistent Workspace Memory

```python
# inference_agent/memory/protocols.py

class Memory(Protocol):

    async def store(self, content: str, category: str,
                    metadata: dict | None = None) -> None:
        """Append to the appropriate memory file."""
        ...

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Hybrid BM25 + vector search over all memory files."""
        ...

    async def get_daily_observations(self, date: str | None = None,
                                      camera_id: str | None = None) -> str:
        """Get observations for a specific day (default: today)."""
        ...

    def load_workspace_files(self) -> dict[str, str]:
        """Load workspace context files for system prompt injection.

        Returns dict: filename -> content for:
        - AGENTS.md (operating instructions)
        - USER.md (user identity and preferences)
        - CAMERAS.md (camera registry)
        - HEARTBEAT.md (heartbeat checklist)
        - Recent daily observations
        """
        ...

    async def save_active_state(self, state: dict) -> None:
        """Persist active pipelines/tasks for restart recovery."""
        ...

    async def load_active_state(self) -> dict | None:
        """Load previously active state (for restart recovery)."""
        ...
```

### 3.5 `UserInterface` — Channel Abstraction

```python
# inference_agent/interfaces/protocols.py

class UserInterface(Protocol):

    async def start(self, event_callback: Callable[[AgentEvent], Awaitable[None]]) -> None:
        """Start the interface. Calls event_callback for proactive alerts."""
        ...

    async def run_conversation(self, agent: AgentCore) -> None:
        """Main loop: receive messages, call agent, display events."""
        ...

    async def emit_event(self, event: AgentEvent) -> None:
        """Push a proactive event to the user (alert, status update)."""
        ...
```

---

## 4. OpenClaw-Inspired Workspace & Lifecycle

### 4.1 Workspace Directory Structure

Following OpenClaw's pattern of plain files as the source of truth:

```
~/.vision_agent/                       # Default workspace
├── config.yaml                        # Agent configuration (validated w/ Pydantic)
├── AGENTS.md                          # Operating instructions for the agent
│                                      #   (injected into system prompt every turn)
├── USER.md                            # User identity, preferences, contact info
│                                      #   (injected into system prompt every turn)
├── CAMERAS.md                         # Camera registry with names and URLs
│                                      #   (injected into system prompt every turn)
├── HEARTBEAT.md                       # Heartbeat checklist
│                                      #   ("Check cameras for anomalies,
│                                      #    report only if something new")
├── skills/                            # Domain-specific instruction packs
│   ├── ppe-compliance/
│   │   └── SKILL.md                   # "How to monitor PPE compliance"
│   ├── parking-lot/
│   │   └── SKILL.md                   # "How to count vehicles and occupancy"
│   ├── package-delivery/
│   │   └── SKILL.md                   # "How to detect package deliveries"
│   └── quality-inspection/
│       └── SKILL.md                   # "How to inspect items on conveyor"
├── memory/
│   ├── KNOWLEDGE.md                   # Long-term learned knowledge
│   ├── 2026-02-11.md                  # Daily observation log (today)
│   ├── 2026-02-10.md                  # Yesterday
│   └── ...
├── sessions/
│   ├── session_abc123.jsonl           # JSONL transcript (append-only)
│   └── ...
├── active_state.json                  # Running pipelines, tasks (for recovery)
└── .index/
    └── memory.db                      # SQLite FTS5 + vec0 index
```

### 4.2 Workspace Files Loaded Into System Prompt

Following OpenClaw's pattern, certain workspace files are injected into the
system prompt on **every turn**. This makes them the primary way to configure
the agent's behavior.

**AGENTS.md** — Operating instructions:
```markdown
# Vision Agent Instructions

## Behavior
- When the user asks you to monitor a camera, always confirm the setup
- When you detect a safety violation, include a frame snapshot in the alert
- Summarize observations hourly, not per-frame
- Use YOLO-World for zero-shot detection unless the user specifies a model

## Defaults
- Default confidence threshold: 0.5
- Default max FPS for monitoring: 5
- Alert channel: Slack (#security-alerts)

## Domain Context
- This is a warehouse facility
- Cameras 1-3 are on the production floor
- Camera 4 is the loading dock
- Shift change at 6am, 2pm, 10pm
```

**CAMERAS.md** — Camera registry:
```markdown
# Cameras

| Name | URL | Location | Notes |
|------|-----|----------|-------|
| cam1 | rtsp://192.168.1.50/stream1 | Production Floor East | 1080p, 15fps |
| cam2 | rtsp://192.168.1.51/stream1 | Production Floor West | 1080p, 15fps |
| cam3 | rtsp://192.168.1.52/stream1 | Assembly Line | 720p, 30fps |
| cam4 | rtsp://192.168.1.53/stream1 | Loading Dock | 1080p, 10fps |
```

**HEARTBEAT.md** — Proactive check checklist:
```markdown
# Heartbeat Checklist

Every 15 minutes, check the following:

1. Are all registered cameras still streaming? If any dropped, alert the user.
2. Check production floor cameras for PPE violations. Alert immediately.
3. Check loading dock for vehicles that have been stationary > 30 minutes.
4. If it's near shift change (±30min), note any unusual activity.
5. If nothing notable, respond with HEARTBEAT_OK (silent, no user notification).
```

### 4.3 Skills — Loadable Domain Expertise

Skills are Markdown files that provide **domain-specific instructions** the
agent loads on demand. They are NOT executable code — they guide the agent's
reasoning and tool usage for specific scenarios.

**Example: `skills/ppe-compliance/SKILL.md`**
```markdown
---
name: ppe-compliance
description: Monitor PPE compliance on camera feeds
models:
  - hard-hat-detection/3   # Roboflow model for hard hats
  - safety-vest/2          # Roboflow model for safety vests
---

# PPE Compliance Monitoring

## Setup
1. Use the `hard-hat-detection/3` model for hard hat detection
2. Use the `safety-vest/2` model for safety vest detection
3. Set confidence threshold to 0.6 (these models are well-calibrated)
4. Add ByteTracker for consistent person tracking across frames

## Workflow Pattern
- Detect people → detect hard hats → detect safety vests
- Use DetectionsFilter to find people WITHOUT matching PPE
- Use DeltaFilter to avoid re-alerting on the same person

## Alert Format
Include: camera name, timestamp, annotated frame, violation type
(missing hard hat / missing vest / both)
```

The agent reads skill files when the user's request matches a skill, or when
explicitly asked: "use the PPE compliance skill on camera 3."

### 4.4 Session Transcripts (JSONL)

Every interaction is logged as append-only JSONL, following OpenClaw's pattern:

```jsonl
{"id":"msg_001","type":"user","content":"Watch cam1 for people without hard hats","ts":"2026-02-11T10:00:00Z"}
{"id":"msg_002","type":"assistant","content":[{"type":"thinking","text":"User wants PPE..."},{"type":"tool_use","id":"tu_01","name":"start_pipeline",...}],"ts":"2026-02-11T10:00:02Z"}
{"id":"msg_003","type":"tool_result","tool_use_id":"tu_01","content":"Pipeline started: pipe_abc","ts":"2026-02-11T10:00:05Z"}
{"id":"msg_004","type":"assistant","content":[{"type":"text","text":"I've started monitoring..."}],"ts":"2026-02-11T10:00:06Z"}
```

This provides: full audit trail, crash recovery (replay from last checkpoint),
debugging, and analytics.

### 4.5 Heartbeat & Cron

**Heartbeat** (inspired by OpenClaw):
- Fires every N minutes (configurable, default 15)
- Reads `HEARTBEAT.md` for the checklist
- Runs an agent turn with a special system prompt:
  "You are running a heartbeat check. Review the checklist and your active
  streams. If nothing notable, respond with HEARTBEAT_OK. Otherwise, compose
  an alert message."
- If the response is `HEARTBEAT_OK`, silently proceed (no user notification)
- If the response contains content, emit it as an alert event

**Cron** (scheduled tasks):
- Stored in `active_state.json`
- Examples: "Daily summary at 6pm", "Capture snapshot every hour"
- Runs in isolated context (doesn't pollute main conversation)

---

## 5. Tool Inventory — Revised

### 5.1 Core Tools (Always Available)

```python
# These are always in the tool list. ~15 tools, fits comfortably in cache.

tools = {
    # === Inference (single image) ===
    "run_inference": {
        "description": "Run inference on a single image using a model or workflow. "
                       "Accepts a Roboflow model_id (e.g. 'my-workspace/model/3') "
                       "or a workflow (by ID or inline spec).",
        "params": {
            "image": "str — file path, URL, or base64",
            "model_id": "str | None — Roboflow model ID",
            "workspace_name": "str | None — for workflow by ID",
            "workflow_id": "str | None — for workflow by ID",
            "workflow_spec": "dict | None — inline workflow specification",
            "parameters": "dict | None — additional workflow parameters",
            "confidence": "float | None — confidence threshold (default 0.4)",
        },
    },

    "ask_about_image": {
        "description": "Ask a vision-language model a question about an image. "
                       "Uses Florence-2, PaliGemma, Qwen2.5-VL, or Claude's own vision.",
        "params": {
            "image": "str — file path, URL, or base64",
            "question": "str — question to ask about the image",
            "model": "str — 'florence2', 'qwen2.5-vl', 'paligemma' (default 'florence2')",
        },
    },

    "detect_zero_shot": {
        "description": "Detect objects by text description without a trained model. "
                       "Uses YOLO-World for fast detection by class name prompts.",
        "params": {
            "image": "str",
            "class_names": "list[str] — objects to detect, e.g. ['person', 'hard hat']",
            "confidence": "float (default 0.3)",
        },
    },

    # === Pipeline Management ===
    "start_pipeline": {
        "description": "Start continuous video processing on a camera/video source. "
                       "The pipeline runs a workflow on every frame. Accepts: "
                       "(1) a Roboflow workflow by workspace_name + workflow_id, "
                       "(2) an inline workflow_spec, or (3) a simple model_id. "
                       "Video sources: RTSP URL, webcam device, video file path.",
        "params": {
            "video_reference": "str — RTSP URL, file path, webcam (0,1,2)",
            "pipeline_name": "str — human-friendly name for this pipeline",
            "workflow_spec": "dict | None — inline workflow specification",
            "workspace_name": "str | None",
            "workflow_id": "str | None",
            "model_id": "str | None — simple detection model",
            "max_fps": "float | None — frame rate limit",
            "parameters": "dict | None",
        },
    },

    "get_pipeline_results": {
        "description": "Get the latest results from a running pipeline. "
                       "Returns detection summaries. Set include_frame=true "
                       "to get an annotated snapshot.",
        "params": {
            "pipeline_id": "str",
            "max_results": "int (default 5)",
            "include_frame": "bool (default false)",
        },
    },

    "list_pipelines": {
        "description": "List all active video processing pipelines with status.",
    },

    "stop_pipeline": {
        "description": "Stop a running video pipeline.",
        "params": {"pipeline_id": "str"},
    },

    "pause_pipeline": {
        "description": "Pause a running pipeline (can be resumed).",
        "params": {"pipeline_id": "str"},
    },

    "resume_pipeline": {
        "description": "Resume a paused pipeline.",
        "params": {"pipeline_id": "str"},
    },

    # === Workflow Discovery (on-demand) ===
    "list_workflow_blocks": {
        "description": "List available workflow blocks, optionally filtered by category "
                       "(model, analytics, visualization, transformation, sink, flow_control, "
                       "classical_cv, fusion, tracking). Use this when you need to compose "
                       "a custom workflow and want to know what blocks exist.",
        "params": {"category": "str | None"},
    },

    "get_block_details": {
        "description": "Get the full schema, parameters, and example usage for a "
                       "specific workflow block. Use this before composing a workflow "
                       "to understand a block's inputs, outputs, and required parameters.",
        "params": {"block_type": "str — e.g. 'roboflow_core/line_counter@v2'"},
    },

    # === Memory ===
    "remember": {
        "description": "Store an observation, fact, or user preference in persistent memory. "
                       "Use this to record significant events, patterns, or things the user "
                       "asks you to remember.",
        "params": {
            "content": "str — what to remember",
            "category": "str — 'observation', 'knowledge', 'preference'",
            "camera_id": "str | None — associated camera",
        },
    },

    "recall": {
        "description": "Search memory for relevant past observations, knowledge, or preferences.",
        "params": {
            "query": "str — search query",
            "max_results": "int (default 5)",
        },
    },

    # === Reasoning ===
    "think": {
        "description": "Use this to think through complex problems step-by-step before "
                       "taking action. Use when you need to plan a workflow, analyze "
                       "ambiguous results, or decide between approaches. No side effects.",
        "params": {"thought": "str — your step-by-step reasoning"},
    },
}
```

### 5.2 Tool Count & Caching Strategy

- **15 tools** total (well within Claude's comfort zone)
- All tool definitions are **prompt-cached** via `cache_control` on the last
  tool definition
- System prompt (identity + workspace files + block summary) is also cached
- Total cached prefix: ~10K tokens → $0.001 per call at cache-read rate

---

## 6. System Prompt Design

### 6.1 Structure

The system prompt is built dynamically per turn, following OpenClaw's pattern:

```python
def build_system_prompt(workspace: Memory, tools: ToolRegistry,
                        active_pipelines: list[PipelineHandle]) -> list[dict]:
    """Build system prompt as cacheable content blocks."""

    # --- Cached section (stable across turns) ---
    cached_blocks = []

    # Core identity + instructions
    cached_blocks.append({
        "type": "text",
        "text": CORE_IDENTITY_PROMPT,
    })

    # Workspace files (AGENTS.md, USER.md, CAMERAS.md)
    ws_files = workspace.load_workspace_files()
    for filename, content in ws_files.items():
        if content.strip():
            cached_blocks.append({
                "type": "text",
                "text": f"# {filename}\n\n{content}",
            })

    # Block summary (available workflow blocks)
    cached_blocks.append({
        "type": "text",
        "text": tools.get_capabilities_summary(),
    })

    # Workflow examples
    cached_blocks.append({
        "type": "text",
        "text": WORKFLOW_EXAMPLES,
    })

    # Mark last cached block for prompt caching
    cached_blocks[-1]["cache_control"] = {"type": "ephemeral"}

    # --- Dynamic section (changes every turn) ---
    dynamic_blocks = []

    # Active pipelines summary
    if active_pipelines:
        summary = format_active_pipelines(active_pipelines)
        dynamic_blocks.append({"type": "text", "text": summary})

    # Available skills
    skills = workspace.list_available_skills()
    if skills:
        skills_text = format_skills_list(skills)
        dynamic_blocks.append({"type": "text", "text": skills_text})

    return cached_blocks + dynamic_blocks
```

### 6.2 Core Identity Prompt

```
You are a Vision Agent — an autonomous computer vision assistant powered by
Roboflow Inference. You help users monitor cameras, analyze images, and detect
events using state-of-the-art CV models.

## How You Work

You have tools for running inference on images and managing video processing
pipelines. For continuous monitoring tasks, you create Workflow specifications
(JSON) that define processing pipelines, then deploy them on video sources.

## Important Behaviors

1. **Ask before assuming.** If the user mentions a model or camera, ask for
   the specific model_id or RTSP URL if not provided. They may have fine-tuned
   models on Roboflow they want to use.

2. **Check for existing workflows.** The user may already have workflows
   configured on Roboflow. Ask if they have one, or if they want you to
   compose a new one.

3. **Use the think tool** for complex tasks. Before composing a workflow,
   think through what blocks you need, how they connect, and what parameters
   to set.

4. **Validate before deploying.** Always check that your workflow spec is
   correct. If deployment fails, read the error and fix the issue.

5. **Be proactive but not noisy.** When monitoring, only alert on significant
   events. Use local filtering in the workflow (DeltaFilter, ContinueIf) to
   avoid redundant alerts.

6. **Show, don't tell.** When reporting detections, include an annotated frame
   snapshot (set include_frame=true in get_pipeline_results).

## Workflow Specification Format

{workflow_format_docs}

## Supported Video Sources
- RTSP streams: rtsp://host/path
- Webcam devices: 0, 1, 2 (integer)
- Video files: /path/to/video.mp4
- MJPEG streams: http://host/stream

## Model References
- Roboflow models: "workspace/project/version" (e.g., "my-ws/hard-hat/3")
- Zero-shot: use the detect_zero_shot tool with class names
- COCO pre-trained: "coco/9" for 80 common object classes
```

---

## 7. Claude API Integration Details

### 7.1 Prompt Caching Placement

```python
# Cache the system prompt + tool definitions for 90% input cost savings.
# Cache order: tools → system → messages

# Tools: add cache_control to LAST tool definition
tools[-1]["cache_control"] = {"type": "ephemeral"}

# System: add cache_control to last stable content block
system_blocks[-1]["cache_control"] = {"type": "ephemeral"}

# This caches ~10K tokens. At Sonnet pricing:
# Cache write: 10K * $3.75/MTok = $0.0375 (first call only)
# Cache read:  10K * $0.30/MTok = $0.003  (every subsequent call)
```

### 7.2 Images in Tool Results

When the agent needs to see a frame (e.g., from `get_pipeline_results` with
`include_frame=true`), the tool result includes image content blocks:

```python
# Tool result with image
tool_result = {
    "type": "tool_result",
    "tool_use_id": call.id,
    "content": [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64_jpeg,  # Resized to ~640px max dimension
            },
        },
        {
            "type": "text",
            "text": "Frame #1234 from cam1 (loading dock). "
                    "3 detections: person (0.92), forklift (0.87), pallet (0.76)",
        },
    ],
}
```

### 7.3 Extended Thinking for Complex Workflows

For complex workflow composition, enable extended thinking with the interleaved
thinking beta header:

```python
response = await self._client.messages.create(
    model=self._model,
    max_tokens=8192,
    thinking={"type": "enabled", "budget_tokens": 4096},
    tools=tool_defs,
    system=system_blocks,
    messages=messages,
    extra_headers={"anthropic-beta": "interleaved-thinking-2025-05-14"},
)
```

This lets Claude reason between tool calls — e.g., after getting block details,
it can think about how to wire blocks together before composing the workflow.

### 7.4 The "think" Tool

In addition to extended thinking, we include a `think` tool (recommended by
Anthropic) that gives Claude an explicit scratchpad during multi-step agentic
workflows:

```python
{
    "name": "think",
    "description": "Think through a problem step-by-step. No side effects.",
    "input_schema": {
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": "Your step-by-step reasoning"
            }
        },
        "required": ["thought"]
    }
}
```

When Claude calls `think`, the executor returns an empty string and continues
the loop. This is especially useful for:
- Planning a workflow before composing it
- Analyzing ambiguous detection results before alerting
- Deciding which backend to use for a task

### 7.5 Conversation History Management

To prevent context overflow during long sessions:

1. **Token budget tracking**: After each turn, estimate total context size
2. **Compaction trigger**: When approaching 80% of context window, summarize
   old messages and write important state to memory files (pre-compaction flush,
   per OpenClaw pattern)
3. **Thinking block handling**: Include thinking blocks in history (Claude
   requires them via signatures), but the API automatically excludes them from
   context budget
4. **Result truncation**: Pipeline results are summarized before sending to
   Claude to control token growth

---

## 8. Work Streams (Parallelizable)

```
                    ┌──────────────────────────┐
                    │  WS0: Interfaces +       │
                    │  Workspace Skeleton       │
                    │  (2-3 days)               │
                    └─────────┬────────────────┘
                              │
        ┌─────────────────────┼──────────────────────┐
        │                     │                      │
        ▼                     ▼                      ▼
┌───────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ WS1: Agent    │  │ WS2: Backends    │  │ WS3: Memory +    │
│ Core + LLM    │  │ + Tools          │  │ Workspace        │
│ + System      │  │ (5-7 days)       │  │ (4-6 days)       │
│ Prompt        │  │                  │  │                  │
│ (5-7 days)    │  │ WS2a: HTTP       │  │                  │
│               │  │ WS2b: Direct SDK │  │                  │
│               │  │ WS2c: Workflow   │  │                  │
│               │  │       Composer   │  │                  │
└───────┬───────┘  └────────┬─────────┘  └────────┬─────────┘
        │                   │                      │
        └───────────────────┼──────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  WS4: Integration     │
                │  + MVP Validation     │
                │  (3-5 days)           │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  WS5: Heartbeat +     │
                │  Proactive Monitoring │
                │  + Skills             │
                │  (5-7 days)           │
                └───────────────────────┘
```

**WS2 can be further parallelized** — the three backends (HTTP, Direct, Workflow
Composer) share the `InferenceBackend` interface and can be developed by
separate people simultaneously.

---

## 9. MVP — Minimum Testable Assumption

### What We're Derisking

The #1 risk is: **Can Claude reliably orchestrate Inference capabilities from
natural language, including composing valid workflows and managing pipelines?**

The #2 risk is: **Can the agent use existing user resources (models, workflows)
on Roboflow, not just auto-discovered capabilities?**

### MVP Scope

```
MVP = CLI REPL + Agent Core + HTTP Backend + 15 Tools
```

**In scope:**
- CLI REPL interface
- Agent loop with Claude API (tool-use + prompt caching + think tool)
- HTTP client backend (InferenceHTTPClient to a running inference server)
- All 15 tools from Section 5.1
- System prompt with workspace file injection (AGENTS.md, CAMERAS.md)
- Workflow composer (block discovery + schema generation)
- Minimal memory (file append, no search yet)
- JSONL session logging

**Out of scope for MVP:**
- Direct SDK backend (added in Phase 2)
- WebRTC backend (added in Phase 2)
- Hybrid memory search (added in Phase 2)
- Heartbeat/proactive monitoring (added in Phase 3)
- Skills system (added in Phase 3)
- Multi-channel interfaces (added in Phase 4)

### MVP Test Scenarios

| # | Instruction | Tests | Success |
|---|------------|-------|---------|
| 1 | "I have a model my-ws/hard-hat/3. Run it on this image." | User-provided model_id, single image | Correct detections returned |
| 2 | "I have a workflow called 'ppe-check' in my 'acme' workspace. Run it on cam1." | User-provided workflow_id + hosted pipeline | Pipeline starts, results consumable |
| 3 | "Watch this video for people." (no model specified) | Agent asks user for model preference or suggests YOLO-World/COCO | Appropriate follow-up question or zero-shot detection |
| 4 | "Count cars crossing the middle of the frame in test.mp4" | Agent composes workflow: detection → tracker → line_counter | Valid workflow, accurate counts |
| 5 | "What blocks are available for tracking?" | Discovery tool usage | Correct block list returned |
| 6 | "What's in this image?" [attach photo] | VLM tool with image attachment | Accurate description |
| 7 | Agent composes invalid workflow | Self-correction loop | Fixes and deploys within 2 attempts |
| 8 | "Stop all pipelines and summarize what you observed." | Pipeline management + recall | Clean shutdown, observation summary |
| 9 | "Use my workflow 'inventory-count' on rtsp://192.168.1.52/stream1 at 2 FPS" | User's workflow by ID + specific params | Hosted pipeline with correct config |
| 10 | Complex: "Monitor cam1 and cam2. Detect people on cam1, vehicles on cam2. Alert me if a person is near a vehicle on cam2." | Multi-pipeline, think tool usage | Two pipelines running, coherent plan |

---

## 10. Risk Register & Derisking

### Risk 1: Tool Schema Explosion → MITIGATED by Design

Two-tier approach: 15 action tools + 2 discovery tools. Block details are
returned on-demand, not embedded in tool definitions. System prompt contains
only a one-line summary per block (~2.5K tokens for 130+ blocks).

### Risk 2: Workflow Composition Accuracy → DERISK in MVP

**Mitigation layers:**
1. Few-shot examples in system prompt (3 working workflow patterns)
2. `think` tool — Claude reasons about the workflow before composing it
3. Validate-before-deploy in `start_pipeline` tool
4. Self-correction: validation errors are returned as `is_error` tool results
5. Template approach: user can reference existing Roboflow workflows instead
   of the agent composing from scratch

**MVP test:** Benchmark 20 natural language instructions. Target: >70% valid
on first attempt, >90% after self-correction.

### Risk 3: User Doesn't Know What They Need → DERISK by UX

The agent should be conversational, not assume. If the user says "watch for
defects," the agent should ask:
- "Do you have a trained defect detection model on Roboflow? If so, what's
  the model ID?"
- "Or would you like me to try zero-shot detection? I can look for specific
  types of defects using YOLO-World."
- "Do you already have a workflow set up for this?"

This is encoded in the system prompt's "ask before assuming" rule.

### Risk 4: Backend Complexity → MITIGATED by Interface

The `InferenceBackend` protocol abstracts all three backends. The agent core
doesn't know or care which backend is in use. The tool executor selects the
backend based on config + task type.

### Risk 5: Cost for Always-On Monitoring → DERISK in Phase 3

Heartbeat approach: Claude is only consulted periodically, not per-frame. The
workflow pipeline runs at 30+ FPS autonomously; the LLM is called at most once
per heartbeat interval (default 15 min). With prompt caching, each heartbeat
check costs ~$0.01.

---

## 11. Dependency Graph & Sequencing

### Phase 0: Foundation (Days 1-3)

| Task | Deliverable |
|------|-------------|
| P0.1 Define Protocol interfaces | All `protocols.py` files |
| P0.2 Project skeleton + pyproject.toml | `inference_agent/` package |
| P0.3 Workspace directory initialization | Default workspace creation |
| P0.4 Contract tests for each Protocol | Pytest fixtures with mock implementations |

### Phase 1: Parallel Streams (Days 3-10)

**WS1: Agent Core + LLM Client**

| Task | Deliverable |
|------|-------------|
| WS1.1 `ClaudeClient` — Anthropic SDK wrapper | `llm/claude_client.py` |
| WS1.2 `VisionAgent` — agentic loop | `core/agent.py` |
| WS1.3 System prompt builder | `core/prompt_builder.py` |
| WS1.4 System prompt engineering + few-shot examples | Prompt text + test results |
| WS1.5 `CLIInterface` | `interfaces/cli.py` |
| WS1.6 Session transcript logger (JSONL) | `core/session_log.py` |
| WS1.7 Unit tests with mock backend | Agent loop tests |

**WS2: Backends + Tools**

| Task | Deliverable |
|------|-------------|
| WS2.1 `HTTPBackend` — InferenceHTTPClient wrapper | `backends/http_backend.py` |
| WS2.2 Block discovery (from HTTP or direct import) | `capabilities/block_discovery.py` |
| WS2.3 Schema generator (block → LLM tool description) | `capabilities/schema_generator.py` |
| WS2.4 `ToolRegistry` + all 15 tool implementations | `tools/` directory |
| WS2.5 Result summarizer (raw output → human-readable) | `tools/result_summarizer.py` |
| WS2.6 Unit tests for each tool | Tool tests with mock backend |

**WS3: Memory + Workspace**

| Task | Deliverable |
|------|-------------|
| WS3.1 Workspace initialization + file management | `memory/workspace.py` |
| WS3.2 File-based memory store (append to .md files) | `memory/store.py` |
| WS3.3 BM25 search (SQLite FTS5) | `memory/search.py` |
| WS3.4 `remember` and `recall` tool implementations | `memory/tools.py` |
| WS3.5 Workspace file loader (for system prompt) | `memory/workspace.py` |
| WS3.6 Active state persistence (JSON) | `memory/state.py` |

### Phase 2: Integration + MVP (Days 8-14)

| Task | Deliverable |
|------|-------------|
| I.1 Wire all modules, end-to-end smoke test | Working agent |
| I.2 Run MVP test scenarios 1-10 | Test results |
| I.3 Benchmark: workflow composition accuracy | Accuracy report |
| I.4 Add Direct SDK backend | `backends/direct_backend.py` |
| I.5 Add WebRTC backend | `backends/webrtc_backend.py` |
| I.6 Vector search (CLIP embeddings) for memory | `memory/search.py` |
| I.7 Performance profiling + cost analysis | Report |

### Phase 3: Proactive Monitoring (Days 12-18)

| Task | Deliverable |
|------|-------------|
| S.1 Heartbeat scheduler | `core/heartbeat.py` |
| S.2 Event detection + LLM escalation logic | `core/event_handler.py` |
| S.3 Cron scheduler for periodic tasks | `core/scheduler.py` |
| S.4 Skills loader + system prompt injection | `skills/loader.py` |
| S.5 Pre-compaction memory flush | `core/context_guard.py` |
| S.6 Multi-camera correlation | Backend enhancements |

### Phase 4: Multi-Channel (Days 16-22)

| Task | Deliverable |
|------|-------------|
| C.1 Slack bot interface | `interfaces/slack.py` |
| C.2 REST API interface | `interfaces/rest.py` |
| C.3 Alert routing (different channels for different events) | `core/alert_router.py` |
| C.4 Web dashboard (optional) | `interfaces/web/` |

---

## 12. Detailed Implementation — Each Work Stream

### WS1: Agent Core + LLM Client

#### WS1.1: ClaudeClient

```python
# inference_agent/llm/claude_client.py

import anthropic
from typing import AsyncIterator

class ClaudeClient:
    """Claude API wrapper with tool-use, caching, thinking, and streaming."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5"):
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._total_usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}

    async def chat(self, messages, tools, system, max_tokens=4096,
                   thinking=None) -> LLMResponse:
        """Core implementation:

        1. Convert ToolDefinition objects to Anthropic API dicts.
           Add cache_control to the LAST tool definition.
        2. Build the request:
           - model, max_tokens, system, tools, messages
           - If thinking: add thinking param + interleaved-thinking beta header
        3. Call self._client.messages.create(**request)
        4. Handle rate limits: retry with exponential backoff on 429/529
        5. Parse response:
           - Extract text blocks → response.text
           - Extract tool_use blocks → response.tool_calls
           - Extract thinking blocks → response.thinking
           - Copy stop_reason and usage
        6. Accumulate usage stats for cost monitoring
        7. Return LLMResponse
        """
        ...

    def format_tool_result(self, tool_call_id: str, content,
                           is_error: bool = False) -> dict:
        """Format a tool result for the messages array.

        If content is a string: simple text result.
        If content is a list: content blocks (text + images).
        If content is a dict: JSON-serialize to string.

        For images: encode as base64 JPEG, max 640px on longest side.
        """
        ...
```

#### WS1.2: VisionAgent (Agent Loop)

```python
# inference_agent/core/agent.py

class VisionAgent:

    async def process_message(self, message, attachments=None):
        """The agentic loop. Pseudocode:

        1. Append user message to self._messages
           - If attachments: include as image content blocks
        2. Build system prompt via PromptBuilder
        3. Loop (max 20 iterations):
           a. Call self._llm.chat(messages, tools, system, thinking=...)
           b. Append assistant response (including thinking blocks) to messages
           c. If stop_reason == "end_turn":
              - Yield AgentEvent(RESPONSE, response.text)
              - Break
           d. If stop_reason == "tool_use":
              - For each tool_call in response.tool_calls:
                - Yield AgentEvent(TOOL_CALL, {name, args})
                - result = await self._tools.execute(tool_call)
                - Yield AgentEvent(TOOL_RESULT, {name, summary})
              - Append tool_result user message to messages
              - Continue loop
           e. If stop_reason == "max_tokens":
              - Yield AgentEvent(ERROR, "Response truncated")
              - Break
        4. Log full turn to session JSONL
        5. Check context size → trigger compaction if needed
        """
        ...

    async def process_heartbeat(self):
        """Heartbeat turn. Pseudocode:

        1. Load HEARTBEAT.md from workspace
        2. Build heartbeat system prompt (lighter than full prompt)
        3. Compose a user message:
           "Heartbeat check. Active pipelines: [list]. Please review
            your checklist."
        4. Run agent loop (same as process_message but in heartbeat context)
        5. If response == "HEARTBEAT_OK": yield nothing
        6. Else: yield AgentEvent(ALERT, response)
        """
        ...
```

#### WS1.3: PromptBuilder

```python
# inference_agent/core/prompt_builder.py

class PromptBuilder:
    """Builds system prompt from workspace files + capabilities."""

    def build(self, workspace: Memory, tools: ToolRegistry,
              active_pipelines: list, mode: str = "full") -> list[dict]:
        """Build system prompt as content blocks.

        mode="full": All sections (normal conversation)
        mode="heartbeat": Lighter prompt (skip examples, skill list)

        Returns list of dicts suitable for the Anthropic API system param.
        Last stable block gets cache_control for prompt caching.
        """
        ...
```

### WS2: Backends + Tools

#### WS2.1: HTTPBackend

```python
# inference_agent/backends/http_backend.py

from inference_sdk import InferenceHTTPClient

class HTTPBackend:
    """InferenceBackend implementation using the HTTP client SDK."""

    def __init__(self, api_url: str, api_key: str | None = None):
        self._client = InferenceHTTPClient(api_url=api_url, api_key=api_key)

    async def run_single(self, image, model_id=None, workflow_spec=None,
                         workspace_name=None, workflow_id=None,
                         parameters=None) -> InferenceResult:
        """Implementation:

        If model_id provided:
            result = self._client.infer(image, model_id=model_id)
        Elif workspace_name and workflow_id:
            result = self._client.run_workflow(
                workspace_name=workspace_name,
                workflow_id=workflow_id,
                images={"image": image},
                parameters=parameters,
            )
        Elif workflow_spec:
            result = self._client.run_workflow(
                specification=workflow_spec,
                images={"image": image},
                parameters=parameters,
            )

        Return InferenceResult(predictions=result, ...)
        """
        ...

    async def start_pipeline(self, video_reference, workflow_spec=None,
                             workspace_name=None, workflow_id=None,
                             max_fps=None, parameters=None) -> PipelineHandle:
        """Implementation:

        response = self._client.start_inference_pipeline_with_workflow(
            video_reference=video_reference,
            workflow_specification=workflow_spec,
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            max_fps=max_fps,
            workflows_parameters=parameters,
        )
        return PipelineHandle(pipeline_id=response["pipeline_id"], ...)
        """
        ...

    async def consume_results(self, pipeline_id, max_results=5):
        """Implementation:

        results = []
        for _ in range(max_results):
            try:
                r = self._client.consume_inference_pipeline_result(pipeline_id)
                results.append(InferenceResult(predictions=r, ...))
            except:
                break
        return results
        """
        ...
```

#### WS2.4: Tool Implementations

```python
# inference_agent/tools/inference_tools.py

class RunInferenceTool:
    """The primary inference tool — runs a model or workflow on an image."""

    name = "run_inference"
    description = "Run inference on a single image..."
    input_schema = { ... }
    category = "inference"

    def __init__(self, backend: InferenceBackend, summarizer: ResultSummarizer):
        self._backend = backend
        self._summarizer = summarizer

    async def execute(self, image: str, model_id: str | None = None,
                      workspace_name: str | None = None,
                      workflow_id: str | None = None,
                      workflow_spec: dict | None = None,
                      parameters: dict | None = None,
                      confidence: float | None = None) -> Any:
        """
        1. Determine which inference path to use:
           - If model_id → direct model inference
           - If workspace_name + workflow_id → user's pre-built workflow
           - If workflow_spec → agent-composed workflow
        2. Call self._backend.run_single(...)
        3. Summarize results via self._summarizer
        4. Return [text_summary, optional_annotated_image]
        """
        ...

class StartPipelineTool:
    """Start a continuous video processing pipeline."""

    async def execute(self, video_reference, pipeline_name, ...) -> str:
        """
        1. If workflow_spec provided: validate using block discovery
        2. If model_id provided but no workflow: compose a simple
           detect → output workflow automatically
        3. Call self._backend.start_pipeline(...)
        4. Register pipeline in the pipeline registry
        5. Return "Pipeline '{name}' started (id: {id}). Processing
           {video_reference} at {max_fps} FPS."
        """
        ...
```

#### WS2.5: Result Summarizer

```python
# inference_agent/tools/result_summarizer.py

class ResultSummarizer:
    """Converts raw inference output to human-readable summaries.

    This is CRITICAL for controlling token usage. Raw workflow output
    can be 10K+ tokens per frame. The summarizer reduces it to ~100 tokens.
    """

    def summarize(self, result: InferenceResult,
                  include_frame: bool = False) -> str | list:
        """
        Input: raw workflow output dict with nested detections
        Output: "Frame #1234: 3 detections - 2 person (0.92, 0.87), 1 car (0.76)"

        If include_frame=True, returns content blocks:
        [
            {"type": "image", "source": {"type": "base64", ...}},
            {"type": "text", "text": "Frame #1234: 3 detections..."},
        ]
        """
        ...
```

### WS3: Memory + Workspace

#### WS3.1: Workspace Manager

```python
# inference_agent/memory/workspace.py

class WorkspaceManager:
    """Manages the agent's workspace directory."""

    def __init__(self, workspace_path: str = "~/.vision_agent"):
        self._path = Path(workspace_path).expanduser()

    def initialize(self):
        """Create workspace directory with default files if not exists.

        Creates:
        - AGENTS.md with default instructions
        - USER.md with placeholder
        - CAMERAS.md with empty table
        - HEARTBEAT.md with default checklist
        - memory/ directory
        - sessions/ directory
        - skills/ directory
        """
        ...

    def load_workspace_files(self) -> dict[str, str]:
        """Load workspace context files for system prompt injection.

        Returns dict of filename → content for all .md files in workspace root.
        Skips files larger than 4K tokens to prevent prompt bloat.
        """
        ...

    def list_available_skills(self) -> list[dict]:
        """Discover skills in skills/ directory.

        Returns [{name, description, path}] for each SKILL.md found.
        """
        ...

    def load_skill(self, skill_name: str) -> str | None:
        """Load a specific skill's SKILL.md content."""
        ...
```

---

## Appendix A: Key Existing Code to Reuse

| Need | Existing Code | How |
|------|--------------|-----|
| HTTP client (full API) | `inference_sdk/http/client.py` → `InferenceHTTPClient` | Wrap in HTTPBackend |
| WebRTC streaming | `inference_sdk/webrtc/client.py` → `WebRTCClient` | Wrap in WebRTCBackend |
| Hosted pipeline mgmt | `InferenceHTTPClient.start_inference_pipeline_with_workflow()` | Call via HTTPBackend |
| Block discovery (HTTP) | `POST /workflows/blocks/describe` endpoint | Call from HTTPBackend |
| Block discovery (direct) | `inference.core.workflows.execution_engine.introspection.blocks_loader` | Import directly |
| Workflow validation | `POST /workflows/validate` or execution engine | Call via backend |
| InferencePipeline | `inference.core.interfaces.stream.inference_pipeline` | Use in DirectBackend |
| Image utilities | `inference.core.utils.image_utils` | Import directly |
| Claude API patterns | `inference/core/workflows/core_steps/models/foundation/anthropic_claude/v3.py` | Reference impl |
| Model manager | `inference.core.managers.base.ModelManager` | Use in DirectBackend |
| WebRTC sources | `inference_sdk.webrtc.sources` → RTSPSource, WebcamSource, etc. | Use in WebRTCBackend |

## Appendix B: Estimated Token Budget (Revised)

| Component | Tokens | Notes |
|-----------|--------|-------|
| Core identity prompt | ~1,500 | Rules, format, workflow docs |
| AGENTS.md | ~500 | User-editable instructions |
| CAMERAS.md | ~200 | Camera registry table |
| USER.md | ~200 | User preferences |
| Block summary (130+ blocks) | ~2,500 | One-line per block |
| Workflow examples (3) | ~1,500 | Few-shot patterns |
| **Total cached system** | **~6,400** | |
| Tool definitions (15 tools) | ~2,500 | Also cached |
| **Total cached prefix** | **~8,900** | $0.003/call at cache read |
| Active pipelines (dynamic) | ~300 | Per-turn |
| Available skills (dynamic) | ~200 | Per-turn |
| **Total per-call overhead** | **~9,400** | |

**Cost per interaction (Sonnet 4.5 with caching):**
- Cached input: 8,900 tokens × $0.30/MTok = $0.003
- Dynamic input: 500 tokens × $3/MTok = $0.002
- User message: ~200 tokens × $3/MTok = $0.001
- Output: ~500 tokens × $15/MTok = $0.008
- **Total: ~$0.014 per interaction**

**Heartbeat cost (every 15 min, 24/7):**
- 96 calls/day × $0.014 = **~$1.35/day** on Sonnet 4.5

## Appendix C: Configuration (Revised)

```yaml
# ~/.vision_agent/config.yaml

llm:
  provider: anthropic
  model: claude-sonnet-4-5           # Primary model
  api_key: ${ANTHROPIC_API_KEY}      # From env var
  max_tokens: 4096
  thinking:
    enabled: false                   # Enable for complex tasks
    budget_tokens: 4096
  cost_limit_daily: 10.00            # USD, hard stop

inference:
  backend: http                      # "http", "direct", or "webrtc"
  server_url: http://localhost:9001
  api_key: ${ROBOFLOW_API_KEY}       # From env var

workspace:
  path: ~/.vision_agent

heartbeat:
  enabled: true
  interval_minutes: 15

monitoring:
  escalation_confidence: 0.5         # Min confidence to escalate to LLM
  max_results_per_consume: 10
  result_image_max_dimension: 640    # Resize frames before sending to LLM

interface:
  type: cli                          # "cli", "slack", "rest"
  # slack:
  #   bot_token: ${SLACK_BOT_TOKEN}
  #   app_token: ${SLACK_APP_TOKEN}
  #   alert_channel: "#security-alerts"
```
