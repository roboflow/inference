# Vision Agent: Implementation Plan

## Table of Contents
1. [Architecture Summary](#1-architecture-summary)
2. [Interface Contracts](#2-interface-contracts)
3. [Work Streams (Parallelizable)](#3-work-streams)
4. [MVP: Minimum Testable Assumption](#4-mvp)
5. [Risk Register & Derisking Strategy](#5-risk-register)
6. [Dependency Graph & Sequencing](#6-dependency-graph)
7. [Detailed Implementation: Each Work Stream](#7-detailed-implementation)

---

## 1. Architecture Summary

The Vision Agent is a Python process that runs an **agentic loop** powered by the
Claude API (tool-use). It receives natural language instructions, reasons about what
Inference capabilities are needed, and orchestrates them autonomously — either by
making direct SDK calls or by dynamically composing Workflow specifications and
deploying them on InferencePipelines.

```
┌─────────────────────────────────────────────────────────────────────┐
│                          AGENT PROCESS                              │
│                                                                     │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────────┐ │
│  │  Interface   │◄──►│  Agent Core  │◄──►│  Tool Executor         │ │
│  │  (CLI/Slack/ │    │  (agentic    │    │  (dispatches to        │ │
│  │   REST/...)  │    │   loop)      │    │   capability modules)  │ │
│  └─────────────┘    └──────┬───────┘    └───────────┬────────────┘ │
│                            │                        │              │
│                     ┌──────┴───────┐    ┌───────────┴────────────┐ │
│                     │  LLM Client  │    │   Capability Registry  │ │
│                     │  (Claude API │    │   (auto-generated      │ │
│                     │   w/ tools)  │    │    tool definitions)   │ │
│                     └──────────────┘    └───────────┬────────────┘ │
│                                                     │              │
│  ┌──────────────────────────────────────────────────┴────────────┐ │
│  │                    Capability Modules                          │ │
│  │  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌────────────────┐  │ │
│  │  │ Workflow  │ │  Stream   │ │  Direct  │ │    Memory      │  │ │
│  │  │ Composer  │ │  Manager  │ │  Infer   │ │   (files +     │  │ │
│  │  │          │ │           │ │          │ │    search)     │  │ │
│  │  └──────────┘ └───────────┘ └──────────┘ └────────────────┘  │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                              │                                     │
├──────────────────────────────┼─────────────────────────────────────┤
│                    INFERENCE (existing)                             │
│  ┌──────────────┐  ┌───────────────────┐  ┌────────────────────┐  │
│  │  Models (38+) │  │  Workflow Engine   │  │  InferencePipeline │  │
│  │  via SDK or   │  │  (103+ blocks)    │  │  (video streams)   │  │
│  │  HTTP client  │  │                   │  │                    │  │
│  └──────────────┘  └───────────────────┘  └────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
```

### Key Architectural Decisions

1. **Build our own agent loop on the raw Anthropic SDK** (`pip install anthropic`),
   not the Claude Agent SDK. Reason: we need fine-grained control over image content
   in tool results, custom tool execution with video hardware, and precise control
   over loop timing for proactive monitoring. The Claude Agent SDK is optimized for
   code-editing agents, not vision agents.

2. **Workflow-first for streaming** — The agent composes Workflow specification JSON
   and deploys it on InferencePipeline. The LLM is never in the frame-processing hot
   path. The existing workflow engine handles 30+ FPS on GPU; Claude is only consulted
   for setup, event interpretation, and user interaction.

3. **Auto-generated tool schemas from block manifests** — Rather than manually
   maintaining tool definitions, we generate them from the existing block manifest
   system. This keeps the agent's knowledge in sync with Inference's capabilities as
   blocks are added/updated.

4. **File-based memory with hybrid search** (OpenClaw pattern) — Markdown + JSON
   files for transparency, with BM25 + vector hybrid search for retrieval. Users can
   inspect, edit, and version-control the agent's memory.

5. **Interface-first design** — All components communicate through well-defined
   Python protocols/ABCs. This enables parallel development and easy testing via mocks.

---

## 2. Interface Contracts

These interfaces are the **critical path** — they must be designed and agreed upon
before parallel work streams begin. Each interface is a Python Protocol or ABC.

### 2.1 `AgentCore` — The Agentic Loop

```python
# inference_agent/core/protocols.py

from typing import Protocol, AsyncIterator, Any
from dataclasses import dataclass

@dataclass
class AgentMessage:
    """A message in the agent's conversation."""
    role: str  # "user", "assistant", "system"
    content: Any  # str, list of content blocks, or tool results
    metadata: dict  # timestamps, source info, etc.

@dataclass
class AgentEvent:
    """Events emitted by the agent during processing."""
    type: str  # "thinking", "tool_call", "tool_result", "response",
               # "stream_started", "alert", "error"
    data: Any
    timestamp: float

class AgentCore(Protocol):
    """The main agent loop. Receives messages, reasons, acts, responds."""

    async def process_message(self, message: str,
                              attachments: list[bytes] | None = None
                              ) -> AsyncIterator[AgentEvent]:
        """Process a user message. Yields events as the agent works.

        The caller consumes events to:
        - Display thinking/progress to the user
        - Show tool calls being made
        - Deliver the final response
        - React to alerts from proactive monitoring

        Args:
            message: User's natural language instruction
            attachments: Optional image bytes (e.g., user sends a photo)

        Yields:
            AgentEvent objects as the agent reasons and acts
        """
        ...

    async def start_monitoring(self) -> None:
        """Start the proactive monitoring loop (heartbeat).

        Periodically checks active streams, runs scheduled analyses,
        and emits alert events when relevant things are detected.
        """
        ...

    async def stop_monitoring(self) -> None:
        """Stop the proactive monitoring loop."""
        ...

    @property
    def active_streams(self) -> dict[str, "StreamInfo"]:
        """Currently active video stream pipelines."""
        ...
```

### 2.2 `LLMClient` — Claude API Wrapper

```python
# inference_agent/llm/protocols.py

from typing import Protocol
from dataclasses import dataclass

@dataclass
class ToolDefinition:
    """A tool the LLM can call."""
    name: str
    description: str
    input_schema: dict  # JSON Schema
    category: str | None = None  # For grouping in system prompt

@dataclass
class ToolCall:
    """A tool call from the LLM."""
    id: str
    name: str
    arguments: dict

@dataclass
class ToolResult:
    """Result of executing a tool, sent back to the LLM."""
    tool_call_id: str
    content: Any  # str, or list of content blocks (text + images)
    is_error: bool = False

@dataclass
class LLMResponse:
    """Response from the LLM."""
    text: str | None  # Final text response (None if tool_use)
    tool_calls: list[ToolCall]  # Tool calls to execute
    stop_reason: str  # "end_turn", "tool_use", "max_tokens"
    thinking: str | None  # Extended thinking content
    usage: dict  # Token usage stats

class LLMClient(Protocol):
    """Wrapper around Claude API with tool-use support."""

    async def chat(
        self,
        messages: list[dict],
        tools: list[ToolDefinition],
        system_prompt: str,
        max_tokens: int = 4096,
        thinking: bool = False,
        thinking_budget: int | None = None,
    ) -> LLMResponse:
        """Send messages to Claude and get a response.

        Handles:
        - Message formatting for the Anthropic API
        - Image content blocks in tool results
        - Extended thinking when enabled
        - Prompt caching for system prompt + tool definitions
        - Streaming (internal, aggregates to LLMResponse)
        - Rate limit retry with exponential backoff
        """
        ...

    async def chat_stream(
        self,
        messages: list[dict],
        tools: list[ToolDefinition],
        system_prompt: str,
        **kwargs,
    ) -> AsyncIterator[AgentEvent]:
        """Streaming variant — yields events as they arrive."""
        ...
```

### 2.3 `ToolExecutor` — Tool Dispatch

```python
# inference_agent/tools/protocols.py

from typing import Protocol, Callable, Awaitable, Any

class Tool(Protocol):
    """A single tool the agent can use."""
    name: str
    description: str
    input_schema: dict  # JSON Schema for parameters
    category: str

    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments. Returns result for LLM."""
        ...

class ToolRegistry(Protocol):
    """Registry of all available tools."""

    @property
    def tools(self) -> list[Tool]:
        """All registered tools."""
        ...

    @property
    def definitions(self) -> list[ToolDefinition]:
        """Tool definitions formatted for the LLM."""
        ...

    def get_tool(self, name: str) -> Tool | None:
        """Look up a tool by name."""
        ...

    async def execute(self, name: str, arguments: dict) -> Any:
        """Execute a tool by name. Returns result content for LLM.

        Result can be:
        - str: Simple text result
        - list: Content blocks (text + images) for multimodal results
        - dict: Structured data (will be JSON-serialized)

        Raises ToolExecutionError on failure.
        """
        ...
```

### 2.4 `CapabilityModule` — Pluggable Capability Groups

```python
# inference_agent/capabilities/protocols.py

from typing import Protocol

class CapabilityModule(Protocol):
    """A group of related tools. Each module provides tools to the registry."""

    @property
    def name(self) -> str:
        """Module name (e.g., 'stream_manager', 'direct_inference')."""
        ...

    def get_tools(self) -> list[Tool]:
        """Return all tools provided by this module."""
        ...

    async def initialize(self) -> None:
        """Called once at startup. Connect to inference server, etc."""
        ...

    async def shutdown(self) -> None:
        """Called on agent shutdown. Clean up resources."""
        ...
```

### 2.5 `Memory` — Persistent Scene Memory

```python
# inference_agent/memory/protocols.py

from typing import Protocol
from dataclasses import dataclass

@dataclass
class MemoryEntry:
    """A single memory entry."""
    content: str
    source: str  # "observation", "user_preference", "configuration", "knowledge"
    timestamp: float
    metadata: dict  # camera_id, stream_id, tags, etc.

@dataclass
class SearchResult:
    """A memory search result with relevance score."""
    entry: MemoryEntry
    score: float  # 0.0 - 1.0
    source_file: str

class Memory(Protocol):
    """Persistent memory with hybrid search."""

    async def store(self, content: str, source: str,
                    metadata: dict | None = None) -> None:
        """Store a new memory entry.

        Appends to the appropriate file based on source:
        - "observation" -> observations/YYYY-MM-DD.md
        - "user_preference" -> preferences.md
        - "configuration" -> config.json
        - "knowledge" -> knowledge.md
        """
        ...

    async def search(self, query: str, max_results: int = 10,
                     source_filter: str | None = None
                     ) -> list[SearchResult]:
        """Hybrid BM25 + vector search over all memory files.

        Args:
            query: Natural language search query
            max_results: Maximum results to return
            source_filter: Optional filter by source type

        Returns:
            Ranked list of matching memory entries
        """
        ...

    async def get_recent_observations(self, camera_id: str | None = None,
                                       hours: int = 1
                                       ) -> list[MemoryEntry]:
        """Get recent observation entries, optionally filtered by camera."""
        ...

    async def get_active_config(self) -> dict:
        """Get the current agent configuration (cameras, tasks, etc.)."""
        ...

    async def save_active_config(self, config: dict) -> None:
        """Persist active configuration for restart recovery."""
        ...
```

### 2.6 `UserInterface` — Channel Abstraction

```python
# inference_agent/interfaces/protocols.py

from typing import Protocol, AsyncIterator

@dataclass
class IncomingMessage:
    """Message from the user."""
    text: str
    attachments: list[bytes]  # Images
    channel_id: str  # Which interface it came from
    user_id: str | None
    metadata: dict

@dataclass
class OutgoingMessage:
    """Message to send to the user."""
    text: str
    images: list[bytes] | None = None  # Annotated frames
    channel_id: str | None = None  # None = broadcast to all

class UserInterface(Protocol):
    """Abstraction for user-facing communication channels."""

    async def start(self) -> None:
        """Start listening for user messages."""
        ...

    async def stop(self) -> None:
        """Stop the interface."""
        ...

    async def receive(self) -> AsyncIterator[IncomingMessage]:
        """Yield incoming user messages."""
        ...

    async def send(self, message: OutgoingMessage) -> None:
        """Send a message to the user."""
        ...

    async def send_alert(self, message: OutgoingMessage) -> None:
        """Send a proactive alert. May use different formatting/priority."""
        ...
```

### 2.7 `WorkflowComposer` — Dynamic Workflow Generation

```python
# inference_agent/capabilities/workflow_composer/protocols.py

from typing import Protocol
from dataclasses import dataclass

@dataclass
class BlockInfo:
    """Information about a workflow block, derived from manifest."""
    type_identifier: str  # e.g., "roboflow_core/roboflow_object_detection_model@v2"
    name: str  # Human-friendly name
    short_description: str
    category: str  # "model", "analytics", "visualization", "sink", etc.
    input_schema: dict  # JSON Schema of parameters
    outputs: list[dict]  # [{name, kind}]
    long_description: str | None = None

class WorkflowComposer(Protocol):
    """Knows about all workflow blocks and can compose workflow specs."""

    @property
    def available_blocks(self) -> list[BlockInfo]:
        """All available workflow blocks."""
        ...

    def get_blocks_summary(self) -> str:
        """Concise summary of all blocks for LLM system prompt."""
        ...

    def validate_workflow(self, spec: dict) -> tuple[bool, str | None]:
        """Validate a workflow specification. Returns (valid, error_msg)."""
        ...

    def get_block_info(self, type_identifier: str) -> BlockInfo | None:
        """Get detailed info about a specific block."""
        ...

    def get_blocks_by_category(self, category: str) -> list[BlockInfo]:
        """Get all blocks in a category."""
        ...
```

---

## 3. Work Streams (Parallelizable)

After the interface contracts are agreed upon (1-2 days of design), the following
work streams can proceed **in parallel**:

```
                    ┌─────────────────────┐
                    │  WS0: Interface     │
                    │  Contract Design    │
                    │  (1-2 days)         │
                    └────────┬────────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                  │
           ▼                 ▼                  ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
│ WS1: Agent Core  │ │ WS2: Capabil-│ │ WS3: Memory      │
│ + LLM Client     │ │ ity Modules  │ │ System           │
│ (5-7 days)       │ │ (5-7 days)   │ │ (3-5 days)       │
└────────┬─────────┘ └──────┬───────┘ └────────┬─────────┘
         │                  │                   │
         │      ┌───────────┼───────────┐       │
         │      │           │           │       │
         ▼      ▼           ▼           ▼       ▼
┌──────────────────────────────────────────────────────┐
│                WS4: Integration + MVP                 │
│                (3-5 days)                             │
└──────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│ WS5: Streaming   │
│ + Proactive      │
│ Monitoring       │
│ (5-7 days)       │
└──────────────────┘
```

---

## 4. MVP: Minimum Testable Assumption

### What We're Testing

The **core risk** is: Can Claude, given tool definitions auto-generated from
Inference's workflow block manifests, reliably compose workflow specifications and
orchestrate inference pipelines from natural language instructions?

This breaks down into three testable assumptions:

| # | Assumption | How We Test It | Success Criteria |
|---|-----------|----------------|-----------------|
| 1 | Claude can understand auto-generated tool schemas derived from workflow block manifests and use them correctly | Give Claude block descriptions + ask it to compose a workflow JSON | Valid workflow specs that pass `validate_workflow` >80% of the time |
| 2 | Claude can orchestrate multi-step vision tasks (detect → track → filter → alert) through tool calls | End-to-end: user says "watch for X" → agent composes workflow → pipeline runs → results consumed | Working pipeline producing correct detections from a test video |
| 3 | The agent loop is responsive enough for interactive use (plan + deploy in <30s) | Measure time from user instruction to pipeline running | <30s for common monitoring tasks, <60s for complex multi-step setups |

### MVP Scope

**In scope:**
- CLI REPL interface (simplest possible UI)
- Agent loop with Claude API tool-use
- 5 core tools:
  1. `list_available_models` — list Roboflow models
  2. `list_workflow_blocks` — list available blocks with descriptions
  3. `get_block_details` — get full schema for a specific block
  4. `deploy_workflow_pipeline` — compose + validate + start InferencePipeline
  5. `get_pipeline_results` — consume latest results from a running pipeline
- Direct SDK mode (agent runs on same machine as inference)
- One test scenario: "Watch this video file and detect all people"

**Out of scope for MVP:**
- Memory system (hardcode in system prompt)
- Proactive monitoring / heartbeat
- Multi-channel interfaces
- HTTP client mode
- RTSP/live streams (use video files)
- Notification sinks
- Persistence across restarts

### MVP File Structure

```
inference_agent/
├── __init__.py
├── __main__.py                    # Entry point: `python -m inference_agent`
├── core/
│   ├── __init__.py
│   ├── agent.py                   # AgentCore implementation
│   ├── protocols.py               # Interface definitions (Section 2)
│   └── config.py                  # AgentConfig dataclass
├── llm/
│   ├── __init__.py
│   ├── claude_client.py           # LLMClient implementation (Anthropic SDK)
│   └── protocols.py
├── tools/
│   ├── __init__.py
│   ├── registry.py                # ToolRegistry implementation
│   └── protocols.py
├── capabilities/
│   ├── __init__.py
│   ├── workflow_composer/
│   │   ├── __init__.py
│   │   ├── composer.py            # WorkflowComposer implementation
│   │   ├── block_discovery.py     # Auto-discover blocks from manifests
│   │   └── schema_generator.py    # Generate LLM tool schemas from blocks
│   ├── stream_manager/
│   │   ├── __init__.py
│   │   └── manager.py             # Pipeline lifecycle management
│   └── direct_inference/
│       ├── __init__.py
│       └── inference.py           # Single-image inference tools
├── memory/                        # Stub for MVP, full impl later
│   ├── __init__.py
│   └── protocols.py
└── interfaces/
    ├── __init__.py
    └── cli.py                     # CLI REPL interface
```

---

## 5. Risk Register & Derisking Strategy

### Risk 1: Tool Schema Explosion (HIGH)

**Risk:** Inference has 130+ workflow blocks. If we generate a tool definition for
each block parameter, the system prompt will be enormous (50K+ tokens), expensive,
and may confuse Claude.

**Derisking strategy:**
- **Two-tier tool design**: High-level tools (`deploy_workflow_pipeline`) that accept
  a workflow spec, plus discovery tools (`list_workflow_blocks`, `get_block_details`)
  that let Claude explore what's available on demand.
- **NOT** one tool per block. Instead, Claude uses discovery tools to learn about
  blocks, then composes a workflow JSON that references them.
- The system prompt contains a **summary** of capabilities (block names + one-line
  descriptions), not full schemas.
- Full block schemas are returned only when Claude calls `get_block_details`.

**Test:** Measure system prompt token count with summary approach vs. full schema
approach. Target: system prompt < 8K tokens.

### Risk 2: Workflow Composition Accuracy (HIGH)

**Risk:** Claude may generate invalid workflow JSON — wrong block type identifiers,
incorrect selector syntax (`$steps.x.y`), incompatible kind connections, etc.

**Derisking strategy:**
- **Validate before deploy**: The `deploy_workflow_pipeline` tool validates the
  workflow spec using the existing `validate_workflow` endpoint before starting the
  pipeline. If validation fails, the error is returned to Claude as a tool result so
  it can self-correct.
- **Few-shot examples**: Include 3-5 working workflow examples in the system prompt
  (common patterns: detect+track, detect+filter+alert, detect+count).
- **Template library**: Pre-built workflow templates for common patterns that Claude
  can customize rather than building from scratch.
- **Iterative refinement**: The agent loop naturally supports Claude trying, failing,
  reading the error, and fixing — just like a human developer.

**Test:** Create a benchmark of 20 natural language instructions and measure what
percentage produce valid workflows on first attempt vs. after self-correction.

### Risk 3: Pipeline Result Interpretation (MEDIUM)

**Risk:** Workflow pipeline outputs are raw dicts with nested detection data
(supervision Detections serialized). Claude needs to understand these to decide
whether to alert the user.

**Derisking strategy:**
- **Result summarizer**: A utility function that converts raw pipeline output into a
  human-readable summary before sending to Claude:
  ```
  "Frame 1234 from source 0: 3 detections (2 person [0.92, 0.87], 1 car [0.76])"
  ```
- **Selective image forwarding**: Only send frame images to Claude when the agent
  needs visual confirmation (anomaly detected, ambiguous result). Most of the time,
  the structured detection data is sufficient.
- **Token budget management**: Limit result verbosity to prevent context overflow
  during long monitoring sessions.

### Risk 4: Proactive Monitoring Cost (MEDIUM)

**Risk:** An always-on agent calling Claude API for every pipeline result would be
prohibitively expensive.

**Derisking strategy:**
- **Event-driven, not poll-driven**: The pipeline runs autonomously. Claude is only
  consulted when:
  1. A relevant event is detected (e.g., new objects appearing, threshold exceeded)
  2. The user asks a question
  3. A scheduled heartbeat fires (configurable: every 15-60 min)
- **Local filtering**: Python code in the result consumer decides whether an event is
  "interesting" enough to escalate to Claude. Simple heuristics (new object count > 0,
  confidence > threshold) avoid unnecessary LLM calls.
- **Model tiering**: Use Haiku 4.5 ($1/$5 per MTok) for routine event classification,
  Sonnet 4.5 ($3/$15) for complex analysis, Opus only when explicitly requested.
- **Prompt caching**: Cache the system prompt + tool definitions (saves 90% on
  repeated input tokens).

### Risk 5: Multi-Camera Complexity (LOW for MVP)

**Deferred**: MVP uses single video files. Multi-camera support is a Phase 2 concern.

---

## 6. Dependency Graph & Sequencing

### Phase 0: Foundation (Week 1, days 1-2)

**Goal:** Agree on interfaces and set up the project skeleton.

| Task | Dependencies | Owner | Deliverable |
|------|-------------|-------|-------------|
| P0.1 Define all Protocol interfaces | None | Architect | `protocols.py` files in each module |
| P0.2 Set up package structure | None | Anyone | `inference_agent/` with `pyproject.toml` |
| P0.3 Write interface tests (contract tests) | P0.1 | Anyone | Tests that validate any implementation of each Protocol |

### Phase 1: Parallel Work Streams (Week 1-2, days 3-10)

All three streams can proceed simultaneously once interfaces are defined.

**WS1: Agent Core + LLM Client**
| Task | Dependencies | Deliverable |
|------|-------------|-------------|
| WS1.1 Implement `ClaudeClient` (LLMClient protocol) | P0.1 | `llm/claude_client.py` |
| WS1.2 Implement `AgentLoop` (AgentCore protocol) | P0.1 | `core/agent.py` |
| WS1.3 System prompt engineering | WS1.1 | System prompt template with block summaries + examples |
| WS1.4 Implement CLI REPL | P0.1 | `interfaces/cli.py` |
| WS1.5 Unit tests with mock tools | WS1.2 | Test agent loop with fake tool implementations |

**WS2: Capability Modules**
| Task | Dependencies | Deliverable |
|------|-------------|-------------|
| WS2.1 Block discovery from manifests | P0.1 | `capabilities/workflow_composer/block_discovery.py` |
| WS2.2 LLM tool schema generation | WS2.1 | `capabilities/workflow_composer/schema_generator.py` |
| WS2.3 WorkflowComposer implementation | WS2.1, WS2.2 | `capabilities/workflow_composer/composer.py` |
| WS2.4 StreamManager capability module | P0.1 | `capabilities/stream_manager/manager.py` |
| WS2.5 DirectInference capability module | P0.1 | `capabilities/direct_inference/inference.py` |
| WS2.6 ToolRegistry implementation | P0.1, WS2.3-5 | `tools/registry.py` |

**WS3: Memory System**
| Task | Dependencies | Deliverable |
|------|-------------|-------------|
| WS3.1 File-based memory store | P0.1 | `memory/store.py` |
| WS3.2 BM25 search (SQLite FTS5) | WS3.1 | `memory/search.py` |
| WS3.3 Vector search (CLIP or sentence-transformers) | WS3.1 | `memory/search.py` |
| WS3.4 Hybrid search with union ranking | WS3.2, WS3.3 | `memory/search.py` |
| WS3.5 Memory tools (remember, recall, get_observations) | WS3.1-4 | `memory/tools.py` |

### Phase 2: Integration + MVP (Week 2-3, days 8-14)

| Task | Dependencies | Deliverable |
|------|-------------|-------------|
| I.1 Wire all modules together | WS1, WS2 | Working agent with real tools |
| I.2 End-to-end test: "detect people in video" | I.1 | Passing integration test |
| I.3 End-to-end test: "count cars" with line counter | I.1 | Passing integration test |
| I.4 End-to-end test: self-correction on invalid workflow | I.1 | Agent recovers from validation errors |
| I.5 Benchmark: workflow composition accuracy | I.1 | Accuracy metrics on 20-task benchmark |
| I.6 Integrate memory module | WS3, I.1 | Agent can remember and recall |
| I.7 Performance profiling | I.1-5 | Response time measurements |

### Phase 3: Streaming + Proactive (Week 3-4)

| Task | Dependencies | Deliverable |
|------|-------------|-------------|
| S.1 Result consumer with event detection | I.2 | Background thread consuming pipeline results |
| S.2 Event → LLM escalation logic | S.1 | Decides when to consult Claude about an event |
| S.3 Heartbeat / periodic analysis loop | S.1 | Configurable timer-based scene analysis |
| S.4 RTSP stream support | S.1 | Live camera streams working |
| S.5 Multi-camera management | S.4 | Multiple streams with per-camera context |
| S.6 Alert routing | S.2 | Route alerts to correct user interface |

---

## 7. Detailed Implementation: Each Work Stream

### WS1: Agent Core + LLM Client

#### WS1.1: ClaudeClient

```python
# inference_agent/llm/claude_client.py

import anthropic
from .protocols import LLMClient, LLMResponse, ToolDefinition, ToolCall

class ClaudeClient:
    """LLMClient implementation using the Anthropic Python SDK."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5",
                 max_retries: int = 3):
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_retries = max_retries

    async def chat(self, messages, tools, system_prompt,
                   max_tokens=4096, thinking=False,
                   thinking_budget=None) -> LLMResponse:
        """Core implementation notes:

        1. Convert ToolDefinition list to Anthropic API format
        2. Use prompt caching on system prompt + tool definitions:
           system=[{"type": "text", "text": prompt,
                    "cache_control": {"type": "ephemeral"}}]
        3. Handle tool_use stop_reason by extracting ToolCall objects
        4. Handle extended thinking: pass thinking blocks back in
           subsequent messages (CRITICAL for multi-turn tool use)
        5. Retry on 429/529 with exponential backoff using
           response headers (retry-after, anthropic-ratelimit-*)
        6. Track token usage for cost monitoring
        """
        ...
```

**Key implementation details:**
- Use `anthropic.AsyncAnthropic` for async support
- Enable prompt caching on system prompt (saves 90% on repeated calls)
- When `thinking=True` and tools are used, preserve thinking blocks in
  the message history (Claude requires this for coherent multi-turn reasoning)
- Parse `tool_use` content blocks into `ToolCall` dataclasses
- Parse images in tool results into proper content blocks:
  ```python
  {
      "type": "tool_result",
      "tool_use_id": call_id,
      "content": [
          {"type": "image", "source": {"type": "base64", ...}},
          {"type": "text", "text": "Frame 1234 from camera front_door"}
      ]
  }
  ```

#### WS1.2: AgentLoop

```python
# inference_agent/core/agent.py

class VisionAgent:
    """Main agent loop implementation."""

    def __init__(self, config: AgentConfig, llm: LLMClient,
                 tools: ToolRegistry, memory: Memory):
        self._config = config
        self._llm = llm
        self._tools = tools
        self._memory = memory
        self._messages: list[dict] = []
        self._monitoring_task: asyncio.Task | None = None

    async def process_message(self, message, attachments=None):
        """Agent loop pseudocode:

        1. Add user message to conversation history
        2. If attachments, include as image content blocks
        3. Build system prompt:
           a. Core identity and instructions
           b. Available capabilities summary (from WorkflowComposer)
           c. Active streams summary
           d. Recent memory context (last observations)
        4. Loop:
           a. Call LLM with messages + tools
           b. If stop_reason == "end_turn":
              - Yield response event
              - Break
           c. If stop_reason == "tool_use":
              - For each tool_call:
                - Yield tool_call event (for UI display)
                - Execute tool via ToolRegistry
                - Yield tool_result event
              - Add assistant message + tool results to history
              - Continue loop
           d. If stop_reason == "max_tokens":
              - Yield warning event
              - Break
        5. Post-processing:
           - If a pipeline was started, register it for monitoring
           - If user preference was expressed, store in memory
        """
        ...
```

**System prompt structure:**

```python
SYSTEM_PROMPT_TEMPLATE = """You are a Vision Agent — an autonomous computer vision
assistant powered by Roboflow Inference.

## Your Capabilities

You can analyze images and video streams using state-of-the-art computer vision
models. You orchestrate these capabilities by composing Workflow specifications
— JSON documents that define processing pipelines.

### Available Model Types
{model_summary}

### Available Analytics Blocks
{analytics_summary}

### Available Visualization Blocks
{viz_summary}

### Available Sink Blocks (for notifications/actions)
{sink_summary}

## How to Deploy a Monitoring Pipeline

To watch a video stream, you compose a workflow specification and deploy it:

1. Use `list_workflow_blocks` to see available blocks (filtered by category)
2. Use `get_block_details` to understand a specific block's parameters
3. Compose a workflow JSON with the required steps
4. Use `deploy_workflow_pipeline` to validate and start it
5. Use `get_pipeline_results` to check what's been detected

### Workflow Specification Format

```json
{{
  "version": "1.0",
  "inputs": [
    {{"type": "WorkflowImage", "name": "image"}},
    {{"type": "WorkflowVideoMetadata", "name": "video_metadata"}}
  ],
  "steps": [
    {{
      "type": "<block_type_identifier>",
      "name": "<unique_step_name>",
      "<param>": "$inputs.<name>" or "$steps.<step>.<output>" or <literal_value>
    }}
  ],
  "outputs": [
    {{"type": "JsonField", "name": "<output_name>",
      "selector": "$steps.<step>.<output>"}}
  ]
}}
```

### Example Workflows

#### Detect and track people
```json
{detect_track_example}
```

#### Count objects crossing a line
```json
{line_count_example}
```

## Currently Active Streams
{active_streams_summary}

## Recent Observations
{recent_observations}

## Rules
- Always validate workflow specs before deployment
- If a workflow fails validation, read the error and fix the issue
- When the user asks about what's happening, check pipeline results first
- For proactive monitoring, set up workflows that filter for events of interest
- When sending annotated frames to the user, resize to reasonable dimensions
- Be concise in responses. The user wants results, not explanations of your process.
"""
```

#### WS1.3: System Prompt Engineering

This is a **critical deliverable**. The system prompt must:
1. Be under 8K tokens (for cost and focus)
2. Include enough block information for Claude to compose workflows
3. Include 3-5 few-shot workflow examples covering common patterns
4. Be dynamically updated with active streams and recent observations

**Key few-shot examples to include:**
1. Simple detection: image → object_detection_model → output
2. Detection + tracking: image → detection → byte_tracker → output
3. Detection + filter + alert: image → detection → filter → continue_if → sink
4. Analytics: image → detection → tracker → line_counter → data_aggregator
5. Multi-model: image → [model_A, model_B] → consensus → output

#### WS1.4: CLI REPL

```python
# inference_agent/interfaces/cli.py

import asyncio
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

class CLIInterface:
    """Simple CLI REPL for interacting with the Vision Agent."""

    def __init__(self, agent: AgentCore):
        self._agent = agent
        self._session = PromptSession(
            history=FileHistory('.vision_agent_history')
        )

    async def run(self):
        """Main REPL loop:

        1. Print welcome message
        2. Loop:
           a. Prompt for user input
           b. Call agent.process_message()
           c. Consume events:
              - "thinking" → show spinner
              - "tool_call" → show "Calling: tool_name..."
              - "tool_result" → show brief summary
              - "response" → print response text
              - "alert" → print with alert formatting
              - "error" → print error
        3. Handle /commands:
           /status — show active streams
           /streams — list pipeline details
           /stop <id> — stop a pipeline
           /memory <query> — search memory
           /quit — exit
        """
        ...
```

### WS2: Capability Modules

#### WS2.1: Block Discovery

```python
# inference_agent/capabilities/workflow_composer/block_discovery.py

"""
Discovers all available workflow blocks and extracts their metadata.

Two modes:
1. Direct (embedded): Import from inference.core.workflows.core_steps.loader
2. HTTP (remote): Call POST /workflows/blocks/describe on inference server

For MVP, we use Direct mode.
"""

from inference.core.workflows.core_steps.loader import load_blocks, load_kinds
from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    describe_available_blocks,
)

def discover_blocks() -> list[BlockInfo]:
    """Discover all available blocks and convert to BlockInfo objects.

    Implementation:
    1. Call describe_available_blocks(dynamic_blocks=[])
    2. For each BlockDescription in result.blocks:
       a. Extract json_schema_extra from block_schema for metadata
          (name, short_description, long_description, block_type, version)
       b. Extract properties from block_schema["properties"],
          skipping "type" and "name" (fixed fields)
       c. For each property, classify as:
          - Selector (has "reference": true in json_schema_extra)
          - Static parameter (no reference)
       d. Extract outputs_manifest as [{name, kind}]
       e. Construct BlockInfo dataclass
    3. Return list of BlockInfo objects

    Key consideration: Filter out deprecated blocks and blocks that
    require enterprise features or specific hardware.
    """
    ...
```

#### WS2.2: Schema Generator

```python
# inference_agent/capabilities/workflow_composer/schema_generator.py

"""
Generates LLM tool schemas from block manifests.

CRITICAL DESIGN DECISION: We do NOT create one tool per block.
Instead, we create a few high-level tools and embed block knowledge
in the system prompt and discoverable via query tools.

Tools generated:
1. list_workflow_blocks(category?) → concise list of blocks
2. get_block_details(block_type) → full schema for one block
3. deploy_workflow_pipeline(spec, video_ref) → validate + start pipeline
4. validate_workflow(spec) → validate without deploying

The actual block schemas are returned as tool results when Claude
calls get_block_details, not embedded in the tool definitions.
"""

def generate_blocks_summary(blocks: list[BlockInfo]) -> str:
    """Generate a concise summary for the system prompt.

    Format:
    ## Models
    - roboflow_core/roboflow_object_detection_model@v2: Detect objects with bounding boxes (YOLO, RF-DETR)
    - roboflow_core/roboflow_instance_segmentation_model@v2: Segment objects with pixel masks
    - roboflow_core/yolo_world_model@v2: Zero-shot object detection from text prompts
    ...

    ## Analytics
    - roboflow_core/line_counter@v2: Count objects crossing a line
    - roboflow_core/time_in_zone@v3: Measure time objects spend in a zone
    ...

    Target: <3K tokens for the entire summary.
    """
    ...

def generate_block_detail_response(block: BlockInfo) -> str:
    """Generate a detailed description of one block for tool result.

    Format:
    Block: roboflow_core/line_counter@v2
    Category: analytics
    Description: Count objects crossing a line in a video stream.

    Required Parameters:
    - metadata: Selector(video_metadata) — Video metadata from $inputs.video_metadata
    - detections: Selector(object_detection_prediction | instance_segmentation_prediction)
      — Tracked detections, e.g., $steps.tracker.tracked_detections
    - line_segment: list | Selector(list_of_values) — Two points [[x1,y1],[x2,y2]]

    Optional Parameters:
    - triggering_anchor: str — Point on bbox crossing the line (default: "CENTER")
      Options: CENTER, TOP_CENTER, BOTTOM_CENTER, ...

    Outputs:
    - count_in: integer — Objects crossing line inward
    - count_out: integer — Objects crossing line outward

    Example step:
    {
      "type": "roboflow_core/line_counter@v2",
      "name": "my_counter",
      "metadata": "$inputs.video_metadata",
      "detections": "$steps.tracker.tracked_detections",
      "line_segment": [[0, 300], [640, 300]]
    }
    """
    ...
```

#### WS2.3: WorkflowComposer

```python
# inference_agent/capabilities/workflow_composer/composer.py

class WorkflowComposerModule:
    """CapabilityModule that provides workflow composition tools."""

    def get_tools(self) -> list[Tool]:
        return [
            ListWorkflowBlocksTool(self._blocks),
            GetBlockDetailsTool(self._blocks),
            ValidateWorkflowTool(),
            DeployWorkflowPipelineTool(self._stream_manager),
        ]

class DeployWorkflowPipelineTool:
    """Validates and deploys a workflow on an InferencePipeline.

    Parameters:
    - workflow_specification: dict — The workflow JSON
    - video_reference: str — Video file path, RTSP URL, or camera device
    - pipeline_name: str — Human-readable name for this pipeline
    - max_fps: float | None — Frame rate limit
    - workflows_parameters: dict | None — Additional workflow parameters

    Implementation:
    1. Validate the workflow spec using the execution engine
    2. If invalid, return the validation error (Claude will self-correct)
    3. Create an InMemoryBufferSink for result consumption
    4. Call InferencePipeline.init_with_workflow() with:
       - workflow_specification=spec
       - video_reference=video_ref
       - on_prediction=buffer_sink.on_prediction
       - max_fps=max_fps
       - serialize_results=True  (so results are JSON-serializable)
    5. Start pipeline on a background thread
    6. Register pipeline + sink in the StreamManager
    7. Return success message with pipeline_id
    """
    ...
```

#### WS2.4: StreamManager

```python
# inference_agent/capabilities/stream_manager/manager.py

class StreamManagerModule:
    """Manages InferencePipeline lifecycle and result consumption."""

    def __init__(self):
        self._pipelines: dict[str, ManagedPipeline] = {}

    def get_tools(self) -> list[Tool]:
        return [
            ListActiveStreamsTool(self),
            GetStreamStatusTool(self),
            GetPipelineResultsTool(self),
            StopPipelineTool(self),
            PausePipelineTool(self),
            ResumePipelineTool(self),
        ]

@dataclass
class ManagedPipeline:
    """A pipeline managed by the agent."""
    pipeline_id: str
    pipeline_name: str  # User-friendly name
    pipeline: InferencePipeline
    buffer_sink: InMemoryBufferSink
    video_reference: str
    workflow_spec: dict
    started_at: float
    status: str  # "running", "paused", "stopped", "error"

class GetPipelineResultsTool:
    """Consume latest results from a pipeline's buffer sink.

    Parameters:
    - pipeline_id: str
    - max_results: int (default 5)
    - include_frame: bool (default False) — whether to include a frame image

    Implementation:
    1. Get ManagedPipeline by id
    2. Consume up to max_results from buffer_sink
    3. For each (predictions, frames) pair:
       a. Summarize predictions into human-readable text:
          "3 objects detected: person (0.92), person (0.87), car (0.76)"
       b. If include_frame and frame is not None:
          - Encode frame as JPEG, base64
          - Return as content block list [image, text]
       c. Else return text summary only
    4. Return list of result summaries

    IMPORTANT: Summarize predictions to control token usage.
    Don't send raw detection dicts to Claude.
    """
    ...
```

#### WS2.5: DirectInference

```python
# inference_agent/capabilities/direct_inference/inference.py

class DirectInferenceModule:
    """Tools for single-image inference (not streaming)."""

    def get_tools(self) -> list[Tool]:
        return [
            DetectObjectsTool(self._model_manager),
            ClassifyImageTool(self._model_manager),
            RunOCRTool(self._model_manager),
            AskVLMTool(self._model_manager),
            DetectZeroShotTool(self._model_manager),
        ]

class DetectObjectsTool:
    """Run object detection on a single image.

    Parameters:
    - image: str — File path or URL
    - model_id: str — Roboflow model ID (e.g., "coco/9")
    - confidence: float (default 0.4)
    - classes: list[str] | None — Filter to specific classes

    Returns: Text summary of detections + optionally annotated image

    Implementation:
    Uses the existing inference model manager and workflow execution engine
    to run a simple single-step workflow on the image.
    """
    ...

class AskVLMTool:
    """Ask a vision-language model a question about an image.

    This is particularly useful for the agent to understand ambiguous
    scenes, verify detections, or answer user questions about what's
    in a frame.

    Parameters:
    - image: str — File path, URL, or base64
    - question: str — Question to ask about the image
    - model: str — "claude", "gpt4", "gemini", "florence2" (default "claude")

    Note: Uses the EXISTING Claude/GPT/Gemini workflow blocks in inference,
    not the agent's own LLM client. This keeps the VLM call in the Inference
    pipeline and avoids double-billing.
    """
    ...
```

### WS3: Memory System

#### WS3.1-3.4: Memory Implementation

```python
# inference_agent/memory/store.py

class FileMemoryStore:
    """File-based memory store with hybrid search.

    Directory layout:
    {workspace}/
    ├── config.json              # Agent config (cameras, tasks, preferences)
    ├── knowledge.md             # Long-term knowledge and patterns
    ├── preferences.md           # User preferences
    ├── observations/
    │   ├── 2024-03-15.md        # Daily observation logs
    │   └── 2024-03-16.md
    └── .index/
        └── memory.db            # SQLite with FTS5 + vec0

    Search approach (from OpenClaw):
    1. Chunk all .md files into ~400-token segments with 80-token overlap
    2. Index in SQLite:
       - FTS5 virtual table for BM25 keyword search
       - vec0 virtual table (sqlite-vec) for vector search
    3. Hybrid search: union of BM25 and vector results
       score = vector_weight * vector_score + text_weight * text_score
       Default weights: vector 0.7, text 0.3
    4. File watcher triggers re-indexing on changes (debounced)

    Embedding provider (for MVP): Use CLIP embeddings from inference
    (already available via the inference SDK). Falls back to BM25-only
    if CLIP is not loaded.

    For production: Support sentence-transformers or OpenAI embeddings.
    """
    ...

# inference_agent/memory/tools.py

class MemoryTools:
    """Tools that the agent can use to interact with memory."""

    def get_tools(self) -> list[Tool]:
        return [
            RememberTool(self._store),       # Store an observation/fact
            RecallTool(self._store),          # Search memory
            GetObservationsTool(self._store), # Recent observations by camera
            GetConfigTool(self._store),       # Read agent config
            SaveConfigTool(self._store),      # Write agent config
        ]

class RememberTool:
    """Store information in persistent memory.

    Parameters:
    - content: str — What to remember
    - category: str — "observation", "knowledge", "preference"
    - camera_id: str | None — Associated camera (for observations)

    The agent calls this to persist:
    - Significant events: "Person entered zone A at 14:23 without hard hat"
    - Learned patterns: "Deliveries usually arrive between 10am-2pm"
    - User preferences: "User prefers Slack alerts for urgent events"
    - Domain knowledge: "The red zone is the loading dock area"
    """
    ...
```

---

## Appendix A: Key Existing Code to Reuse

| What We Need | Existing Code | How to Reuse |
|-------------|--------------|-------------|
| Block manifests & discovery | `inference/core/workflows/core_steps/loader.py` → `load_blocks()` | Import directly, call to get all block classes |
| Block introspection | `inference/core/workflows/execution_engine/introspection/blocks_loader.py` → `describe_available_blocks()` | Import directly, generates full block metadata |
| Kind type system | `inference/core/workflows/execution_engine/entities/types.py` | Import Kind constants for type checking |
| Workflow validation | Execution engine compilation + `/workflows/validate` endpoint | Use execution engine directly or HTTP call |
| InferencePipeline | `inference/core/interfaces/stream/inference_pipeline.py` → `InferencePipeline.init_with_workflow()` | Direct instantiation |
| InMemoryBufferSink | `inference/core/interfaces/stream/sinks.py` → `InMemoryBufferSink` | Direct instantiation for result consumption |
| Claude API patterns | `inference/core/workflows/core_steps/models/foundation/anthropic_claude/v3.py` | Reference for image encoding, API key handling, response parsing |
| Image encoding | `inference/core/utils/image_utils.py` → `load_image()`, `encode_image_to_jpeg_bytes()` | Import directly |
| Model manager | `inference/core/managers/base.py` → `ModelManager` | Instantiate for direct inference mode |
| Workflow examples | `inference/development/workflows_examples/` | Reference for few-shot examples in system prompt |

## Appendix B: Estimated Token Budget

| Component | Tokens | Notes |
|-----------|--------|-------|
| System prompt (core) | ~2,000 | Identity, rules, format instructions |
| Block summary | ~2,500 | One-line per block, grouped by category |
| Workflow examples (3) | ~1,500 | Few-shot examples of common patterns |
| Active streams context | ~500 | Updated each turn |
| Recent observations | ~1,000 | Last few observations from memory |
| **Total system prompt** | **~7,500** | Well within budget |
| Tool definitions (5 MVP) | ~1,500 | Cached via prompt caching |
| **Total cached input** | **~9,000** | At $0.30/MTok cache read (Sonnet) = $0.003/call |

With prompt caching, the per-call cost for a routine interaction is approximately:
- Cached input (9K tokens): $0.003
- New input (user message, ~200 tokens): $0.0006
- Output (~500 tokens): $0.0075
- **Total: ~$0.01 per interaction**

For always-on monitoring with 6 LLM calls/hour:
- **~$1.50/day** on Sonnet 4.5 with prompt caching

## Appendix C: MVP Test Scenarios

| # | User Instruction | Expected Agent Behavior | Success Criteria |
|---|-----------------|------------------------|-----------------|
| 1 | "Detect all people in this video: test.mp4" | Compose workflow: detection → output. Deploy pipeline. Report detections. | Pipeline runs, detections returned |
| 2 | "Watch test.mp4 and count people crossing the middle of the frame" | Compose workflow: detection → tracker → line_counter. Deploy. | Line counts accumulate correctly |
| 3 | "What objects can you detect?" | Call list_workflow_blocks(category="model"). Report available models. | Accurate list of model blocks |
| 4 | "Use the YOLO-World model to find all 'coffee cups' in image.jpg" | Call direct inference with YOLO-World, class prompt "coffee cup". | Detections with correct class |
| 5 | "Stop all pipelines" | Call stop on all active pipelines. Confirm. | All pipelines terminated |
| 6 | "Set up PPE detection — alert when someone doesn't have a hard hat" | Compose workflow: detection(hard-hat model) → filter(no-hardhat class) → output. | Pipeline detects PPE violations |
| 7 | "What's in this image?" [with attachment] | Use AskVLM tool. Return description. | Accurate image description |
| 8 | Invalid workflow (agent makes a mistake) | Validation fails. Agent reads error. Fixes workflow. Redeploys. | Self-correction within 2 attempts |

## Appendix D: Configuration

```python
# inference_agent/core/config.py

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class AgentConfig:
    """Configuration for the Vision Agent."""

    # LLM
    anthropic_api_key: str
    model: str = "claude-sonnet-4-5"
    thinking_model: str = "claude-sonnet-4-5"  # For complex planning
    max_tokens: int = 4096
    enable_thinking: bool = False
    thinking_budget: int = 8000

    # Inference
    inference_mode: str = "direct"  # "direct" or "http"
    inference_server_url: str = "http://localhost:9001"
    roboflow_api_key: str | None = None

    # Memory
    workspace_path: str = "~/.vision_agent"
    memory_vector_weight: float = 0.7
    memory_text_weight: float = 0.3

    # Monitoring
    heartbeat_interval_minutes: int = 30
    event_escalation_confidence: float = 0.5  # Min confidence to escalate to LLM
    max_results_per_consume: int = 10

    # Cost control
    max_llm_calls_per_hour: int = 60
    max_image_tokens_per_call: int = 2000  # Resize frames to stay under this

    # Interface
    interface: str = "cli"  # "cli", "slack", "rest"
```
