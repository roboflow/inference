"""Protocol interfaces defining the boundaries between all agent components."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable, Optional, Protocol


# ---------------------------------------------------------------------------
# Event types emitted by the agent
# ---------------------------------------------------------------------------

class EventType(Enum):
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    RESPONSE = "response"
    ALERT = "alert"
    STREAM_EVENT = "stream_event"
    ERROR = "error"
    ASK_USER = "ask_user"


@dataclass
class AgentEvent:
    type: EventType
    data: Any
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# LLM types
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: dict
    cache_control: Optional[dict] = None


@dataclass
class LLMResponse:
    content_blocks: list[dict]
    text: Optional[str]
    tool_calls: list[ToolCall]
    stop_reason: str
    usage: dict


# ---------------------------------------------------------------------------
# Inference backend types
# ---------------------------------------------------------------------------

@dataclass
class PipelineHandle:
    pipeline_id: str
    backend: str
    video_reference: str
    workflow_description: str
    started_at: float
    status: str = "running"


@dataclass
class InferenceResult:
    predictions: dict
    frame: Optional[bytes] = None
    frame_id: Optional[int] = None
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Search result for memory
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    content: str
    source: str
    score: float
    metadata: Optional[dict] = None


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class AgentCore(Protocol):

    async def process_message(
        self,
        message: str,
        attachments: Optional[list[bytes]] = None,
    ) -> AsyncIterator[AgentEvent]:
        """Process a user message. Yields events as the agent works."""
        ...

    async def process_heartbeat(self) -> AsyncIterator[AgentEvent]:
        """Run a heartbeat check."""
        ...

    async def shutdown(self) -> None:
        """Graceful shutdown: stop streams, flush memory, save state."""
        ...


class LLMClient(Protocol):

    async def chat(
        self,
        messages: list[dict],
        tools: list[ToolDefinition],
        system: list[dict],
        max_tokens: int = 4096,
        thinking: Optional[dict] = None,
    ) -> LLMResponse:
        """Send messages to Claude. Handles caching, retries, thinking."""
        ...


class InferenceBackend(Protocol):

    async def run_single(
        self,
        image: str | bytes,
        model_id: Optional[str] = None,
        workflow_spec: Optional[dict] = None,
        workspace_name: Optional[str] = None,
        workflow_id: Optional[str] = None,
        parameters: Optional[dict] = None,
    ) -> InferenceResult:
        ...

    async def start_pipeline(
        self,
        video_reference: str,
        workflow_spec: Optional[dict] = None,
        workspace_name: Optional[str] = None,
        workflow_id: Optional[str] = None,
        max_fps: Optional[float] = None,
        parameters: Optional[dict] = None,
    ) -> PipelineHandle:
        ...

    async def consume_results(
        self,
        pipeline_id: str,
        max_results: int = 5,
    ) -> list[InferenceResult]:
        ...

    async def stop_pipeline(self, pipeline_id: str) -> None: ...
    async def pause_pipeline(self, pipeline_id: str) -> None: ...
    async def resume_pipeline(self, pipeline_id: str) -> None: ...
    async def list_pipelines(self) -> list[PipelineHandle]: ...
    async def get_pipeline_status(self, pipeline_id: str) -> PipelineHandle: ...


class Memory(Protocol):

    async def store(
        self, content: str, category: str, metadata: Optional[dict] = None
    ) -> None:
        ...

    async def search(
        self, query: str, max_results: int = 10
    ) -> list[SearchResult]:
        ...

    async def get_daily_observations(
        self, date: Optional[str] = None, camera_id: Optional[str] = None
    ) -> str:
        ...

    def load_workspace_files(self) -> dict[str, str]:
        ...

    async def save_active_state(self, state: dict) -> None:
        ...

    async def load_active_state(self) -> Optional[dict]:
        ...


class UserInterface(Protocol):

    async def start(
        self, event_callback: Callable[[AgentEvent], Awaitable[None]]
    ) -> None:
        ...

    async def run_conversation(self, agent: AgentCore) -> None:
        ...

    async def emit_event(self, event: AgentEvent) -> None:
        ...
