"""Vision Agent: the agentic loop that orchestrates inference via Claude."""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import AsyncIterator, Optional

from inference_agent.core.protocols import (
    AgentEvent,
    EventType,
    LLMResponse,
    ToolCall,
)
from inference_agent.core.prompt_builder import PromptBuilder
from inference_agent.core.session_log import SessionLog
from inference_agent.llm.claude_client import ClaudeClient
from inference_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

MAX_TOOL_LOOP_ITERATIONS = 25
HEARTBEAT_OK = "HEARTBEAT_OK"


class VisionAgent:
    """The core agentic loop.

    Receives user messages, builds system prompts, calls Claude with tools,
    executes tool calls, and feeds results back until Claude produces a
    final response.
    """

    def __init__(
        self,
        llm: ClaudeClient,
        tools: ToolRegistry,
        prompt_builder: PromptBuilder,
        session_log: SessionLog,
        thinking_config: Optional[dict] = None,
    ):
        self._llm = llm
        self._tools = tools
        self._prompt_builder = prompt_builder
        self._session_log = session_log
        self._thinking = thinking_config
        self._messages: list[dict] = []
        self._active_pipelines: list = []

    async def process_message(
        self,
        message: str,
        attachments: Optional[list[bytes]] = None,
    ) -> AsyncIterator[AgentEvent]:
        """Process a user message through the agentic loop.

        Yields AgentEvents as the agent reasons, calls tools, and responds.
        """
        # Build user message with optional image attachments
        user_content = _build_user_content(message, attachments)
        self._messages.append({"role": "user", "content": user_content})
        self._session_log.log_user_message(message, attachments)

        # Build system prompt
        system = self._prompt_builder.build(
            active_pipelines=self._active_pipelines, mode="full"
        )
        tool_defs = self._tools.list_definitions()

        # Agentic loop
        for iteration in range(MAX_TOOL_LOOP_ITERATIONS):
            response = await self._llm.chat(
                messages=self._messages,
                tools=tool_defs,
                system=system,
                thinking=self._thinking,
            )

            # Append assistant response to conversation
            self._messages.append({
                "role": "assistant",
                "content": response.content_blocks,
            })
            self._session_log.log_assistant_response(response.content_blocks)

            # Yield thinking events
            for block in response.content_blocks:
                if block.get("type") == "thinking":
                    yield AgentEvent(
                        type=EventType.THINKING,
                        data=block.get("thinking", ""),
                    )

            # If the model finished, yield the final text response
            if response.stop_reason == "end_turn":
                if response.text:
                    yield AgentEvent(type=EventType.RESPONSE, data=response.text)
                break

            # If the model wants to use tools, execute them
            if response.stop_reason == "tool_use" and response.tool_calls:
                tool_results = []
                for tool_call in response.tool_calls:
                    yield AgentEvent(
                        type=EventType.TOOL_CALL,
                        data={"name": tool_call.name, "arguments": tool_call.arguments},
                    )
                    self._session_log.log_tool_call(
                        tool_call.name, tool_call.id, tool_call.arguments
                    )

                    # Execute the tool
                    try:
                        result = await self._tools.execute(
                            tool_call.name, tool_call.arguments
                        )
                        is_error = False
                    except Exception as e:
                        result = f"Error: {e}"
                        is_error = True

                    self._session_log.log_tool_result(
                        tool_call.name, tool_call.id, result, is_error
                    )

                    # Format the result
                    formatted = ClaudeClient.format_tool_result(
                        tool_call.id, result, is_error
                    )
                    tool_results.append(formatted)

                    # Yield tool result event (summary for display)
                    result_summary = str(result)
                    if len(result_summary) > 200:
                        result_summary = result_summary[:200] + "..."
                    yield AgentEvent(
                        type=EventType.TOOL_RESULT,
                        data={
                            "name": tool_call.name,
                            "result": result_summary,
                            "is_error": is_error,
                        },
                    )

                # Append tool results as a user message
                self._messages.append({
                    "role": "user",
                    "content": tool_results,
                })
                continue

            # Unexpected stop reason
            if response.stop_reason == "max_tokens":
                yield AgentEvent(
                    type=EventType.ERROR,
                    data="Response was truncated (max_tokens reached).",
                )
                break

            # Safety: break on unknown stop reason
            logger.warning("Unexpected stop_reason: %s", response.stop_reason)
            break

        else:
            # Exceeded max iterations
            yield AgentEvent(
                type=EventType.ERROR,
                data=f"Agent loop exceeded {MAX_TOOL_LOOP_ITERATIONS} iterations.",
            )

    async def process_heartbeat(self) -> AsyncIterator[AgentEvent]:
        """Run a heartbeat check.

        Uses a lighter system prompt and checks active pipelines.
        """
        system = self._prompt_builder.build(
            active_pipelines=self._active_pipelines, mode="heartbeat"
        )
        tool_defs = self._tools.list_definitions()

        # Build heartbeat message
        pipeline_summary = "No active pipelines."
        if self._active_pipelines:
            lines = []
            for p in self._active_pipelines:
                lines.append(f"  - {p.pipeline_id}: {p.status}")
            pipeline_summary = "\n".join(lines)

        heartbeat_message = (
            f"Heartbeat check. Active pipelines:\n{pipeline_summary}\n\n"
            "Please review your checklist. If nothing notable, respond with HEARTBEAT_OK."
        )

        messages = [{"role": "user", "content": heartbeat_message}]

        for _ in range(5):  # Heartbeat should be quick
            response = await self._llm.chat(
                messages=messages,
                tools=tool_defs,
                system=system,
            )
            messages.append({"role": "assistant", "content": response.content_blocks})

            if response.stop_reason == "end_turn":
                if response.text and HEARTBEAT_OK not in response.text:
                    yield AgentEvent(type=EventType.ALERT, data=response.text)
                break

            if response.stop_reason == "tool_use" and response.tool_calls:
                tool_results = []
                for tc in response.tool_calls:
                    try:
                        result = await self._tools.execute(tc.name, tc.arguments)
                        is_error = False
                    except Exception as e:
                        result = f"Error: {e}"
                        is_error = True
                    tool_results.append(
                        ClaudeClient.format_tool_result(tc.id, result, is_error)
                    )
                messages.append({"role": "user", "content": tool_results})
                continue

            break

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Vision Agent shutting down...")
        self._session_log.log_event("shutdown", {"pipelines": len(self._active_pipelines)})

    @property
    def session_id(self) -> str:
        return self._session_log.session_id

    @property
    def message_count(self) -> int:
        return len(self._messages)


def _build_user_content(
    message: str, attachments: Optional[list[bytes]] = None
) -> str | list:
    """Build user message content, including image attachments if any."""
    if not attachments:
        return message

    content: list[dict] = []
    for attachment in attachments:
        b64 = base64.b64encode(attachment).decode("ascii")
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64,
            },
        })
    content.append({"type": "text", "text": message})
    return content
