"""CLI REPL interface for the Vision Agent."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from inference_agent.core.agent import VisionAgent
from inference_agent.core.protocols import AgentEvent, EventType

logger = logging.getLogger(__name__)

# ANSI colors
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


class CLIInterface:
    """Terminal-based REPL interface for the Vision Agent."""

    def __init__(self, agent: VisionAgent):
        self._agent = agent

    async def run_conversation(self) -> None:
        """Main REPL loop."""
        _print_banner(self._agent.session_id)

        while True:
            try:
                user_input = await _async_input(f"\n{GREEN}You:{RESET} ")
            except (EOFError, KeyboardInterrupt):
                print(f"\n{DIM}Goodbye!{RESET}")
                await self._agent.shutdown()
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            # Handle special commands
            if user_input.lower() in ("/quit", "/exit", "/q"):
                print(f"{DIM}Shutting down...{RESET}")
                await self._agent.shutdown()
                break

            if user_input.lower() == "/help":
                _print_help()
                continue

            if user_input.lower() == "/status":
                self._print_status()
                continue

            if user_input.lower() == "/usage":
                self._print_usage()
                continue

            # Check for image attachment
            attachments = None
            if user_input.startswith("/image "):
                parts = user_input.split(" ", 2)
                if len(parts) >= 3:
                    image_path = parts[1]
                    message = parts[2]
                    attachments = _load_image(image_path)
                    user_input = message
                else:
                    print(f"{YELLOW}Usage: /image <path> <message>{RESET}")
                    continue

            # Process the message
            print(f"\n{CYAN}Agent:{RESET} ", end="", flush=True)

            try:
                async for event in self._agent.process_message(
                    user_input, attachments
                ):
                    self._render_event(event)
            except Exception as e:
                print(f"\n{RED}Error: {e}{RESET}")
                logger.error("Error processing message", exc_info=True)

    def _render_event(self, event: AgentEvent) -> None:
        """Render an agent event to the terminal."""
        if event.type == EventType.THINKING:
            # Show thinking in dim text
            text = str(event.data)
            if len(text) > 200:
                text = text[:200] + "..."
            print(f"{DIM}(thinking: {text}){RESET}")

        elif event.type == EventType.TOOL_CALL:
            name = event.data.get("name", "?")
            args = event.data.get("arguments", {})
            args_str = ", ".join(f"{k}={_truncate(str(v))}" for k, v in args.items())
            print(f"{DIM}[calling {name}({args_str})]{RESET}")

        elif event.type == EventType.TOOL_RESULT:
            name = event.data.get("name", "?")
            is_error = event.data.get("is_error", False)
            result = event.data.get("result", "")
            if is_error:
                print(f"{RED}[{name} error: {result}]{RESET}")
            elif name != "think":
                # Don't show think tool results (they're empty)
                print(f"{DIM}[{name} → {_truncate(result, 100)}]{RESET}")

        elif event.type == EventType.RESPONSE:
            print(event.data)

        elif event.type == EventType.ALERT:
            print(f"\n{YELLOW}⚠ ALERT: {event.data}{RESET}")

        elif event.type == EventType.ERROR:
            print(f"\n{RED}Error: {event.data}{RESET}")

    def _print_status(self) -> None:
        """Print agent status."""
        print(f"\n{BOLD}Session:{RESET} {self._agent.session_id}")
        print(f"{BOLD}Messages:{RESET} {self._agent.message_count}")

    def _print_usage(self) -> None:
        """Print token usage."""
        usage = self._agent._llm.total_usage
        print(f"\n{BOLD}Token Usage:{RESET}")
        for key, value in usage.items():
            print(f"  {key}: {value:,}")


def _print_banner(session_id: str) -> None:
    print(f"""
{BOLD}╔══════════════════════════════════════════════════╗
║           Vision Agent v0.1.0                    ║
║     Autonomous CV Assistant by Roboflow          ║
╚══════════════════════════════════════════════════╝{RESET}

{DIM}Session: {session_id}
Type /help for commands, /quit to exit.{RESET}
""")


def _print_help() -> None:
    print(f"""
{BOLD}Commands:{RESET}
  /help              Show this help
  /image <path> <msg>  Send an image with a message
  /status            Show session status
  /usage             Show token usage
  /quit              Exit the agent
""")


def _truncate(s: str, max_len: int = 80) -> str:
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def _load_image(path: str) -> Optional[list[bytes]]:
    """Load an image file as bytes."""
    p = Path(path).expanduser()
    if not p.exists():
        print(f"{RED}Image not found: {path}{RESET}")
        return None
    try:
        return [p.read_bytes()]
    except Exception as e:
        print(f"{RED}Failed to read image: {e}{RESET}")
        return None


async def _async_input(prompt: str) -> str:
    """Read input asynchronously to not block the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: input(prompt))
