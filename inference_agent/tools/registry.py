"""Tool registry: holds all tool definitions, dispatches execution."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Awaitable, Optional

from inference_agent.core.protocols import ToolDefinition

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """A registered tool with its definition and executor."""

    definition: ToolDefinition
    execute: Callable[..., Awaitable[Any]]
    category: str = "general"


class ToolRegistry:
    """Central registry of all agent tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.definition.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_definitions(self) -> list[ToolDefinition]:
        """Return all tool definitions for the Claude API."""
        return [t.definition for t in self._tools.values()]

    async def execute(self, name: str, arguments: dict) -> Any:
        """Execute a tool by name with the given arguments.

        Returns the tool result (string, list of content blocks, or dict).
        Raises KeyError if tool not found.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Unknown tool: {name}")
        try:
            return await tool.execute(**arguments)
        except TypeError as e:
            # Handle argument mismatch gracefully
            logger.error("Tool %s argument error: %s", name, e)
            return f"Error: Invalid arguments for tool '{name}': {e}"
        except Exception as e:
            logger.error("Tool %s execution error: %s", name, e, exc_info=True)
            return f"Error executing tool '{name}': {e}"

    def get_capabilities_summary(self) -> str:
        """One-line summary per tool for the system prompt."""
        lines = ["## Available Tools\n"]
        by_category: dict[str, list[Tool]] = {}
        for tool in self._tools.values():
            by_category.setdefault(tool.category, []).append(tool)

        for category, tools in sorted(by_category.items()):
            lines.append(f"### {category.title()}")
            for tool in tools:
                # First sentence of description
                desc = tool.definition.description.split(".")[0] + "."
                lines.append(f"- **{tool.definition.name}**: {desc}")
            lines.append("")
        return "\n".join(lines)
