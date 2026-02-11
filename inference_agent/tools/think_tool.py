"""Think tool: explicit reasoning scratchpad for the agent."""

from __future__ import annotations

from inference_agent.core.protocols import ToolDefinition
from inference_agent.tools.registry import Tool

THINK_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {
            "type": "string",
            "description": "Your step-by-step reasoning about the problem.",
        },
    },
    "required": ["thought"],
}


def create_think_tool() -> Tool:
    async def execute(thought: str) -> str:
        # The think tool has no side effects. It exists so Claude can
        # reason explicitly in multi-step agentic workflows.
        return ""

    return Tool(
        definition=ToolDefinition(
            name="think",
            description=(
                "Think through complex problems step-by-step before taking action. "
                "Use when you need to plan a workflow, analyze ambiguous results, "
                "or decide between approaches. No side effects."
            ),
            input_schema=THINK_SCHEMA,
        ),
        execute=execute,
        category="reasoning",
    )
