"""Memory tools: remember, recall."""

from __future__ import annotations

from typing import Any, Optional

from inference_agent.core.protocols import Memory, ToolDefinition
from inference_agent.tools.registry import Tool


# ---------------------------------------------------------------------------
# remember
# ---------------------------------------------------------------------------

REMEMBER_SCHEMA = {
    "type": "object",
    "properties": {
        "content": {
            "type": "string",
            "description": "What to remember — an observation, fact, or user preference.",
        },
        "category": {
            "type": "string",
            "enum": ["observation", "knowledge", "preference"],
            "description": "Category: 'observation' (event seen), 'knowledge' (learned fact), 'preference' (user pref).",
        },
        "camera_id": {
            "type": "string",
            "description": "Associated camera name/ID (optional).",
        },
    },
    "required": ["content", "category"],
}


def create_remember_tool(memory: Memory) -> Tool:
    async def execute(
        content: str,
        category: str = "observation",
        camera_id: Optional[str] = None,
    ) -> str:
        metadata = {}
        if camera_id:
            metadata["camera_id"] = camera_id

        await memory.store(content, category, metadata if metadata else None)
        return f"Remembered ({category}): {content[:100]}{'...' if len(content) > 100 else ''}"

    return Tool(
        definition=ToolDefinition(
            name="remember",
            description=(
                "Store an observation, fact, or user preference in persistent memory. "
                "Use this to record significant events, patterns, or things the user "
                "asks you to remember."
            ),
            input_schema=REMEMBER_SCHEMA,
        ),
        execute=execute,
        category="memory",
    )


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------

RECALL_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query for memory.",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results (default 5).",
        },
    },
    "required": ["query"],
}


def create_recall_tool(memory: Memory) -> Tool:
    async def execute(query: str, max_results: int = 5) -> str:
        results = await memory.search(query, max_results=max_results)
        if not results:
            return "No memories found matching that query."

        lines = [f"Found {len(results)} relevant memories:"]
        for r in results:
            source_info = f" [{r.source}]" if r.source else ""
            lines.append(f"  - {r.content}{source_info}")
        return "\n".join(lines)

    return Tool(
        definition=ToolDefinition(
            name="recall",
            description=(
                "Search memory for relevant past observations, knowledge, or preferences."
            ),
            input_schema=RECALL_SCHEMA,
        ),
        execute=execute,
        category="memory",
    )
