"""Workflow discovery tools: list_workflow_blocks, get_block_details."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from inference_agent.core.protocols import ToolDefinition
from inference_agent.tools.registry import Tool

logger = logging.getLogger(__name__)


class BlockDiscovery:
    """Discovers available workflow blocks from the inference server or local engine.

    Supports two modes:
    - HTTP: queries the /workflows/blocks/describe endpoint
    - Direct: imports from inference.core.workflows
    """

    def __init__(self, mode: str = "http", api_url: Optional[str] = None, api_key: Optional[str] = None):
        self._mode = mode
        self._api_url = api_url
        self._api_key = api_key
        self._blocks_cache: Optional[dict] = None

    def _load_blocks_direct(self) -> dict:
        """Load blocks by directly importing the workflow engine."""
        try:
            from inference.core.workflows.execution_engine.introspection.blocks_loader import (
                describe_available_blocks,
            )
            description = describe_available_blocks(dynamic_blocks=[])
            blocks = {}
            for block in description.blocks:
                blocks[block.manifest_type_identifier] = {
                    "name": block.human_friendly_block_name,
                    "type": block.manifest_type_identifier,
                    "schema": block.block_schema,
                    "outputs": [
                        {"name": o.name, "kind": [str(k) for k in o.kind]}
                        for o in block.outputs_manifest
                    ],
                }
            return blocks
        except ImportError:
            logger.warning("Cannot import workflow engine directly; falling back to HTTP")
            return self._load_blocks_http()

    def _load_blocks_http(self) -> dict:
        """Load blocks via the HTTP API."""
        import requests

        url = f"{self._api_url}/workflows/blocks/describe"
        headers = {}
        if self._api_key:
            headers["api_key"] = self._api_key
        try:
            resp = requests.post(url, json={}, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            blocks = {}
            for block in data.get("blocks", []):
                block_type = block.get("manifest_type_identifier", block.get("type", ""))
                blocks[block_type] = {
                    "name": block.get("human_friendly_block_name", block_type),
                    "type": block_type,
                    "schema": block.get("block_schema", {}),
                    "outputs": block.get("outputs_manifest", []),
                }
            return blocks
        except Exception as e:
            logger.error("Failed to load blocks via HTTP: %s", e)
            return {}

    def get_blocks(self) -> dict:
        """Get all available blocks (cached after first load)."""
        if self._blocks_cache is None:
            if self._mode == "direct":
                self._blocks_cache = self._load_blocks_direct()
            else:
                self._blocks_cache = self._load_blocks_http()
        return self._blocks_cache

    def get_block_summary(self) -> str:
        """One-line summary per block for the system prompt."""
        blocks = self.get_blocks()
        if not blocks:
            return "No workflow blocks discovered yet."

        # Group by category based on type prefix
        categories: dict[str, list[str]] = {}
        for block_type, info in sorted(blocks.items()):
            # Extract category from type like "roboflow_core/foo@v1"
            parts = block_type.split("/")
            category = parts[0] if len(parts) > 1 else "other"
            name = info.get("name", block_type)
            categories.setdefault(category, []).append(f"  - `{block_type}`: {name}")

        lines = [f"## Available Workflow Blocks ({len(blocks)} total)\n"]
        for category, items in sorted(categories.items()):
            lines.append(f"### {category}")
            lines.extend(items[:30])  # Limit per category to control tokens
            if len(items) > 30:
                lines.append(f"  ... and {len(items) - 30} more")
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# list_workflow_blocks tool
# ---------------------------------------------------------------------------

LIST_BLOCKS_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "description": (
                "Filter by category: 'model', 'analytics', 'visualization', "
                "'transformation', 'sink', 'flow_control', 'classical_cv', "
                "'fusion', 'tracking'. Omit to list all."
            ),
        },
    },
}


def create_list_workflow_blocks_tool(discovery: BlockDiscovery) -> Tool:
    async def execute(category: Optional[str] = None) -> str:
        blocks = discovery.get_blocks()
        if not blocks:
            return "No workflow blocks available. The inference server may not be reachable."

        filtered = blocks
        if category:
            category_lower = category.lower()
            filtered = {
                k: v for k, v in blocks.items()
                if category_lower in k.lower() or category_lower in v.get("name", "").lower()
            }

        if not filtered:
            return f"No blocks found matching category '{category}'."

        lines = [f"Found {len(filtered)} blocks:"]
        for block_type, info in sorted(filtered.items()):
            name = info.get("name", block_type)
            lines.append(f"  - `{block_type}`: {name}")
        return "\n".join(lines)

    return Tool(
        definition=ToolDefinition(
            name="list_workflow_blocks",
            description=(
                "List available workflow blocks, optionally filtered by category. "
                "Use this when you need to compose a custom workflow and want to "
                "know what blocks exist."
            ),
            input_schema=LIST_BLOCKS_SCHEMA,
        ),
        execute=execute,
        category="discovery",
    )


# ---------------------------------------------------------------------------
# get_block_details tool
# ---------------------------------------------------------------------------

GET_BLOCK_DETAILS_SCHEMA = {
    "type": "object",
    "properties": {
        "block_type": {
            "type": "string",
            "description": "Block type identifier, e.g. 'roboflow_core/line_counter@v2'.",
        },
    },
    "required": ["block_type"],
}


def create_get_block_details_tool(discovery: BlockDiscovery) -> Tool:
    async def execute(block_type: str) -> str:
        blocks = discovery.get_blocks()
        block = blocks.get(block_type)

        if block is None:
            # Try partial match
            matches = [
                (k, v) for k, v in blocks.items()
                if block_type.lower() in k.lower()
            ]
            if matches:
                lines = [f"Block '{block_type}' not found. Did you mean:"]
                for k, v in matches[:5]:
                    lines.append(f"  - `{k}`: {v.get('name', k)}")
                return "\n".join(lines)
            return f"Block '{block_type}' not found."

        schema = block.get("schema", {})
        outputs = block.get("outputs", [])

        # Format the schema in a readable way
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        lines = [
            f"## {block.get('name', block_type)}",
            f"Type: `{block_type}`\n",
            "### Input Parameters:",
        ]

        for prop_name, prop_info in properties.items():
            if prop_name in ("type", "name"):
                continue  # Skip meta fields
            req_marker = " (required)" if prop_name in required else ""
            prop_type = prop_info.get("type", "any")
            desc = prop_info.get("description", "")
            lines.append(f"  - `{prop_name}`: {prop_type}{req_marker} — {desc}")

        if outputs:
            lines.append("\n### Outputs:")
            for output in outputs:
                if isinstance(output, dict):
                    oname = output.get("name", "?")
                    okind = output.get("kind", [])
                    lines.append(f"  - `{oname}`: {', '.join(str(k) for k in okind)}")

        return "\n".join(lines)

    return Tool(
        definition=ToolDefinition(
            name="get_block_details",
            description=(
                "Get the full schema, parameters, and outputs for a specific "
                "workflow block. Use this before composing a workflow to understand "
                "a block's inputs, outputs, and required parameters."
            ),
            input_schema=GET_BLOCK_DETAILS_SCHEMA,
        ),
        execute=execute,
        category="discovery",
    )
