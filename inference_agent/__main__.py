"""Entry point: python -m inference_agent"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

from inference_agent.config import AgentConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="vision-agent",
        description="Vision Agent: autonomous CV assistant powered by Roboflow Inference",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config.yaml (default: ~/.vision_agent/config.yaml)",
    )
    parser.add_argument(
        "--workspace", "-w",
        type=str,
        default=None,
        help="Workspace directory (default: ~/.vision_agent)",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        help="Inference server URL (default: http://localhost:9001)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Claude model to use (default: claude-sonnet-4-5-20250929)",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        default=False,
        help="Enable extended thinking for complex reasoning",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )
    return parser.parse_args()


def build_agent(config: AgentConfig):
    """Wire all components together and return the agent + CLI."""
    from inference_agent.backends.http_backend import HTTPBackend
    from inference_agent.core.agent import VisionAgent
    from inference_agent.core.prompt_builder import PromptBuilder
    from inference_agent.core.session_log import SessionLog
    from inference_agent.interfaces.cli import CLIInterface
    from inference_agent.llm.claude_client import ClaudeClient
    from inference_agent.memory.store import FileMemoryStore
    from inference_agent.memory.workspace import WorkspaceManager
    from inference_agent.tools.discovery_tools import (
        BlockDiscovery,
        create_get_block_details_tool,
        create_list_workflow_blocks_tool,
    )
    from inference_agent.tools.inference_tools import (
        create_ask_about_image_tool,
        create_detect_zero_shot_tool,
        create_run_inference_tool,
    )
    from inference_agent.tools.memory_tools import (
        create_recall_tool,
        create_remember_tool,
    )
    from inference_agent.tools.pipeline_tools import (
        create_get_pipeline_results_tool,
        create_list_pipelines_tool,
        create_pause_pipeline_tool,
        create_resume_pipeline_tool,
        create_start_pipeline_tool,
        create_stop_pipeline_tool,
    )
    from inference_agent.tools.registry import ToolRegistry
    from inference_agent.tools.result_summarizer import ResultSummarizer
    from inference_agent.tools.think_tool import create_think_tool

    # Initialize workspace
    workspace_path = Path(config.workspace_path).expanduser()
    workspace = WorkspaceManager(str(workspace_path))
    workspace.initialize()

    # LLM client
    llm = ClaudeClient(
        api_key=config.llm.api_key,
        model=config.llm.model,
        max_tokens=config.llm.max_tokens,
    )

    # Inference backend
    backend = HTTPBackend(
        api_url=config.inference.server_url,
        api_key=config.inference.api_key or None,
    )

    # Memory
    memory = FileMemoryStore(workspace_path)

    # Block discovery
    block_discovery = BlockDiscovery(
        mode="http",
        api_url=config.inference.server_url,
        api_key=config.inference.api_key or None,
    )

    # Result summarizer
    summarizer = ResultSummarizer(
        max_image_dimension=config.monitoring.result_image_max_dimension
    )

    # Tool registry — register all 15 tools
    tools = ToolRegistry()

    # Inference tools (3)
    tools.register(create_run_inference_tool(backend, summarizer))
    tools.register(create_ask_about_image_tool(backend))
    tools.register(create_detect_zero_shot_tool(backend, summarizer))

    # Pipeline tools (6)
    tools.register(create_start_pipeline_tool(backend))
    tools.register(create_get_pipeline_results_tool(backend, summarizer))
    tools.register(create_list_pipelines_tool(backend))
    tools.register(create_stop_pipeline_tool(backend))
    tools.register(create_pause_pipeline_tool(backend))
    tools.register(create_resume_pipeline_tool(backend))

    # Discovery tools (2)
    tools.register(create_list_workflow_blocks_tool(block_discovery))
    tools.register(create_get_block_details_tool(block_discovery))

    # Memory tools (2)
    tools.register(create_remember_tool(memory))
    tools.register(create_recall_tool(memory))

    # Reasoning tools (1)
    tools.register(create_think_tool())

    # Prompt builder
    prompt_builder = PromptBuilder(
        workspace=workspace,
        tool_registry=tools,
        block_discovery=block_discovery,
    )

    # Session log
    session_log = SessionLog(workspace.get_sessions_dir())

    # Thinking config
    thinking = None
    if config.llm.thinking_enabled:
        thinking = {
            "type": "enabled",
            "budget_tokens": config.llm.thinking_budget_tokens,
        }

    # Agent core
    agent = VisionAgent(
        llm=llm,
        tools=tools,
        prompt_builder=prompt_builder,
        session_log=session_log,
        thinking_config=thinking,
    )

    # CLI interface
    cli = CLIInterface(agent)

    return agent, cli


def main():
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    config = AgentConfig.load(args.config)

    # Apply CLI overrides
    if args.workspace:
        config.workspace_path = args.workspace
    if args.server_url:
        config.inference.server_url = args.server_url
    if args.model:
        config.llm.model = args.model
    if args.thinking:
        config.llm.thinking_enabled = True

    # Validate required keys
    if not config.llm.api_key:
        print("Error: ANTHROPIC_API_KEY not set. Set it in environment or config.yaml.")
        sys.exit(1)

    # Build and run
    agent, cli = build_agent(config)

    try:
        asyncio.run(cli.run_conversation())
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
