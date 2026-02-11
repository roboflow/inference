"""Agent configuration with Pydantic validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-5-20250929"
    api_key: str = Field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    max_tokens: int = 4096
    thinking_enabled: bool = False
    thinking_budget_tokens: int = 4096
    cost_limit_daily: float = 10.0


class InferenceConfig(BaseModel):
    backend: str = "http"  # "http", "direct", or "webrtc"
    server_url: str = "http://localhost:9001"
    api_key: str = Field(default_factory=lambda: os.environ.get("ROBOFLOW_API_KEY", ""))


class HeartbeatConfig(BaseModel):
    enabled: bool = False
    interval_minutes: int = 15


class MonitoringConfig(BaseModel):
    escalation_confidence: float = 0.5
    max_results_per_consume: int = 10
    result_image_max_dimension: int = 640


class InterfaceConfig(BaseModel):
    type: str = "cli"


class AgentConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    interface: InterfaceConfig = Field(default_factory=InterfaceConfig)
    workspace_path: str = "~/.vision_agent"

    @classmethod
    def load(cls, path: Optional[str] = None) -> AgentConfig:
        """Load config from YAML file, falling back to defaults."""
        if path is None:
            path = os.path.expanduser("~/.vision_agent/config.yaml")
        config_path = Path(path)
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
            # Resolve env var references like ${ANTHROPIC_API_KEY}
            data = _resolve_env_vars(data)
            return cls.model_validate(data)
        return cls()

    def save(self, path: Optional[str] = None) -> None:
        """Save config to YAML file."""
        if path is None:
            path = os.path.expanduser(
                str(Path(self.workspace_path) / "config.yaml")
            )
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)


def _resolve_env_vars(data: dict) -> dict:
    """Recursively resolve ${ENV_VAR} references in config values."""
    resolved = {}
    for key, value in data.items():
        if isinstance(value, dict):
            resolved[key] = _resolve_env_vars(value)
        elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            resolved[key] = os.environ.get(env_var, "")
        else:
            resolved[key] = value
    return resolved
