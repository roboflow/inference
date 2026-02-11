"""Workspace manager: directory structure, workspace file loading, skills."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_AGENTS_MD = """\
# Vision Agent Instructions

## Behavior
- When the user asks you to monitor a camera, always confirm the setup before starting
- When you detect something notable, include a frame snapshot in the report
- Summarize observations periodically, not per-frame
- Use zero-shot detection (YOLO-World) when the user doesn't specify a model

## Defaults
- Default confidence threshold: 0.4
- Default max FPS for monitoring: 5
- Always ask the user before starting a new pipeline

## Tips
- You can edit this file to customize the agent's behavior
"""

DEFAULT_USER_MD = """\
# User Profile

<!-- Fill in your details so the agent can personalize responses -->

- Name:
- Role:
- Preferences:
"""

DEFAULT_CAMERAS_MD = """\
# Cameras

<!-- Register your cameras here. The agent reads this on every turn. -->

| Name | URL | Location | Notes |
|------|-----|----------|-------|
| <!-- cam1 --> | <!-- rtsp://... --> | <!-- Location --> | <!-- Notes --> |
"""

DEFAULT_HEARTBEAT_MD = """\
# Heartbeat Checklist

<!-- The agent checks this list periodically. Customize it for your setup. -->

Every 15 minutes, check the following:

1. Are all registered cameras still streaming? Alert if any dropped.
2. Check active pipelines for errors or stalls.
3. If nothing notable, respond with HEARTBEAT_OK (silent).
"""


class WorkspaceManager:
    """Manages the agent's workspace directory and files."""

    def __init__(self, workspace_path: str = "~/.vision_agent"):
        self._path = Path(workspace_path).expanduser()

    @property
    def path(self) -> Path:
        return self._path

    def initialize(self) -> None:
        """Create workspace directory with default files if not exists."""
        self._path.mkdir(parents=True, exist_ok=True)

        defaults = {
            "AGENTS.md": DEFAULT_AGENTS_MD,
            "USER.md": DEFAULT_USER_MD,
            "CAMERAS.md": DEFAULT_CAMERAS_MD,
            "HEARTBEAT.md": DEFAULT_HEARTBEAT_MD,
        }

        for filename, content in defaults.items():
            filepath = self._path / filename
            if not filepath.exists():
                filepath.write_text(content)
                logger.info("Created default workspace file: %s", filepath)

        # Create subdirectories
        for subdir in ("memory", "sessions", "skills"):
            (self._path / subdir).mkdir(exist_ok=True)

        # Create default knowledge file
        knowledge = self._path / "memory" / "KNOWLEDGE.md"
        if not knowledge.exists():
            knowledge.write_text("# Knowledge Base\n\n<!-- Long-term learned facts -->\n")

    def load_workspace_files(self) -> dict[str, str]:
        """Load workspace .md files for system prompt injection.

        Returns dict of filename -> content for all .md files in workspace root.
        Skips files larger than ~4K tokens (~16KB) to prevent prompt bloat.
        """
        files = {}
        max_size = 16_000  # ~4K tokens

        for filepath in sorted(self._path.glob("*.md")):
            if filepath.stat().st_size > max_size:
                logger.warning(
                    "Skipping %s (too large: %d bytes)", filepath.name, filepath.stat().st_size
                )
                continue
            try:
                files[filepath.name] = filepath.read_text()
            except Exception as e:
                logger.warning("Failed to read %s: %s", filepath.name, e)

        return files

    def list_available_skills(self) -> list[dict]:
        """Discover skills in the skills/ directory.

        Returns [{name, description, path}] for each SKILL.md found.
        """
        skills_dir = self._path / "skills"
        if not skills_dir.exists():
            return []

        skills = []
        for skill_dir in sorted(skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if skill_file.exists():
                content = skill_file.read_text()
                # Parse frontmatter for description
                description = _extract_frontmatter_field(content, "description")
                skills.append({
                    "name": skill_dir.name,
                    "description": description or skill_dir.name,
                    "path": str(skill_file),
                })
        return skills

    def load_skill(self, skill_name: str) -> Optional[str]:
        """Load a specific skill's SKILL.md content."""
        skill_file = self._path / "skills" / skill_name / "SKILL.md"
        if skill_file.exists():
            return skill_file.read_text()
        return None

    def get_sessions_dir(self) -> Path:
        return self._path / "sessions"

    def get_memory_dir(self) -> Path:
        return self._path / "memory"

    def get_active_state_path(self) -> Path:
        return self._path / "active_state.json"


def _extract_frontmatter_field(content: str, field: str) -> Optional[str]:
    """Extract a field from YAML frontmatter in a Markdown file."""
    if not content.startswith("---"):
        return None
    try:
        end = content.index("---", 3)
        frontmatter = content[3:end]
        for line in frontmatter.strip().split("\n"):
            if line.startswith(f"{field}:"):
                return line[len(field) + 1:].strip()
    except ValueError:
        pass
    return None
