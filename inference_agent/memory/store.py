"""File-based memory store: append to Markdown files, basic search."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from inference_agent.core.protocols import SearchResult

logger = logging.getLogger(__name__)


class FileMemoryStore:
    """Memory implementation using plain Markdown files.

    Stores observations in daily files, knowledge in KNOWLEDGE.md,
    and preferences in a separate section. Search is basic substring
    matching (MVP — to be upgraded to FTS5 + vector search later).
    """

    def __init__(self, workspace_path: Path):
        self._workspace = workspace_path
        self._memory_dir = workspace_path / "memory"
        self._memory_dir.mkdir(parents=True, exist_ok=True)

    async def store(
        self,
        content: str,
        category: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Append content to the appropriate memory file."""
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%H:%M:%S UTC")

        if category == "observation":
            # Daily observation file
            date_str = now.strftime("%Y-%m-%d")
            filepath = self._memory_dir / f"{date_str}.md"
            camera_id = (metadata or {}).get("camera_id", "")
            camera_tag = f" [{camera_id}]" if camera_id else ""
            entry = f"- **{timestamp}**{camera_tag}: {content}\n"
            self._append_to_file(filepath, entry, header=f"# Observations — {date_str}\n\n")

        elif category == "knowledge":
            filepath = self._memory_dir / "KNOWLEDGE.md"
            entry = f"- {content} *(learned {now.strftime('%Y-%m-%d')})*\n"
            self._append_to_file(filepath, entry)

        elif category == "preference":
            # Store preferences in USER.md
            filepath = self._workspace / "USER.md"
            entry = f"- {content}\n"
            self._append_to_file(filepath, entry)

        else:
            logger.warning("Unknown memory category: %s", category)
            filepath = self._memory_dir / "KNOWLEDGE.md"
            entry = f"- [{category}] {content}\n"
            self._append_to_file(filepath, entry)

    async def search(
        self, query: str, max_results: int = 10
    ) -> list[SearchResult]:
        """Basic substring search across all memory files.

        This is the MVP implementation. Will be upgraded to SQLite FTS5 +
        vector search in a later phase.
        """
        results: list[SearchResult] = []
        query_lower = query.lower()

        # Search all .md files in memory directory
        for filepath in sorted(self._memory_dir.glob("*.md"), reverse=True):
            try:
                content = filepath.read_text()
                for line in content.split("\n"):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if query_lower in line.lower():
                        results.append(SearchResult(
                            content=line.lstrip("- "),
                            source=filepath.name,
                            score=1.0,
                        ))
                        if len(results) >= max_results:
                            return results
            except Exception as e:
                logger.warning("Error reading %s: %s", filepath, e)

        return results

    async def get_daily_observations(
        self, date: Optional[str] = None, camera_id: Optional[str] = None
    ) -> str:
        """Get observations for a specific day."""
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        filepath = self._memory_dir / f"{date}.md"
        if not filepath.exists():
            return f"No observations for {date}."

        content = filepath.read_text()
        if camera_id:
            lines = [
                line for line in content.split("\n")
                if camera_id in line or line.startswith("#")
            ]
            return "\n".join(lines)
        return content

    def load_workspace_files(self) -> dict[str, str]:
        """Delegate to workspace manager."""
        # This is called by the workspace manager, not directly
        raise NotImplementedError("Use WorkspaceManager.load_workspace_files()")

    async def save_active_state(self, state: dict) -> None:
        """Persist active pipelines/tasks for restart recovery."""
        state_path = self._workspace / "active_state.json"
        state_path.write_text(json.dumps(state, default=str, indent=2))

    async def load_active_state(self) -> Optional[dict]:
        """Load previously active state."""
        state_path = self._workspace / "active_state.json"
        if state_path.exists():
            try:
                return json.loads(state_path.read_text())
            except Exception as e:
                logger.warning("Failed to load active state: %s", e)
        return None

    def _append_to_file(
        self, filepath: Path, entry: str, header: Optional[str] = None
    ) -> None:
        """Append entry to file, creating with header if needed."""
        if not filepath.exists() and header:
            filepath.write_text(header)
        with open(filepath, "a") as f:
            f.write(entry)
