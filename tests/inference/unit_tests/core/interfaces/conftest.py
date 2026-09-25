import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Generator, Optional

import pytest


@pytest.fixture(scope="function")
def empty_directory() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def inference_caplog(
    caplog: pytest.LogCaptureFixture,
) -> Generator[pytest.LogCaptureFixture, None, None]:
    """caplog attaches to the root logger, but the ``inference`` logger (and
    its descendants) has ``propagate = False`` (see
    ``inference/core/logger.py``), so records never reach it. Attach
    caplog's handler directly to the ``inference`` logger instead.
    """
    inference_logger = logging.getLogger("inference")
    inference_logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        inference_logger.removeHandler(caplog.handler)


def _missing_git_history_reason(sha: str, *, project_root: Path) -> Optional[str]:
    """Return why git history for ``sha`` is unavailable, or None if it is.

    Private helper for `require_git_baseline_history`; kept separate so the
    three checks (executable, repository metadata, commit reachability) each
    produce a distinct, actionable message. Only the two conditions a
    portable checkout actually hits - no git repository here, and the commit
    missing from the object database - count as "unavailable". Any other
    git failure (permissions, corrupt repo, bad config, ...) is a real
    problem and fails immediately via `pytest.fail`, in every mode.
    """
    if shutil.which("git") is None:
        return f"git executable not found on PATH (need it for baseline commit {sha})"

    has_metadata = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
    )
    if has_metadata.returncode != 0:
        if "not a git repository" in has_metadata.stderr.lower():
            return (
                f"{project_root} has no git repository metadata "
                f"(need it for baseline commit {sha})"
            )
        pytest.fail(
            f"git rev-parse failed unexpectedly in {project_root}: "
            f"{has_metadata.stderr}"
        )

    has_commit = subprocess.run(
        ["git", "-C", str(project_root), "cat-file", "-e", f"{sha}^{{commit}}"],
        capture_output=True,
        text=True,
    )
    if has_commit.returncode != 0:
        if "not a valid object name" in has_commit.stderr.lower():
            return (
                f"baseline commit {sha} unreachable from local git history; "
                f"run `git fetch --depth 1 origin {sha}`"
            )
        pytest.fail(
            f"git cat-file failed unexpectedly for {sha} in {project_root}: "
            f"{has_commit.stderr}"
        )

    return None


def require_git_baseline_history(sha: str, *, project_root: Path) -> None:
    """Skip or fail a test that needs local git history for a baseline commit.

    Reuses the availability-check pattern from
    `tests/inference/unit_tests/test_workflows_package_compatibility.py`
    (`test_inventory_matches_historical_git_tree`), generalized to a shared
    helper: checks that `git` is on PATH, that `project_root` has git
    repository metadata, and that `sha` is reachable there. Ordinary runs
    skip with an actionable reason naming `sha` when history is unavailable
    (e.g. a shallow checkout or an exported source tree). Setting the
    `STREAMS_REQUIRE_BASELINE_HISTORY=1` environment variable turns that
    skip into a hard failure, so CI cannot silently lose this coverage if its
    baseline fetch breaks. Any other git failure fails immediately in every
    mode; see `_missing_git_history_reason`.

    Args:
        sha: Commit SHA the calling test needs history for.
        project_root: Repository root to run git commands against.
    """
    reason = _missing_git_history_reason(sha, project_root=project_root)
    if reason is None:
        return

    if os.environ.get("STREAMS_REQUIRE_BASELINE_HISTORY") == "1":
        pytest.fail(reason)

    pytest.skip(reason)
