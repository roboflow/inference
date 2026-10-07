"""Ensure CPU build workflows exercise the runtime checks on their own image."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = [
    ROOT / f".github/workflows/docker.cpu{suffix}.yml"
    for suffix in ("", ".slim", ".parallel")
]


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda path: path.name)
def test_cpu_build_runs_runtime_checks(workflow):
    """Run opt-in checks inside the loaded build, including non-publishing builds.

    Args:
        workflow (Path): CPU build workflow to check.
    """
    steps = yaml.safe_load(workflow.read_text())["jobs"]["docker"]["steps"]
    build_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("uses") == "depot/build-push-action@v1"
    )
    build = steps[build_index]
    assert build["with"].get("load") is True
    runtime_steps = [
        step
        for step in steps[build_index + 1 :]
        if "test_cpu_torch_runtime.py" in step.get("run", "")
    ]
    assert len(runtime_steps) == 1
    runtime = runtime_steps[0]
    assert "if" not in runtime
    assert runtime["env"]["IMAGE_TAGS"] == build["with"]["tags"]
    command = runtime["run"]
    assert 'image="${IMAGE_TAGS%%,*}"' in command
    assert "docker run --rm --pull=never --entrypoint /bin/sh" in command
    assert "-e CPU_IMAGE_TEST=true" in command
    assert "${{ github.workspace }}/tests/docker/test_cpu_torch_runtime.py:" in command
    assert ":ro" in command
    assert '"$image" -c' in command
    assert "python3 -m pytest /test_cpu_torch_runtime.py" in command
