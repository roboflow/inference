import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).parents[3]
    / "development"
    / "video_poc"
    / "processor"
    / "build_staging_overlays.py"
)
SPEC = importlib.util.spec_from_file_location("build_staging_overlays", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


GIT_SHA = "a" * 40
DIGEST = "sha256:" + "b" * 64
GPU_BASE = f"{MODULE.GPU_BASE_REPOSITORY}@{DIGEST}"
CPU_BASE = f"{MODULE.CPU_BASE_REPOSITORY}@{DIGEST}"


def test_build_plan_is_staging_only_and_commit_derived():
    plan = MODULE.build_plan(GPU_BASE, CPU_BASE, GIT_SHA)

    assert plan["environment"] == "staging"
    assert plan["project"] == "roboflow-staging"
    assert plan["outputs"] == {
        "gpu": f"{MODULE.GPU_OUTPUT_REPOSITORY}:{GIT_SHA}",
        "cpu": f"{MODULE.CPU_OUTPUT_REPOSITORY}:{GIT_SHA}",
    }
    assert "development/video_poc/processor/processor.py" in plan["sourceFiles"]
    assert "development/video_poc/processor/video_ingest.py" in plan["sourceFiles"]
    assert "development/video_poc/processor/Dockerfile.overlay" in plan["sourceFiles"]
    command = plan["submitCommand"]
    assert command[:3] == ["gcloud", "builds", "submit"]
    assert "--project=roboflow-staging" in command
    substitutions = next(value for value in command if value.startswith("--substitutions="))
    assert GPU_BASE in substitutions
    assert CPU_BASE in substitutions
    assert f"_GIT_SHA={GIT_SHA}" in substitutions


@pytest.mark.parametrize(
    "value,repository",
    (
        (f"{MODULE.GPU_BASE_REPOSITORY}:latest", MODULE.GPU_BASE_REPOSITORY),
        (f"{MODULE.GPU_BASE_REPOSITORY}:2e4a97ee5", MODULE.GPU_BASE_REPOSITORY),
        (
            "us-central1-docker.pkg.dev/roboflow-platform/video-proc/"
            f"video-processor@{DIGEST}",
            MODULE.GPU_BASE_REPOSITORY,
        ),
        (f"{MODULE.GPU_BASE_REPOSITORY}@sha256:short", MODULE.GPU_BASE_REPOSITORY),
    ),
)
def test_validate_digest_ref_rejects_mutable_or_non_staging_input(
    value, repository
):
    with pytest.raises(ValueError):
        MODULE.validate_digest_ref(value, repository)


@pytest.mark.parametrize("git_sha", ("short", "A" * 40, "a" * 39, "a" * 41))
def test_output_ref_requires_full_lowercase_commit(git_sha):
    with pytest.raises(ValueError):
        MODULE.output_ref(MODULE.GPU_OUTPUT_REPOSITORY, git_sha)


def test_resolve_digest_returns_immutable_ref(monkeypatch):
    class Result:
        stdout = DIGEST + "\n"

    monkeypatch.setattr(MODULE.subprocess, "run", lambda *args, **kwargs: Result())

    image = f"{MODULE.GPU_OUTPUT_REPOSITORY}:{GIT_SHA}"
    assert MODULE.resolve_digest(image) == (
        f"{MODULE.GPU_OUTPUT_REPOSITORY}@{DIGEST}"
    )


def test_resolve_digest_rejects_unexpected_registry_output(monkeypatch):
    class Result:
        stdout = "not-a-digest\n"

    monkeypatch.setattr(MODULE.subprocess, "run", lambda *args, **kwargs: Result())

    with pytest.raises(RuntimeError):
        MODULE.resolve_digest(f"{MODULE.GPU_OUTPUT_REPOSITORY}:{GIT_SHA}")


def test_verify_source_revision_rejects_content_drift(monkeypatch):
    class Result:
        def __init__(self, returncode=0):
            self.returncode = returncode

    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return Result(returncode=1 if "diff" in command else 0)

    monkeypatch.setattr(MODULE.subprocess, "run", run)
    plan = MODULE.build_plan(GPU_BASE, CPU_BASE, GIT_SHA)

    with pytest.raises(RuntimeError, match="source files differ"):
        MODULE.verify_source_revision(plan)
    assert any("cat-file" in command for command in calls)
    assert any("diff" in command for command in calls)
