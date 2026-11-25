"""Prepared source handoff and coordinator identity, without publishing anything."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".github/scripts"))
sys.path.insert(0, str(ROOT / ".release/pypi"))
import image_inputs
import release
from test_pypi_preparation import (
    _build,
    _package,
    _project,
    _publish,
    _PyPI,
    _wheel,
    candidate,
    dependencies,
    planner,
    prepare,
    workspace,
)


def _run(**changes):
    return {
        "id": 100,
        "path": ".github/workflows/publish.pypi.yml",
        "head_sha": "a" * 40,
        "head_branch": "v2.0rc1",
        "event": "release",
        "status": "completed",
        "conclusion": "success",
        **changes,
    }


@pytest.fixture
def image_event(workspace, monkeypatch):
    monkeypatch.setenv("GITHUB_EVENT_NAME", "release")
    monkeypatch.setenv("RELEASE_TAG", "v2.0rc1")
    monkeypatch.setenv("GITHUB_REF", "refs/tags/v2.0rc1")
    monkeypatch.setenv("GITHUB_RUN_ID", "900")
    return workspace


@pytest.mark.parametrize(
    "fault", ["workflow", "sha", "tag", "event", "duplicate", "none"]
)
def test_coordinator_selection_rejects_wrong_or_ambiguous_release(
    image_event, monkeypatch, fault
):
    run = _run()
    key = {
        "workflow": "path",
        "sha": "head_sha",
        "tag": "head_branch",
        "event": "event",
    }.get(fault)
    if key:
        run[key] = "wrong"
    now = [0]
    monkeypatch.setattr(dependencies.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        dependencies.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    monkeypatch.setattr(
        dependencies,
        "_github",
        lambda *args, **kwargs: {
            "workflow_runs": (
                [] if fault == "none" else [run, run] if fault == "duplicate" else [run]
            )
        },
    )
    with pytest.raises(ValueError):
        image_inputs._select_parent(wait_seconds=1)
    assert not Path("release-selection").exists()


@pytest.mark.parametrize(
    "fault",
    ["failed", "expired", "missing", "duplicate", "different-parent", "wrong-sha"],
)
def test_image_gate_never_falls_back_to_checkout_locks(image_event, monkeypatch, fault):
    monkeypatch.setattr(dependencies, "_github", lambda *args, **kwargs: _run())
    image_inputs._select_parent(parent_run_id=100)
    artifact = {
        "id": 42,
        "name": "candidate-image-inputs",
        "expired": fault == "expired",
    }

    def github(path, **kwargs):
        if "/artifacts?" in path:
            return {
                "artifacts": (
                    []
                    if fault == "missing"
                    else [artifact, artifact] if fault == "duplicate" else [artifact]
                )
            }
        return _run(
            conclusion="failure" if fault == "failed" else "success",
            head_sha="b" * 40 if fault == "wrong-sha" else "a" * 40,
        )

    monkeypatch.setattr(dependencies, "_github", github)
    with pytest.raises(ValueError):
        image_inputs._wait(
            parent_run_id=101 if fault == "different-parent" else 100, wait_seconds=1
        )
    assert not Path("release-evidence/image-selection.json").exists()


def test_waits_for_original_coordinator_and_pins_retry_selection(
    image_event, monkeypatch
):
    calls = []
    runs = iter([_run(status="in_progress"), _run()])

    def github(path, **kwargs):
        calls.append(path)
        if "/runs?" in path:
            return {"workflow_runs": [_run(status="queued")]}
        if "/artifacts?" in path:
            return {
                "artifacts": [
                    {"id": 42, "name": "candidate-image-inputs", "expired": False}
                ]
            }
        return next(runs)

    monkeypatch.setattr(dependencies, "_github", github)
    monkeypatch.setattr(dependencies.time, "sleep", lambda seconds: None)
    image_inputs._select_parent()
    candidate._seal("image-selection")
    original = candidate._files(Path("release-candidate/image-selection"))
    shutil.rmtree("release-selection")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    candidate._restore("image-selection")
    image_inputs._wait(wait_seconds=20)
    selected = json.loads(Path("release-evidence/image-selection.json").read_text())
    assert (selected["run_id"], selected["artifact_id"]) == (100, 42)
    assert candidate._files(Path("release-candidate/image-selection")) == original
    assert sum("/runs?" in path for path in calls) == 1


@pytest.mark.parametrize("reuse", [False, True])
def test_real_prepared_locks_reach_image_checkout_and_survive_retry(
    image_event, monkeypatch, reuse
):
    monkeypatch.setenv("GITHUB_RUN_ID", "100")
    catalog = {
        "inference-sdk": _package("inference-sdk", "2.0rc1"),
        "inference-models": _package("inference-models", "2.0rc1"),
        "roboflow-workflows": _package(
            "roboflow-workflows",
            "2.0rc1",
            ["inference-sdk==2.0rc1", "inference-models==2.0rc1"],
        ),
        "streamvision": _package(
            "streamvision", "2.0rc1", ["roboflow-workflows==2.0rc1"]
        ),
    }
    for name in dependencies.PROJECTS:
        _project(name, requirements=catalog[name]["requires_dist"])
    available = list(catalog.values()) if reuse else [catalog["inference-sdk"]]
    api = _PyPI(monkeypatch, available)
    for package in available:
        _publish(image_event, package)
    runtime_dir = Path("dist/pypi")
    runtime_dir.mkdir(parents=True)
    filename, data = _wheel(
        _package("inference", "2.0rc1", [f"{name}==2.0rc1" for name in catalog])
    )
    (runtime_dir / filename).write_bytes(data)
    consumers = dependencies._local_wheels(runtime_dir)
    plan = planner._plan(
        consumers, catalog=catalog, registry=planner._Registry(wait_seconds=120)
    )
    plan["source_revision"] = "a" * 40
    # Each actual publisher has an independent dist directory.
    shutil.rmtree("dist")
    publisher_locks = {}

    def dispatch(name, **kwargs):
        child = 200 + len(publisher_locks)
        with monkeypatch.context() as child_env:
            child_env.setenv("GITHUB_RUN_ID", str(child))
            prepare._prepare(name, plan=plan)
            output = _build(name)
            candidate._seal(name)
            publisher_locks[name] = prepare._paths(name)[1].read_bytes()
            shutil.rmtree(output)
        api.add(catalog[name])
        _publish(image_event, catalog[name])
        prepare._paths(name)[1].unlink()
        prepare._paths(name)[2].unlink()
        return child

    monkeypatch.setattr(planner, "_dispatch", dispatch)
    planner._coordinate(plan, wait_seconds=120)
    runtime_dir.mkdir(parents=True)
    (runtime_dir / filename).write_bytes(data)
    release._write_record(Path("dist/dependency-plan.json"), **plan)
    image_inputs._assemble()
    candidate._seal("image-inputs")
    original = candidate._files(Path("release-candidate/image-inputs"))
    locks = {p: prepare._paths(p)[1].read_bytes() for p in dependencies.PROJECTS}
    if not reuse:
        assert locks == publisher_locks
    for profile in dependencies.PROJECTS:
        prepare._paths(profile)[1].write_bytes(b"stale checkout lock")

    monkeypatch.setenv("GITHUB_RUN_ID", "900")
    image_inputs._restore(parent_run_id=100)
    for profile, expected in locks.items():
        lock = prepare._paths(profile)[1]
        assert lock.read_bytes() == expected
        # The real consumer check accepts exactly the lock restored into the
        # build context. It never runs a mutating lock command.
        subprocess.run(
            ["uv", "lock", "--check", "--offline"], cwd=lock.parent, check=True
        )

    def no_resolution(*args, **kwargs):
        pytest.fail("Image retry must not resolve or rebuild package candidates")

    monkeypatch.setattr(subprocess, "run", no_resolution)
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    image_inputs._restore(parent_run_id=100)
    assert candidate._files(Path("release-candidate/image-inputs")) == original
    receipt = json.loads(Path("release-evidence/image-inputs.json").read_text())
    assert receipt["coordinator_run_id"] == 100
    assert receipt["source_revision"] == "a" * 40
    # Changed metadata in even the last package must block ALL lock overlays.
    Path("stream_vision/pyproject.toml").write_text('name = "changed"')
    Path("inference_models/uv.lock").write_text("sentinel")
    with pytest.raises(ValueError, match="metadata"):
        image_inputs._restore(parent_run_id=100)
    assert Path("inference_models/uv.lock").read_text() == "sentinel"


def test_all_release_image_builds_wait_restore_then_use_local_context():
    covered = []
    for path in (ROOT / ".github/workflows").glob("docker*.yml"):
        workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        if "release" not in workflow["on"]:
            continue
        covered.append(path.name)
        gate = workflow["jobs"]["release_inputs"]
        assert gate["uses"] == "./.github/workflows/prepare.release-image-inputs.yml"
        for job in workflow["jobs"].values():
            steps = job.get("steps", [])
            builds = [
                i
                for i, step in enumerate(steps)
                if step.get("uses", "").startswith(
                    ("depot/build-push-action@", "depot/bake-action@")
                )
            ]
            if not builds:
                continue
            assert "release_inputs" in job["needs"]
            restore = next(
                i
                for i, step in enumerate(steps)
                if step.get("uses") == "./.github/actions/restore-release-inputs"
            )
            checkout = [
                i
                for i, step in enumerate(steps)
                if step.get("uses", "").startswith("actions/checkout@")
            ]
            assert max(checkout) < restore < min(builds)
            assert job["permissions"]["actions"] == "read"
            for i in builds:
                if "build-push" in steps[i]["uses"]:
                    assert steps[i]["with"]["context"] in {".", "./inference_models"}
                else:
                    bake = (ROOT / "docker/docker-bake.jetson.7.2.0.hcl").read_text()
                    assert 'target "jetson-server-jp72" {\n  context    = "."' in bake
    assert len(covered) == 13


def test_input_bundle_requires_successful_runtime_publication():
    workflow = yaml.load(
        (ROOT / ".github/workflows/publish.pypi.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    job = workflow["jobs"]["prepare_image_inputs"]
    assert "publish_runtime" in job["needs"]
    assert "if" not in job  # Default success() blocks failed/skipped publication.
    step = next(
        s for s in job["steps"] if s.get("name") == "Assemble prepared image inputs"
    )
    assert step["if"] == "${{ steps.candidate.outputs.reused != 'true' }}"
