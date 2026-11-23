"""Retained-candidate and publisher-identity regression boundaries."""

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".release/pypi"))
import dependencies


def test_coordinator_retry_reuses_original_publisher_run(monkeypatch):
    monkeypatch.setenv("SOURCE_REVISION", "a" * 40)
    monkeypatch.setenv("GITHUB_REF", "refs/heads/release/1.8.7rc1")
    monkeypatch.setenv("GITHUB_RUN_ID", "100")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    calls = []
    original = {
        "id": 42,
        "head_sha": "a" * 40,
        "path": ".github/workflows/publish.pypi.inference_exp.yml",
        "event": "workflow_dispatch",
        "display_title": "pypi-inference-models-parent-100",
        "status": "completed",
        "conclusion": "success",
        "run_attempt": 1,
    }

    def github(path, *, payload=None, **kwargs):
        calls.append((path, payload))
        if path.startswith("commits/"):
            return {"sha": "a" * 40}
        if "/runs?" in path:
            return {"workflow_runs": [original]}
        if path.endswith("/dispatches"):
            return {"workflow_run_id": 43}
        return {**original, "id": int(path.rsplit("/", 1)[1])}

    monkeypatch.setattr(dependencies, "_github", github)
    assert (
        dependencies._dispatch("inference-models", version="2", wait_seconds=20) == 42
    )
    assert not any(path.endswith("/dispatches") for path, _ in calls)


def test_sdk_publication_smokes_retained_wheel_in_existing_image():
    workflow = yaml.load(
        (ROOT / ".github/workflows/publish.pypi.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    steps = workflow["jobs"]["publish_sdk"]["steps"]
    smoke = next(
        i
        for i, step in enumerate(steps)
        if "run_sdk_tests.sh retained" in step.get("run", "")
    )
    upload = next(
        i for i, step in enumerate(steps) if step.get("uses", "").startswith("pypa/")
    )
    assert smoke < upload
    assert steps[smoke].get("continue-on-error", "false") == "false"


@pytest.fixture
def retained(tmp_path, monkeypatch):
    import candidate

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOURCE_REVISION", "a" * 40)
    monkeypatch.setenv("GITHUB_REPOSITORY", "roboflow/inference")
    monkeypatch.setenv("GITHUB_RUN_ID", "100")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    directory = tmp_path / "dist/sdk"
    directory.mkdir(parents=True)
    (directory / "sdk.whl").write_bytes(b"original wheel bytes")
    return candidate


def test_retry_restores_identical_bytes_without_building(retained, monkeypatch):
    import shutil

    retained._seal("sdk")
    shutil.rmtree("dist")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "7")
    retained._restore("sdk")
    assert Path("dist/sdk/sdk.whl").read_bytes() == b"original wheel bytes"
    retained._verify("sdk", restored=True)


@pytest.mark.parametrize("change", ["bytes", "missing", "extra", "symlink"])
def test_corrupt_retained_candidate_cannot_restore(retained, change):
    import shutil

    retained._seal("sdk")
    shutil.rmtree("dist")
    wheel = Path("release-candidate/sdk/files/sdk/sdk.whl")
    if change == "bytes":
        wheel.write_bytes(b"different wheel")
    elif change == "missing":
        wheel.unlink()
    elif change == "extra":
        wheel.with_name("extra.whl").write_bytes(b"extra")
    else:
        wheel.unlink()
        wheel.symlink_to("/etc/hostname")
    with pytest.raises(ValueError):
        retained._restore("sdk")
    assert not Path("dist").exists()


@pytest.mark.parametrize(
    "variable", ["SOURCE_REVISION", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"]
)
def test_wrong_candidate_identity_fails(retained, monkeypatch, variable):
    retained._seal("sdk")
    monkeypatch.setenv(variable, "different")
    with pytest.raises(ValueError, match="identity"):
        retained._verify("sdk")


def test_smoke_cannot_change_the_upload_candidate(retained):
    retained._seal("sdk")
    Path("dist/sdk/sdk.whl").write_bytes(b"changed after testing")
    with pytest.raises(ValueError, match="SHA-256"):
        retained._verify("sdk", restored=True)


@pytest.mark.parametrize("attempt", ["1", "2", "9"])
def test_artifact_selection_reuses_original_id(retained, monkeypatch, attempt):
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", attempt)
    monkeypatch.setattr(
        dependencies,
        "_github",
        lambda *args, **kwargs: {
            "artifacts": [{"name": "candidate-sdk", "id": 17, "expired": False}]
        },
    )
    assert retained._select("sdk") == {
        "name": "candidate-sdk",
        "id": 17,
        "reused": True,
    }


@pytest.mark.parametrize(
    "problem", ["expired", "duplicate", "deleted", "interrupted", "missing-history"]
)
def test_missing_or_ambiguous_artifact_never_rebuilds(retained, monkeypatch, problem):
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    artifact = {"name": "candidate-sdk", "id": 17, "expired": problem == "expired"}
    artifacts = (
        [artifact]
        if problem == "expired"
        else [artifact, artifact] if problem == "duplicate" else []
    )
    step = {
        "name": "Retain candidate sdk",
        "status": "in_progress" if problem == "interrupted" else "completed",
        "conclusion": None if problem == "interrupted" else "success",
    }

    def github(path, **kwargs):
        return (
            {"artifacts": artifacts}
            if "/artifacts?" in path
            else {
                "jobs": (
                    []
                    if problem == "missing-history"
                    else [{"run_attempt": 1, "steps": [step]}]
                )
            }
        )

    monkeypatch.setattr(dependencies, "_github", github)
    with pytest.raises(ValueError):
        retained._select("sdk")


def test_failed_build_before_retention_can_be_retried(retained, monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")

    def github(path, **kwargs):
        return (
            {"artifacts": []}
            if "/artifacts?" in path
            else {
                "jobs": [
                    {
                        "run_attempt": 1,
                        "steps": [
                            {
                                "name": "Retain candidate sdk",
                                "status": "completed",
                                "conclusion": "skipped",
                            }
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr(dependencies, "_github", github)
    assert retained._select("sdk")["reused"] is False


def test_candidate_lookup_paginates(retained, monkeypatch):
    calls = []

    def github(path, **kwargs):
        calls.append(path)
        return {
            "artifacts": (
                [{"name": "unrelated"}] * 100
                if path.endswith("&page=1")
                else [{"name": "candidate-sdk", "id": 17, "expired": False}]
            )
        }

    monkeypatch.setattr(dependencies, "_github", github)
    assert retained._select("sdk")["id"] == 17
    assert len(calls) == 2


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "timed_out"])
def test_parent_retry_resumes_failed_child_and_waits_for_new_attempt(
    monkeypatch, conclusion
):
    monkeypatch.setenv("SOURCE_REVISION", "a" * 40)
    monkeypatch.setenv("GITHUB_REF", "refs/heads/moved-branch")
    monkeypatch.setenv("GITHUB_RUN_ID", "100")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    original = {
        "id": 42,
        "head_sha": "a" * 40,
        "path": ".github/workflows/publish.pypi.inference_exp.yml",
        "event": "workflow_dispatch",
        "display_title": "pypi-inference-models-parent-100",
        "status": "completed",
        "conclusion": conclusion,
        "run_attempt": 1,
    }
    polls = iter(
        [
            original,
            {**original, "status": "in_progress", "run_attempt": 2},
            {**original, "conclusion": "success", "run_attempt": 2},
        ]
    )
    posts = []

    def github(path, *, payload=None, **kwargs):
        if "/runs?" in path:
            return {"workflow_runs": [original]}
        if payload is not None:
            posts.append((path, payload))
            return {}
        assert path == "actions/runs/42"
        return next(polls)

    monkeypatch.setattr(dependencies, "_github", github)
    monkeypatch.setattr(dependencies.time, "sleep", lambda seconds: None)
    assert (
        dependencies._dispatch("inference-models", version="2", wait_seconds=20) == 42
    )
    assert posts == [("actions/runs/42/rerun-failed-jobs", {})]


def test_sdk_smoke_requires_single_wheel_without_installing(tmp_path):
    import sdk_smoke

    with pytest.raises(ValueError, match="exactly one"):
        sdk_smoke.smoke(tmp_path)
    (tmp_path / "one.whl").write_bytes(b"one")
    (tmp_path / "two.whl").write_bytes(b"two")
    with pytest.raises(ValueError, match="exactly one"):
        sdk_smoke.smoke(tmp_path)


def _first_party_requirements(requirements):
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    return {
        (canonicalize_name(req.name), str(req.specifier), tuple(sorted(req.extras)))
        for req in map(Requirement, requirements)
        if canonicalize_name(req.name) in dependencies.OWNED
    }


def _assert_floor_contract(project, locked, wheel):
    declared = _first_party_requirements(
        project["dependencies"]
        + [
            requirement
            for requirements in project.get("optional-dependencies", {}).values()
            for requirement in requirements
        ]
    )
    lock_requirements = [
        entry["name"]
        + ("[" + ",".join(entry["extras"]) + "]" if entry.get("extras") else "")
        + entry.get("specifier", "")
        for entry in locked["metadata"]["requires-dist"]
    ]
    assert _first_party_requirements(lock_requirements) == declared
    assert _first_party_requirements(wheel["requires_dist"]) == declared


@pytest.mark.parametrize("profile", ["roboflow-workflows", "streamvision"])
def test_real_dependency_floors_match_manifest_lock_and_built_wheel(tmp_path, profile):
    import shutil
    import subprocess
    import tomllib

    directory = dependencies.PROJECTS[profile][0]
    source = ROOT / directory
    project = tomllib.loads((source / "pyproject.toml").read_text())["project"]
    lock = tomllib.loads((source / "uv.lock").read_text())
    locked = next(p for p in lock["package"] if p["name"] == profile)
    shutil.copytree(
        source,
        tmp_path / directory,
        ignore=shutil.ignore_patterns(".venv", "dist", "build", "*.egg-info", "tests"),
    )
    subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--offline",
            "--no-build-isolation",
            "--out-dir",
            str(tmp_path / "dist"),
        ],
        cwd=tmp_path / directory,
        check=True,
        capture_output=True,
    )
    wheel_path = next((tmp_path / "dist").glob("*.whl"))
    wheel = dependencies._wheel(wheel_path.read_bytes(), filename=wheel_path.name)
    _assert_floor_contract(project, locked, wheel)


@pytest.mark.parametrize("floor", [">=0.39.9rc1", ">=0.39.10", ">=99.4"])
def test_updated_floors_are_allowed_but_stale_lock_or_wheel_fails(floor):
    project = {"dependencies": ["inference-models" + floor]}
    locked = {
        "metadata": {
            "requires-dist": [{"name": "inference-models", "specifier": floor}]
        }
    }
    wheel = {"requires_dist": ["inference-models" + floor]}
    _assert_floor_contract(project, locked, wheel)
    with pytest.raises(AssertionError):
        _assert_floor_contract(
            project, locked, {"requires_dist": ["inference-models>=0.1"]}
        )
    locked["metadata"]["requires-dist"][0]["specifier"] = ">=0.1"
    with pytest.raises(AssertionError):
        _assert_floor_contract(project, locked, wheel)


def test_missing_previously_started_child_cannot_redispatch(monkeypatch):
    monkeypatch.setenv("SOURCE_REVISION", "a" * 40)
    monkeypatch.setenv("GITHUB_REF", "refs/heads/candidate")
    monkeypatch.setenv("GITHUB_RUN_ID", "100")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")

    def github(path, *, payload=None, **kwargs):
        assert (
            payload is None
        ), "Never dispatch a replacement for missing publisher history"
        return {"workflow_runs": []}

    monkeypatch.setattr(dependencies, "_github", github)
    with pytest.raises(ValueError, match="Previously started publisher"):
        dependencies._dispatch("inference-models", version="2", started=True)


def test_dispatch_intent_survives_lost_post_response(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("SOURCE_REVISION", "a" * 40)
    monkeypatch.setenv("GITHUB_REF", "refs/heads/candidate")
    monkeypatch.setenv("GITHUB_RUN_ID", "100")
    plan = {
        "order": ["inference-models"],
        "packages": {"inference-models": {"action": "publish", "version": "2"}},
    }
    path = tmp_path / "plan.json"

    def github(api_path, *, payload=None, **kwargs):
        if "/runs?" in api_path:
            return {"workflow_runs": []}
        if api_path.startswith("commits/"):
            return {"sha": "a" * 40}
        assert (
            json.loads(path.read_text())["packages"]["inference-models"][
                "publisher_started"
            ]
            is True
        )
        raise ValueError("lost dispatch response")

    monkeypatch.setattr(dependencies, "_github", github)
    with pytest.raises(ValueError, match="lost dispatch response"):
        dependencies._coordinate(plan, plan_file=path)
    assert (
        json.loads(path.read_text())["packages"]["inference-models"][
            "publisher_started"
        ]
        is True
    )


def test_checkpoint_keeps_partial_progress_and_rejects_replanning():
    import copy

    original = {
        "source_revision": "a" * 40,
        "verification_run_id": "100",
        "packages": {"inference-models": {"action": "publish", "version": "2"}},
    }
    progress = copy.deepcopy(original)
    progress["packages"]["inference-models"].update(
        publisher_started=True, publisher_started_run_id=42
    )
    assert dependencies._resume_plan(original, progress) == progress
    progress["packages"]["inference-models"]["version"] = "3"
    with pytest.raises(ValueError, match="changed"):
        dependencies._resume_plan(original, progress)


@pytest.mark.parametrize("checkpoint", ["missing", "expired", "available"])
def test_coordinator_requires_checkpoint_after_started_attempt(
    retained, monkeypatch, checkpoint
):
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "3")

    def github(path, **kwargs):
        return {
            "jobs": [
                {
                    "run_attempt": 2,
                    "steps": [
                        {
                            "name": "Publish missing upstream versions in dependency order",
                            "status": "completed",
                            "conclusion": "cancelled",
                        }
                    ],
                }
            ]
        }

    monkeypatch.setattr(dependencies, "_github", github)
    artifacts = (
        []
        if checkpoint == "missing"
        else [
            {"name": "dependency-plan-2", "id": 55, "expired": checkpoint == "expired"}
        ]
    )
    if checkpoint == "available":
        assert retained._checkpoint(artifacts, run_id="100", deadline=9999999999) == 55
    else:
        with pytest.raises(ValueError, match="checkpoint"):
            retained._checkpoint(artifacts, run_id="100", deadline=9999999999)
