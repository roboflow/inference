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


@pytest.mark.parametrize("new_execution", [False, True])
@pytest.mark.parametrize("new_checkpoint", [False, True])
def test_checkpoint_uses_execution_attempt_not_copied_job_attempt(
    retained, monkeypatch, new_execution, new_checkpoint
):
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "4")
    job = {
        "name": "prepare_dependencies",
        "started_at": "2026-11-21T00:00:52Z",
        "run_attempt": 1,
        "steps": [
            {
                "name": "Publish missing upstream versions in dependency order",
                "status": "completed",
                "conclusion": "success",
            }
        ],
    }
    # GitHub gives copied jobs different IDs and later attempt numbers, but
    # preserves their start time. Put the original on the second API page.
    later = {**job, "id": 200, "run_attempt": 2}
    if new_execution:
        later["started_at"] = "2026-11-21T01:00:52Z"
    jobs = [{**later, "id": 300, "run_attempt": 3}, later]
    jobs += [{"run_attempt": 1, "steps": []}] * 98
    calls = []

    def github(path, **kwargs):
        calls.append(path)
        return {"jobs": jobs if path.endswith("page=1") else [job]}

    monkeypatch.setattr(dependencies, "_github", github)
    artifacts = [{"name": "dependency-plan-1", "id": 51, "expired": False}]
    if new_checkpoint:
        artifacts.append({"name": "dependency-plan-2", "id": 52, "expired": False})
    if new_execution and not new_checkpoint:
        with pytest.raises(ValueError, match="dependency-plan-2"):
            retained._checkpoint(artifacts, run_id="100", deadline=9999999999)
    else:
        assert retained._checkpoint(artifacts, run_id="100", deadline=9999999999) == (
            52 if new_execution else 51
        )
    assert len(calls) == 2


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "timed_out"])
@pytest.mark.parametrize("earlier_success", [False, True])
def test_failed_retention_without_artifact_can_retry_unless_previously_retained(
    retained, monkeypatch, conclusion, earlier_success
):
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "3")
    step = {
        "name": "Retain candidate sdk",
        "status": "completed",
        "conclusion": conclusion,
    }
    jobs = [{"run_attempt": 2, "steps": [step]}]
    if earlier_success:
        jobs.append({"run_attempt": 1, "steps": [{**step, "conclusion": "success"}]})
    monkeypatch.setattr(
        dependencies,
        "_github",
        lambda path, **kwargs: (
            {"artifacts": []} if "/artifacts?" in path else {"jobs": jobs}
        ),
    )
    if earlier_success:
        with pytest.raises(ValueError, match="do not rebuild"):
            retained._select("sdk")
    else:
        assert retained._select("sdk")["reused"] is False


def test_fork_candidate_lookup_is_read_only(retained, monkeypatch):
    import io

    monkeypatch.setenv("GITHUB_REPOSITORY", "contributor/inference")
    monkeypatch.setenv("GH_TOKEN", "test-only")
    requests = []

    def open_url(request, **kwargs):
        requests.append((request.full_url, request.get_method()))
        return io.BytesIO(b'{"artifacts": []}')

    monkeypatch.setattr(dependencies.urllib.request, "urlopen", open_url)
    assert retained._select("sdk")["reused"] is False
    assert requests == [
        (
            "https://api.github.com/repos/contributor/inference/actions/runs/100/artifacts?&per_page=100&page=1",
            "GET",
        )
    ]
    with pytest.raises(ValueError, match="restricted"):
        dependencies._github(
            "actions/workflows/publisher.yml/dispatches",
            payload={},
            deadline=9999999999,
        )
    assert len(requests) == 1


def test_candidate_rejects_symlink_even_with_identical_bytes(retained):
    wheel = Path("dist/sdk/sdk.whl")
    target = Path("original.whl").resolve()
    target.write_bytes(wheel.read_bytes())
    wheel.unlink()
    wheel.symlink_to(target)
    with pytest.raises(ValueError, match="Symlink"):
        retained._seal("sdk")


def test_empty_candidate_cannot_be_sealed(retained):
    Path("dist/sdk/sdk.whl").unlink()
    with pytest.raises(ValueError, match="Empty candidate"):
        retained._seal("sdk")


def test_restore_preserves_existing_directory(retained):
    retained._seal("sdk")
    Path("dist/sdk/sdk.whl").write_bytes(b"unrelated existing bytes")
    with pytest.raises(ValueError, match="overwrite"):
        retained._restore("sdk")
    assert Path("dist/sdk/sdk.whl").read_bytes() == b"unrelated existing bytes"


@pytest.mark.parametrize(
    "field,value",
    [
        ("head_sha", "b" * 40),
        ("path", ".github/workflows/unrelated.yml"),
        ("event", "push"),
        ("display_title", "pypi-inference-models-parent-999"),
    ],
)
def test_supplied_publisher_identity_is_checked_before_rerun(monkeypatch, field, value):
    monkeypatch.setenv("SOURCE_REVISION", "a" * 40)
    monkeypatch.setenv("GITHUB_REF", "refs/heads/candidate")
    monkeypatch.setenv("GITHUB_RUN_ID", "100")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    original = {
        "id": 42,
        "head_sha": "a" * 40,
        "path": ".github/workflows/publish.pypi.inference_exp.yml",
        "event": "workflow_dispatch",
        "display_title": "pypi-inference-models-parent-100",
        "status": "completed",
        "conclusion": "failure",
        "run_attempt": 1,
        field: value,
    }
    calls = []

    def github(path, *, payload=None, **kwargs):
        calls.append(path)
        assert payload is None, "Reject another publisher before requesting a rerun"
        return original

    monkeypatch.setattr(dependencies, "_github", github)
    with pytest.raises(ValueError, match="identity"):
        dependencies._dispatch("inference-models", version="2", run_id=42)
    assert calls == ["actions/runs/42"]


@pytest.mark.parametrize(
    "change",
    [
        "top-level",
        "packages",
        "unexpected",
        "reuse-progress",
        "id",
        "bool-id",
        "started-type",
    ],
)
def test_checkpoint_rejects_changes_outside_valid_publisher_progress(change):
    import copy

    original = {
        "source_revision": "a" * 40,
        "order": ["inference-models"],
        "packages": {"inference-models": {"action": "publish", "version": "2"}},
    }
    progress = copy.deepcopy(original)
    entry = progress["packages"]["inference-models"]
    if change == "top-level":
        progress["source_revision"] = "b" * 40
    elif change == "packages":
        progress["packages"]["streamvision"] = entry.copy()
    elif change == "unexpected":
        entry["arbitrary"] = True
    elif change == "reuse-progress":
        original["packages"]["inference-models"]["action"] = "reuse"
        entry.update(action="reuse", publisher_run_id=42)
    elif change == "started-type":
        entry["publisher_started"] = "false"
    else:
        entry["publisher_started_run_id"] = True if change == "bool-id" else -42
    with pytest.raises(ValueError):
        dependencies._resume_plan(original, progress)


@pytest.mark.parametrize("case", ["unrelated", "ambiguous", "first-attempt"])
def test_publisher_lookup_does_not_adopt_or_rerun_wrong_execution(monkeypatch, case):
    monkeypatch.setenv("SOURCE_REVISION", "a" * 40)
    monkeypatch.setenv("GITHUB_REF", "refs/heads/candidate")
    monkeypatch.setenv("GITHUB_RUN_ID", "100")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1" if case == "first-attempt" else "2")
    original = {
        "id": 42,
        "head_sha": "a" * 40,
        "path": ".github/workflows/publish.pypi.inference_exp.yml",
        "event": "workflow_dispatch",
        "display_title": "pypi-inference-models-parent-100",
        "status": "completed",
        "conclusion": "failure",
        "run_attempt": 1,
    }
    runs = [original]
    if case == "ambiguous":
        runs.append({**original, "id": 43})
    elif case == "unrelated":
        runs = [{**original, "display_title": "manual publish"}]

    def github(path, *, payload=None, **kwargs):
        assert payload is None, "No mutation allowed for ambiguous or missing history"
        assert "/runs?" in path
        return {"workflow_runs": runs}

    monkeypatch.setattr(dependencies, "_github", github)
    with pytest.raises(ValueError, match="missing|Ambiguous|rerun it"):
        dependencies._dispatch("inference-models", version="2", started=True)


@pytest.mark.parametrize("run_id", [True, False, 0, -1, "42", "../../runs/42"])
def test_invalid_checkpoint_run_id_is_rejected_before_network(monkeypatch, run_id):
    monkeypatch.setenv("SOURCE_REVISION", "a" * 40)
    monkeypatch.setenv("GITHUB_REF", "refs/heads/candidate")
    monkeypatch.setattr(
        dependencies,
        "_github",
        lambda *args, **kwargs: pytest.fail("Invalid ID reached network"),
    )
    with pytest.raises(ValueError, match="run ID"):
        dependencies._dispatch("inference-models", version="2", run_id=run_id)


@pytest.mark.parametrize(
    "failure", ["unsent", 401, 403, 404, 422, 408, 429, 503, "lost-response"]
)
def test_dispatch_intent_cleared_only_when_request_definitely_not_started(
    tmp_path, monkeypatch, failure
):
    import io
    import json
    import urllib.error

    monkeypatch.setenv("GH_TOKEN", "test-only")
    monkeypatch.setenv("GITHUB_REPOSITORY", "roboflow/inference")
    monkeypatch.setenv("SOURCE_REVISION", "a" * 40)
    monkeypatch.setenv("GITHUB_REF", "refs/heads/candidate")
    monkeypatch.setenv("GITHUB_RUN_ID", "100")
    clock = [0]
    monkeypatch.setattr(dependencies.time, "monotonic", lambda: clock[0])
    path = tmp_path / "plan.json"
    plan = {
        "order": ["inference-models"],
        "packages": {"inference-models": {"action": "publish", "version": "2"}},
    }
    write = dependencies.release._write_record

    def write_record(path, **record):
        write(path, **record)
        if failure == "unsent" and record["packages"]["inference-models"].get(
            "publisher_started"
        ):
            clock[0] = 21

    monkeypatch.setattr(dependencies.release, "_write_record", write_record)
    posts = []

    def open_url(request, **kwargs):
        if request.get_method() == "GET":
            body = (
                {"workflow_runs": []}
                if "/runs?" in request.full_url
                else {"sha": "a" * 40}
            )
            return io.BytesIO(json.dumps(body).encode())
        posts.append(request.full_url)
        assert (
            json.loads(path.read_text())["packages"]["inference-models"][
                "publisher_started"
            ]
            is True
        )
        if failure == "lost-response":
            raise ConnectionResetError("lost")
        assert (
            failure != "unsent"
        ), "Deadline must be checked before opening the request"
        raise urllib.error.HTTPError(request.full_url, failure, "fixture", None, None)

    monkeypatch.setattr(dependencies.urllib.request, "urlopen", open_url)
    with pytest.raises(ValueError):
        dependencies._coordinate(plan, deadline=20, plan_file=path)
    expected_started = failure in {408, 429, 503, "lost-response"}
    assert (
        plan["packages"]["inference-models"].get("publisher_started", False)
        is expected_started
    )
    saved = json.loads(path.read_text())
    assert saved["packages"] == plan["packages"]
    assert saved["order"] == plan["order"]
    assert len(posts) == (0 if failure == "unsent" else 1)


def test_sdk_smoke_checks_digest_after_install_and_import(tmp_path, monkeypatch):
    import zipfile

    import sdk_smoke

    wheel = tmp_path / "inference_sdk-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "inference_sdk-1.0.dist-info/METADATA",
            "Name: inference-sdk\nVersion: 1.0\n",
        )
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if "-I" in argv:
            wheel.write_bytes(wheel.read_bytes() + b"changed during import")

    monkeypatch.setattr(sdk_smoke.subprocess, "run", run)
    with pytest.raises(ValueError, match="changed during smoke"):
        sdk_smoke.smoke(tmp_path)
    assert len(calls) == 3


@pytest.mark.parametrize(
    "mode,target",
    [
        ("unit", "sdk-tests"),
        ("integration", "sdk-tests"),
        ("e2e", "sdk-e2e"),
        ("install", "sdk-install"),
        ("retained", "base"),
    ],
)
def test_sdk_launcher_preserves_docker_arguments(tmp_path, mode, target):
    import json
    import os
    import subprocess

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "docker.jsonl"
    docker = fake_bin / "docker"
    docker.write_text(
        f"#!{sys.executable}\nimport json, sys\nwith open({str(log)!r}, 'a') as log: log.write(json.dumps(sys.argv[1:]) + '\\n')\nif sys.argv[1:3] == ['image', 'inspect']: print('sha256:fixture')\n"
    )
    git = fake_bin / "git"
    git.write_text(
        f"#!/bin/sh\nif [ \"$2\" = --show-toplevel ]; then printf '%s\\n' '{ROOT}'; else printf '%s\\n' fixture; fi\n"
    )
    docker.chmod(0o755)
    git.chmod(0o755)
    subprocess.run(
        [
            os.environ.get("SDK_TEST_BASH", "bash"),
            str(ROOT / ".github/scripts/run_sdk_tests.sh"),
            mode,
        ],
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
        check=True,
        capture_output=True,
        text=True,
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0][calls[0].index("--target") + 1] == target
    run = calls[-1]
    assert run[:3] == ["run", "--rm", "--shm-size=1g"]
    expected_options = (
        ["-v", f"{ROOT}:/work:ro", "-w", "/tmp"] if mode == "retained" else []
    )
    assert run[3 : 4 + len(expected_options)] == [*expected_options, "sha256:fixture"]
    assert run[4 + len(expected_options)] == "python"
