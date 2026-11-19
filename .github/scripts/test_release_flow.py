"""Behavioral boundaries for the one-run release pipeline."""

import importlib.util
import json
import os
import subprocess
import sys
import threading
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".release/pypi"))
import flow


def _load_images():
    spec = importlib.util.spec_from_file_location(
        "release_images", ROOT / ".github/scripts/release_images.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(path, *args):
    return subprocess.run(
        ["git", "-C", str(path), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-b", "release-candidates-9.9.0rc1")
    _git(source, "config", "user.name", "Fixture")
    _git(source, "config", "user.email", "fixture@example.invalid")
    for lock in flow.LOCKS:
        path = source / lock
        path.parent.mkdir(parents=True)
        path.write_text("version = 1\nrevision = 3\n")
    (source / "app.py").write_text("answer = 42\n")
    _git(source, "add", ".")
    _git(source, "commit", "-m", "Fixture")
    sha = _git(source, "rev-parse", "HEAD")
    remote = tmp_path / "remote.git"
    subprocess.run(
        ["git", "clone", "--bare", str(source), str(remote)],
        check=True,
        capture_output=True,
    )
    monkeypatch.chdir(source)
    monkeypatch.setenv("GH_TOKEN", "fixture-token")
    original_git = flow._git

    def local_git(*args, env=None):
        args = tuple(
            str(remote) if arg == "https://github.com/roboflow/inference.git" else arg
            for arg in args
        )
        return original_git(*args, env=env)

    monkeypatch.setattr(flow, "_git", local_git)
    state = {
        "channel": "rc",
        "version": "9.9.0rc1",
        "source_sha": sha,
        "trigger_sha": sha,
        "application": flow._fingerprint(include_locks=False),
        "branch": "release-candidates-9.9.0rc1",
        "run_id": "123",
    }
    flow._write(flow.STATE / "release.json", state)
    flow._save_locks()
    return source, remote, state


def test_lock_commit_targets_release_branch_and_has_no_ci_marker(candidate):
    source, remote, state = candidate
    locks = json.loads((flow.STATE / "locks.json").read_text())
    locks[flow.LOCKS[-1]] += 'requires-python = ">=3.10"\n'
    flow._write(flow.STATE / "locks.json", locks)
    flow._commit_locks()
    new = flow._state()["source_sha"]
    assert new != state["source_sha"]
    assert _git(remote, "rev-parse", state["branch"]) == new
    assert _git(source, "show", "-s", "--format=%P", new) == state["source_sha"]
    assert "[skip ci]" in _git(source, "show", "-s", "--format=%B", new)
    assert (
        _git(source, "diff", "--name-only", state["source_sha"], new) == flow.LOCKS[-1]
    )
    # A failed artifact upload can rerun the commit job from the original SHA.
    _git(source, "checkout", "--detach", state["source_sha"])
    flow._write(flow.STATE / "release.json", state)
    flow._commit_locks()
    assert flow._state()["source_sha"] == new
    assert _git(remote, "rev-list", "--count", state["branch"]) == "2"


def test_empty_lock_update_does_not_create_commit(candidate):
    _, remote, state = candidate
    flow._commit_locks()
    assert flow._state()["source_sha"] == state["source_sha"]
    assert _git(remote, "rev-list", "--count", state["branch"]) == "1"


@pytest.mark.parametrize("fault", [None, "lock", "application", "commit"])
def test_event_checkout_restores_exact_generated_release_tree(candidate, fault):
    source, _, state = candidate
    locks = json.loads((flow.STATE / "locks.json").read_text())
    locks[flow.LOCKS[-1]] += 'requires-python = ">=3.10"\n'
    flow._write(flow.STATE / "locks.json", locks)
    flow._commit_locks()
    committed_tree = _git(source, "rev-parse", "HEAD^{tree}")
    _git(source, "checkout", "--detach", state["trigger_sha"])
    if fault == "lock":
        locks[flow.LOCKS[-1]] += "corrupted = true\n"
        flow._write(flow.STATE / "locks.json", locks)
    elif fault in {"application", "commit"}:
        Path("app.py").write_text("answer = 0\n")
        if fault == "commit":
            _git(source, "commit", "-am", "Unrelated commit")
    if fault:
        with pytest.raises(ValueError, match="Checkout is not|Restored source differs"):
            flow._restore_source()
    else:
        flow._restore_source()
        assert _git(source, "write-tree") == committed_tree
        assert _git(source, "rev-parse", "HEAD") == state["trigger_sha"]


def test_branch_movement_is_not_overwritten(candidate):
    source, remote, state = candidate
    other = source.parent / "other"
    subprocess.run(
        ["git", "clone", str(remote), str(other)], check=True, capture_output=True
    )
    _git(other, "config", "user.name", "Other")
    _git(other, "config", "user.email", "other@example.invalid")
    (other / "app.py").write_text("answer = 99\n")
    _git(other, "commit", "-am", "Concurrent change")
    _git(other, "push", "origin", state["branch"])
    moved = _git(other, "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="changed concurrently"):
        flow._commit_locks()
    assert _git(remote, "rev-parse", state["branch"]) == moved


def test_application_edit_cannot_hide_in_lock_commit(candidate):
    Path("app.py").write_text("answer = 99\n")
    with pytest.raises(ValueError, match="Application changed"):
        flow._commit_locks()


@pytest.mark.parametrize(
    "attempts,failures,expected,count", [(3, 2, 0, 3), (3, 3, 7, 3), (1, 1, 7, 1)]
)
def test_real_retry_wrapper_retains_attempts(
    tmp_path, attempts, failures, expected, count
):
    counter = tmp_path / "counter"
    command = (
        f'n=0; if [ -f "{counter}" ]; then n=$(cat "{counter}"); fi; '
        f'n=$((n+1)); echo "$n" > "{counter}"; echo "attempt $n"; '
        f'if [ "$n" -le {failures} ]; then exit 7; fi'
    )
    result = subprocess.run(
        ["bash", str(ROOT / ".github/scripts/retry.sh")],
        env={
            **os.environ,
            "RETRY_COMMAND": command,
            "RETRY_ATTEMPTS": str(attempts),
            "RETRY_LOG_DIR": str(tmp_path / "logs"),
            "RETRY_DELAY_SECONDS": "0",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == expected
    assert int(counter.read_text()) == count
    assert not list((tmp_path / "logs").glob("attempt-*.log"))
    assert len((tmp_path / "logs/attempts.txt").read_text().splitlines()) == count


def test_retry_does_not_swallow_cancellation(tmp_path):
    result = subprocess.run(
        ["bash", str(ROOT / ".github/scripts/retry.sh")],
        env={
            **os.environ,
            "RETRY_COMMAND": "exit 130",
            "RETRY_ATTEMPTS": "3",
            "RETRY_LOG_DIR": str(tmp_path / "logs"),
            "RETRY_DELAY_SECONDS": "0",
        },
        capture_output=True,
    )
    assert result.returncode == 130
    assert (tmp_path / "logs/attempts.txt").read_text() == "1 130\n"


def test_equivalence_only_normalizes_release_metadata():
    rc = b'__version__ = "1.2.3rc2"\ndef guard():\n    return False\n'
    stable = rc.replace(b"1.2.3rc2", b"1.2.3")
    assert flow._canonical("inference/core/version.py", rc) == flow._canonical(
        "inference/core/version.py", stable
    )
    assert flow._canonical("inference/core/version.py", rc) != flow._canonical(
        "inference/core/version.py", stable.replace(b"False", b"True")
    )
    assert (
        flow._requirement("inference-models>=0.39.9rc2; python_version>='3.10'")
        == 'inference-models>=0.39.9; python_version >= "3.10"'
    )
    assert flow._requirement("numpy==2.0.0rc1") == "numpy==2.0.0rc1"


def test_lock_equivalence_keeps_third_party_hashes_and_sources():
    rc = {
        "package": [
            {
                "name": "inference-models",
                "version": "1.2.3rc1",
                "wheels": [{"hash": "sha256:a"}],
            },
            {"name": "numpy", "version": "2.0.0", "wheels": [{"hash": "sha256:b"}]},
        ]
    }
    final = json.loads(json.dumps(rc))
    final["package"][0].update(version="1.2.3", wheels=[{"hash": "sha256:c"}])
    assert flow._normalize(rc) == flow._normalize(final)
    final["package"][1]["wheels"][0]["hash"] = "sha256:changed"
    assert flow._normalize(rc) != flow._normalize(final)


@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped"])
def test_rc_gate_rejects_incomplete_suite(tmp_path, monkeypatch, result):
    monkeypatch.chdir(tmp_path)
    flow._write(
        flow.STATE / "release.json",
        {
            "channel": "rc",
            "source_sha": "new-lock-sha",
            "run_id": "1",
            "policy": {"rc_tests": ["unit.yml"]},
        },
    )
    monkeypatch.setenv(
        "RELEASE_RESULTS",
        json.dumps({"rc_unit": {"result": result}, "smoke": {"result": "success"}}),
    )
    writes = []

    def api(path, *, payload=None):
        if payload:
            writes.append(payload)
            return {}
        return {"jobs": [{"id": 1, "name": "rc_unit / tests", "conclusion": result}]}

    monkeypatch.setattr(flow, "_api", api)
    with pytest.raises(ValueError, match="validation failed"):
        flow._validate()
    assert writes[0]["head_sha"] == "new-lock-sha"
    assert writes[0]["conclusion"] == "failure"


def test_stable_requires_smoke_but_does_not_run_full_rc_suite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    flow._write(
        flow.STATE / "release.json",
        {"channel": "stable", "policy": {"rc_tests": ["unit.yml"]}},
    )
    monkeypatch.setenv(
        "RELEASE_RESULTS",
        json.dumps({"rc_unit": {"result": "skipped"}, "smoke": {"result": "success"}}),
    )
    flow._validate()
    monkeypatch.setenv(
        "RELEASE_RESULTS",
        json.dumps({"rc_unit": {"result": "skipped"}, "smoke": {"result": "failure"}}),
    )
    with pytest.raises(ValueError, match="smoke"):
        flow._validate()


def _wheel(index, name, version, requirements=()):
    filename = f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
    project = index / name
    project.mkdir(parents=True, exist_ok=True)
    file = project / filename
    info = f"{name.replace('-', '_')}-{version}.dist-info"
    with zipfile.ZipFile(file, "w") as archive:
        archive.writestr(
            info + "/METADATA",
            f"Metadata-Version: 2.3\nName: {name}\nVersion: {version}\n"
            + "".join(f"Requires-Dist: {req}\n" for req in requirements),
        )
        archive.writestr(
            info + "/WHEEL",
            "Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(info + "/RECORD", "")
    with (project / "index.html").open("a") as stream:
        stream.write(f'<a href="{filename}">{filename}</a>\n')


def test_real_uv_relocks_after_upstream_appears_and_rejects_third_party_drift(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    index = tmp_path / "index"
    _wheel(index, "inference-sdk", "1.0.0rc1")
    _wheel(
        index,
        "roboflow-workflows",
        "1.0.0rc1",
        ["fixture-dependency>=1", "inference-sdk>=1.0.0rc1"],
    )
    _wheel(index, "fixture-dependency", "1.0.0")
    handler = partial(SimpleHTTPRequestHandler, directory=str(index))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for path in flow.LOCKS:
            file = Path(path)
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text('version = 1\nrequires-python = ">=3.12"\n')
        project = Path("stream_vision/pyproject.toml")
        project.write_text(
            '[project]\nname = "streamvision"\nversion = "1.0.0rc1"\nrequires-python = ">=3.12"\n'
            'dependencies = ["roboflow-workflows==1.0.0rc1"]\n'
            f'[[tool.uv.index]]\nurl = "http://127.0.0.1:{server.server_port}"\ndefault = true\n'
        )
        monkeypatch.setattr(
            flow,
            "_versions",
            lambda: {
                "inference-sdk": "1.0.0rc2",
                "roboflow-workflows": next(
                    iter(
                        flow.Requirement(
                            flow.tomllib.loads(project.read_text())["project"][
                                "dependencies"
                            ][0]
                        ).specifier
                    )
                ).version,
            },
        )
        monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setenv("UV_PYTHON_DOWNLOADS", "never")
        monkeypatch.setenv("UV_NO_CACHE", "1")
        subprocess.run(
            ["uv", "lock"], cwd="stream_vision", check=True, capture_output=True
        )
        project.write_text(
            project.read_text().replace(
                "roboflow-workflows==1.0.0rc1", "roboflow-workflows==1.0.0rc2"
            )
        )
        with pytest.raises(subprocess.CalledProcessError):
            flow._prepare_lock("streamvision")
        _wheel(index, "inference-sdk", "1.0.0rc2")
        _wheel(
            index,
            "roboflow-workflows",
            "1.0.0rc2",
            ["fixture-dependency>=1", "inference-sdk>=1.0.0rc1"],
        )
        _wheel(index, "fixture-dependency", "2.0.0")
        flow._prepare_lock("streamvision")
        lock = Path("stream_vision/uv.lock").read_text()
        assert 'version = "1.0.0rc2"' in lock
        assert 'version = "1.0.0"' in lock
        assert 'version = "2.0.0"' not in lock
        selected = {
            item["name"]: item["version"]
            for item in flow.tomllib.loads(lock)["package"]
        }
        assert selected["inference-sdk"] == "1.0.0rc2"
        project.write_text(
            project.read_text().replace(
                "roboflow-workflows==1.0.0rc2", "roboflow-workflows==1.0.0rc3"
            )
        )
        _wheel(
            index,
            "roboflow-workflows",
            "1.0.0rc3",
            ["fixture-dependency>=2", "inference-sdk>=1.0.0rc1"],
        )
        with pytest.raises(ValueError, match="third-party dependency change"):
            flow._prepare_lock("streamvision")
    finally:
        server.shutdown()
        server.server_close()


def test_image_retry_reuses_digest_and_existing_completed_upload(tmp_path, monkeypatch):
    images = _load_images()
    monkeypatch.chdir(tmp_path)
    digest = "sha256:" + "a" * 64
    store = {"roboflow/test@" + digest: digest}
    plan = {
        "key": "cpu",
        "repository": "roboflow/test",
        "repositories": ["roboflow/test", "example.test/test"],
        "tag": "1.0.0rc1",
        "channel": "rc",
        "source_sha": "source",
        "platforms": ["linux/amd64"],
    }
    flow._write(
        Path("image-smoke/amd64/image-smoke.json"),
        {
            "key": "cpu",
            "digest": digest,
            "platform": "linux/amd64",
            "inventory": {"numpy": "2.0.0"},
        },
    )
    calls = []
    fail = [True]

    def inspect(reference, *, allow_missing=False):
        return store.get(reference)

    def copy(reference, *, target):
        assert reference == "roboflow/test@" + digest
        if target.startswith("example.test") and fail[0]:
            fail[0] = False
            raise RuntimeError("transient registry failure")
        calls.append(target)
        store[target] = digest

    monkeypatch.setattr(images, "_digest", inspect)
    monkeypatch.setattr(images, "_copy", copy)
    with pytest.raises(RuntimeError, match="transient"):
        images._promote(plan, digest=digest)
    images._promote(plan, digest=digest)
    assert calls == ["roboflow/test:1.0.0rc1", "example.test/test:1.0.0rc1"]
    assert json.loads(Path("image-receipt.json").read_text())["digest"] == digest
    store["roboflow/test:1.0.0rc1"] = "sha256:" + "b" * 64
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        images._promote(plan, digest=digest)


def test_artifact_redirect_strips_token_on_storage_host():
    request = flow.urllib.request.Request(
        "https://api.github.com/artifact",
        headers={"Authorization": "Bearer test-token"},
    )
    handler = flow._ArtifactRedirect()
    redirected = handler.redirect_request(
        request, None, 302, "Found", {}, "https://storage.example.test/signed"
    )
    assert redirected.get_header("Authorization") is None
    same_host = handler.redirect_request(
        request, None, 302, "Found", {}, "https://api.github.com/other"
    )
    assert same_host.get_header("Authorization") == "Bearer test-token"
    with pytest.raises(ValueError, match="HTTPS"):
        handler.redirect_request(
            request, None, 302, "Found", {}, "http://storage.example.test/signed"
        )


def test_successful_rc_receipt_is_found_after_more_than_one_artifact_page(
    candidate, monkeypatch
):
    _, _, state = candidate
    run = {
        "id": 12,
        "head_sha": state["source_sha"],
        "head_branch": state["branch"],
        "head_repository": {"full_name": flow.REPOSITORY},
    }
    receipt = {
        "schema": 1,
        "channel": "rc",
        "version": "1.2.3rc2",
        "run_id": "12",
        "trigger_sha": state["source_sha"],
        "source_sha": state["source_sha"],
        "branch": state["branch"],
        "repository": flow.REPOSITORY,
        "complete": True,
    }
    calls = []

    def api(path):
        calls.append(path)
        if "workflows/" in path:
            return {"workflow_runs": [run]}
        if path.endswith("page=1"):
            return {
                "artifacts": [
                    {"id": i, "name": f"test-logs-{i}", "expired": False}
                    for i in range(100)
                ]
            }
        return {
            "artifacts": [{"id": 101, "name": "release-receipt-1", "expired": False}]
        }

    monkeypatch.setattr(flow, "_api", api)
    monkeypatch.setattr(flow, "_download_receipt", lambda artifact_id: receipt)
    assert flow._find_rc("1.2.3") == receipt
    assert any(path.endswith("page=2") for path in calls)
    receipt["trigger_sha"] = "wrong"
    with pytest.raises(ValueError, match="Untrusted"):
        flow._find_rc("1.2.3")


@pytest.mark.parametrize(
    "steps", [[], [{"name": "Tests [release-test]", "conclusion": "skipped"}]]
)
def test_successful_caller_cannot_hide_skipped_test_steps(tmp_path, monkeypatch, steps):
    monkeypatch.chdir(tmp_path)
    flow._write(
        flow.STATE / "release.json",
        {
            "channel": "rc",
            "source_sha": "source",
            "run_id": "1",
            "policy": {"rc_tests": ["unit.yml"]},
        },
    )
    monkeypatch.setenv(
        "RELEASE_RESULTS",
        json.dumps({"rc_unit": {"result": "success"}, "smoke": {"result": "success"}}),
    )

    def api(path, *, payload=None):
        if payload:
            return {}
        return {
            "jobs": [
                {
                    "id": 1,
                    "name": "rc_unit / tests",
                    "conclusion": "success",
                    "steps": steps,
                }
            ]
        }

    monkeypatch.setattr(flow, "_api", api)
    with pytest.raises(ValueError, match="validation failed"):
        flow._validate()


def test_cross_registry_copy_preserves_all_platforms(monkeypatch):
    images = _load_images()
    calls = []
    monkeypatch.setattr(images, "_command", lambda *args: calls.append(args))
    images._copy(
        "roboflow/test@sha256:" + "a" * 64,
        target="us-docker.pkg.dev/project/repo/image:1.0.0",
    )
    assert calls[0][:5] == (
        "skopeo",
        "copy",
        "--all",
        "--preserve-digests",
        "--retry-times",
    )
    assert calls[0][-2] == "docker://roboflow/test@sha256:" + "a" * 64
    assert calls[0][-1] == "docker://us-docker.pkg.dev/project/repo/image:1.0.0"


def test_package_inventory_normalizes_only_first_party_rc_metadata():
    import smoke

    packages = [
        {"name": "inference_sdk", "version": "1.2.3rc1"},
        {"name": "numpy", "version": "2.0.0rc1"},
    ]
    assert smoke._inventory(packages) == {"inference-sdk": "1.2.3", "numpy": "2.0.0rc1"}


def test_release_receipt_requires_complete_package_and_image_smoke_evidence(candidate):
    _, _, state = candidate
    state["policy"] = {"images": [{"key": "cpu"}], "python_versions": ["3.12"]}
    flow._write(flow.STATE / "release.json", state)
    flow._write(
        flow.STATE / "images/cpu/image-receipt.json",
        {"key": "cpu", "source_sha": state["source_sha"]},
    )
    flow._write(flow.STATE / "dependency-receipt.json", {})
    flow._write(flow.STATE / "runtime-receipt.json", {})
    for name in flow.FIRST_PARTY:
        flow._write(flow.STATE / "smoke" / f"{name}-py3.12.json", {name: "1.0.0"})
    flow._receipt()
    receipt = json.loads((flow.STATE / "receipt.json").read_text())
    assert receipt["complete"] is True
    assert len(receipt["smoke"]) == len(flow.FIRST_PARTY)
    (flow.STATE / "smoke/inference-sdk-py3.12.json").unlink()
    with pytest.raises(ValueError, match="Missing package smoke"):
        flow._receipt()


@pytest.mark.parametrize(
    "version,branch,packages,channel",
    [
        ("9.9.0rc1", "release-candidates-9.9.0rc1", "all", "rc"),
        ("9.9.0", "release-9.9.0", "sdk", "sdk"),
        ("9.9.0", "release-9.9.0", "all", None),
        ("9.9.0rc1", "main", "all", None),
    ],
)
def test_manual_channel_requires_rc_release_branch(
    candidate, monkeypatch, version, branch, packages, channel
):
    _, _, state = candidate
    monkeypatch.setattr(flow.release, "_expected_version", lambda: version)
    monkeypatch.setattr(flow, "POLICY", ROOT / ".release/release-policy.json")
    for name, value in {
        "GITHUB_REPOSITORY": flow.REPOSITORY,
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_SHA": state["source_sha"],
        "GITHUB_REF": f"refs/heads/{branch}",
        "GITHUB_RUN_ID": "1",
        "PACKAGES": packages,
        "PUBLISH": "true",
    }.items():
        monkeypatch.setenv(name, value)
    flow._write(flow.STATE / "event.json", {})
    if channel:
        flow._prepare()
        assert flow._state()["channel"] == channel
    else:
        with pytest.raises(ValueError, match="requires an RC|requires a release"):
            flow._prepare()


@pytest.mark.parametrize(
    "candidate,final",
    [("1.8.5rc1.post1", "1.8.5.post1"), ("1.8.5-rc2.post3", "1.8.5-post3")],
)
def test_post_release_identity_preserves_post_number(candidate, final):
    assert flow._base(candidate) == flow._base(final)
    assert flow._base(candidate) != flow._base("1.8.5rc1")
    assert flow._base(candidate) != flow._base("1.8.5.post4")
    rc = f"inference-models=={candidate} # keep this pin aligned"
    stable = f"inference-models=={final} # keep this pin aligned"
    assert flow._requirement(rc) == flow._requirement(stable)
    assert flow._canonical(
        "inference/core/version.py", f'__version__ = "{candidate}"'.encode()
    ) == flow._canonical(
        "inference/core/version.py", f'__version__ = "{final}"'.encode()
    )
    assert flow._requirement(f"numpy=={candidate}") != flow._requirement(
        f"numpy=={final}"
    )


def test_post_smoke_inventory_does_not_accept_different_post_release():
    import smoke

    assert smoke._inventory(
        [{"name": "inference-sdk", "version": "1.8.5rc1.post1"}]
    ) == {"inference-sdk": "1.8.5.post1"}
    assert smoke._inventory([{"name": "inference-sdk", "version": "1.8.5.post2"}]) != {
        "inference-sdk": "1.8.5.post1"
    }


@pytest.mark.parametrize(
    "channel,runtime,models,exists,expected",
    [
        ("rc", "1.8.6rc1", "0.39.9", True, "cpu-1.8.6rc1"),
        ("stable", "1.8.6", "0.39.9", True, "cpu-0.39.9"),
        ("stable", "1.8.7", "0.39.9", True, "cpu-0.39.9"),
        ("stable", "1.8.7", "0.39.9", "moved", None),
        ("rc", "1.8.6rc1", "0.40.0rc1", False, "cpu-1.8.6rc1"),
        ("stable", "1.8.6", "0.40.0", False, "cpu-0.40.0"),
    ],
)
@pytest.mark.parametrize("image_key", ["models-cpu", "models-jp51"])
def test_models_plan_reuses_independent_version_with_retained_content_identity(
    tmp_path, monkeypatch, channel, runtime, models, exists, expected, image_key
):
    images = _load_images()
    monkeypatch.chdir(tmp_path)
    directory = Path("inference_models")
    directory.mkdir()
    (directory / "pyproject.toml").write_text(
        f'[project]\nname="inference-models"\nversion="{models}"\n'
    )
    package_directory = flow.STATE / "packages/inference-models"
    package_directory.mkdir(parents=True)
    wheel = package_directory / f"inference_models-{models}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("inference_models/__init__.py", "answer = 42\n")
        archive.writestr("inference_models/data/config.json", '{"safe": true}\n')
    policy = json.loads((ROOT / ".release/release-policy.json").read_text())
    item = next(image for image in policy["images"] if image["key"] == image_key)
    core_only = item.get("core_only", False)
    platform = item["platforms"][0]
    expected = expected.replace("cpu-", item["tag_prefix"]) if expected else None
    if core_only:
        wheel.unlink()  # The existing Jetson core contains no Models package.
    flow._write(
        flow.STATE / "release.json",
        {
            "channel": channel,
            "version": runtime,
            "source_sha": "verified-source",
            "run_id": "7",
            "policy": {"images": [item]},
        },
    )
    flow._write(
        flow.STATE / "accepted-rc.json",
        {
            "images": [
                {
                    "key": image_key,
                    "digest": "sha256:" + ("a" if exists is True else "b") * 64,
                    "smoke": {platform: {"inference-models": flow._base(models)}},
                }
            ]
        },
    )
    digest = "sha256:" + "a" * 64
    inspected = []

    def inspect(reference, *, allow_missing=False):
        inspected.append(reference)
        return digest if exists else None

    monkeypatch.setattr(images, "_digest", inspect)
    monkeypatch.setattr(
        images,
        "_stable_context",
        lambda plan, digest: ("final-context", "final-Dockerfile"),
    )
    if exists == "moved":
        with pytest.raises(ValueError, match="differs from the accepted RC digest"):
            images._plan(image_key)
        return
    images._plan(image_key)
    plan = json.loads(Path("image-plan.json").read_text())
    assert plan["tag"] == expected
    assert plan["reuse_digest"] == (
        digest
        if exists
        else "sha256:" + "b" * 64 if core_only and channel == "stable" else ""
    )
    if channel == "stable" and not exists and not core_only:
        assert plan["context"] == "final-context"
        assert plan["dockerfile"] == "final-Dockerfile"
        assert not plan["bake"]
    assert plan["models_version"] == models
    if core_only:
        assert "models_content_sha256" not in plan
    else:
        assert len(plan["models_content_sha256"]) == 64
    if flow.Version(models).is_prerelease:
        assert not inspected
    else:
        assert inspected == [f"roboflow/inference-exp:{item['tag_prefix']}{models}"]
    # A same-version content change is observable and cannot inherit the image's evidence.
    if not core_only:
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("inference_models/__init__.py", "answer = 0\n")
        assert images._models_identity(models) != plan["models_content_sha256"]
    if exists and channel == "stable":
        flow._write(
            Path("image-smoke/cpu/image-smoke.json"),
            {
                "key": image_key,
                "digest": digest,
                "platform": platform,
                "inventory": {"inference-models": models},
            },
        )
        monkeypatch.setattr(flow, "_api", lambda path: {"tag_name": "v9.0.0"})
        monkeypatch.setattr(
            images,
            "_copy",
            lambda *args, **kwargs: pytest.fail(
                "An unchanged Models tag must not be rebuilt or overwritten"
            ),
        )
        images._promote(plan, digest=digest)
        images._promote(plan, digest=digest)
        assert json.loads(Path("image-receipt.json").read_text())["reused"] is True


@pytest.mark.parametrize("fault", [None, "changed", "missing", "extra", "version"])
def test_reused_models_image_content_matches_retained_wheel(
    tmp_path, monkeypatch, fault
):
    from types import SimpleNamespace

    images = _load_images()
    spec = importlib.util.spec_from_file_location(
        "image_smoke", ROOT / ".github/scripts/release_image_smoke.py"
    )
    image_smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(image_smoke)
    monkeypatch.chdir(tmp_path)
    directory = flow.STATE / "packages/inference-models"
    directory.mkdir(parents=True)
    package = tmp_path / "installed/inference_models"
    package.mkdir(parents=True)
    with zipfile.ZipFile(
        directory / "inference_models-0.39.9-py3-none-any.whl", "w"
    ) as archive:
        for name, contents in {
            "__init__.py": "answer = 42\n",
            "data.json": "{}\n",
        }.items():
            archive.writestr("inference_models/" + name, contents)
            (package / name).write_text(contents)
    profile = {
        "models_version": "0.39.9",
        "models_content_sha256": images._models_identity("0.39.9"),
    }
    monkeypatch.setattr(
        image_smoke,
        "version",
        lambda name: "0.39.8" if fault == "version" else "0.39.9",
    )
    (package / "__pycache__").mkdir()
    (package / "__pycache__/generated.pyc").write_bytes(b"ignored interpreter cache")
    if fault == "changed":
        (package / "__init__.py").write_text("answer = 0\n")
    elif fault == "missing":
        (package / "data.json").unlink()
    elif fault == "extra":
        (package / "unexpected.py").write_text("raise RuntimeError()\n")
    if fault:
        with pytest.raises(AssertionError):
            image_smoke._verify_models_content(
                SimpleNamespace(__file__=str(package / "__init__.py")), profile
            )
    else:
        image_smoke._verify_models_content(
            SimpleNamespace(__file__=str(package / "__init__.py")), profile
        )


@pytest.mark.parametrize("fault", [None, "missing", "duplicate", "digest"])
def test_stable_image_build_uses_accepted_digest_and_offline_retained_wheels(
    tmp_path, monkeypatch, fault
):
    import shutil

    images = _load_images()
    monkeypatch.chdir(tmp_path)
    scripts = Path(".github/scripts")
    scripts.mkdir(parents=True)
    shutil.copy2(ROOT / ".github/scripts/release_image_finalize.py", scripts)
    wheels = flow.STATE / "packages"
    wheels.mkdir(parents=True)
    for name in flow.FIRST_PARTY:
        if fault == "missing" and name == "inference-sdk":
            continue
        (wheels / f"{name.replace('-', '_')}-1.2.3-py3-none-any.whl").write_bytes(
            b"retained"
        )
    if fault == "duplicate":
        (wheels / "inference_sdk-1.2.4-py3-none-any.whl").write_bytes(b"duplicate")
    Path("inference/core").mkdir(parents=True)
    Path("inference/core/version.py").write_text('__version__ = "1.2.3"\n')
    Path("inference/ignored.py").write_text("do not copy\n")
    digest = "sha256:" + "a" * 64 if fault != "digest" else "latest"
    plan = {"repository": "roboflow/test", "version_source": "runtime"}
    if fault:
        with pytest.raises(ValueError):
            images._stable_context(plan, digest)
        return
    context, dockerfile = images._stable_context(plan, digest)
    text = Path(dockerfile).read_text()
    assert text.startswith(f"FROM roboflow/test@{digest}\n")
    assert "RUN --network=none" in text
    assert (
        set(json.loads((Path(context) / "manifest.json").read_text()))
        == flow.FIRST_PARTY
    )
    assert (Path(context) / "vendor/pip/__main__.py").is_file()
    assert (
        Path(context) / "sources/inference/core/version.py"
    ).read_text() == '__version__ = "1.2.3"\n'
    assert not (Path(context) / "sources/inference/ignored.py").exists()


def test_stable_image_finalization_installs_offline_without_third_party_drift(tmp_path):
    import shutil

    import pip

    def wheel(name, version, destination):
        package = name.replace("-", "_")
        info = f"{package}-{version}.dist-info"
        target = destination / f"{package}-{version}-py3-none-any.whl"
        files = {
            f"{package}/__init__.py": f'__version__ = "{version}"\n',
            f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\nRequires-Dist: unavailable-release-test-dependency==999\n",
            f"{info}/WHEEL": "Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        }
        files[f"{info}/RECORD"] = (
            "".join(f"{path},,\n" for path in files) + f"{info}/RECORD,,\n"
        )
        with zipfile.ZipFile(target, "w") as archive:
            for path, content in files.items():
                archive.writestr(path, content)
        return target

    environment = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
    python = str(environment / "bin/python")
    rc = wheel("inference-sdk", "1.2.3rc1", tmp_path)
    dependency = wheel("third-party-fixture", "7.8.9", tmp_path)
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            str(rc),
            str(dependency),
        ],
        check=True,
        capture_output=True,
    )
    root = tmp_path / "promotion"
    (root / "wheels").mkdir(parents=True)
    stable = wheel("inference-sdk", "1.2.3", root / "wheels")
    flow._write(
        root / "manifest.json",
        {"inference-sdk": {"version": "1.2.3", "file": stable.name}},
    )
    shutil.copytree(Path(pip.__file__).parent, root / "vendor/pip")
    shutil.copy2(
        ROOT / ".github/scripts/release_image_finalize.py", root / "finalize.py"
    )
    source = root / "sources/inference/core"
    source.mkdir(parents=True)
    (source / "version.py").write_text('__version__ = "1.2.3"\n')
    app = tmp_path / "app"
    (app / "inference/core").mkdir(parents=True)
    (app / "inference/core/version.py").write_text('__version__ = "1.2.3rc1"\n')
    (app / "inference/unchanged.py").write_text("preserved\n")
    for _ in range(2):
        subprocess.run(
            [python, str(root / "finalize.py")],
            cwd=app,
            check=True,
            capture_output=True,
        )
    assert (app / "inference/core/version.py").read_text() == '__version__ = "1.2.3"\n'
    assert (app / "inference/unchanged.py").read_text() == "preserved\n"
    result = subprocess.run(
        [
            python,
            "-c",
            "from importlib.metadata import version; import inference_sdk; assert version('inference-sdk') == '1.2.3'; assert version('third-party-fixture') == '7.8.9'; assert inference_sdk.__version__ == '1.2.3'",
        ],
        cwd=app,
        check=True,
        capture_output=True,
    )
    assert result.returncode == 0


@pytest.mark.parametrize(
    "specifier", ["==0.39.9rc1", ">=0.39.9rc1", ">=0.39.9rc1,<1.0"]
)
def test_lock_metadata_normalizes_named_first_party_specifiers(specifier):
    rc = {
        "name": "inference-models",
        "specifier": specifier,
        "marker": "python_version >= '3.10'",
    }
    stable = {**rc, "specifier": specifier.replace("rc1", "")}
    assert flow._normalize(rc) == flow._normalize(stable)
    third_party = {**rc, "name": "numpy"}
    assert flow._normalize(third_party) != flow._normalize(
        {**third_party, "specifier": stable["specifier"]}
    )
    assert flow._normalize(rc) != flow._normalize({**stable, "specifier": ">=0.40.0"})


def test_real_lock_first_party_metadata_survives_finalization():
    import tomllib

    rc = (ROOT / "workflows/uv.lock").read_bytes()
    assert b'name = "inference-models", specifier = ">=0.39.9rc1"' in rc
    # Finalization changes only the two named first-party requirement spellings.
    stable = rc.replace(
        b'specifier = ">=0.39.9rc1"', b'specifier = ">=0.39.9"'
    ).replace(b'specifier = ">=1.8.5rc1"', b'specifier = ">=1.8.5"')
    assert flow._canonical("workflows/uv.lock", rc) == flow._canonical(
        "workflows/uv.lock", stable
    )
    changed = tomllib.loads(stable.decode())
    changed["revision"] += 1
    assert flow._normalize(tomllib.loads(rc.decode())) != flow._normalize(changed)


@pytest.mark.parametrize(
    "path",
    ["inference_server/pyproject.toml", "inference_model_manager/pyproject.toml"],
)
def test_server_stack_metadata_accepts_only_rc_final_changes(path):
    rc = b'[project]\nname="server-fixture"\ndependencies=["inference-models>=0.39.9rc1,<1.0", "numpy>=2.0.0rc1"]\n'
    stable = rc.replace(b"0.39.9rc1", b"0.39.9")
    assert flow._canonical(path, rc) == flow._canonical(path, stable)
    assert flow._canonical(path, rc) != flow._canonical(
        path, stable.replace(b"0.39.9", b"0.40.0")
    )
    assert flow._canonical(path, rc) != flow._canonical(
        path, stable.replace(b"2.0.0rc1", b"2.0.0")
    )


def test_historical_stable_bump_distinguishes_notes_from_dependency_changes():
    fixture = json.loads(
        (ROOT / ".github/scripts/fixtures/release_1_8_5_metadata.json").read_text()
    )
    for path, versions in fixture["files"].items():
        equivalent = flow._canonical(
            path, versions["before"].encode()
        ) == flow._canonical(path, versions["after"].encode())
        assert equivalent is (path == "stream_vision/CHANGELOG.md")


def test_retry_artifact_payload_contains_only_attempt_numbers_and_exit_codes(tmp_path):
    secret = "fixture-private-api-key"
    result = subprocess.run(
        ["bash", str(ROOT / ".github/scripts/retry.sh")],
        env={
            **os.environ,
            "RETRY_COMMAND": f"echo 'https://example.invalid/?api_key={secret}'; exit 7",
            "RETRY_ATTEMPTS": "1",
            "RETRY_LOG_DIR": str(tmp_path / "logs"),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 7
    assert (
        secret in result.stdout
    )  # The runner masks its console; artifacts cannot rely on it.
    assert {p.name for p in (tmp_path / "logs").iterdir()} == {"attempts.txt"}
    assert (tmp_path / "logs/attempts.txt").read_text() == "1 7\n"


def test_retry_artifact_uploads_never_include_test_output():
    import yaml

    for path in (ROOT / ".github/workflows").glob("*.yml"):
        workflow = yaml.safe_load(path.read_text())
        for job in workflow.get("jobs", {}).values():
            for step in job.get("steps", []):
                if step.get("uses", "").startswith(
                    "actions/upload-artifact@"
                ) and step.get("with", {}).get("name", "").startswith("rc-"):
                    paths = step["with"]["path"].splitlines()
                    assert paths and all(
                        p.endswith("/attempts.txt") for p in paths
                    ), path


@pytest.mark.parametrize(
    "fault", [None, "feature-branch", "unrelated-source", "wrong-branch"]
)
def test_rc_receipt_requires_release_branch_and_source_ancestry(
    candidate, monkeypatch, fault
):
    source, _, state = candidate
    source_sha = state["source_sha"]
    if fault == "unrelated-source":
        _git(source, "checkout", "--orphan", "unrelated")
        _git(source, "commit", "--allow-empty", "-m", "Unrelated root")
        source_sha = _git(source, "rev-parse", "HEAD")
        _git(source, "checkout", "--detach", state["source_sha"])
    run = {
        "id": 12,
        "head_sha": state["source_sha"],
        "head_branch": state["branch"],
        "head_repository": {"full_name": flow.REPOSITORY},
    }
    receipt = {
        "schema": 1,
        "channel": "rc",
        "version": "9.9.0rc1",
        "run_id": "12",
        "trigger_sha": state["source_sha"],
        "source_sha": source_sha,
        "branch": state["branch"],
        "repository": flow.REPOSITORY,
        "complete": True,
    }
    if fault == "feature-branch":
        run["head_branch"] = receipt["branch"] = "feature/forged-receipt"
    elif fault == "wrong-branch":
        receipt["branch"] = "release/other"

    def api(path):
        if "workflows/" in path:
            return {"workflow_runs": [run]}
        return {
            "artifacts": [{"id": 101, "name": "release-receipt-1", "expired": False}]
        }

    monkeypatch.setattr(flow, "_api", api)
    monkeypatch.setattr(flow, "_download_receipt", lambda artifact_id: receipt)
    if fault:
        with pytest.raises(ValueError, match="No successful|ancestor|Untrusted"):
            flow._find_rc("9.9.0")
        assert not (flow.STATE / "accepted-rc.json").exists()
    else:
        assert flow._find_rc("9.9.0") == receipt


@pytest.mark.parametrize(
    "fault", [None, "lock-structure", "application", "third-party"]
)
def test_stable_full_source_and_lock_equivalence_runs_in_prepare(
    candidate, monkeypatch, fault
):
    source, _, state = candidate
    version_path = Path("inference/core/version.py")
    version_path.parent.mkdir(parents=True)
    version_path.write_text('__version__ = "9.9.0rc1"\n')
    _git(source, "add", "inference/core/version.py")
    _git(source, "commit", "-m", "Prepared RC")
    policy = json.loads((ROOT / ".release/release-policy.json").read_text())
    receipt = {
        "run_id": "12",
        "application": flow._fingerprint(include_locks=False),
        "fingerprint": flow._fingerprint(),
        "third_party": {p: [] for p in flow.LOCKS},
        "policy": policy,
    }
    version_path.write_text('__version__ = "9.9.0"\n')
    if fault == "lock-structure":
        Path(flow.LOCKS[0]).write_text("version = 1\nrevision = 99\n")
    elif fault == "application":
        Path("app.py").write_text("answer = 0\n")
    elif fault == "third-party":
        Path(flow.LOCKS[0]).write_text(
            'version = 1\nrevision = 3\n[[package]]\nname = "numpy"\nversion = "2.0.0"\n'
        )
    # Only tracked source enters the tag; release-state stays outside Git.
    _git(source, "add", "inference/core/version.py", "app.py", *flow.LOCKS)
    _git(source, "commit", "-m", "Prepared final")
    _git(source, "tag", "v9.9.0")
    sha = _git(source, "rev-parse", "HEAD")
    monkeypatch.setattr(flow, "POLICY", ROOT / ".release/release-policy.json")
    monkeypatch.setattr(flow, "_find_rc", lambda version: receipt)
    monkeypatch.setattr(flow.release, "_expected_version", lambda: "9.9.0")
    for name, value in {
        "GITHUB_REPOSITORY": flow.REPOSITORY,
        "GITHUB_EVENT_NAME": "release",
        "GITHUB_SHA": sha,
        "GITHUB_REF": "refs/tags/v9.9.0",
        "GITHUB_RUN_ID": "15",
    }.items():
        monkeypatch.setenv(name, value)
    flow._write(
        flow.STATE / "event.json",
        {
            "action": "published",
            "release": {"tag_name": "v9.9.0", "draft": False, "prerelease": False},
        },
    )
    if fault:
        with pytest.raises(ValueError, match="Stable source/locks/policy differ"):
            flow._prepare()
    else:
        flow._prepare()
        assert flow._state()["accepted_rc_run_id"] == "12"


@pytest.mark.parametrize("tag", ["1.8.5-post1", "1.8.5.post1"])
def test_runtime_image_tag_preserves_official_post_release_spelling(
    tmp_path, monkeypatch, tag
):
    images = _load_images()
    monkeypatch.chdir(tmp_path)
    item = next(
        image
        for image in json.loads((ROOT / ".release/release-policy.json").read_text())[
            "images"
        ]
        if image["version_source"] == "runtime"
    )
    flow._write(
        flow.STATE / "release.json",
        {
            "channel": "stable",
            "version": "1.8.5.post1",
            "image_version": tag,
            "source_sha": "source",
            "run_id": "1",
            "policy": {"images": [item]},
        },
    )
    flow._write(
        flow.STATE / "accepted-rc.json",
        {"images": [{"key": item["key"], "digest": "sha256:" + "a" * 64, "smoke": {}}]},
    )
    monkeypatch.setattr(
        images, "_stable_context", lambda plan, digest: ("context", "Dockerfile")
    )
    images._plan(item["key"])
    assert (
        json.loads(Path("image-plan.json").read_text())["tag"]
        == item["tag_prefix"] + tag
    )


@pytest.mark.parametrize(
    "package", ["inference-models", "roboflow-workflows", "streamvision"]
)
@pytest.mark.parametrize(
    "channel,version,upload,allowed",
    [
        ("rc", "1.2.3", True, False),
        ("rc", "1.2.3", False, True),
        ("rc", "1.2.3rc1", True, True),
        ("stable", "1.2.3", True, True),
    ],
)
def test_rc_reuses_but_never_uploads_new_stable_dependencies(
    tmp_path, monkeypatch, package, channel, version, upload, allowed
):
    monkeypatch.chdir(tmp_path)
    flow._write(flow.STATE / "release.json", {"channel": channel})
    flow._write(flow.STATE / f"{package}-prepare.json", {"upload": upload})
    monkeypatch.setattr(flow, "_versions", lambda: {package: version})
    if allowed:
        flow._check_upload(package)
    else:
        with pytest.raises(ValueError, match="cannot publish a new stable version"):
            flow._check_upload(package)
