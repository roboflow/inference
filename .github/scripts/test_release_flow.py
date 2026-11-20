"""Behavioral boundaries for the one-run release pipeline."""

import importlib.util
import io
import json
import os
import subprocess
import sys
import threading
import urllib.error
import zipfile
from functools import partial
from http.client import IncompleteRead
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
    monkeypatch.setattr(flow, "POLICY", ROOT / ".release/release-policy.json")
    monkeypatch.setenv("GH_TOKEN", "fixture-token")
    original_git = flow._git

    def local_git(*args, env=None, **kwargs):
        args = tuple(
            str(remote) if arg == "https://github.com/roboflow/inference.git" else arg
            for arg in args
        )
        return original_git(*args, env=env, **kwargs)

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


def test_retry_stdin_consumer_cannot_swallow_remaining_script(tmp_path):
    result = subprocess.run(
        ["bash", str(ROOT / ".github/scripts/retry.sh")],
        env={
            **os.environ,
            "RETRY_COMMAND": "cat >/dev/null\nexit 7",
            "RETRY_ATTEMPTS": "3",
            "RETRY_LOG_DIR": str(tmp_path / "logs"),
            "RETRY_DELAY_SECONDS": "0",
        },
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 7
    assert (tmp_path / "logs/attempts.txt").read_text() == "1 7\n2 7\n3 7\n"


@pytest.mark.parametrize("recover", [True, False])
def test_retry_times_out_hung_attempts_and_can_recover(tmp_path, recover):
    marker = tmp_path / "attempted"
    command = f'touch "{marker}"; sleep 60'
    if recover:
        command = f'[ ! -f "{marker}" ] || exit 0; ' + command
    result = subprocess.run(
        ["bash", str(ROOT / ".github/scripts/retry.sh")],
        env={
            **os.environ,
            "RETRY_COMMAND": command,
            "RETRY_ATTEMPTS": "3",
            "RETRY_TIMEOUT_SECONDS": "1",
            "RETRY_LOG_DIR": str(tmp_path / "logs"),
            "RETRY_DELAY_SECONDS": "0",
        },
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == (0 if recover else 124)
    assert (tmp_path / "logs/attempts.txt").read_text() == (
        "1 124\n2 0\n" if recover else "1 124\n2 124\n3 124\n"
    )


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
    _, remote, state = candidate
    run = {
        "id": 12,
        "head_sha": state["source_sha"],
        "head_branch": state["branch"],
        "head_repository": {"full_name": flow.REPOSITORY},
    }
    receipt = {
        **_rc_evidence(state),
        "schema": 1,
        "channel": "rc",
        "version": "1.2.3rc2",
        "run_id": "12",
        "evidence_ref": flow._evidence_ref("12"),
        "trigger_sha": state["source_sha"],
        "source_sha": state["source_sha"],
        "branch": state["branch"],
        "repository": flow.REPOSITORY,
        "complete": True,
        "fingerprint": flow._fingerprint(),
    }
    flow._git("push", str(remote), f"{state['source_sha']}:{receipt['evidence_ref']}")
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
        {
            "key": "cpu",
            "source_sha": state["source_sha"],
            "smoke": {"linux/amd64": {"numpy": "2.0.0"}},
        },
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
    "fault",
    [
        None,
        "feature-branch",
        "null-branch",
        "unrelated-source",
        "wrong-branch",
        "malformed-source",
    ],
)
def test_rc_receipt_requires_release_branch_and_source_ancestry(
    candidate, monkeypatch, fault
):
    source, remote, state = candidate
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
        **_rc_evidence(state),
        "schema": 1,
        "channel": "rc",
        "version": "9.9.0rc1",
        "run_id": "12",
        "evidence_ref": flow._evidence_ref("12"),
        "trigger_sha": state["source_sha"],
        "source_sha": source_sha,
        "branch": state["branch"],
        "repository": flow.REPOSITORY,
        "complete": True,
        "fingerprint": flow._fingerprint(),
    }
    flow._git("push", str(remote), f"{source_sha}:{receipt['evidence_ref']}")
    if fault == "feature-branch":
        run["head_branch"] = receipt["branch"] = "feature/forged-receipt"
    elif fault == "null-branch":
        run["head_branch"] = None
    elif fault == "malformed-source":
        receipt["source_sha"] = "HEAD"
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


def _mock_rc_evidence(monkeypatch, receipts, *, retain=True):
    if retain:
        for receipt in receipts:
            receipt["evidence_ref"] = flow._evidence_ref(receipt["run_id"])
            flow._git(
                "push",
                f"https://github.com/{flow.REPOSITORY}.git",
                f"{receipt['source_sha']}:{receipt['evidence_ref']}",
            )
    runs = [
        {
            "id": receipt["run_id"],
            "head_sha": receipt["trigger_sha"],
            "head_branch": receipt["branch"],
            "head_repository": {"full_name": flow.REPOSITORY},
        }
        for receipt in receipts
    ]
    by_id = {str(receipt["run_id"]): receipt for receipt in receipts}

    def api(path):
        if "workflows/" in path:
            return {"workflow_runs": runs}
        run_id = path.split("/")[2]
        return {
            "artifacts": [{"id": run_id, "name": "release-receipt-1", "expired": False}]
        }

    monkeypatch.setattr(flow, "_api", api)
    monkeypatch.setattr(
        flow, "_download_receipt", lambda artifact_id: by_id[str(artifact_id)]
    )


def _rc_evidence(state):
    policy = json.loads((ROOT / ".release/release-policy.json").read_text())
    smoke = {f"{name}-py3.12": {name: "9.9.0"} for name in flow.dependencies.OWNED}
    smoke.update(
        {
            f"{name}-py{version}": {name: "9.9.0"}
            for name in flow.release.RUNTIME
            for version in policy["python_versions"]
        }
    )
    images = [
        {
            "key": item["key"],
            "source_sha": flow._git("rev-parse", "HEAD"),
            "digest": "sha256:" + "a" * 64,
            "smoke": {platform: {"numpy": "2.0.0"} for platform in item["platforms"]},
        }
        for item in policy["images"]
    ]
    return {
        **state,
        "schema": 1,
        "repository": flow.REPOSITORY,
        "source_sha": flow._git("rev-parse", "HEAD"),
        "application": flow._fingerprint(include_locks=False),
        "fingerprint": flow._fingerprint(),
        "files": flow._file_fingerprints(),
        "policy": json.loads((ROOT / ".release/release-policy.json").read_text()),
        "third_party": {p: [] for p in flow.LOCKS},
        "images": images,
        "smoke": smoke,
        "complete": True,
    }


@pytest.mark.parametrize("squash", [False, True])
@pytest.mark.parametrize("tag_suffix", ["", "-post1", ".post1"])
def test_stable_prepare_accepts_equivalent_squash_with_distinct_trigger_and_source(
    candidate, monkeypatch, squash, tag_suffix
):
    source, _, state = candidate
    base = state["source_sha"]
    version_path = Path("inference/core/version.py")
    version_path.parent.mkdir(parents=True)
    rc_version = "9.9.0rc1" + (".post1" if tag_suffix else "")
    final_version = "9.9.0" + (".post1" if tag_suffix else "")
    version_path.write_text(f'__version__ = "{rc_version}"\n')
    _git(source, "add", str(version_path))
    _git(source, "commit", "-m", "Prepare RC")
    trigger = _git(source, "rev-parse", "HEAD")
    Path(flow.LOCKS[0]).write_text("version = 1\nrevision = 4\n")
    _git(source, "add", flow.LOCKS[0])
    _git(source, "commit", "-m", "Bot lock commit")
    receipt = _rc_evidence({**state, "trigger_sha": trigger, "version": rc_version})
    assert receipt["source_sha"] != trigger
    _mock_rc_evidence(monkeypatch, [receipt])
    if squash:
        _git(source, "checkout", "-b", "main", base)
        _git(source, "merge", "--squash", receipt["source_sha"])
    version_path.write_text(f'__version__ = "{final_version}"\n')
    _git(source, "add", str(version_path))
    _git(source, "commit", "-m", "Release final")
    sha = _git(source, "rev-parse", "HEAD")
    tag = "v9.9.0" + tag_suffix
    _git(source, "tag", tag)
    assert len({trigger, receipt["source_sha"], sha}) == 3
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", receipt["source_sha"], "HEAD"]
    )
    assert ancestor.returncode == (1 if squash else 0)
    monkeypatch.setattr(flow, "POLICY", ROOT / ".release/release-policy.json")
    for name, value in {
        "GITHUB_REPOSITORY": flow.REPOSITORY,
        "GITHUB_EVENT_NAME": "release",
        "GITHUB_SHA": sha,
        "GITHUB_REF": "refs/tags/" + tag,
        "GITHUB_RUN_ID": "15",
    }.items():
        monkeypatch.setenv(name, value)
    flow._write(
        flow.STATE / "event.json",
        {
            "action": "published",
            "release": {"tag_name": tag, "draft": False, "prerelease": False},
        },
    )
    flow._prepare()
    assert flow._state()["accepted_rc_run_id"] == state["run_id"]
    assert flow._state()["image_version"] == tag.removeprefix("v")
    assert (
        json.loads((flow.STATE / "accepted-rc.json").read_text())["source_sha"]
        == receipt["source_sha"]
    )


@pytest.mark.parametrize("newer", ["non-ancestor", "different-content", "null-branch"])
def test_rc_selection_skips_newer_ineligible_receipt(candidate, monkeypatch, newer):
    source, _, state = candidate
    Path(flow.LOCKS[0]).write_text("version = 1\nrevision = 4\n")
    _git(source, "add", flow.LOCKS[0])
    _git(source, "commit", "-m", "Bot lock commit")
    good = _rc_evidence(state)
    if newer == "non-ancestor":
        _git(source, "checkout", "--orphan", "parallel-release")
        _git(source, "commit", "-m", "Unrelated trigger with identical content")
        bad = {**good, "trigger_sha": _git(source, "rev-parse", "HEAD")}
    elif newer == "different-content":
        Path("app.py").write_text("answer = 99\n")
        _git(source, "add", "app.py")
        _git(source, "commit", "-m", "Next RC")
        bad = _rc_evidence(state)
    else:
        bad = {**good, "branch": None}
    bad["run_id"] = "124"
    _git(source, "checkout", "--detach", good["source_sha"])
    _mock_rc_evidence(monkeypatch, [bad, good])
    assert flow._find_rc("9.9.0")["run_id"] == good["run_id"]


def test_rc_selection_reports_differing_paths_and_preflight_detects_them(
    candidate, monkeypatch, capsys
):
    source, _, state = candidate
    receipt = _rc_evidence(state)
    _mock_rc_evidence(monkeypatch, [receipt])
    monkeypatch.setattr(flow.release, "_expected_version", lambda: "9.9.0")
    flow._preflight()
    assert "successful RC run 123" in capsys.readouterr().out
    Path("app.py").write_text("answer = 99\n")
    Path(flow.LOCKS[0]).write_text("version = 1\nrevision = 4\n")
    Path("added.py").write_text("new = True\n")
    _git(source, "add", "added.py")
    with pytest.raises(ValueError, match="No successful equivalent") as error:
        flow._preflight()
    assert all(
        path in str(error.value) for path in ["app.py", "added.py", flow.LOCKS[0]]
    )


@pytest.mark.parametrize(
    "tag",
    [
        "V9.9.0",
        "v9.9.0.0",
        "v9.9.0_post1",
        "9.9.0",
        "v9.9.0rc1",
        "v09.9.0",
        "v9.09.0",
        "v9.9.00",
        "v9.9.0-post01",
        "v9.9.0.post01",
    ],
)
def test_stable_prepare_rejects_noncanonical_image_tag_spellings(
    candidate, monkeypatch, tag
):
    monkeypatch.setenv("GITHUB_REPOSITORY", flow.REPOSITORY)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "release")
    monkeypatch.setattr(flow.release, "_expected_version", lambda: "9.9.0")
    flow._write(
        flow.STATE / "event.json",
        {
            "action": "published",
            "release": {"tag_name": tag, "draft": False, "prerelease": False},
        },
    )
    with pytest.raises(ValueError, match="Stable tag must"):
        flow._prepare()


@pytest.mark.parametrize("table", ["tool.setuptools.package-data", "tool.custom"])
def test_pyproject_equivalence_preserves_non_dependency_strings(table):
    rc = f'[{table}]\ninference = ["inference==1.2.3rc1"]\n'.encode()
    for version in [b"1.2.3", b"1.2.3rc2"]:
        assert flow._canonical("pyproject.toml", rc) != flow._canonical(
            "pyproject.toml", rc.replace(b"1.2.3rc1", version)
        )


@pytest.mark.parametrize(
    "table", ["project.optional-dependencies", "dependency-groups"]
)
def test_pyproject_equivalence_normalizes_only_first_party_dependencies(table):
    rc = f'[{table}]\ntest = ["inference==1.2.3rc1", "numpy==2.0.0rc1"]\n'.encode()
    assert flow._canonical("pyproject.toml", rc) == flow._canonical(
        "pyproject.toml", rc.replace(b"1.2.3rc1", b"1.2.3")
    )
    assert flow._canonical("pyproject.toml", rc) != flow._canonical(
        "pyproject.toml", rc.replace(b"2.0.0rc1", b"2.0.0")
    )


@pytest.mark.parametrize("version", ["1.2.3a1", "1.2.3b2", "1.2.3.dev1", "1.2.3+local"])
def test_non_rc_dependency_versions_are_preserved(version):
    requirement = "inference==" + version
    assert flow._requirement(requirement) == requirement
    assert flow._requirement(requirement) != flow._requirement("inference==1.2.3")


def test_api_retries_share_a_bounded_evidence_lookup_budget(monkeypatch):
    import urllib.error

    clock = [0.0]
    requests = []
    monkeypatch.setenv("GH_TOKEN", "fixture-token")
    flow._api_deadline.cache_clear()
    monkeypatch.setattr(flow.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        flow.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )

    def unavailable(request, *, timeout):
        requests.append(timeout)
        clock[0] += timeout
        raise urllib.error.URLError("fixture unavailable")

    monkeypatch.setattr(flow.urllib.request, "urlopen", unavailable)
    for _ in range(2):
        with pytest.raises(urllib.error.URLError):
            flow._api("actions/runs/1/jobs")
    with pytest.raises(flow._EvidenceDeadlineExceeded, match="10-minute budget"):
        flow._api("actions/runs/1/jobs?page=2")
    assert clock[0] == 600
    assert requests[-1] <= 30


def test_rc_attempt_and_job_budgets_cover_retries_and_setup():
    import re

    import yaml

    def minutes(value, *, release=True):
        if isinstance(value, int) or value.isdigit():
            return int(value)

        pattern = r"&& (\d+) " if release else r"\|\| (\d+) "
        return int(re.search(pattern, value)[1])

    actions = {
        action: yaml.safe_load((ROOT / action / "action.yml").read_text())
        for action in (
            ".github/actions/retry-tests",
            ".github/actions/server-tests-cpu",
        )
    }

    for path in (ROOT / ".github/workflows").glob("*.yml"):
        workflow = yaml.safe_load(path.read_text())
        for name, job in workflow.get("jobs", {}).items():
            steps = job.get("steps", [])
            retry_steps = [
                step
                for step in steps
                if step.get("uses")
                in {
                    "./.github/actions/retry-tests",
                    "./.github/actions/server-tests-cpu",
                }
            ]
            if not retry_steps or "release_source" not in str(
                job.get("timeout-minutes", "")
            ):
                continue

            normal_job_minutes = minutes(job["timeout-minutes"], release=False)
            for step in steps:
                if "timeout-minutes" in step:
                    assert (
                        minutes(step["timeout-minutes"], release=False)
                        <= normal_job_minutes
                    ), (path, name, step["name"], "normal step exceeds job")
            for step in retry_steps:
                action = actions[step["uses"].removeprefix("./")]
                inputs = step.get("with", {})
                attempts = inputs.get(
                    "attempts", action["inputs"]["attempts"]["default"]
                )
                assert minutes(attempts) == 3, (path, name, "RC attempts")
                assert minutes(attempts, release=False) == 1, (
                    path,
                    name,
                    "normal attempts",
                )
                if "timeout-seconds" in action["inputs"]:
                    seconds = int(
                        inputs.get(
                            "timeout-seconds",
                            action["inputs"]["timeout-seconds"]["default"],
                        )
                    )
                else:
                    seconds = int(
                        next(
                            inner["env"]["RETRY_TIMEOUT_SECONDS"]
                            for inner in action["runs"]["steps"]
                            if "RETRY_TIMEOUT_SECONDS" in inner.get("env", {})
                        )
                    )
                assert minutes(step["timeout-minutes"]) * 60 >= 3 * seconds + 110, (
                    path,
                    name,
                    step["name"],
                    "RC retries need kill grace, retry waits and 60 seconds spare",
                )
            step_budget = sum(
                minutes(step["timeout-minutes"])
                for step in steps
                if "timeout-minutes" in step
            )
            assert minutes(job["timeout-minutes"]) >= step_budget + 30, (path, name)


@pytest.fixture(autouse=True)
def reset_evidence_request_budget():
    flow._api_deadline.cache_clear()
    yield
    flow._api_deadline.cache_clear()


@pytest.mark.parametrize("attempts", [1, 3])
def test_retry_command_cannot_consume_inherited_input(tmp_path, attempts):
    sentinel = "caller-input-must-not-reach-tests"
    result = subprocess.run(
        ["bash", str(ROOT / ".github/scripts/retry.sh")],
        env={
            **os.environ,
            "RETRY_COMMAND": "cat; exit 0",
            "RETRY_ATTEMPTS": str(attempts),
            "RETRY_LOG_DIR": str(tmp_path / "logs"),
        },
        input=sentinel,
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 0
    assert sentinel not in result.stdout


def test_single_attempt_keeps_existing_ci_step_deadline(tmp_path):
    result = subprocess.run(
        ["bash", str(ROOT / ".github/scripts/retry.sh")],
        env={
            **os.environ,
            "RETRY_COMMAND": "sleep 1.2",
            "RETRY_ATTEMPTS": "1",
            "RETRY_TIMEOUT_SECONDS": "1",
            "RETRY_LOG_DIR": str(tmp_path / "logs"),
        },
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 0
    assert (tmp_path / "logs/attempts.txt").read_text() == "1 0\n"


def test_successful_slow_api_read_still_exceeds_shared_budget(monkeypatch):
    import io

    clock = [0.0]
    calls = []
    monkeypatch.setenv("GH_TOKEN", "fixture-token")
    monkeypatch.setattr(flow.time, "monotonic", lambda: clock[0])

    class SlowResponse(io.BytesIO):
        def read(self, *args):
            clock[0] = 601.0
            return super().read(*args)

    def open_response(request, *, timeout):
        calls.append(timeout)
        return SlowResponse(b"{}")

    monkeypatch.setattr(flow.urllib.request, "urlopen", open_response)
    with pytest.raises(flow._EvidenceDeadlineExceeded, match="10-minute budget"):
        flow._api("actions/runs/1")
    assert calls == [30]


def test_rc_source_tag_is_created_without_lock_changes_and_never_overwritten(candidate):
    source, remote, state = candidate
    flow._commit_locks()
    ref = flow._state()["evidence_ref"]
    assert ref == "refs/tags/rc-evidence/123"
    assert _git(remote, "rev-parse", ref) == state["source_sha"]
    flow._commit_locks()
    _git(source, "commit", "--allow-empty", "-m", "Different commit")
    with pytest.raises(ValueError, match="already identifies a different commit"):
        flow._retain_rc_source(
            _git(source, "rev-parse", "HEAD"),
            run_id="123",
            remote=str(remote),
            env=None,
        )
    assert _git(remote, "rev-parse", ref) == state["source_sha"]


@pytest.mark.parametrize("fetch_all_tags", [False, True])
@pytest.mark.parametrize("fault", [None, "missing-tag", "wrong-tag", "main-moved"])
def test_stable_from_fresh_clone_after_squash_and_release_branch_deletion(
    candidate, monkeypatch, tmp_path, fetch_all_tags, fault
):
    source, remote, state = candidate
    base = state["source_sha"]
    version = Path("inference/core/version.py")
    version.parent.mkdir(parents=True)
    version.write_text('__version__ = "9.9.0rc1"\n')
    _git(source, "add", str(version))
    _git(source, "commit", "-m", "Prepare RC")
    trigger = _git(source, "rev-parse", "HEAD")
    _git(source, "push", str(remote), f"HEAD:refs/heads/{state['branch']}")
    state.update(
        source_sha=trigger,
        trigger_sha=trigger,
        application=flow._fingerprint(include_locks=False),
    )
    flow._write(flow.STATE / "release.json", state)
    locks = json.loads((flow.STATE / "locks.json").read_text())
    locks[flow.LOCKS[0]] += 'requires-python = ">=3.10"\n'
    flow._write(flow.STATE / "locks.json", locks)
    flow._commit_locks()
    receipt = _rc_evidence(flow._state())
    assert receipt["source_sha"] != trigger
    assert receipt["evidence_ref"] == "refs/tags/rc-evidence/123"
    assert _git(remote, "rev-parse", receipt["evidence_ref"]) == receipt["source_sha"]
    _mock_rc_evidence(monkeypatch, [receipt], retain=False)
    _git(source, "checkout", "-b", "main", base)
    _git(source, "merge", "--squash", receipt["source_sha"])
    version.write_text('__version__ = "9.9.0"\n')
    if fault == "main-moved":
        Path("main-only.py").write_text("untested = True\n")
        _git(source, "add", "main-only.py")
    _git(source, "add", str(version))
    _git(source, "commit", "-m", "Release squash")
    stable_sha = _git(source, "rev-parse", "HEAD")
    _git(source, "tag", "v9.9.0")
    _git(source, "push", str(remote), "refs/heads/main", "refs/tags/v9.9.0")
    _git(remote, "symbolic-ref", "HEAD", "refs/heads/main")
    _git(source, "push", str(remote), "--delete", state["branch"])
    if fault == "missing-tag":
        _git(remote, "update-ref", "-d", receipt["evidence_ref"])
    elif fault == "wrong-tag":
        _git(remote, "update-ref", receipt["evidence_ref"], stable_sha)
    _git(remote, "reflog", "expire", "--expire=now", "--all")
    _git(remote, "gc", "--prune=now")
    fresh = tmp_path / "fresh-stable"
    subprocess.run(
        [
            "git",
            "clone",
            "--no-local",
            "--no-tags",
            "--single-branch",
            "--branch",
            "main",
            str(remote),
            str(fresh),
        ],
        check=True,
        capture_output=True,
    )
    missing = subprocess.run(
        ["git", "-C", str(fresh), "cat-file", "-e", receipt["source_sha"]],
        capture_output=True,
    )
    assert missing.returncode != 0
    assert not _git(remote, "for-each-ref", f"refs/heads/{state['branch']}")
    _git(fresh, "fetch", "origin", "refs/tags/v9.9.0:refs/tags/v9.9.0")
    if fetch_all_tags:
        _git(
            fresh,
            "fetch",
            "origin",
            "+refs/heads/*:refs/remotes/origin/*",
            "+refs/tags/*:refs/tags/*",
        )
    monkeypatch.chdir(fresh)
    monkeypatch.setattr(flow, "POLICY", ROOT / ".release/release-policy.json")
    for name, value in {
        "GITHUB_REPOSITORY": flow.REPOSITORY,
        "GITHUB_EVENT_NAME": "release",
        "GITHUB_SHA": stable_sha,
        "GITHUB_REF": "refs/tags/v9.9.0",
        "GITHUB_RUN_ID": "456",
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
        with pytest.raises(ValueError, match="No successful equivalent") as error:
            flow._prepare()
        if fault == "main-moved":
            assert "main-only.py" in str(error.value)
        assert not (flow.STATE / "accepted-rc.json").exists()
    else:
        flow._prepare()
        assert flow._state()["accepted_rc_run_id"] == "123"
        assert _git(fresh, "rev-parse", "HEAD") == stable_sha
        assert flow._fingerprint(ref=receipt["source_sha"]) == receipt["fingerprint"]


@pytest.mark.parametrize("attack", ["source-tree", "trigger-workflow"])
def test_forged_rc_receipt_cannot_substitute_git_source(candidate, monkeypatch, attack):
    source, _, state = candidate
    workflow = Path(".github/workflows/publish.pypi.yml")
    workflow.parent.mkdir(parents=True)
    workflow.write_text("trusted pipeline\n")
    _git(source, "add", str(workflow))
    _git(source, "commit", "-m", "Reviewed pipeline")
    trusted = _git(source, "rev-parse", "HEAD")
    fingerprint = flow._fingerprint()
    workflow.write_text("forged receipt producer\n")
    _git(source, "commit", "-am", "Untrusted pipeline")
    trigger = _git(source, "rev-parse", "HEAD")
    if attack == "trigger-workflow":
        workflow.write_text("trusted pipeline\n")
        _git(source, "commit", "-am", "Restore reviewed tree after forged run")
    receipt = _rc_evidence({**state, "trigger_sha": trigger})
    receipt["fingerprint"] = fingerprint
    _mock_rc_evidence(monkeypatch, [receipt])
    _git(source, "checkout", "--detach", trusted)
    with pytest.raises(
        ValueError, match="fingerprint does not match|other than generated locks"
    ):
        flow._find_rc("9.9.0")
    assert not (flow.STATE / "accepted-rc.json").exists()


@pytest.mark.parametrize(
    "fault", ["schema", "source", "missing-tag", "missing-object-mismatch", "json"]
)
def test_unusable_newer_receipt_cannot_hide_valid_older_evidence(
    candidate, monkeypatch, fault
):
    _, remote, state = candidate
    good = _rc_evidence(state)
    bad = {**good, "run_id": "124"}
    _mock_rc_evidence(monkeypatch, [bad, good])
    if fault == "schema":
        bad["schema"] = 999
    elif fault == "source":
        bad["source_sha"] = "HEAD"
    elif fault == "missing-tag":
        _git(remote, "update-ref", "-d", bad["evidence_ref"])
    elif fault == "missing-object-mismatch":
        bad["source_sha"] = "a" * 40
        bad["fingerprint"] = "b" * 64
    else:
        download = flow._download_receipt

        def broken_json(artifact_id):
            if str(artifact_id) == "124":
                raise json.JSONDecodeError("invalid artifact", "", 0)
            return download(artifact_id)

        monkeypatch.setattr(flow, "_download_receipt", broken_json)
    fetches = []
    git = flow._git

    def record_git(*args, **kwargs):
        if args[0] == "fetch":
            fetches.append(args[-1])
        return git(*args, **kwargs)

    monkeypatch.setattr(flow, "_git", record_git)
    assert flow._find_rc("9.9.0")["run_id"] == "123"
    if fault in {"schema", "source", "missing-object-mismatch", "json"}:
        assert fetches == [good["evidence_ref"]]


def test_git_blob_fingerprint_matches_worktree_for_modes_and_binary_files(candidate):
    source, _, _ = candidate
    Path("bytes.bin").write_bytes(b"\x00\xff\xfe\n")
    Path("script.sh").write_text("#!/bin/sh\nexit 0\n")
    Path("script.sh").chmod(0o755)
    Path("link").symlink_to("script.sh")
    _git(source, "add", "bytes.bin", "script.sh", "link")
    _git(source, "commit", "-m", "Mixed blobs")
    assert flow._file_fingerprints(ref="HEAD") == flow._file_fingerprints()
    assert flow._fingerprint(ref="HEAD") == flow._fingerprint()


def test_rc_lock_rename_cannot_hide_non_lock_deletion(candidate, monkeypatch):
    source, _, state = candidate
    lock = flow.LOCKS[0]
    Path("app.py").write_text("version = 1\nrevision = 99\n")
    Path(lock).unlink()
    _git(source, "add", "-A")
    _git(source, "commit", "-m", "Prepare trigger without generated lock")
    state["trigger_sha"] = _git(source, "rev-parse", "HEAD")
    _git(source, "mv", "app.py", lock)
    _git(source, "commit", "-m", "Rename application into lock")
    receipt = _rc_evidence(state)
    _mock_rc_evidence(monkeypatch, [receipt])
    assert _git(
        source, "diff", "--name-status", state["trigger_sha"], "HEAD"
    ).startswith("R100\tapp.py\t")
    with pytest.raises(ValueError, match="other than generated locks"):
        flow._find_rc("9.9.0")
    assert not (flow.STATE / "accepted-rc.json").exists()


@pytest.mark.parametrize(
    "field", ["application", "policy", "third_party", "images", "smoke"]
)
@pytest.mark.parametrize("fault", ["missing", "wrong-shape", "incompatible"])
def test_rc_selection_skips_incomplete_consumer_metadata(
    candidate, monkeypatch, field, fault
):
    _, _, state = candidate
    good = _rc_evidence(state)
    bad = json.loads(json.dumps({**good, "run_id": "124"}))
    if fault == "missing":
        del bad[field]
    elif fault == "wrong-shape":
        bad[field] = [] if field != "images" else {}
    elif field == "application":
        bad[field] = "b" * 64
    elif field == "policy":
        bad[field]["python_versions"] = ["3.99"]
    elif field == "third_party":
        bad[field][flow.LOCKS[0]] = [{"name": "numpy", "version": "999"}]
    elif field == "images":
        bad[field][0]["smoke"] = {}
    else:
        bad[field].pop(next(iter(bad[field])))
    _mock_rc_evidence(monkeypatch, [bad, good])
    assert flow._find_rc("9.9.0")["run_id"] == "123"
    assert json.loads((flow.STATE / "accepted-rc.json").read_text())["run_id"] == "123"


@pytest.mark.parametrize("fault", ["digest", "source", "duplicate", "inventory"])
def test_rc_selection_rejects_unusable_image_evidence(candidate, monkeypatch, fault):
    _, _, state = candidate
    good = _rc_evidence(state)
    bad = json.loads(json.dumps({**good, "run_id": "124"}))
    image = bad["images"][0]
    if fault == "digest":
        image["digest"] = "mutable-tag"
    elif fault == "source":
        image["source_sha"] = "b" * 40
    elif fault == "duplicate":
        bad["images"].append(image.copy())
    else:
        image["smoke"][next(iter(image["smoke"]))] = []
    _mock_rc_evidence(monkeypatch, [bad, good])
    assert flow._find_rc("9.9.0")["run_id"] == "123"


@pytest.mark.parametrize("fault", [None, [], {}, {"numpy": None}, {"numpy": ""}])
def test_rc_selection_rejects_unusable_package_inventory(candidate, monkeypatch, fault):
    _, _, state = candidate
    good = _rc_evidence(state)
    bad = json.loads(json.dumps({**good, "run_id": "124"}))
    bad["smoke"][next(iter(bad["smoke"]))] = fault
    _mock_rc_evidence(monkeypatch, [bad, good])
    assert flow._find_rc("9.9.0")["run_id"] == "123"


@pytest.mark.parametrize("failure", ["http", "url", "read_oserror", "read_reset"])
@pytest.mark.parametrize("exhausted", [False, True])
def test_rc_download_failure_respects_candidate_and_shared_budgets(
    candidate, monkeypatch, failure, exhausted
):
    _, _, state = candidate
    good = _rc_evidence(state)
    bad = {**good, "run_id": "124"}
    download = flow._download_receipt
    _mock_rc_evidence(monkeypatch, [bad, good])
    downloads = []

    def record_download(artifact_id):
        downloads.append(str(artifact_id))
        return download(artifact_id)

    monkeypatch.setattr(flow, "_download_receipt", record_download)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("receipt.json", json.dumps(good))
    now = [0]
    monkeypatch.setattr(flow, "_api_deadline", lambda: 600)
    monkeypatch.setattr(flow.time, "monotonic", lambda: now[0])
    calls = []

    class Opener:
        def open(self, request, *, timeout):
            calls.append(request.full_url)
            if "/124/" in request.full_url:
                now[0] = 601 if exhausted else 1
                if failure == "http":
                    raise urllib.error.HTTPError(
                        request.full_url, 410, "Gone", {}, None
                    )
                if failure == "url":
                    raise urllib.error.URLError("connection failed")

                class BrokenResponse(io.BytesIO):
                    def read(self, *args):
                        error = (
                            ConnectionResetError if failure == "read_reset" else OSError
                        )
                        raise error("receipt body interrupted")

                return BrokenResponse()
            return io.BytesIO(payload.getvalue())

    monkeypatch.setattr(flow.urllib.request, "build_opener", lambda *args: Opener())
    if exhausted:
        with pytest.raises(flow._EvidenceDeadlineExceeded):
            flow._find_rc("9.9.0")
        assert len(calls) == 1
        assert downloads == ["124"]
        assert not (flow.STATE / "accepted-rc.json").exists()
    else:
        assert flow._find_rc("9.9.0")["run_id"] == "123"
        assert len(calls) == 2


@pytest.mark.parametrize("outcome", ["timeout", "slow-success", "already-expired"])
@pytest.mark.parametrize("exhausted", [False, True])
def test_rc_fetch_respects_candidate_and_shared_budgets(
    candidate, monkeypatch, outcome, exhausted
):
    _, _, state = candidate
    good = _rc_evidence(state)
    bad = {**good, "run_id": "124"}
    _mock_rc_evidence(monkeypatch, [bad, good])
    now = [601 if outcome == "already-expired" else 0]
    monkeypatch.setattr(flow, "_api_deadline", lambda: 600)
    monkeypatch.setattr(flow.time, "monotonic", lambda: now[0])
    run = subprocess.run
    fetches = []

    def fetch(args, **kwargs):
        if args[:2] == ["git", "fetch"]:
            fetches.append((args[-1], kwargs.get("timeout")))
            if args[-1] == bad["evidence_ref"]:
                now[0] = 601 if exhausted else 31
                if outcome == "timeout":
                    raise subprocess.TimeoutExpired(args, kwargs.get("timeout", 30))
        return run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fetch)
    if exhausted or outcome == "already-expired":
        with pytest.raises(flow._EvidenceDeadlineExceeded):
            flow._find_rc("9.9.0")
        assert len(fetches) == (0 if outcome == "already-expired" else 1)
        assert not (flow.STATE / "accepted-rc.json").exists()
    else:
        expected = "123" if outcome == "timeout" else "124"
        assert flow._find_rc("9.9.0")["run_id"] == expected
        assert all(timeout == 30 for _, timeout in fetches)


@pytest.mark.parametrize("fault", ["wrong-tag", "evidence-ref"])
def test_rc_tag_identity_requires_exact_commit_and_receipt_ref(
    candidate, monkeypatch, fault
):
    source, remote, state = candidate
    receipt = _rc_evidence(state)
    _mock_rc_evidence(monkeypatch, [receipt])
    if fault == "wrong-tag":
        _git(source, "commit", "--allow-empty", "-m", "Same tree different commit")
        other = _git(source, "rev-parse", "HEAD")
        assert other != receipt["source_sha"]
        assert flow._fingerprint(ref=other) == receipt["fingerprint"]
        _git(source, "push", str(remote), f"+{other}:{receipt['evidence_ref']}")
    else:
        receipt["evidence_ref"] = "refs/tags/rc-evidence/999"
    with pytest.raises(ValueError, match="does not match the receipt source|Untrusted"):
        flow._find_rc("9.9.0")
    assert not (flow.STATE / "accepted-rc.json").exists()


@pytest.mark.parametrize("fault", ["oid", "kind", "separator"])
def test_git_blob_fingerprint_rejects_corrupt_batch_headers(
    candidate, monkeypatch, fault
):
    run = subprocess.run

    def corrupt(args, **kwargs):
        result = run(args, **kwargs)
        if args == ["git", "cat-file", "--batch"]:
            header, content = result.stdout.split(b"\n", 1)
            oid, kind, size = header.split()
            if fault == "oid":
                oid = b"0" * 40
            elif fault == "kind":
                kind = b"tree"
            else:
                offset = int(size)
                content = content[:offset] + b"!" + content[offset + 1 :]
            result.stdout = b" ".join([oid, kind, size]) + b"\n" + content
        return result

    monkeypatch.setattr(subprocess, "run", corrupt)
    with pytest.raises(ValueError, match="Invalid Git blob evidence"):
        flow._fingerprint(ref="HEAD")


@pytest.mark.parametrize(
    "fault",
    [
        "deflate",
        "bzip2",
        "lzma",
        "encrypted",
        "unsupported-compression",
        "incomplete-read",
    ],
)
@pytest.mark.parametrize("exhausted", [False, True])
def test_rc_malformed_download_falls_back_within_shared_budget(
    candidate, monkeypatch, fault, exhausted
):
    _, _, state = candidate
    good = _rc_evidence(state)
    bad = {**good, "run_id": "124"}
    download = flow._download_receipt
    _mock_rc_evidence(monkeypatch, [bad, good])
    downloads = []

    def record_download(artifact_id):
        downloads.append(str(artifact_id))
        return download(artifact_id)

    monkeypatch.setattr(flow, "_download_receipt", record_download)
    buffer_type = io.BytesIO
    payload = buffer_type()
    compression = {"bzip2": zipfile.ZIP_BZIP2, "lzma": zipfile.ZIP_LZMA}.get(
        fault, zipfile.ZIP_DEFLATED
    )
    with zipfile.ZipFile(payload, "w", compression=compression) as archive:
        archive.writestr("receipt.json", json.dumps(good))
    valid = payload.getvalue()
    broken = bytearray(valid)
    if fault == "deflate":
        broken[30 + len("receipt.json")] = 7  # Reserved DEFLATE block type.
    elif fault == "bzip2":
        broken[30 + len("receipt.json")] = 0
    elif fault == "lzma":
        offset = 30 + len("receipt.json")
        broken[offset + 2 : offset + 4] = b"\x00\x00"
    elif fault == "encrypted":
        broken[6] |= 1
        central = broken.index(b"PK\x01\x02")
        broken[central + 8] |= 1
    elif fault == "unsupported-compression":
        broken[8:10] = (99).to_bytes(2, "little")
        central = broken.index(b"PK\x01\x02")
        broken[central + 10 : central + 12] = (99).to_bytes(2, "little")

    now = [0]
    monkeypatch.setattr(flow, "_api_deadline", lambda: 600)
    monkeypatch.setattr(flow.time, "monotonic", lambda: now[0])

    class ArchiveBuffer(buffer_type):
        def read(self, *args):
            if exhausted:
                now[0] = 601
            return super().read(*args)

    class BrokenBody(buffer_type):
        def read(self, *args):
            if exhausted:
                now[0] = 601
            raise IncompleteRead(b"partial", 100)

    class Opener:
        def open(self, request, *, timeout):
            if "/124/" in request.full_url:
                if fault == "incomplete-read":
                    return BrokenBody()
                return buffer_type(broken)
            return buffer_type(valid)

    monkeypatch.setattr(flow.urllib.request, "build_opener", lambda *args: Opener())
    monkeypatch.setattr(flow.io, "BytesIO", ArchiveBuffer)
    if exhausted:
        with pytest.raises(flow._EvidenceDeadlineExceeded):
            flow._find_rc("9.9.0")
        assert downloads == ["124"]
        assert not (flow.STATE / "accepted-rc.json").exists()
    else:
        assert flow._find_rc("9.9.0")["run_id"] == "123"
        assert downloads == ["124", "123"]


@pytest.mark.parametrize(
    "compression",
    [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA],
)
def test_rc_receipt_download_accepts_supported_compression(monkeypatch, compression):
    receipt = {"schema": 1}
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=compression) as archive:
        archive.writestr("receipt.json", json.dumps(receipt))

    class Opener:
        def open(self, request, *, timeout):
            return io.BytesIO(payload.getvalue())

    monkeypatch.setenv("GH_TOKEN", "fixture-token")
    monkeypatch.setattr(flow.urllib.request, "build_opener", lambda *args: Opener())
    assert flow._download_receipt("123") == receipt


@pytest.mark.parametrize("image_key", ["cpu", "jetson51", "models-jp51"])
@pytest.mark.parametrize("version", ["1.1build1", "1.1+build1"])
def test_rc_image_inventory_is_validated_before_recording(
    tmp_path, monkeypatch, image_key, version
):
    images = _load_images()
    policy = json.loads((ROOT / ".release/release-policy.json").read_text())
    plan = {
        **next(item for item in policy["images"] if item["key"] == image_key),
        "channel": "rc",
    }
    platform = plan["platforms"][0]
    inventory = [{"name": "distro-package", "version": version}]
    commands = []

    def command(*args, **kwargs):
        commands.append(args)
        return subprocess.CompletedProcess(
            args, 0, stdout=json.dumps(inventory), stderr=""
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(images, "_command", command)
    monkeypatch.setattr(
        images.subprocess, "run", lambda args, **kwargs: command(*args, **kwargs)
    )
    if version == "1.1build1":
        with pytest.raises(ValueError, match="1.1build1"):
            images._smoke(plan, digest="sha256:" + "a" * 64, platform=platform)
        assert not Path("image-smoke.json").exists()
    else:
        images._smoke(plan, digest="sha256:" + "a" * 64, platform=platform)
        assert json.loads(Path("image-smoke.json").read_text())["inventory"] == {
            "distro-package": version
        }
    if image_key in {"jetson51", "models-jp51"}:
        assert platform == "linux/arm64"
        assert any("--runtime" in args and "nvidia" in args for args in commands)


@pytest.mark.parametrize("version", ["1.1build1", "1.1+build1"])
def test_rc_package_inventory_is_validated_before_recording(
    tmp_path, monkeypatch, version
):
    import smoke

    packages = [
        {"name": "inference-sdk", "version": "9.9.0rc1"},
        {"name": "distro-package", "version": version},
    ]
    monkeypatch.setattr(
        smoke.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, 0, stdout=json.dumps(packages)
        ),
    )
    state = tmp_path / "release-state"
    wheel = tmp_path / "inference_sdk-9.9.0rc1-py3-none-any.whl"
    destination = state / "smoke/inference-sdk-py3.12.json"
    if version == "1.1build1":
        with pytest.raises(ValueError, match="1.1build1"):
            smoke._smoke(wheel, python="3.12", state=state)
        assert not destination.exists()
    else:
        smoke._smoke(wheel, python="3.12", state=state)
        assert json.loads(destination.read_text()) == {
            "inference-sdk": "9.9.0",
            "distro-package": version,
        }


@pytest.mark.parametrize("origin", ["image", "package"])
@pytest.mark.parametrize("version", ["1.1build1", "1.1+build1"])
def test_rc_receipt_revalidates_recorded_inventories(candidate, origin, version):
    _, _, state = candidate
    state["policy"] = {"images": [{"key": "cpu"}], "python_versions": ["3.12"]}
    flow._write(flow.STATE / "release.json", state)
    image_inventory = {"distro-package": version if origin == "image" else "1.0.0"}
    flow._write(
        flow.STATE / "images/cpu/image-receipt.json",
        {
            "key": "cpu",
            "source_sha": state["source_sha"],
            "smoke": {"linux/amd64": image_inventory},
        },
    )
    flow._write(flow.STATE / "dependency-receipt.json", {})
    flow._write(flow.STATE / "runtime-receipt.json", {})
    for name in flow.FIRST_PARTY:
        inventory = {
            name: "9.9.0",
            "distro-package": version if origin == "package" else "1.0.0",
        }
        flow._write(flow.STATE / "smoke" / f"{name}-py3.12.json", inventory)
    if version == "1.1build1":
        with pytest.raises(ValueError, match="1.1build1"):
            flow._receipt()
        assert not (flow.STATE / "receipt.json").exists()
    else:
        flow._receipt()
        receipt = json.loads((flow.STATE / "receipt.json").read_text())
        assert receipt["complete"] is True
        recorded = (
            receipt["images"][0]["smoke"]["linux/amd64"]
            if origin == "image"
            else receipt["smoke"]["inference-sdk-py3.12"]
        )
        assert recorded["distro-package"] == version


@pytest.mark.parametrize("origin", ["image", "package"])
def test_stable_receipt_rejects_invalid_recorded_inventory(candidate, origin):
    _, _, state = candidate
    receipt = _rc_evidence(state)
    receipt["evidence_ref"] = flow._evidence_ref(receipt["run_id"])
    if origin == "image":
        inventory = next(iter(receipt["images"][0]["smoke"].values()))
    else:
        inventory = next(iter(receipt["smoke"].values()))
    inventory["distro-package"] = "1.1build1"
    run = {
        "id": receipt["run_id"],
        "head_sha": receipt["trigger_sha"],
        "head_branch": receipt["branch"],
    }
    with pytest.raises(ValueError, match="1.1build1"):
        flow._validate_rc_receipt(receipt, run=run)


def test_package_smoke_import_does_not_require_tomllib():
    script = """
import runpy
import sys

class WithoutTomllib:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tomllib":
            raise ModuleNotFoundError("tomllib is unavailable on Python 3.10")

sys.meta_path.insert(0, WithoutTomllib())
sys.path.insert(0, sys.argv[1])
smoke = runpy.run_path(sys.argv[1] + "/smoke.py")
assert smoke["_inventory"]([{"name": "distro-package", "version": "1.1+build1"}]) == {"distro-package": "1.1+build1"}
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(ROOT / ".release/pypi")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
