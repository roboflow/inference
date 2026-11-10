"""Publication contracts: run only in the .release/pypi/Dockerfile image."""

import hashlib
import importlib.metadata
import importlib.util
import io
import json
import shutil
import subprocess
import urllib.error
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "pypi_release", ROOT / ".release/pypi/release.py"
)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)
VERSION = "9.8.7rc2"
FILENAME = f"inference_sdk-{VERSION}-py3-none-any.whl"
DIST_INFO = f"inference_sdk-{VERSION}.dist-info"


def _sdk_wheel(*, changed=None, timestamp=(2026, 1, 1, 0, 0, 0), mode=0o100644):
    entries = {
        "inference_sdk/__init__.py": b"class InferenceHTTPClient: pass\n",
        f"{DIST_INFO}/METADATA": (
            f"Metadata-Version: 2.4\nName: inference-sdk\nVersion: {VERSION}\n"
            "Requires-Dist: requests>=2\nRequires-Python: >=3.10,<3.14\n\nREADME\n"
        ).encode(),
        f"{DIST_INFO}/WHEEL": b"Wheel-Version: 1.0\nTag: py3-none-any\n",
        f"{DIST_INFO}/RECORD": b"fixture record\n",
        f"{DIST_INFO}/licenses/LICENSE": b"License text\n",
    }
    entries.update(changed or {})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, contents in entries.items():
            info = zipfile.ZipInfo(name, date_time=timestamp)
            info.external_attr = mode << 16
            archive.writestr(info, contents)

    return buffer.getvalue()


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PUBLISH", raising=False)
    monkeypatch.delenv("EXPECTED_VERSION", raising=False)
    version_file = tmp_path / "inference/core/version.py"
    version_file.parent.mkdir(parents=True)
    version_file.write_text(f'__version__ = "{VERSION}"\n')
    wheel = tmp_path / "dist/sdk" / FILENAME
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(_sdk_wheel())
    return tmp_path


def _registry(monkeypatch, *, data=None, status=None, mutate=None):
    payload = data or _sdk_wheel()
    metadata = {
        "info": {"version": VERSION},
        "urls": [
            {
                "filename": FILENAME,
                "url": f"https://files.pythonhosted.org/{FILENAME}",
                "digests": {"sha256": hashlib.sha256(payload).hexdigest()},
                "yanked": False,
            }
        ],
    }
    if mutate:
        mutate(metadata)

    calls = []

    def read(url):
        calls.append(url)
        if url.endswith("/json"):
            if status:
                raise urllib.error.HTTPError(url, status, "fixture", {}, None)
            return json.dumps(metadata).encode()

        return payload

    monkeypatch.setattr(release, "_read_url", read)
    return calls


def test_missing_sdk_stages_only_the_original_wheel(candidate, monkeypatch):
    _registry(monkeypatch, status=404)
    release._run("prepare-sdk", wait_seconds=0)
    staged = candidate / "dist/sdk-upload"
    assert [p.name for p in staged.iterdir()] == [FILENAME]
    assert (staged / FILENAME).read_bytes() == _sdk_wheel()
    assert json.loads((candidate / "dist/sdk-decision.json").read_text())["upload"]
    assert not (candidate / "dist/sdk-receipt.json").exists()


def test_prepublished_sdk_with_different_zip_timestamps_is_verified(
    candidate, monkeypatch
):
    published = _sdk_wheel(timestamp=(2026, 2, 2, 2, 2, 2))
    assert published != _sdk_wheel()
    _registry(monkeypatch, data=published)
    release._run("prepare-sdk", wait_seconds=0)
    assert list((candidate / "dist/sdk-upload").iterdir()) == []
    assert not json.loads((candidate / "dist/sdk-decision.json").read_text())["upload"]
    receipt = json.loads((candidate / "dist/sdk-receipt.json").read_text())
    assert receipt["published_file"]["sha256"] == hashlib.sha256(published).hexdigest()
    assert receipt["published_file"]["sha256"] != receipt["candidate_file"]["sha256"]


@pytest.mark.parametrize(
    "entry",
    [
        "inference_sdk/__init__.py",
        f"{DIST_INFO}/METADATA",
        f"{DIST_INFO}/WHEEL",
        f"{DIST_INFO}/RECORD",
        f"{DIST_INFO}/licenses/LICENSE",
    ],
)
def test_same_version_different_contents_cannot_be_reused(
    candidate, monkeypatch, entry
):
    with zipfile.ZipFile(io.BytesIO(_sdk_wheel())) as archive:
        changed = {entry: archive.read(entry) + b"changed\n"}
    _registry(monkeypatch, data=_sdk_wheel(changed=changed))
    with pytest.raises(ValueError, match="contents differ"):
        release._run("prepare-sdk", wait_seconds=0)
    assert not (candidate / "dist/sdk-decision.json").exists()


def test_file_permission_changes_are_not_ignored(candidate, monkeypatch):
    _registry(monkeypatch, data=_sdk_wheel(mode=0o100755))
    with pytest.raises(ValueError, match="contents differ"):
        release._run("verify-sdk", wait_seconds=0)


def test_retry_after_completed_upload_does_not_upload_again(candidate, monkeypatch):
    _registry(monkeypatch, status=404)
    release._run("prepare-sdk", wait_seconds=0)
    original = (candidate / "dist/sdk-upload" / FILENAME).read_bytes()
    _registry(monkeypatch, data=original)
    release._run("prepare-sdk", wait_seconds=0)
    assert list((candidate / "dist/sdk-upload").iterdir()) == []
    assert not json.loads((candidate / "dist/sdk-decision.json").read_text())["upload"]


def test_file_visibility_delay_never_authorizes_another_upload(candidate, monkeypatch):
    calls = _registry(monkeypatch)
    read = release._read_url
    missing = [True]

    def delayed(url):
        if not url.endswith("/json") and missing:
            missing.pop()
            raise urllib.error.HTTPError(url, 404, "file not visible", {}, None)
        return read(url)

    monkeypatch.setattr(release, "_read_url", delayed)
    monkeypatch.setattr(release.time, "sleep", lambda seconds: None)
    release._run("prepare-sdk", wait_seconds=30)
    assert len(calls) == 3
    assert not json.loads((candidate / "dist/sdk-decision.json").read_text())["upload"]


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
def test_registry_errors_never_authorize_an_upload(candidate, monkeypatch, status):
    _registry(monkeypatch, status=status)
    with pytest.raises((ValueError, urllib.error.HTTPError)):
        release._run("prepare-sdk", wait_seconds=0)
    assert not (candidate / "dist/sdk-decision.json").exists()


def test_missing_sdk_blocks_consumers_until_timeout(candidate, monkeypatch):
    calls = _registry(monkeypatch, status=404)
    now = [0]
    monkeypatch.setattr(release.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        release.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    expected = release._inventory(
        candidate / "dist/sdk", names=release.SDK, version=VERSION
    )[0]
    with pytest.raises(ValueError, match="timed out without publishing consumers"):
        release._reconcile_sdk(expected, allow_missing=False, wait_seconds=20)
    assert len(calls) == 3


def test_gate_waits_for_sdk_visibility_then_verifies_bytes(candidate, monkeypatch):
    _registry(monkeypatch)
    read = release._read_url
    failures = [404, 503]

    def delayed(url):
        if url.endswith("/json") and failures:
            raise urllib.error.HTTPError(url, failures.pop(0), "fixture", {}, None)
        return read(url)

    monkeypatch.setattr(release, "_read_url", delayed)
    monkeypatch.setattr(release.time, "sleep", lambda seconds: None)
    release._run("verify-sdk", wait_seconds=60)
    assert failures == []
    assert (
        json.loads((candidate / "dist/sdk-receipt.json").read_text())["status"]
        == "verified"
    )


@pytest.mark.parametrize("fault", ["hash", "yanked", "extra-file", "empty", "version"])
def test_invalid_registry_releases_are_rejected(candidate, monkeypatch, fault):
    def mutate(metadata):
        if fault == "hash":
            metadata["urls"][0]["digests"]["sha256"] = "0" * 64
        elif fault == "yanked":
            metadata["urls"][0]["yanked"] = True
        elif fault == "extra-file":
            metadata["urls"].append(dict(metadata["urls"][0]))
        elif fault == "empty":
            metadata["urls"] = []
        else:
            metadata["info"]["version"] = "9.8.6"

    _registry(monkeypatch, mutate=mutate)
    with pytest.raises(ValueError):
        release._run("prepare-sdk", wait_seconds=0)


@pytest.mark.parametrize(
    "url", ["http://pypi.org/file", "https://example.com/file", "file:///etc/passwd"]
)
def test_unexpected_download_urls_are_rejected(url):
    with pytest.raises(ValueError, match="unexpected download URL"):
        release._read_url(url)


def test_wrong_expected_version_fails_before_build_or_registry(candidate, monkeypatch):
    monkeypatch.setenv("PUBLISH", "true")
    monkeypatch.setenv("EXPECTED_VERSION", "9.8.7")
    with pytest.raises(ValueError, match="does not match source"):
        release._run("build-sdk", wait_seconds=0)


def test_publication_requires_explicit_version(candidate, monkeypatch):
    monkeypatch.setenv("PUBLISH", "true")
    with pytest.raises(ValueError, match="requires expected_version"):
        release._run("prepare-sdk", wait_seconds=0)


def test_release_tag_and_later_rc_versions_are_accepted(candidate, monkeypatch):
    monkeypatch.setenv("PUBLISH", "true")
    monkeypatch.setenv("EXPECTED_VERSION", f"v{VERSION}")
    assert release._expected_version() == VERSION


def test_sdk_rejects_foreign_package_payload():
    with pytest.raises(ValueError, match="only inference_sdk"):
        release._wheel(
            _sdk_wheel(changed={"inference/__init__.py": b""}),
            filename=FILENAME,
            version=VERSION,
        )


def test_workflow_ownership_and_credential_boundaries():
    for filename in ["publish.pypi.yml", "publish.pypi.inference_sdk.yml"]:
        workflow = yaml.load(
            (ROOT / ".github/workflows" / filename).read_text(), Loader=yaml.BaseLoader
        )
        assert workflow["on"]["release"]["types"] == ["created"]
        assert (
            workflow["on"]["workflow_dispatch"]["inputs"]["publish"]["default"]
            == "false"
        )
        assert "id-token" not in workflow["permissions"]
        assert workflow["jobs"]["publish"]["permissions"]["id-token"] == "write"
        assert workflow["jobs"]["publish"]["needs"] == "build"
        uploads = [
            step
            for step in workflow["jobs"]["publish"]["steps"]
            if step.get("uses", "").startswith("pypa/")
        ]
        assert len(uploads) == 1
        expected = "dist/sdk-upload/" if "inference_sdk" in filename else "dist/pypi/"
        assert uploads[0]["with"]["packages-dir"] == expected


def test_sdk_build_needs_no_runtime_cli_or_standalone_source(tmp_path, monkeypatch):
    checkout = tmp_path / "sdk-only"
    checkout.mkdir()
    for relative in [
        "Makefile",
        "README.md",
        "LICENSE",
        "LICENSE.core",
        "inference/core/version.py",
        ".release/pypi/inference.sdk.setup.py",
        "requirements/requirements.sdk.http.txt",
        "requirements/requirements.sdk.webrtc.txt",
    ]:
        destination = checkout / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    shutil.copytree(ROOT / "inference_sdk", checkout / "inference_sdk")
    monkeypatch.chdir(checkout)
    monkeypatch.delenv("PUBLISH", raising=False)
    monkeypatch.delenv("EXPECTED_VERSION", raising=False)
    release._run("build-sdk", wait_seconds=0)
    files = list((checkout / "dist/sdk").iterdir())
    assert len(files) == 1 and files[0].name.startswith("inference_sdk-")
    assert not (checkout / "dist/pypi").exists()


def test_real_split_build_preserves_all_six_package_contents(tmp_path, monkeypatch):
    """Exercise actual setup scripts, independent of registry fixture wheels."""
    assert importlib.util.find_spec("torch") is None
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        importlib.metadata.version("inference-models")
    checkout = tmp_path / "checkout"
    shutil.copytree(
        ROOT,
        checkout,
        ignore=shutil.ignore_patterns(
            ".git", "dist", "build", "*.egg-info", "__pycache__", ".pytest_cache"
        ),
    )
    monkeypatch.chdir(checkout)
    monkeypatch.delenv("PUBLISH", raising=False)
    monkeypatch.delenv("EXPECTED_VERSION", raising=False)
    version = release._source_version()
    # setup.py writes SDK/CLI version modules. Use independent clean trees so
    # one build cannot change the other build's inputs.
    baseline_checkout = tmp_path / "baseline"
    shutil.copytree(checkout, baseline_checkout)
    subprocess.run(
        ["make", "create_pypi_wheels"],
        cwd=baseline_checkout,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    baseline = release._inventory(
        baseline_checkout / "dist/pypi",
        names=release.SDK | release.RUNTIME,
        version=version,
    )
    release._run("build-runtime", wait_seconds=0)
    split = release._inventory(
        checkout / "dist/pypi", names=release.RUNTIME, version=version
    )
    split += release._inventory(
        checkout / "dist/sdk", names=release.SDK, version=version
    )
    assert {file["name"]: file["content_sha256"] for file in split} == {
        file["name"]: file["content_sha256"] for file in baseline
    }
    assert not list((checkout / "dist/pypi").glob("inference_sdk*"))
    assert len(split) == 6

    # Exercise the real consumer gate with the real built SDK and registry I/O
    # replaced at the transport boundary only.
    sdk_path = next((checkout / "dist/sdk").glob("*.whl"))
    sdk_data = sdk_path.read_bytes()
    api_url = f"https://pypi.org/pypi/inference-sdk/{version}/json"
    file_url = f"https://files.pythonhosted.org/{sdk_path.name}"
    responses = {
        api_url: json.dumps(
            {
                "info": {"version": version},
                "urls": [
                    {
                        "filename": sdk_path.name,
                        "url": file_url,
                        "digests": {"sha256": hashlib.sha256(sdk_data).hexdigest()},
                        "yanked": False,
                    }
                ],
            }
        ).encode(),
        file_url: sdk_data,
    }
    monkeypatch.setattr(release, "_read_url", lambda url: responses[url])
    release._run("gate-sdk", wait_seconds=0)
    assert (
        json.loads((checkout / "dist/sdk-receipt.json").read_text())["published_file"][
            "sha256"
        ]
        == hashlib.sha256(sdk_data).hexdigest()
    )

    # A leaked SDK wheel must stop the consumer publication gate before network I/O.
    shutil.copyfile(
        next((checkout / "dist/sdk").glob("*.whl")),
        checkout / "dist/pypi" / next((checkout / "dist/sdk").glob("*.whl")).name,
    )
    with pytest.raises(ValueError, match="Unexpected publication inventory"):
        release._run("gate-sdk", wait_seconds=0)
