"""Publication contracts: run only in the .release/pypi/Dockerfile image."""

import hashlib
import http.client
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
from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "pypi_release", ROOT / ".release/pypi/release.py"
)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)
VERSION = "9.8.7rc2"
FILENAME = f"inference_sdk-{VERSION}-py3-none-any.whl"
DIST_INFO = f"inference_sdk-{VERSION}.dist-info"


def _sdk_wheel(
    *,
    changed=None,
    timestamp=(2026, 1, 1, 0, 0, 0),
    mode=0o100644,
    compression=zipfile.ZIP_STORED,
    omitted=(),
):
    entries = {
        "inference_sdk/__init__.py": b"class InferenceHTTPClient: pass\n",
        f"{DIST_INFO}/METADATA": (
            f"Metadata-Version: 2.4\nName: inference-sdk\nVersion: {VERSION}\n"
            "Requires-Dist: requests>=2\nRequires-Python: >=3.10,<3.14\n\nREADME\n"
        ).encode(),
        f"{DIST_INFO}/WHEEL": b"Wheel-Version: 1.0\nGenerator: setuptools (83.0.0)\nTag: py3-none-any\n",
        f"{DIST_INFO}/RECORD": b"fixture record\n",
        f"{DIST_INFO}/licenses/LICENSE": b"License text\n",
    }
    entries.update(changed or {})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, contents in entries.items():
            if name in omitted:
                continue
            info = zipfile.ZipInfo(name, date_time=timestamp)
            info.external_attr = mode << 16
            info.compress_type = compression
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


def _runtime_wheels(candidate, *, requirements=None, invalid_package=None):
    directory = candidate / "dist/pypi"
    directory.mkdir(parents=True, exist_ok=True)
    if requirements is None:
        requirements = [f"inference-sdk=={VERSION}"]
    for name in release.RUNTIME:
        distribution = name.replace("-", "_")
        path = directory / f"{distribution}-{VERSION}-py3-none-any.whl"
        with zipfile.ZipFile(path, "w") as archive:
            metadata = f"Name: {name}\nVersion: {VERSION}\n"
            package_requirements = (
                [f"inference-sdk=={VERSION}"]
                if invalid_package is not None and name != invalid_package
                else requirements
            )
            metadata += "".join(
                f"Requires-Dist: {raw}\n" for raw in package_requirements
            )
            archive.writestr(f"{distribution}-{VERSION}.dist-info/METADATA", metadata)


@pytest.mark.parametrize("operation", ["gate-sdk", "verify-sdk"])
def test_missing_sdk_blocks_run_without_relying_on_receipt_upload(
    candidate, monkeypatch, operation
):
    _runtime_wheels(candidate)
    calls = _registry(monkeypatch, status=404)
    with pytest.raises(ValueError, match="timed out without publishing consumers"):
        release._run(operation, wait_seconds=0)
    assert len(calls) == 1
    assert not (candidate / "dist/sdk-receipt.json").exists()


@pytest.mark.parametrize("operation", ["prepare-sdk", "verify-sdk", "gate-sdk"])
@pytest.mark.parametrize("entry", [f"{DIST_INFO}/WHEEL", f"{DIST_INFO}/METADATA"])
def test_sdk_publisher_and_consumer_use_distinct_content_policies(
    candidate, monkeypatch, operation, entry
):
    _runtime_wheels(candidate)
    with zipfile.ZipFile(io.BytesIO(_sdk_wheel())) as archive:
        original = archive.read(entry)
    changed = (
        original.replace(b"setuptools (83.0.0)", b"setuptools (84.2.1)")
        if entry.endswith("/WHEEL")
        else original.replace(b"README", b"Earlier README")
    )
    published = _sdk_wheel(changed={entry: changed})
    _registry(monkeypatch, data=published)
    monkeypatch.setenv("SOURCE_REVISION", "fixture-source")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    if operation != "gate-sdk":
        with pytest.raises(ValueError, match="contents differ"):
            release._run(operation, wait_seconds=0)
        assert not (candidate / "dist/sdk-receipt.json").exists()
        return

    release._run(operation, wait_seconds=0)
    receipt = json.loads((candidate / "dist/sdk-receipt.json").read_text())
    assert receipt["verification_policy"] == "registry-integrity"
    assert receipt["published_file"]["sha256"] == hashlib.sha256(published).hexdigest()
    assert receipt["source_revision"] == "fixture-source"
    assert receipt["verification_run_id"] == "123"


@pytest.mark.parametrize(
    "error",
    [
        http.client.IncompleteRead(b"truncated"),
        http.client.HTTPException("transport"),
        ConnectionResetError("reset"),
    ],
)
@pytest.mark.parametrize("operation", ["prepare-sdk", "verify-sdk", "gate-sdk"])
def test_transport_failures_retry_then_verify(candidate, monkeypatch, error, operation):
    _runtime_wheels(candidate)
    calls = _registry(monkeypatch)
    read = release._read_url
    failures = [error]

    def flaky(url):
        if failures:
            raise failures.pop()
        return read(url)

    monkeypatch.setattr(release, "_read_url", flaky)
    sleeps = []
    monkeypatch.setattr(release.time, "sleep", sleeps.append)
    release._run(operation, wait_seconds=20)
    assert len(sleeps) == 1
    assert len(calls) == 2


@pytest.mark.parametrize(
    "error",
    [
        http.client.IncompleteRead(b"truncated"),
        http.client.HTTPException("transport"),
        ConnectionResetError("reset"),
    ],
)
def test_transport_timeout_is_a_clean_cli_failure(candidate, monkeypatch, error):
    now = [0]
    calls = []

    def unavailable(url):
        calls.append(url)
        raise error

    monkeypatch.setattr(release, "_read_url", unavailable)
    monkeypatch.setattr(release.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        release.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    result = CliRunner().invoke(release._main, ["verify-sdk", "--wait-seconds", "20"])
    assert result.exit_code == 1
    assert "Error: Registry unavailable:" in result.output
    assert "timed out without publishing consumers" in result.output
    assert "Traceback" not in result.output
    assert len(calls) == 3


@pytest.mark.parametrize(
    "requirements",
    [
        [],
        [f"inference-sdk=={VERSION}", f"inference-sdk=={VERSION}"],
        [f"inference-sdk>={VERSION}"],
        ["inference-sdk==9.8.6"],
        [f'inference-sdk=={VERSION}; python_version >= "3.10"'],
        ["inference-sdk @ https://example.com/inference_sdk.whl"],
    ],
    ids=["missing", "duplicate", "loose", "wrong-version", "marker", "direct-url"],
)
@pytest.mark.parametrize("package", sorted(release.RUNTIME))
def test_runtime_pins_fail_before_network(
    candidate, monkeypatch, requirements, package
):
    _runtime_wheels(candidate, requirements=requirements, invalid_package=package)
    calls = _registry(monkeypatch)
    with pytest.raises(
        ValueError, match=f"Expected unconditional inference-sdk=={VERSION}"
    ):
        release._run("gate-sdk", wait_seconds=0)
    assert calls == []
    assert not (candidate / "dist/sdk-receipt.json").exists()


def test_runtime_exact_pin_allows_consumer_gate(candidate, monkeypatch):
    _runtime_wheels(candidate)
    calls = _registry(monkeypatch)
    release._run("gate-sdk", wait_seconds=0)
    assert len(calls) == 2


@pytest.mark.parametrize(
    "helper", ["published", "reconcile-required", "reconcile-optional"]
)
def test_reconcile_defaults_to_strict_publisher_policy(candidate, monkeypatch, helper):
    _registry(monkeypatch, data=_sdk_wheel(changed={f"{DIST_INFO}/WHEEL": b"older"}))
    expected = release._inventory(
        candidate / "dist/sdk", names=release.SDK, version=VERSION
    )[0]
    with pytest.raises(ValueError, match="contents differ"):
        if helper == "published":
            release._published_sdk(expected)
        elif helper == "reconcile-required":
            release._reconcile_sdk(expected, allow_missing=False, wait_seconds=0)
        else:
            release._reconcile_sdk(expected, allow_missing=True, wait_seconds=0)


@pytest.mark.parametrize("operation", ["prepare-sdk", "verify-sdk", "gate-sdk"])
@pytest.mark.parametrize(
    "fault",
    [
        "identity",
        "version",
        "ownership",
        "missing-sdk",
        "missing-metadata",
        "malformed",
    ],
)
def test_invalid_published_wheel_fails_immediately(
    candidate, monkeypatch, operation, fault
):
    _runtime_wheels(candidate)
    if fault == "identity":
        data = _sdk_wheel(
            changed={
                f"{DIST_INFO}/METADATA": f"Name: foreign\nVersion: {VERSION}\n".encode()
            }
        )
        message = "Filename and METADATA disagree"
    elif fault == "version":
        data = _sdk_wheel(
            changed={f"{DIST_INFO}/METADATA": b"Name: inference-sdk\nVersion: 9.8.6\n"}
        )
        message = "Filename and METADATA disagree"
    elif fault == "ownership":
        data = _sdk_wheel(changed={"inference/__init__.py": b"foreign"})
        message = "only inference_sdk"
    elif fault == "missing-sdk":
        data = _sdk_wheel(omitted=["inference_sdk/__init__.py"])
        message = "only inference_sdk"
    elif fault == "missing-metadata":
        data = _sdk_wheel(omitted=[f"{DIST_INFO}/METADATA"])
        message = "Expected one METADATA"
    else:
        data = b"not a zip"
        message = "File is not a zip file"
    calls = _registry(monkeypatch, data=data)
    sleeps = []
    monkeypatch.setattr(release.time, "sleep", sleeps.append)
    with pytest.raises((ValueError, zipfile.BadZipFile), match=message):
        release._run(operation, wait_seconds=20)
    assert len(calls) == 2
    assert sleeps == []


@pytest.mark.parametrize("status", [401, 403])
def test_auth_errors_are_not_retried_or_absent(candidate, monkeypatch, status):
    calls = _registry(monkeypatch, status=status)
    sleeps = []
    monkeypatch.setattr(release.time, "sleep", sleeps.append)
    with pytest.raises(urllib.error.HTTPError):
        release._run("prepare-sdk", wait_seconds=20)
    assert len(calls) == 1
    assert sleeps == []


def test_local_os_errors_are_not_registry_retries(candidate, monkeypatch):
    def disk_error(url):
        raise OSError("local disk failure")

    monkeypatch.setattr(release, "_read_url", disk_error)
    sleeps = []
    monkeypatch.setattr(release.time, "sleep", sleeps.append)
    with pytest.raises(OSError, match="local disk failure"):
        release._run("verify-sdk", wait_seconds=20)
    assert sleeps == []


def _response(monkeypatch, *, final_url, payload=b"body"):
    reads = []

    class Response(io.BytesIO):
        def geturl(self):
            return final_url

        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    monkeypatch.setattr(
        release.urllib.request, "urlopen", lambda request, timeout: Response(payload)
    )
    return reads


@pytest.mark.parametrize(
    "final_url", ["http://pypi.org/file", "https://example.com/file"]
)
def test_registry_redirect_allowlist(monkeypatch, final_url):
    reads = _response(monkeypatch, final_url=final_url)
    with pytest.raises(ValueError, match="Unexpected registry redirect"):
        release._read_url("https://pypi.org/file")
    assert reads == []


@pytest.mark.parametrize("host", ["pypi.org", "files.pythonhosted.org"])
@pytest.mark.parametrize("size", [8, 9])
def test_registry_download_byte_boundary(monkeypatch, host, size):
    monkeypatch.setattr(release, "MAX_RESPONSE_BYTES", 8)
    reads = _response(
        monkeypatch, final_url=f"https://{host}/file", payload=b"a" * size
    )
    if size == 8:
        assert release._read_url(f"https://{host}/file") == b"a" * size
    else:
        with pytest.raises(
            ValueError, match="Registry response exceeds the SDK size limit"
        ):
            release._read_url(f"https://{host}/file")
    assert reads == [9]


def test_sdk_uncompressed_size_limit(monkeypatch):
    data = _sdk_wheel(
        changed={"inference_sdk/large.py": b"x" * 10000},
        compression=zipfile.ZIP_DEFLATED,
    )
    assert len(data) < 4096
    monkeypatch.setattr(release, "MAX_RESPONSE_BYTES", 4096)
    with pytest.raises(ValueError, match="Uncompressed SDK exceeds the size limit"):
        release._wheel(data, filename=FILENAME, version=VERSION)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        exact_size = sum(entry.file_size for entry in archive.infolist())
    monkeypatch.setattr(release, "MAX_RESPONSE_BYTES", exact_size)
    assert (
        release._wheel(data, filename=FILENAME, version=VERSION)["name"]
        == "inference-sdk"
    )


@pytest.mark.parametrize("fault", ["traversal", "absolute", "duplicate"])
def test_invalid_zip_entries_rejected_without_sdk_ownership_masking(fault):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            f"inference-{VERSION}.dist-info/METADATA",
            f"Name: inference\nVersion: {VERSION}\n",
        )
        if fault == "duplicate":
            archive.writestr("inference/payload.py", b"first")
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("inference/payload.py", b"second")
        else:
            archive.writestr(
                "../escape.py" if fault == "traversal" else "/escape.py", b"payload"
            )
    with pytest.raises(ValueError, match="Invalid wheel entries"):
        release._wheel(
            buffer.getvalue(),
            filename=f"inference-{VERSION}-py3-none-any.whl",
            version=VERSION,
        )


def test_missing_sdk_stages_only_the_original_wheel(candidate, monkeypatch):
    _registry(monkeypatch, status=404)
    release._run("prepare-sdk", wait_seconds=0)
    staged = candidate / "dist/sdk-upload"
    assert [p.name for p in staged.iterdir()] == [FILENAME]
    assert (staged / FILENAME).read_bytes() == _sdk_wheel()
    assert json.loads((candidate / "dist/sdk-decision.json").read_text())["upload"]
    assert not (candidate / "dist/sdk-receipt.json").exists()


@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_prepublished_sdk_with_different_zip_timestamps_is_verified(
    candidate, monkeypatch, compression
):
    published = _sdk_wheel(timestamp=(2026, 2, 2, 2, 2, 2), compression=compression)
    assert published != _sdk_wheel()
    _registry(monkeypatch, data=published)
    release._run("prepare-sdk", wait_seconds=0)
    assert list((candidate / "dist/sdk-upload").iterdir()) == []
    assert not json.loads((candidate / "dist/sdk-decision.json").read_text())["upload"]
    receipt = json.loads((candidate / "dist/sdk-receipt.json").read_text())
    assert receipt["verification_policy"] == "candidate-content"
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
@pytest.mark.parametrize("operation", ["prepare-sdk", "verify-sdk"])
def test_same_version_different_contents_cannot_be_reused(
    candidate, monkeypatch, entry, operation
):
    with zipfile.ZipFile(io.BytesIO(_sdk_wheel())) as archive:
        changed = {entry: archive.read(entry) + b"changed\n"}
    _registry(monkeypatch, data=_sdk_wheel(changed=changed))
    with pytest.raises(ValueError, match="contents differ"):
        release._run(operation, wait_seconds=0)
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


@pytest.mark.parametrize(
    "fault", ["hash", "yanked", "extra-file", "empty", "version", "filename"]
)
@pytest.mark.parametrize("operation", ["prepare-sdk", "verify-sdk", "gate-sdk"])
def test_invalid_registry_releases_are_rejected(
    candidate, monkeypatch, fault, operation
):
    _runtime_wheels(candidate)

    def mutate(metadata):
        if fault == "hash":
            metadata["urls"][0]["digests"]["sha256"] = "0" * 64
        elif fault == "yanked":
            metadata["urls"][0]["yanked"] = True
        elif fault == "extra-file":
            metadata["urls"].append(dict(metadata["urls"][0]))
        elif fault == "empty":
            metadata["urls"] = []
        elif fault == "version":
            metadata["info"]["version"] = "9.8.6"
        else:
            metadata["urls"][0]["filename"] = "foreign-9.8.7rc2-py3-none-any.whl"

    _registry(monkeypatch, mutate=mutate)
    with pytest.raises(ValueError):
        release._run(operation, wait_seconds=0)


@pytest.mark.parametrize(
    "url", ["http://pypi.org/file", "https://example.com/file", "file:///etc/passwd"]
)
def test_unexpected_download_urls_are_rejected(url, monkeypatch):
    reads = _response(monkeypatch, final_url="https://pypi.org/file")
    with pytest.raises(ValueError, match="unexpected download URL"):
        release._read_url(url)
    assert reads == []


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
    workflow = yaml.load(
        (ROOT / ".github/workflows/publish.pypi.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    assert not (ROOT / ".github/workflows/publish.pypi.inference_sdk.yml").exists()
    assert workflow["on"]["release"]["types"] == ["created"]
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert inputs["publish"]["default"] == "false"
    assert inputs["packages"]["default"] == "all"
    assert inputs["packages"]["options"] == ["sdk", "runtime", "all", "dependencies"]
    assert "id-token" not in workflow["permissions"]
    jobs = workflow["jobs"]
    assert jobs["publish_sdk"]["needs"] == "build_sdk"
    assert set(jobs["publish_runtime"]["needs"]) == {
        "build_runtime",
        "publish_sdk",
        "prepare_dependencies",
    }

    for name, job in jobs.items():
        assert "environment" not in job
        assert "uses" not in job  # Upload stays in the already trusted workflow.
        uploads = [
            step for step in job["steps"] if step.get("uses", "").startswith("pypa/")
        ]
        if name in {"build_sdk", "build_runtime", "prepare_dependencies"}:
            assert "id-token" not in job.get("permissions", {})
            assert not uploads
            continue

        assert job["permissions"]["id-token"] == "write"
        assert len(uploads) == 1
        expected = "dist/sdk-upload/" if name == "publish_sdk" else "dist/pypi/"
        assert uploads[0]["with"]["packages-dir"] == expected

    # The real registry gate must run before the consumer upload in every mode.
    steps = jobs["publish_runtime"]["steps"]
    gate = next(i for i, step in enumerate(steps) if "gate-sdk" in step.get("run", ""))
    upload = next(
        i for i, step in enumerate(steps) if step.get("uses", "").startswith("pypa/")
    )
    assert gate < upload
    assert "if" not in steps[gate]


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


@pytest.mark.parametrize("local_environments", [False, True])
def test_real_split_build_preserves_all_six_package_contents(
    tmp_path, monkeypatch, local_environments
):
    """Exercise actual setup scripts, independent of registry fixture wheels."""
    assert importlib.util.find_spec("torch") is None
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        importlib.metadata.version("inference-models")
    if local_environments:
        source = tmp_path / "source-with-environments"
        shutil.copytree(
            ROOT,
            source,
            symlinks=True,
            ignore=shutil.ignore_patterns(".git", ".venv", "venv", "__pycache__"),
        )
        for name in [".venv", "venv"]:
            binary = source / name / "bin/python"
            binary.parent.mkdir(parents=True)
            binary.symlink_to("/missing/host/python")
        monkeypatch.setitem(globals(), "ROOT", source)
    checkout = tmp_path / "checkout"
    shutil.copytree(
        ROOT,
        checkout,
        ignore=shutil.ignore_patterns(
            ".git",
            "dist",
            "build",
            "*.egg-info",
            "__pycache__",
            ".pytest_cache",
            ".venv",
            "venv",
        ),
        symlinks=True,
    )
    monkeypatch.chdir(checkout)
    monkeypatch.delenv("PUBLISH", raising=False)
    monkeypatch.delenv("EXPECTED_VERSION", raising=False)
    version = release._source_version()
    # setup.py writes SDK/CLI version modules. Use independent clean trees so
    # one build cannot change the other build's inputs.
    baseline_checkout = tmp_path / "baseline"
    shutil.copytree(checkout, baseline_checkout, symlinks=True)
    assert not (checkout / ".venv").exists()
    assert not (checkout / "venv").exists()
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
