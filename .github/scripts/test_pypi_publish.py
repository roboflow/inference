"""Per-file publication recovery contracts, executed in the release Docker image."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_runtime_upload_is_preflighted_and_stages_only_missing_files():
    config = yaml.load(
        (ROOT / ".github/workflows/publish.pypi.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    steps = config["jobs"]["publish_runtime"]["steps"]
    prepare = next(
        i for i, step in enumerate(steps) if "publish.sh prepare" in step.get("run", "")
    )
    upload = next(
        i for i, step in enumerate(steps) if step.get("uses", "").startswith("pypa/")
    )
    verify = next(
        i for i, step in enumerate(steps) if "publish.sh verify" in step.get("run", "")
    )
    assert prepare < upload < verify
    assert steps[upload]["with"]["packages-dir"] == "dist/runtime-upload/"
    assert "skip-existing" not in steps[upload]["with"]


import hashlib
import importlib.util
import io
import json
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.error
import zipfile

import pytest
from click.testing import CliRunner

sys.path.insert(0, str(ROOT / ".release/pypi"))
SPEC = importlib.util.spec_from_file_location(
    "pypi_publish", ROOT / ".release/pypi/publish.py"
)
publish = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publish)
VERSION = "9.8.7rc2"


@pytest.mark.parametrize("operation", ["prepare", "verify"])
def test_publication_requests_share_absolute_deadline(
    candidate, monkeypatch, operation
):
    _build(candidate, "runtime")
    api = _Registry(monkeypatch, candidate)
    clock = [0.0]
    calls = []
    monkeypatch.setattr(publish.time, "monotonic", lambda: clock[0])

    def slow_read(url, **kwargs):
        calls.append((url, kwargs.get("deadline")))
        clock[0] += 4
        return api.read(url)

    monkeypatch.setattr(publish.release, "_read_url", slow_read)
    with pytest.raises(ValueError, match="timed out"):
        _run(candidate, operation=operation, wait_seconds=10)
    assert len(calls) == 3
    assert [deadline for _, deadline in calls] == [10, 10, 10]
    assert clock[0] == 12
    assert not Path("upload").exists()


@pytest.mark.parametrize("operation", ["prepare", "verify"])
def test_zero_publication_budget_stops_before_network(
    candidate, monkeypatch, operation
):
    _build(candidate, "runtime")
    api = _Registry(monkeypatch, candidate)
    with pytest.raises(ValueError, match="timed out"):
        _run(candidate, operation=operation, wait_seconds=0)
    assert api.calls == []


@pytest.mark.parametrize("profile", list(publish.PROJECTS))
def test_real_uv_output_passes_strict_prepare_and_rejects_incomplete_builds(
    tmp_path, monkeypatch, profile
):
    project = publish.PROJECTS[profile]
    shutil.copytree(
        ROOT / project,
        tmp_path / project,
        ignore=shutil.ignore_patterns(".venv", "dist", "build", "*.egg-info", "tests"),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("EXPECTED_VERSION", raising=False)
    monkeypatch.delenv("EXPECTED_REVISION", raising=False)
    version = publish._version(profile)
    if profile == "inference-models":
        workflow = yaml.safe_load(
            (ROOT / ".github/workflows/publish.pypi.inference_exp.yml").read_text()
        )
        command = next(
            line.strip()
            for step in workflow["jobs"]["build"]["steps"]
            for line in step.get("run", "").splitlines()
            if "uv build" in line
        )
        directory = tmp_path / project / "dist"
    else:
        command = next(
            line.split("&&", 1)[1].strip()
            for line in (ROOT / "Makefile").read_text().splitlines()
            if line.strip().startswith(f"cd {project} && uv build ")
        )
        directory = tmp_path / "dist"

    # Use the real publisher command and backend, with installed build tools offline.
    result = subprocess.run(
        [*shlex.split(command), "--offline", "--no-build-isolation"],
        cwd=tmp_path / project,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = []

    def absent(url, **kwargs):
        calls.append(url)
        raise urllib.error.HTTPError(url, 404, "absent", {}, None)

    monkeypatch.setattr(publish.release, "_read_url", absent)
    receipt = _run(directory, profile=profile, wait_seconds=20)
    expected = publish._filenames(profile, version)
    assert {path.name for path in directory.iterdir()} == expected
    assert set(receipt["missing_files"]) == expected
    assert {path.name for path in Path("upload").iterdir()} == expected

    calls.clear()
    for filename in expected:
        path = directory / filename
        data = path.read_bytes()
        path.unlink()
        with pytest.raises(ValueError, match="Unexpected publication inventory"):
            _run(directory, profile=profile, wait_seconds=20)
        path.write_bytes(data)
    (directory / ".gitignore").write_text("*\n")
    with pytest.raises(ValueError, match="Unexpected publication inventory"):
        _run(directory, profile=profile, wait_seconds=20)
    assert calls == []


def _wheel(
    name,
    *,
    version=VERSION,
    change=None,
    timestamp=(2026, 1, 1, 0, 0, 0),
    mode=0o100644,
    compression=zipfile.ZIP_STORED,
):
    stem = f"{name.replace('-', '_')}-{version}"
    entries = {
        f"{name.replace('-', '_')}/__init__.py": b"payload\n",
        f"{stem}.dist-info/METADATA": (
            f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n"
            f"Requires-Dist: inference-sdk=={version}\nRequires-Python: >=3.10,<3.14\n\n"
        ).encode(),
        f"{stem}.dist-info/WHEEL": b"Wheel-Version: 1.0\nTag: py3-none-any\n",
        f"{stem}.dist-info/RECORD": b"record\n",
        f"{stem}.dist-info/licenses/LICENSE": b"license\n",
    }
    if change:
        key = next(key for key in entries if key.endswith(change))
        entries[key] += b"changed\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path, contents in entries.items():
            entry = zipfile.ZipInfo(path, date_time=timestamp)
            entry.external_attr = mode << 16
            entry.compress_type = compression
            archive.writestr(entry, contents)
    return buffer.getvalue()


def _sdist(
    name,
    *,
    version=VERSION,
    change=False,
    mode=0o644,
    timestamp=0,
    owner=0,
    reverse=False,
    fault=None,
):
    stem = f"{name.replace('-', '_')}-{version}"
    metadata = f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n\n".encode()
    entries = [
        (stem, None),
        (f"{stem}/PKG-INFO", metadata),
        (f"{stem}/{name.replace('-', '_')}.egg-info/PKG-INFO", metadata),
        (f"{stem}/source.py", b"changed" if change else b"payload"),
    ]
    if fault == "root":
        entries = [
            (path.replace(stem, "other"), contents) for path, contents in entries
        ]
    if fault == "traversal":
        entries.append((f"{stem}/../evil", b"bad"))
    if fault == "absolute":
        entries.append(("/evil", b"bad"))
    if fault == "backslash":
        entries.append((f"{stem}/a\\b", b"bad"))
    if fault == "duplicate":
        entries.append(entries[-1])
    if fault == "case-collision":
        entries.append((f"{stem}/SOURCE.py", b"different"))
    if fault == "file-trailing-slash":
        entries[-1] = (f"{stem}/source.py/", b"payload")
    if reverse:
        entries.reverse()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path, contents in entries:
            entry = tarfile.TarInfo(path)
            entry.mode = mode if contents is not None else 0o755
            entry.mtime = timestamp
            entry.uid = entry.gid = owner
            entry.uname = entry.gname = str(owner)
            if contents is None:
                entry.type = tarfile.DIRTYPE
            else:
                entry.size = len(contents)
            if fault in {"symlink", "hardlink", "fifo"} and path.endswith("source.py"):
                entry.type = {
                    "symlink": tarfile.SYMTYPE,
                    "hardlink": tarfile.LNKTYPE,
                    "fifo": tarfile.FIFOTYPE,
                }[fault]
                entry.linkname = "/outside"
            if fault == "size" and path.endswith("source.py"):
                entry.size = publish.release.MAX_RESPONSE_BYTES + 1
                archive.addfile(entry)
                break
            if fault == "identity" and path.endswith("/PKG-INFO"):
                contents = metadata.replace(b"Name: ", b"Name: other-")
                entry.size = len(contents)
            if fault == "missing-metadata" and path == f"{stem}/PKG-INFO":
                continue
            archive.addfile(
                entry, io.BytesIO(contents) if contents is not None else None
            )
    return buffer.getvalue()


class _Registry:
    def __init__(self, monkeypatch, directory, *, absent=False):
        self.files = {path.name: path.read_bytes() for path in directory.iterdir()}
        self.documents = {}
        self.calls = []
        for filename, data in self.files.items():
            file = publish._artifact(data, filename=filename, version=VERSION)
            document = self.documents.setdefault(
                file["name"],
                {"info": {"name": file["name"], "version": VERSION}, "urls": []},
            )
            document["urls"].append(
                {
                    "filename": filename,
                    "url": f"https://files.pythonhosted.org/{filename}",
                    "digests": {"sha256": hashlib.sha256(data).hexdigest()},
                    "yanked": False,
                }
            )
        if absent:
            self.documents.clear()
        monkeypatch.setattr(publish.release, "_read_url", self.read)

    def read(self, url, **kwargs):
        self.calls.append(url)
        if url.endswith("/json"):
            name = url.split("/")[-3]
            if name not in self.documents:
                raise urllib.error.HTTPError(url, 404, "missing version", {}, None)
            return json.dumps(self.documents[name]).encode()
        return self.files[url.rsplit("/", 1)[-1]]

    def replace(self, filename, data):
        self.files[filename] = data
        for document in self.documents.values():
            for file in document["urls"]:
                if file["filename"] == filename:
                    file["digests"]["sha256"] = hashlib.sha256(data).hexdigest()


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(publish.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        publish.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PUBLISH", "true")
    monkeypatch.setenv("EXPECTED_VERSION", VERSION)
    monkeypatch.delenv("EXPECTED_REVISION", raising=False)
    path = tmp_path / "inference/core/version.py"
    path.parent.mkdir(parents=True)
    path.write_text(f'__version__ = "{VERSION}"\n')
    directory = tmp_path / "dist"
    directory.mkdir()
    return directory


def _build(directory, profile, *, version=VERSION):
    names = publish.release.RUNTIME if profile == "runtime" else {profile}
    for name in names:
        for filename in publish._filenames(name, version):
            data = (
                _wheel(name, version=version)
                if filename.endswith(".whl")
                else _sdist(name, version=version)
            )
            (directory / filename).write_bytes(data)
        if profile != "runtime":
            path = Path(publish.PROJECTS[name]) / "pyproject.toml"
            path.parent.mkdir(parents=True)
            path.write_text(f'[project]\nname="{name}"\nversion="{version}"\n')


def _run(directory, *, profile="runtime", operation="prepare", wait_seconds=20):
    publish._run(
        operation,
        profile=profile,
        directory=directory,
        upload_directory=Path("upload"),
        receipt=Path(f"evidence/{operation}.json"),
        wait_seconds=wait_seconds,
    )
    return json.loads(Path(f"evidence/{operation}.json").read_text())


@pytest.mark.parametrize("profile", ["runtime", *publish.PROJECTS])
@pytest.mark.parametrize("state", ["absent", "present", "subset"])
def test_preflight_stages_exactly_missing_files_and_verifies_all(
    candidate, monkeypatch, profile, state
):
    _build(candidate, profile)
    api = _Registry(monkeypatch, candidate, absent=state == "absent")
    if state == "subset":
        name = sorted(api.documents)[0]
        if profile in publish.SDISTS:
            api.documents[name]["urls"] = api.documents[name]["urls"][:1]
        else:
            del api.documents[name]
    present = {
        file["filename"]
        for document in api.documents.values()
        for file in document["urls"]
    }
    receipt = _run(candidate, profile=profile)
    assert {p.name for p in Path("upload").iterdir()} == set(api.files) - present
    assert receipt["upload"] == (state != "present")
    assert {file["filename"] for file in receipt["candidate_files"]} == set(api.files)
    assert {file["filename"] for file in receipt["published_files"]} == present
    assert receipt["status"] == "prepared"
    if state != "present":
        with pytest.raises(ValueError, match="Incomplete published inventory"):
            _run(candidate, profile=profile, operation="verify")
        assert (
            json.loads(Path("evidence/verify.json").read_text())["status"] == "failed"
        )
    api = _Registry(monkeypatch, candidate)
    final = _run(candidate, profile=profile, operation="verify")
    assert final["status"] == "verified"
    assert len(final["published_files"]) == len(api.files)
    assert not final["upload"]


@pytest.mark.parametrize("present_suffix", [".whl", ".tar.gz"])
@pytest.mark.parametrize("profile", sorted(publish.SDISTS))
def test_either_standalone_file_can_be_missing(
    candidate, monkeypatch, profile, present_suffix
):
    _build(candidate, profile)
    api = _Registry(monkeypatch, candidate)
    api.documents[profile]["urls"] = [
        file
        for file in api.documents[profile]["urls"]
        if file["filename"].endswith(present_suffix)
    ]
    receipt = _run(candidate, profile=profile)
    assert len(receipt["published_files"]) == 1
    assert len(receipt["missing_files"]) == 1
    assert not receipt["missing_files"][0].endswith(present_suffix)


def test_completed_upload_rerun_clears_stale_stage_and_is_noop(candidate, monkeypatch):
    _build(candidate, "runtime")
    _Registry(monkeypatch, candidate, absent=True)
    assert _run(candidate)["upload"]
    Path("upload/unexpected.txt").write_text("stale")
    _Registry(monkeypatch, candidate)
    assert not _run(candidate)["upload"]
    assert list(Path("upload").iterdir()) == []


@pytest.mark.parametrize(
    "change", ["__init__.py", "/METADATA", "/WHEEL", "/RECORD", "/LICENSE", "mode"]
)
def test_batch_conflict_blocks_all_staging_and_retains_hashes(
    candidate, monkeypatch, change
):
    _build(candidate, "runtime")
    api = _Registry(monkeypatch, candidate)
    name = sorted(api.documents)[-1]
    for absent in sorted(api.documents)[:-1]:
        del api.documents[absent]
    filename = api.documents[name]["urls"][0]["filename"]
    api.replace(
        filename,
        _wheel(
            name,
            change=None if change == "mode" else change,
            mode=0o100755 if change == "mode" else 0o100644,
        ),
    )
    with pytest.raises(ValueError, match="contents differ"):
        _run(candidate)
    assert not Path("upload").exists()
    receipt = json.loads(Path("evidence/prepare.json").read_text())
    assert receipt["status"] == "failed"
    assert not receipt["upload"]
    assert receipt["candidate_files"] and receipt["published_files"]


@pytest.mark.parametrize("profile", ["runtime", "inference-models"])
def test_timestamp_compression_owner_and_tar_order_variations_match_contents(
    candidate, monkeypatch, profile
):
    _build(candidate, profile)
    api = _Registry(monkeypatch, candidate)
    for filename in list(api.files):
        name = publish._artifact(
            api.files[filename], filename=filename, version=VERSION
        )["name"]
        data = (
            _wheel(
                name,
                timestamp=(2026, 2, 2, 2, 2, 2),
                compression=zipfile.ZIP_DEFLATED,
            )
            if filename.endswith(".whl")
            else _sdist(name, timestamp=9999, owner=123, reverse=True)
        )
        if filename.endswith(".whl"):
            with zipfile.ZipFile(io.BytesIO(api.files[filename])) as candidate_zip:
                assert {entry.compress_type for entry in candidate_zip.infolist()} == {
                    zipfile.ZIP_STORED
                }
            with zipfile.ZipFile(io.BytesIO(data)) as published_zip:
                assert {entry.compress_type for entry in published_zip.infolist()} == {
                    zipfile.ZIP_DEFLATED
                }
        api.replace(filename, data)
    receipt = _run(candidate, profile=profile)
    assert not receipt["upload"]
    assert all(file["match"] == "contents" for file in receipt["matches"])


@pytest.mark.parametrize("change", ["payload", "mode"])
def test_sdist_content_or_mode_change_blocks_reuse(candidate, monkeypatch, change):
    _build(candidate, "inference-models")
    api = _Registry(monkeypatch, candidate)
    filename = next(file for file in api.files if file.endswith(".tar.gz"))
    api.replace(
        filename,
        _sdist(
            "inference-models",
            change=change == "payload",
            mode=0o755 if change == "mode" else 0o644,
        ),
    )
    with pytest.raises(ValueError, match="contents differ"):
        _run(candidate, profile="inference-models")
    assert not Path("upload").exists()


@pytest.mark.parametrize(
    "fault",
    [
        "sha",
        "malformed-sha",
        "yanked",
        "missing-yank",
        "unexpected",
        "empty",
        "duplicate",
        "identity",
        "version",
        "unsafe-url",
        "credentials",
        "malformed",
        "wrong-file",
    ],
)
def test_unsafe_registry_inventory_never_authorizes_upload(
    candidate, monkeypatch, fault
):
    _build(candidate, "runtime")
    api = _Registry(monkeypatch, candidate)
    document = api.documents[sorted(api.documents)[0]]
    file = document["urls"][0]
    if fault == "sha":
        file["digests"]["sha256"] = "0" * 64
    elif fault == "malformed-sha":
        file["digests"]["sha256"] = "oops"
    elif fault == "yanked":
        file["yanked"] = True
    elif fault == "missing-yank":
        del file["yanked"]
    elif fault == "unexpected":
        file["filename"] = "other.whl"
    elif fault == "empty":
        document["urls"] = []
    elif fault == "duplicate":
        document["urls"].append(dict(file))
    elif fault == "identity":
        document["info"]["name"] = "wrong"
    elif fault == "version":
        document["info"]["version"] = "1"
    elif fault == "unsafe-url":
        file["url"] = "https://evil.example/file"
    elif fault == "credentials":
        file["url"] = "https://token@files.pythonhosted.org/file"
    elif fault == "malformed":
        document["urls"] = {}
    else:
        api.replace(file["filename"], b"not a wheel")
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        _run(candidate)
    assert not Path("upload").exists()


@pytest.mark.parametrize(
    "status", [401, 403, 429, 500, 503, "timeout", "urlerror", "cdn404"]
)
def test_registry_and_cdn_failure_never_means_absence(candidate, monkeypatch, status):
    _build(candidate, "runtime")
    api = _Registry(monkeypatch, candidate)
    read = api.read

    def fail(url, **kwargs):
        if status == "cdn404" and url.endswith("/json"):
            return read(url)
        if status == "timeout":
            raise TimeoutError("unavailable")
        if status == "urlerror":
            raise urllib.error.URLError("unavailable")
        raise urllib.error.HTTPError(
            url, 404 if status == "cdn404" else status, "failure", {}, None
        )

    monkeypatch.setattr(publish.release, "_read_url", fail)
    with pytest.raises((ValueError, urllib.error.HTTPError)):
        _run(candidate)
    assert not Path("upload").exists()


@pytest.mark.parametrize("failure", ["cdn404", "transient", "version404"])
def test_bounded_visibility_retry_then_full_verification(
    candidate, monkeypatch, failure
):
    _build(candidate, "runtime")
    api = _Registry(monkeypatch, candidate)
    read = api.read
    clock = [0]
    attempts = []

    def delayed(url, **kwargs):
        target = (
            not url.endswith("/json") if failure == "cdn404" else url.endswith("/json")
        )
        if target and not attempts:
            attempts.append(url)
            raise urllib.error.HTTPError(
                url, 503 if failure == "transient" else 404, "delay", {}, None
            )
        return read(url)

    monkeypatch.setattr(publish.release, "_read_url", delayed)
    monkeypatch.setattr(publish.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        publish.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    receipt = _run(candidate, operation="verify", wait_seconds=20)
    assert receipt["status"] == "verified"
    assert clock[0] == 10


@pytest.mark.parametrize(
    "fault",
    [
        "extra",
        "sdk",
        "missing",
        "wrong-identity",
        "wrong-version",
        "conditional-sdk",
        "wrong-sdk",
    ],
)
def test_exact_local_inventory_and_runtime_sdk_contract_precede_network(
    candidate, monkeypatch, fault
):
    _build(candidate, "runtime")
    if fault in {"extra", "sdk"}:
        (
            candidate
            / (
                "extra.txt"
                if fault == "extra"
                else f"inference_sdk-{VERSION}-py3-none-any.whl"
            )
        ).write_bytes(b"bad")
    else:
        path = sorted(candidate.iterdir())[0]
        if fault == "missing":
            path.unlink()
        elif fault in {"wrong-identity", "wrong-version"}:
            path.write_bytes(
                _wheel(
                    "other" if fault == "wrong-identity" else "inference",
                    version="1" if fault == "wrong-version" else VERSION,
                )
            )
        else:
            buffer = io.BytesIO()
            with zipfile.ZipFile(path) as source, zipfile.ZipFile(
                buffer, "w"
            ) as target:
                for entry in source.infolist():
                    data = source.read(entry)
                    if entry.filename.endswith("/METADATA"):
                        data = data.replace(
                            f"inference-sdk=={VERSION}".encode(),
                            (
                                b"inference-sdk==1"
                                if fault == "wrong-sdk"
                                else f'inference-sdk=={VERSION}; python_version>="3.10"'.encode()
                            ),
                        )
                    target.writestr(entry, data)
            path.write_bytes(buffer.getvalue())
    monkeypatch.setattr(
        publish.release,
        "_read_url",
        lambda url: pytest.fail("Invalid local inventory must stop before network"),
    )
    with pytest.raises(ValueError):
        _run(candidate)


@pytest.mark.parametrize("version", ["9.8.7rc3", "9.8.7.post2"])
@pytest.mark.parametrize("profile", ["runtime", "inference-models"])
def test_rc_and_post_versions_are_preserved_and_wrong_expectations_rejected(
    candidate, monkeypatch, version, profile
):
    _build(candidate, profile, version=version)
    Path("inference/core/version.py").write_text(f'__version__="{version}"\n')
    monkeypatch.setenv("EXPECTED_VERSION", f"v{version}")
    monkeypatch.setattr(
        publish.release,
        "_read_url",
        lambda url, **kwargs: (_ for _ in ()).throw(
            urllib.error.HTTPError(url, 404, "missing", {}, None)
        ),
    )
    receipt = _run(candidate, profile=profile)
    assert all(file["version"] == version for file in receipt["candidate_files"])
    monkeypatch.setenv("EXPECTED_VERSION", "9.8.7")
    monkeypatch.setattr(
        publish.release,
        "_read_url",
        lambda url: pytest.fail("Wrong version must not query registry"),
    )
    with pytest.raises(ValueError, match="version"):
        _run(candidate, profile=profile)


@pytest.mark.parametrize(
    "fault",
    [
        "root",
        "traversal",
        "absolute",
        "backslash",
        "duplicate",
        "case-collision",
        "file-trailing-slash",
        "symlink",
        "hardlink",
        "fifo",
        "identity",
        "missing-metadata",
        "size",
    ],
)
def test_malformed_sdist_is_rejected(fault):
    with pytest.raises((ValueError, tarfile.TarError, EOFError)):
        publish._artifact(
            _sdist("inference-models", fault=fault),
            filename=f"inference_models-{VERSION}.tar.gz",
            version=VERSION,
        )


@pytest.mark.parametrize(
    "fault", ["duplicate", "traversal", "symlink", "oversized", "malformed"]
)
def test_malformed_wheel_is_rejected(monkeypatch, fault):
    filename = f"inference-{VERSION}-py3-none-any.whl"
    buffer = io.BytesIO(_wheel("inference"))
    if fault == "malformed":
        data = b"not a zip"
    else:
        with zipfile.ZipFile(buffer, "a") as archive:
            entry = zipfile.ZipInfo(
                "../evil"
                if fault == "traversal"
                else (
                    "inference/__init__.py"
                    if fault == "duplicate"
                    else "inference/extra.py"
                )
            )
            if fault == "symlink":
                entry.external_attr = (stat.S_IFLNK | 0o777) << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, "a" * 100000 if fault == "oversized" else "evil")
        data = buffer.getvalue()
        if fault == "oversized":
            assert len(data) < 5000
            monkeypatch.setattr(publish.release, "MAX_RESPONSE_BYTES", 5000)
    messages = {
        "duplicate": "Invalid wheel entries",
        "traversal": "Unsafe archive entry",
        "symlink": "Unsafe wheel entry type",
        "oversized": "Uncompressed wheel exceeds",
        "malformed": "not a zip",
    }
    with pytest.raises((ValueError, zipfile.BadZipFile), match=messages[fault]):
        publish._artifact(data, filename=filename, version=VERSION)


@pytest.mark.parametrize(
    "kind", ["same", "parent", "child", "receipt-source", "receipt-stage", "symlink"]
)
def test_unsafe_staging_and_receipt_paths_are_rejected(candidate, monkeypatch, kind):
    _build(candidate, "runtime")
    stage, receipt = Path("upload"), Path("evidence/prepare.json")
    if kind == "same":
        stage = candidate
    elif kind == "parent":
        stage = candidate.parent
    elif kind == "child":
        stage = candidate / "upload"
    elif kind == "receipt-source":
        receipt = candidate / "receipt.json"
    elif kind == "receipt-stage":
        receipt = stage / "receipt.json"
    else:
        stage.symlink_to(candidate, target_is_directory=True)
    monkeypatch.setattr(
        publish.release,
        "_read_url",
        lambda url: pytest.fail("Unsafe paths must stop before network"),
    )
    with pytest.raises(ValueError):
        publish._run(
            "prepare",
            profile="runtime",
            directory=candidate,
            upload_directory=stage,
            receipt=receipt,
            wait_seconds=0,
        )


def test_cli_emits_reviewable_receipt_and_provenance(candidate, monkeypatch):
    _build(candidate, "runtime")
    _Registry(monkeypatch, candidate)
    monkeypatch.setenv("SOURCE_REVISION", "candidate-sha")
    monkeypatch.setenv("RELEASE_IMAGE_ID", "sha256:image")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    result = CliRunner().invoke(
        publish._main,
        [
            "prepare",
            "--profile",
            "runtime",
            "--directory",
            str(candidate),
            "--upload-directory",
            "upload",
            "--receipt",
            "evidence/prepare.json",
            "--wait-seconds",
            "20",
        ],
    )
    assert result.exit_code == 0, result.output
    receipt = json.loads(Path("evidence/prepare.json").read_text())
    assert receipt["source_revision"] == "candidate-sha"
    assert receipt["image_id"] == "sha256:image"
    assert receipt["verification_run_id"] == "42"


def test_standalone_workflows_gate_prepare_upload_missing_then_verify():
    for profile, filename in [
        ("inference-models", "inference_exp"),
        ("roboflow-workflows", "roboflow_workflows"),
        ("streamvision", "streamvision"),
    ]:
        config = yaml.load(
            (ROOT / f".github/workflows/publish.pypi.{filename}.yml").read_text(),
            Loader=yaml.BaseLoader,
        )
        assert config["permissions"] == {"contents": "read", "id-token": "write"}
        job = config["jobs"]["build"]
        assert job["concurrency"]["cancel-in-progress"] == "false"
        assert f"pypi-{profile}" in job["concurrency"]["group"]
        assert "github.run_id" in job["concurrency"]["group"]
        assert "environment" not in job and "uses" not in job
        steps = job["steps"]
        gate = next(
            i
            for i, step in enumerate(steps)
            if "dependencies.sh gate" in step.get("run", "")
        )
        prepare = next(
            i
            for i, step in enumerate(steps)
            if "publish.sh prepare" in step.get("run", "")
        )
        upload = next(
            i
            for i, step in enumerate(steps)
            if step.get("uses", "").startswith("pypa/")
        )
        verify = next(
            i
            for i, step in enumerate(steps)
            if "publish.sh verify" in step.get("run", "")
        )
        assert gate < prepare < upload < verify
        for index in (gate, prepare, upload, verify):
            assert "github.event.inputs.publish == 'true'" in steps[index]["if"]
        assert "steps.prepare.outputs.upload == 'true'" in steps[upload]["if"]
        assert "upload" not in steps[verify]["if"]
        assert steps[upload]["with"] == {"packages-dir": "release-upload/"}
        for index in (prepare, verify):
            assert f"--profile {profile}" in steps[index]["run"]
            assert "--receipt release-evidence/" in steps[index]["run"]
        assert any(
            "always()" in step.get("if", "")
            and step.get("with", {}).get("path") == "release-evidence/"
            for step in steps
        )


def test_runtime_modes_and_dependency_gates_precede_recovery():
    config = yaml.load(
        (ROOT / ".github/workflows/publish.pypi.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    job = config["jobs"]["publish_runtime"]
    assert "inputs.packages != 'dependencies'" in job["if"]
    assert "inputs.publish == true" in job["if"]
    assert job["concurrency"]["cancel-in-progress"] == "false"
    steps = job["steps"]
    prepare = next(
        i for i, step in enumerate(steps) if "publish.sh prepare" in step.get("run", "")
    )
    gates = [
        i
        for i, step in enumerate(steps)
        if "gate-sdk" in step.get("run", "")
        or "dependencies.sh verify" in step.get("run", "")
    ]
    assert len(gates) == 2 and max(gates) < prepare
    assert any(
        "always()" in step.get("if", "")
        and "runtime-publication" in step.get("with", {}).get("name", "")
        for step in steps
    )
    coordinator = config["jobs"]["prepare_dependencies"]["steps"]
    coordinate = next(
        step
        for step in coordinator
        if "dependencies.sh coordinate" in step.get("run", "")
    )
    assert (
        coordinate["if"]
        == "${{ github.event_name == 'release' || inputs.publish == true }}"
    )


def test_contract_workflow_covers_new_helper_and_tests():
    config = yaml.load(
        (ROOT / ".github/workflows/test.pypi.release.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    for trigger in ("pull_request", "push"):
        assert ".github/scripts/test_pypi_publish.py" in config["on"][trigger]["paths"]
    run = next(
        step["run"]
        for step in config["jobs"]["contracts"]["steps"]
        if "python -m pytest" in step.get("run", "")
    )
    assert "docker run" in run
    assert run.count(".release/pypi/publish.py") == 2
    assert run.count(".github/scripts/test_pypi_publish.py") == 3


@pytest.mark.parametrize("document", [None, [], "invalid"])
def test_successful_http_with_invalid_json_document_is_not_absence(
    candidate, monkeypatch, document
):
    _build(candidate, "runtime")
    monkeypatch.setattr(
        publish.release,
        "_read_url",
        lambda url, **kwargs: json.dumps(document).encode(),
    )
    with pytest.raises(ValueError, match="Invalid registry document"):
        _run(candidate)
    assert not Path("upload").exists()


def test_oversized_compressed_tar_extension_is_rejected_before_parsing(monkeypatch):
    stem = f"inference_models-{VERSION}"
    buffer = io.BytesIO()
    metadata = f"Name: inference-models\nVersion: {VERSION}\n\n".encode()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        entry = tarfile.TarInfo(f"{stem}/PKG-INFO")
        entry.size = len(metadata)
        entry.pax_headers = {"comment": "a" * 100000}
        archive.addfile(entry, io.BytesIO(metadata))
    assert len(buffer.getvalue()) < 5000
    monkeypatch.setattr(publish.release, "MAX_RESPONSE_BYTES", 5000)
    with pytest.raises(ValueError, match="Uncompressed sdist exceeds"):
        publish._artifact(buffer.getvalue(), filename=f"{stem}.tar.gz", version=VERSION)


def test_standalone_manual_publication_preserves_optional_expected_version(
    candidate, monkeypatch
):
    _build(candidate, "inference-models")
    _Registry(monkeypatch, candidate)
    monkeypatch.delenv("EXPECTED_VERSION")
    assert not _run(candidate, profile="inference-models")["upload"]


def test_runtime_publication_still_requires_expected_version(candidate, monkeypatch):
    _build(candidate, "runtime")
    monkeypatch.delenv("EXPECTED_VERSION")
    monkeypatch.setattr(
        publish.release,
        "_read_url",
        lambda url: pytest.fail("Missing expected_version must precede network"),
    )
    with pytest.raises(ValueError, match="requires expected_version"):
        _run(candidate)
