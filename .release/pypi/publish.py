"""Verify immutable release contents and stage only missing PyPI files."""

import gzip
import hashlib
import io
import json
import os
import re
import shutil
import stat
import tarfile
import time
import tomllib
import urllib.error
import urllib.parse
import zipfile
from email.parser import BytesParser
from http.client import HTTPException
from pathlib import Path, PurePosixPath

import click
import release
from packaging.utils import canonicalize_name, parse_sdist_filename
from packaging.version import Version

PROJECTS = {
    "inference-models": "inference_models",
    "roboflow-workflows": "workflows",
    "streamvision": "stream_vision",
}
SDISTS = {"inference-models", "roboflow-workflows"}
MAX_ENTRIES = 10000


def _filenames(name, version):
    stem = f"{name.replace('-', '_')}-{version}"
    filenames = {f"{stem}-py3-none-any.whl"}
    if name in SDISTS:
        filenames.add(f"{stem}.tar.gz")
    return filenames


def _safe_name(name):
    if (
        not name
        or name.startswith("/")
        or "\\" in name
        or "\x00" in name
        or any(part in {"", ".", ".."} for part in name.rstrip("/").split("/"))
    ):
        raise ValueError(f"Unsafe archive entry: {name!r}")


def _metadata_fields(metadata):
    fields = {
        "requires_dist": metadata.get_all("Requires-Dist", []),
        "provides_extra": metadata.get_all("Provides-Extra", []),
        "requires_python": metadata.get("Requires-Python", ""),
    }
    return fields


def _artifact(data, *, filename, version):
    if len(data) > release.MAX_RESPONSE_BYTES:
        raise ValueError("Archive exceeds the size limit")

    if filename.endswith(".whl"):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if (
                len(entries) > MAX_ENTRIES
                or sum(e.file_size for e in entries) > release.MAX_RESPONSE_BYTES
            ):
                raise ValueError("Uncompressed wheel exceeds the size limit")
            for entry in entries:
                _safe_name(entry.filename)
                kind = stat.S_IFMT(entry.external_attr >> 16)
                if kind not in {0, stat.S_IFREG, stat.S_IFDIR}:
                    raise ValueError("Unsafe wheel entry type")
            metadata_paths = [
                p for p in archive.namelist() if p.endswith(".dist-info/METADATA")
            ]
            if len(metadata_paths) != 1:
                raise ValueError("Expected one wheel METADATA file")
            metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
            if any(
                len(metadata.get_all(field, [])) != 1 for field in ("Name", "Version")
            ):
                raise ValueError("Expected one metadata name and version")
        result = release._wheel(data, filename=filename, version=version)
    else:
        name, actual_version = parse_sdist_filename(filename)
        if not filename.endswith(".tar.gz") or actual_version != Version(version):
            raise ValueError(f"Wrong sdist version or format: {filename}")
        root = filename[:-7]
        entries = []
        seen = set()
        total = 0
        metadata = None
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as compressed:
            expanded = compressed.read(release.MAX_RESPONSE_BYTES + 1)
        if len(expanded) > release.MAX_RESPONSE_BYTES:
            raise ValueError("Uncompressed sdist exceeds the size limit")
        with tarfile.open(fileobj=io.BytesIO(expanded), mode="r:") as archive:
            for entry in archive:
                entry_name = entry.name.rstrip("/") if entry.isdir() else entry.name
                _safe_name(entry_name)
                if entry.isfile() and entry_name.endswith("/"):
                    raise ValueError("Regular sdist file name ends with a slash")
                if entry_name.casefold() in seen or len(seen) >= MAX_ENTRIES:
                    raise ValueError("Duplicate or excessive sdist entries")
                seen.add(entry_name.casefold())
                if PurePosixPath(entry_name).parts[0] != root:
                    raise ValueError("Unexpected sdist root")
                if not (entry.isfile() or entry.isdir()):
                    raise ValueError("Unsafe sdist entry type")
                if entry.isdir() and entry.size:
                    raise ValueError("Unexpected directory data in sdist")
                total += entry.size
                if total > release.MAX_RESPONSE_BYTES or entry.size < 0:
                    raise ValueError("Uncompressed sdist exceeds the size limit")
                contents = archive.extractfile(entry).read() if entry.isfile() else b""
                if entry_name == f"{root}/PKG-INFO" and entry.isfile():
                    metadata = BytesParser().parsebytes(contents)
                entries.append(
                    (
                        entry_name,
                        "file" if entry.isfile() else "directory",
                        entry.mode,
                        hashlib.sha256(contents).hexdigest(),
                    )
                )
        if metadata is None:
            raise ValueError("Missing root PKG-INFO")
        if any(len(metadata.get_all(field, [])) != 1 for field in ("Name", "Version")):
            raise ValueError("Expected one metadata name and version")
        if (
            canonicalize_name(metadata["Name"]) != name
            or Version(metadata["Version"]) != actual_version
        ):
            raise ValueError("Filename and PKG-INFO disagree")
        result = {
            "name": str(name),
            "version": str(actual_version),
            "filename": filename,
            "sha256": hashlib.sha256(data).hexdigest(),
            "content_sha256": hashlib.sha256(
                json.dumps(sorted(entries)).encode()
            ).hexdigest(),
        }
    result.update(_metadata_fields(metadata))
    return result


def _version(profile):
    if profile == "runtime":
        return release._expected_version()
    project = tomllib.loads((Path(PROJECTS[profile]) / "pyproject.toml").read_text())[
        "project"
    ]
    if canonicalize_name(project["name"]) != profile:
        raise ValueError("Wrong source project identity")
    version = str(Version(project["version"]))
    expected = os.environ.get("EXPECTED_VERSION", "")
    if expected and Version(expected) != Version(version):
        raise ValueError("Package version differs from expected_version")
    revision = os.environ.get("EXPECTED_REVISION", "")
    if revision and revision != os.environ.get("SOURCE_REVISION", ""):
        raise ValueError("Candidate revision differs from expected_revision")
    return version


def _inventory(directory, *, profile, version):
    names = release.RUNTIME if profile == "runtime" else {profile}
    expected = {filename for name in names for filename in _filenames(name, version)}
    paths = sorted(directory.iterdir())
    if {p.name for p in paths} != expected or any(
        p.is_symlink()
        or not p.is_file()
        or p.stat().st_size > release.MAX_RESPONSE_BYTES
        for p in paths
    ):
        raise ValueError(f"Unexpected publication inventory in {directory}")
    files = [_artifact(p.read_bytes(), filename=p.name, version=version) for p in paths]
    if profile == "runtime":
        release._inventory(directory, names=release.RUNTIME, version=version)
    return files


def _retry(operation, *, deadline, retry_not_found=True):
    while True:
        if time.monotonic() >= deadline:
            raise ValueError("Publication verification timed out")
        try:
            result = operation()
            if time.monotonic() >= deadline:
                raise ValueError("Publication verification timed out")
            return result
        except urllib.error.HTTPError as error:
            if (
                not (error.code == 404 and retry_not_found)
                and error.code != 429
                and error.code < 500
            ):
                raise
            reason = f"Registry HTTP {error.code}"
        except (
            urllib.error.URLError,
            TimeoutError,
            HTTPException,
            ConnectionError,
        ) as error:
            reason = f"Registry unavailable: {error}"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError(f"{reason}; publication verification timed out")
        print(f"{reason}; retrying publication verification", flush=True)
        time.sleep(min(10, remaining))


def _remote_inventory(name, *, version, wait_seconds=600, deadline=None):
    deadline = time.monotonic() + wait_seconds if deadline is None else deadline

    def read():
        try:
            document = json.loads(
                release._read_url(
                    f"https://pypi.org/pypi/{name}/{version}/json", deadline=deadline
                )
            )
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise
        if not isinstance(document, dict):
            raise ValueError("Invalid registry document")
        return document

    document = _retry(read, deadline=deadline)
    if document is None:
        return None
    if canonicalize_name(document["info"]["name"]) != name or Version(
        document["info"]["version"]
    ) != Version(version):
        raise ValueError(f"Registry identity mismatch for {name}=={version}")
    remote = document["urls"]
    expected = _filenames(name, str(Version(document["info"]["version"])))
    if (
        not isinstance(remote, list)
        or not remote
        or any(not isinstance(file, dict) for file in remote)
    ):
        raise ValueError(f"Invalid registry inventory for {name}=={version}")
    filenames = [file["filename"] for file in remote]
    if len(filenames) != len(set(filenames)) or not set(filenames) <= expected:
        raise ValueError(f"Unexpected registry inventory for {name}=={version}")
    for file in remote:
        if file.get("yanked") is not False:
            raise ValueError(
                f"Required file is yanked or missing yank status: {file['filename']}"
            )
        digest = file["digests"]["sha256"]
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("Invalid registry SHA-256")
        url = urllib.parse.urlparse(file["url"])
        if (
            url.scheme != "https"
            or url.hostname != "files.pythonhosted.org"
            or url.username
            or url.password
            or url.port not in {None, 443}
            or url.fragment
        ):
            raise ValueError("Registry returned an unsafe download URL")
    files = []
    for remote_file in remote:
        data = _retry(
            lambda: release._read_url(remote_file["url"], deadline=deadline),
            deadline=deadline,
        )
        if hashlib.sha256(data).hexdigest() != remote_file["digests"]["sha256"]:
            raise ValueError(f"Registry SHA-256 mismatch for {name}=={version}")
        file = _artifact(data, filename=remote_file["filename"], version=version)
        if file["name"] != name:
            raise ValueError("Downloaded artifact identity mismatch")
        files.append(file)
    if time.monotonic() >= deadline:
        raise ValueError("Publication verification timed out")
    files.sort(key=lambda file: file["filename"])
    return files


def _matches(expected, published):
    candidates = {file["filename"]: file for file in expected}
    decisions = [
        {
            "filename": file["filename"],
            "match": (
                "different"
                if file["filename"] not in candidates
                or file["content_sha256"]
                != candidates[file["filename"]]["content_sha256"]
                else (
                    "bytes"
                    if file["sha256"] == candidates[file["filename"]]["sha256"]
                    else "contents"
                )
            ),
        }
        for file in published
    ]
    return decisions


def _reconcile(expected, *, allow_missing, wait_seconds, receipt=None):
    deadline = time.monotonic() + wait_seconds
    missing = sorted(file["filename"] for file in expected)
    while True:
        if time.monotonic() >= deadline:
            raise ValueError(
                f"Incomplete published inventory: {missing}; verification timed out"
            )
        published = []
        for name in sorted({file["name"] for file in expected}):
            version = next(file["version"] for file in expected if file["name"] == name)
            files = _remote_inventory(name, version=version, deadline=deadline)
            published.extend(files or [])
        candidates = {file["filename"]: file for file in expected}
        if receipt is not None:
            record = json.loads(receipt.read_text())
            record["published_files"] = published
            record["matches"] = _matches(expected, published)
            receipt.write_text(json.dumps(record, indent=2) + "\n")
        for actual in published:
            if actual["filename"] not in candidates:
                raise ValueError("Published filename differs from this candidate")
            if (
                actual["content_sha256"]
                != candidates[actual["filename"]]["content_sha256"]
            ):
                raise ValueError(
                    f"Published contents differ from this candidate: {actual['filename']}; use a new version"
                )
        missing = sorted(candidates.keys() - {file["filename"] for file in published})
        if allow_missing or not missing:
            return published, missing
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError(
                f"Incomplete published inventory: {missing}; verification timed out"
            )
        time.sleep(min(10, remaining))


def _paths(directory, *, upload_directory, receipt):
    for path in (directory, upload_directory, receipt):
        if any(parent.is_symlink() for parent in (path, *path.parents)):
            raise ValueError("Publication paths must not contain symlinks")
    source, stage, record = (
        directory.resolve(),
        upload_directory.resolve(),
        receipt.resolve(),
    )
    if (
        source == stage
        or source in stage.parents
        or stage in source.parents
        or record == source
        or source in record.parents
        or record == stage
        or stage in record.parents
    ):
        raise ValueError("Source, staging directory and receipt must be separate")
    if (
        stage == Path(stage.anchor)
        or stage == Path.cwd().resolve()
        or stage in Path.cwd().resolve().parents
        or Path.cwd().resolve() not in stage.parents
        or ".git" in stage.relative_to(Path.cwd().resolve()).parts
    ):
        raise ValueError("Unsafe upload directory")


def _run(operation, *, profile, directory, upload_directory, receipt, wait_seconds):
    _paths(directory, upload_directory=upload_directory, receipt=receipt)
    try:
        release._write_record(
            receipt,
            status="checking",
            operation=operation,
            profile=profile,
            upload=False,
        )
        version = _version(profile)
        expected = _inventory(directory, profile=profile, version=version)
        release._write_record(
            receipt,
            status="checking",
            operation=operation,
            profile=profile,
            candidate_files=expected,
            upload=False,
        )
        published, missing = _reconcile(
            expected,
            allow_missing=operation == "prepare",
            wait_seconds=wait_seconds,
            receipt=receipt,
        )
        if operation == "prepare":
            if upload_directory.exists():
                if not upload_directory.is_dir():
                    raise ValueError("Upload directory is not a directory")
                shutil.rmtree(upload_directory)
            upload_directory.mkdir(parents=True)
            for filename in missing:
                shutil.copyfile(directory / filename, upload_directory / filename)
        release._write_record(
            receipt,
            status="prepared" if operation == "prepare" else "verified",
            operation=operation,
            profile=profile,
            candidate_files=expected,
            published_files=published,
            matches=_matches(expected, published),
            missing_files=missing,
            upload=bool(missing) and operation == "prepare",
        )
    except Exception as error:
        fields = json.loads(receipt.read_text()) if receipt.is_file() else {}
        for field in (
            "schema",
            "source_revision",
            "image_id",
            "repository",
            "verification_run_id",
            "run_attempt",
        ):
            fields.pop(field, None)
        release._write_record(
            receipt,
            **{**fields, "status": "failed", "upload": False, "error": str(error)},
        )
        raise


@click.command()
@click.argument("operation", type=click.Choice(["prepare", "verify"]))
@click.option(
    "--profile",
    type=click.Choice(["runtime", *PROJECTS]),
    required=True,
)
@click.option(
    "--directory",
    type=click.Path(path_type=Path),
    required=True,
)
@click.option(
    "--upload-directory",
    type=click.Path(path_type=Path),
    required=True,
)
@click.option(
    "--receipt",
    type=click.Path(path_type=Path),
    required=True,
)
@click.option(
    "--wait-seconds",
    type=click.FloatRange(min=0, max=600),
    default=600,
)
def _main(operation, *, profile, directory, upload_directory, receipt, wait_seconds):
    try:
        _run(
            operation,
            profile=profile,
            directory=directory,
            upload_directory=upload_directory,
            receipt=receipt,
            wait_seconds=wait_seconds,
        )
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        zipfile.BadZipFile,
        tarfile.TarError,
        EOFError,
    ) as error:
        raise click.ClickException(str(error)) from error


if __name__ == "__main__":
    _main()
