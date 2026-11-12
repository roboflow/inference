"""Build explicit PyPI inventories and verify the exact SDK dependency."""

import ast
import hashlib
import http.client
import io
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from email.parser import BytesParser
from pathlib import Path

import click
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

SDK = {"inference-sdk"}
RUNTIME = {
    "inference",
    "inference-core",
    "inference-cpu",
    "inference-gpu",
    "inference-cli",
}
MAX_RESPONSE_BYTES = 64 * 1024 * 1024


def _source_version():
    tree = ast.parse(Path("inference/core/version.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            version = Version(ast.literal_eval(node.value))
            return str(version)

    raise ValueError("Missing source version")


def _expected_version():
    version = _source_version()
    expected = os.environ.get("EXPECTED_VERSION", "")
    if os.environ.get("PUBLISH") == "true" and not expected:
        raise ValueError("Publication requires expected_version (or a release tag)")

    if expected and Version(expected) != Version(version):
        raise ValueError(f"Expected version {expected} does not match source {version}")

    return version


def _wheel(data, *, filename, version):
    name, wheel_version, _, _ = parse_wheel_filename(filename)
    if wheel_version != Version(version):
        raise ValueError(f"Wrong wheel version: {filename}; expected {version}")

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if (
            name == "inference-sdk"
            and sum(entry.file_size for entry in archive.infolist())
            > MAX_RESPONSE_BYTES
        ):
            raise ValueError("Uncompressed SDK exceeds the size limit")

        if len(names) != len(set(names)) or any(
            entry.startswith("/") or ".." in entry.split("/") for entry in names
        ):
            raise ValueError(f"Invalid wheel entries: {filename}")

        metadata_paths = [p for p in names if p.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise ValueError(f"Expected one METADATA file: {filename}")

        metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
        if (
            canonicalize_name(metadata["Name"]) != name
            or Version(metadata["Version"]) != wheel_version
        ):
            raise ValueError(f"Filename and METADATA disagree: {filename}")

        if name == "inference-sdk":
            dist_info = metadata_paths[0].split("/")[0] + "/"
            if "inference_sdk/__init__.py" not in names or any(
                not entry.startswith(("inference_sdk/", dist_info)) for entry in names
            ):
                raise ValueError(
                    "SDK wheel must contain only inference_sdk and metadata"
                )

        # Publisher reuse compares all entry contents and permissions, excluding ZIP encoding.
        entries = [
            (
                entry.filename,
                entry.external_attr,
                hashlib.sha256(archive.read(entry)).hexdigest(),
            )
            for entry in sorted(archive.infolist(), key=lambda item: item.filename)
        ]
        content_sha256 = hashlib.sha256(json.dumps(entries).encode()).hexdigest()

    result = {
        "name": str(name),
        "version": str(wheel_version),
        "filename": filename,
        "sha256": hashlib.sha256(data).hexdigest(),
        "content_sha256": content_sha256,
        "requires_dist": metadata.get_all("Requires-Dist", []),
    }
    return result


def _inventory(directory, *, names, version):
    paths = sorted(directory.iterdir())
    if len(paths) != len(names) or any(path.suffix != ".whl" for path in paths):
        raise ValueError(f"Unexpected publication inventory in {directory}")

    files = [
        _wheel(path.read_bytes(), filename=path.name, version=version) for path in paths
    ]
    if {file["name"] for file in files} != names:
        raise ValueError(f"Wrong packages in {directory}: expected {sorted(names)}")

    for file in files:
        if file["name"] in RUNTIME:
            sdk_requirements = [
                Requirement(raw)
                for raw in file["requires_dist"]
                if canonicalize_name(Requirement(raw).name) == "inference-sdk"
            ]
            if len(sdk_requirements) != 1 or (
                str(sdk_requirements[0].specifier) != f"=={version}"
                or sdk_requirements[0].marker is not None
                or sdk_requirements[0].url is not None
            ):
                raise ValueError(f"Expected unconditional inference-sdk=={version}")

    return files


def _write_record(path, **fields):
    record = {
        "schema": 1,
        "source_revision": os.environ.get("SOURCE_REVISION", ""),
        "image_id": os.environ.get("RELEASE_IMAGE_ID", ""),
        "repository": os.environ.get("GITHUB_REPOSITORY", ""),
        "verification_run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
        **fields,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")


def _read_url(url):
    allowed_hosts = {"pypi.org", "files.pythonhosted.org"}
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
        raise ValueError("Registry returned an unexpected download URL")

    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        final_url = urllib.parse.urlparse(response.geturl())
        if final_url.scheme != "https" or final_url.hostname not in allowed_hosts:
            raise ValueError("Unexpected registry redirect")

        data = response.read(MAX_RESPONSE_BYTES + 1)
        if len(data) > MAX_RESPONSE_BYTES:
            raise ValueError("Registry response exceeds the SDK size limit")

    return data


def _published_sdk(expected, *, compare_candidate=True):
    url = f"https://pypi.org/pypi/inference-sdk/{expected['version']}/json"
    try:
        release = json.loads(_read_url(url))
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None

        raise

    if Version(release["info"]["version"]) != Version(expected["version"]):
        raise ValueError("Registry returned the wrong SDK version")

    files = release["urls"]
    if not files:
        raise ValueError("Existing SDK release has no files; inspect it manually")

    if len(files) != 1 or files[0]["filename"] != expected["filename"]:
        raise ValueError("Existing SDK release has an unexpected file inventory")

    file = files[0]
    if file.get("yanked"):
        raise ValueError("Required SDK file is yanked")

    data = _read_url(file["url"])
    if hashlib.sha256(data).hexdigest() != file["digests"]["sha256"]:
        raise ValueError("Downloaded SDK does not match the registry SHA-256")

    actual = _wheel(data, filename=file["filename"], version=expected["version"])
    if compare_candidate and actual["content_sha256"] != expected["content_sha256"]:
        raise ValueError(
            "Published SDK contents differ from this candidate; use a new version"
        )

    return actual


def _reconcile_sdk(expected, *, allow_missing, wait_seconds, compare_candidate=True):
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            actual = _published_sdk(expected, compare_candidate=compare_candidate)
            if actual is not None or allow_missing:
                return actual

            reason = "Required SDK is not published yet"
        except urllib.error.HTTPError as error:
            # Only the release endpoint's 404 means absent; file CDN failures retry.
            if error.code not in {404, 429} and error.code < 500:
                raise

            reason = f"Registry HTTP {error.code}"
        except (
            urllib.error.URLError,
            TimeoutError,
            http.client.HTTPException,
            ConnectionResetError,
        ) as error:
            reason = f"Registry unavailable: {error}"

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError(f"{reason}; timed out without publishing consumers")

        print(f"{reason}; waiting for verified SDK availability", flush=True)
        time.sleep(min(10, remaining))


def _run(operation, *, wait_seconds):
    version = _expected_version()
    sdk_dir = Path("dist/sdk")
    if operation in {"build-sdk", "build-runtime"}:
        if operation == "build-runtime":
            subprocess.run(["make", "create_pypi_runtime_wheels"], check=True)
            files = _inventory(Path("dist/pypi"), names=RUNTIME, version=version)
            _write_record(Path("dist/runtime-manifest.json"), files=files)

        # The runtime SDK reference stays outside its upload directory.
        subprocess.run(["make", "create_pypi_sdk_wheel"], check=True)
        files = _inventory(sdk_dir, names=SDK, version=version)
        _write_record(Path("dist/sdk-manifest.json"), files=files)
        Path("dist/version.txt").write_text(version + "\n")
        return

    expected = _inventory(sdk_dir, names=SDK, version=version)[0]
    if operation == "gate-sdk":
        _inventory(Path("dist/pypi"), names=RUNTIME, version=version)

    actual = _reconcile_sdk(
        expected,
        allow_missing=operation == "prepare-sdk",
        wait_seconds=wait_seconds,
        compare_candidate=operation != "gate-sdk",
    )
    if operation == "prepare-sdk":
        upload_dir = Path("dist/sdk-upload")
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
        upload_dir.mkdir(parents=True)
        if actual is None:
            shutil.copyfile(
                sdk_dir / expected["filename"], upload_dir / expected["filename"]
            )

        _write_record(Path("dist/sdk-decision.json"), upload=actual is None)

    if actual is not None:
        _write_record(
            Path("dist/sdk-receipt.json"),
            status="verified",
            verification_policy=(
                "registry-integrity" if operation == "gate-sdk" else "candidate-content"
            ),
            candidate_file=expected,
            published_file=actual,
        )
        print(f"Verified inference-sdk=={version}: {actual['sha256']}")


@click.command()
@click.argument(
    "operation",
    type=click.Choice(
        ["build-sdk", "build-runtime", "prepare-sdk", "verify-sdk", "gate-sdk"]
    ),
)
@click.option(
    "--wait-seconds",
    type=click.FloatRange(min=0, max=600),
    default=600,
    show_default=True,
)
def _main(operation, *, wait_seconds):
    try:
        _run(operation, wait_seconds=wait_seconds)
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as error:
        raise click.ClickException(str(error)) from error


if __name__ == "__main__":
    _main()
