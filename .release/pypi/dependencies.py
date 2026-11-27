"""Resolve and verify first-party dependencies without installing runtime packages."""

import gzip
import hashlib
import io
import json
import os
import re
import tarfile
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from email.parser import BytesParser
from http.client import HTTPException
from pathlib import Path

import click
import publish
import release
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version
from publish import _retry

PROJECTS = {
    "inference-models": ("inference_models", "publish.pypi.inference_exp.yml"),
    "roboflow-workflows": ("workflows", "publish.pypi.roboflow_workflows.yml"),
    "streamvision": ("stream_vision", "publish.pypi.streamvision.yml"),
}
OWNED = {*PROJECTS, "inference-sdk"}


def _requirements(files):
    requirements = []
    for file in files:
        for raw in file["requires_dist"]:
            requirement = Requirement(raw)
            if canonicalize_name(requirement.name) not in OWNED:
                continue

            if requirement.url:
                raise ValueError(f"Direct URL first-party dependency: {raw}")

            # All extras/platform declarations are checked conservatively. This
            # prevents a release that only works with the coordinator's extras/OS.
            requirements.append(requirement)

    return requirements


def _edges(file):
    edges = sorted(
        {
            (
                canonicalize_name(req.name),
                str(req.specifier),
                tuple(sorted({canonicalize_name(extra) for extra in req.extras})),
            )
            for req in _requirements([file])
        }
    )
    return edges


def _wheel(data, *, filename):
    _, version, _, tags = parse_wheel_filename(filename)
    if not any(str(tag) == "py3-none-any" for tag in tags):
        raise ValueError(f"Expected a universal Python wheel: {filename}")

    file = release._wheel(data, filename=filename, version=str(version))
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        path = next(p for p in archive.namelist() if p.endswith(".dist-info/METADATA"))
        metadata = BytesParser().parsebytes(archive.read(path))

    file["provides_extra"] = metadata.get_all("Provides-Extra", [])
    file["requires_python"] = metadata.get("Requires-Python", "")
    return file


def _local_wheels(directory):
    files = [
        _wheel(p.read_bytes(), filename=p.name) for p in sorted(directory.glob("*.whl"))
    ]
    if not files:
        raise ValueError(f"No consumer wheels in {directory}")

    return files


def _catalog():
    catalog = {}
    for name, (directory, _) in PROJECTS.items():
        project = tomllib.loads((Path(directory) / "pyproject.toml").read_text())[
            "project"
        ]
        if canonicalize_name(project["name"]) != name:
            raise ValueError(f"Wrong project name in {directory}")

        optional = project.get("optional-dependencies", {})
        catalog[name] = {
            "name": name,
            "version": str(Version(project["version"])),
            "requires_dist": project.get("dependencies", [])
            + [raw for dependencies in optional.values() for raw in dependencies],
            "provides_extra": list(optional),
            "requires_python": project.get("requires-python", ""),
        }

    sdk = Path("dist/sdk")
    if sdk.exists():
        files = _local_wheels(sdk)
        if len(files) != 1 or files[0]["name"] != "inference-sdk":
            raise ValueError("Expected one SDK reference wheel")

        catalog["inference-sdk"] = files[0]

    return catalog


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ValueError("Dependency operation timed out")

    return remaining


class _Registry:
    def __init__(self, *, wait_seconds=600, deadline=None):
        self.deadline = (
            time.monotonic() + wait_seconds if deadline is None else deadline
        )
        self.cache = {}

    def _json(self, path):
        def read():
            try:
                result = json.loads(
                    release._read_url(
                        f"https://pypi.org/pypi/{path}/json", deadline=self.deadline
                    )
                )
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    return None
                raise

            return result

        result = _retry(read, deadline=self.deadline)
        return result

    def _versions(self, name):
        _remaining(self.deadline)
        project = self._json(name)
        if project is None:
            return []

        versions = []
        for version, files in project["releases"].items():
            if any(
                not file.get("yanked")
                and (
                    file["filename"].endswith("-py3-none-any.whl")
                    or (
                        name in publish.SDISTS
                        and not any(f["filename"].endswith(".whl") for f in files)
                        and file["filename"].endswith(".tar.gz")
                    )
                )
                for file in files
            ):
                versions.append(Version(version))

        ordered = sorted(versions, reverse=True)
        return ordered

    def _file(self, name, version):
        _remaining(self.deadline)
        key = (name, str(version))
        if key in self.cache:
            return self.cache[key]

        if name in PROJECTS:
            files = publish._remote_inventory(
                name, version=str(version), deadline=self.deadline
            )
            if files is None:
                return None
            file = next(
                (file for file in files if file["filename"].endswith(".whl")),
                files[0],
            )
            file = {
                **file,
                "complete": {file["filename"] for file in files}
                == publish._filenames(name, file["version"]),
                "published_files": files,
            }
            if file["complete"]:
                self.cache[key] = file
            return file

        published = self._json(f"{name}/{version}")
        if published is None:
            return None

        if canonicalize_name(published["info"]["name"]) != name or Version(
            published["info"]["version"]
        ) != Version(version):
            raise ValueError(f"Registry identity mismatch for {name}=={version}")

        wheels = [
            file
            for file in published["urls"]
            if file["filename"].endswith("-py3-none-any.whl")
        ]
        if len(wheels) != 1 or wheels[0].get("yanked"):
            raise ValueError(
                f"Required non-yanked universal wheel missing for {name}=={version}"
            )

        remote = wheels[0]
        data = _retry(
            lambda: release._read_url(remote["url"], deadline=self.deadline),
            deadline=self.deadline,
        )
        if hashlib.sha256(data).hexdigest() != remote["digests"]["sha256"]:
            raise ValueError(f"Registry SHA-256 mismatch for {name}=={version}")

        file = _wheel(data, filename=remote["filename"])
        if file["name"] != name or Version(file["version"]) != Version(version):
            raise ValueError(f"Wheel identity mismatch for {name}=={version}")

        self.cache[key] = file
        return file


def _compatible(file, requirements):
    constraints = [
        req for req in requirements if canonicalize_name(req.name) == file["name"]
    ]
    specifier = SpecifierSet(",".join(str(req.specifier) for req in constraints))
    extras = {canonicalize_name(extra) for req in constraints for extra in req.extras}
    supplied = {canonicalize_name(extra) for extra in file.get("provides_extra", [])}
    python = SpecifierSet(file.get("requires_python", ""))
    compatible = (
        Version(file["version"]) in specifier
        and extras <= supplied
        and all(f"3.{minor}.0" in python for minor in range(10, 14))
    )
    return compatible


def _order(selected):
    ordered = []
    visiting = set()

    def visit(name):
        if name in visiting:
            raise ValueError(f"Dependency cycle at {name}")
        if name in ordered:
            return

        visiting.add(name)
        for requirement in _requirements([selected[name]]):
            dependency = canonicalize_name(requirement.name)
            if dependency not in selected:
                raise ValueError(f"Unplanned dependency: {dependency}")
            if dependency != name:
                visit(dependency)
        visiting.remove(name)
        ordered.append(name)

    for name in sorted(selected):
        visit(name)
    return ordered


def _plan(consumers, *, catalog, registry):
    root = _requirements(consumers)

    def resolve(selected):
        _remaining(registry.deadline)
        requirements = root + _requirements(selected.values())
        if any(not _compatible(file, requirements) for file in selected.values()):
            return None

        remaining = {
            canonicalize_name(req.name) for req in requirements
        } - selected.keys()
        if not remaining:
            try:
                _order(selected)
            except ValueError:
                return None

            return selected

        # Resolve consumers first where possible; backtracking still handles
        # constraints introduced by a later candidate's actual wheel metadata.
        name = next(
            n
            for n in (
                "streamvision",
                "roboflow-workflows",
                "inference-models",
                "inference-sdk",
            )
            if n in remaining
        )
        constraints = [
            req for req in requirements if canonicalize_name(req.name) == name
        ]
        specifier = SpecifierSet(",".join(str(req.specifier) for req in constraints))
        exact = {
            spec.version
            for spec in specifier
            if spec.operator == "==" and "*" not in spec.version
        }
        versions = (
            sorted({Version(v) for v in exact}, reverse=True)
            if exact
            else registry._versions(name)
        )
        for version in versions:
            if version not in specifier:
                continue
            try:
                file = registry._file(name, str(version))
            except (
                ValueError,
                KeyError,
                TypeError,
                zipfile.BadZipFile,
                tarfile.TarError,
                gzip.BadGzipFile,
                EOFError,
            ) as error:
                _remaining(registry.deadline)
                if exact:
                    raise
                print(f"Skipping invalid {name}=={version}: {error}", flush=True)
                continue
            if file is None:
                continue

            if not file.get("complete", True):
                local = catalog.get(name)
                if not local or Version(local["version"]) != version:
                    continue
                candidate = {**local, "action": "publish"}
            else:
                candidate = {**file, "action": "reuse"}

            result = resolve({**selected, name: candidate})
            if result is not None:
                return result

        local = catalog.get(name)
        if local and _compatible(local, requirements):
            # The version endpoint can lead the project listing during publication.
            file = registry._file(name, local["version"])
            if file is not None and file.get("complete", True):
                candidate = {**file, "action": "reuse"}
            else:
                action = "sdk" if name == "inference-sdk" else "publish"
                candidate = {**local, "action": action}
            result = resolve({**selected, name: candidate})
            if result is not None:
                return result
        return None

    selected = resolve({})
    if selected is None:
        raise ValueError(
            "No compatible first-party dependency plan (constraints or dependency cycle); prepare versions/pins/locks before releasing"
        )

    plan = {"consumers": consumers, "packages": selected, "order": _order(selected)}
    return plan


def _verify_file(expected, *, registry):
    def read():
        actual = registry._file(expected["name"], expected["version"])
        if actual is None or not actual.get("complete", True):
            raise urllib.error.HTTPError(
                "https://pypi.org", 404, "Dependency not visible", None, None
            )
        return actual

    actual = _retry(read, deadline=registry.deadline)

    if _edges(actual) != _edges(expected) or {
        canonicalize_name(extra) for extra in actual["provides_extra"]
    } != {canonicalize_name(extra) for extra in expected["provides_extra"]}:
        raise ValueError(
            f"Published dependency metadata differs from the plan: {expected['name']}"
        )

    retained = expected["action"] == "reuse" or (
        expected["action"] == "publish" and expected.get("publisher_run_id")
    )
    if retained:
        if not expected.get("sha256"):
            raise ValueError(f"Missing retained SHA-256 for {expected['name']}")

        if actual["sha256"] != expected["sha256"] or (
            expected.get("content_sha256")
            and actual["content_sha256"] != expected["content_sha256"]
        ):
            raise ValueError(
                f"Published dependency changed since planning: {expected['name']}"
            )

    if retained and actual.get("published_files") != expected.get("published_files"):
        raise ValueError(
            f"Published dependency inventory changed since planning: {expected['name']}"
        )

    return actual


def _validate_plan(plan, *, consumers):
    if plan.get("source_revision") != os.environ.get("SOURCE_REVISION", ""):
        raise ValueError("Dependency plan belongs to a different source revision")

    if plan["consumers"] != consumers:
        raise ValueError("Consumer wheels differ from the dependency plan")

    if plan["order"] != _order(plan["packages"]):
        raise ValueError("Incomplete or invalid dependency order")


def _verify_plan(plan, *, registry, consumers):
    _validate_plan(plan, consumers=consumers)
    for name in plan["order"]:
        entry = plan["packages"][name]
        if entry["action"] == "publish" and not all(
            entry.get(key) for key in ("sha256", "content_sha256", "publisher_run_id")
        ):
            raise ValueError(f"Missing retained publisher receipt for {name}")

    files = [
        _verify_file(plan["packages"][name], registry=registry)
        for name in plan["order"]
    ]
    constraints = _requirements(consumers + files)
    required = {canonicalize_name(req.name) for req in constraints}
    if required != {file["name"] for file in files} or any(
        not _compatible(file, constraints) for file in files
    ):
        raise ValueError("Published dependencies do not satisfy the consumer metadata")

    return files


class _RequestNotStarted(ValueError):
    """The API rejected a mutation, or its deadline expired before sending it."""


def _github(path, *, payload=None, deadline, candidate_read=False):
    token = os.environ["GH_TOKEN"]
    repository = os.environ.get("GITHUB_REPOSITORY")
    read_only_candidate = (
        candidate_read
        and payload is None
        and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository or "")
        and re.fullmatch(r"actions/runs/[0-9]+/(artifacts|jobs)\?[A-Za-z0-9_=&]*", path)
    )
    if repository != "roboflow/inference" and not read_only_candidate:
        raise ValueError("Publishing coordination is restricted to roboflow/inference")

    request = urllib.request.Request(
        f"https://api.github.com/repos/{repository}/{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2026-03-10",
        },
    )

    def read(*, timeout=None):
        with urllib.request.urlopen(
            request,
            timeout=min(30, _remaining(deadline)) if timeout is None else timeout,
        ) as response:
            body = response.read()
            result = json.loads(body) if body else {}
        _remaining(deadline)
        return result

    # Never retry POST: a lost response may have already started a publisher.
    if payload is not None:
        try:
            timeout = min(30, _remaining(deadline))
        except ValueError as error:
            raise _RequestNotStarted(
                "Dispatch deadline expired before sending"
            ) from error
        try:
            result = read(timeout=timeout)
        except urllib.error.HTTPError as error:
            # Explicit client rejections did not enqueue a publisher. Treat
            # timeouts/rate limits/server errors conservatively as ambiguous.
            if error.code in {400, 401, 403, 404, 405, 410, 422}:
                raise _RequestNotStarted(
                    f"Dispatch rejected with HTTP {error.code}; fix the request/access before retrying"
                ) from error
            raise ValueError(
                "Dispatch response failed; a publisher may have started; inspect workflow runs before retrying"
            ) from error
        except (OSError, HTTPException, ValueError) as error:
            raise ValueError(
                "Dispatch response failed; a publisher may have started; inspect workflow runs before retrying"
            ) from error
    else:
        result = _retry(read, deadline=deadline, retry_not_found=False)
    return result


def _dispatch(
    name,
    *,
    version,
    wait_seconds=9000,
    deadline=None,
    started=False,
    run_id=None,
    on_started=None,
    on_not_started=None,
):
    deadline = time.monotonic() + wait_seconds if deadline is None else deadline
    _remaining(deadline)
    workflow = PROJECTS[name][1]
    revision = os.environ["SOURCE_REVISION"]
    ref = os.environ["GITHUB_REF"]
    if not ref.startswith(("refs/heads/", "refs/tags/")):
        raise ValueError("Publishers require a branch or tag ref")

    parent = os.environ.get("GITHUB_RUN_ID")
    if run_id is not None and (type(run_id) is not int or run_id <= 0):
        raise ValueError("Invalid publisher run ID in coordinator checkpoint")
    original = _github(f"actions/runs/{run_id}", deadline=deadline) if run_id else None
    if parent and original is None:
        title = f"pypi-{name}-parent-{parent}"
        matches = []
        for page in range(1, 11):
            response = _github(
                f"actions/workflows/{workflow}/runs?event=workflow_dispatch"
                f"&head_sha={revision}&per_page=100&page={page}",
                deadline=deadline,
            )
            matches.extend(
                run
                for run in response["workflow_runs"]
                if run.get("display_title") == title
            )
            if len(response["workflow_runs"]) < 100:
                break
        else:
            raise ValueError(
                "Publisher history exceeds lookup limit; inspect before retrying"
            )
        if len(matches) > 1:
            raise ValueError(f"Ambiguous publisher runs for {title}")
        if matches:
            original = matches[0]

    if started and original is None:
        raise ValueError(
            f"Previously started publisher for {name} is missing; do not rebuild"
        )
    minimum_attempt = 1
    if original:
        run_id = int(original["id"])
        if (
            original["head_sha"] != revision
            or original["path"] != f".github/workflows/{workflow}"
            or original["event"] != "workflow_dispatch"
            or (
                parent
                and original.get("display_title") != f"pypi-{name}-parent-{parent}"
            )
        ):
            raise ValueError(f"Unexpected publisher identity/revision in run {run_id}")
        if original["status"] == "completed" and original["conclusion"] != "success":
            if int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")) <= 1:
                raise ValueError(
                    f"Publisher run {run_id} failed; rerun it before retrying"
                )
            minimum_attempt = int(original["run_attempt"]) + 1
            _github(
                f"actions/runs/{run_id}/rerun-failed-jobs",
                payload={},
                deadline=deadline,
            )
    else:
        current = _github(
            "commits/" + urllib.parse.quote(ref, safe=""), deadline=deadline
        )
        if current["sha"] != revision:
            raise ValueError("Candidate ref moved; freeze the ref before publishing")
        inputs = {
            "publish": "true",
            "expected_version": version,
            "expected_revision": revision,
        }
        if parent:
            inputs["coordinator_run_id"] = parent
        if on_started is not None:
            on_started(None)  # Persist intent before the non-retryable POST.
        try:
            dispatched = _github(
                f"actions/workflows/{workflow}/dispatches",
                payload={"ref": ref, "inputs": inputs},
                deadline=deadline,
            )
        except _RequestNotStarted:
            if on_not_started is not None:
                on_not_started()
            raise
        run_id = int(dispatched["workflow_run_id"])
    if on_started is not None:
        on_started(run_id)
    print(
        f"Publisher {name}=={version}: https://github.com/roboflow/inference/actions/runs/{run_id}",
        flush=True,
    )
    try:
        while True:
            _remaining(deadline)
            run = _github(f"actions/runs/{run_id}", deadline=deadline)
            if (
                run["head_sha"] != revision
                or run["path"] != f".github/workflows/{workflow}"
                or run["event"] != "workflow_dispatch"
                or (
                    parent
                    and run.get("display_title") != f"pypi-{name}-parent-{parent}"
                )
            ):
                raise ValueError(
                    f"Unexpected publisher identity/revision in run {run_id}"
                )

            if (
                run["status"] == "completed"
                and int(run.get("run_attempt", 1)) >= minimum_attempt
            ):
                if run["conclusion"] != "success":
                    raise ValueError(
                        f"Publisher run {run_id} ended with {run['conclusion']}"
                    )
                return run_id

            time.sleep(min(15, _remaining(deadline)))
    except (ValueError, OSError, HTTPException) as error:
        raise ValueError(
            f"Publisher run {run_id}: {error}; inspect it before retrying"
        ) from error


def _coordinate(plan, *, wait_seconds=9000, deadline=None, plan_file=None):
    deadline = time.monotonic() + wait_seconds if deadline is None else deadline
    runs = {}
    try:
        for name in plan["order"]:
            _remaining(deadline)
            expected = plan["packages"][name]
            if expected["action"] == "publish":

                def remember(run_id):
                    expected["publisher_started"] = True
                    if run_id is not None:
                        expected["publisher_started_run_id"] = run_id
                    if plan_file is not None:
                        release._write_record(plan_file, **plan)

                def forget_unsent():
                    expected.pop("publisher_started", None)
                    expected.pop("publisher_started_run_id", None)
                    if plan_file is not None:
                        release._write_record(plan_file, **plan)

                runs[name] = _dispatch(
                    name,
                    version=expected["version"],
                    deadline=deadline,
                    started=expected.get("publisher_started", False),
                    run_id=expected.get("publisher_started_run_id")
                    or expected.get("publisher_run_id"),
                    on_started=remember,
                    on_not_started=forget_unsent,
                )
            actual = _verify_file(expected, registry=_Registry(deadline=deadline))
            if name in runs:
                expected.update(
                    {key: actual[key] for key in ("sha256", "content_sha256")}
                )
                expected["published_files"] = actual["published_files"]
                expected["publisher_run_id"] = runs[name]
                if plan_file is not None:
                    release._write_record(plan_file, **plan)
    except (ValueError, OSError, HTTPException) as error:
        raise ValueError(f"{error}; already-started publisher runs: {runs}") from error

    return runs


def _resume_plan(original, checkpoint):
    if {k: v for k, v in checkpoint.items() if k != "packages"} != {
        k: v for k, v in original.items() if k != "packages"
    }:
        raise ValueError("Coordinator checkpoint differs from the retained plan")
    if checkpoint["packages"].keys() != original["packages"].keys():
        raise ValueError("Coordinator checkpoint changed package selection")
    added = {
        "publisher_started",
        "publisher_started_run_id",
        "publisher_run_id",
        "sha256",
        "content_sha256",
        "published_files",
    }
    for name, expected in original["packages"].items():
        actual = checkpoint["packages"][name]
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Coordinator checkpoint changed {name}")
        allowed = added if expected["action"] == "publish" else set()
        if actual.keys() - expected.keys() - allowed:
            raise ValueError(f"Unexpected coordinator checkpoint fields for {name}")
        if (
            "publisher_started" in actual
            and type(actual["publisher_started"]) is not bool
        ):
            raise ValueError(f"Invalid publisher intent for {name}")
        for field in ("publisher_started_run_id", "publisher_run_id"):
            if field in actual and (
                type(actual[field]) is not int or actual[field] <= 0
            ):
                raise ValueError(f"Invalid publisher run ID for {name}")
    return checkpoint


def _guard(package):
    expected = os.environ.get("EXPECTED_REVISION", "")
    if expected and expected != os.environ.get("SOURCE_REVISION"):
        raise ValueError(
            "Candidate revision differs from the coordinator's expected revision"
        )

    expected_version = os.environ.get("EXPECTED_VERSION", "")
    if expected_version and Version(expected_version) != Version(
        _catalog()[package]["version"]
    ):
        raise ValueError(
            "Package version differs from the coordinator's expected version"
        )


@click.command()
@click.argument(
    "operation",
    type=click.Choice(["plan", "coordinate", "verify", "gate", "guard", "resume"]),
)
@click.option(
    "--directory",
    type=click.Path(path_type=Path),
    default="dist/pypi",
)
@click.option(
    "--plan-file",
    type=click.Path(path_type=Path),
    default="dist/dependency-plan.json",
)
@click.option(
    "--checkpoint-file",
    type=click.Path(path_type=Path),
    default="release-checkpoint/dependency-plan.json",
)
@click.option(
    "--package",
    type=click.Choice(sorted(PROJECTS)),
)
@click.option(
    "--wait-seconds",
    type=click.FloatRange(
        min=0,
        max=10800,
    ),
    default=600,
    show_default=True,
)
def _main(operation, *, directory, plan_file, checkpoint_file, package, wait_seconds):
    try:
        if operation == "resume":
            plan = _resume_plan(
                json.loads(plan_file.read_text()),
                json.loads(checkpoint_file.read_text()),
            )
            release._write_record(plan_file, **plan)
            return
        if operation == "guard":
            if not package:
                raise ValueError("guard requires --package")
            _guard(package)
            return

        deadline = time.monotonic() + wait_seconds
        consumers = _local_wheels(directory)
        if operation in {"plan", "gate"}:
            catalog = _catalog() if operation != "gate" else {}
            plan = _plan(
                consumers,
                catalog=catalog,
                registry=_Registry(deadline=deadline),
            )
            release._write_record(plan_file, **plan)
            plan = json.loads(plan_file.read_text())
            print(json.dumps(plan, indent=2), flush=True)
            if operation == "plan":
                return
        else:
            plan = json.loads(plan_file.read_text())

        if operation == "coordinate":
            _validate_plan(plan, consumers=consumers)

        runs = (
            _coordinate(plan, deadline=deadline, plan_file=plan_file)
            if operation == "coordinate"
            else {}
        )
        files = _verify_plan(
            plan,
            registry=_Registry(deadline=deadline),
            consumers=consumers,
        )
        release._write_record(
            plan_file.with_name("dependency-receipt.json"),
            files=files,
            publisher_runs=runs,
        )
    except (
        ValueError,
        KeyError,
        OSError,
        HTTPException,
        zipfile.BadZipFile,
        tarfile.TarError,
        EOFError,
    ) as error:
        runs = {
            name: entry.get("publisher_started_run_id") or entry["publisher_run_id"]
            for name, entry in locals().get("plan", {}).get("packages", {}).items()
            if entry.get("publisher_started_run_id") or entry.get("publisher_run_id")
        }
        raise click.ClickException(
            f"{error}; already-started publisher runs: {runs}"
        ) from error


if __name__ == "__main__":
    _main()
