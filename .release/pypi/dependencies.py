"""Resolve and verify first-party dependencies without installing runtime packages."""

import hashlib
import io
import json
import os
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
import release
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

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


def _retry(operation, *, deadline, retry_not_found=True):
    while True:
        _remaining(deadline)
        try:
            result = operation()
            _remaining(deadline)
            return result
        except urllib.error.HTTPError as error:
            if (
                not (error.code == 404 and retry_not_found)
                and error.code != 429
                and error.code < 500
            ):
                raise

            reason = f"HTTP {error.code}"
        except (
            urllib.error.URLError,
            TimeoutError,
            HTTPException,
            ConnectionError,
        ) as error:
            reason = f"Network unavailable: {error}"

        remaining = _remaining(deadline)
        print(f"{reason}; retrying dependency operation", flush=True)
        time.sleep(min(10, remaining))


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
                and file["filename"].endswith("-py3-none-any.whl")
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
            file = registry._file(name, str(version))
            if file is None:
                continue

            result = resolve({**selected, name: {**file, "action": "reuse"}})
            if result is not None:
                return result

        local = catalog.get(name)
        if (
            local
            and _compatible(local, requirements)
            and registry._file(name, local["version"]) is None
        ):
            action = "sdk" if name == "inference-sdk" else "publish"
            result = resolve({**selected, name: {**local, "action": action}})
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
        if actual is None:
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


def _github(path, *, payload=None, deadline):
    token = os.environ["GH_TOKEN"]
    repository = os.environ.get("GITHUB_REPOSITORY")
    if repository != "roboflow/inference":
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

    def read():
        with urllib.request.urlopen(
            request, timeout=min(30, _remaining(deadline))
        ) as response:
            result = json.load(response)
        _remaining(deadline)
        return result

    # Never retry POST: a lost response may have already started a publisher.
    if payload is not None:
        try:
            result = read()
        except (OSError, HTTPException, ValueError) as error:
            raise ValueError(
                "Dispatch response failed; a publisher may have started; inspect workflow runs before retrying"
            ) from error
    else:
        result = _retry(read, deadline=deadline, retry_not_found=False)
    return result


def _dispatch(name, *, version, wait_seconds=9000, deadline=None):
    deadline = time.monotonic() + wait_seconds if deadline is None else deadline
    _remaining(deadline)
    workflow = PROJECTS[name][1]
    revision = os.environ["SOURCE_REVISION"]
    ref = os.environ["GITHUB_REF"]
    if not ref.startswith(("refs/heads/", "refs/tags/")):
        raise ValueError("Publishers require a branch or tag ref")

    current = _github("commits/" + urllib.parse.quote(ref, safe=""), deadline=deadline)
    if current["sha"] != revision:
        raise ValueError("Candidate ref moved; freeze the ref before publishing")

    dispatched = _github(
        f"actions/workflows/{workflow}/dispatches",
        payload={
            "ref": ref,
            "inputs": {
                "publish": "true",
                "expected_version": version,
                "expected_revision": revision,
            },
        },
        deadline=deadline,
    )
    run_id = int(dispatched["workflow_run_id"])
    print(
        f"Dispatched {name}=={version}: https://github.com/roboflow/inference/actions/runs/{run_id}",
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
            ):
                raise ValueError(
                    f"Unexpected publisher identity/revision in run {run_id}"
                )

            if run["status"] == "completed":
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
                runs[name] = _dispatch(
                    name, version=expected["version"], deadline=deadline
                )
            actual = _verify_file(expected, registry=_Registry(deadline=deadline))
            if name in runs:
                expected.update(
                    {key: actual[key] for key in ("sha256", "content_sha256")}
                )
                expected["publisher_run_id"] = runs[name]
                if plan_file is not None:
                    release._write_record(plan_file, **plan)
    except (ValueError, OSError, HTTPException) as error:
        raise ValueError(f"{error}; already-started publisher runs: {runs}") from error

    return runs


def _verify_published(package, *, directory, plan_file, deadline):
    files = _local_wheels(directory)
    expected = _catalog()[package]
    if (
        len(files) != 1
        or files[0]["name"] != package
        or Version(files[0]["version"]) != Version(expected["version"])
    ):
        raise ValueError(
            f"Expected exactly one built {package}=={expected['version']} wheel"
        )

    candidate = files[0]
    actual = _verify_file(
        {**candidate, "action": "publish", "sha256": None, "content_sha256": None},
        registry=_Registry(deadline=deadline),
    )
    release._write_record(
        plan_file.with_name("published-content-receipt.json"),
        candidate=candidate,
        published=actual,
    )
    if actual["content_sha256"] != candidate["content_sha256"]:
        raise ValueError(
            f"Published {package} contents differ from the built candidate; use a new version"
        )

    return actual


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
    type=click.Choice(
        ["plan", "coordinate", "verify", "gate", "guard", "verify-published"]
    ),
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
def _main(operation, *, directory, plan_file, package, wait_seconds):
    try:
        if operation == "guard":
            if not package:
                raise ValueError("guard requires --package")
            _guard(package)
            return

        deadline = time.monotonic() + wait_seconds
        if operation == "verify-published":
            if not package:
                raise ValueError("verify-published requires --package")

            _verify_published(
                package, directory=directory, plan_file=plan_file, deadline=deadline
            )
            return

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
    except (ValueError, KeyError, OSError, HTTPException, zipfile.BadZipFile) as error:
        runs = {
            name: entry["publisher_run_id"]
            for name, entry in locals().get("plan", {}).get("packages", {}).items()
            if entry.get("publisher_run_id")
        }
        raise click.ClickException(
            f"{error}; already-started publisher runs: {runs}"
        ) from error


if __name__ == "__main__":
    _main()
