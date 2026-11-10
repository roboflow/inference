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
            (canonicalize_name(req.name), str(req.specifier), tuple(sorted(req.extras)))
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


def _retry(operation, *, wait_seconds):
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            result = operation()
            return result
        except urllib.error.HTTPError as error:
            if error.code not in {404, 429} and error.code < 500:
                raise

            reason = f"Registry HTTP {error.code}"
        except (urllib.error.URLError, TimeoutError) as error:
            reason = f"Registry unavailable: {error}"

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError(f"{reason}; dependency verification timed out")

        print(f"{reason}; retrying dependency verification", flush=True)
        time.sleep(min(10, remaining))


class _Registry:
    def __init__(self, *, wait_seconds=600):
        self.wait_seconds = wait_seconds
        self.cache = {}

    def _json(self, path):
        def read():
            try:
                result = json.loads(
                    release._read_url(f"https://pypi.org/pypi/{path}/json")
                )
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    return None
                raise

            return result

        result = _retry(read, wait_seconds=self.wait_seconds)
        return result

    def _versions(self, name):
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
            lambda: release._read_url(remote["url"]), wait_seconds=self.wait_seconds
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
            visit(dependency)
        visiting.remove(name)
        ordered.append(name)

    for name in sorted(selected):
        visit(name)
    return ordered


def _plan(consumers, *, catalog, registry):
    root = _requirements(consumers)

    def resolve(selected):
        requirements = root + _requirements(selected.values())
        if any(not _compatible(file, requirements) for file in selected.values()):
            return None

        remaining = {
            canonicalize_name(req.name) for req in requirements
        } - selected.keys()
        if not remaining:
            _order(selected)
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
            "No compatible first-party dependency plan; prepare versions/pins/locks before releasing"
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

    actual = _retry(read, wait_seconds=registry.wait_seconds)

    if _edges(actual) != _edges(expected) or set(actual["provides_extra"]) != set(
        expected["provides_extra"]
    ):
        raise ValueError(
            f"Published dependency metadata differs from the plan: {expected['name']}"
        )

    if expected["action"] == "reuse" and actual["sha256"] != expected["sha256"]:
        raise ValueError(
            f"Published dependency changed since planning: {expected['name']}"
        )

    return actual


def _verify_plan(plan, *, registry, consumers):
    if plan.get("source_revision") != os.environ.get("SOURCE_REVISION", ""):
        raise ValueError("Dependency plan belongs to a different source revision")

    if plan["consumers"] != consumers:
        raise ValueError("Consumer wheels differ from the dependency plan")

    if plan["order"] != _order(plan["packages"]):
        raise ValueError("Incomplete or invalid dependency order")

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


def _github(path, *, payload=None):
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
    # Do not retry POST: a lost response could otherwise start a second publisher.
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    return result


def _dispatch(name, *, version, wait_seconds):
    workflow = PROJECTS[name][1]
    revision = os.environ["SOURCE_REVISION"]
    ref = os.environ["GITHUB_REF"]
    if not ref.startswith(("refs/heads/", "refs/tags/")):
        raise ValueError("Publishers require a branch or tag ref")

    # The child independently checks this revision before any build/upload.
    current = _github("commits/" + urllib.parse.quote(ref, safe=""))
    if current["sha"] != revision:
        raise ValueError("Candidate ref moved; freeze the ref before publishing")

    dispatched = _github(
        f"actions/workflows/{workflow}/dispatches",
        payload={
            "ref": ref,
            "return_run_details": True,
            "inputs": {
                "publish": "true",
                "expected_version": version,
                "expected_revision": revision,
            },
        },
    )
    run_id = int(dispatched["workflow_run_id"])
    print(
        f"Dispatched {name}=={version}: https://github.com/roboflow/inference/actions/runs/{run_id}",
        flush=True,
    )
    deadline = time.monotonic() + wait_seconds
    while True:
        run = _github(f"actions/runs/{run_id}")
        if (
            run["head_sha"] != revision
            or run["path"] != f".github/workflows/{workflow}"
            or run["event"] != "workflow_dispatch"
        ):
            raise ValueError(f"Unexpected publisher identity/revision in run {run_id}")

        if run["status"] == "completed":
            if run["conclusion"] != "success":
                raise ValueError(
                    f"Publisher run {run_id} ended with {run['conclusion']}"
                )
            return run_id

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError(
                f"Publisher run {run_id} timed out; inspect it before retrying"
            )
        time.sleep(min(15, remaining))


def _coordinate(plan, *, wait_seconds):
    runs = {}
    for name in plan["order"]:
        expected = plan["packages"][name]
        if expected["action"] == "publish":
            runs[name] = _dispatch(
                name, version=expected["version"], wait_seconds=wait_seconds
            )
        _verify_file(expected, registry=_Registry(wait_seconds=min(600, wait_seconds)))

    return runs


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
    "operation", type=click.Choice(["plan", "coordinate", "verify", "gate", "guard"])
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
    type=click.FloatRange(min=0, max=3600),
    default=1800,
)
def _main(operation, *, directory, plan_file, package, wait_seconds):
    try:
        if operation == "guard":
            if not package:
                raise ValueError("guard requires --package")
            _guard(package)
            return

        consumers = _local_wheels(directory)
        if operation in {"plan", "gate", "coordinate"}:
            catalog = _catalog() if operation != "gate" else {}
            plan = _plan(
                consumers,
                catalog=catalog,
                registry=_Registry(wait_seconds=min(600, wait_seconds)),
            )
            release._write_record(plan_file, **plan)
            plan = json.loads(plan_file.read_text())
            print(json.dumps(plan, indent=2), flush=True)
            if operation == "plan":
                return
        else:
            plan = json.loads(plan_file.read_text())

        runs = (
            _coordinate(plan, wait_seconds=wait_seconds)
            if operation == "coordinate"
            else {}
        )
        files = _verify_plan(
            plan,
            registry=_Registry(wait_seconds=min(600, wait_seconds)),
            consumers=consumers,
        )
        release._write_record(
            plan_file.with_name("dependency-receipt.json"),
            files=files,
            publisher_runs=runs,
        )
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as error:
        raise click.ClickException(str(error)) from error


if __name__ == "__main__":
    _main()
