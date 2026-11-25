"""Resolve a companion's release lock only when creating a new candidate."""

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from time import sleep

import click
import dependencies
import release


def _paths(profile):
    directory = Path(dependencies.PROJECTS[profile][0])
    paths = [
        directory / "pyproject.toml",
        directory / "uv.lock",
        Path("release-evidence") / f"preparation-{profile}.json",
    ]
    return paths


def _sha256(data):
    digest = hashlib.sha256(data).hexdigest()
    return digest


def _verify(profile, *, root=Path("."), run_id=None):
    project, lock, receipt = _paths(profile)
    record = json.loads((root / receipt).read_text())
    expected = {
        "profile": profile,
        "source_revision": os.environ["SOURCE_REVISION"],
        "repository": os.environ["GITHUB_REPOSITORY"],
        "verification_run_id": str(run_id or os.environ["GITHUB_RUN_ID"]),
        "pyproject_sha256": _sha256((root / project).read_bytes()),
        "lock_sha256": _sha256((root / lock).read_bytes()),
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Prepared source/lock identity differs from this candidate")


def _upstreams(profile, *, plan):
    selected = {}
    visiting = set()

    def visit(package):
        for requirement in dependencies._requirements([package]):
            name = dependencies.canonicalize_name(requirement.name)
            if name == profile or name in visiting:
                raise ValueError("Cyclic first-party release dependency")

            entry = plan["packages"][name]
            if not dependencies._compatible(entry, [requirement]):
                raise ValueError(f"Prepared source requires a different {name} version")

            if name in selected:
                continue

            visiting.add(name)
            visit(entry)
            visiting.remove(name)
            selected[name] = entry["version"]

    visit(dependencies._catalog()[profile])
    return selected


def _resolve(command, *, directory):
    deadline = time.monotonic() + 600
    for attempt in range(3):
        try:
            subprocess.run(
                command,
                cwd=directory,
                check=True,
                timeout=dependencies._remaining(deadline),
            )
            return
        except subprocess.CalledProcessError:
            if attempt == 2:
                raise

            delay = min(10 * (attempt + 1), dependencies._remaining(deadline))
            click.echo(f"Lock resolution failed; retrying in {delay:g}s", err=True)
            sleep(delay)


def _prepare(profile, *, plan=None):
    if (Path("release-candidate") / profile).exists():
        raise ValueError("Retained candidate exists; restore it, do not resolve again")

    dependencies._guard(profile)
    project, lock, receipt = _paths(profile)
    if receipt.exists():
        raise ValueError("Preparation already exists; do not resolve it again")

    original = lock.read_bytes() if lock.exists() else None
    project_bytes = project.read_bytes()
    command = [sys.executable, "-m", "uv", "lock", "--python", sys.executable]
    # Targeted upgrades replace compatible RCs while retaining unrelated pins.
    upstreams = _upstreams(profile, plan=plan) if plan is not None else None
    for name in sorted(dependencies.OWNED - {profile}):
        requirement = (
            f"{name}=={upstreams[name]}" if upstreams and name in upstreams else name
        )
        command.extend(["--upgrade-package", requirement])

    try:
        _resolve(command, directory=project.parent)
        subprocess.run(
            [sys.executable, "-m", "uv", "lock", "--check", "--offline"],
            cwd=project.parent,
            check=True,
            timeout=60,
        )
        if project.read_bytes() != project_bytes:
            raise ValueError("Lock preparation changed package metadata")

        if upstreams is not None:
            resolved = {}
            for package in tomllib.loads(lock.read_text())["package"]:
                if package["name"] in dependencies.OWNED - {profile}:
                    resolved.setdefault(package["name"], set()).add(package["version"])
            if resolved != {name: {version} for name, version in upstreams.items()}:
                raise ValueError(
                    "Prepared lock differs from the release dependency plan"
                )

        release._write_record(
            receipt,
            profile=profile,
            uv_version=importlib.metadata.version("uv"),
            command=command,
            upstreams=upstreams,
            pyproject_sha256=_sha256(project_bytes),
            original_lock_sha256=_sha256(original) if original is not None else None,
            lock_sha256=_sha256(lock.read_bytes()),
        )
        _verify(profile)
    except (OSError, ValueError, subprocess.SubprocessError):
        # Failed preparation leaves no candidate and restores the original lock.
        if original is None:
            lock.unlink(missing_ok=True)
        else:
            lock.write_bytes(original)
        receipt.unlink(missing_ok=True)
        raise


@click.command()
@click.option(
    "--profile",
    type=click.Choice(sorted(dependencies.PROJECTS)),
    required=True,
)
def _main(*, profile):
    try:
        plan = None
        parent = os.environ.get("COORDINATOR_RUN_ID")
        if parent:
            import candidate

            candidate._verify("dependencies", run_id=parent)
            plan = json.loads(
                Path(
                    "release-candidate/dependencies/files/dependency-plan.json"
                ).read_text()
            )
            expected = plan["packages"][profile]
            current = dependencies._catalog()[profile]
            if expected["version"] != current["version"] or dependencies._edges(
                expected
            ) != dependencies._edges(current):
                raise ValueError("Publisher source differs from the coordinator plan")
        _prepare(profile, plan=plan)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        raise click.ClickException(
            f"Lock preparation failed for {profile}: {error}. "
            "Required upstream versions must be published before this publisher; "
            "retry the original coordinator after fixing the prerequisite."
        ) from error


if __name__ == "__main__":
    _main()
