"""Keep one immutable, source-bound distribution inventory per workflow run."""

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import click
import dependencies
import prepare

DIRECTORIES = {
    "sdk": "dist",
    "runtime": "dist",
    "dependencies": "release-plan",
    "inference-models": "inference_models/dist",
    "roboflow-workflows": "dist",
    "streamvision": "dist",
    "image-inputs": "release-inputs",
    "image-selection": "release-selection",
}


def _identity(profile, *, run_id=None):
    return {
        "schema": 2 if profile in dependencies.PROJECTS else 1,
        "profile": profile,
        "source_revision": os.environ["SOURCE_REVISION"],
        "repository": os.environ["GITHUB_REPOSITORY"],
        "run_id": str(run_id or os.environ["GITHUB_RUN_ID"]),
    }


def _files(directory):
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink in candidate: {path}")
        if path.is_file():
            files[path.relative_to(directory).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    if not files:
        raise ValueError(f"Empty candidate: {directory}")
    return files


def _pages(path, key, *, deadline):
    for page in range(1, 101):
        result = dependencies._github(
            f"{path}&per_page=100&page={page}", deadline=deadline, candidate_read=True
        )
        entries = result[key]
        yield from entries
        if len(entries) < 100:
            return
    raise ValueError("GitHub pagination limit reached; cannot safely select candidate")


def _select(profile):
    run_id = os.environ["GITHUB_RUN_ID"]
    deadline = time.monotonic() + 120
    name = f"candidate-{profile}"
    all_artifacts = list(
        _pages(f"actions/runs/{run_id}/artifacts?", "artifacts", deadline=deadline)
    )
    artifacts = [artifact for artifact in all_artifacts if artifact["name"] == name]
    extra = {}
    if profile == "dependencies":
        extra["checkpoint_id"] = _checkpoint(
            all_artifacts, run_id=run_id, deadline=deadline
        )
    if len(artifacts) > 1:
        raise ValueError(f"Ambiguous retained candidate: {name}")
    if artifacts:
        artifact = artifacts[0]
        if artifact["expired"]:
            raise ValueError(f"Retained candidate expired: {name}; do not rebuild")
        return {"name": name, "id": artifact["id"], "reused": True, **extra}

    attempt = int(os.environ.get("GITHUB_RUN_ATTEMPT", "1"))
    if attempt > 1:
        jobs = list(
            _pages(f"actions/runs/{run_id}/jobs?filter=all", "jobs", deadline=deadline)
        )
        previous = [job for job in jobs if job["run_attempt"] < attempt]
        if not previous:
            raise ValueError("Missing prior job history; cannot safely rebuild")
        for job in previous:
            for step in job["steps"]:
                if step["name"] == f"Retain candidate {profile}" and (
                    step["status"] == "in_progress"
                    or step.get("conclusion")
                    not in {None, "skipped", "failure", "cancelled", "timed_out"}
                ):
                    raise ValueError(
                        f"Retained candidate missing: {name}; do not rebuild. "
                        "Restore the original artifact or start a new candidate."
                    )
    return {"name": name, "id": "", "reused": False, **extra}


def _checkpoint(artifacts, *, run_id, deadline):
    attempt = int(os.environ.get("GITHUB_RUN_ATTEMPT", "1"))
    if attempt == 1:
        return ""
    jobs = list(
        _pages(f"actions/runs/{run_id}/jobs?filter=all", "jobs", deadline=deadline)
    )
    previous = [job for job in jobs if job["run_attempt"] < attempt]
    if not previous:
        raise ValueError("Missing coordinator history; cannot safely resume")

    # Re-run failed jobs copies untouched successful jobs into later attempts.
    # Their start time stays unchanged; only an actual execution writes a new
    # checkpoint. Keep the first attempt for each execution, across all pages.
    executions = {}
    for index, job in enumerate(previous):
        identity = (
            (job["name"], job["started_at"])
            if job.get("name") and job.get("started_at")
            else index
        )
        if identity not in executions or (
            job["run_attempt"] < executions[identity]["run_attempt"]
        ):
            executions[identity] = job
    started = [
        job["run_attempt"]
        for job in executions.values()
        if any(
            step["name"] == "Publish missing upstream versions in dependency order"
            and (
                step["status"] == "in_progress"
                or step.get("conclusion") not in {None, "skipped"}
            )
            for step in job["steps"]
        )
    ]
    if not started:
        return ""
    name = f"dependency-plan-{max(started)}"
    matches = [artifact for artifact in artifacts if artifact["name"] == name]
    if len(matches) != 1 or matches[0]["expired"]:
        raise ValueError(
            f"Missing coordinator checkpoint {name}; inspect original publishers, do not redispatch. "
            "If it cannot be restored, reconcile those publishers before starting a new coordinator run."
        )
    return matches[0]["id"]


def _seal(profile):
    if profile in dependencies.PROJECTS:
        prepare._verify(profile)
    directory = Path(DIRECTORIES[profile])
    bundle = Path("release-candidate") / profile
    bundle.mkdir(parents=True, exist_ok=False)
    shutil.copytree(directory, bundle / "files", symlinks=True)
    manifest = {**_identity(profile), "files": _files(bundle / "files")}
    if profile in dependencies.PROJECTS:
        for path in prepare._paths(profile):
            target = bundle / "inputs" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        manifest["inputs"] = _files(bundle / "inputs")
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _verify(profile, *, restored=False, run_id=None):
    bundle = Path("release-candidate") / profile
    manifest = json.loads((bundle / "manifest.json").read_text())
    identity = _identity(profile, run_id=run_id)
    if {key: manifest.get(key) for key in identity} != identity:
        raise ValueError("Retained candidate identity does not match this run/source")
    if _files(bundle / "files") != manifest["files"]:
        raise ValueError("Retained candidate inventory or SHA-256 changed")
    if profile in dependencies.PROJECTS:
        expected_paths = {path.as_posix() for path in prepare._paths(profile)}
        if (
            set(manifest["inputs"]) != expected_paths
            or _files(bundle / "inputs") != manifest["inputs"]
        ):
            raise ValueError("Retained preparation inventory or SHA-256 changed")
        prepare._verify(profile, root=bundle / "inputs", run_id=run_id)
        project, lock, receipt = prepare._paths(profile)
        if project.read_bytes() != (bundle / "inputs" / project).read_bytes():
            raise ValueError("Checkout metadata differs from the prepared candidate")
        if restored:
            prepare._verify(profile, run_id=run_id)
            if any(
                path.read_bytes() != (bundle / "inputs" / path).read_bytes()
                for path in (lock, receipt)
            ):
                raise ValueError("Restored preparation differs from the candidate")
    if restored and _files(Path(DIRECTORIES[profile])) != manifest["files"]:
        raise ValueError("Restored candidate inventory or SHA-256 changed")


def _restore(profile):
    _verify(profile)
    directory = Path(DIRECTORIES[profile])
    if directory.exists():
        raise ValueError(
            f"Refusing to overwrite existing candidate directory: {directory}"
        )
    if profile in dependencies.PROJECTS:
        # The checkout intentionally still has its original lock. Restore the
        # tested lock and receipt, never resolve dependencies on a retry.
        for path in prepare._paths(profile)[1:]:
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path("release-candidate") / profile / "inputs" / path, path)
    shutil.copytree(Path("release-candidate") / profile / "files", directory)
    _verify(profile, restored=True)


@click.command()
@click.argument("operation", type=click.Choice(["select", "seal", "restore", "verify"]))
@click.option(
    "--profile",
    type=click.Choice(sorted(DIRECTORIES)),
    required=True,
)
def _main(operation, *, profile):
    try:
        if operation == "select":
            selection = _select(profile)
            output = Path("release-evidence") / f"selection-{profile}.json"
            output.parent.mkdir(exist_ok=True)
            output.write_text(json.dumps(selection, indent=2) + "\n")
            print(json.dumps(selection))
        elif operation == "seal":
            _seal(profile)
        elif operation == "restore":
            _restore(profile)
        else:
            _verify(profile, restored=True)
    except (OSError, ValueError, KeyError) as error:
        raise click.ClickException(str(error)) from error


if __name__ == "__main__":
    _main()
