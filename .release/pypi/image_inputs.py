"""Bind release image builds to one coordinator and its prepared source inputs."""

import json
import os
import shutil
import time
import urllib.parse
from pathlib import Path

import candidate
import click
import dependencies
import prepare
import release

WORKFLOW = ".github/workflows/publish.pypi.yml"


def _release_tag():
    tag = os.environ.get("RELEASE_TAG", "")
    if os.environ.get("GITHUB_EVENT_NAME") == "release" and not tag:
        raise ValueError("Release event is missing its exact tag")
    return tag


def _validate_parent(run):
    tag = _release_tag()
    if (
        run["path"] != WORKFLOW
        or run["head_sha"] != os.environ["SOURCE_REVISION"]
        or run["event"] not in {"release", "workflow_dispatch"}
        or (tag and (run["event"] != "release" or run["head_branch"] != tag))
    ):
        raise ValueError("Coordinator workflow, source or release tag does not match")


def _select_parent(*, parent_run_id=None, wait_seconds=600):
    deadline = time.monotonic() + wait_seconds
    if parent_run_id:
        run = dependencies._github(f"actions/runs/{parent_run_id}", deadline=deadline)
        _validate_parent(run)
    else:
        tag = _release_tag()
        if not tag:
            raise ValueError("Manual image builds require an explicit release_run_id")
        while True:
            query = urllib.parse.urlencode(
                {"event": "release", "head_sha": os.environ["SOURCE_REVISION"]}
            )
            matches = [
                run
                for run in candidate._pages(
                    f"actions/workflows/publish.pypi.yml/runs?{query}",
                    "workflow_runs",
                    deadline=deadline,
                )
                if run["head_branch"] == tag
            ]
            if len(matches) > 1:
                raise ValueError(
                    "Ambiguous release coordinators; reconcile the original runs"
                )
            if matches:
                run = matches[0]
                _validate_parent(run)
                break
            time.sleep(min(15, dependencies._remaining(deadline)))

    release._write_record(
        Path("release-selection/coordinator.json"),
        coordinator_run_id=int(run["id"]),
        release_tag=_release_tag(),
    )


def _wait(*, parent_run_id=None, wait_seconds=21000):
    selection = json.loads(Path("release-selection/coordinator.json").read_text())
    selected = selection["coordinator_run_id"]
    if (
        type(selected) is not int
        or selected <= 0
        or (parent_run_id and parent_run_id != selected)
    ):
        raise ValueError("Image retry selected a different coordinator")
    if selection["release_tag"] != _release_tag():
        raise ValueError("Image retry changed release tag")

    deadline = time.monotonic() + wait_seconds
    while True:
        run = dependencies._github(f"actions/runs/{selected}", deadline=deadline)
        _validate_parent(run)
        if run["status"] == "completed":
            if run["conclusion"] != "success":
                raise ValueError(
                    f"Coordinator {selected} failed; retry it before the image run"
                )
            break
        time.sleep(min(15, dependencies._remaining(deadline)))

    artifacts = [
        artifact
        for artifact in candidate._pages(
            f"actions/runs/{selected}/artifacts?", "artifacts", deadline=deadline
        )
        if artifact["name"] == "candidate-image-inputs"
    ]
    if len(artifacts) != 1 or artifacts[0]["expired"]:
        raise ValueError(
            "Prepared image inputs are missing, expired or ambiguous; do not rebuild"
        )
    release._write_record(
        Path("release-evidence/image-selection.json"),
        run_id=selected,
        artifact_id=artifacts[0]["id"],
    )


def _assemble():
    plan_path = Path("dist/dependency-plan.json")
    plan = json.loads(plan_path.read_text())
    dependencies._validate_plan(
        plan, consumers=dependencies._local_wheels(Path("dist/pypi"))
    )
    output = Path("release-inputs")
    output.mkdir(exist_ok=False)
    origins = {}
    catalog = dependencies._catalog()
    for profile in dependencies.PROJECTS:
        entry = plan["packages"][profile]
        if entry["version"] != catalog[profile]["version"] or dependencies._edges(
            entry
        ) != dependencies._edges(catalog[profile]):
            raise ValueError(
                f"Image source for {profile} differs from the selected release package"
            )
        if entry["action"] == "publish":
            origin = entry["publisher_run_id"]
            candidate._verify(profile, run_id=origin)
            bundle = Path("release-candidate") / profile
            for path in prepare._paths(profile):
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(bundle / "inputs" / path, path)
            publisher = output / "publishers" / f"{profile}.json"
            publisher.parent.mkdir(exist_ok=True)
            shutil.copyfile(bundle / "manifest.json", publisher)
        elif entry["action"] == "reuse":
            # Reused PyPI packages have no child run in this release. Their image
            # sources still need a fresh, retained lock with the selected versions.
            origin = int(os.environ["GITHUB_RUN_ID"])
            prepare._prepare(profile, plan=plan)
        else:
            raise ValueError(f"Unexpected companion action for {profile}")
        prepare._verify(profile, run_id=origin)
        for path in prepare._paths(profile):
            target = output / "inputs" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        origins[profile] = {"run_id": origin, "action": entry["action"]}

    shutil.copyfile(plan_path, output / "dependency-plan.json")
    release._write_record(
        output / "release-inputs.json",
        source_ref=os.environ["GITHUB_REF"],
        release_tag=_release_tag(),
        profiles=origins,
    )


def _restore(*, parent_run_id):
    candidate._verify("image-inputs", run_id=parent_run_id)
    bundle = Path("release-candidate/image-inputs")
    files = bundle / "files"
    record = json.loads((files / "release-inputs.json").read_text())
    if (
        record["source_revision"] != os.environ["SOURCE_REVISION"]
        or record["repository"] != os.environ["GITHUB_REPOSITORY"]
        or record["verification_run_id"] != str(parent_run_id)
        or set(record["profiles"]) != set(dependencies.PROJECTS)
        or (_release_tag() and record["release_tag"] != _release_tag())
    ):
        raise ValueError("Prepared image inputs do not match the release/source")

    # Validate ALL metadata before changing any checkout lock. No arbitrary
    # archive paths become overlays; only the three known lock paths are copied.
    for profile, origin in record["profiles"].items():
        prepare._verify(profile, root=files / "inputs", run_id=origin["run_id"])
        project, _, _ = prepare._paths(profile)
        if project.read_bytes() != (files / "inputs" / project).read_bytes():
            raise ValueError(f"Checkout metadata changed for {profile}")
    for profile in dependencies.PROJECTS:
        _, lock, _ = prepare._paths(profile)
        shutil.copyfile(files / "inputs" / lock, lock)
    release._write_record(
        Path("release-evidence/image-inputs.json"),
        coordinator_run_id=parent_run_id,
        inputs_sha256=prepare._sha256((bundle / "manifest.json").read_bytes()),
        locks={
            str(prepare._paths(p)[1]): prepare._sha256(
                prepare._paths(p)[1].read_bytes()
            )
            for p in dependencies.PROJECTS
        },
    )


@click.command()
@click.argument(
    "operation", type=click.Choice(["select", "wait", "assemble", "restore"])
)
@click.option(
    "--parent-run-id",
    type=click.IntRange(min=1),
)
@click.option(
    "--wait-seconds",
    type=click.FloatRange(min=1, max=21000),
    default=600,
)
def _main(operation, *, parent_run_id, wait_seconds):
    try:
        if operation == "select":
            _select_parent(parent_run_id=parent_run_id, wait_seconds=wait_seconds)
        elif operation == "wait":
            _wait(parent_run_id=parent_run_id, wait_seconds=wait_seconds)
        elif operation == "assemble":
            _assemble()
        else:
            if not parent_run_id:
                raise ValueError("restore requires --parent-run-id")
            _restore(parent_run_id=parent_run_id)
    except (OSError, ValueError, KeyError) as error:
        raise click.ClickException(str(error)) from error


if __name__ == "__main__":
    _main()
