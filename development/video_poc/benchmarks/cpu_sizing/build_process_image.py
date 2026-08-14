#!/usr/bin/env python3
"""Build a CPU process overlay from exact committed bytes and retain evidence.

Dry-run is the default. ``--execute`` creates a temporary clean build context
exclusively from ``git show REVISION:path``; it never submits the caller's
working tree. Executing Cloud Build is a staging external write and requires
separate authorization.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import render_variant

PROJECT = "roboflow-staging"
BASE = (
    "us-central1-docker.pkg.dev/roboflow-staging/video-proc/"
    "video-processor-cpu-telemetry@sha256:"
    "106cf74b916f55d1cfc4322d9b361fb1fbed86889dd8d8eb435904c950801d6a"
)
OUTPUT_REPOSITORY = (
    "us-central1-docker.pkg.dev/roboflow-staging/video-proc/"
    "video-processor-cpu-telemetry"
)
PROCESSOR_DIRECTORY = Path("development/video_poc/processor")


def _run_json(command):
    result = subprocess.run(command, check=True, capture_output=True)
    return json.loads(result.stdout)


def materialize_exact_context(repository, revision, destination):
    manifest = render_variant._exact_revision_source_manifest(repository, revision)
    for path in manifest:
        relative = Path(path).relative_to(PROCESSOR_DIRECTORY)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        content = subprocess.run(
            ["git", "-C", str(repository), "show", f"{revision}:{path}"],
            check=True,
            capture_output=True,
        ).stdout
        if render_variant._sha256_bytes(content) != manifest[path]:
            raise RuntimeError(f"source changed while materializing {path}")
        target.write_bytes(content)
    written = {
        str(
            PROCESSOR_DIRECTORY / path.relative_to(destination)
        ): render_variant._sha256_bytes(path.read_bytes())
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    if written != manifest:
        raise RuntimeError("clean build context differs from exact revision manifest")
    return manifest


def build_commands(revision, context, image=None):
    output = f"{OUTPUT_REPOSITORY}:{revision}"
    substitutions = ",".join(
        (
            f"_BASE_PROCESSOR_IMAGE={BASE}",
            f"_OUTPUT={output}",
            f"_GIT_SHA={revision}",
            "_RUNTIME_VARIANT=cpu-process",
        )
    )
    commands = {
        "build": [
            "gcloud",
            "builds",
            "submit",
            str(context),
            f"--project={PROJECT}",
            f"--config={context / 'cloudbuild.process-overlay.yaml'}",
            f"--substitutions={substitutions}",
            "--format=json",
        ]
    }
    if image:
        commands["registry"] = [
            "gcloud",
            "artifacts",
            "docker",
            "images",
            "describe",
            image,
            f"--project={PROJECT}",
            "--format=json",
        ]
        commands["smoke"] = [
            "gcloud",
            "builds",
            "submit",
            "--no-source",
            f"--project={PROJECT}",
            f"--config={context / 'cloudbuild.job-process-smoke.yaml'}",
            f"--substitutions=_IMAGE={image}",
            "--format=json",
        ]
    return commands


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return {
        "path": path.name,
        "sha256": render_variant._sha256_bytes(path.read_bytes()),
    }


def execute(repository, revision, output_dir):
    output_dir = Path(output_dir).resolve()
    try:
        output_dir.relative_to(render_variant.PROVENANCE_DIR.resolve())
    except ValueError as error:
        raise ValueError(
            "build evidence must be retained under cpu_sizing/provenance"
        ) from error
    minimum = json.loads((Path(__file__).with_name("size_classes.json")).read_text())[
        "imagePolicy"
    ]["processMinimumSourceRevision"]
    if (
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "merge-base",
                "--is-ancestor",
                minimum,
                revision,
            ],
            check=False,
        ).returncode
        != 0
    ):
        raise ValueError("revision does not contain bounded process cleanup")
    output_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="cpu-process-build-") as temporary:
        context = Path(temporary) / "processor"
        context.mkdir()
        manifest = materialize_exact_context(repository, revision, context)
        build = _run_json(build_commands(revision, context)["build"])
        if build.get("status") != "SUCCESS":
            raise RuntimeError("Cloud Build did not succeed")
        images = render_variant._build_images(build)
        if len(images) != 1:
            raise RuntimeError(f"expected one immutable build image, found {images}")
        image = images[0]
        commands = build_commands(revision, context, image)
        registry = _run_json(commands["registry"])
        smoke = _run_json(commands["smoke"])

    descriptors = {
        "build": _write_json(output_dir / "build.json", build),
        "registry": _write_json(output_dir / "registry.json", registry),
        "smoke": _write_json(output_dir / "smoke.json", smoke),
    }
    provenance = {
        "schemaVersion": 1,
        "environment": "staging",
        "image": image,
        "baseImage": BASE,
        "sourceRevision": revision,
        "sourceFilesSha256": manifest,
        "sourceManifestSha256": render_variant._manifest_hash(manifest),
        "evidence": descriptors,
    }
    provenance_path = output_dir / "provenance.json"
    render_variant.validate_process_provenance(
        json.loads(Path(__file__).with_name("size_classes.json").read_text()),
        provenance,
        provenance_path,
    )
    _write_json(provenance_path, provenance)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if render_variant.GIT_SHA.fullmatch(args.revision) is None:
        parser.error("--revision must be an exact lowercase git SHA")
    repository = Path(__file__).resolve().parents[4]
    with tempfile.TemporaryDirectory(prefix="cpu-process-plan-") as temporary:
        context = Path(temporary) / "processor"
        context.mkdir()
        manifest = materialize_exact_context(repository, args.revision, context)
        plan = {
            "environment": "staging",
            "sourceRevision": args.revision,
            "sourceFilesSha256": manifest,
            "sourceManifestSha256": render_variant._manifest_hash(manifest),
            "commands": build_commands(args.revision, context),
        }
    if args.execute:
        result = execute(repository, args.revision, args.output_dir.resolve())
    else:
        result = plan
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
