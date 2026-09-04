#!/usr/bin/env python3
"""Render one exact staging CPU worker strategic-merge patch.

Rendering is offline. The output is JSON (also valid YAML) so it can be passed
to ``kubectl patch --type=strategic --patch-file`` only after a separate live
write authorization and the runbook preflight.
"""

import argparse
import base64
import hashlib
import json
import re
import subprocess
from pathlib import Path

DEFAULT_CATALOG = Path(__file__).with_name("size_classes.json")
PROVENANCE_DIR = Path(__file__).with_name("provenance")
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
SAFE_VARIANT = re.compile(r"[a-z0-9-]{1,63}\Z")
PROCESSOR_PREFIX = "development/video_poc/processor/"


def load_catalog(path):
    document = json.loads(Path(path).read_text())
    if document.get("schemaVersion") != 1:
        raise ValueError("size-class schemaVersion must be 1")
    if document.get("environment") != "staging":
        raise ValueError("size-class catalog must target staging")
    if document.get("clusterContext") != "ck8s-stg":
        raise ValueError("size-class catalog must target ck8s-stg")
    return document


def validate_image(image, repository):
    prefix = f"{repository}@"
    if not str(image).startswith(prefix):
        raise ValueError(f"image must use exact staging repository {repository}")
    if not DIGEST.fullmatch(str(image)[len(prefix) :]):
        raise ValueError("image must use an immutable sha256 digest")
    return str(image)


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _exact_revision_source_manifest(repository, revision):
    listing = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "ls-tree",
            "-r",
            "--name-only",
            revision,
            "--",
            PROCESSOR_PREFIX,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if not listing:
        raise ValueError("source revision has no video processor build context")
    result = {}
    for path in listing:
        content = subprocess.run(
            ["git", "-C", str(repository), "show", f"{revision}:{path}"],
            check=True,
            capture_output=True,
        ).stdout
        result[path] = _sha256_bytes(content)
    return result


def _manifest_hash(manifest):
    return _sha256_bytes(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    )


def _context_source_manifest(manifest):
    prefix = PROCESSOR_PREFIX
    return {
        path.removeprefix(prefix): digest
        for path, digest in manifest.items()
        if path.startswith(prefix)
    }


def _cloud_build_source_manifest(build):
    result = {}
    hashes = (build.get("sourceProvenance") or {}).get("fileHashes") or {}
    for path, descriptor in hashes.items():
        candidates = (descriptor or {}).get("fileHash") or []
        sha256_values = [
            item.get("value")
            for item in candidates
            if item.get("type") == "SHA256" and item.get("value")
        ]
        if len(sha256_values) != 1:
            raise ValueError(f"Cloud Build lacks one SHA256 for source file {path}")
        try:
            result[path] = base64.b64decode(sha256_values[0], validate=True).hex()
        except ValueError as error:
            raise ValueError(
                f"Cloud Build source hash is invalid for {path}"
            ) from error
    return result


def _load_evidence_file(provenance_path, descriptor):
    if set(descriptor) != {"path", "sha256"}:
        raise ValueError("evidence descriptor must contain path and sha256")
    base = Path(provenance_path).resolve().parent
    path = (base / descriptor["path"]).resolve()
    try:
        path.relative_to(PROVENANCE_DIR.resolve())
    except ValueError as error:
        raise ValueError(
            "process evidence must live in cpu_sizing/provenance"
        ) from error
    content = path.read_bytes()
    if _sha256_bytes(content) != descriptor["sha256"]:
        raise ValueError(f"process evidence hash mismatch: {path.name}")
    return json.loads(content)


def _build_images(build):
    images = []
    for item in (build.get("results") or {}).get("images") or []:
        name = item.get("name")
        digest = item.get("digest")
        if name and digest:
            images.append(f"{name.split('@', 1)[0].split(':', 1)[0]}@{digest}")
    return images


def _build_image_records(build):
    return [
        {"name": item.get("name"), "digest": item.get("digest")}
        for item in (build.get("results") or {}).get("images") or []
    ]


def validate_process_provenance(
    catalog,
    provenance,
    provenance_path=None,
    ancestor_check=None,
    source_manifest_loader=None,
    evidence_loader=None,
):
    if provenance.get("schemaVersion") != 1:
        raise ValueError("process provenance schemaVersion must be 1")
    if provenance.get("environment") != "staging":
        raise ValueError("process provenance must target staging")
    image = validate_image(
        provenance.get("image"), catalog["imagePolicy"]["repository"]
    )
    if image == catalog["observedBaseline"]["image"]:
        raise ValueError("process image cannot equal the thread-only baseline")
    if provenance.get("baseImage") != catalog["observedBaseline"]["image"]:
        raise ValueError("process image must use the exact observed CPU base")
    source_revision = str(provenance.get("sourceRevision") or "")
    if GIT_SHA.fullmatch(source_revision) is None:
        raise ValueError("process provenance needs an exact source revision")
    minimum = catalog["imagePolicy"]["processMinimumSourceRevision"]
    repository = Path(__file__).resolve().parents[4]
    if ancestor_check is None:

        def ancestor_check(base, revision):
            return (
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(repository),
                        "merge-base",
                        "--is-ancestor",
                        base,
                        revision,
                    ],
                    check=False,
                ).returncode
                == 0
            )

    if not ancestor_check(minimum, source_revision):
        raise ValueError("process source is not a descendant of the cleanup gate")

    if source_manifest_loader is None:
        source_manifest_loader = lambda revision: _exact_revision_source_manifest(
            repository, revision
        )
    exact_manifest = source_manifest_loader(source_revision)
    if provenance.get("sourceFilesSha256") != exact_manifest:
        raise ValueError(
            "process source manifest differs from exact git revision bytes"
        )
    if provenance.get("sourceManifestSha256") != _manifest_hash(exact_manifest):
        raise ValueError("process source manifest hash is invalid")

    if provenance_path is None and evidence_loader is None:
        raise ValueError("process provenance requires retained build evidence")
    if evidence_loader is None:
        evidence_loader = lambda descriptor: _load_evidence_file(
            provenance_path, descriptor
        )
    evidence = provenance.get("evidence") or {}
    if set(evidence) != {"build", "registry", "smoke"}:
        raise ValueError("process provenance needs build, registry, and smoke evidence")
    build = evidence_loader(evidence["build"])
    registry = evidence_loader(evidence["registry"])
    smoke = evidence_loader(evidence["smoke"])
    if build.get("status") != "SUCCESS":
        raise ValueError("process image build did not succeed")
    substitutions = build.get("substitutions") or {}
    expected_output = f'{catalog["imagePolicy"]["repository"]}:{source_revision}'
    expected_substitutions = {
        "_BASE_PROCESSOR_IMAGE": catalog["observedBaseline"]["image"],
        "_OUTPUT": expected_output,
        "_GIT_SHA": source_revision,
        "_RUNTIME_VARIANT": "cpu-process",
    }
    if any(
        substitutions.get(key) != value for key, value in expected_substitutions.items()
    ):
        raise ValueError("process build substitutions do not match exact source/base")
    if not (build.get("sourceProvenance") or {}).get("resolvedStorageSource"):
        raise ValueError("process build lacks resolved uploaded-source provenance")
    build_source_manifest = _cloud_build_source_manifest(build)
    if build_source_manifest != _context_source_manifest(exact_manifest):
        raise ValueError("Cloud Build input hashes differ from exact revision bytes")
    expected_digest = image.split("@", 1)[1]
    expected_result = {"name": expected_output, "digest": expected_digest}
    if _build_image_records(build) != [expected_result]:
        raise ValueError(
            "process build result tag/digest differs from the requested output"
        )
    if image not in _build_images(build):
        raise ValueError("process build results do not contain the retained digest")
    registry_digest = (registry.get("image_summary") or {}).get(
        "fully_qualified_digest"
    )
    if registry_digest != image:
        raise ValueError("registry evidence does not resolve to the process image")
    if smoke.get("status") != "SUCCESS":
        raise ValueError("process image smoke did not succeed")
    if (smoke.get("substitutions") or {}).get("_IMAGE") != image:
        raise ValueError("process smoke did not test the retained image digest")
    return {"image": image, "sourceRevision": source_revision}


def load_tracked_process_provenance(
    path, tracked_check=None, committed_content_loader=None
):
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(PROVENANCE_DIR.resolve())
    except ValueError as error:
        raise ValueError(
            "process provenance must live in cpu_sizing/provenance"
        ) from error
    repository = Path(__file__).resolve().parents[4]
    relative = resolved.relative_to(repository)
    if tracked_check is None:

        def tracked_check(candidate):
            return (
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(repository),
                        "ls-files",
                        "--error-unmatch",
                        str(candidate),
                    ],
                    check=False,
                    capture_output=True,
                ).returncode
                == 0
            )

    if committed_content_loader is None:

        def committed_content_loader(candidate):
            return subprocess.run(
                ["git", "-C", str(repository), "show", f"HEAD:{candidate}"],
                check=True,
                capture_output=True,
            ).stdout

    if not tracked_check(relative):
        raise ValueError("process provenance must be committed before rendering")
    if committed_content_loader(relative) != resolved.read_bytes():
        raise ValueError("process provenance differs from the committed HEAD bytes")
    return resolved, json.loads(resolved.read_text())


def render_patch(
    catalog,
    size_name,
    topology_name,
    process_provenance=None,
    process_provenance_path=None,
):
    try:
        size = catalog["sizeClasses"][size_name]
    except KeyError as error:
        raise ValueError(f"unknown CPU size class: {size_name}") from error
    try:
        topology = catalog["topologies"][topology_name]
    except KeyError as error:
        raise ValueError(f"unknown CPU topology: {topology_name}") from error

    if topology_name == "process":
        if process_provenance is None:
            raise ValueError(
                "process topology requires a validated process provenance record"
            )
        validated = validate_process_provenance(
            catalog, process_provenance, process_provenance_path
        )
        image = validated["image"]
        source_revision = validated["sourceRevision"]
    else:
        image = catalog["observedBaseline"]["image"]
        source_revision = catalog["observedBaseline"]["sourceRevision"]
    image = validate_image(image, catalog["imagePolicy"]["repository"])
    if GIT_SHA.fullmatch(str(source_revision)) is None:
        raise ValueError("source revision must be an exact lowercase git SHA")
    runtime_variant = f"{size_name}-{topology_name}"
    if SAFE_VARIANT.fullmatch(runtime_variant) is None:
        raise ValueError("rendered runtime variant is invalid")

    resources = {
        "requests": {"cpu": size["cpu"], "memory": size["memory"]},
        "limits": {"cpu": size["cpu"], "memory": size["memory"]},
    }
    patch = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": catalog["deployment"],
            "namespace": catalog["namespace"],
        },
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "video-poc.roboflow.com/cpu-benchmark-variant": runtime_variant
                    }
                },
                "spec": {
                    "containers": [
                        {
                            "name": catalog["container"],
                            "image": image,
                            "env": [
                                {
                                    "name": "VIDEO_PROC_IMAGE",
                                    "value": image,
                                },
                                {
                                    "name": "VIDEO_PROC_RUNTIME_VARIANT",
                                    "value": runtime_variant,
                                },
                                {
                                    "name": "VIDEO_PROC_GIT_SHA",
                                    "value": str(source_revision),
                                },
                                {
                                    "name": "MAX_CONCURRENT_JOBS",
                                    "value": str(size["maxJobs"]),
                                },
                                {
                                    "name": "ONNX_INTRA_OP_THREADS",
                                    "value": str(size["intraOpThreads"]),
                                },
                                {
                                    "name": "OMP_NUM_THREADS",
                                    "value": str(size["intraOpThreads"]),
                                },
                                {
                                    "name": "PROCESSOR_JOB_EXECUTION_MODE",
                                    "value": topology["jobExecutionMode"],
                                },
                                {
                                    "name": "PROCESSOR_EXECUTION_DOMAIN_MODE",
                                    "value": topology["executionDomainMode"],
                                },
                            ],
                            "resources": resources,
                        }
                    ]
                },
            }
        },
    }
    # Keep the renderer and the live preflight on one exact schema. Importing
    # here avoids making offline provenance helpers depend on kubectl.
    from verify_rollout_patch import validate_patch_document

    validate_patch_document(catalog, patch)
    return patch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--size", required=True)
    parser.add_argument("--topology", choices=("thread", "process"), required=True)
    parser.add_argument(
        "--process-provenance",
        type=Path,
        help="required checked/retained build record for process topology",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    loaded = (
        load_tracked_process_provenance(args.process_provenance)
        if args.process_provenance
        else None
    )
    provenance_path, provenance = loaded if loaded else (None, None)
    patch = render_patch(
        load_catalog(args.catalog),
        args.size,
        args.topology,
        provenance,
        provenance_path,
    )
    rendered = json.dumps(patch, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
