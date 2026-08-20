#!/usr/bin/env python3
"""Fail-closed, read-only preflight for a staging CPU Deployment patch.

The helper reads the live Deployment/pods and asks the API server to render a
dry-run strategic merge. It never issues a persistent Kubernetes write.
"""

import argparse
import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path

CONTEXT = "ck8s-stg"
NAMESPACE = "video-proc"
DEPLOYMENT = "video-processor-pool-cpu"
CONTAINER = "processor"
EXPERIMENT_ENVS = {
    "VIDEO_PROC_IMAGE",
    "VIDEO_PROC_RUNTIME_VARIANT",
    "VIDEO_PROC_GIT_SHA",
    "MAX_CONCURRENT_JOBS",
    "ONNX_INTRA_OP_THREADS",
    "OMP_NUM_THREADS",
    "PROCESSOR_JOB_EXECUTION_MODE",
    "PROCESSOR_EXECUTION_DOMAIN_MODE",
}
PATCH_ANNOTATION = "video-poc.roboflow.com/cpu-benchmark-variant"
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
CPU_VARIANT = re.compile(r"(cpu-[248])-(thread|process)\Z")


def _require_exact_keys(document, expected, location):
    if not isinstance(document, dict) or set(document) != set(expected):
        actual = (
            sorted(document) if isinstance(document, dict) else type(document).__name__
        )
        raise ValueError(
            f"{location} keys must be exactly {sorted(expected)}; found {actual}"
        )


def validate_patch_document(catalog, patch):
    """Reject extra metadata/spec fields before asking Kubernetes to render."""
    _require_exact_keys(patch, {"apiVersion", "kind", "metadata", "spec"}, "patch")
    if patch["apiVersion"] != "apps/v1" or patch["kind"] != "Deployment":
        raise ValueError("patch must target an apps/v1 Deployment")
    _require_exact_keys(patch["metadata"], {"name", "namespace"}, "patch metadata")
    if patch["metadata"] != {
        "name": catalog["deployment"],
        "namespace": catalog["namespace"],
    }:
        raise ValueError("patch metadata differs from the audited Deployment")
    _require_exact_keys(patch["spec"], {"template"}, "patch spec")
    template = patch["spec"]["template"]
    _require_exact_keys(template, {"metadata", "spec"}, "pod template")
    _require_exact_keys(template["metadata"], {"annotations"}, "pod metadata")
    annotations = template["metadata"]["annotations"]
    _require_exact_keys(annotations, {PATCH_ANNOTATION}, "pod annotations")
    _require_exact_keys(template["spec"], {"containers"}, "pod spec")
    containers = template["spec"]["containers"]
    if not isinstance(containers, list) or len(containers) != 1:
        raise ValueError("patch must contain exactly one container")
    container = containers[0]
    _require_exact_keys(
        container, {"name", "image", "env", "resources"}, "processor container"
    )
    if container["name"] != catalog["container"]:
        raise ValueError("patch container differs from the audited processor")
    env = container["env"]
    if not isinstance(env, list) or len(env) != len(EXPERIMENT_ENVS):
        raise ValueError("patch must contain each experiment environment exactly once")
    for index, item in enumerate(env):
        _require_exact_keys(item, {"name", "value"}, f"environment entry {index}")
        if not isinstance(item["name"], str) or not isinstance(item["value"], str):
            raise ValueError("patch environment names and values must be strings")
    env_map = _env_map(container)
    if set(env_map) != EXPERIMENT_ENVS or len(env_map) != len(env):
        raise ValueError("patch experiment environment allowlist is incomplete")
    variant = env_map["VIDEO_PROC_RUNTIME_VARIANT"]["value"]
    match = CPU_VARIANT.fullmatch(variant)
    if match is None:
        raise ValueError("patch runtime variant is not a known CPU size/topology")
    size_name, topology_name = match.groups()
    try:
        size = catalog["sizeClasses"][size_name]
        topology = catalog["topologies"][topology_name]
    except KeyError as error:
        raise ValueError("patch runtime variant is absent from the catalog") from error
    if annotations[PATCH_ANNOTATION] != variant:
        raise ValueError("patch annotation and runtime variant differ")
    if env_map["VIDEO_PROC_IMAGE"]["value"] != container["image"]:
        raise ValueError("patch image environment and container image differ")
    image_prefix = f'{catalog["imagePolicy"]["repository"]}@'
    if (
        not container["image"].startswith(image_prefix)
        or DIGEST.fullmatch(container["image"][len(image_prefix) :]) is None
    ):
        raise ValueError("patch image is not an immutable staging CPU digest")
    revision = env_map["VIDEO_PROC_GIT_SHA"]["value"]
    if GIT_SHA.fullmatch(revision) is None:
        raise ValueError("patch source revision is not an exact Git SHA")
    expected_environment = {
        "MAX_CONCURRENT_JOBS": str(size["maxJobs"]),
        "ONNX_INTRA_OP_THREADS": str(size["intraOpThreads"]),
        "OMP_NUM_THREADS": str(size["intraOpThreads"]),
        "PROCESSOR_JOB_EXECUTION_MODE": topology["jobExecutionMode"],
        "PROCESSOR_EXECUTION_DOMAIN_MODE": topology["executionDomainMode"],
    }
    if any(
        env_map[name]["value"] != value for name, value in expected_environment.items()
    ):
        raise ValueError("patch environment differs from the catalog size/topology")
    if topology_name == "thread" and (
        container["image"] != catalog["observedBaseline"]["image"]
        or revision != catalog["observedBaseline"]["sourceRevision"]
    ):
        raise ValueError("thread patch must use the exact audited baseline identity")
    if (
        topology_name == "process"
        and container["image"] == catalog["observedBaseline"]["image"]
    ):
        raise ValueError("process patch cannot use the thread-only baseline image")
    resources = container["resources"]
    _require_exact_keys(resources, {"requests", "limits"}, "resources")
    for key in ("requests", "limits"):
        _require_exact_keys(resources[key], {"cpu", "memory"}, f"resources {key}")
    if resources["requests"] != resources["limits"]:
        raise ValueError("CPU benchmark requests and limits must be equal")
    if resources["limits"] != {"cpu": size["cpu"], "memory": size["memory"]}:
        raise ValueError("patch resources differ from the catalog size")
    return True


def _container(document):
    containers = document["spec"]["template"]["spec"]["containers"]
    matches = [item for item in containers if item.get("name") == CONTAINER]
    if len(matches) != 1:
        raise ValueError("expected exactly one processor container")
    return matches[0]


def _env_map(container):
    return {item["name"]: item for item in container.get("env") or []}


def validate_live_and_rendered(
    catalog, patch, live, rendered, working_pods, expected_current_image
):
    validate_patch_document(catalog, patch)
    baseline = catalog["observedBaseline"]
    if live["metadata"].get("uid") != baseline["deploymentUid"]:
        raise ValueError("live Deployment UID differs from the audited target")
    if live["metadata"].get("namespace") != NAMESPACE:
        raise ValueError("live Deployment namespace differs from video-proc")
    if live["metadata"].get("name") != DEPLOYMENT:
        raise ValueError("live Deployment name differs from CPU pool")
    if live["spec"]["template"]["spec"].get("nodeSelector") != catalog["nodeSelector"]:
        raise ValueError("live CPU pool placement drifted from the audited selector")
    if working_pods:
        raise ValueError("working video processor pods exist")
    status = live.get("status") or {}
    if live["spec"].get("replicas") != 1:
        raise ValueError("CPU Deployment must have exactly one desired replica")
    if status.get("observedGeneration") != live["metadata"].get("generation"):
        raise ValueError("CPU Deployment controller has not observed this generation")
    for field in ("replicas", "updatedReplicas", "readyReplicas", "availableReplicas"):
        if status.get(field) != 1:
            raise ValueError(f"CPU Deployment must have exactly one {field}")
    if status.get("unavailableReplicas", 0) != 0:
        raise ValueError("CPU Deployment has unavailable replicas")

    live_container = _container(live)
    rendered_container = _container(rendered)
    if live_container.get("image") != expected_current_image:
        raise ValueError("live image differs from the operator's rollback anchor")
    for item in (live_container, rendered_container):
        resource_keys = set((item.get("resources") or {}).get("requests") or {}) | set(
            (item.get("resources") or {}).get("limits") or {}
        )
        if not resource_keys <= {"cpu", "memory"}:
            raise ValueError("CPU processor has an unexpected extended resource")

    expected_image = _container(patch)["image"]
    if rendered_container.get("image") != expected_image:
        raise ValueError("server render did not retain the intended image")
    if rendered_container.get("resources") != _container(patch).get("resources"):
        raise ValueError("server render did not retain intended resources")
    expected_annotation = patch["spec"]["template"]["metadata"]["annotations"]
    rendered_annotations = (
        rendered["spec"]["template"].get("metadata", {}).get("annotations") or {}
    )
    if any(
        rendered_annotations.get(key) != value
        for key, value in expected_annotation.items()
    ):
        raise ValueError("server render did not retain the rollout annotation")
    expected_env = _env_map(_container(patch))
    rendered_env = _env_map(rendered_container)
    if set(expected_env) != EXPERIMENT_ENVS:
        raise ValueError("patch experiment env allowlist is incomplete")
    for name, value in expected_env.items():
        if rendered_env.get(name) != value:
            raise ValueError(f"server render changed experiment env {name}")

    before = copy.deepcopy(live["spec"])
    after = copy.deepcopy(rendered["spec"])
    for spec in (before, after):
        template = spec["template"]
        annotations = template.setdefault("metadata", {}).setdefault("annotations", {})
        annotations.pop(PATCH_ANNOTATION, None)
        container = _container({"spec": {"template": template}})
        container.pop("image", None)
        container.pop("resources", None)
        env = _env_map(container)
        container["env"] = [
            value for name, value in sorted(env.items()) if name not in EXPERIMENT_ENVS
        ]
    if before != after:
        raise ValueError("server render changes fields outside the CPU allowlist")
    return True


def kubectl_json(*arguments):
    command = ["kubectl", "--context", CONTEXT, "-n", NAMESPACE, *arguments]
    return json.loads(subprocess.run(command, check=True, capture_output=True).stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patch", type=Path)
    parser.add_argument("--expected-current-image", required=True)
    parser.add_argument(
        "--catalog", type=Path, default=Path(__file__).with_name("size_classes.json")
    )
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    patch = json.loads(args.patch.read_text())
    live = kubectl_json("get", "deployment", DEPLOYMENT, "-o", "json")
    rendered = kubectl_json(
        "patch",
        "deployment",
        DEPLOYMENT,
        "--type=strategic",
        "--dry-run=server",
        "-o",
        "json",
        f"--patch-file={args.patch.resolve()}",
    )
    working = (
        kubectl_json(
            "get",
            "pods",
            "-l",
            "app=video-processor,pool=working,processor-tier=cpu",
            "-o",
            "json",
        ).get("items")
        or []
    )
    validate_live_and_rendered(
        catalog,
        patch,
        live,
        rendered,
        working,
        args.expected_current_image,
    )
    encoded_live = json.dumps(live, sort_keys=True).encode()
    encoded_rendered = json.dumps(rendered, sort_keys=True).encode()
    print(
        json.dumps(
            {
                "valid": True,
                "environment": "staging",
                "context": CONTEXT,
                "deploymentUid": live["metadata"]["uid"],
                "resourceVersion": live["metadata"].get("resourceVersion"),
                "liveSha256": hashlib.sha256(encoded_live).hexdigest(),
                "renderedSha256": hashlib.sha256(encoded_rendered).hexdigest(),
                "patchSha256": hashlib.sha256(args.patch.read_bytes()).hexdigest(),
                "liveImage": _container(live).get("image"),
                "renderedImage": _container(rendered).get("image"),
                "renderedResources": _container(rendered).get("resources"),
                "nodeSelector": rendered["spec"]["template"]["spec"].get(
                    "nodeSelector"
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
