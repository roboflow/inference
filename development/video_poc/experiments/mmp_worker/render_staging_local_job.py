#!/usr/bin/env python3
"""Render a bounded, standalone staging Pod for a local video workflow job."""

import argparse
import json
import re


IMAGE_RE = re.compile(
    r"^us-central1-docker\.pkg\.dev/roboflow-staging/video-proc/"
    r"video-processor-mmp@sha256:[0-9a-f]{64}$"
)
DNS_LABEL_RE = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")


def dns_label(value, field, max_length=63):
    value = str(value or "")
    if len(value) > max_length or not DNS_LABEL_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase Kubernetes DNS label")
    return value


def workspace_name(value):
    value = str(value or "").strip()
    if not value or len(value) > 128 or any(ord(char) < 32 for char in value):
        raise ValueError("workspace must be a non-empty staging workspace name")
    return value


def _workflow():
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v1",
                "name": "detection",
                "images": "$inputs.image",
                "model_id": "microsoft-coco-obj-det/8",
            },
            {
                "type": "roboflow_core/bounding_box_visualization@v1",
                "name": "visualization",
                "image": "$inputs.image",
                "predictions": "$steps.detection.predictions",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.detection.predictions",
            },
            {
                "type": "JsonField",
                "name": "visualization",
                "coordinates_system": "own",
                "selector": "$steps.visualization.image",
            },
        ],
    }


def render(image, run_id, workspace, api_key_secret, backend="subprocess"):
    if not IMAGE_RE.fullmatch(str(image or "")):
        raise ValueError("image must be an immutable staging MMP worker digest")
    run_id = dns_label(run_id, "run ID", max_length=40)
    workspace = workspace_name(workspace)
    api_key_secret = dns_label(api_key_secret, "API key Secret")
    if backend not in {"subprocess", "direct"}:
        raise ValueError("backend must be subprocess or direct")
    manager_mode = f"mmp-bundled-{backend}"
    job_config_name = f"video-mmp-{run_id}-job"
    job = {
        "id": f"local-mmp-{run_id}",
        "workspace": workspace,
        "sourceUrl": (
            "https://media.roboflow.com/supervision/"
            "video-examples/vehicles.mp4"
        ),
        "mode": "batch",
        "imageOutput": "visualization",
        "workflowSpecification": _workflow(),
    }
    return {
        "apiVersion": "v1",
        "kind": "List",
        "items": [
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {
                    "name": job_config_name,
                    "namespace": "video-proc",
                    "labels": {
                        "roboflow.com/environment": "staging",
                        "roboflow.com/experiment": "mmp-video-worker",
                        "roboflow.com/run-id": run_id,
                    },
                },
                "data": {"job.json": json.dumps(job, sort_keys=True)},
            },
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {
                    "name": f"video-mmp-{run_id}",
                    "namespace": "video-proc",
                    "labels": {
                        "app.kubernetes.io/name": "video-mmp-local-job",
                        "roboflow.com/environment": "staging",
                        "roboflow.com/experiment": "mmp-video-worker",
                        "roboflow.com/run-id": run_id,
                    },
                },
                "spec": {
                    "activeDeadlineSeconds": 900,
                    "automountServiceAccountToken": False,
                    "imagePullSecrets": [{"name": "gcp-ar-pull-secret"}],
                    "nodeSelector": {"gpu_type": "L40S"},
                    "restartPolicy": "Never",
                    "terminationGracePeriodSeconds": 45,
                    "tolerations": [
                        {"effect": "NoSchedule", "key": "gpu", "value": "true"}
                    ],
                    "containers": [
                        {
                            "name": "worker",
                            "image": image,
                            "imagePullPolicy": "IfNotPresent",
                            "args": [
                                "--job-file",
                                "/var/run/video-job/job.json",
                                "--max-jobs",
                                "1",
                                "--tier",
                                "gpu",
                            ],
                            "env": [
                                {"name": "PROJECT", "value": "roboflow-staging"},
                                {
                                    "name": "ROBOFLOW_ENVIRONMENT",
                                    "value": "staging",
                                },
                                {
                                    "name": "PROCESSOR_MODEL_MANAGER_MODE",
                                    "value": manager_mode,
                                },
                                {
                                    "name": "LEGACY_MMP_ADAPTER_MODE",
                                    "value": "bundled",
                                },
                                {
                                    "name": "LEGACY_MMP_ADAPTER_BUNDLED_BACKEND",
                                    "value": backend,
                                },
                                {"name": "INFERENCE_N_SLOTS", "value": "8"},
                                {"name": "INFERENCE_INPUT_MB", "value": "12"},
                                {"name": "INFERENCE_BATCH_MAX_SIZE", "value": "8"},
                                {
                                    "name": "INFERENCE_BATCH_MAX_WAIT_MS",
                                    "value": "5",
                                },
                                {
                                    "name": "MODEL_CACHE_DIR",
                                    "value": "/tmp/model-cache",
                                },
                                {
                                    "name": "ROBOFLOW_API_KEY",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "key": "api-key",
                                            "name": api_key_secret,
                                        }
                                    },
                                },
                            ],
                            "ports": [{"containerPort": 8890, "name": "http"}],
                            "resources": {
                                "requests": {
                                    "cpu": "4",
                                    "memory": "16Gi",
                                    "nvidia.com/gpu": "1",
                                },
                                "limits": {
                                    "cpu": "8",
                                    "memory": "32Gi",
                                    "nvidia.com/gpu": "1",
                                },
                            },
                            "volumeMounts": [
                                {"mountPath": "/dev/shm", "name": "dshm"},
                                {
                                    "mountPath": "/var/run/video-job",
                                    "name": "job",
                                    "readOnly": True,
                                },
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"},
                            "name": "dshm",
                        },
                        {
                            "name": "job",
                            "configMap": {
                                "defaultMode": 0o400,
                                "items": [
                                    {"key": "job.json", "path": "job.json"}
                                ],
                                "name": job_config_name,
                            },
                        },
                    ],
                },
            },
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--api-key-secret", required=True)
    parser.add_argument(
        "--backend", choices=("subprocess", "direct"), default="subprocess"
    )
    args = parser.parse_args(argv)
    try:
        manifest = render(
            args.image,
            args.run_id,
            args.workspace,
            args.api_key_secret,
            args.backend,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
