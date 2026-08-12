#!/usr/bin/env python3
"""Render a digest-pinned, dedicated staging MMP experiment Deployment."""

import argparse
import json
import re
from pathlib import Path


STAGING_IMAGE = re.compile(
    r"^us-central1-docker\.pkg\.dev/roboflow-staging/video-proc/"
    r"mmp-benchmark@sha256:[0-9a-f]{64}$"
)
NAMESPACE = "video-proc-bench-mmp"
NAME = "mmp-benchmark-server"


def render(image, source_revision, api_key_secret, mps=False):
    if not STAGING_IMAGE.fullmatch(str(image or "")):
        raise ValueError(
            "image must be the staging mmp-benchmark repository at an exact digest"
        )
    if not re.fullmatch(r"[0-9a-f]{40}", str(source_revision or "")):
        raise ValueError("source revision must be an exact 40-character git SHA")
    if not re.fullmatch(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?", api_key_secret):
        raise ValueError("API key Secret name must be a Kubernetes DNS label")

    env = [
        {"name": "API_BASE_URL", "value": "https://api.roboflow.one"},
        {"name": "NUM_WORKERS", "value": "4"},
        {"name": "INFERENCE_N_SLOTS", "value": "128"},
        {"name": "INFERENCE_INPUT_MB", "value": "20"},
        {"name": "INFERENCE_BATCH_MAX_SIZE", "value": "0"},
        {"name": "INFERENCE_BATCH_MAX_WAIT_MS", "value": "5"},
        {"name": "NVIDIA_MPS", "value": "1" if mps else "0"},
        {"name": "CUDA_MPS_PIPE_DIRECTORY", "value": "/var/run/nvidia-mps"},
        {"name": "CUDA_MPS_LOG_DIRECTORY", "value": "/var/log/nvidia-mps"},
        {"name": "MMP_BENCHMARK_IMAGE_REF", "value": image},
        {
            "name": "RF_BENCH_TENANT_A_KEY",
            "valueFrom": {
                "secretKeyRef": {"name": api_key_secret, "key": "tenant-a"}
            },
        },
        {
            "name": "RF_BENCH_TENANT_B_KEY",
            "valueFrom": {
                "secretKeyRef": {"name": api_key_secret, "key": "tenant-b"}
            },
        },
    ]
    labels = {
        "app": NAME,
        "app.kubernetes.io/managed-by": "video-poc-benchmark",
    }
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": NAME,
            "namespace": NAMESPACE,
            "labels": labels,
            "annotations": {
                "roboflow.com/environment": "staging-only",
                "roboflow.com/source-revision": source_revision,
                "roboflow.com/mps": "enabled" if mps else "disabled",
            },
        },
        "spec": {
            "replicas": 1,
            "strategy": {"type": "Recreate"},
            "selector": {"matchLabels": {"app": NAME}},
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "nodeSelector": {"gpu_type": "L40S"},
                    "tolerations": [
                        {"key": "gpu", "value": "true", "effect": "NoSchedule"}
                    ],
                    "terminationGracePeriodSeconds": 60,
                    "containers": [
                        {
                            "name": "server",
                            "image": image,
                            "imagePullPolicy": "IfNotPresent",
                            "env": env,
                            "ports": [{"name": "http", "containerPort": 8000}],
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/v2/server/health",
                                    "port": "http",
                                },
                                "initialDelaySeconds": 20,
                                "periodSeconds": 10,
                                "timeoutSeconds": 3,
                                "failureThreshold": 12,
                            },
                            "resources": {
                                "requests": {
                                    "nvidia.com/gpu": 1,
                                    "cpu": "4",
                                    "memory": "16Gi",
                                },
                                "limits": {
                                    "nvidia.com/gpu": 1,
                                    "cpu": "8",
                                    "memory": "32Gi",
                                },
                            },
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                            },
                            "volumeMounts": [
                                {"name": "dshm", "mountPath": "/dev/shm"},
                                {"name": "results", "mountPath": "/results"},
                                {"name": "model-cache", "mountPath": "/models/cache"},
                                {
                                    "name": "mps-pipe",
                                    "mountPath": "/var/run/nvidia-mps",
                                },
                                {"name": "mps-log", "mountPath": "/var/log/nvidia-mps"},
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "dshm",
                            "emptyDir": {"medium": "Memory", "sizeLimit": "4Gi"},
                        },
                        {"name": "results", "emptyDir": {}},
                        {"name": "model-cache", "emptyDir": {}},
                        {"name": "mps-pipe", "emptyDir": {"medium": "Memory"}},
                        {"name": "mps-log", "emptyDir": {}},
                    ],
                },
            },
        },
    }
    return {
        "apiVersion": "v1",
        "kind": "List",
        "items": [
            {
                "apiVersion": "v1",
                "kind": "Namespace",
                "metadata": {
                    "name": NAMESPACE,
                    "labels": {
                        "roboflow.com/environment": "staging-only",
                        "app.kubernetes.io/managed-by": "video-poc-benchmark",
                    },
                },
            },
            deployment,
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--api-key-secret", default="mmp-benchmark-api-keys")
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    document = render(
        args.image, args.source_revision, args.api_key_secret, mps=args.mps
    )
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
