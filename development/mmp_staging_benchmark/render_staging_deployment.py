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
RUN_ID = re.compile(r"^[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?$")


def render(
    image,
    source_revision,
    *,
    run_id,
    mps=False,
    decoder="imagecodecs",
    active_thread_percentage=None,
    node_name=None,
):
    if not STAGING_IMAGE.fullmatch(str(image or "")):
        raise ValueError(
            "image must be the staging mmp-benchmark repository at an exact digest"
        )
    if not re.fullmatch(r"[0-9a-f]{40}", str(source_revision or "")):
        raise ValueError("source revision must be an exact 40-character git SHA")
    if not RUN_ID.fullmatch(str(run_id or "")):
        raise ValueError("run ID must be a Kubernetes DNS label of at most 63 chars")
    if decoder not in {"imagecodecs", "nvjpeg"}:
        raise ValueError("decoder must be imagecodecs or nvjpeg")
    if active_thread_percentage is not None:
        if not mps:
            raise ValueError("active thread percentage requires MPS")
        if not 1 <= int(active_thread_percentage) <= 100:
            raise ValueError("active thread percentage must be in [1, 100]")
    if node_name is not None and not RUN_ID.fullmatch(str(node_name)):
        raise ValueError("node name must be a Kubernetes DNS label")

    env = [
        {"name": "API_BASE_URL", "value": "https://api.roboflow.one"},
        {"name": "NUM_WORKERS", "value": "4"},
        {"name": "INFERENCE_N_SLOTS", "value": "128"},
        {"name": "INFERENCE_INPUT_MB", "value": "20"},
        {"name": "INFERENCE_BATCH_MAX_SIZE", "value": "0"},
        {"name": "INFERENCE_BATCH_MAX_WAIT_MS", "value": "5"},
        {"name": "INFERENCE_DECODER", "value": decoder},
        {"name": "NVIDIA_MPS", "value": "1" if mps else "0"},
        {"name": "CUDA_MPS_PIPE_DIRECTORY", "value": "/var/run/nvidia-mps"},
        {"name": "CUDA_MPS_LOG_DIRECTORY", "value": "/var/log/nvidia-mps"},
        {"name": "MMP_BENCHMARK_IMAGE_REF", "value": image},
        {"name": "MMP_BENCHMARK_RUN_ID", "value": run_id},
        {
            "name": "MMP_BENCHMARK_MODE",
            "value": "mps" if mps else "non-mps",
        },
        {
            "name": "POD_NAME",
            "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}},
        },
        {
            "name": "NODE_NAME",
            "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}},
        },
    ]
    if active_thread_percentage is not None:
        env.append(
            {
                "name": "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
                "value": str(active_thread_percentage),
            }
        )
    labels = {
        "app": NAME,
        "app.kubernetes.io/managed-by": "video-poc-benchmark",
    }
    annotations = {
        "roboflow.com/environment": "staging-only",
        "roboflow.com/source-revision": source_revision,
        "roboflow.com/mps": "enabled" if mps else "disabled",
        "roboflow.com/run-id": run_id,
        "roboflow.com/decoder": decoder,
    }
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": NAME,
            "namespace": NAMESPACE,
            "labels": labels,
            "annotations": annotations,
        },
        "spec": {
            "replicas": 1,
            "strategy": {"type": "Recreate"},
            "selector": {"matchLabels": {"app": NAME}},
            "template": {
                "metadata": {
                    "labels": labels,
                    "annotations": annotations,
                },
                "spec": {
                    "automountServiceAccountToken": False,
                    "nodeSelector": {
                        "gpu_type": "L40S",
                        **({"kubernetes.io/hostname": node_name} if node_name else {}),
                    },
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
                                "capabilities": {"drop": ["ALL"]},
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
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mps", action="store_true")
    parser.add_argument(
        "--decoder", choices=("imagecodecs", "nvjpeg"), default="imagecodecs"
    )
    parser.add_argument("--active-thread-percentage", type=int)
    parser.add_argument("--node-name")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    document = render(
        args.image,
        args.source_revision,
        run_id=args.run_id,
        mps=args.mps,
        decoder=args.decoder,
        active_thread_percentage=args.active_thread_percentage,
        node_name=args.node_name,
    )
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
