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


def render(
    image,
    run_id,
    workspace,
    api_key_secret,
    backend="subprocess",
    concurrency=1,
    max_fps=5.0,
    mode="batch",
    input_height=None,
):
    if not IMAGE_RE.fullmatch(str(image or "")):
        raise ValueError("image must be an immutable staging MMP worker digest")
    run_id = dns_label(run_id, "run ID", max_length=40)
    workspace = workspace_name(workspace)
    api_key_secret = dns_label(api_key_secret, "API key Secret")
    if backend not in {"legacy", "subprocess", "direct"}:
        raise ValueError("backend must be legacy, subprocess, or direct")
    if not isinstance(concurrency, int) or not 1 <= concurrency <= 8:
        raise ValueError("concurrency must be an integer between 1 and 8")
    if not isinstance(max_fps, (int, float)) or not 0 < max_fps <= 30:
        raise ValueError("max FPS must be greater than 0 and at most 30")
    if mode not in {"batch", "stream"}:
        raise ValueError("mode must be batch or stream")
    if input_height is not None:
        if mode != "stream":
            raise ValueError("input height is supported only in stream mode")
        if not isinstance(input_height, int) or not 64 <= input_height <= 2160:
            raise ValueError("input height must be an integer from 64 to 2160")
    manager_mode = (
        "legacy" if backend == "legacy" else f"mmp-bundled-{backend}"
    )
    job_config_name = f"video-mmp-{run_id}-job"
    jobs = []
    for ordinal in range(1, concurrency + 1):
        jobs.append(
            {
                "id": f"local-{run_id}-{ordinal}",
                "workspace": workspace,
                "sourceUrl": (
                    "rtsp://127.0.0.1:8554/fixture-scaled"
                    if input_height is not None
                    else (
                        "https://media.roboflow.com/supervision/"
                        "video-examples/vehicles.mp4"
                    )
                ),
                "mode": mode,
                "maxFps": float(max_fps),
                "imageOutput": "visualization",
                "workflowSpecification": _workflow(),
                **(
                    {
                        "simPublishUrl": (
                            "rtsp://127.0.0.1:8554/"
                            f"sim-{run_id}-{ordinal}"
                        )
                    }
                    if mode == "stream" and input_height is None
                    else {}
                ),
            }
        )
    job_data = {
        f"job-{ordinal}.json": json.dumps(job, sort_keys=True)
        for ordinal, job in enumerate(jobs, start=1)
    }
    job_args = []
    for ordinal in range(1, concurrency + 1):
        job_args.extend(
            ["--job-file", f"/var/run/video-job/job-{ordinal}.json"]
        )
    job_args.extend(["--max-jobs", str(concurrency), "--tier", "gpu"])
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
                "data": {
                    **job_data,
                    "mediamtx.yml": (
                        "logLevel: warn\n"
                        "rtsp: yes\n"
                        "rtspAddress: :8554\n"
                        "rtspTransports: [tcp]\n"
                        "webrtc: no\n"
                        "hls: no\n"
                        "rtmp: no\n"
                        "srt: no\n"
                        "paths:\n  all_others:\n"
                    ),
                },
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
                            **(
                                {
                                    "command": ["/bin/sh", "-c"],
                                    "args": [
                                        "sleep 8; exec python /app/processor.py \"$@\"",
                                        "processor",
                                        *job_args,
                                    ],
                                }
                                if input_height is not None
                                else {"args": job_args}
                            ),
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
                                    "value": (
                                        "off" if backend == "legacy" else "bundled"
                                    ),
                                },
                                {
                                    "name": "LEGACY_MMP_ADAPTER_BUNDLED_BACKEND",
                                    "value": (
                                        "subprocess"
                                        if backend == "legacy"
                                        else backend
                                    ),
                                },
                                {"name": "INFERENCE_N_SLOTS", "value": "8"},
                                {"name": "INFERENCE_INPUT_MB", "value": "32"},
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
                        },
                        {
                            "name": "relay",
                            "image": (
                                "docker.io/bluenviron/mediamtx@sha256:"
                                "59aaad04627c7c8f40ceb01a5ff1c43f91e01939da147c3419f1aaa0c78d6cf5"
                            ),
                            "args": ["/etc/mediamtx/mediamtx.yml"],
                            "ports": [
                                {"containerPort": 8554, "name": "local-rtsp"}
                            ],
                            "resources": {
                                "requests": {"cpu": "100m", "memory": "64Mi"},
                                "limits": {"cpu": "500m", "memory": "256Mi"},
                            },
                            "volumeMounts": [
                                {
                                    "mountPath": "/etc/mediamtx",
                                    "name": "relay-config",
                                    "readOnly": True,
                                }
                            ],
                        },
                        *(
                            [
                                {
                                    "name": "fixture-scaler",
                                    "image": image,
                                    "imagePullPolicy": "IfNotPresent",
                                    "command": ["/bin/sh", "-c"],
                                    "args": [
                                        "while true; do "
                                        "ffmpeg -hide_banner -loglevel warning "
                                        "-re -stream_loop -1 -i "
                                        "https://media.roboflow.com/supervision/"
                                        "video-examples/vehicles.mp4 "
                                        f"-vf scale=-2:{input_height} -an "
                                        "-c:v mpeg4 -q:v 5 -pix_fmt yuv420p "
                                        "-g 30 "
                                        "-f rtsp -rtsp_transport tcp "
                                        "rtsp://127.0.0.1:8554/fixture-scaled; "
                                        "sleep 1; done"
                                    ],
                                    "resources": {
                                        "requests": {
                                            "cpu": "1",
                                            "memory": "512Mi",
                                        },
                                        "limits": {
                                            "cpu": "4",
                                            "memory": "2Gi",
                                        },
                                    },
                                }
                            ]
                            if input_height is not None
                            else []
                        ),
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
                                    {
                                        "key": f"job-{ordinal}.json",
                                        "path": f"job-{ordinal}.json",
                                    }
                                    for ordinal in range(1, concurrency + 1)
                                ],
                                "name": job_config_name,
                            },
                        },
                        {
                            "name": "relay-config",
                            "configMap": {
                                "defaultMode": 0o400,
                                "items": [
                                    {
                                        "key": "mediamtx.yml",
                                        "path": "mediamtx.yml",
                                    }
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
        "--backend",
        choices=("legacy", "subprocess", "direct"),
        default="subprocess",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-fps", type=float, default=5.0)
    parser.add_argument("--mode", choices=("batch", "stream"), default="batch")
    parser.add_argument("--input-height", type=int)
    args = parser.parse_args(argv)
    try:
        manifest = render(
            args.image,
            args.run_id,
            args.workspace,
            args.api_key_secret,
            args.backend,
            args.concurrency,
            args.max_fps,
            args.mode,
            args.input_height,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
