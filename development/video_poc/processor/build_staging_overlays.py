"""Build reproducible staging video-processor overlay images with Cloud Build."""

import argparse
import json
import re
import subprocess
from pathlib import Path

PROJECT = "roboflow-staging"
REGISTRY_PREFIX = "us-central1-docker.pkg.dev/roboflow-staging/video-proc/"
GPU_BASE_REPOSITORY = f"{REGISTRY_PREFIX}video-processor"
CPU_BASE_REPOSITORY = f"{REGISTRY_PREFIX}video-processor-cpu"
GPU_OUTPUT_REPOSITORY = f"{REGISTRY_PREFIX}video-processor-telemetry"
CPU_OUTPUT_REPOSITORY = f"{REGISTRY_PREFIX}video-processor-cpu-telemetry"
SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")
OVERLAY_FILES = (
    "development/video_poc/processor/Dockerfile.overlay",
    "development/video_poc/processor/cloudbuild.overlay.yaml",
    "development/video_poc/processor/processor.py",
    "development/video_poc/processor/processor_metrics.py",
    "development/video_poc/processor/job_telemetry.py",
    "development/video_poc/processor/job_process.py",
    "development/video_poc/processor/claim_proof.py",
    "development/video_poc/processor/inference_runtime_compat.py",
    "development/video_poc/processor/run_lifecycle.py",
    "development/video_poc/processor/worker_lifecycle.py",
    "development/video_poc/processor/execution_domains.py",
    "development/video_poc/processor/security.py",
    "development/video_poc/processor/video_ingest.py",
    "development/video_poc/processor/file_replay.py",
    "development/video_poc/processor/low_latency_producer.py",
)


def validate_digest_ref(value: str, repository: str) -> str:
    prefix = f"{repository}@"
    if not value.startswith(prefix):
        raise ValueError(f"image must use exact staging repository {repository}")
    digest = value[len(prefix) :]
    if not SHA256_RE.fullmatch(digest):
        raise ValueError("base image must use an immutable sha256 digest")
    return value


def output_ref(repository: str, git_sha: str) -> str:
    if not GIT_SHA_RE.fullmatch(git_sha):
        raise ValueError("git SHA must be exactly 40 lowercase hexadecimal characters")
    return f"{repository}:{git_sha}"


def build_plan(gpu_base: str, cpu_base: str, git_sha: str) -> dict:
    validate_digest_ref(gpu_base, GPU_BASE_REPOSITORY)
    validate_digest_ref(cpu_base, CPU_BASE_REPOSITORY)
    gpu_output = output_ref(GPU_OUTPUT_REPOSITORY, git_sha)
    cpu_output = output_ref(CPU_OUTPUT_REPOSITORY, git_sha)
    processor_dir = Path(__file__).resolve().parent
    config = processor_dir / "cloudbuild.overlay.yaml"
    substitutions = ",".join(
        (
            f"_GPU_BASE={gpu_base}",
            f"_CPU_BASE={cpu_base}",
            f"_GPU_OUTPUT={gpu_output}",
            f"_CPU_OUTPUT={cpu_output}",
            f"_GIT_SHA={git_sha}",
        )
    )
    return {
        "schemaVersion": 1,
        "environment": "staging",
        "project": PROJECT,
        "gitSha": git_sha,
        "sourceFiles": list(OVERLAY_FILES),
        "inputs": {"gpu": gpu_base, "cpu": cpu_base},
        "outputs": {"gpu": gpu_output, "cpu": cpu_output},
        "submitCommand": [
            "gcloud",
            "builds",
            "submit",
            str(processor_dir),
            f"--project={PROJECT}",
            f"--config={config}",
            f"--substitutions={substitutions}",
        ],
    }


def verify_source_revision(plan: dict) -> None:
    repository = Path(__file__).resolve().parents[3]
    revision = plan["gitSha"]
    subprocess.run(
        ["git", "-C", str(repository), "cat-file", "-e", f"{revision}^{{commit}}"],
        check=True,
    )
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "diff",
            "--quiet",
            revision,
            "--",
            *plan["sourceFiles"],
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(
            "overlay source files differ from --git-sha; commit them or use "
            "the exact source revision"
        )


def resolve_digest(image: str) -> str:
    command = [
        "gcloud",
        "artifacts",
        "docker",
        "images",
        "describe",
        image,
        f"--project={PROJECT}",
        "--format=value(image_summary.digest)",
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    digest = result.stdout.strip()
    if not SHA256_RE.fullmatch(digest):
        raise RuntimeError(f"registry returned invalid digest for {image}")
    return f"{image.split(':', 1)[0]}@{digest}"


def execute_plan(plan: dict) -> dict:
    subprocess.run(
        [
            "gcloud",
            "projects",
            "describe",
            PROJECT,
            "--format=value(projectId)",
        ],
        check=True,
    )
    subprocess.run(plan["submitCommand"], check=True)
    immutable_outputs = {
        tier: resolve_digest(image) for tier, image in plan["outputs"].items()
    }
    return {**plan, "immutableOutputs": immutable_outputs}


def write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-base", required=True)
    parser.add_argument("--cpu-base", required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="submit the Cloud Build; otherwise print a dry build plan",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_plan(args.gpu_base, args.cpu_base, args.git_sha)
    verify_source_revision(plan)
    result = execute_plan(plan) if args.execute else plan
    if args.result_json:
        write_result(args.result_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
