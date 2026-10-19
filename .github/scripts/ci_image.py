"""Private CI image transport. All configuration is supplied through environment.

Never print subprocess output: registry errors and build logs can contain private
locations. GitHub outputs contain only digests; detailed metadata stays private.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


def _run(args: list[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise ValueError("CI image operation failed; private output withheld")

    return result.stdout


def _digest(value: str) -> str:
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise ValueError("Invalid image digest")

    return value


def _emit(path_variable: str, name: str, value: str) -> None:
    if "\n" in value or "\r" in value:
        raise ValueError("Invalid output")

    with Path(os.environ[path_variable]).open("a") as handle:
        handle.write(f"{name}={value}\n")


def _trusted() -> None:
    if (
        os.environ.get("GITHUB_REPOSITORY") != "roboflow/inference"
        or os.environ.get("GITHUB_REF") != "refs/heads/main"
        or os.environ.get("GITHUB_EVENT_NAME") not in {"push", "workflow_dispatch"}
        or os.environ.get("CI_IMAGE_REUSE_ENABLED") != "true"
    ):
        raise ValueError("Private image transport requires enabled, trusted main")

    sha = os.environ.get("GITHUB_SHA", "")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Invalid candidate SHA")

    if _run(["git", "rev-parse", "HEAD"]).strip() != sha:
        raise ValueError("Checkout differs from candidate SHA")


def _repository() -> str:
    prefix = os.environ.get("CI_REGISTRY_PREFIX", "")
    # An approved Artifact Registry namespace, supplied as a secret. No public
    # release registry or caller-controlled repository fallback is permitted.
    if not re.fullmatch(r"[a-z0-9-]+-docker\.pkg\.dev/[a-z0-9-]+/[a-z0-9_-]+", prefix):
        raise ValueError("Configure an approved private registry namespace")

    family = os.environ.get("IMAGE_FAMILY", "")
    if family not in {"cpu", "gpu"}:
        raise ValueError("Invalid image family")

    repository = f"{prefix}/inference-ci-{family}"
    return repository


def _validate() -> None:
    if os.environ.get("MODEL_MODE") not in {"true", "false"}:
        raise ValueError("Invalid model mode")

    tag = os.environ.get("IMAGE_TAG", "")
    digest = os.environ.get("IMAGE_DIGEST", "")
    private = os.environ.get("PRIVATE_IMAGE") == "true"
    if tag and digest:
        raise ValueError("Specify only one of image_tag and image_digest")

    if tag and not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", tag):
        raise ValueError("Invalid image tag")

    if digest:
        _digest(digest)

    if private:
        _trusted()
        _repository()
        if not digest:
            raise ValueError("Private consumption requires a producer digest")


def _manifest_digests(manifest: dict, *, family: str) -> dict[str, str]:
    expected = {"amd64", "arm64"} if family == "cpu" else {"amd64"}
    result = {}
    for item in manifest.get("manifests", []):
        platform = item.get("platform", {})
        architecture = platform.get("architecture")
        # BuildKit provenance descriptors have unknown/unknown platform.
        if architecture == "unknown" and platform.get("os") == "unknown":
            continue
        if platform.get("os") != "linux" or architecture not in expected:
            raise ValueError("Unexpected image platform")
        if architecture in result:
            raise ValueError("Ambiguous image platform")

        result[architecture] = _digest(item["digest"])
    if set(result) != expected:
        raise ValueError("Missing required image platform")

    return result


def _preflight() -> None:
    _trusted()
    _repository()


def _build() -> None:
    _trusted()
    repository = _repository()
    family = os.environ["IMAGE_FAMILY"]
    sha = os.environ["GITHUB_SHA"]
    run_id = os.environ["GITHUB_RUN_ID"]
    attempt = os.environ["GITHUB_RUN_ATTEMPT"]
    if not run_id.isdigit() or not attempt.isdigit():
        raise ValueError("Invalid producer identity")

    if not os.environ.get("DEPOT_PROJECT_ID"):
        raise ValueError("Configure the private builder project")

    # Export tracked source only; checkout tokens, auth files, local logs and
    # runner state must never enter COPY . . or a remote builder cache.
    with TemporaryDirectory(dir=os.environ["RUNNER_TEMP"]) as directory:
        root = Path(directory)
        context = root / "source"
        context.mkdir()
        archive = root / "source.tar"
        _run(["git", "archive", "--format=tar", f"--output={archive}", sha])
        _run(["tar", "-xf", str(archive), "-C", str(context)])
        metadata = root / "metadata.json"
        _run(
            [
                "depot",
                "build",
                "--push",
                "--progress=plain",
                "--platform",
                "linux/amd64,linux/arm64" if family == "cpu" else "linux/amd64",
                "--provenance=mode=min",
                "--metadata-file",
                str(metadata),
                "--label",
                f"org.opencontainers.image.revision={sha}",
                "--tag",
                f"{repository}:ci-{sha}-{run_id}-{attempt}",
                "--file",
                str(context / f"docker/dockerfiles/Dockerfile.onnx.{family}"),
                str(context),
            ]
        )
        index = _digest(json.loads(metadata.read_text())["containerimage.digest"])
        manifest = json.loads(
            _run(
                [
                    "docker",
                    "buildx",
                    "imagetools",
                    "inspect",
                    "--raw",
                    f"{repository}@{index}",
                ]
            )
        )
        if "manifests" in manifest:
            children = _manifest_digests(manifest, family=family)
        elif family == "gpu":
            children = {"amd64": index}
        else:
            raise ValueError("CPU build must include both architectures")

        _emit("GITHUB_OUTPUT", "index_digest", index)
        for architecture, digest in children.items():
            _emit("GITHUB_OUTPUT", f"{architecture}_digest", digest)


def _pull() -> None:
    _validate()
    family = os.environ["IMAGE_FAMILY"]
    if family not in {"cpu", "gpu"}:
        raise ValueError("Invalid image family")

    private = os.environ.get("PRIVATE_IMAGE") == "true"
    repository = (
        _repository() if private else f"roboflow/roboflow-inference-server-{family}"
    )
    digest = os.environ.get("IMAGE_DIGEST", "")
    tag = os.environ.get("IMAGE_TAG", "")
    if not digest and not tag:
        raise ValueError("Prebuilt image reference required")

    reference = f"{repository}@{digest}" if digest else f"{repository}:{tag}"
    _run(["docker", "pull", "--platform", "linux/amd64", reference])
    info = json.loads(_run(["docker", "image", "inspect", reference]))[0]
    if info.get("Os") != "linux" or info.get("Architecture") != "amd64":
        raise ValueError("Image architecture mismatch")
    if digest and reference not in info.get("RepoDigests", []):
        raise ValueError("Pulled image digest mismatch")
    if (
        private
        and (info.get("Config", {}).get("Labels") or {}).get(
            "org.opencontainers.image.revision"
        )
        != os.environ["GITHUB_SHA"]
    ):
        raise ValueError("Image source differs from candidate SHA")

    # Run by immutable local image ID, never a shared mutable :test alias.
    image_id = _digest(info["Id"])
    _emit("GITHUB_ENV", "INFERENCE_TEST_IMAGE", image_id)


def _port() -> None:
    container = os.environ["INFERENCE_TEST_CONTAINER"]
    value = _run(["docker", "port", container, "9101/tcp"]).strip()
    if not re.fullmatch(r"127\.0\.0\.1:[0-9]+", value):
        raise ValueError("Unexpected test container port binding")

    _emit("GITHUB_ENV", "TEST_PORT", value.rsplit(":", 1)[1])


if __name__ == "__main__":
    try:
        {
            "preflight": _preflight,
            "validate": _validate,
            "build": _build,
            "pull": _pull,
            "port": _port,
        }[os.environ["CI_IMAGE_ACTION"]]()
    except (ValueError, KeyError, OSError, IndexError):
        # Do not render the exception, command, response or user input.
        print("::error::CI image validation or transport failed; no fallback performed")
        raise SystemExit(1) from None
