"""Public prebuilt-image consumption and shared helpers for CI image transport.

Saved CI images (private_image) are produced and pulled by depot_ci_image.py;
this module validates consumer inputs, pulls public release images by tag or
digest, and allocates the test port. Never print subprocess output.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path


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
    # Pushes to main consume saved images only while reuse is enabled; a manual
    # dispatch from any same-repository branch may exercise the path before that.
    ref = os.environ.get("GITHUB_REF", "")
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    enabled = os.environ.get("CI_IMAGE_REUSE_ENABLED") == "true"
    main_push = event == "push" and ref == "refs/heads/main" and enabled
    manual = event == "workflow_dispatch" and ref.startswith("refs/heads/")
    if os.environ.get("GITHUB_REPOSITORY") != "roboflow/inference" or not (
        main_push or manual
    ):
        raise ValueError("Saved CI images require trusted same-repository execution")

    sha = os.environ.get("GITHUB_SHA", "")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Invalid candidate SHA")

    if _run(["git", "rev-parse", "HEAD"]).strip() != sha:
        raise ValueError("Checkout differs from candidate SHA")


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


def _pull() -> None:
    _validate()
    family = os.environ["IMAGE_FAMILY"]
    if family not in {"cpu", "gpu"}:
        raise ValueError("Invalid image family")

    if os.environ.get("PRIVATE_IMAGE") == "true":
        raise ValueError("Saved CI images are pulled by the Depot transport")

    repository = f"roboflow/roboflow-inference-server-{family}"
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
            "validate": _validate,
            "pull": _pull,
            "port": _port,
        }[os.environ["CI_IMAGE_ACTION"]]()
    except (ValueError, KeyError, OSError, IndexError):
        # Do not render the exception, command, response or user input.
        print("::error::CI image validation or transport failed; no fallback performed")
        raise SystemExit(1) from None
