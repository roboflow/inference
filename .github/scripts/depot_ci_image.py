"""Opt-in fast-track Depot transport; never print commands, metadata or errors."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from ci_image import _digest, _emit, _manifest_digests


def _run(args: list[str], *, env: dict | None = None, stdin: str | None = None) -> str:
    result = subprocess.run(
        args, env=env, input=stdin, capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise ValueError("Private operation failed")

    return result.stdout


def _authorize() -> None:
    ref = os.environ.get("GITHUB_REF", "")
    sha = os.environ.get("GITHUB_SHA", "")
    workflows = {
        f"roboflow/inference/.github/workflows/{name}@{ref}"
        for name in ("fast_track_depot_images.yml", "fast_track_integration_tests.yml")
    }
    if (
        os.environ.get("GITHUB_REPOSITORY") != "roboflow/inference"
        or os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
        or os.environ.get("CI_DEPOT_REUSE_ENABLED") != "true"
        or not re.fullmatch(
            r"refs/heads/fast-track/post-v\d+\.\d+\.\d+-pt[1-9]\d*", ref
        )
        or ref != os.environ.get("CI_DEPOT_APPROVED_REF")
        or not re.fullmatch(r"[0-9a-f]{40}", sha)
        or sha != os.environ.get("CI_DEPOT_APPROVED_SHA")
        or os.environ.get("GITHUB_WORKFLOW_REF") not in workflows
    ):
        raise ValueError("Unapproved execution")

    if _run(["git", "rev-parse", "HEAD"]).strip() != sha:
        raise ValueError("Checkout differs from approval")


def _repository() -> str:
    # A project-scoped Depot repository, supplied through a secret. No arbitrary
    # registry host, nested path, mutable tag or public-registry fallback.
    repository = os.environ.get("CI_DEPOT_REGISTRY", "")
    if not re.fullmatch(
        r"(?:[a-z0-9][a-z0-9-]*\.registry\.depot\.dev|registry\.depot\.dev)/[a-z0-9]+",
        repository,
    ):
        raise ValueError("Invalid Depot repository")

    return repository


def _validate() -> None:
    _authorize()
    _repository()
    if os.environ.get("IMAGE_FAMILY") not in {"cpu", "gpu"}:
        raise ValueError("Invalid image family")


def _login(*, env: dict, token: str) -> None:
    if not token or "\n" in token or "\r" in token:
        raise ValueError("Missing or invalid credential")

    _run(
        [
            "docker",
            "login",
            _repository().split("/", 1)[0],
            "--username",
            "x-token",
            "--password-stdin",
        ],
        env=env,
        stdin=token,
    )


def _build() -> None:
    _validate()
    repository = _repository()
    project = os.environ.get("DEPOT_PROJECT_ID", "")
    token = os.environ.get("DEPOT_TOKEN", "")
    if project != repository.rsplit("/", 1)[1] or not token:
        raise ValueError("Configure the matching builder project and token")

    family = os.environ["IMAGE_FAMILY"]
    sha = os.environ["GITHUB_SHA"]
    # Buildx is installed before this helper. Keep its records for its post hook.
    config = Path(os.environ["DOCKER_CONFIG"]) / "config.json"
    try:
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
                    "--save",
                    "--progress=plain",
                    "--platform",
                    "linux/amd64,linux/arm64" if family == "cpu" else "linux/amd64",
                    "--provenance=mode=min",
                    "--metadata-file",
                    str(metadata),
                    "--label",
                    f"org.opencontainers.image.revision={sha}",
                    "--label",
                    f"io.roboflow.ci.image-family={family}",
                    "--file",
                    str(context / f"docker/dockerfiles/Dockerfile.onnx.{family}"),
                    str(context),
                ]
            )
            result = json.loads(metadata.read_text())
            saved = result.get("depot.build", {})
            if saved.get("projectID") != project or not saved.get("buildID"):
                raise ValueError("Missing saved build identity")

            index = _digest(result["containerimage.digest"])
            _login(env=dict(os.environ), token=token)
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
                raise ValueError("CPU requires both architectures")

            _emit("GITHUB_OUTPUT", "index_digest", index)
            for architecture, digest in children.items():
                _emit("GITHUB_OUTPUT", f"{architecture}_digest", digest)
    finally:
        config.unlink(missing_ok=True)


def _pull() -> None:
    _validate()
    if os.environ.get("MODEL_MODE") not in {"true", "false"}:
        raise ValueError("Invalid model mode")

    digest = _digest(os.environ.get("IMAGE_DIGEST", ""))
    reference = f"{_repository()}@{digest}"
    # The pull credential never enters GITHUB_ENV or a test-step environment.
    # No Depot CLI/API token or OIDC permission is needed by consumers.
    with TemporaryDirectory(dir=os.environ["RUNNER_TEMP"]) as directory:
        env = dict(os.environ, DOCKER_CONFIG=directory)
        _login(env=env, token=os.environ.get("CI_DEPOT_PULL_TOKEN", ""))
        _run(["docker", "pull", "--platform", "linux/amd64", reference], env=env)
        info = json.loads(_run(["docker", "image", "inspect", reference], env=env))[0]
        labels = info.get("Config", {}).get("Labels") or {}
        if (
            info.get("Os") != "linux"
            or info.get("Architecture") != "amd64"
            or reference not in info.get("RepoDigests", [])
            or labels.get("org.opencontainers.image.revision")
            != os.environ["GITHUB_SHA"]
            or labels.get("io.roboflow.ci.image-family") != os.environ["IMAGE_FAMILY"]
        ):
            raise ValueError("Image identity mismatch")

        image_id = _digest(info["Id"])

    _emit("GITHUB_ENV", "INFERENCE_TEST_IMAGE", image_id)


if __name__ == "__main__":
    try:
        {
            "authorize": _authorize,
            "validate": _validate,
            "build": _build,
            "pull": _pull,
        }[os.environ["CI_DEPOT_ACTION"]]()
    except Exception:
        # Even malformed metadata, login errors and unexpected exceptions must
        # not reveal private resource names or secret-bearing subprocess output.
        print("::error::Depot image operation failed; private diagnostics withheld")
        raise SystemExit(1) from None
