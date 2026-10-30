"""Depot Registry transport for CI image reuse.

Never print commands, metadata or errors: they can carry private details.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from ci_image import _digest, _emit, _manifest_digests

_ORCHESTRATOR = "roboflow/inference/.github/workflows/ci_image_reuse.yml"
# Only a push to main may write into the CI project; the release project is
# never a CI target. Manual dispatches must use a sandbox project.
_PROTECTED_PROJECTS = frozenset({"9x3fbbxlnm", "grl7ffzxd7"})


def _run(args: list[str], *, env: dict | None = None, stdin: str | None = None) -> str:
    result = subprocess.run(
        args, env=env, input=stdin, capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise ValueError("Private operation failed")

    return result.stdout


def _authorize() -> None:
    # Pushes to main use saved images; a manual dispatch of the orchestrator
    # from any same-repository branch proves the path early.
    ref = os.environ.get("GITHUB_REF", "")
    sha = os.environ.get("GITHUB_SHA", "")
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    main_push = event == "push" and ref == "refs/heads/main"
    manual = event == "workflow_dispatch" and ref.startswith("refs/heads/")
    if (
        os.environ.get("GITHUB_REPOSITORY") != "roboflow/inference"
        or not re.fullmatch(r"[0-9a-f]{40}", sha)
        or not (main_push or manual)
        or os.environ.get("GITHUB_WORKFLOW_REF") != f"{_ORCHESTRATOR}@{ref}"
    ):
        raise ValueError("Unapproved execution")

    if _run(["git", "rev-parse", "HEAD"]).strip() != sha:
        raise ValueError("Checkout differs from approval")


def _project() -> str:
    # The orchestrator passes one Depot project ID to every producer and consumer.
    project = os.environ.get("DEPOT_PROJECT_ID", "")
    if not re.fullmatch(r"[a-z0-9]+", project):
        raise ValueError("Invalid Depot project")

    if project in _PROTECTED_PROJECTS and os.environ.get("GITHUB_EVENT_NAME") != "push":
        raise ValueError("Manual runs must use a sandbox project")

    return project


def _repository() -> str:
    # Saved images live in the project-scoped Depot Registry repository, addressed
    # by digest only. No host override, nested path or mutable tag.
    repository = f"registry.depot.dev/{_project()}"
    return repository


def _validate() -> None:
    _authorize()
    _project()
    if os.environ.get("IMAGE_FAMILY") not in {"cpu", "gpu"}:
        raise ValueError("Invalid image family")


def _pull_token(project: str) -> str:
    # The CLI authenticates with GitHub OIDC through the org-level Depot
    # connection; the registry only ever sees a one-hour read-only token.
    token = _run(["depot", "pull-token", "--project", project]).strip()
    if not token or any(character.isspace() for character in token):
        raise ValueError("Missing or invalid pull token")

    return token


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
    project = _project()
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
                    # Readable name in the Depot Registry UI; consumers still
                    # verify by digest and labels, never by tag.
                    "--save-tag",
                    f"ci-{family}-{sha}",
                    "--progress=plain",
                    "--project",
                    project,
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
            _login(env=dict(os.environ), token=_pull_token(project))
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
    # The one-hour pull token lives only in a temporary DOCKER_CONFIG; it never
    # enters GITHUB_ENV or a test-step environment.
    with TemporaryDirectory(dir=os.environ["RUNNER_TEMP"]) as directory:
        env = dict(os.environ, DOCKER_CONFIG=directory)
        _login(env=env, token=_pull_token(_project()))
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
            "validate": _validate,
            "build": _build,
            "pull": _pull,
        }[os.environ["CI_DEPOT_ACTION"]]()
    except Exception:
        # Even malformed metadata, login errors and unexpected exceptions must
        # not reveal private resource names or secret-bearing subprocess output.
        print("::error::Depot image operation failed; private diagnostics withheld")
        raise SystemExit(1) from None
