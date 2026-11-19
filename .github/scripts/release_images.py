"""Plan, smoke-test and promote retained image digests without rebuilding."""

import json
import os
import re
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".release/pypi"))
import flow
from packaging.utils import canonicalize_name
from packaging.version import Version


def _command(*args, check=True):
    result = subprocess.run(args, text=True, capture_output=True, check=check)
    return result


def _digest(reference, *, allow_missing=False):
    result = _command(
        "docker",
        "buildx",
        "imagetools",
        "inspect",
        reference,
        "--format",
        "{{json .Manifest}}",
        check=False,
    )
    if result.returncode:
        if allow_missing and re.search(
            r"manifest unknown|not found|no such manifest", result.stderr, re.I
        ):
            return None
        raise ValueError(f"Cannot inspect image {reference}: {result.stderr}")
    value = json.loads(result.stdout)["digest"]
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", value):
        raise ValueError("Invalid registry digest")
    return value


def _plan(key):
    state = flow._state()
    item = next(image for image in state["policy"]["images"] if image["key"] == key)
    version = state["version"]
    if item["version_source"] == "models":
        version = tomllib.loads(Path("inference_models/pyproject.toml").read_text())[
            "project"
        ]["version"]
        # An unchanged stable Models version must not give an RC a stable image tag.
        if state["channel"] == "rc":
            version = state["version"]
    tag = item["tag_prefix"] + version
    repositories = [item["repository"]]
    if item["gcp"]:
        repositories.append(
            "us-docker.pkg.dev/roboflow-artifacts/" + item["repository"]
        )
    tests = []
    for platform in item["platforms"]:
        runner = (
            ["ubuntu-24.04-arm"] if platform.endswith("arm64") else ["ubuntu-latest"]
        )
        if item["accelerator"] == "cuda":
            runner = ["Roboflow-GPU-VM-Runner"]
        elif item["accelerator"] == "jetson":
            runner = item["runner"]
        tests.append({"platform": platform, "runner": runner})
    result = {
        **item,
        "tag": tag,
        "repositories": repositories,
        "candidate": item["repository"] + f":candidate-{key}-{state['run_id']}",
        "channel": state["channel"],
        "release_version": state["version"],
        "source_sha": state["source_sha"],
        "smoke_matrix": {"include": tests},
    }
    if state["channel"] == "stable":
        receipt = json.loads((flow.STATE / "accepted-rc.json").read_text())
        result["rc_smoke"] = next(
            image["smoke"] for image in receipt["images"] if image["key"] == key
        )
    flow._write(Path("image-plan.json"), result)
    flow._outputs(
        candidate=result["candidate"],
        smoke_matrix=json.dumps(result["smoke_matrix"]),
        platforms=",".join(item["platforms"]),
    )


def _smoke(plan, *, digest, platform):
    reference = plan["repository"] + "@" + digest
    _command("docker", "pull", "--platform", platform, reference)
    options = ["--gpus", "all"] if plan["accelerator"] == "cuda" else []
    if plan["accelerator"] == "jetson":
        options = ["--runtime", "nvidia"]
    script = Path(".github/scripts/release_image_smoke.py").resolve()
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--platform",
            platform,
            *options,
            "--entrypoint",
            "python3",
            "-e",
            "RELEASE_IMAGE_PROFILE=" + json.dumps(plan),
            "-v",
            f"{script}:/release_image_smoke.py:ro",
            reference,
            "/release_image_smoke.py",
        ],
        check=True,
    )
    inventory_result = _command(
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--platform",
        platform,
        "--entrypoint",
        "python3",
        reference,
        "-c",
        'import importlib.metadata as m,json; print(json.dumps([{"name":d.metadata["Name"],"version":d.version} for d in m.distributions()]))',
    )
    inventory = {
        canonicalize_name(item["name"]): (
            Version(item["version"]).base_version
            if canonicalize_name(item["name"]) in flow.FIRST_PARTY
            else item["version"]
        )
        for item in json.loads(inventory_result.stdout)
    }
    if plan["channel"] == "stable" and inventory != plan["rc_smoke"][platform]:
        raise ValueError("Image Python dependency inventory differs from tested RC")
    flow._write(
        Path("image-smoke.json"),
        {
            "key": plan["key"],
            "digest": digest,
            "platform": platform,
            "inventory": inventory,
        },
    )
    if not plan["http"]:
        return
    container = _command(
        "docker",
        "run",
        "-d",
        "--pull=never",
        "--platform",
        platform,
        *options,
        "-e",
        "PORT=9001",
        reference,
    ).stdout.strip()
    try:
        for _ in range(60):
            result = _command(
                "docker",
                "exec",
                container,
                "python3",
                "-c",
                "import urllib.request; r=urllib.request.urlopen('http://127.0.0.1:9001/info', timeout=3); assert r.status == 200",
                check=False,
            )
            if result.returncode == 0:
                return
            time.sleep(5)
        logs = _command("docker", "logs", container, check=False)
        print(logs.stdout)
        print(logs.stderr)
        raise ValueError("Image HTTP startup/health smoke failed")
    finally:
        _command("docker", "rm", "-f", container, check=False)


def _copy(reference, *, target):
    # Copy all blobs and platforms, not just an index pointing into Docker Hub.
    _command(
        "skopeo",
        "copy",
        "--all",
        "--preserve-digests",
        "--retry-times",
        "2",
        "--authfile",
        str(Path.home() / ".docker/config.json"),
        "docker://" + reference,
        "docker://" + target,
    )


def _promote(plan, *, digest):
    reference = plan["repository"] + "@" + digest
    if _digest(reference) != digest:
        raise ValueError("Retained build digest is unavailable")
    latest = False
    if plan["channel"] == "stable":
        latest = (
            flow._api("releases/latest")["tag_name"].removeprefix("v")
            == plan["release_version"]
        )
    smoke = {}
    for path in Path("image-smoke").rglob("image-smoke.json"):
        receipt = json.loads(path.read_text())
        if receipt["key"] != plan["key"] or receipt["digest"] != digest:
            raise ValueError("Image smoke receipt does not match retained digest")
        platform = receipt["platform"]
        if platform in smoke and smoke[platform] != receipt["inventory"]:
            raise ValueError("Conflicting image smoke inventories")
        smoke[platform] = receipt["inventory"]
    if set(smoke) != set(plan["platforms"]):
        raise ValueError("Missing target-platform smoke receipts")
    tags = []
    for repository in plan["repositories"]:
        target = repository + ":" + plan["tag"]
        existing = _digest(target, allow_missing=True)
        if existing and existing != digest:
            raise ValueError(f"Refusing to overwrite released image {target}")
        if existing is None:
            _copy(reference, target=target)
        if _digest(target) != digest:
            raise ValueError(f"Publication verification failed for {target}")
        tags.append(target)
    if latest:
        for repository in plan["repositories"]:
            target = repository + ":" + plan["tag_prefix"] + "latest"
            _copy(reference, target=target)
            if _digest(target) != digest:
                raise ValueError(f"Latest digest verification failed for {target}")
            tags.append(target)
    flow._write(
        Path("image-receipt.json"),
        {
            "key": plan["key"],
            "source_sha": plan["source_sha"],
            "digest": digest,
            "tags": tags,
            "smoke": smoke,
        },
    )


@click.command()
@click.argument("operation", type=click.Choice(["plan", "smoke", "promote"]))
@click.option(
    "--key",
)
@click.option(
    "--digest",
)
@click.option(
    "--platform",
)
def _main(operation, *, key, digest, platform):
    if operation == "plan":
        _plan(key)
        return
    if not digest or not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
        raise click.ClickException("A retained build digest is required")
    plan = json.loads(Path("image-plan.json").read_text())
    if operation == "smoke":
        if platform not in plan["platforms"]:
            raise click.ClickException("Unexpected test platform")
        _smoke(plan, digest=digest, platform=platform)
    else:
        _promote(plan, digest=digest)


if __name__ == "__main__":
    _main()
