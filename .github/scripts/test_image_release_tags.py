"""Execute image tag decisions and their workflow consumers without publishing."""

import json
import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
ACTION = ROOT / ".github/actions/determine-tags/action.yml"
LEGACY = {
    "docker.cpu.slim.yml": "roboflow/roboflow-inference-server-cpu-slim",
    "docker.cpu.parallel.yml": "roboflow/roboflow-inference-server-cpu-parallel",
    "docker.gpu.slim.yml": "roboflow/roboflow-inference-server-gpu-slim",
    "docker.gpu.parallel.yml": "roboflow/roboflow-inference-server-gpu-parallel",
    "docker.gpu.udp.yml": "roboflow/roboflow-inference-server-udp-gpu",
    "docker.trt.yml": "roboflow/roboflow-inference-server-trt",
}


def _tags(tmp_path, *, latest="v1.8.7", api_failure=False, api_fields=None, **values):
    tmp_path.mkdir(parents=True, exist_ok=True)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "api-calls"
    curl = fake_bin / "curl"
    body = json.dumps(
        {"tag_name": latest, "prerelease": False, "draft": False, **(api_fields or {})}
    )
    curl.write_text(
        "#!/bin/sh\n"
        + f"echo called >> {shlex.quote(str(calls))}\n"
        + ("exit 22\n" if api_failure else f"printf '%s\\n' {shlex.quote(body)}\n")
    )
    curl.chmod(0o755)
    output = tmp_path / "output"
    env = {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CUSTOM_TAG": "",
        "VERSION": "1.8.7",
        "BASE_IMAGE": "roboflow/image",
        "TOKEN": "test-only",
        "FORCE_PUSH": "false",
        "BRANCH": "v1.8.7",
        "EVENT_NAME": "release",
        "TARGET_BRANCH": "main",
        "REPOSITORY": "roboflow/inference",
        "RELEASE_PRERELEASE": "false",
        "RELEASE_VERSION": "",
        "TAG_PREFIX": "",
        "GITHUB_OUTPUT": str(output),
        **values,
    }
    script = yaml.safe_load(ACTION.read_text())["runs"]["steps"][0]["run"]
    result = subprocess.run(
        ["bash", "-c", script],
        env=env,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=10,
    )
    tags = (
        output.read_text().strip().removeprefix("image_tags=")
        if output.exists()
        else ""
    )
    return result, tags, calls.exists()


@pytest.mark.parametrize("suffix", ["rc1", "-rc.1", "a1", "b2", ".dev3"])
def test_prereleases_never_get_latest_or_query_latest(tmp_path, suffix):
    version = "1.8.7" + suffix
    result, tags, queried = _tags(
        tmp_path, VERSION=version, BRANCH="v" + version, RELEASE_PRERELEASE="true"
    )
    assert result.returncode == 0, result.stderr
    assert tags == "roboflow/image:" + version
    assert not queried


@pytest.mark.parametrize(
    "latest,expected",
    [
        ("v1.8.7", "roboflow/image:1.8.7,roboflow/image:latest"),
        ("v1.9.0", "roboflow/image:1.8.7"),
    ],
)
def test_only_selected_stable_release_gets_latest(tmp_path, latest, expected):
    result, tags, queried = _tags(tmp_path, latest=latest)
    assert result.returncode == 0, result.stderr
    assert tags == expected
    assert queried


@pytest.mark.parametrize(
    "version,release,prerelease",
    [
        ("1.8.7rc1", "v1.8.7rc1", "false"),
        ("1.8.7", "v1.8.7", "true"),
        ("1.8.7", "v1.8.6", "false"),
    ],
)
def test_release_metadata_disagreement_stops_before_push(
    tmp_path, version, release, prerelease
):
    result, tags, _ = _tags(
        tmp_path, VERSION=version, BRANCH=release, RELEASE_PRERELEASE=prerelease
    )
    assert result.returncode != 0
    assert not tags


@pytest.mark.parametrize("force", ["true", "false"])
@pytest.mark.parametrize("custom", ["", "test-rc1"])
def test_manual_publication_never_gets_latest(tmp_path, force, custom):
    result, tags, queried = _tags(
        tmp_path, EVENT_NAME="workflow_dispatch", CUSTOM_TAG=custom, FORCE_PUSH=force
    )
    assert result.returncode == 0, result.stderr
    assert tags == "roboflow/image:1.8.7" + ("-" + custom if custom else "")
    assert not queried


def test_failed_latest_lookup_does_not_publish(tmp_path):
    result, tags, _ = _tags(tmp_path, api_failure=True)
    assert result.returncode != 0
    assert not tags


@pytest.mark.parametrize("workflow", LEGACY)
@pytest.mark.parametrize("event", ["release", "workflow_dispatch"])
def test_legacy_workflow_consumers_cannot_publish_latest_for_rc(
    tmp_path, workflow, event
):
    config = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())
    steps = config["jobs"]["docker"]["steps"]
    version = "1.8.7rc1"
    decision = next(
        (
            step
            for step in steps
            if step.get("uses") == "./.github/actions/determine-tags"
        ),
        None,
    )
    repository = LEGACY[workflow]
    configured_repository = (
        decision["with"]["base_image"]
        if decision
        else config.get("env", {}).get("BASE_IMAGE", repository)
    )
    if configured_repository == "${{ env.BASE_IMAGE }}":
        configured_repository = config["env"]["BASE_IMAGE"]
    assert configured_repository == repository
    selected = ""
    if decision:
        result, selected, _ = _tags(
            tmp_path,
            VERSION=version,
            BRANCH="v" + version,
            RELEASE_PRERELEASE="true",
            EVENT_NAME=event,
            BASE_IMAGE=repository,
        )
        assert result.returncode == 0, result.stderr
    values = {
        "env.VERSION": version,
        "env.BASE_IMAGE": repository,
        "steps.tags.outputs.image_tags": selected,
    }
    outbound = []
    for step in steps:
        for expression in [
            step.get("with", {}).get("tags"),
            step.get("env", {}).get("IMAGE_TAGS"),
        ]:
            if expression:
                actual = re.sub(
                    r"\$\{\{\s*(.*?)\s*\}\}", lambda match: values[match[1]], expression
                )
                outbound.extend(actual.split(","))
    assert outbound, "Check the actual tags passed to the build/promotion steps"
    assert all(tag == repository + ":" + version for tag in outbound), outbound


@pytest.mark.parametrize("platform", ["cpu", "cu118", "cu124", "jp51"])
@pytest.mark.parametrize(
    "version,expected_version",
    [("0.39.11rc1", "0.39.11rc1"), ("0.39.11", "0.39.11-1.8.7rc1")],
)
def test_experimental_workflow_preserves_platform_prefix_without_latest_for_rc(
    tmp_path, platform, version, expected_version
):
    config = yaml.safe_load(
        (ROOT / ".github/workflows/docker.inference-exp.yml").read_text()
    )
    step = next(
        step
        for step in config["jobs"]["build"]["steps"]
        if step.get("name") == "Determine image tags"
    )
    result, selected, _ = _tags(
        tmp_path / "decision",
        VERSION=version,
        RELEASE_VERSION="1.8.7rc1",
        BRANCH="v1.8.7rc1",
        RELEASE_PRERELEASE="true",
        TAG_PREFIX=platform + "-",
        BASE_IMAGE="roboflow/inference-exp",
    )
    assert result.returncode == 0, result.stderr
    output = tmp_path / "output"
    run = subprocess.run(
        ["bash", "-c", step["run"]],
        env={
            "PATH": os.environ["PATH"],
            "BASE_TAG": version,
            "PLATFORM": platform,
            "EVENT_NAME": "release",
            "GIT_REF": "refs/tags/v1.8.7rc1",
            "RELEASE_IMAGE_TAGS": selected,
            "GITHUB_OUTPUT": str(output),
        },
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert run.returncode == 0, run.stderr
    assert (
        output.read_text().strip()
        == f"tags=roboflow/inference-exp:{platform}-{expected_version}"
    )


@pytest.mark.parametrize("suffix", ["post1", ".post2", "-post3"])
def test_final_post_release_remains_eligible_for_latest(tmp_path, suffix):
    version = "1.8.7" + suffix
    result, tags, _ = _tags(
        tmp_path, VERSION=version, BRANCH="v" + version, latest="v" + version
    )
    assert result.returncode == 0, result.stderr
    assert tags == f"roboflow/image:{version},roboflow/image:latest"


@pytest.mark.parametrize("release", ["v1.8.7.post1", "v1.8.7post1", "v1.8.7-post1"])
def test_equivalent_post_release_spellings_keep_checkout_image_version(
    tmp_path, release
):
    result, tags, queried = _tags(
        tmp_path, VERSION="1.8.7.post1", BRANCH=release, latest=release
    )
    assert result.returncode == 0, result.stderr
    assert tags == "roboflow/image:1.8.7.post1,roboflow/image:latest"
    assert queried


@pytest.mark.parametrize(
    "event,prerelease",
    [("workflow_dispatch", "false"), ("push", "false"), ("release", "true")],
)
@pytest.mark.parametrize("prefix", ["", "cu124-"])
def test_reserved_alias_cannot_be_used_as_candidate_version(
    tmp_path, event, prerelease, prefix
):
    result, tags, _ = _tags(
        tmp_path,
        EVENT_NAME=event,
        VERSION="latest",
        BRANCH="latest",
        RELEASE_PRERELEASE=prerelease,
        TAG_PREFIX=prefix,
    )
    assert result.returncode != 0
    assert not tags


@pytest.mark.parametrize(
    "fields",
    [{"tag_name": None}, {"tag_name": ""}, {"prerelease": True}, {"draft": True}],
)
def test_invalid_latest_api_response_stops_publication(tmp_path, fields):
    result, tags, _ = _tags(tmp_path, api_fields=fields)
    assert result.returncode != 0
    assert not tags


@pytest.mark.parametrize(
    "latest,expected",
    [
        (
            "v1.8.7",
            "roboflow/inference-exp:cu124-0.39.11,roboflow/inference-exp:cu124-latest",
        ),
        ("v1.9.0", "roboflow/inference-exp:cu124-0.39.11"),
    ],
)
def test_independent_image_version_uses_runtime_release_for_latest_selection(
    tmp_path, latest, expected
):
    result, tags, _ = _tags(
        tmp_path,
        VERSION="0.39.11",
        RELEASE_VERSION="1.8.7",
        TAG_PREFIX="cu124-",
        BASE_IMAGE="roboflow/inference-exp",
        latest=latest,
    )
    assert result.returncode == 0, result.stderr
    assert tags == expected


def test_rc_cannot_overwrite_final_independent_image_version(tmp_path):
    result, tags, queried = _tags(
        tmp_path,
        VERSION="0.39.11",
        RELEASE_VERSION="1.8.7rc1",
        BRANCH="v1.8.7rc1",
        RELEASE_PRERELEASE="true",
        TAG_PREFIX="cpu-",
    )
    assert result.returncode == 0, result.stderr
    assert tags == "roboflow/image:cpu-0.39.11-1.8.7rc1"
    assert not queried


def test_stable_runtime_cannot_promote_prerelease_image(tmp_path):
    result, tags, _ = _tags(
        tmp_path, VERSION="0.39.11rc1", RELEASE_VERSION="1.8.7", TAG_PREFIX="cpu-"
    )
    assert result.returncode != 0
    assert not tags


def test_release_prerelease_flag_and_independent_version_are_wired():
    action = yaml.safe_load(ACTION.read_text())["runs"]["steps"][0]
    assert (
        action["env"]["RELEASE_PRERELEASE"] == "${{ github.event.release.prerelease }}"
    )
    config = yaml.safe_load(
        (ROOT / ".github/workflows/docker.inference-exp.yml").read_text()
    )
    steps = config["jobs"]["build"]["steps"]
    decision = next(step for step in steps if step.get("id") == "release-tags")
    consumer = next(step for step in steps if step.get("id") == "image-tags")
    assert decision["uses"] == "./.github/actions/determine-tags"
    assert decision["with"]["version"] == "${{ needs.determine-tags.outputs.base-tag }}"
    assert (
        decision["with"]["release_version"]
        == "${{ needs.determine-tags.outputs.release-version }}"
    )
    assert (
        consumer["env"]["RELEASE_IMAGE_TAGS"]
        == "${{ steps.release-tags.outputs.image_tags }}"
    )
    assert steps.index(decision) < steps.index(consumer)


@pytest.mark.parametrize(
    "event,branch,base_tag,expected",
    [
        ("push", "refs/heads/main", "custom", "cpu-test"),
        ("workflow_dispatch", "refs/heads/candidate", "custom", "cpu-custom"),
        ("workflow_dispatch", "refs/heads/candidate", "latest", None),
    ],
)
def test_experimental_non_release_tags_remain_separate(
    tmp_path, event, branch, base_tag, expected
):
    config = yaml.safe_load(
        (ROOT / ".github/workflows/docker.inference-exp.yml").read_text()
    )
    step = next(
        step
        for step in config["jobs"]["build"]["steps"]
        if step.get("id") == "image-tags"
    )
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", "-c", step["run"]],
        env={
            "PATH": os.environ["PATH"],
            "BASE_TAG": base_tag,
            "PLATFORM": "linux/amd64",
            "EVENT_NAME": event,
            "GIT_REF": branch,
            "GITHUB_OUTPUT": str(output),
        },
        text=True,
        capture_output=True,
        timeout=10,
    )
    if expected is None:
        assert result.returncode != 0
        assert not output.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert output.read_text().strip() == f"tags=roboflow/inference-exp:{expected}"
