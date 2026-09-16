"""Execute actual workflow shell bodies locally, without registry or API operations."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[4]
WORKFLOWS = ROOT / ".github/workflows"
INSTALLERS = [
    "build-windows-installer-cpu.yml",
    "build-windows-installer-gpu.yml",
    "build-osx-bundle.yml",
]


def run_step(tmp_path, workflow, step_name, **values):
    config = yaml.safe_load((WORKFLOWS / workflow).read_text())
    step = next(
        step
        for job in config["jobs"].values()
        for step in job["steps"]
        if step.get("name") == step_name
    )
    assert "${{" not in step["run"]
    output = tmp_path / "output"
    env_output = tmp_path / "environment"
    env = {
        "PATH": "/usr/bin:/bin:/opt/homebrew/bin",
        "GITHUB_OUTPUT": str(output),
        "GITHUB_ENV": str(env_output),
        **values,
    }
    result = subprocess.run(
        ["bash", "-c", step["run"]],
        env=env,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=10,
    )
    return (
        result,
        output.read_text() if output.exists() else "",
        env_output.read_text() if env_output.exists() else "",
    )


BAD_VALUES = [
    "$(touch sentinel)",
    'x"; touch sentinel; #',
    "valid\ninjected=value",
    "valid\rvalue",
    "x" * 129,
    "space tag",
]


@pytest.mark.parametrize("value", BAD_VALUES)
@pytest.mark.parametrize(
    "workflow,step,field,defaults",
    [
        (
            "docker.inference-exp.yml",
            "generate tags",
            "CUSTOM_TAG",
            {"VERSION": "1.2.3"},
        ),
        (
            "docker.inference-exp.yml",
            "Check if should build",
            "IMAGE_VARIANT",
            {"EVENT_NAME": "workflow_dispatch", "PLATFORM": "linux/amd64"},
        ),
        (
            "docker.inference-exp.yml",
            "Determine image tags",
            "BASE_TAG",
            {"EVENT_NAME": "release", "PLATFORM": "cpu", "GIT_REF": "refs/tags/v1.2.3"},
        ),
        ("docker.jetson.7.2.0.yml", "Validate media rebuild tag", "CUSTOM_TAG", {}),
        (
            "docker.jetson.7.2.0.yml",
            "Use current-ref media image",
            "CUSTOM_TAG",
            {
                "MEDIA_DIGEST": "sha256:" + "a" * 64,
                "MEDIA_BASE_IMAGE": "roboflow/media",
            },
        ),
    ],
)
def test_workflows_reject_unsafe_whole_inputs(
    tmp_path, value, workflow, step, field, defaults
):
    result, output, environment = run_step(
        tmp_path, workflow, step, **defaults, **{field: value}
    )
    assert result.returncode != 0
    assert not output and not environment
    assert not (tmp_path / "sentinel").exists()


@pytest.mark.parametrize("workflow", INSTALLERS)
@pytest.mark.parametrize("version", BAD_VALUES)
def test_installer_manual_version_is_data_and_validated(tmp_path, workflow, version):
    result, output, _ = run_step(
        tmp_path,
        workflow,
        "Determine Version",
        EVENT_NAME="workflow_dispatch",
        MANUAL_VERSION=version,
        GIT_REF="refs/heads/main",
    )
    assert result.returncode != 0
    assert not output
    assert not (tmp_path / "sentinel").exists()


@pytest.mark.parametrize("workflow", INSTALLERS)
@pytest.mark.parametrize(
    "event,manual,ref",
    [
        ("workflow_dispatch", "v1.2.3-rc.1", "refs/heads/main"),
        ("release", "", "refs/tags/v1.2.3-rc.1"),
    ],
)
def test_installers_preserve_manual_and_release_versions(
    tmp_path, workflow, event, manual, ref
):
    result, output, _ = run_step(
        tmp_path,
        workflow,
        "Determine Version",
        EVENT_NAME=event,
        MANUAL_VERSION=manual,
        GIT_REF=ref,
    )
    assert result.returncode == 0, result.stderr
    key = (
        "BUILD_VERSION_ENV"
        if workflow == "build-osx-bundle.yml"
        else "installer_version_out"
    )
    assert output == f"{key}=1.2.3-rc.1\n"


@pytest.mark.parametrize(
    "step,values,expected",
    [
        ("generate tags", {"CUSTOM_TAG": "", "VERSION": "1.2.3"}, "base-tag=1.2.3\n"),
        (
            "generate tags",
            {"CUSTOM_TAG": "canary", "VERSION": "1.2.3"},
            "base-tag=canary\n",
        ),
        (
            "Check if should build",
            {
                "IMAGE_VARIANT": "cpu",
                "PLATFORM": "linux/amd64",
                "EVENT_NAME": "workflow_dispatch",
            },
            "should-build=true\n",
        ),
        (
            "Determine image tags",
            {
                "BASE_TAG": "1.2.3",
                "PLATFORM": "linux/amd64",
                "EVENT_NAME": "release",
                "GIT_REF": "refs/tags/v1.2.3",
            },
            "tags=roboflow/inference-exp:cpu-1.2.3,roboflow/inference-exp:cpu-latest\n",
        ),
        (
            "Determine image tags",
            {
                "BASE_TAG": "1.2.3",
                "PLATFORM": "cu124",
                "EVENT_NAME": "push",
                "GIT_REF": "refs/heads/main",
            },
            "tags=roboflow/inference-exp:cu124-test\n",
        ),
    ],
)
def test_experimental_workflow_preserves_tag_and_platform_rules(
    tmp_path, step, values, expected
):
    result, output, _ = run_step(tmp_path, "docker.inference-exp.yml", step, **values)
    assert result.returncode == 0, result.stderr
    assert output == expected


def test_jetson_media_reference_is_validated_and_preserved(tmp_path):
    result, _, environment = run_step(
        tmp_path,
        "docker.jetson.7.2.0.yml",
        "Use current-ref media image",
        CUSTOM_TAG="canary",
        MEDIA_BASE_IMAGE="roboflow/media",
        MEDIA_DIGEST="sha256:" + "a" * 64,
    )
    assert result.returncode == 0, result.stderr
    assert (
        environment
        == "JETSON_MEDIA_IMAGE=roboflow/media:canary-media@sha256:" + "a" * 64 + "\n"
    )


def test_exp_final_tag_accounts_for_platform_prefix_length(tmp_path):
    result, output, _ = run_step(
        tmp_path,
        "docker.inference-exp.yml",
        "Determine image tags",
        BASE_TAG="a" * 128,
        PLATFORM="cpu",
        EVENT_NAME="release",
        GIT_REF="refs/tags/v1.2.3",
    )
    assert result.returncode != 0
    assert not output


def test_mergeable_branch_ref_is_encoded_data_without_network(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gh = fake_bin / "gh"
    gh.write_text(
        '#!/bin/sh\nprintf \'%s\' "$2" > "$CAPTURE_PATH"\nprintf \'%s\\n\' \'{"status":"ahead"}\'\n'
    )
    gh.chmod(0o755)
    capture = tmp_path / "url"
    branch = "fix/$(touch sentinel)"
    result, output, _ = run_step(
        tmp_path,
        "check_if_branch_is_mergeable.yml",
        "Check if branch contains all commits from main via GitHub CLI",
        SOURCE_BRANCH=branch,
        TARGET_REPO="roboflow/inference",
        SOURCE_OWNER="roboflow",
        PATH=f"{fake_bin}:/usr/bin:/bin:/opt/homebrew/bin",
        CAPTURE_PATH=str(capture),
    )
    assert result.returncode == 0, result.stderr
    assert output == "mergeable_state=clean\n"
    assert "fix%2F%24" in capture.read_text()
    assert "touch%20sentinel" in capture.read_text()
    assert not (tmp_path / "sentinel").exists()
