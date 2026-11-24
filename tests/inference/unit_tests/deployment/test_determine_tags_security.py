import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[4]
ACTION = ROOT / ".github/actions/determine-tags/action.yml"


def run_tags(tmp_path, **overrides):
    script = yaml.safe_load(ACTION.read_text())["runs"]["steps"][0]["run"]
    assert "${{" not in script
    output = tmp_path / "outputs"
    env = dict(
        PATH=os.environ["PATH"],
        CUSTOM_TAG="",
        VERSION="1.2.3",
        BASE_IMAGE="roboflow/inference",
        TOKEN="test-only",
        FORCE_PUSH="false",
        BRANCH="main",
        EVENT_NAME="workflow_dispatch",
        TARGET_BRANCH="",
        REPOSITORY="roboflow/inference",
        GITHUB_OUTPUT=str(output),
    )
    env.update(overrides)
    result = subprocess.run(
        ["bash", "-c", script],
        env=env,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=10,
    )
    return result, output.read_text() if output.exists() else ""


@pytest.mark.parametrize(
    "custom",
    [
        'bad"; false; #',
        "$(false)",
        "valid\ninjected=value",
        "valid\rvalue",
        "a" * 128,
        "space tag",
    ],
)
def test_rejects_unsafe_or_oversized_complete_tags(tmp_path, custom):
    result, output = run_tags(tmp_path, CUSTOM_TAG=custom)
    assert result.returncode != 0
    assert output == ""


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, "1.2.3"),
        ({"CUSTOM_TAG": "test_01"}, "1.2.3-test_01"),
        ({"EVENT_NAME": "push", "BRANCH": "Feature/JP7.2"}, "feature-jp7.2"),
    ],
)
def test_preserves_valid_manual_and_push_tags(tmp_path, overrides, expected):
    result, output = run_tags(tmp_path, **overrides)
    assert result.returncode == 0, result.stderr
    assert output == f"image_tags=roboflow/inference:{expected}\n"


@pytest.mark.parametrize("field", ["VERSION", "BASE_IMAGE", "BRANCH", "TARGET_BRANCH"])
def test_rejects_line_breaks_in_all_contexts(tmp_path, field):
    result, output = run_tags(tmp_path, **{field: "valid\nother"})
    assert result.returncode != 0
    assert not output


@pytest.mark.parametrize(
    "release,expected",
    [
        ("v1.2.3", "1.2.3,roboflow/inference:latest"),
        ("v1.2.2", "1.2.2"),
        ("1.2.3-rc1", "1.2.3-rc1"),
    ],
)
def test_preserves_release_tag_rules_without_network(tmp_path, release, expected):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    # Test-only deterministic API response; the real curl is never invoked.
    for name, body in {
        "curl": "#!/bin/sh\nprintf '%s\\n' '{\"tag_name\":\"v1.2.3\"}'\n",
        "jq": "#!/bin/sh\ncat >/dev/null\nprintf 'v1.2.3\\n'\n",
    }.items():
        executable = fake_bin / name
        executable.write_text(body)
        executable.chmod(0o755)
    result, output = run_tags(
        tmp_path, EVENT_NAME="release", BRANCH=release, PATH=f"{fake_bin}:/usr/bin:/bin"
    )
    assert result.returncode == 0, result.stderr
    assert output == f"image_tags=roboflow/inference:{expected}\n"


@pytest.mark.parametrize("release", ["1.2.3,evil", "1.2.3,roboflow/inference:evil"])
def test_release_cannot_inject_additional_repository_or_tag(tmp_path, release):
    result, output = run_tags(tmp_path, EVENT_NAME="release", BRANCH=release)
    assert result.returncode != 0
    assert not output


@pytest.mark.parametrize("custom", ["$(touch sentinel)", 'bad"; touch sentinel; #'])
def test_custom_tag_never_executes_submitted_shell(tmp_path, custom):
    result, output = run_tags(tmp_path, CUSTOM_TAG=custom)
    assert result.returncode != 0
    assert not output
    assert not (tmp_path / "sentinel").exists()
