import os
import subprocess
import sys
from typing import Dict

import pytest

_SETTING_NAMES = (
    "ACTIVE_LEARNING_ENABLED",
    "ACTIVE_LEARNING_TAGS",
    "OFFLINE_MODE",
    "_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START",
)


def _run(code: str, env: Dict[str, str]) -> subprocess.CompletedProcess:
    base_env = {
        name: value for name, value in os.environ.items() if name not in _SETTING_NAMES
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"from inference_server import configuration as c; {code}",
        ],
        env={**base_env, **env},
        capture_output=True,
        text=True,
    )

    return result


def test_active_learning_settings_default_to_legacy_values() -> None:
    result = _run(
        "assert c.ACTIVE_LEARNING_ENABLED is True; "
        "assert c.ACTIVE_LEARNING_TAGS is None",
        env={},
    )

    assert result.returncode == 0, result.stderr


def test_active_learning_can_be_disabled() -> None:
    result = _run(
        "assert c.ACTIVE_LEARNING_ENABLED is False",
        env={"ACTIVE_LEARNING_ENABLED": "False"},
    )

    assert result.returncode == 0, result.stderr


def test_offline_mode_forces_active_learning_off() -> None:
    result = _run(
        "assert c.LEGACY_OFFLINE_MODE is True; "
        "assert c.ACTIVE_LEARNING_ENABLED is False",
        env={
            "ACTIVE_LEARNING_ENABLED": "True",
            "_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START": "true",
        },
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("factory-x,line-y", ["factory-x", "line-y"]),
        ("single", ["single"]),
        (" spaced , kept ", [" spaced ", " kept "]),
        ("", [""]),
    ],
)
def test_active_learning_tags_are_split_on_commas_as_legacy(raw, expected) -> None:
    result = _run(
        f"assert c.ACTIVE_LEARNING_TAGS == {expected!r}, c.ACTIVE_LEARNING_TAGS",
        env={"ACTIVE_LEARNING_TAGS": raw},
    )

    assert result.returncode == 0, result.stderr
