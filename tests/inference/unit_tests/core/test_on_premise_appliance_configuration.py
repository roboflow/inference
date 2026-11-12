import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
PRINT_FLAGS = (
    "from inference.core import env; "
    "print(env.ON_PREMISE_APPLIANCE, env.MODELS_CACHE_AUTH_ENABLED)"
)


def _import_env(**overrides: str) -> subprocess.CompletedProcess:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"ON_PREMISE_APPLIANCE", "MODELS_CACHE_AUTH_ENABLED"}
    }
    environment.update(overrides)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT)]
        + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
    )

    return subprocess.run(
        [sys.executable, "-c", PRINT_FLAGS],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, "False False"),
        ({"MODELS_CACHE_AUTH_ENABLED": "True"}, "False True"),
        ({"ON_PREMISE_APPLIANCE": "True"}, "True True"),
        (
            {"ON_PREMISE_APPLIANCE": "True", "MODELS_CACHE_AUTH_ENABLED": "True"},
            "True True",
        ),
    ],
)
def test_appliance_mode_turns_on_per_model_checks_by_default(
    overrides, expected
) -> None:
    result = _import_env(**overrides)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == expected


def test_appliance_mode_refuses_per_model_checks_turned_off() -> None:
    result = _import_env(ON_PREMISE_APPLIANCE="True", MODELS_CACHE_AUTH_ENABLED="False")

    assert result.returncode != 0
    assert "ON_PREMISE_APPLIANCE requires MODELS_CACHE_AUTH_ENABLED=True" in (
        result.stderr
    )
