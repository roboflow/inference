"""Behavioral tests for the process-wide OFFLINE_MODE startup latch."""

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[4])
_OFFLINE_MODE_LATCH = "_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"
_CONTROLLED_VARIABLES = [
    "OFFLINE_MODE",
    _OFFLINE_MODE_LATCH,
    "SAM3_EXEC_MODE",
    "SAM3_FINE_TUNED_MODELS_ENABLED",
    "WORKFLOWS_STEP_EXECUTION_MODE",
    "WEBRTC_MODAL_TOKEN_ID",
    "WEBRTC_MODAL_TOKEN_SECRET",
    "OTEL_TRACING_ENABLED",
    "OTEL_METRICS_ENABLED",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "DISABLE_VERSION_CHECK",
    "INFERENCE_WARNINGS_DISABLED",
    "PYTHONWARNINGS",
]


def _run_with_env(code: str, extra_env: dict) -> None:
    env = {**os.environ}
    for variable in _CONTROLLED_VARIABLES:
        env.pop(variable, None)
    env.update(extra_env)
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_runtime_environment_change_cannot_enable_offline_mode() -> None:
    _run_with_env(
        """
import importlib
import os

from inference.core import env as core_env

assert core_env.OFFLINE_MODE is False
os.environ["OFFLINE_MODE"] = "True"
os.environ["_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"] = "True"
importlib.reload(core_env)
models_config = importlib.import_module("inference_models.configuration")
assert core_env.OFFLINE_MODE is False
assert models_config.OFFLINE_MODE is False
assert os.environ["_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"] == "False"
assert "HF_HUB_OFFLINE" not in os.environ
assert "TRANSFORMERS_OFFLINE" not in os.environ
""",
        {
            "OFFLINE_MODE": "False",
            "DISABLE_VERSION_CHECK": "True",
        },
    )


def test_both_packages_share_latch_and_offline_forces_local_execution() -> None:
    _run_with_env(
        """
import importlib
import os

models_config = importlib.import_module("inference_models.configuration")
assert models_config.OFFLINE_MODE is True
os.environ["OFFLINE_MODE"] = "False"
os.environ["_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"] = "False"
from inference.core import env as core_env
assert core_env.OFFLINE_MODE is True
assert core_env.SAM3_EXEC_MODE == "local"
assert core_env.SAM3_FINE_TUNED_MODELS_ENABLED is True
assert core_env.WORKFLOWS_STEP_EXECUTION_MODE == "local"
assert core_env.WEBRTC_MODAL_TOKEN_ID is None
assert core_env.WEBRTC_MODAL_TOKEN_SECRET is None
assert core_env.OTEL_TRACING_ENABLED is False
assert core_env.OTEL_METRICS_ENABLED is False
assert os.environ["HF_HUB_OFFLINE"] == "1"
assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
importlib.reload(models_config)
assert models_config.OFFLINE_MODE is True
assert os.environ["_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"] == "True"
""",
        {
            "OFFLINE_MODE": "True",
            "SAM3_EXEC_MODE": "remote",
            "WORKFLOWS_STEP_EXECUTION_MODE": "remote",
            "WEBRTC_MODAL_TOKEN_ID": "runtime-token-id",
            "WEBRTC_MODAL_TOKEN_SECRET": "runtime-token-secret",
            "OTEL_TRACING_ENABLED": "True",
            "OTEL_METRICS_ENABLED": "True",
        },
    )
