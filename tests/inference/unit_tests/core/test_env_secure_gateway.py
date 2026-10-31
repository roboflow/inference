"""Behavioral tests for SECURE_GATEWAY side-effects in inference.core.env.

The flags are computed at import time, so each case runs in a fresh
interpreter instead of reloading the module in-process (reloading would
leave mutated globals behind for other tests).
"""

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[4])

# Flags that drive the behavior under test - ambient values (CI machines,
# developer shells, repo-root .env files) must not leak into the subprocess.
_CONTROLLED_VARIABLES = [
    "SECURE_GATEWAY",
    "LICENSE_SERVER",
    "WORKFLOWS_STEP_EXECUTION_MODE",
    "WORKFLOWS_REMOTE_API_TARGET",
    "DISABLE_VERSION_CHECK",
    "VERSION_CHECK_MODE",
    "INFERENCE_WARNINGS_DISABLED",
    "PYTHONWARNINGS",
]

_ASSERT_GATEWAY_FORCES_LOCAL = (
    "from inference.core import env; "
    "assert env.WORKFLOWS_STEP_EXECUTION_MODE == 'local', "
    "env.WORKFLOWS_STEP_EXECUTION_MODE; "
    "assert env.DISABLE_VERSION_CHECK is True"
)

_ASSERT_SELF_HOSTED_PRESERVED = (
    "from inference.core import env; "
    "assert env.WORKFLOWS_STEP_EXECUTION_MODE == 'remote', "
    "env.WORKFLOWS_STEP_EXECUTION_MODE"
)

_ASSERT_DEFAULTS_UNCHANGED = (
    "from inference.core import env; "
    "assert env.WORKFLOWS_STEP_EXECUTION_MODE == 'local'; "
    "assert env.DISABLE_VERSION_CHECK is False"
)


def _run_with_env(code: str, extra_env: dict) -> None:
    env = {**os.environ}
    for variable in _CONTROLLED_VARIABLES:
        env.pop(variable, None)
    # hard-empty instead of unset: env.py's load_dotenv() fills missing keys
    # from a repo-root .env, which could re-introduce a gateway setting
    env["SECURE_GATEWAY"] = ""
    env["LICENSE_SERVER"] = ""
    env.update(extra_env)
    # `python -c` resolves `inference` from the subprocess CWD - pin the repo
    # checkout explicitly so a stale site-packages install is never imported.
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


def test_secure_gateway_forces_local_execution_and_disables_version_check() -> None:
    _run_with_env(
        _ASSERT_GATEWAY_FORCES_LOCAL,
        {
            "SECURE_GATEWAY": "gateway.local:8080",
            "WORKFLOWS_STEP_EXECUTION_MODE": "remote",
            "WORKFLOWS_REMOTE_API_TARGET": "hosted",
            "DISABLE_VERSION_CHECK": "False",
        },
    )


def test_secure_gateway_preserves_remote_self_hosted_execution() -> None:
    _run_with_env(
        _ASSERT_SELF_HOSTED_PRESERVED,
        {
            "SECURE_GATEWAY": "gateway.local:8080",
            "WORKFLOWS_STEP_EXECUTION_MODE": "remote",
            "WORKFLOWS_REMOTE_API_TARGET": "self-hosted",
        },
    )


def test_defaults_unchanged_without_secure_gateway() -> None:
    _run_with_env(
        _ASSERT_DEFAULTS_UNCHANGED,
        {
            "WORKFLOWS_STEP_EXECUTION_MODE": "local",
            "DISABLE_VERSION_CHECK": "False",
        },
    )
