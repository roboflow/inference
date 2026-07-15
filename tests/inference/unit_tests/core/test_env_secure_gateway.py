"""Behavioral tests for SECURE_GATEWAY side-effects in inference.core.env.

The flags are computed at import time, so each case runs in a fresh
interpreter instead of reloading the module in-process (reloading would
leave mutated globals behind for other tests).
"""

import os
import subprocess
import sys

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
    env = {**os.environ, **extra_env}
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
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
