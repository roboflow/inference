import importlib
import os
import subprocess
import sys

import pytest

from inference.core import env as env_module


def _resolve_runs_on_jetson(monkeypatch, runs_on_jetson=None, running_on_jetson=None):
    with monkeypatch.context() as env_context:
        if runs_on_jetson is None:
            env_context.delenv("RUNS_ON_JETSON", raising=False)
        else:
            env_context.setenv("RUNS_ON_JETSON", runs_on_jetson)
        if running_on_jetson is None:
            env_context.delenv("RUNNING_ON_JETSON", raising=False)
        else:
            env_context.setenv("RUNNING_ON_JETSON", running_on_jetson)
        importlib.reload(env_module)
        result = env_module.RUNS_ON_JETSON
    importlib.reload(env_module)
    return result


def test_runs_on_jetson_uses_running_on_jetson_fallback_when_unset(
    monkeypatch,
) -> None:
    assert (
        _resolve_runs_on_jetson(
            monkeypatch, runs_on_jetson=None, running_on_jetson="True"
        )
        is True
    )


def test_runs_on_jetson_explicit_false_overrides_running_on_jetson(
    monkeypatch,
) -> None:
    assert (
        _resolve_runs_on_jetson(
            monkeypatch, runs_on_jetson="False", running_on_jetson="True"
        )
        is False
    )


def test_runs_on_jetson_preserves_false_default_when_both_names_unset(
    monkeypatch,
) -> None:
    assert (
        _resolve_runs_on_jetson(
            monkeypatch, runs_on_jetson=None, running_on_jetson=None
        )
        is False
    )


def test_assume_identity_service_access_token_uses_roboflow_env_name() -> None:
    original_scoped_token = os.environ.get(
        "ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN"
    )
    original_token = os.environ.get("ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN")
    try:
        os.environ.pop("ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN", None)
        os.environ["ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN"] = "assume-token"

        importlib.reload(env_module)

        assert (
            env_module.ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN == "assume-token"
        )
    finally:
        if original_scoped_token is None:
            os.environ.pop("ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN", None)
        else:
            os.environ["ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN"] = (
                original_scoped_token
            )
        if original_token is None:
            os.environ.pop("ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN", None)
        else:
            os.environ["ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN"] = original_token
        importlib.reload(env_module)


def test_workflows_remote_api_key_transport_allowed_values_stay_in_sync_with_sdk() -> (
    None
):
    # given - env.py duplicates the allowed values on purpose (it must not
    # import inference_sdk); this test is the enforcement of the KEEP IN SYNC
    # comment next to _ALLOWED_WORKFLOWS_REMOTE_API_KEY_TRANSPORTS.
    from inference.core.env import _ALLOWED_WORKFLOWS_REMOTE_API_KEY_TRANSPORTS
    from inference_sdk.http.entities import ApiKeyTransport

    # then
    assert set(_ALLOWED_WORKFLOWS_REMOTE_API_KEY_TRANSPORTS) == {
        transport.value for transport in ApiKeyTransport
    }


def test_workflows_remote_api_key_transport_rejects_invalid_value() -> None:
    # given - env.py validates at import time, hence the subprocess
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env["WORKFLOWS_REMOTE_API_KEY_TRANSPORT"] = "invalid"

    # when
    result = subprocess.run(
        [sys.executable, "-c", "import inference.core.env"],
        capture_output=True,
        text=True,
        env=env,
    )

    # then
    assert result.returncode != 0
    assert "Invalid WORKFLOWS_REMOTE_API_KEY_TRANSPORT" in result.stderr


@pytest.mark.parametrize("offline_mode", [False, True])
@pytest.mark.parametrize("cache_auth", [False, True])
@pytest.mark.parametrize("local_packages", [False, True])
def test_local_packages_require_model_authorization_to_be_disabled(
    offline_mode: bool, cache_auth: bool, local_packages: bool
) -> None:
    env = {
        **os.environ,
        "DISABLE_VERSION_CHECK": "True",
        "USE_INFERENCE_MODELS": "True",
        "OFFLINE_MODE": str(offline_mode),
        "_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START": str(offline_mode),
        "ALLOW_OFFLINE_MODEL_CACHE_AUTH_BYPASS": "True",
        "MODELS_CACHE_AUTH_ENABLED": str(cache_auth),
        "ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES": str(local_packages),
    }
    result = subprocess.run(
        [sys.executable, "-c", "import inference.core.env"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )

    if cache_auth and local_packages and not offline_mode:
        assert result.returncode != 0
        assert "cannot authorize local model paths" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
