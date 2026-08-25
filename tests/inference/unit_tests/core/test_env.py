import importlib
import os

from inference.core import env as env_module


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
