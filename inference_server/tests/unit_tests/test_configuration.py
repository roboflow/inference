import os
import subprocess
import sys

from inference_server import configuration


def test_app_port_default_is_9001():
    assert configuration.APP_PORT_DEFAULT == 9001


def test_offline_mode_forces_local_sam3_execution():
    code = (
        "from inference_server import configuration as c; "
        "assert c.SAM3_EXEC_MODE == 'local', c.SAM3_EXEC_MODE; "
        "assert c.SAM3_FINE_TUNED_MODELS_ENABLED is False, "
        "c.SAM3_FINE_TUNED_MODELS_ENABLED"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "OFFLINE_MODE": "true",
            "SAM3_EXEC_MODE": "remote",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_hosted_flags_default_to_legacy_values():
    code = (
        "from inference_server import configuration as c; "
        "assert c.LAMBDA is False; "
        "assert c.GCP_SERVERLESS is False; "
        "assert c.ENFORCE_CREDITS_VERIFICATION is False; "
        "assert c.DEDICATED_DEPLOYMENT_ID is None; "
        "assert c.DEDICATED_DEPLOYMENT_WORKSPACE_ID is None; "
        "assert c.DEDICATED_DEPLOYMENT_WORKSPACE_URL is None; "
        "assert c.WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT is None; "
        "assert c.TRANSIENT_ROBOFLOW_API_ERRORS == set(); "
        "assert c.TRANSIENT_ROBOFLOW_API_ERRORS_RETRIES == 3; "
        "assert c.TRANSIENT_ROBOFLOW_API_ERRORS_RETRY_INTERVAL == 1; "
        "assert c.RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API is False; "
        "assert c.ROBOFLOW_SERVICE_SECRET is None; "
        "assert c.ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN is None"
    )
    env = {
        name: value
        for name, value in os.environ.items()
        if name
        not in {
            "LAMBDA",
            "GCP_SERVERLESS",
            "ENFORCE_CREDITS_VERIFICATION",
            "DEDICATED_DEPLOYMENT_ID",
            "DEDICATED_DEPLOYMENT_WORKSPACE_ID",
            "DEDICATED_DEPLOYMENT_WORKSPACE_URL",
            "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT",
            "ROBOFLOW_SERVICE_SECRET",
            "ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN",
            "ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN",
            "TRANSIENT_ROBOFLOW_API_ERRORS",
            "TRANSIENT_ROBOFLOW_API_ERRORS_RETRIES",
            "TRANSIENT_ROBOFLOW_API_ERRORS_RETRY_INTERVAL",
            "RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API",
        }
    }
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_hosted_flags_read_legacy_spellings():
    code = (
        "from inference_server import configuration as c; "
        "assert c.LAMBDA is True; "
        "assert c.GCP_SERVERLESS is True; "
        "assert c.ENFORCE_CREDITS_VERIFICATION is True; "
        "assert c.DEDICATED_DEPLOYMENT_WORKSPACE_URL == 'ws-url'; "
        "assert c.WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT == ['a', 'b']; "
        "assert c.ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN == 'legacy-tok'; "
        "assert c.TRANSIENT_ROBOFLOW_API_ERRORS == {502, 503}; "
        "assert c.TRANSIENT_ROBOFLOW_API_ERRORS_RETRIES == 5; "
        "assert c.TRANSIENT_ROBOFLOW_API_ERRORS_RETRY_INTERVAL == 2; "
        "assert c.RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API is True"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "LAMBDA": "True",
            "GCP_SERVERLESS": "true",
            "ENFORCE_CREDITS_VERIFICATION": "1",
            "DEDICATED_DEPLOYMENT_WORKSPACE_URL": "ws-url",
            "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT": " a, b ",
            "ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN": "legacy-tok",
            "TRANSIENT_ROBOFLOW_API_ERRORS": "502, 503",
            "TRANSIENT_ROBOFLOW_API_ERRORS_RETRIES": "5",
            "TRANSIENT_ROBOFLOW_API_ERRORS_RETRY_INTERVAL": "2",
            "RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API": "true",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
