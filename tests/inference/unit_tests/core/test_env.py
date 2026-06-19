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
