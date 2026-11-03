import os

import pytest

from inference.server_defaults import (
    SELF_HOSTED_SERVER_DEFAULTS,
    apply_self_hosted_server_defaults,
)

_ALL_NAMES = [
    *SELF_HOSTED_SERVER_DEFAULTS,
    "LAMBDA",
    "GCP_SERVERLESS",
    "DEDICATED_DEPLOYMENT_WORKSPACE_URL",
    "OFFLINE_MODE",
]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in _ALL_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_self_hosted_server_is_locked_down_by_default() -> None:
    apply_self_hosted_server_defaults()

    assert os.environ["REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT"] == "True"
    assert os.environ["ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS"] == "False"
    assert os.environ["ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM"] == "False"
    assert (
        os.environ["ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES"] == "False"
    )
    assert os.environ["ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE"] == "False"
    assert os.environ["ALLOW_URL_TO_NON_GLOBAL_ADDRESSES"] == "False"
    assert os.environ["VALIDATE_IMAGE_URL_REDIRECTS"] == "True"


def test_explicit_environment_wins(monkeypatch) -> None:
    monkeypatch.setenv("ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS", "True")
    monkeypatch.setenv("REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT", "False")

    apply_self_hosted_server_defaults()

    assert os.environ["ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS"] == "True"
    assert os.environ["REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT"] == "False"


@pytest.mark.parametrize(
    "name,value",
    [
        ("LAMBDA", "True"),
        ("GCP_SERVERLESS", "true"),
        ("DEDICATED_DEPLOYMENT_WORKSPACE_URL", "some-workspace"),
    ],
)
def test_roboflow_hosted_runtimes_keep_their_defaults(monkeypatch, name, value) -> None:
    monkeypatch.setenv(name, value)

    apply_self_hosted_server_defaults()

    for default_name in SELF_HOSTED_SERVER_DEFAULTS:
        assert default_name not in os.environ


def test_offline_mode_does_not_require_api_key(monkeypatch) -> None:
    monkeypatch.setenv("OFFLINE_MODE", "True")

    apply_self_hosted_server_defaults()

    assert "REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT" not in os.environ
    assert os.environ["ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS"] == "False"
