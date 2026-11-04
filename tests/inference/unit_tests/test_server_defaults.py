import os
import subprocess
import sys

import pytest

from inference import server_defaults
from inference.server_defaults import (
    SELF_HOSTED_SERVER_DEFAULTS,
    apply_self_hosted_server_defaults,
)

HOST_ACCESS_SETTINGS = [
    "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS",
    "ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM",
    "ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES",
    "ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE",
    "ALLOW_UNSAFE_GSTREAMER_PIPELINES",
]

_ALL_NAMES = [
    *SELF_HOSTED_SERVER_DEFAULTS,
    "LAMBDA",
    "GCP_SERVERLESS",
    "DEDICATED_DEPLOYMENT_WORKSPACE_URL",
    "OFFLINE_MODE",
    "HOST",
    "PUBLISHED_HOST",
    "ROBOFLOW_API_KEY",
    "API_KEY",
    "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT",
    *HOST_ACCESS_SETTINGS,
]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch, tmp_path):
    # The code under test writes os.environ directly (setdefault, load_dotenv);
    # a copy keeps those writes out of later tests.
    monkeypatch.setattr(
        os,
        "environ",
        {k: v for k, v in os.environ.items() if k not in _ALL_NAMES},
    )
    monkeypatch.setattr(server_defaults, "LOOPBACK_ONLY", False)
    monkeypatch.chdir(tmp_path)


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
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("ROBOFLOW_API_KEY", "some-key")
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


@pytest.mark.parametrize(
    "env,api_key_required",
    [
        ({"HOST": "127.0.0.1"}, False),
        ({"HOST": "localhost"}, False),
        ({"HOST": "::1"}, False),
        # A container binds 0.0.0.0 but may be published on loopback only.
        ({"HOST": "0.0.0.0", "PUBLISHED_HOST": "127.0.0.1"}, False),
        ({"HOST": "127.0.0.1", "PUBLISHED_HOST": "0.0.0.0"}, True),
        ({"HOST": "127.0.0.1", "ROBOFLOW_API_KEY": "some-key"}, True),
        ({"HOST": "127.0.0.1", "API_KEY": "some-key"}, True),
        (
            {"HOST": "127.0.0.1", "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT": "ws"},
            True,
        ),
        ({"HOST": "192.168.1.10"}, True),
    ],
)
def test_api_key_is_required_unless_only_local_callers_without_a_workspace(
    monkeypatch, env, api_key_required
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    apply_self_hosted_server_defaults()

    assert ("REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT" in os.environ) is api_key_required
    assert os.environ["ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS"] == "False"


@pytest.mark.parametrize("setting", HOST_ACCESS_SETTINGS)
@pytest.mark.parametrize(
    "env",
    [
        {"OFFLINE_MODE": "True"},
        {"REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT": "False"},
        # A container published on all interfaces.
        {
            "HOST": "127.0.0.1",
            "PUBLISHED_HOST": "0.0.0.0",
            "REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT": "False",
        },
        # inference.core.env parses this to an empty whitelist.
        {
            "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT": " , ",
            "REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT": "False",
        },
    ],
)
def test_exposed_server_without_key_checks_refuses_host_access_settings(
    monkeypatch, env, setting
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(setting, "true")

    with pytest.raises(SystemExit, match=setting):
        apply_self_hosted_server_defaults()


@pytest.mark.parametrize(
    "env",
    [
        # Loopback: only local callers.
        {"HOST": "127.0.0.1", "REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT": "False"},
        # Key checks on.
        {"ROBOFLOW_API_KEY": "some-key"},
        {
            "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT": "some-ws",
            "REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT": "False",
        },
    ],
)
def test_host_access_settings_are_allowed_locally_or_behind_key_checks(
    monkeypatch, env
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS", "True")

    apply_self_hosted_server_defaults()

    assert os.environ["ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS"] == "True"


def test_dot_env_file_is_honoured(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PUBLISHED_HOST", "127.0.0.1")
    (tmp_path / ".env").write_text(
        "OFFLINE_MODE=True\nROBOFLOW_API_KEY=some-key\n"
        "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS=True\n"
    )

    apply_self_hosted_server_defaults()

    assert "REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT" not in os.environ
    assert os.environ["ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS"] == "True"
    assert os.environ["ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM"] == "False"


def test_dot_env_host_does_not_make_the_server_loopback(monkeypatch, tmp_path) -> None:
    # Launchers such as run_uvicorn.sh bind before ./.env is read, but
    # importing inference loads ./.env into the environment first.
    monkeypatch.setenv("HOST", "127.0.0.1")
    (tmp_path / ".env").write_text("HOST=127.0.0.1\n")

    apply_self_hosted_server_defaults()

    assert os.environ["REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT"] == "True"
    assert server_defaults.LOOPBACK_ONLY is False


def test_defaults_reach_inference_core_env() -> None:
    # Catches a misspelled name, which would silently leave a protection off.
    script = (
        "from inference.server_defaults import apply_self_hosted_server_defaults\n"
        "apply_self_hosted_server_defaults()\n"
        "from inference.core import env\n"
        "print(env.REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT,"
        " env.ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS,"
        " env.ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM,"
        " env.ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES,"
        " env.ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE,"
        " env.ALLOW_URL_TO_NON_GLOBAL_ADDRESSES,"
        " env.VALIDATE_IMAGE_URL_REDIRECTS)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )

    assert result.stdout.split()[-7:] == [
        "True",
        "False",
        "False",
        "False",
        "False",
        "False",
        "True",
    ]
