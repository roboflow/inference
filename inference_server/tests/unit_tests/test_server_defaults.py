import json
import os
import subprocess
import sys

import pytest

from inference_server import configuration, server_defaults
from inference_server.server_defaults import (
    HOST_ACCESS_SETTINGS,
    SERVER_DEFAULTS,
    apply_server_defaults,
)

_ALL_NAMES = [
    *SERVER_DEFAULTS,
    *HOST_ACCESS_SETTINGS,
    "LAMBDA",
    "GCP_SERVERLESS",
    "DEDICATED_DEPLOYMENT_WORKSPACE_URL",
    "OFFLINE_MODE",
    "HOST",
    "PUBLISHED_HOST",
    "ROBOFLOW_API_KEY",
    "API_KEY",
    "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT",
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
    monkeypatch.chdir(tmp_path)


def _set(monkeypatch, env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)


def test_server_is_locked_down_by_default() -> None:
    apply_server_defaults()

    assert {name: os.environ[name] for name in SERVER_DEFAULTS} == {
        "REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT": "True",
        "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS": "False",
        "ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM": "False",
        "ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES": "False",
        "ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE": "False",
        "ALLOW_URL_TO_NON_GLOBAL_ADDRESSES": "False",
        "VALIDATE_IMAGE_URL_REDIRECTS": "True",
        "ALLOW_WEBHOOK_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES": "False",
    }


def test_server_defaults_win_over_the_permissive_legacy_defaults() -> None:
    from inference_server.legacy_env import apply_legacy_env

    apply_server_defaults()
    apply_legacy_env()

    assert os.environ["ALLOW_URL_TO_NON_GLOBAL_ADDRESSES"] == "False"
    assert os.environ["ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM"] == "False"


def test_explicit_environment_wins(monkeypatch) -> None:
    _set(
        monkeypatch,
        {
            "HOST": "127.0.0.1",
            "ROBOFLOW_API_KEY": "some-key",
            "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS": "True",
            "REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT": "False",
        },
    )

    apply_server_defaults()

    assert os.environ["ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS"] == "True"
    assert os.environ["REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT"] == "False"


@pytest.mark.parametrize(
    "name,value",
    [
        ("LAMBDA", "True"),
        ("GCP_SERVERLESS", "1"),
        ("DEDICATED_DEPLOYMENT_WORKSPACE_URL", "some-workspace"),
    ],
)
def test_roboflow_hosted_runtimes_keep_their_defaults(monkeypatch, name, value) -> None:
    monkeypatch.setenv(name, value)
    monkeypatch.setenv("ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS", "True")

    apply_server_defaults()

    assert [n for n in SERVER_DEFAULTS if n in os.environ] == [
        "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS"
    ]


def test_offline_mode_does_not_require_api_key(monkeypatch) -> None:
    monkeypatch.setenv("OFFLINE_MODE", "yes")

    apply_server_defaults()

    assert "REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT" not in os.environ
    assert os.environ["ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS"] == "False"


@pytest.mark.parametrize(
    "env,api_key_required",
    [
        ({}, True),
        ({"HOST": "0.0.0.0"}, True),
        ({"HOST": "127.0.0.1"}, False),
        ({"HOST": "127.1.2.3"}, False),
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
        (
            {"HOST": "127.0.0.1", "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT": " , "},
            False,
        ),
        ({"HOST": "192.168.1.10"}, True),
        ({"HOST": "localhost.evil.example"}, True),
    ],
)
def test_api_key_is_required_unless_only_local_callers_without_a_workspace(
    monkeypatch, env, api_key_required
) -> None:
    _set(monkeypatch, env)

    apply_server_defaults()

    assert ("REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT" in os.environ) is api_key_required
    assert os.environ["ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS"] == "False"


@pytest.mark.parametrize("setting", HOST_ACCESS_SETTINGS)
@pytest.mark.parametrize(
    "env",
    [
        {"OFFLINE_MODE": "True"},
        {"REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT": "False"},
        {"ROBOFLOW_API_KEY": "k", "REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT": "no"},
        # A container published on all interfaces.
        {
            "HOST": "127.0.0.1",
            "PUBLISHED_HOST": "0.0.0.0",
            "REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT": "False",
        },
        # configuration.py parses this to an empty whitelist.
        {
            "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT": " , ",
            "REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT": "False",
        },
    ],
)
@pytest.mark.parametrize("value", ["true", "1", "on"])
def test_exposed_server_without_key_checks_refuses_host_access_settings(
    monkeypatch, env, setting, value
) -> None:
    _set(monkeypatch, env)
    monkeypatch.setenv(setting, value)

    with pytest.raises(SystemExit, match=setting):
        apply_server_defaults()


@pytest.mark.parametrize(
    "env",
    [
        # Loopback: only local callers.
        {"HOST": "127.0.0.1", "REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT": "False"},
        # Key checks on.
        {"ROBOFLOW_API_KEY": "some-key"},
        {},
        {
            "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT": "some-ws",
            "REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT": "False",
        },
    ],
)
def test_host_access_settings_are_allowed_locally_or_behind_key_checks(
    monkeypatch, env
) -> None:
    _set(monkeypatch, env)
    monkeypatch.setenv("ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS", "True")

    apply_server_defaults()

    assert os.environ["ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS"] == "True"


def test_dot_env_file_is_honoured(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PUBLISHED_HOST", "127.0.0.1")
    (tmp_path / ".env").write_text(
        "OFFLINE_MODE=True\nROBOFLOW_API_KEY=some-key\n"
        "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS=True\n"
    )

    apply_server_defaults()

    assert "REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT" not in os.environ
    assert os.environ["ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS"] == "True"
    assert os.environ["ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM"] == "False"


def test_dot_env_key_counts_as_a_workspace_on_loopback(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOST", "127.0.0.1")
    (tmp_path / ".env").write_text("ROBOFLOW_API_KEY=some-key\n")

    apply_server_defaults()

    assert os.environ["REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT"] == "True"


@pytest.mark.parametrize("name", ["HOST", "PUBLISHED_HOST"])
def test_dot_env_host_does_not_make_the_server_loopback(
    monkeypatch, tmp_path, name
) -> None:
    # run_uvicorn.sh binds before ./.env is read, but importing inference_models
    # loads ./.env into the environment first.
    monkeypatch.setenv("HOST", "127.0.0.1")
    (tmp_path / ".env").write_text(f"{name}=127.0.0.1\n")

    apply_server_defaults()

    assert os.environ["REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT"] == "True"
    assert server_defaults.is_loopback_only() is False


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, False),
        ({"REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT": True}, True),
        ({"REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT": True, "LAMBDA": True}, False),
        ({"REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT": True, "GCP_SERVERLESS": True}, False),
        ({"WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT": ["ws"]}, True),
        ({"WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT": []}, False),
        ({"DEDICATED_DEPLOYMENT_WORKSPACE_URL": "ws"}, True),
    ],
)
def test_workspace_auth_enabled(monkeypatch, overrides, expected) -> None:
    base = {
        "REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT": False,
        "LAMBDA": False,
        "GCP_SERVERLESS": False,
        "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT": None,
        "DEDICATED_DEPLOYMENT_WORKSPACE_URL": None,
    }
    for name, value in {**base, **overrides}.items():
        monkeypatch.setattr(configuration, name, value)

    assert server_defaults.workspace_auth_enabled() is expected


@pytest.mark.parametrize(
    "host,overrides,expected",
    [
        ("127.0.0.1", {}, True),
        ("0.0.0.0", {}, False),
        ("127.0.0.1", {"REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT": True}, False),
        ("127.0.0.1", {"GCP_SERVERLESS": True}, False),
        ("127.0.0.1", {"LAMBDA": True}, False),
    ],
)
def test_local_callers_only(monkeypatch, host, overrides, expected) -> None:
    monkeypatch.setenv("HOST", host)
    base = {
        "REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT": False,
        "LAMBDA": False,
        "GCP_SERVERLESS": False,
        "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT": None,
        "DEDICATED_DEPLOYMENT_WORKSPACE_URL": None,
    }
    for name, value in {**base, **overrides}.items():
        monkeypatch.setattr(configuration, name, value)

    assert server_defaults.local_callers_only() is expected


_PASSED_THROUGH = ("PATH", "HOME", "TMPDIR", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def _clean_subprocess_env() -> dict:
    # Only what Python needs: settings leaked into os.environ by earlier tests
    # must not decide what the server sees.
    env = {k: os.environ[k] for k in _PASSED_THROUGH if k in os.environ}
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    env["HOST"] = "0.0.0.0"

    return env


def test_entrypoint_applies_server_defaults_before_configuration_is_read(
    tmp_path,
) -> None:
    # Catches a misspelled name or a late hook, either of which would silently
    # leave a protection off in the shipped image.
    script = (
        "import json\n"
        "import inference_server.entrypoints.cpu_http\n"
        "from inference_server import configuration as c\n"
        "from inference_server.app import app\n"
        "from inference_server.workflows.host import SERVER_WORKFLOWS_CONFIGURATION"
        " as w\n"
        "e = w.engine\n"
        "print(json.dumps([c.REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT,"
        " c.ALLOW_URL_TO_NON_GLOBAL_ADDRESSES,"
        " c.ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM,"
        " c.VALIDATE_IMAGE_URL_REDIRECTS,"
        " e.allow_custom_python_execution,"
        " e.allow_blocks_accessing_local_storage,"
        " e.allow_blocks_accessing_environmental_variables,"
        " e.allow_webhook_sink_to_non_global_addresses,"
        " [m.cls.__name__ for m in app.user_middleware]]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=_clean_subprocess_env(),
        timeout=300,
    )

    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout.strip().splitlines()[-1])
    assert values[:8] == [True, False, False, True, False, False, False, False]
    assert "DedicatedAuthMiddleware" in values[8]


@pytest.mark.parametrize(
    "command",
    [
        ["-c", "import inference_server.entrypoints.gpu_http"],
        ["-m", "inference_server.app"],
    ],
)
def test_server_entry_refuses_to_start_exposed_with_host_access(
    tmp_path, command
) -> None:
    env = {
        **_clean_subprocess_env(),
        "REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT": "False",
        "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS": "True",
    }
    result = subprocess.run(
        [sys.executable, *command],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        timeout=300,
    )

    assert result.returncode == 1
    assert "Refusing to start: ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS" in (
        result.stderr
    )
