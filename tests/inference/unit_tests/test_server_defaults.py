import json
import os
import subprocess
import sys
from pathlib import Path

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
    "ALLOW_NUMPY_INPUT",
    "ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES",
    "ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES",
]

REPO_ROOT = Path(__file__).resolve().parents[3]

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
    assert os.environ["ALLOW_WEBHOOK_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES"] == "False"


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


def test_refusing_uvicorn_worker_stops_its_supervisor(monkeypatch) -> None:
    # uvicorn (0.30+) starts a spawned worker again when it exits.
    monkeypatch.setenv("REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT", "False")
    monkeypatch.setenv("ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS", "True")
    monkeypatch.setattr(server_defaults.multiprocessing, "parent_process", object)
    signals = []
    monkeypatch.setattr(server_defaults.os, "kill", lambda *args: signals.append(args))

    with pytest.raises(SystemExit, match="Refusing to start"):
        apply_self_hosted_server_defaults()

    assert signals == [(os.getppid(), server_defaults.signal.SIGTERM)]


def test_refusing_gunicorn_worker_exits_with_the_boot_error_code(
    monkeypatch, capsys
) -> None:
    # gunicorn forks its workers and halts only when one exits with code 3.
    monkeypatch.setenv("REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT", "False")
    monkeypatch.setenv("ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS", "True")
    monkeypatch.setenv("SERVER_SOFTWARE", "gunicorn/23.0.0")
    monkeypatch.setitem(sys.modules, "gunicorn.arbiter", object())

    def exit_now(code):
        raise SystemExit(code)

    monkeypatch.setattr(server_defaults.os, "_exit", exit_now)

    with pytest.raises(SystemExit) as refusal:
        apply_self_hosted_server_defaults()

    assert refusal.value.code == 3
    assert "Refusing to start" in capsys.readouterr().err


@pytest.mark.parametrize(
    "env",
    [{"OFFLINE_MODE": "True"}, {"REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT": "False"}],
)
def test_exposed_server_without_key_checks_warns(monkeypatch, capsys, env) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    apply_self_hosted_server_defaults()

    stderr = capsys.readouterr().err
    assert stderr.startswith("SECURITY:")
    assert "anyone who can reach its port can use every route" in stderr


@pytest.mark.parametrize(
    "env",
    [
        {"HOST": "127.0.0.1", "REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT": "False"},
        {"ROBOFLOW_API_KEY": "some-key"},
        {"OFFLINE_MODE": "True", "LAMBDA": "True"},
    ],
)
def test_local_or_key_checked_server_does_not_warn(monkeypatch, capsys, env) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    apply_self_hosted_server_defaults()

    assert "SECURITY:" not in capsys.readouterr().err


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
        " env.VALIDATE_IMAGE_URL_REDIRECTS,"
        " env.ALLOW_WEBHOOK_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES)\n"
        "from roboflow_workflows.core_steps.sinks.webhook import v1\n"
        "print(v1.ALLOW_WEBHOOK_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )

    assert result.stdout.split()[-9:] == [
        "True",
        "False",
        "False",
        "False",
        "False",
        "False",
        "True",
        "False",
        # The webhook sink block reads it too.
        "False",
    ]


# Stops the import of inference.core.env and reports the environment it would
# have read. Missing optional packages (celery) are stubbed; nothing past the
# first inference.core.env import runs.
_ENTRY_MODULE_SCRIPT = """
import importlib
import importlib.abc
import importlib.util
import json
import os
import runpy
import sys
from unittest.mock import MagicMock


class EnvImported(BaseException):
    pass


class StopAtEnv(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "inference.core.env":
            raise EnvImported(
                os.environ.get("ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS")
            )
        return None


for optional in ("celery",):
    if importlib.util.find_spec(optional) is None:
        sys.modules[optional] = MagicMock()
sys.meta_path.insert(0, StopAtEnv())
kind, target = sys.argv[1], sys.argv[2]
try:
    if kind == "module":
        importlib.import_module(target)
    else:
        runpy.run_path(target, run_name="entry_module_under_test")
except EnvImported as stop:
    result = {"env_imported": True, "value": stop.args[0]}
else:
    result = {
        "env_imported": False,
        "value": os.environ.get("ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS"),
    }
print(json.dumps(result))
"""


@pytest.mark.parametrize(
    "kind,target,imports_env",
    [
        ("path", "docker/config/cpu_http.py", True),
        ("path", "docker/config/gpu_http.py", True),
        ("path", "inference/enterprise/parallel/parallel_http_config.py", True),
        # Started with `celery -A inference.enterprise.parallel.tasks`.
        ("module", "inference.enterprise.parallel.tasks", True),
        # Started with `python3 inference/enterprise/parallel/infer.py`.
        ("path", "inference/enterprise/parallel/infer.py", True),
        # These import the server only under __main__, after the module body.
        ("path", "app_bundles/osx/run_inference.py", False),
        ("path", "app_bundles/windows/run_inference.py", False),
    ],
)
def test_entry_modules_apply_defaults_before_reading_inference_core_env(
    tmp_path, kind, target, imports_env
) -> None:
    if kind == "path":
        target = str(REPO_ROOT / target)
    result = subprocess.run(
        [sys.executable, "-c", _ENTRY_MODULE_SCRIPT, kind, target],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(sys.path),
        },
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "env_imported": imports_env,
        "value": "False",
    }
