"""WP-A03: every server launcher starts the stream manager through the bootstrap.

Each launcher module is executed for real, with only its HTTP-server half
stubbed, and its stream manager process is spawned: explicitly by the GPU
launcher, by forcing the default start method for the other three - the
default on macOS and Windows, where the app bundles run. The spawned manager
must install its configuration and host descriptor before it imports the
manager app or any runtime module the app imports; see `test_host_spawn.py`
for how the import order is recorded.
"""

import socket
from pathlib import Path

import pytest

from tests.inference.unit_tests.core.interfaces.stream_manager.manager_app.test_host_spawn import (
    PIPELINE_RUNTIME_MODULES,
    REPO_ROOT,
    run_spawn_driver,
)

MANAGER_APP_MODULE = "inference.core.interfaces.stream_manager.manager_app.app"

_LAUNCHER_DRIVER = """
import json
import os


def _stub_module(name):
    import types
    from unittest.mock import MagicMock

    module = types.ModuleType(name)
    module.__getattr__ = lambda attribute: MagicMock(name=f"{name}.{attribute}")
    return module


def _wait_until_listening(process, port):
    import socket
    import time

    deadline = time.monotonic() + 240
    while time.monotonic() < deadline and process.is_alive():
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def main():
    import multiprocessing
    import runpy
    import sys

    if os.environ["FORCE_SPAWN"] == "1":
        multiprocessing.set_start_method("spawn", force=True)
    # Only the stream manager launch is under test: the model manager and HTTP
    # interface the launcher also builds are replaced.
    for name in (
        "inference.core.cache",
        "inference.core.interfaces.http.http_api",
        "inference.core.managers.active_learning",
        "inference.core.managers.base",
        "inference.core.managers.decorators.fixed_size_cache",
        "inference.core.registries.roboflow",
        "inference.models.utils",
    ):
        sys.modules[name] = _stub_module(name)

    from inference.core.interfaces.streams_configuration import (
        server_streams_configuration,
    )

    namespace = runpy.run_path(os.environ["LAUNCHER_PATH"], run_name="launcher")
    process = namespace["stream_manager_process"]
    listening = _wait_until_listening(process, int(os.environ["STREAM_MANAGER_PORT"]))
    process.terminate()
    process.join(timeout=60)
    result = {
        "driver_pid": os.getpid(),
        "manager_pid": process.pid,
        "listening": listening,
        "exit_code": process.exitcode,
        "configuration": repr(server_streams_configuration()),
    }
    with open(os.environ["SPAWN_DRIVER_RESULT"], "w") as result_file:
        json.dump(result, result_file)


if __name__ == "__main__":
    main()
"""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "launcher, force_spawn",
    [
        ("docker/config/cpu_http.py", True),
        ("docker/config/gpu_http.py", False),
        ("app_bundles/osx/cpu_http.py", True),
        ("app_bundles/windows/cpu_http.py", True),
    ],
)
def test_launcher_installs_settings_before_the_manager_app_is_imported(
    launcher: str,
    force_spawn: bool,
    tmp_path: Path,
) -> None:
    result, import_records = run_spawn_driver(
        _LAUNCHER_DRIVER,
        tmp_path=tmp_path,
        watched_modules=(MANAGER_APP_MODULE, *PIPELINE_RUNTIME_MODULES),
        environment={
            "LAUNCHER_PATH": str(REPO_ROOT / launcher),
            "FORCE_SPAWN": "1" if force_spawn else "0",
            "ENABLE_STREAM_API": "True",
            "STREAM_API_PRELOADED_PROCESSES": "0",
            "STREAM_MANAGER_HOST": "127.0.0.1",
            "STREAM_MANAGER_PORT": str(_free_port()),
        },
    )

    assert result["listening"] is True
    assert result["exit_code"] == 0
    manager_records = [
        record for record in import_records if record["pid"] == result["manager_pid"]
    ]
    # Recorded only because the manager was spawned: a forked child would have
    # inherited these modules instead of importing them.
    assert MANAGER_APP_MODULE in {record["module"] for record in manager_records}
    for record in manager_records:
        assert "bootstrap.py:run_stream_manager" in record["stack"], record
        assert "legacy_stream.host:LegacyPipelineHost" in record["descriptor"], record
        assert record["configuration"] == result["configuration"], record
