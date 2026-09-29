import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from streamvision.__main__ import build_parser

PACKAGE_ROOT = Path(__file__).resolve().parents[2]

FAKE_HOST = """
import json
import os
import sys

import streamvision.stream.configuration as configuration

observations = {
    "runtime_imported": "streamvision.stream_manager.manager_app.app" in sys.modules,
    "configuration_installed": configuration._CONFIGURATION is not None,
}
with open(os.environ["FAKE_HOST_OBSERVATIONS"], "w") as file:
    json.dump(observations, file)


def create_host(**settings):
    raise AssertionError("no pipeline is started in this test")
"""

BLOCK_AIORTC = 'import sys\nsys.modules["aiortc"] = None\n'

BLOCK_ROBOFLOW_WORKFLOWS = 'import sys\nsys.modules["roboflow_workflows"] = None\n'

FAKE_HOST_WITH_CONFIG = """
import os

from streamvision.stream.configuration import StreamsConfiguration, configure_process

configure_process(
    StreamsConfiguration(stream_manager_port=int(os.environ["FAKE_HOST_CONFIG_PORT"]))
)


def create_host(**settings):
    raise AssertionError("no pipeline is started in this test")
"""


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    return port


def _wait_for_port(port: int, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        assert process.poll() is None, process.communicate()[1]
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.2)

    raise AssertionError(f"stream manager did not listen on {port} in {timeout}s")


def test_parser_parses_all_options() -> None:
    # when
    args = build_parser().parse_args(
        [
            "--host-factory",
            "my_host:create_host",
            "--host-setting",
            "api_key=abc",
            "--host-setting",
            "mode=x=y",
            "--warm-pipelines",
            "2",
        ]
    )

    # then
    assert args.host_factory == "my_host:create_host"
    assert dict(args.host_setting) == {"api_key": "abc", "mode": "x=y"}
    assert args.warm_pipelines == 2


def test_parser_defaults_and_rejects_malformed_setting() -> None:
    # when
    args = build_parser().parse_args(["--host-factory", "my_host:create_host"])

    # then
    assert args.host_setting == []
    assert args.warm_pipelines == 0
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--host-factory", "h:f", "--host-setting", "x"])


def test_parser_rejects_negative_or_non_integer_warm_pipelines() -> None:
    # when / then
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--host-factory", "h:f", "--warm-pipelines", "-1"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--host-factory", "h:f", "--warm-pipelines", "abc"])


def test_launcher_imports_host_before_runtime_and_serves_until_sigterm(
    tmp_path: Path,
) -> None:
    # given
    host_directory = tmp_path / "host"
    host_directory.mkdir()
    (host_directory / "fake_launcher_host.py").write_text(FAKE_HOST)
    observations_path = tmp_path / "observations.json"
    port = _free_port()
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [
                str(host_directory),
                str(PACKAGE_ROOT),
                *filter(None, [os.environ.get("PYTHONPATH")]),
            ]
        ),
        "STREAM_MANAGER_HOST": "127.0.0.1",
        "STREAM_MANAGER_PORT": str(port),
        "FAKE_HOST_OBSERVATIONS": str(observations_path),
    }

    # when
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamvision",
            "--host-factory",
            "fake_launcher_host:create_host",
        ],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_port(port, process=process, timeout=60)
        process.send_signal(signal.SIGTERM)
        _, stderr = process.communicate(timeout=15)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()

    # then
    assert process.returncode == 0, stderr
    assert json.loads(observations_path.read_text()) == {
        "runtime_imported": False,
        "configuration_installed": False,
    }


def test_launcher_preserves_host_installed_configuration(tmp_path: Path) -> None:
    # given
    host_directory = tmp_path / "host_with_config"
    host_directory.mkdir()
    (host_directory / "fake_config_host.py").write_text(FAKE_HOST_WITH_CONFIG)
    port = _free_port()
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [
                str(host_directory),
                str(PACKAGE_ROOT),
                *filter(None, [os.environ.get("PYTHONPATH")]),
            ]
        ),
        "FAKE_HOST_CONFIG_PORT": str(port),
    }
    environment.pop("STREAM_MANAGER_PORT", None)

    # when
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamvision",
            "--host-factory",
            "fake_config_host:create_host",
        ],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_port(port, process=process, timeout=60)
        process.send_signal(signal.SIGTERM)
        _, stderr = process.communicate(timeout=15)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()

    # then
    assert process.returncode == 0, stderr


def test_launcher_exits_when_the_webrtc_extra_is_missing(tmp_path: Path) -> None:
    # given
    block_directory = tmp_path / "block"
    block_directory.mkdir()
    (block_directory / "sitecustomize.py").write_text(BLOCK_AIORTC)
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [
                str(block_directory),
                str(PACKAGE_ROOT),
                *filter(None, [os.environ.get("PYTHONPATH")]),
            ]
        ),
    }

    # when
    result = subprocess.run(
        [sys.executable, "-m", "streamvision", "--host-factory", "x.y:z"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    # then
    assert result.returncode != 0
    assert "streamvision[webrtc]" in result.stderr


def test_launcher_exits_when_the_workflows_extra_is_missing(tmp_path: Path) -> None:
    # given
    block_directory = tmp_path / "block"
    block_directory.mkdir()
    (block_directory / "sitecustomize.py").write_text(BLOCK_ROBOFLOW_WORKFLOWS)
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [
                str(block_directory),
                str(PACKAGE_ROOT),
                *filter(None, [os.environ.get("PYTHONPATH")]),
            ]
        ),
    }

    # when
    result = subprocess.run(
        [sys.executable, "-m", "streamvision", "--host-factory", "x.y:z"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    # then
    assert result.returncode != 0
    assert "streamvision[webrtc,workflows]" in result.stderr
