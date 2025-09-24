import os
import subprocess
import sys

import pytest

from inference_cli.lib.container_adapter import (
    find_running_inference_containers,
    terminate_running_containers,
)

PYTHON_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
) + ":".join(sys.path)


@pytest.mark.timeout(600)
def test_server_start() -> None:
    # when
    result = subprocess.run(f"python -m inference_cli.main server start".split())
    running_containers = find_running_inference_containers()

    # then
    try:
        assert result.returncode == 0
        assert len(running_containers) == 1
    finally:
        terminate_running_containers(
            containers=running_containers, interactive_mode=False
        )


@pytest.mark.timeout(600)
def test_server_start_when_env_file_is_pointed(example_env_file_path: str) -> None:
    # when
    result = subprocess.run(
        f"python -m inference_cli.main server start -e {example_env_file_path}".split()
    )
    running_containers = find_running_inference_containers()

    # then
    try:
        assert result.returncode == 0
        assert len(running_containers) == 1
        container_env_variables = running_containers[0].attrs["Config"]["Env"]
        assert (
            "SOME=MY_VALUE" in container_env_variables
        ), "Custom env variable defined in file is exported"
        assert (
            "API_KEY=MY_API_KEY" in container_env_variables
        ), "API key defined in env file is exported"
        assert (
            "PORT=9001" in container_env_variables
        ), "Explicitly defined port value is over-shadowing the env file"
    finally:
        terminate_running_containers(
            containers=running_containers, interactive_mode=False
        )


@pytest.mark.timeout(600)
def test_server_stop_when_no_inference_containers_are_running() -> None:
    # when
    result = subprocess.run(f"python -m inference_cli.main server start".split())

    # then
    assert result.returncode == 0


@pytest.mark.timeout(600)
def test_server_stop_when_inference_container_is_running() -> None:
    # given
    subprocess.run(f"python -m inference_cli.main server start".split())
    running_containers = find_running_inference_containers()
    assert len(running_containers) > 0

    # when
    result = subprocess.run(f"python -m inference_cli.main server stop".split())
    running_containers = find_running_inference_containers()

    # then
    try:
        assert result.returncode == 0
        assert len(running_containers) == 0
    finally:
        terminate_running_containers(
            containers=running_containers, interactive_mode=False
        )
