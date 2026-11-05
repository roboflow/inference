from unittest import mock
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from inference_cli import server
from inference_cli.lib import container_adapter
from inference_cli.lib.container_adapter import (
    announce_authentication,
    prepare_container_environment,
    start_inference_container,
)


@mock.patch.object(container_adapter, "read_env_file")
def test_prepare_container_environment_when_env_file_defined(
    read_env_file_mock: MagicMock,
) -> None:
    # given
    read_env_file_mock.return_value = {
        "API_KEY": "my_api_key",
        "DEVICE_ID": "other",
        "SOME": "VALUE",
    }

    # when
    result = prepare_container_environment(
        port=8080,
        project="roboflow-platform",
        metrics_enabled=False,
        device_id="some",
        num_workers=3,
        api_key=None,
        env_file_path="my_env_file",
    )

    # then
    assert sorted(result) == sorted(
        [
            "HOST=0.0.0.0",
            "PORT=8080",
            "PROJECT=roboflow-platform",
            "METRICS_ENABLED=False",
            "DEVICE_ID=some",
            "NUM_WORKERS=3",
            "API_KEY=my_api_key",
            "SOME=VALUE",
        ]
    )
    read_env_file_mock.assert_called_once_with(path="my_env_file")


@mock.patch.object(container_adapter, "pull_image")
@mock.patch.object(
    container_adapter, "find_running_inference_containers", return_value=[]
)
@mock.patch.object(container_adapter, "docker")
def test_start_inference_container_default_tmp_volume_always_present(
    docker_mock: MagicMock,
    _find_containers_mock: MagicMock,
    _pull_image_mock: MagicMock,
) -> None:
    # when
    start_inference_container(image="roboflow/roboflow-inference-server-cpu:latest")

    # then
    _, kwargs = docker_mock.from_env.return_value.containers.run.call_args
    assert "/tmp" in kwargs["volumes"]
    assert kwargs["volumes"]["/tmp"] == {"bind": "/tmp", "mode": "rw"}


@mock.patch.object(container_adapter, "pull_image")
@mock.patch.object(
    container_adapter, "find_running_inference_containers", return_value=[]
)
@mock.patch.object(container_adapter, "docker")
def test_start_inference_container_user_volumes_merged_with_tmp(
    docker_mock: MagicMock,
    _find_containers_mock: MagicMock,
    _pull_image_mock: MagicMock,
) -> None:
    # given
    user_volumes = {"/home/user/data": {"bind": "/data", "mode": "rw"}}

    # when
    start_inference_container(
        image="roboflow/roboflow-inference-server-cpu:latest",
        volumes=user_volumes,
    )

    # then
    _, kwargs = docker_mock.from_env.return_value.containers.run.call_args
    assert "/tmp" in kwargs["volumes"]
    assert "/home/user/data" in kwargs["volumes"]
    assert kwargs["volumes"]["/home/user/data"] == {"bind": "/data", "mode": "rw"}


def test_prepare_container_environment_when_env_file_not_defined() -> None:
    # when
    result = prepare_container_environment(
        port=8080,
        project="roboflow-platform",
        metrics_enabled=False,
        device_id="some",
        num_workers=3,
        api_key=None,
        env_file_path=None,
    )

    # then
    assert sorted(result) == sorted(
        [
            "HOST=0.0.0.0",
            "PORT=8080",
            "PROJECT=roboflow-platform",
            "METRICS_ENABLED=False",
            "DEVICE_ID=some",
            "NUM_WORKERS=3",
        ]
    )


@mock.patch.object(container_adapter, "pull_image")
@mock.patch.object(
    container_adapter, "find_running_inference_containers", return_value=[]
)
@mock.patch.object(container_adapter, "docker")
def test_start_inference_container_publishes_ports_on_loopback_by_default(
    docker_mock: MagicMock,
    _find_containers_mock: MagicMock,
    _pull_image_mock: MagicMock,
) -> None:
    # when
    start_inference_container(
        image="roboflow/roboflow-inference-server-cpu:latest",
        port=9001,
        development=True,
    )

    # then
    _, kwargs = docker_mock.from_env.return_value.containers.run.call_args
    assert kwargs["ports"] == {"9001": ("127.0.0.1", 9001), "9002": ("127.0.0.1", 9002)}
    # The server decides from this whether it may run without API key checks.
    assert "PUBLISHED_HOST=127.0.0.1" in kwargs["environment"]


@mock.patch.object(container_adapter, "pull_image")
@mock.patch.object(
    container_adapter, "find_running_inference_containers", return_value=[]
)
@mock.patch.object(container_adapter, "docker")
def test_start_inference_container_publishes_ports_on_all_interfaces_for_jetson(
    docker_mock: MagicMock,
    _find_containers_mock: MagicMock,
    _pull_image_mock: MagicMock,
) -> None:
    # when
    start_inference_container(
        image="roboflow/roboflow-inference-server-jetson-6.2.0:latest",
        port=9001,
    )

    # then
    _, kwargs = docker_mock.from_env.return_value.containers.run.call_args
    assert kwargs["ports"] == {"9001": ("0.0.0.0", 9001)}


@mock.patch.object(container_adapter, "pull_image")
@mock.patch.object(
    container_adapter, "find_running_inference_containers", return_value=[]
)
@mock.patch.object(container_adapter, "docker")
def test_start_inference_container_bind_address_overrides_jetson_default(
    docker_mock: MagicMock,
    _find_containers_mock: MagicMock,
    _pull_image_mock: MagicMock,
) -> None:
    # when
    start_inference_container(
        image="roboflow/roboflow-inference-server-jetson-6.2.0:latest",
        port=9001,
        bind_address="127.0.0.1",
    )

    # then
    _, kwargs = docker_mock.from_env.return_value.containers.run.call_args
    assert kwargs["ports"] == {"9001": ("127.0.0.1", 9001)}


@mock.patch.object(container_adapter, "pull_image")
@mock.patch.object(
    container_adapter, "find_running_inference_containers", return_value=[]
)
@mock.patch.object(container_adapter, "docker")
def test_start_inference_container_publishes_ports_on_requested_bind_address(
    docker_mock: MagicMock,
    _find_containers_mock: MagicMock,
    _pull_image_mock: MagicMock,
) -> None:
    # when
    start_inference_container(
        image="roboflow/roboflow-inference-server-cpu:latest",
        port=9001,
        bind_address="0.0.0.0",
    )

    # then
    _, kwargs = docker_mock.from_env.return_value.containers.run.call_args
    assert kwargs["ports"] == {"9001": ("0.0.0.0", 9001)}
    assert "PUBLISHED_HOST=0.0.0.0" in kwargs["environment"]


def test_development_environment_uses_a_random_notebook_token() -> None:
    def notebook_tokens():
        environment = prepare_container_environment(
            port=9001,
            project="roboflow-platform",
            metrics_enabled=False,
            device_id=None,
            num_workers=1,
            api_key=None,
            env_file_path=None,
            development=True,
        )
        return [
            entry.split("=", 1)[1]
            for entry in environment
            if entry.startswith("NOTEBOOK_PASSWORD=")
        ]

    first, second = notebook_tokens(), notebook_tokens()

    assert len(first) == 1 and len(first[0]) >= 24
    assert first != second
    assert first[0] != "roboflow"


@mock.patch.object(
    container_adapter, "read_env_file", return_value={"NOTEBOOK_PASSWORD": "mine"}
)
def test_development_environment_keeps_notebook_token_from_env_file(_) -> None:
    environment = prepare_container_environment(
        port=9001,
        project="roboflow-platform",
        metrics_enabled=False,
        device_id=None,
        num_workers=1,
        api_key=None,
        env_file_path="some.env",
        development=True,
    )

    assert "NOTEBOOK_PASSWORD=mine" in environment


@pytest.mark.parametrize(
    "environment",
    [
        [],
        ["ROBOFLOW_API_KEY="],
        ["API_KEY=", "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT= , "],
        ["REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT=True"],
        ["REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT=true", "ROBOFLOW_API_KEY="],
        ["OFFLINE_MODE=False", "DEDICATED_DEPLOYMENT_WORKSPACE_URL="],
    ],
)
def test_announce_authentication_warns_when_exposed_server_has_no_workspace(
    environment: list, capsys
) -> None:
    announce_authentication(environment=environment, bind_address="0.0.0.0")

    assert "will reject every request" in capsys.readouterr().out


@pytest.mark.parametrize(
    "environment",
    [
        ["ROBOFLOW_API_KEY=key"],
        ["API_KEY=key"],
        ["WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT=a, b"],
        # The server runs without key checks in these cases.
        ["OFFLINE_MODE=True"],
        ["OFFLINE_MODE=true"],
        ["DEDICATED_DEPLOYMENT_WORKSPACE_URL=https://ws.example.com"],
        ["REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT=False"],
    ],
)
def test_announce_authentication_is_silent_when_exposed_server_will_not_reject_everything(
    environment: list, capsys
) -> None:
    announce_authentication(environment=environment, bind_address="0.0.0.0")

    assert capsys.readouterr().out == ""


def test_announce_authentication_tells_loopback_user_that_a_key_enables_key_checks(
    capsys,
) -> None:
    announce_authentication(
        environment=["ROBOFLOW_API_KEY=key"], bind_address="127.0.0.1"
    )

    output = capsys.readouterr().out
    assert "will require an API key" in output
    assert "builder" in output and "dashboard" in output
    assert "unset ROBOFLOW_API_KEY" in output


@pytest.mark.parametrize(
    "environment",
    [
        [],
        ["ROBOFLOW_API_KEY="],
        ["ROBOFLOW_API_KEY=key", "REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT=false"],
        ["ROBOFLOW_API_KEY=key", "OFFLINE_MODE=True"],
    ],
)
def test_announce_authentication_is_silent_on_loopback_without_key_checks(
    environment: list, capsys
) -> None:
    announce_authentication(environment=environment, bind_address="127.0.0.1")

    assert capsys.readouterr().out == ""


@mock.patch.object(container_adapter, "pull_image")
@mock.patch.object(
    container_adapter, "find_running_inference_containers", return_value=[]
)
@mock.patch.object(container_adapter, "docker")
@mock.patch.object(server, "ensure_container_runtime_is_running")
def test_exported_roboflow_api_key_is_announced_on_loopback_server_start(
    _runtime_mock: MagicMock,
    _docker_mock: MagicMock,
    _find_containers_mock: MagicMock,
    _pull_image_mock: MagicMock,
) -> None:
    # when
    result = CliRunner().invoke(
        server.server_app, ["start"], env={"ROBOFLOW_API_KEY": "exported-key"}
    )

    # then
    assert result.exit_code == 0, result.output
    assert "will require an API key" in result.output
    assert "unset ROBOFLOW_API_KEY" in result.output
