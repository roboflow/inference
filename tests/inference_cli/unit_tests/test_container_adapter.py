from unittest import mock
from unittest.mock import MagicMock, call

from inference_cli.lib import container_adapter
from inference_cli.lib.container_adapter import (
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
