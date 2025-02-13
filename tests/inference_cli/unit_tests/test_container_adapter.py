from unittest import mock
from unittest.mock import MagicMock

from inference_cli.lib import container_adapter
from inference_cli.lib.container_adapter import prepare_container_environment


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
