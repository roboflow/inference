from unittest import mock
from unittest.mock import MagicMock

from rich.progress import Progress

from inference_cli.lib import container_adapter
from inference_cli.lib.container_adapter import (
    find_running_inference_containers,
    is_container_running,
    is_inference_server_container,
    kill_containers,
    show_progress,
    terminate_running_containers,
)


def test_is_inference_server_container_when_tag_matches() -> None:
    # given
    container = MagicMock()
    container.image.tags = ["some", "roboflow/roboflow-inference-server-cpu:latest"]

    # when
    result = is_inference_server_container(container=container)

    # then
    assert (
        result is True
    ), "All containers starting from roboflow/roboflow-inference-server are assumed inference containers"


def test_is_inference_server_container_when_tag_does_not_match() -> None:
    # given
    container = MagicMock()
    container.image.tags = ["some", "other"]

    # when
    result = is_inference_server_container(container=container)

    # then
    assert (
        result is False
    ), "No containers starting from roboflow/roboflow-inference-server given"


def test_is_container_running_when_stopped_container_given() -> None:
    # given
    container = MagicMock()
    container.attrs = {"State": {"Status": "stopped"}}

    # when
    result = is_container_running(container=container)

    # then
    assert result is False, "Stopped container must not be recognised as running"


def test_is_container_running_when_container_in_undefined_state_given() -> None:
    # given
    container = MagicMock()
    container.attrs = {"State": {}}

    # when
    result = is_container_running(container=container)

    # then
    assert (
        result is False
    ), "Container with undefined status is assumed not to be running"


def test_is_container_running_when_running_container_given() -> None:
    # given
    container = MagicMock()
    container.attrs = {"State": {"Status": "running"}}

    # when
    result = is_container_running(container=container)

    # then
    assert result is True, "Running container must be recognised as running"


def test_kill_containers_purges_all_containers_passed() -> None:
    # given
    containers = [MagicMock(), MagicMock()]

    # when
    kill_containers(containers=containers)

    # then
    containers[0].kill.assert_called_once()
    containers[1].kill.assert_called_once()


@mock.patch.object(container_adapter, "docker")
def test_find_running_inference_containers(docker_mock: MagicMock) -> None:
    # given
    containers = [MagicMock(), MagicMock(), MagicMock()]
    # inference container - not running
    containers[0].image.tags = ["some", "roboflow/roboflow-inference-server-cpu:latest"]
    containers[0].attrs = {"State": {"Status": "stopped"}}
    # inference container - running
    containers[1].image.tags = ["some", "roboflow/roboflow-inference-server-cpu:latest"]
    containers[1].attrs = {"State": {"Status": "running"}}
    # other container - running
    containers[2].image.tags = ["some", "other"]
    containers[2].attrs = {"State": {"Status": "running"}}
    docker_mock.from_env.return_value.containers.list.return_value = containers

    # when
    result = find_running_inference_containers()

    # then
    assert result == [containers[1]]


@mock.patch.object(container_adapter, "ask_user_to_kill_container")
def test_terminate_running_containers_in_interactive_mode(
    ask_user_to_kill_container_mock: MagicMock,
) -> None:
    # given
    containers = [MagicMock(), MagicMock(), MagicMock()]
    containers[0].attrs = {"State": {"Status": "stopped"}}
    containers[1].attrs = {"State": {"Status": "running"}}
    containers[2].attrs = {"State": {"Status": "running"}}
    ask_user_to_kill_container_mock.side_effect = [
        False,
        True,
    ]  # container 1, 2 should be queried, only 2 to terminate

    # when
    result = terminate_running_containers(containers=containers, interactive_mode=True)

    # then
    assert result is True, "Container 1 must be preserved"
    containers[0].kill.assert_not_called()
    containers[1].kill.assert_not_called()
    containers[2].kill.assert_called_once()


def test_terminate_running_containers_in_non_interactive_mode() -> None:
    # given
    containers = [MagicMock(), MagicMock(), MagicMock()]
    containers[0].attrs = {"State": {"Status": "stopped"}}
    containers[1].attrs = {"State": {"Status": "running"}}
    containers[2].attrs = {"State": {"Status": "running"}}

    # when
    result = terminate_running_containers(containers=containers, interactive_mode=False)

    # then
    assert result is False, "All running containers should be terminated"
    containers[0].kill.assert_not_called()
    containers[1].kill.assert_called_once()
    containers[2].kill.assert_called_once()


def test_show_progress_when_new_downloading_status_encountered() -> None:
    # given
    with Progress() as progress:
        progress_tasks = {}
        log_line = {
            "id": 1,
            "status": "Downloading",
            "progressDetail": {"total": 100.0},
        }

        # when
        show_progress(
            log_line=log_line, progress=progress, progress_tasks=progress_tasks
        )

        # then
        assert len(progress_tasks) == 1, "One new task is expected to be added"
        assert (
            "[red][Downloading 1]" in progress_tasks
        ), "ID for new task should be [red][Downloading 1]"


def test_show_progress_when_update_on_existing_download_encountered() -> None:
    # given
    with Progress() as progress:
        progress_tasks = {
            "[red][Downloading 1]": progress.add_task(
                "[red][Downloading 1]", total=100.0
            )
        }
        log_line = {
            "id": 1,
            "status": "Downloading",
            "progressDetail": {"current": 3.0},
        }

        # when
        show_progress(
            log_line=log_line, progress=progress, progress_tasks=progress_tasks
        )

        # then
        assert len(progress_tasks) == 1, "No new task to be created"
        assert (
            abs(progress.tasks[0].percentage - 3.0) < 1e-5
        ), "Progress should match 3% (3.0 / 100.0)"


def test_show_progress_when_new_extracting_status_encountered() -> None:
    # given
    with Progress() as progress:
        progress_tasks = {}
        log_line = {"id": 1, "status": "Extracting", "progressDetail": {"total": 100.0}}

        # when
        show_progress(
            log_line=log_line, progress=progress, progress_tasks=progress_tasks
        )

        # then
        assert len(progress_tasks) == 1, "One new task is expected to be added"
        assert (
            "[green][Extracting 1]" in progress_tasks
        ), "ID for new task should be [green][Extracting 1]"


def test_show_progress_when_update_on_existing_extracting_encountered() -> None:
    # given
    with Progress() as progress:
        progress_tasks = {
            "[green][Extracting 1]": progress.add_task(
                "[green][Extracting 1]", total=100.0
            )
        }
        log_line = {"id": 1, "status": "Extracting", "progressDetail": {"current": 3.0}}

        # when
        show_progress(
            log_line=log_line, progress=progress, progress_tasks=progress_tasks
        )

        # then
        assert len(progress_tasks) == 1, "No new task to be created"
        assert (
            abs(progress.tasks[0].percentage - 3.0) < 1e-5
        ), "Progress should match 3% (3.0 / 100.0)"


def test_show_progress_when_unknown_status_given() -> None:
    # given
    with Progress() as progress:
        progress_tasks = {}
        log_line = {"id": 1, "status": "unknown", "progressDetail": {"total": 100.0}}

        # when
        show_progress(
            log_line=log_line, progress=progress, progress_tasks=progress_tasks
        )

        # then
        assert (
            len(progress_tasks) == 0
        ), "No new task should be added on the update which is not recognised"
