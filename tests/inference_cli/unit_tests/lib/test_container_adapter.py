import os
from unittest import mock
from unittest.mock import MagicMock, mock_open, patch

import pytest
from rich.progress import Progress

from inference_cli.lib import container_adapter
from inference_cli.lib.container_adapter import (
    _JETSON_IMAGES,
    _detect_jetson,
    _get_jetpack_image,
    _image_for_l4t,
    _parse_tegra_release,
    find_running_inference_containers,
    get_image,
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


# --- Tests for Jetson introspection and image selection ---


def test_jetson_images_table_is_sorted_descending() -> None:
    """The table must be sorted (l4t_major DESC, l4t_minor_min DESC) so that
    first-match lookup returns the most specific entry."""
    keys = [(e.l4t_major, e.l4t_minor_min) for e in _JETSON_IMAGES]
    assert keys == sorted(keys, reverse=True), (
        "_JETSON_IMAGES is not sorted descending by (l4t_major, l4t_minor_min). "
        "New entries must be inserted in the correct position."
    )


JETSON_450 = "roboflow/roboflow-inference-server-jetson-4.5.0:latest"
JETSON_461 = "roboflow/roboflow-inference-server-jetson-4.6.1:latest"
JETSON_511 = "roboflow/roboflow-inference-server-jetson-5.1.1:latest"
JETSON_600 = "roboflow/roboflow-inference-server-jetson-6.0.0:latest"
JETSON_620 = "roboflow/roboflow-inference-server-jetson-6.2.0:latest"


class TestParseTegraRelease:
    def test_parses_valid_tegra_release(self) -> None:
        content = "# R36 (release), REVISION: 4.0, GCID: 12345, BOARD: generic"
        with patch("builtins.open", mock_open(read_data=content)):
            assert _parse_tegra_release() == (36, 4)

    def test_parses_r35_tegra_release(self) -> None:
        content = "# R35 (release), REVISION: 2.1, GCID: 99999, BOARD: generic"
        with patch("builtins.open", mock_open(read_data=content)):
            assert _parse_tegra_release() == (35, 2)

    def test_parses_r32_tegra_release(self) -> None:
        content = "# R32 (release), REVISION: 7.1, GCID: 55555, BOARD: t186ref"
        with patch("builtins.open", mock_open(read_data=content)):
            assert _parse_tegra_release() == (32, 7)

    def test_returns_none_when_file_missing(self) -> None:
        with patch("builtins.open", side_effect=FileNotFoundError):
            assert _parse_tegra_release() is None

    def test_returns_none_for_malformed_content(self) -> None:
        with patch("builtins.open", mock_open(read_data="garbage content")):
            assert _parse_tegra_release() is None


class TestImageForL4t:
    @pytest.mark.parametrize(
        "l4t_major, l4t_minor, expected",
        [
            (32, 0, JETSON_450),
            (32, 5, JETSON_450),
            (32, 6, JETSON_461),
            (32, 7, JETSON_461),
            (35, 0, JETSON_511),
            (35, 2, JETSON_511),
            (35, 4, JETSON_511),
            (36, 0, JETSON_600),
            (36, 2, JETSON_600),
            (36, 3, JETSON_600),
            (36, 4, JETSON_620),
            (36, 5, JETSON_620),
        ],
    )
    def test_l4t_to_image(
        self, l4t_major: int, l4t_minor: int, expected: str
    ) -> None:
        assert _image_for_l4t(l4t_major, l4t_minor) == expected

    def test_returns_none_for_unknown_l4t_major(self) -> None:
        assert _image_for_l4t(99, 1) is None


class TestGetJetpackImage:
    @pytest.mark.parametrize(
        "version, expected_image",
        [
            ("4.5", JETSON_450),
            ("4.5.1", JETSON_450),
            ("4.6", JETSON_461),
            ("4.6.1", JETSON_461),
            ("5", JETSON_511),
            ("5.0", JETSON_511),
            ("5.1.1", JETSON_511),
            ("6.0", JETSON_600),
            ("6.1", JETSON_600),
            ("6.2", JETSON_620),
            ("6.2.0", JETSON_620),
        ],
    )
    def test_returns_correct_image(self, version: str, expected_image: str) -> None:
        assert _get_jetpack_image(version) == expected_image

    def test_raises_for_unsupported_version(self) -> None:
        with pytest.raises(RuntimeError, match="not supported"):
            _get_jetpack_image("3.0")


class TestDetectJetson:
    def test_detects_from_tegra_release(self) -> None:
        content = "# R36 (release), REVISION: 4.0, GCID: 12345, BOARD: generic"
        with patch("builtins.open", mock_open(read_data=content)):
            result = _detect_jetson()
        assert result is not None
        image, source = result
        assert image == JETSON_620
        assert "/etc/nv_tegra_release" in source

    @patch.object(container_adapter, "_parse_tegra_release", return_value=None)
    @patch.object(container_adapter, "_get_jetpack_version_from_dpkg", return_value="6.0")
    def test_falls_back_to_dpkg(self, _dpkg_mock: MagicMock, _tegra_mock: MagicMock) -> None:
        result = _detect_jetson()
        assert result is not None
        image, source = result
        assert image == JETSON_600
        assert "dpkg" in source

    @patch.object(container_adapter, "_parse_tegra_release", return_value=None)
    @patch.object(container_adapter, "_get_jetpack_version_from_dpkg", return_value=None)
    def test_returns_none_when_not_jetson(self, _dpkg_mock: MagicMock, _tegra_mock: MagicMock) -> None:
        assert _detect_jetson() is None


class TestGetImage:
    @mock.patch.dict(os.environ, {"JETSON_JETPACK": "6.2"}, clear=False)
    def test_uses_env_var_when_set(self) -> None:
        assert get_image() == JETSON_620

    @mock.patch.dict(os.environ, {}, clear=False)
    @patch.object(container_adapter, "_detect_jetson", return_value=(JETSON_511, "test"))
    def test_uses_introspection_when_no_env_var(self, _mock: MagicMock) -> None:
        os.environ.pop("JETSON_JETPACK", None)
        assert get_image() == JETSON_511

    @mock.patch.dict(os.environ, {}, clear=False)
    @patch.object(container_adapter, "_detect_jetson", return_value=None)
    @patch("subprocess.check_output")
    def test_falls_back_to_gpu_when_nvidia_smi_works(
        self, nvidia_smi_mock: MagicMock, _detect_mock: MagicMock
    ) -> None:
        os.environ.pop("JETSON_JETPACK", None)
        nvidia_smi_mock.return_value = b"some output"
        assert get_image() == "roboflow/roboflow-inference-server-gpu:latest"

    @mock.patch.dict(os.environ, {}, clear=False)
    @patch.object(container_adapter, "_detect_jetson", return_value=None)
    @patch("subprocess.check_output", side_effect=FileNotFoundError)
    def test_falls_back_to_cpu_when_no_gpu(
        self, _nvidia_mock: MagicMock, _detect_mock: MagicMock
    ) -> None:
        os.environ.pop("JETSON_JETPACK", None)
        assert get_image() == "roboflow/roboflow-inference-server-cpu:latest"
