import subprocess
from unittest import mock
from unittest.mock import MagicMock

import pytest

from inference.core.workflows.core_steps.sinks.v4l2_camera import v1
from inference.core.workflows.core_steps.sinks.v4l2_camera.v1 import (
    CONTROL_MAP,
    V4L2CameraControlBlockV1,
    V4L2CameraControlManifest,
    _run_v4l2_ctl,
    _validate_device_path,
)


def test_manifest_parsing_when_input_is_valid() -> None:
    raw_manifest = {
        "type": "roboflow_core/v4l2_camera_control@v1",
        "name": "cam_ctrl",
        "device_path": "/dev/video0",
        "brightness": 128,
        "gain": 10,
        "gamma": 200,
    }

    result = V4L2CameraControlManifest.model_validate(raw_manifest)

    assert result.type == "roboflow_core/v4l2_camera_control@v1"
    assert result.device_path == "/dev/video0"
    assert result.brightness == 128
    assert result.gain == 10
    assert result.gamma == 200


def test_manifest_parsing_with_defaults() -> None:
    raw_manifest = {
        "type": "roboflow_core/v4l2_camera_control@v1",
        "name": "cam_ctrl",
        "device_path": "/dev/video0",
    }

    result = V4L2CameraControlManifest.model_validate(raw_manifest)

    assert result.device_path == "/dev/video0"
    assert result.cooldown_seconds == 0
    assert result.disable_sink is False
    for field_name in CONTROL_MAP:
        assert getattr(result, field_name) is None


def test_manifest_parsing_with_selectors() -> None:
    raw_manifest = {
        "type": "roboflow_core/v4l2_camera_control@v1",
        "name": "cam_ctrl",
        "device_path": "$inputs.device",
        "brightness": "$inputs.brightness_val",
    }

    result = V4L2CameraControlManifest.model_validate(raw_manifest)

    assert result.device_path == "$inputs.device"
    assert result.brightness == "$inputs.brightness_val"


def test_validate_device_path_accepts_valid_paths() -> None:
    _validate_device_path("/dev/video0")
    _validate_device_path("/dev/video12")


def test_validate_device_path_rejects_invalid_paths() -> None:
    with pytest.raises(ValueError, match="Invalid device path"):
        _validate_device_path("/tmp/evil")
    with pytest.raises(ValueError, match="Invalid device path"):
        _validate_device_path("video0")


@mock.patch.object(v1, "subprocess")
def test_run_v4l2_ctl_when_binary_not_found(mock_subprocess) -> None:
    mock_subprocess.run.side_effect = FileNotFoundError
    mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired

    with pytest.raises(RuntimeError, match="v4l2-ctl not found"):
        _run_v4l2_ctl(["-d", "/dev/video0", "--list-ctrls"])


@mock.patch.object(v1, "subprocess")
def test_run_v4l2_ctl_when_command_times_out(mock_subprocess) -> None:
    mock_subprocess.run.side_effect = subprocess.TimeoutExpired(cmd="v4l2-ctl", timeout=5)
    mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired

    with pytest.raises(RuntimeError, match="timed out"):
        _run_v4l2_ctl(["-d", "/dev/video0", "--list-ctrls"])


@mock.patch.object(v1, "subprocess")
def test_run_v4l2_ctl_when_permission_denied(mock_subprocess) -> None:
    mock_subprocess.run.return_value = MagicMock(
        returncode=1, stderr="Permission denied", stdout=""
    )
    mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired

    with pytest.raises(RuntimeError, match="Permission denied"):
        _run_v4l2_ctl(["-d", "/dev/video0", "--list-ctrls"])


@mock.patch.object(v1, "subprocess")
def test_run_v4l2_ctl_when_device_not_found(mock_subprocess) -> None:
    mock_subprocess.run.return_value = MagicMock(
        returncode=1, stderr="No such device", stdout=""
    )
    mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired

    with pytest.raises(RuntimeError, match="Device not found"):
        _run_v4l2_ctl(["-d", "/dev/video99", "--list-ctrls"])


@mock.patch.object(v1, "subprocess")
def test_run_v4l2_ctl_when_successful(mock_subprocess) -> None:
    mock_subprocess.run.return_value = MagicMock(
        returncode=0, stdout="ok", stderr=""
    )
    mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired

    result = _run_v4l2_ctl(["-d", "/dev/video0", "--list-ctrls"])

    assert result == "ok"


@mock.patch.object(v1, "_set_control")
def test_control_block_sets_single_control(mock_set_control) -> None:
    block = V4L2CameraControlBlockV1()

    result = block.run(device_path="/dev/video0", brightness=128)

    assert result["success"] is True
    assert result["throttling_status"] is False
    assert result["controls_set"] == 1
    mock_set_control.assert_called_once_with("/dev/video0", "brightness", 128)


@mock.patch.object(v1, "_set_control")
def test_control_block_sets_multiple_controls(mock_set_control) -> None:
    block = V4L2CameraControlBlockV1()

    result = block.run(device_path="/dev/video0", brightness=128, gain=10)

    assert result["success"] is True
    assert result["controls_set"] == 2
    assert mock_set_control.call_count == 2


@mock.patch.object(v1, "_set_control")
def test_control_block_skips_none_values(mock_set_control) -> None:
    block = V4L2CameraControlBlockV1()

    result = block.run(device_path="/dev/video0")

    assert result["success"] is True
    assert result["controls_set"] == 0
    mock_set_control.assert_not_called()


@mock.patch.object(v1, "_set_control")
def test_control_block_rejects_out_of_range_value(mock_set_control) -> None:
    block = V4L2CameraControlBlockV1()

    result = block.run(device_path="/dev/video0", brightness=999)

    assert result["success"] is False
    assert "out of range" in result["error"]
    mock_set_control.assert_not_called()


@mock.patch.object(v1, "_set_control")
def test_control_block_when_subprocess_fails(mock_set_control) -> None:
    mock_set_control.side_effect = RuntimeError("Device not found: No such device")
    block = V4L2CameraControlBlockV1()

    result = block.run(device_path="/dev/video0", brightness=128)

    assert result["success"] is False
    assert result["controls_set"] == 0
    assert "Device not found" in result["error"]


@mock.patch.object(v1, "_set_control")
def test_control_block_partial_failure(mock_set_control) -> None:
    def side_effect(device, control_name, value):
        if control_name == "gain":
            raise RuntimeError("gain control failed")

    mock_set_control.side_effect = side_effect
    block = V4L2CameraControlBlockV1()

    result = block.run(device_path="/dev/video0", brightness=128, gain=10)

    assert result["success"] is False
    assert result["controls_set"] == 1
    assert "gain" in result["error"]


@mock.patch.object(v1, "_set_control")
def test_control_block_disable_sink(mock_set_control) -> None:
    block = V4L2CameraControlBlockV1()

    result = block.run(device_path="/dev/video0", disable_sink=True, brightness=128)

    assert result["success"] is True
    assert result["controls_set"] == 0
    mock_set_control.assert_not_called()


@mock.patch.object(v1, "_set_control")
def test_control_block_cooldown(mock_set_control) -> None:
    block = V4L2CameraControlBlockV1()

    result1 = block.run(device_path="/dev/video0", cooldown_seconds=60, brightness=128)
    assert result1["success"] is True
    assert result1["controls_set"] == 1

    result2 = block.run(device_path="/dev/video0", cooldown_seconds=60, brightness=200)
    assert result2["success"] is True
    assert result2["throttling_status"] is True
    assert result2["controls_set"] == 0
    assert mock_set_control.call_count == 1


def test_control_block_rejects_invalid_device_path() -> None:
    block = V4L2CameraControlBlockV1()

    result = block.run(device_path="/tmp/evil", brightness=128)

    assert result["success"] is False
    assert "Invalid device path" in result["error"]
