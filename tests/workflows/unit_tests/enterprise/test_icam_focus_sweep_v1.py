import time
from unittest.mock import MagicMock, patch

from inference.enterprise.workflows.enterprise_blocks.streams.icam_focus_sweep.v1 import (
    IcamFocusSweepBlockV1,
)
from inference.enterprise.workflows.stream_camera_parameters.entities import (
    ApplyCameraParametersResult,
)
from inference.enterprise.workflows.stream_camera_parameters.focus_sweep import (
    FocusSweepState,
    next_focus_value,
    run_focus_sweep_tick,
    should_tick,
)


class TestFocusSweepMath:

    def test_next_focus_wraps_at_max(self):
        assert next_focus_value(95, 5, 100) == 100
        assert next_focus_value(100, 5, 100) == 0

    def test_next_focus_increments(self):
        assert next_focus_value(10, 5, 100) == 15

    def test_should_tick_after_interval(self):
        assert should_tick(0.0, 10.0, 10.0) is True
        assert should_tick(5.0, 14.9, 10.0) is False
        assert should_tick(5.0, 15.0, 10.0) is True


class TestRunFocusSweepTick:

    @patch(
        "inference.enterprise.workflows.stream_camera_parameters.focus_sweep.configure_usb_camera"
    )
    def test_applies_lens_position(self, mock_configure):
        mock_configure.return_value = ApplyCameraParametersResult(
            success=True,
            applied=["lens_position"],
        )
        state = FocusSweepState()
        new_state, updated, result = run_focus_sweep_tick(
            state,
            interval_seconds=10,
            step=5,
            max_focus=100,
            video_reference="0",
            edge_base_url="http://192.168.0.15:8000",
            now=100.0,
        )
        assert updated is True
        assert new_state.focus_value == 5
        assert result is not None and result.success is True
        mock_configure.assert_called_once_with(
            "0",
            {"lens_position": 5},
            base_url="http://192.168.0.15:8000",
        )

    @patch(
        "inference.enterprise.workflows.stream_camera_parameters.focus_sweep.configure_usb_camera"
    )
    def test_skips_within_interval(self, mock_configure):
        state = FocusSweepState(focus_value=20, last_tick_at=100.0)
        new_state, updated, result = run_focus_sweep_tick(
            state,
            interval_seconds=10,
            step=5,
            max_focus=100,
            video_reference="0",
            edge_base_url="http://127.0.0.1:8000",
            now=105.0,
        )
        assert updated is False
        assert result is None
        assert new_state.focus_value == 20
        mock_configure.assert_not_called()


class TestIcamFocusSweepBlock:

    @patch(
        "inference.enterprise.workflows.stream_camera_parameters.focus_sweep.configure_usb_camera"
    )
    def test_block_accumulates_focus_over_time(self, mock_configure):
        mock_configure.return_value = ApplyCameraParametersResult(
            success=True,
            applied=["lens_position"],
        )
        block = IcamFocusSweepBlockV1()
        base_args = dict(
            interval_seconds=10.0,
            step=5,
            max_focus=100,
            video_reference="0",
            edge_base_url="http://192.168.0.15:8000",
            depends_on=MagicMock(),
        )

        with patch(
            "inference.enterprise.workflows.stream_camera_parameters.focus_sweep.time.time",
            return_value=1000.0,
        ):
            first = block.run(**base_args)
        assert first["updated"] is True
        assert first["focus"] == 5

        with patch(
            "inference.enterprise.workflows.stream_camera_parameters.focus_sweep.time.time",
            return_value=1005.0,
        ):
            second = block.run(**base_args)
        assert second["updated"] is False
        assert second["focus"] == 5

        with patch(
            "inference.enterprise.workflows.stream_camera_parameters.focus_sweep.time.time",
            return_value=1010.0,
        ):
            third = block.run(**base_args)
        assert third["updated"] is True
        assert third["focus"] == 10
