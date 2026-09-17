# note most tests are in test_workflow_with_onvif due to need for camera interaction

import pytest

from inference.core.workflows.core_steps.sinks.onvif_movement import v1, v1_tensor
from inference.core.workflows.core_steps.sinks.onvif_movement.v1 import (
    BlockManifest,
    CameraWrapper,
)
from inference.core.workflows.prototypes.block import Runtime, Severity


def test_manifest_parsing_when_the_input_is_valid() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/onvif_sink@v1",
        "name": "onvif_control",
        "predictions": "$steps.byte_tracker.tracked_detections",
        "camera_ip": "localhost",
        "camera_username": "admin",
        "camera_password": "123456",
        "default_position_preset": "1",
        "pid_ki": "0",
        "pid_kp": "0.25",
        "pid_kd": "1",
        "zoom_if_able": "False",
        "move_to_position_after_idle_seconds": "0",
        "minimum_camera_speed": "10",
        "dead_zone": "10",
        "movement_type": "Follow",
        "camera_update_rate_limit": "500",
        "camera_port": "1981",
        "flip_x_movement": "True",
        "flip_y_movement": "True",
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result == BlockManifest(
        type="roboflow_core/onvif_sink@v1",
        name="onvif_control",
        predictions="$steps.byte_tracker.tracked_detections",
        camera_ip="localhost",
        camera_username="admin",
        camera_password="123456",
        default_position_preset="1",
        pid_ki=0,
        pid_kp=0.25,
        pid_kd=1,
        zoom_if_able=False,
        move_to_position_after_idle_seconds=0,
        minimum_camera_speed=10,
        dead_zone=10,
        movement_type="Follow",
        camera_update_rate_limit=500,
        camera_port=1981,
        flip_x_movement=True,
        flip_y_movement=True,
    )


@pytest.mark.parametrize("module", [v1, v1_tensor])
def test_onvif_block_restrictions_do_not_gate_on_step_execution_mode(module) -> None:
    # Both variants share one HARD "no LAN from hosted" restriction and no
    # longer take step_execution_mode: the sink always runs in-process.
    restrictions = module.BlockManifest.get_restrictions()

    assert restrictions == [v1.NO_LAN_FROM_HOSTED_RESTRICTION]
    assert restrictions[0].severity is Severity.HARD
    assert restrictions[0].applies_to_step_execution_modes is None
    assert restrictions[0].applies_to_input_modes is None
    assert restrictions[0].applies_to_runtimes == [
        Runtime.HOSTED_SERVERLESS,
        Runtime.DEDICATED_DEPLOYMENT,
    ]
    assert "step_execution_mode" not in module.ONVIFSinkBlockV1.get_init_parameters()
