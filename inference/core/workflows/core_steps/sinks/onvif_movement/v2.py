from typing import Literal, Type, Union

import supervision as sv

from inference.core.workflows.core_steps.sinks.noop import (
    DisableSink,
    versioned_sink_manifest_config,
)
from inference.core.workflows.core_steps.sinks.onvif_movement.v1 import (
    PREDICTIONS_OUTPUT_KEY,
    SEEKING_OUTPUT_KEY,
)
from inference.core.workflows.core_steps.sinks.onvif_movement.v1 import (
    BlockManifest as BlockManifestV1,
)
from inference.core.workflows.core_steps.sinks.onvif_movement.v1 import ONVIFSinkBlockV1
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest


class BlockManifest(BlockManifestV1):
    model_config = versioned_sink_manifest_config(BlockManifestV1, version="v2")
    type: Literal["roboflow_core/onvif_sink@v2"]
    disable_sink: DisableSink = False


class ONVIFSinkBlockV2(ONVIFSinkBlockV1):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        predictions: sv.Detections,
        camera_ip: str,
        camera_port: int,
        camera_username: str,
        camera_password: str,
        movement_type: str,
        default_position_preset: Union[str, None],
        zoom_if_able: bool,
        follow_tracker: bool,
        dead_zone: int,
        camera_update_rate_limit: int,
        flip_y_movement: bool,
        flip_x_movement: bool,
        move_to_position_after_idle_seconds: int,
        pid_kp: float,
        pid_ki: float,
        pid_kd: float,
        minimum_camera_speed: float,
        simulate_variable_speed: bool,
        disable_sink: bool = False,
    ) -> BlockResult:
        if disable_sink:
            return {
                PREDICTIONS_OUTPUT_KEY: predictions,
                SEEKING_OUTPUT_KEY: False,
            }
        return super().run(
            predictions=predictions,
            camera_ip=camera_ip,
            camera_port=camera_port,
            camera_username=camera_username,
            camera_password=camera_password,
            movement_type=movement_type,
            default_position_preset=default_position_preset,
            zoom_if_able=zoom_if_able,
            follow_tracker=follow_tracker,
            dead_zone=dead_zone,
            camera_update_rate_limit=camera_update_rate_limit,
            flip_y_movement=flip_y_movement,
            flip_x_movement=flip_x_movement,
            move_to_position_after_idle_seconds=move_to_position_after_idle_seconds,
            pid_kp=pid_kp,
            pid_ki=pid_ki,
            pid_kd=pid_kd,
            minimum_camera_speed=minimum_camera_speed,
            simulate_variable_speed=simulate_variable_speed,
        )
