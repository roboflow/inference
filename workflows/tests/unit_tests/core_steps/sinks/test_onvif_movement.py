# note most tests are in test_workflow_with_onvif due to need for camera interaction

import asyncio
from types import SimpleNamespace

import pytest
from onvif import ONVIFError
from roboflow_workflows.core_steps.sinks.onvif_movement import v1, v1_tensor
from roboflow_workflows.core_steps.sinks.onvif_movement.v1 import (
    BlockManifest,
    CameraWrapper,
)
from roboflow_workflows.prototypes.block import Runtime, Severity


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


class _FakeOnvifService:
    def __init__(self, camera: "_FakeOnvifCamera") -> None:
        self._camera = camera

    def create_type(self, name: str) -> SimpleNamespace:
        return SimpleNamespace()

    async def GetProfiles(self) -> list:
        await asyncio.sleep(0)
        ptz_configuration = SimpleNamespace(token="ptz-configuration")
        profile = SimpleNamespace(token="profile", PTZConfiguration=ptz_configuration)

        return [profile]

    async def GetConfigurationOptions(self, request) -> SimpleNamespace:
        await asyncio.sleep(0)
        speed = SimpleNamespace(Min=-1.0, Max=1.0)
        pan_tilt_space = SimpleNamespace(XRange=speed, YRange=speed)
        spaces = SimpleNamespace(
            ContinuousPanTiltVelocitySpace=[pan_tilt_space],
            ContinuousZoomVelocitySpace=[],
        )
        options = SimpleNamespace(Spaces=spaces)

        return options

    async def GetPresets(self, request) -> list:
        await asyncio.sleep(0)

        return [{"Name": "1", "token": "1"}]

    async def ContinuousMove(self, request) -> None:
        await asyncio.sleep(0)
        self._camera.moves.append(request.Velocity)


class _FakeOnvifCamera:
    # Like onvif-zeep-async's ONVIFCamera: update_xaddrs() empties the service
    # addresses while it asks the camera for them, and a service needs an address.
    def __init__(self) -> None:
        self.xaddrs = {}
        self.update_xaddrs_calls = 0
        self.moves = []

    async def update_xaddrs(self) -> None:
        self.update_xaddrs_calls += 1
        self.xaddrs = {}
        await asyncio.sleep(0)
        self.xaddrs = {"media": "http://camera/media", "ptz": "http://camera/ptz"}

    async def create_media_service(self) -> _FakeOnvifService:
        return self._service("media")

    async def create_ptz_service(self) -> _FakeOnvifService:
        return self._service("ptz")

    def _service(self, name: str) -> _FakeOnvifService:
        if name not in self.xaddrs:
            raise ONVIFError(f"Device doesn`t support service: {name}")

        service = _FakeOnvifService(self)

        return service


@pytest.mark.parametrize("module", [v1, v1_tensor])
def test_first_move_survives_a_second_camera_configuration(module) -> None:
    # given - on the first frame both connect_camera_async and
    # continuous_move_async configure the camera
    fake_camera = _FakeOnvifCamera()
    camera = module.CameraWrapper(
        max_update_rate=500,
        move_to_position_after_idle_seconds=0,
        run_loop=module.CameraWrapper.create_event_loop(),
    )
    camera.camera = fake_camera

    async def first_frame() -> None:
        move = asyncio.ensure_future(camera.continuous_move_async(-0.5, -0.5, 0))
        while "ptz" not in fake_camera.xaddrs:
            await asyncio.sleep(0)

        # connect_camera_async's configuration starts while the move's runs
        await camera.configure_async()
        await move

    # when
    camera.schedule(first_frame()).result(timeout=5)

    # then
    assert fake_camera.update_xaddrs_calls == 1
    assert not camera._has_config_error
    assert camera.seeking() is True
    assert len(fake_camera.moves) == 1
