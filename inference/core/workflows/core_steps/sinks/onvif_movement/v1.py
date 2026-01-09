import asyncio
import concurrent
import importlib
import os
import threading
import time
from threading import Thread
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import supervision as sv
import zeep
from onvif import ONVIFCamera, ONVIFService
from pydantic import ConfigDict, Field, PositiveInt
from simple_pid import PID

from inference.core import logger
from inference.core.utils.function import experimental
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    SECRET_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

# max number of seconds to switch to zoom only (no xy movement)
ZOOM_MODE_SECONDS = 2

# After the first zoom mode, multiply pan/tilt speed by this much to
# help with control. Will revert once the camera goes back to the preset
# This could be improved in the future by more accurately measuring
# zoom level (note not all cameras can provide coordinates)
ZOOM_MODE_SPEED_REDUCER = 0.5

PREDICTIONS_OUTPUT_KEY: str = "predictions"
SEEKING_OUTPUT_KEY: str = "seeking"

LONG_DESCRIPTION = """
Control an ONVIF-compatible PTZ (Pan-Tilt-Zoom) camera to automatically follow detected objects, move to preset positions, and maintain smooth tracking using PID control for surveillance, security monitoring, and automated camera control workflows.

## How This Block Works

This block controls PTZ cameras through the ONVIF protocol to automatically track and follow objects in real-time. The block:

1. Receives object detection or instance segmentation predictions from upstream blocks
2. Connects to an ONVIF-compatible PTZ camera using IP address, port, username, and password credentials
3. Determines the movement mode (Follow or Go To Preset):

   **For Follow Mode:**
   - Selects the target object to track: either the highest confidence detection or a specific tracked object (if tracker IDs are present)
   - Calculates the position error by comparing the object's bounding box center to the frame center
   - Uses PID (Proportional-Integral-Derivative) control to calculate smooth movement commands:
     - Proportional (Kp): Responds to current position error
     - Integral (Ki): Corrects steady-state errors over time
     - Derivative (Kd): Predicts future error and dampens oscillations
   - Normalizes movement commands to the camera's velocity limits and applies rate limiting to prevent command flooding
   - Sends continuous movement commands to the camera via ONVIF ContinuousMove service
   - Monitors the dead zone (region around center where camera stops moving) to prevent hunting behavior
   - Optionally zooms into the object when it's centered, adjusting zoom speed to fill the frame
   - Maintains tracking of a specific object using tracker IDs until it disappears or tracking is reset
   - Automatically moves to a preset position after a configurable idle period when no objects are detected

   **For Go To Preset Mode:**
   - Moves the camera to a predefined preset position using the ONVIF GotoPreset service
   - Requires the camera to support preset functionality and have presets configured
   - Uses the preset name specified in the configuration

4. Handles camera communication asynchronously using a separate event loop to prevent blocking workflow execution
5. Manages camera state including seeking status, tracked object IDs, zoom state, and movement history
6. Returns two outputs:
   - **predictions**: The detection being tracked (empty if no object is being followed)
   - **seeking**: Boolean indicating whether the camera is currently moving/seeking an object

The block uses PID control to calculate smooth, proportional movement commands based on the distance between the object center and frame center. Movement is normalized as a percentage of the camera's maximum speed, and commands are rate-limited to prevent overwhelming the camera with updates. The dead zone prevents small movements when the object is near the center, reducing hunting behavior. When zoom is enabled, the camera first centers the object with pan/tilt, then zooms in to fill the frame while maintaining the object in view.

## Common Use Cases

- **Surveillance and Security**: Automatically track individuals or vehicles in surveillance scenarios (e.g., follow suspicious activity, track intruders, monitor security perimeters), enabling automated surveillance workflows
- **Sports and Event Coverage**: Track athletes or objects during sports events or performances (e.g., follow players on field, track ball movement, cover event action), enabling automated sports coverage workflows
- **Wildlife Monitoring**: Follow animals or wildlife in natural habitats (e.g., track bird movements, follow animals in reserves, monitor wildlife behavior), enabling wildlife observation workflows
- **Industrial Monitoring**: Automatically follow objects or personnel in industrial settings (e.g., track equipment movement, monitor worker activities, follow vehicles in facilities), enabling industrial automation workflows
- **Traffic Monitoring**: Track vehicles or objects in traffic scenarios (e.g., follow vehicles through intersections, track traffic violations, monitor road activity), enabling automated traffic monitoring workflows
- **Retail Analytics**: Track customers or products in retail environments (e.g., follow customer paths, track product interactions, monitor shopping behavior), enabling retail analytics workflows

## Connecting to Other Blocks

This block receives predictions and produces camera control commands and tracking status:

- **After object detection or instance segmentation blocks** to track detected objects with the camera (e.g., follow detected people, track detected vehicles, monitor detected objects), enabling detection-to-camera-tracking workflows
- **After Byte Tracker blocks** to follow specific tracked objects with consistent IDs (e.g., follow tracked person across frames, maintain tracking of specific vehicle, monitor tracked object persistently), enabling tracking-to-camera workflows
- **After detection filter blocks** to track specific object classes or filtered detections (e.g., track only specific classes, follow filtered detections, monitor selected objects), enabling filtered-tracking workflows
- **Before visualization blocks** to display camera movement status and tracked objects (e.g., visualize tracking status, display seeking indicator, show camera control feedback), enabling camera control visualization workflows
- **Before notification blocks** to alert when camera starts or stops tracking (e.g., notify when tracking begins, alert on tracking loss, report camera status), enabling camera status notification workflows
- **In surveillance and monitoring pipelines** where automated camera control is part of a larger security or monitoring system (e.g., automated security systems, monitoring pipelines, camera control chains), enabling comprehensive surveillance workflows

## Requirements

This block requires an ONVIF-compatible PTZ camera with network access. The camera must support the ONVIF ContinuousMove service for Follow mode and GotoPreset service for preset movement. For optimal performance, use a camera with variable speed movement capability - cameras without variable speed can use the simulate_variable_speed option but may experience jerky movement. The block must run in local execution mode (not suitable for remote/cloud execution). PID tuning is recommended to achieve smooth tracking without overshooting or hunting - adjust pid_kp, pid_ki, and pid_kd parameters based on camera responsiveness and video latency. For accurate tracking, use an eager buffer consumption strategy to minimize lag between camera movement and video feedback. The camera must have presets configured if using preset movement or auto-reset functionality.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "PTZ Tracking (ONVIF)",
            "version": "v1",
            "short_description": "Control an ONVIF compatible PTZ camera to follow an object",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
            "ui_manifest": {
                "section": "video",
                "icon": "fal fa-camera-cctv",
                "blockPriority": 1,
                "popular": False,
            },
        }
    )
    type: Literal["roboflow_core/onvif_sink@v1"]
    predictions: Selector(
        kind=[OBJECT_DETECTION_PREDICTION_KIND, INSTANCE_SEGMENTATION_PREDICTION_KIND]
    ) = Field(  # type: ignore
        description="Object detection or instance segmentation predictions to track. In Follow mode, the block will follow the highest confidence prediction or a tracked object if tracker IDs are present. Predictions should include bounding box coordinates and optionally tracker IDs for persistent tracking.",
        examples=["$steps.object_detection_model.predictions"],
    )
    camera_ip: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Camera IP address or hostname for ONVIF connection. Must be reachable from the workflow execution environment.",
    )
    camera_port: Union[Selector(kind=[INTEGER_KIND]), PositiveInt] = Field(
        description="Camera ONVIF service port (typically 80, 8080, or camera-specific port). Must match the camera's ONVIF configuration.",
        ge=0,
        le=65535,
    )
    camera_username: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Camera username for ONVIF authentication. Must have PTZ control permissions on the camera.",
    )
    camera_password: Union[Selector(kind=[SECRET_KIND]), str] = Field(
        description="Camera password for ONVIF authentication. Should be stored as a secret for security.",
    )
    movement_type: Literal["Follow", "Go To Preset"] = Field(
        default="Follow",
        description="Movement mode for the camera. 'Follow' mode tracks detected objects using PID control. 'Go To Preset' mode moves the camera to a predefined preset position (requires default_position_preset to be configured).",
        examples=["Follow", "Go To Preset", "$inputs.movement_type"],
    )
    simulate_variable_speed: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="Enable variable speed simulation for cameras without native variable speed support. When enabled, sends 100% speed commands followed by stop commands to approximate percentage speeds. May result in jerky movement - only use if camera lacks variable speed capability.",
        examples=[True, False, "$inputs.simulate_variable_speed"],
    )
    zoom_if_able: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="Enable automatic zoom to fill frame with the tracked object. When enabled, camera first centers the object with pan/tilt, then zooms in until the object fills the frame. Requires camera to support zoom functionality.",
        examples=[True, False, "$inputs.zoom_if_able"],
    )
    follow_tracker: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Enable persistent tracking using tracker IDs. When enabled, camera locks onto the tracker ID of the highest confidence detection and continues following that specific object until it disappears or tracking resets. Requires a Byte Tracker block in the workflow to assign tracker IDs.",
        examples=[True, False, "$inputs.follow_tracker"],
    )
    dead_zone: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        default=50,
        description="Dead zone size in pixels around frame center where camera movement stops. Prevents hunting behavior when object is near center. Larger values reduce pan/tilt hunting but may cause zoom hunting. Smaller values improve zoom stability but may cause pan/tilt oscillations. Typical range: 30-100 pixels.",
        examples=[50, "$inputs.dead_zone"],
    )
    default_position_preset: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Preset name for default/home position. Camera will return to this preset after idle period (if move_to_position_after_idle_seconds is set) or when using Go To Preset mode. Must match exactly a preset name configured on the camera. Required for preset movement functionality.",
        default="",
        examples=["", "$inputs.default_position_preset"],
    )
    move_to_position_after_idle_seconds: Union[Selector(kind=[INTEGER_KIND]), int] = (
        Field(
            default=0,
            description="Auto-reset time in seconds. After camera stops seeking/moving for this duration, automatically moves to default_position_preset. Set to 0 to disable auto-reset. Requires default_position_preset to be configured.",
        )
    )
    camera_update_rate_limit: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        default=250,
        description="Minimum time in milliseconds between camera movement commands. Rate limits ONVIF updates to prevent overwhelming the camera. Lower values provide more responsive tracking but may overload slower cameras. Higher values reduce camera load but may cause less smooth movement. Typical range: 100-500ms.",
    )
    flip_x_movement: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        examples=[True, False],
        description="Invert horizontal (pan) movement direction. Enable if the camera image is mirrored horizontally and movement appears reversed. Use to correct camera movement when image is flipped.",
    )
    flip_y_movement: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        examples=[True, False],
        description="Invert vertical (tilt) movement direction. Enabled by default as many cameras have inverted Y-axis. Disable if vertical movement appears reversed.",
    )
    minimum_camera_speed: Union[float, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(
        default=0.05,
        description="Minimum movement speed as percentage (0.0-1.0). Movement commands below this threshold are boosted to this minimum. Some cameras ignore very low speeds - increase if camera doesn't respond to small movements. Typical range: 0.02-0.1 (2%-10%).",
    )
    pid_kp: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=0.25,
        description="PID proportional gain (Kp). Controls response to position error - higher values make camera respond faster but may cause overshooting and hunting. Lower values reduce hunting but make tracking slower. Start with default and adjust based on camera responsiveness. Typical range: 0.1-0.5.",
    )
    pid_ki: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=0.0,
        description="PID integral gain (Ki). Eliminates steady-state error by accumulating error over time. Usually kept at 0 as it can cause oscillations. Increase slightly (0.01-0.1) if camera consistently stops slightly off-center despite small errors.",
    )
    pid_kd: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=1,
        description="PID derivative gain (Kd). Predicts future error and dampens oscillations. Higher values improve stability with video lag but excessive values can cause hunting. Increase (1-5) if there's significant delay between camera movement and video feedback. Decrease if tracking appears jerky.",
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=PREDICTIONS_OUTPUT_KEY,
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                ],
            ),
            OutputDefinition(
                name=SEEKING_OUTPUT_KEY,
                kind=[
                    BOOLEAN_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


# primarily used for rate limiting
def now() -> int:
    return int(round(time.time() * 1000))


class Limits:
    min: float
    max: float

    def __init__(self, range):
        self.min = range.Min
        self.max = range.Max

    def __repr__(self):
        return f"({self.min},{self.max})"

    def __eq__(self, value):
        value.min == self.min and value.max == self.max


class VelocityLimits:
    x: Limits
    y: Limits
    z: Union[Limits, None]

    def __init__(self, x: Limits, y: Limits, z: Limits):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"x:{self.x} y:{self.y} z:{self.z}"


def run_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


# camera wrapper is used to store config info so that we don't have
# to keep querying it from the camera on successive commands
class CameraWrapper:
    camera: Optional[ONVIFCamera] = None
    thread: Thread
    run_loop: asyncio.AbstractEventLoop = None
    media_profile = zeep.AnyObject
    configuration_options = zeep.AnyObject
    _media_profile_token: str = None
    tracked_object: Optional[int] = None
    _velocity_limits: Optional[VelocityLimits]
    _presets: Optional[Dict[str, zeep.AnyObject]] = None
    _last_update_ms: Optional[int] = None
    _max_update_rate: int = 0
    _move_to_position_after_idle_seconds: int = 0
    _existing_preset_task: Optional[asyncio.Task] = None
    _seeking: Optional[bool] = None
    _prev_x: float = 100
    _prev_y: float = 100
    _prev_z: float = 100
    _start_zoom_time: Optional[int] = (
        None  # don't allow xy movements when not None, zoom only
    )
    _stop_preset: Optional[str] = None
    _is_zoomed: bool = False
    _has_config_error: bool = False
    _x_count: int = 0
    _y_count: int = 0
    _z_count: int = 0

    @experimental(
        reason="Usage of CameraWrapper is an experimental feature. Please report any issues "
        "here: https://github.com/roboflow/inference/issues"
    )
    # create a new camera wrapper with an asyncio event loop
    def __init__(
        self,
        max_update_rate: int,
        move_to_position_after_idle_seconds: int,
        run_loop: asyncio.AbstractEventLoop,
    ):
        self._max_update_rate = max_update_rate
        self._move_to_position_after_idle_seconds = move_to_position_after_idle_seconds
        self.run_loop = run_loop

    @classmethod
    def create_event_loop(cls) -> asyncio.AbstractEventLoop:
        async_run_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=run_loop, args=(async_run_loop,), daemon=True)
        thread.start()
        return async_run_loop

    def connect_camera(
        self, camera_ip, camera_port, camera_username, camera_password
    ) -> concurrent.futures._base.Future[None]:
        return self.schedule(
            self.connect_camera_async(
                camera_ip, camera_port, camera_username, camera_password
            )
        )

    async def connect_camera_async(
        self, camera_ip, camera_port, camera_username, camera_password
    ):
        if not self.camera:
            # wsdls are in package directory, ex: "/usr/local/lib/python3.9/site-packages/onvif/wsdl"
            spec = importlib.util.find_spec("onvif")
            wdsl_path = f"{os.path.dirname(spec.origin)}/wsdl"
            self.camera = ONVIFCamera(
                camera_ip, camera_port, camera_username, camera_password, wdsl_path
            )
            await self.configure_async()
        else:
            logger.debug("camera is already connected")

    def set_stop_preset(self, stop_preset: Optional[str]):
        self._stop_preset = stop_preset

    # schedule a future inside the camera's event loop
    def schedule(self, cor) -> concurrent.futures._base.Future[None]:
        return asyncio.run_coroutine_threadsafe(cor, loop=self.run_loop)

    # pushes out the next scheduled move to preset (reset)
    def schedule_next_reset(self):
        if self._stop_preset:
            self.schedule(self.next_reset())

    def clear_next_reset(self):
        if self._existing_preset_task:
            self._existing_preset_task.cancel()

    async def next_reset(self):
        self.clear_next_reset()
        if self._stop_preset:
            self._existing_preset_task = asyncio.create_task(self.reset_task())
            await self._existing_preset_task

    # sleep will expire after camera hasn't moved for idle seconds, and move to preset
    async def reset_task(self):
        await asyncio.sleep(self._move_to_position_after_idle_seconds)
        logger.debug(
            f'camera is idle for {self._move_to_position_after_idle_seconds}s: moving to preset "{self._stop_preset}"'
        )
        # if we've been tracking an object, we want to clear it here
        self.tracked_object = None
        # note stop preset can be cleared even after task is scheduled
        if self._stop_preset:
            await self.go_to_preset_async(self._stop_preset)

    # true if movement update hasn't happened within max_update_rate
    def _can_update(self) -> bool:
        return (
            self._last_update_ms is None
            or now() - self._last_update_ms > self._max_update_rate
        )

    # this is mainly used to allow stop commands through on new zero speeds
    def save_last_speeds(self, x, y, z) -> Tuple[bool, bool, bool]:
        x_changed = x != self._prev_x
        y_changed = y != self._prev_y
        z_changed = z != self._prev_z
        self._prev_x = x
        self._prev_y = y
        self._prev_z = z
        return (x_changed, y_changed, z_changed)

    async def ptz_service(self) -> Union[None, ONVIFService]:
        """
        Creates the ONVIF PTZ service
        This has to run on every command requiring the service - a service can't be awaited twice
        """
        if not self.camera:
            return None
        return await self.camera.create_ptz_service()

    async def media_service(self) -> Union[None, ONVIFService]:
        """
        Creates the ONVIF media service
        This is primarily used to get the media token
        """
        if not self.camera:
            return None
        return await self.camera.create_media_service()

    async def configure_async(self):
        """
        Does initial configuration and gathers all camera info
        Doesn't currently run in init since it needs to be awaited
        """
        if not self.camera:
            raise ValueError(f"Tried to configure camera, but camera was not created")
        if self._has_config_error:
            return

        try:
            await self.camera.update_xaddrs()
            # capabilities = await self.camera.get_capabilities() # <- could be useful in the future
            media = await self.media_service()
            self.media_profile = (await media.GetProfiles())[0]
            ptz = await self.ptz_service()
            config_request = ptz.create_type("GetConfigurationOptions")
            config_request.ConfigurationToken = (
                self.media_profile.PTZConfiguration.token
            )
            config_options = ptz.GetConfigurationOptions(config_request)
            self.configuration_options = await config_options
            pan_tilt_space = (
                self.configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0]
                if hasattr(
                    self.configuration_options.Spaces, "ContinuousPanTiltVelocitySpace"
                )
                and len(
                    self.configuration_options.Spaces.ContinuousPanTiltVelocitySpace
                )
                > 0
                else None
            )
            # pan tilt space is necessary
            if pan_tilt_space is None:
                logger.error("Could not get pan tilt space for camera")
                raise ValueError("Could not get pan tilt space for camera")
            zoom_space = (
                self.configuration_options.Spaces.ContinuousZoomVelocitySpace[0]
                if hasattr(
                    self.configuration_options.Spaces, "ContinuousZoomVelocitySpace"
                )
                and len(self.configuration_options.Spaces.ContinuousZoomVelocitySpace)
                > 0
                else None
            )
            self._velocity_limits = VelocityLimits(
                x=Limits(pan_tilt_space.XRange),
                y=Limits(pan_tilt_space.YRange),
                z=Limits(zoom_space.XRange) if zoom_space else None,
            )
            self._media_profile_token = self.media_profile.token
            presets = await ptz.GetPresets({"ProfileToken": self._media_profile_token})
            # reconfigure into a dict keyed by preset name
            self._presets = {preset["Name"]: preset for preset in presets}
        except Exception as e:
            self._has_config_error = True
            logger.warning(f"Config error: {e}")

    def seeking(self) -> Optional[bool]:
        return self._seeking

    # stops the camera movement
    def stop_camera(self):
        # don't stop if already stopped as stop commands aren't rate limited
        if self._seeking or self._seeking is None:
            logger.debug("stop camera")
            self.continuous_move(0, 0, 0)
            self.schedule_next_reset()

        self._seeking = False

    def go_to_preset(self, preset_name: str, limit_rate: bool = False):
        self.schedule(self.go_to_preset_async(preset_name, limit_rate))

    async def go_to_preset_async(self, preset_name: str, limit_rate: bool = False):
        """
        Tells the camera to move to a preset
        This is not rate limited - all commands will be sent to the camera

        Args:
            preset_name: The preset name to move to (this varies by camera)
        """

        # this is only used for blocks in go to preset mode
        if limit_rate:
            if not self._can_update():
                return
            self._last_update_ms = now()

        if self._media_profile_token is None:
            await self.configure_async()

        preset = self._presets.get(preset_name)
        if not preset:
            # TODO: since this is thrown in another thread, it isn't getting thrown, print to logs
            logger.error(
                f'Camera does not have preset "{preset_name}" - valid presets are {list(self._presets.keys())}'
            )
            raise ValueError(
                f'Camera does not have preset "{preset_name}" - valid presets are {list(self._presets.keys())}'
            )

        ptz = await self.ptz_service()
        request = ptz.create_type("GotoPreset")
        request.ProfileToken = self._media_profile_token
        request.PresetToken = preset["token"]
        request.Speed = {"PanTilt": {"x": 1.0, "y": 1.0}, "Zoom": {"x": 1.0}}

        self._seeking = False
        self._is_zoomed = False
        self.tracked_object = None
        await ptz.GotoPreset(request)

    def zoom(self, z: float):
        # start limited time zoom mode
        if not self._start_zoom_time:
            self._start_zoom_time = now()
            # even though the zoom command should 0 out pan/tilt, sending an explicit stop on all axes seems to help
            self.stop_camera()
        self.schedule(self.continuous_move_async(0, 0, z))

    def stop_zoom(self):
        if self._start_zoom_time is not None:
            self._start_zoom_time = None

    def zooming(self) -> bool:
        global ZOOM_MODE_SECONDS
        if (
            self._start_zoom_time
            and now() > self._start_zoom_time + ZOOM_MODE_SECONDS * 1000
        ):
            self.stop_zoom()
        return self._start_zoom_time is not None

    # tells the camera to move at a continuous velocity
    def continuous_move(
        self, x: float, y: float, z: float, simulate_variable_speed: bool = False
    ):
        self.schedule(self.continuous_move_async(x, y, z, simulate_variable_speed))

    def simulate_variable_speed(self, speed: float, count: int) -> Tuple[float, int]:
        count = count + 1

        if speed != 0 and count >= int(1.0 / speed):
            speed = np.sign(speed)
            if self._can_update():
                count = 0
        else:
            speed = 0

        # stop count until we let the next update through
        return speed, count

    # x and y are velocities from -1 to 1
    async def continuous_move_async(
        self, x: float, y: float, z: float, simulate_variable_speed: bool = False
    ):
        """
        Tells the camera to move at a continuous velocity, or 0 to stop
        Note this is rate limited, some commands will be ignored

        Args:
            x: The x velocity normalized as -1 to 1
            y: The y velocity normalized as -1 to 1
            z: The zoom velocity normalized as -1 to 1
        """

        if self._media_profile_token is None:
            await self.configure_async()

        # clear out any scheduled position resets
        # they'll be rescheduled on the next stop
        if x != 0 and y != 0 and z != 0:
            self.clear_next_reset()

        # This option simulates a % speed by sending a number of move and stop
        # commands that approximate the required % - so for 25% we'll send
        # one 100% move command followed by one stop command.
        if simulate_variable_speed:
            x, self._x_count = self.simulate_variable_speed(x, self._x_count)
            y, self._y_count = self.simulate_variable_speed(y, self._y_count)
            z, self._z_count = self.simulate_variable_speed(z, self._z_count)

        # try to avoid hunting by allowing immediate stop
        x_changed, y_changed, z_changed = self.save_last_speeds(x, y, z)

        # don't rate limit stop commands
        if (x == 0 and x_changed) or (y == 0 and y_changed) or (z == 0 and z_changed):
            pass
        elif not self._can_update():
            return

        ptz = await self.ptz_service()

        # https://www.onvif.org/onvif/ver20/ptz/wsdl/ptz.wsdl#op.AbsoluteMove

        request = ptz.create_type("ContinuousMove")
        request.ProfileToken = self._media_profile_token

        limits = self._velocity_limits

        # normalize to camera's velocity limits
        x_limit = limits.x.min if x < 0 else limits.x.max
        y_limit = limits.y.min if x < 0 else limits.y.max
        if limits.z:
            z_limit = limits.z.min if x < 0 else limits.z.max
        else:
            z_limit = 0

        x = abs(x_limit) * x
        y = abs(y_limit) * y
        z = abs(z_limit) * z

        if self._is_zoomed:
            x = x * ZOOM_MODE_SPEED_REDUCER
            y = y * ZOOM_MODE_SPEED_REDUCER

        request.Velocity = {"PanTilt": {"x": x, "y": y}, "Zoom": {"x": z}}

        logger.debug(
            f"ptz continuous move update: {x},{y},{z} in_zoom:{self._is_zoomed} tracker:{self.tracked_object}"
        )

        # Execute the movement
        await ptz.ContinuousMove(request)
        self._last_update_ms = now()

        if z > 0:
            self._is_zoomed = True

        # prevent stops from re-flagging themselves as seeking
        if x != 0 or y != 0 or z != 0:
            self._seeking = True

    def stop_tracking(self):
        self.stop_camera()
        self.tracked_object = None

    # this just sends a stop command - should be ok if other blocks are controlling camera
    def __del__(self):
        self.stop_camera()


class ONVIFSinkBlockV1(WorkflowBlock):

    def __init__(
        self,
        step_execution_mode: StepExecutionMode,
    ):
        self._step_execution_mode = step_execution_mode
        # all commands will be send to the camera normalized to -1 to 1
        # all setpoints are 0, which represents the center of the frame
        self.x_pid = PID(0, 0, 0, setpoint=0)
        self.x_pid.output_limits = (-1, 1)
        self.y_pid = PID(0, 0, 0, setpoint=0)
        self.y_pid.output_limits = (-1, 1)
        self.z_pid = PID(0, 0, 0, setpoint=0)
        self.z_pid.output_limits = (-1, 1)
        self.event_loop = CameraWrapper.create_event_loop()
        # pool of camera services can can be used in block
        self.cameras: Dict[Tuple[str, int], CameraWrapper] = {}
        self._lock = threading.Lock()

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["step_execution_mode"]

    # gets the CameraWrapper from the static cameras collection
    def get_camera(
        self,
        camera_ip: str,
        camera_port: int,
        camera_username: str,
        camera_password: str,
        max_update_rate: int,
        move_to_position_after_idle_seconds: int,
        event_loop: asyncio.AbstractEventLoop,
    ) -> Optional[CameraWrapper]:
        cameras = self.cameras
        camera_key = (camera_ip, camera_port)
        with self._lock:
            if camera_key not in cameras:
                try:
                    cameras[camera_key] = CameraWrapper(
                        max_update_rate, move_to_position_after_idle_seconds, event_loop
                    )
                    cameras[camera_key].connect_camera(
                        camera_ip, camera_port, camera_username, camera_password
                    )
                except Exception as e:
                    if camera_key in cameras:
                        del cameras[camera_key]
                    raise ValueError(
                        f"Error connecting to camera at {camera_ip}:{camera_port}: {e}"
                    )
        return cameras.get(camera_key)

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
    ) -> BlockResult:

        # this is hard coded: if intermittent move signals are less
        # than 10% then it's unlikely the camera will ever move
        if simulate_variable_speed:
            minimum_camera_speed = max(minimum_camera_speed, 0.1)

        if self._step_execution_mode != StepExecutionMode.LOCAL:
            raise ValueError("Inference must be run locally for the ONVIF block")

        if move_to_position_after_idle_seconds and not default_position_preset:
            raise ValueError(
                "Move to position after idle is set, but no default position is set"
            )

        # disable the stop preset if necessary
        stop_preset = (
            default_position_preset if move_to_position_after_idle_seconds else None
        )

        camera = self.get_camera(
            camera_ip,
            camera_port,
            camera_username,
            camera_password,
            camera_update_rate_limit,
            move_to_position_after_idle_seconds,
            self.event_loop,
        )
        camera.set_stop_preset(stop_preset)

        if movement_type == "Follow":

            # for v1 use the same constants for all axes
            self.x_pid.tunings = (pid_kp, pid_ki, pid_kd)
            self.y_pid.tunings = (pid_kp, pid_ki, pid_kd)
            self.z_pid.tunings = (pid_kp, pid_ki, pid_kd)

            if len(predictions.xyxy) == 0:
                # get/create the camera first so that we can move it to the preset
                camera.stop_tracking()

                return {
                    PREDICTIONS_OUTPUT_KEY: sv.Detections.empty(),
                    SEEKING_OUTPUT_KEY: camera.seeking() if camera else False,
                }

            tracked_object = camera.tracked_object

            max_confidence_prediction = None

            # if there's a tracked object, continue to use it
            if tracked_object:
                tracked_predictions = predictions[
                    predictions.tracker_id == tracked_object
                ]
                if len(tracked_predictions.xyxy) > 0:
                    max_confidence_prediction = tracked_predictions[0]

            # if there's no tracked object, use the max confidence prediction
            if not max_confidence_prediction:
                max_confidence = predictions.confidence.max()
                max_confidence_prediction = predictions[
                    predictions.confidence == max_confidence
                ][0]
                # if we're not tracking at the moment, start tracking this one
                if follow_tracker and max_confidence_prediction.tracker_id:
                    if len(max_confidence_prediction.tracker_id) > 0:
                        tracked_object = int(max_confidence_prediction.tracker_id[0])

                camera.tracked_object = tracked_object

            # adjust PID here as necessary, and send commands to wrapper
            self.move_camera(
                camera,
                max_confidence_prediction,
                zoom_if_able,
                dead_zone,
                flip_x_movement,
                flip_y_movement,
                minimum_camera_speed,
                simulate_variable_speed,
            )

            return {
                PREDICTIONS_OUTPUT_KEY: max_confidence_prediction,
                SEEKING_OUTPUT_KEY: camera.seeking() if camera else False,
            }

        elif movement_type == "Go To Preset":
            camera.stop_camera()
            camera.go_to_preset(default_position_preset, True)

        return {
            PREDICTIONS_OUTPUT_KEY: sv.Detections.empty(),
            SEEKING_OUTPUT_KEY: camera.seeking() if camera else False,
        }

    def move_camera(
        self,
        camera: CameraWrapper,
        prediction: sv.Detections,
        zoom_if_able: bool,
        dead_zone: int,
        flip_x_movement: bool,
        flip_y_movement: bool,
        minimum_camera_speed: float,
        simulate_variable_speed: bool = False,
    ):
        """
        This is where the PID changes are adjusted before the movement
        commands are sent to CameraWrapper

        Args:
            camera: CameraWrapper
            prediction: The prediction containing a single object to point the camera to
            zoom_if_able: True to try zooming in, False to ignore
            dead_zone: The camera will send stop commands when the object is within this zone
            flip_x_movement: Use to reverse the sign on the movement command if image is flipped
            flip_y_movement: Use to reverse the sign on the movement command if image is flipped
            minimum_camera_speed: Expressed as % from 0-1, movement commands are limited to this minimum
            stop_preset: The name of the preset for the camera to go to after being idle
            simulate_variable_speed: All speeds will be max % with start/stops used to simulate speed
        """

        # get dimensions of image and bounding box from prediction
        image_dimensions = prediction.data["root_parent_dimensions"]
        xyxy = prediction.xyxy

        # calculate centers
        (x1, y1, x2, y2) = tuple(xyxy[0])
        (image_height, image_width) = tuple(image_dimensions[0])
        center_point = (x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2)

        # calculate deltas from center and edge
        zoom_delta = min([x1, image_width - x2, y1, image_height - y2])
        box_at_edge = zoom_delta <= 1  # allow 1px tolerance
        # make the deltas x, y, zoom
        delta = (image_width / 2 - center_point[0], image_height / 2 - center_point[1])

        # if we're locked into zoom only mode, and the object goes to the edge, unlock it and go back to pan/tilt
        if camera.zooming() and zoom_delta < dead_zone and not box_at_edge:
            camera.stop_zoom()

        abs_delta = np.abs(delta)

        if np.all(abs_delta < dead_zone) or camera.zooming():
            # if within tolerance or currently zooming, then camera is in zoom mode if available
            # this can continue for up to 5 seconds, or as long as the box is at the edge
            if zoom_if_able:
                if zoom_delta < dead_zone and not box_at_edge:
                    camera.stop_camera()
                else:
                    # we want zoom_delta just past the dead zone, but not at the edge
                    # zoom delta normalized to % image like with pan/tilt
                    # this means constants mostly stay the same regardless of image size
                    normalized_zoom_delta = abs(zoom_delta - dead_zone / 2) / max(
                        image_width, image_height
                    )
                    control_output_z = self.z_pid(normalized_zoom_delta)

                    if abs(control_output_z) < minimum_camera_speed:
                        control_output_z = minimum_camera_speed * np.sign(
                            control_output_z
                        )

                    logger.debug(
                        f"zdelta:{normalized_zoom_delta} output:{control_output_z}"
                    )

                    # in the case where we've overshot, there's no signal to use for PID
                    # back off slowly if the box is at the edge as we likely just missed it
                    camera.zoom(
                        minimum_camera_speed * -1
                        if box_at_edge
                        else control_output_z * -1
                    )

            else:
                camera.stop_camera()
        else:
            # if not in zoom mode, then allow xy (pan/tilt) motion

            # normalize delta in terms of % for PID loop
            # this means constants mostly stay the same regardless of image size
            normalized_delta = delta / np.array([image_width, image_height])

            control_output_x = self.x_pid(normalized_delta[0])
            control_output_y = self.y_pid(normalized_delta[1])

            if abs(control_output_x) < minimum_camera_speed:
                control_output_x = minimum_camera_speed * np.sign(control_output_x)
            if abs(control_output_y) < minimum_camera_speed:
                control_output_y = minimum_camera_speed * np.sign(control_output_y)

            logger.debug(
                f"delta:{normalized_delta} output:{control_output_x} {control_output_y}"
            )

            # larger axis moves at max speed, so normalize to 100%
            speeds = abs(delta / abs_delta.max())

            # hard stop within deadzone seems to help, even though first pass will likely
            # overshoot if there's a lot of lag
            x = speeds[0] * control_output_x if abs_delta[0] > dead_zone else 0
            y = speeds[1] * control_output_y if abs_delta[1] > dead_zone else 0

            # flip movement as necessary based on settings
            # this is actually reversed (delta is backwards)
            x_modifier = 1 if flip_x_movement else -1
            y_modifier = 1 if flip_y_movement else -1

            camera.continuous_move(
                x * x_modifier, y * y_modifier, 0, simulate_variable_speed
            )

    def __del__(self):
        self.event_loop.stop()
