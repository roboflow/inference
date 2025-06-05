from concurrent.futures import ThreadPoolExecutor
import importlib
import os
from threading import Thread
import threading
import time
from typing import Dict, List, Literal, Optional, Type, Union, Tuple
from onvif2 import ONVIFService
import supervision as sv
import numpy as np


from onvif import ONVIFCamera
import asyncio

from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    AllOperationsType,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    build_operations_chain,
)
from inference.core.workflows.errors import WorkflowError
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    INTEGER_KIND,
    STRING_KIND,
    SECRET_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    FLOAT_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

# speed will start to be reduced this many tolerances from tolerance zone
SPEED_REDUCTION_TOLERANCE_MULTIPLIER = 2

PREDICTIONS_OUTPUT_KEY: str = "predictions"
MOVING_OUTPUT_KEY: str = "moving"

LONG_DESCRIPTION = """
This **ONVIF** block allows a workflow to control an ONVIF capable PTZ camera to follow a detected object.

The block returns two booleans:
* predictions - indicates whether or not it's following a valid prediction
* moving - indicates whether or not the camera is currently moving

Note that since the camera runs independently of the block, both booleans might not be updated immediately

There are two modes:

Follow:
The object it follows is the maximum confidence prediction out of all predictions passed into it. To follow
a specific object, use the appropriate filters on the predictiion object to specify the object you want to
follow. Additionally if a tracker is used, the camera will follow the tracked object until it disappears.
Additionally, zoom can be toggled to get the camera to zoom into a position.

Move to Preset:
The camera can also move to a defined preset position. The camera must support the GotoPreset service.

Note that the tracking block uses the ONVIF continuous movement service. Tracking is adjusted on each successive
workflow execution. If workflow execution stops, and the camera is currently moving, the camera will continue
moving until it reaches the limits and will no longer be following an object.

"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "ONVIF Control",
            "version": "v1",
            "short_description": "Control a PTZ camera to follow an object",
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
        description="Object predictions",
        examples=["$steps.object_detection_model.predictions"],
    )
    camera_ip: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Camera IP Address",
    )
    camera_port: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        description="Camera ONVIF Port",
    )
    camera_username: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Camera Username",
    )
    camera_password: Union[Selector(kind=[SECRET_KIND]), str] = Field(
        description="Camera Password",
    )
    movement_type: Literal["Follow", "Go To Preset"] = Field(
        description="Follow Object or Go To Preset On Execution",
    )
    zoom_if_able: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        examples=[True, False],
        description="Zoom If Able",
    )
    follow_tracker: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        examples=[True, False],
        description="Follow the track of the highest confidence prediction (Byte Tracker must be added to the workflow)",
    )
    center_tolerance: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        default=100,
        description="Camera will stop once bounding box is within this many pixels of FoV center (or border for zoom)",
    )
    default_position_preset: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Preset Name for Default Position",
    )
    move_to_position_after_idle_seconds: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        default=0,
        description="Move to the default position after this many seconds if idle (0 to disable)",
    )
    camera_update_rate_limit: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        default=1000,
        description="Minimum number of milliseconds between ONVIF movement updates",
    )
    flip_x_movement: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        examples=[True, False],
        description="Flip X movement if image is mirrored horizontally",
    )
    flip_y_movement: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        examples=[True, False],
        description="Flip Y movement if image is mirrored vertically",
    )
    movement_speed_percent: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=100,
        description="Percent of maximum speed to move camera at (0-100)",
    )
    reduce_speed_near_zone: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        examples=[True, False],
        description="Reduce speed near the tolerance zone to avoid hunting in cases of RTSP lag",
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=PREDICTIONS_OUTPUT_KEY,
                kind=[
                    BOOLEAN_KIND,
                ],
            ),
            OutputDefinition(
                name=MOVING_OUTPUT_KEY,
                kind=[
                    BOOLEAN_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

# gets the CameraWrapper from the static cameras collection
def get_camera(camera_ip:str,camera_port:int,camera_username:str,camera_password:str,max_update_rate:int,move_to_position_after_idle_seconds:int):
    global cameras
    mycam = None
    camera_key = (camera_ip,camera_port)
    if camera_key not in cameras:
        try:
            #"/usr/local/lib/python3.9/site-packages/onvif/wsdl"
            cameras[camera_key] = CameraWrapper(max_update_rate,move_to_position_after_idle_seconds)
            #mycam = ONVIFCamera(camera_ip, camera_port, camera_username, camera_password, f"{os.path.dirname(spec.origin)}/wsdl")
            cameras[camera_key].connect_camera(camera_ip, camera_port, camera_username, camera_password)
        except Exception as e:
            if camera_key in cameras:
                del cameras[camera_key]
            raise ValueError(
                f"Error connecting to camera at {camera_ip}:{camera_port}: {e}"
            )
    return cameras[camera_key]

def camera_seeking(camera_ip:str,camera_port:int) -> bool:
    global cameras
    camera_key = (camera_ip,camera_port)
    camera = cameras.get(camera_key)
    if not camera:
        return False
    return camera.seeking()

# TODO: might just be easier to save XRange/YRange
def limits(s) -> Tuple[float]:
    return (s.XRange.Min,s.XRange.Max,s.YRange.Min,s.YRange.Max)

# primarily used for rate limiting
def now() -> int:
    return int(round(time.time() * 1000))

class Limits:
    min:float
    max:float

    def __init__(self,range):
        self.min = range.Min
        self.max = range.Max

    def __repr__(self):
        return f"({self.min},{self.max})"

class VelocityLimits:
    x:Limits
    y:Limits
    z:Union[Limits,None]

    def __init__(self,x:Limits,y:Limits,z:limits):
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
    camera:Union[ONVIFCamera,None] = None
    thread:Thread
    run_loop = None
    media_profile = None
    configuration_options = None
    media_profile_token = None
    #velocity_limits: Union[Tuple[float,float,float,float],None] = None
    velocity_limits:Union[VelocityLimits,None]
    presets = None
    last_update_ms:Union[int,None] = None
    max_update_rate:int = 0
    move_to_position_after_idle_seconds:int = 0
    existing_preset_task:Union[None,asyncio.Task] = None
    _seeking:Union[bool,None] = None
    _prev_x:float = 100
    _prev_y:float = 100
    _prev_z:float = 100

    # create a new camera wrapper with an asyncio event loop
    def __init__(self,max_update_rate:int,move_to_position_after_idle_seconds:int):
        self.max_update_rate = max_update_rate
        self.move_to_position_after_idle_seconds = move_to_position_after_idle_seconds
        self.run_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=run_loop, args=(self.run_loop,))
        thread.daemon = True
        thread.start()

    def connect_camera(self,camera_ip, camera_port, camera_username, camera_password) -> asyncio.Future:
        return self.schedule(self.connect_camera_async(camera_ip, camera_port, camera_username, camera_password))

    async def connect_camera_async(self,camera_ip, camera_port, camera_username, camera_password):
        spec = importlib.util.find_spec('onvif')
        wdsl_path = f"{os.path.dirname(spec.origin)}/wsdl"
        self.camera = ONVIFCamera(camera_ip, camera_port, camera_username, camera_password, wdsl_path)
        await self.configure_async()

    # schedule a future inside the camera's event loop
    def schedule(self,cor) -> asyncio.Future:
        return asyncio.run_coroutine_threadsafe(cor, loop=self.run_loop)

    # pushes out the next scheduled reset
    def schedule_next_reset(self,preset_name:str):
        self.schedule(self.next_reset(preset_name))

    async def next_reset(self,preset_name:str):
        if self.existing_preset_task:
            self.existing_preset_task.cancel()
        self.existing_preset_task = asyncio.create_task(self.reset_task(preset_name))

    async def reset_task(self,preset_name:str):
        await asyncio.sleep(self.move_to_position_after_idle_seconds)
        print(f"camera is idle for {self.move_to_position_after_idle_seconds}s: moving to preset")
        await self.go_to_preset_async(preset_name)

    # true if movement update hasn't happened within max_update_rate
    def _can_update(self) -> bool:
        return self.last_update_ms is None or now()-self.last_update_ms>self.max_update_rate

    # this is mainly used to allow stop commands through on new zero speeds
    def save_last_speeds(self,x,y,z) -> Tuple[bool,bool]:
        x_changed = x!=self._prev_x
        y_changed = y!=self._prev_y
        z_changed = y!=self._prev_z
        self._prev_x = x
        self._prev_y = y
        self._prev_z = z
        return (x_changed,y_changed,z_changed)

    async def ptz_service(self) -> Union[None,ONVIFService]:
        """
        Creates the ONVIF PTZ service
        This has to run on every command requiring the service - a service can't be awaited twice
        """
        if not self.camera:
            return None
        return await self.camera.create_ptz_service()

    async def media_service(self) -> Union[None,ONVIFService]:
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
            raise WorkflowError(
                f"Tried to configure camera, but camera was not created"
            )

        await self.camera.update_xaddrs()
        # capabilities = await self.camera.get_capabilities() # <- could be useful in the future
        media = await self.media_service()
        self.media_profile = (await media.GetProfiles())[0]
        ptz = await self.ptz_service()
        config_request = ptz.create_type('GetConfigurationOptions')
        config_request.ConfigurationToken = self.media_profile.PTZConfiguration.token
        self.configuration_options = ptz.GetConfigurationOptions(config_request)
        config_options = (await self.configuration_options)
        #print(config_options.Spaces.__dict__)
        pan_tilt_space = config_options.Spaces.ContinuousPanTiltVelocitySpace[0]
        zoom_space = config_options.Spaces.ContinuousZoomVelocitySpace[0] if hasattr(config_options.Spaces, 'ContinuousZoomVelocitySpace') else None
        self.velocity_limits = VelocityLimits(
            x = Limits(pan_tilt_space.XRange),
            y = Limits(pan_tilt_space.YRange),
            z = Limits(zoom_space.XRange if zoom_space else None)
        )

        #self.velocity_limits = limits(config_options.Spaces.ContinuousPanTiltVelocitySpace[0])
        #self.zoom_velocity_limits = limits(config_options.Spaces.ContinuousZoomVelocitySpace[0])
        #print(f"camera velocity limits: {self.velocity_limits}")
        self.media_profile_token = self.media_profile.token
        presets = (await ptz.GetPresets({'ProfileToken':self.media_profile_token}))
        # reconfigure into a dict keyed by preset name
        self.presets = {preset['Name']:preset for preset in presets}

    def seeking(self):
        return self._seeking

    # stops the camera movement
    def stop_camera(self,preset_name:Union[None,str]):
        # don't stop if already stopped as stop commands aren't rate limited
        if self._seeking or self._seeking is None:
            self.continuous_move(0,0,0,0)
            if preset_name:
                self.schedule_next_reset(preset_name)
            '''
            # not all cameras support stop
            try:
                ptz = await self.ptz_service()
                ptz.stop()
            except Exception as e:
                pass
            '''

        self._seeking = False

    def go_to_preset(self, preset_name:str):
        self.schedule(self.go_to_preset_async(preset_name))

    async def go_to_preset_async(self, preset_name:str):
        """
        Tells the camera to move to a preset
        This is not rate limited - all commands will be sent to the camera

        Args:
            preset_name: The preset name to move to (this varies by camera)
        """

        if self.media_profile_token is None:
            await self.configure_async()

        preset = self.presets.get(preset_name)
        if not preset:
            raise ValueError(
                f"Camera does not have preset \"{preset_name}\" - valid presets are {list(self.presets.keys())}"
            )

        ptz = await self.ptz_service()
        request = ptz.create_type('GotoPreset')
        request.ProfileToken = self.media_profile_token
        request.PresetToken = preset['token']
        request.Speed = {
            'PanTilt': {
                'x': 1.0,
                'y': 1.0
            },
            'Zoom': {
                'x':1.0
            }
        }

        self._seeking = False
        await ptz.GotoPreset(request)

    # tells the camera to move at a continuous velocity
    def continuous_move(self,x:float,y:float,z:float,movement_speed_percent:float):
        self.schedule(self.continuous_move_async(x,y,z,movement_speed_percent))

    # x and y are velocities from -1 to 1
    async def continuous_move_async(self,x:float,y:float,z:float,movement_speed_percent:float):
        """
        Tells the camera to move at a continuous velocity, or 0 to stop
        Note this is rate limited, some commands will be ignored

        Args:
            x: The x velocity as -1 to 1 where -1 and 1 are maximums
            y: The y velocity as -1 to 1 where -1 and 1 are maximums
            z: The zoom velocity as -1 to 1 where -1 and 1 are maximums
            movement_speed_percent: percent of maximum movement speed as 0-1
        """

        if self.media_profile_token is None:
            await self.configure_async()

        # try to avoid hunting by allowing immediate stop
        x_changed, y_changed, z_changed = self.save_last_speeds(x,y,z)
        #print(f"changed: {x_changed} {y_changed} {z_changed} {x} {y} {z}")
        if (x==0 and x_changed) or (y==0 and y_changed) or (y==z and z_changed):
            #print(f"forcing stop {self._can_update()}")
            pass
        elif not self._can_update():
            return

        ptz = await self.ptz_service()

        #https://www.onvif.org/onvif/ver20/ptz/wsdl/ptz.wsdl#op.AbsoluteMove

        request = ptz.create_type('ContinuousMove')
        request.ProfileToken = self.media_profile_token

        limits = self.velocity_limits

        #print(f"x:{x} y:{y} z:{z} limits:{limits} speed_percent:{movement_speed_percent}")
        # normalize to camera's velocity limits
        x_limit = limits.x.min if x<0 else limits.x.max
        y_limit = limits.y.min if x<0 else limits.y.max
        z_limit = limits.z.min if x<0 else limits.z.max

        x = abs(x_limit) * x * movement_speed_percent
        y = abs(y_limit) * y * movement_speed_percent
        z = abs(z_limit) * z * movement_speed_percent

        request.Velocity = {
            'PanTilt': {
                'x': x,
                'y': y
            },
            'Zoom': {
                'x':z
            }
        }

        print(f"ptz continuous move update: {x},{y},{z}")

        # Execute the movement
        await ptz.ContinuousMove(request)
        self.last_update_ms = now()
        self._seeking = True

    def __del__(self):
        self.stop_camera(None)
        self.run_loop.stop()

# pool of camera services can can be used across blocks
cameras: Dict[Tuple[str,int],CameraWrapper] = {}

class ONVIFSinkBlockV1(WorkflowBlock):

    def __init__(
        self,
        step_execution_mode: StepExecutionMode,
    ):
        self.bacnet_app = None
        self._step_execution_mode = step_execution_mode

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["step_execution_mode"]

    def run(
        self,
        predictions: sv.Detections,
        camera_ip: str,
        camera_port: int,
        camera_username: str,
        camera_password: str,
        movement_type: str,
        default_position_preset: Union[str,None],
        zoom_if_able: bool,
        follow_tracker: bool,
        center_tolerance: int,
        camera_update_rate_limit:int,
        flip_y_movement:int,
        flip_x_movement:int,
        movement_speed_percent:float,
        move_to_position_after_idle_seconds:int,
        reduce_speed_near_zone:bool
    ) -> BlockResult:

        if self._step_execution_mode != StepExecutionMode.LOCAL:
            raise ValueError(
                "Inference must be run locally for the ONVIF block"
            )

        # disable the stop preset if necessary
        stop_preset = default_position_preset if move_to_position_after_idle_seconds else None

        if movement_type=="Follow":

            if len(predictions.xyxy)==0:
                # get/create the camera first so that we can move it to the preset
                if stop_preset:
                    get_camera(camera_ip,camera_port,camera_username,camera_password,camera_update_rate_limit,move_to_position_after_idle_seconds)
                self.stop_camera(camera_ip, camera_port, stop_preset)
                return {PREDICTIONS_OUTPUT_KEY:False,MOVING_OUTPUT_KEY:camera_seeking(camera_ip,camera_port)}

            # get max confidence prediction
            max_confidence = predictions.confidence.max()
            max_confidence_prediction = predictions[predictions.confidence==max_confidence][0]
            self.async_move(camera_ip, camera_port, camera_username, camera_password, max_confidence_prediction,zoom_if_able,center_tolerance,default_position_preset,camera_update_rate_limit,move_to_position_after_idle_seconds,flip_x_movement,flip_y_movement,movement_speed_percent/100.0,reduce_speed_near_zone,stop_preset)

        elif movement_type=="Go To Preset":
            self.go_to_preset(camera_ip, camera_port, camera_username, camera_password, default_position_preset,camera_update_rate_limit,move_to_position_after_idle_seconds)

        return {PREDICTIONS_OUTPUT_KEY:True,MOVING_OUTPUT_KEY:camera_seeking(camera_ip,camera_port)}

    def stop_camera(self,camera_ip:str,camera_port:int,stop_preset:str):
        global cameras
        camera = cameras.get((camera_ip,camera_port))
        if camera:
            camera.stop_camera(stop_preset)

    def async_move(self,camera_ip:str,camera_port:int,camera_username:str,camera_password:str,prediction:OBJECT_DETECTION_PREDICTION_KIND,zoom_if_able:bool,center_tolerance:int,preset:str,max_update_rate:int,move_to_position_after_idle_seconds:int,flip_x_movement:bool,flip_y_movement:bool,movement_speed_percent:float,reduce_speed_near_zone:bool,stop_preset:str):

        camera = get_camera(camera_ip,camera_port,camera_username,camera_password,max_update_rate,move_to_position_after_idle_seconds)

        # use as stop command, more generic
        #await camera.continuous_move(0,0)

        # adjust speed so that camera moves proportionally towards bounding box
        image_dimensions = prediction.data['root_parent_dimensions']
        image_center = image_dimensions/2
        xyxy = prediction.xyxy
        (x1,y1,x2,y2) = tuple(xyxy[0])
        center_point = np.array([x1+(x2-x1)/2,y1+(y2-y1)/2])

        # delta represents the amount of relative movement to get the object to the center
        zoom_delta = min([x1,image_dimensions[0][0]-x1,y1,image_dimensions[0][1]-y1])
        # make the deltas x, y, zoom
        delta = (center_point-image_center)[0]

        print(f"object center:{center_point} delta:{delta}")

        #print(f"delta:{delta}")
        abs_delta = np.abs(delta)
        if np.all(abs_delta < center_tolerance):
            #await camera.stop_camera(False)
            # zoom is only allowed if the camera is stopped - helps to minimize hunting
            # this could be integrated into the pan/tilt movement with better motion control
            if zoom_if_able:
                if zoom_delta>center_tolerance:
                    print(f"zoom delta: {zoom_delta}")
                    z_speed = 0.5 if delta[1]<SPEED_REDUCTION_TOLERANCE_MULTIPLIER*center_tolerance and reduce_speed_near_zone else 1
                    camera.continuous_move(0,0,z_speed,movement_speed_percent)
                else:
                    camera.stop_camera(stop_preset)
            else:
                camera.stop_camera(stop_preset)
        else:
            # larger axis moves at max speed, so normalize to 100%
            speeds = delta/abs_delta.max()

            # reduce speed by half if close to the zone
            # this is a bit simpler than a P/PID control for v1
            x_reduction_factor = 0.5 if abs_delta[0]<SPEED_REDUCTION_TOLERANCE_MULTIPLIER*center_tolerance and reduce_speed_near_zone else 1
            y_reduction_factor = 0.5 if abs_delta[1]<SPEED_REDUCTION_TOLERANCE_MULTIPLIER*center_tolerance and reduce_speed_near_zone else 1

            # set speed to 0 for axis within tolerance
            # this might not be necessary and slows the process down a bit, but trying to avoid hunting
            # with a good PID loop, this will no longer be necessary
            x = speeds[0]*x_reduction_factor if abs_delta[0]>center_tolerance else 0
            y = speeds[1]*y_reduction_factor if abs_delta[1]>center_tolerance else 0

            # flip movement as necessary based on settings
            x_modifier = -1 if flip_x_movement else 1
            y_modifier = -1 if flip_y_movement else 1

            camera.continuous_move(x*x_modifier,y*y_modifier,0,movement_speed_percent)


    def go_to_preset(self,camera_ip:str,camera_port:int,camera_username:str,camera_password:str,preset:str,max_update_rate:int,move_to_position_after_idle_seconds:int):
        camera = get_camera(camera_ip,camera_port,camera_username,camera_password,max_update_rate,move_to_position_after_idle_seconds)
        camera.stop_camera(None)
        camera.go_to_preset(preset)
