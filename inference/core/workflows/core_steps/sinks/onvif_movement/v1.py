import importlib
import os
import time
from typing import Dict, List, Literal, Optional, Type, Union, Tuple
import supervision as sv
import numpy as np


from onvif import ONVIFCamera
import asyncio

from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    AllOperationsType,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    build_operations_chain,
)
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
    proportional_constant: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=30,
        description="Speed is further reduced by this percentage (0-100) as the object gets closer to the tolerance to avoid hunting",
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
async def get_camera(camera_ip:str,camera_port:int,camera_username:str,camera_password:str,max_update_rate:int):
    global cameras
    mycam = None
    camera_key = (camera_ip,camera_port)
    if camera_key not in cameras:
        try:
            spec = importlib.util.find_spec('onvif')
            #"/usr/local/lib/python3.9/site-packages/onvif/wsdl"
            mycam = ONVIFCamera(camera_ip, camera_port, camera_username, camera_password, f"{os.path.dirname(spec.origin)}/wsdl")
            cameras[camera_key] = CameraWrapper(mycam,max_update_rate)
            await cameras[camera_key].configure()
        except Exception as e:
            if camera_key in cameras:
                del cameras[camera_key]
            raise ValueError(
                f"Error connecting to camera at {camera_ip}:{camera_port}: {e}"
            )
    return cameras[camera_key]

def camera_moving(camera_ip:str,camera_port:int) -> bool:
    global cameras
    camera_key = (camera_ip,camera_port)
    camera = cameras.get(camera_key)
    if not camera:
        return False
    return camera.moving()

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

# camera wrapper is used to store config info so that we don't have
# to keep querying it from the camera on successive commands
class CameraWrapper:
    camera:ONVIFCamera
    media_profile = None
    configuration_options = None
    media_profile_token = None
    #velocity_limits: Union[Tuple[float,float,float,float],None] = None
    velocity_limits:Union[VelocityLimits,None]
    presets = None
    last_update_ms:Union[int,None] = None
    max_update_rate:int = 0
    _moving:Union[bool,None] = None
    _prev_x:float = 100
    _prev_y:float = 100
    _prev_z:float = 100

    def __init__(self,camera:ONVIFCamera,max_update_rate:int):
        self.camera = camera
        self.max_update_rate = max_update_rate

    # true if movement update hasn't happened within max_update_rate
    def _can_update(self):
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

    async def ptz_service(self):
        """
        Creates the ONVIF PTZ service
        This has to run on every command requiring the service - a service can't be awaited twice
        """
        return await self.camera.create_ptz_service()

    async def media_service(self):
        """
        Creates the ONVIF media service
        This is primarily used to get the media token
        """
        return await self.camera.create_media_service()

    async def configure(self):
        """
        Does initial configuration and gathers all camera info
        Doesn't currently run in init since it needs to be awaited
        """
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

    def moving(self):
        return self._moving

    # stops the camera movement
    async def stop_camera(self,zoom_out:bool):
        # don't stop if already stopped as stop commands aren't rate limited
        if self._moving or self._moving is None:
            print("stopping camera")
            # make sure movement speed percent is 100% in this case for zoom
            await self.continuous_move(0,0,-1 if zoom_out else 0,100)
            # not all cameras support stop
            if not zoom_out:
                try:
                    ptz = await self.ptz_service()
                    ptz.stop()
                except Exception as e:
                    pass

        self._moving = False

    async def go_to_preset(self, preset_name:str):
        """
        Tells the camera to move to a preset
        This is not rate limited - all commands will be sent to the camera

        Args:
            preset_name: The preset name to move to (this varies by camera)
        """

        if self.media_profile_token is None:
            await self.configure()

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

        await ptz.GotoPreset(request)
        self._moving = True

    # tells the camera to move at a continuous velocity

    # x and y are velocities from -1 to 1
    async def continuous_move(self,x:float,y:float,z:float,movement_speed_percent:float):
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
            await self.configure()

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

        print(f"x:{x} y:{y} z:{z} limits:{limits} speed_percent:{movement_speed_percent}")
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

        #self.save_last_speeds(x,y)

        print(f"ptz continuous move update: {x},{y},{z}")

        # Execute the movement
        await ptz.ContinuousMove(request)
        self.last_update_ms = now()
        self._moving = True

# pool of camera services can can be used across blocks
cameras: Dict[Tuple[str,int],CameraWrapper] = {}

class ONVIFSinkBlockV1(WorkflowBlock):

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
        default_position_preset: str,
        zoom_if_able: bool,
        follow_tracker: bool,
        center_tolerance: int,
        camera_update_rate_limit:int,
        flip_y_movement:int,
        flip_x_movement:int,
        movement_speed_percent:float,
        move_to_position_after_idle_seconds:int,
        proportional_constant:float
    ) -> BlockResult:

        if movement_type=="Follow":


            if len(predictions.xyxy)==0:
                # TODO: going to a preset is ok, but we don't want to do that if just one frame is missing a prediction - maybe add some time factor (ex. no prediction in 10 seconds?)
                #asyncio.run(self.go_to_preset(camera_ip, camera_port, camera_username, camera_password, default_position_preset,camera_update_rate_limit))
                #print(f"No predictions to move the camera to, moving to preset \"{default_position_preset}\"")
                asyncio.run(self.stop_camera(camera_ip, camera_port, camera_username, camera_password,camera_update_rate_limit,True if zoom_if_able else False))
                return {PREDICTIONS_OUTPUT_KEY:False,MOVING_OUTPUT_KEY:camera_moving(camera_ip,camera_port)}


            # get max confidence prediction
            max_confidence = predictions.confidence.max()
            max_confidence_prediction = predictions[predictions.confidence==max_confidence][0]
            asyncio.run(self.async_move(camera_ip, camera_port, camera_username, camera_password, max_confidence_prediction,zoom_if_able,center_tolerance,default_position_preset,camera_update_rate_limit,flip_x_movement,flip_y_movement,movement_speed_percent/100.0,proportional_constant/100.0))

        elif movement_type=="Go To Preset":
            asyncio.run(self.go_to_preset(camera_ip, camera_port, camera_username, camera_password, default_position_preset,camera_update_rate_limit))

        return {PREDICTIONS_OUTPUT_KEY:True,MOVING_OUTPUT_KEY:camera_moving(camera_ip,camera_port)}

    async def stop_camera(self,camera_ip:str,camera_port:int,camera_username:str,camera_password:str,max_update_rate:int,zoom_out:bool):
        camera = await get_camera(camera_ip,camera_port,camera_username,camera_password,max_update_rate)
        await camera.stop_camera(zoom_out)

    async def async_move(self,camera_ip:str,camera_port:int,camera_username:str,camera_password:str,prediction:OBJECT_DETECTION_PREDICTION_KIND,zoom_if_able:bool,center_tolerance:int,preset:str,max_update_rate:int,flip_x_movement:bool,flip_y_movement:bool,movement_speed_percent:float,proportional_constant:float):

        camera = await get_camera(camera_ip,camera_port,camera_username,camera_password,max_update_rate)

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

        print(f"object center: xyxy:{tuple(xyxy[0])} {center_point} {image_dimensions} {image_center} delta:{delta}")

        #print(f"delta:{delta}")
        if np.all(np.abs(delta) < center_tolerance):
            #await camera.stop_camera(False)
            # zoom is only allowed if the camera is stopped - helps to minimize hunting
            # this could be integrated into the pan/tilt movement with better motion control
            if zoom_if_able:
                '''
                z_reduction_factor = 0 if proportional_constant==0 else max(min(0.5,abs((zoom_delta-center_tolerance)/zoom_delta)),0)*proportional_constant
                z = (1-z_reduction_factor) if abs(zoom_delta)>center_tolerance else 1
                if zoom_delta>center_tolerance:
                    print(f"z movement speeds:{z} factor: {z_reduction_factor}")

                    await camera.continuous_move(0,0,z,movement_speed_percent)
                else:
                '''
                await camera.stop_camera(False)
            else:
                await camera.stop_camera(False)
        else:
            # larger axis moves at max speed, so normalize to 100%
            speeds = delta/np.abs(delta).max()

            # This is simple psuedo-P controller, but could be improved a lot
            # v2 of this block should focus on motion control
            x_multiplier = delta[0]/center_tolerance
            y_multiplier = delta[1]/center_tolerance
            x_reduction_factor = 0 if proportional_constant==0 or x_multiplier>2 else min(max(abs((x_multiplier-1)*proportional_constant),0),0.5)
            y_reduction_factor = 0 if proportional_constant==0 or y_multiplier>2 else min(max(abs((y_multiplier-1)*proportional_constant),0),0.5)

            x = speeds[0]*(1-x_reduction_factor)
            y = speeds[1]*(1-y_reduction_factor)

            print(f"xy movement speeds:{speeds} {x} {y} factor: {x_reduction_factor} {y_reduction_factor}")

            # flip movement as necessary
            x_modifier = -1 if flip_x_movement else 1
            y_modifier = -1 if flip_y_movement else 1

            await camera.continuous_move(x*x_modifier,y*y_modifier,0,movement_speed_percent)


    async def go_to_preset(self,camera_ip:str,camera_port:int,camera_username:str,camera_password:str,preset:str,max_update_rate:int):
        camera = await get_camera(camera_ip,camera_port,camera_username,camera_password,max_update_rate)
        await camera.stop_camera(False)
        await camera.go_to_preset(preset)

    # if the block is destroyed, don't let the camera keep drifting
    def __del__(self):
        asyncio.run(self.stop_camera(False))
