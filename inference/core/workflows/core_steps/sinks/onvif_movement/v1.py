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

OUTPUT_KEY: str = "success"

LONG_DESCRIPTION = """
This **ONVIF** block allows a workflow to control an ONVIF capable PTZ camera to follow a detected object.

The object it follows is the maximum confidence prediction out of all predictions passed into it. To follow
a specific object, use the appropriate filters on the predictiion object to specify the object you want to
follow.

The camera can also move to a defined preset position. This is done automatically when there are no objects
in the prediction.

The block returns booleans indicating success (it's communicating with the camera), whether or not it's
moving, and whether or not it's currently tracking an object.

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
    zoom_if_able: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        examples=[True, False],
        description="Zoom If Able",
    )
    follow_tracker: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        examples=[True, False],
        description="Follow the track of the highest confidence prediction (if available)",
    )
    center_tolerance: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        default=100,
        description="Camera will stop once bounding box is within this many pixels of FoV center (or border for zoom)",
    )
    movement_type: Literal["Follow", "Go To Preset"] = Field(
        description="Follow Object or Go To Preset On Execution",
    )
    default_position_preset: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Preset Name for Default Position",
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
    movement_speed_percent: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=1.0,
        description="Percent of maximum speed to move camera at",
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_KEY,
                kind=[
                    BOOLEAN_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

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

def limits(s):
    return (s.XRange.Min,s.XRange.Max,s.YRange.Min,s.YRange.Max)

def now():
    return int(round(time.time() * 1000))

class CameraWrapper:
    camera:ONVIFCamera
    media_profile = None
    configuration_options = None
    #config_token = None
    media_profile_token = None
    velocity_limits: Union[Tuple[float,float,float,float],None] = None
    presets = None
    last_update_ms:Union[int,None] = None
    max_update_rate:int = 0
    _moving: bool = False

    def __init__(self,camera:ONVIFCamera,max_update_rate:int):
        self.camera = camera
        self.max_update_rate = max_update_rate
        #self.media = media

    # true if movement update hasn't happened within max_update_rate
    def _can_update(self):
        return self.last_update_ms is None or now()-self.last_update_ms>self.max_update_rate

    async def ptz_service(self):
        return await self.camera.create_ptz_service()

    async def media_service(self):
        return await self.camera.create_media_service()

    async def configure(self):
        await self.camera.update_xaddrs()
        capabilities = await self.camera.get_capabilities()
        #print(f"Camera capabilities: {capabilities}")
        media = await self.media_service()
        self.media_profile = (await media.GetProfiles())[0]
        ptz = await self.ptz_service()
        config_request = ptz.create_type('GetConfigurationOptions')
        config_request.ConfigurationToken = self.media_profile.PTZConfiguration.token
        self.configuration_options = ptz.GetConfigurationOptions(config_request)
        config_options = (await self.configuration_options)
        #self.abs_pan_tilt_position_space = config_options.Spaces.AbsolutePanTiltPositionSpace[0]
        self.velocity_limits = limits(config_options.Spaces.ContinuousPanTiltVelocitySpace[0])
        #self.pan_tilt_speed_space = config_options.Spaces.PanTiltSpeedSpace[0]
        self.media_profile_token = self.media_profile.token
        presets = (await ptz.GetPresets({'ProfileToken':self.media_profile_token}))
        # reconfigure into a dict keyed by preset name
        self.presets = {preset['Name']:preset for preset in presets}
        #print(self.presets)

    async def stop_camera(self):
        # don't stop if already stopped as stop commands aren't rate limited
        if self._moving:
            await self.continuous_move(0,0,False,0)
            # not all cameras support stop
            try:
                ptz = await self.ptz_service()
                ptz.stop()
            except Exception as e:
                pass
        self._moving = False

    async def go_to_preset(self, preset_name:str):
        if self.media_profile_token is None:
            await self.configure()

        if not self._can_update():
            return

        preset = self.presets.get(preset_name)
        if not preset:
            raise ValueError(
                f"Camera does not have preset \"{preset_name}\" - valid presets are {list(self.presets.keys())}"
            )

        ptz = await self.ptz_service()
        request = ptz.create_type('GotoPreset')
        #request = ptz.create_type('GotoHomePosition')
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

        #print(request)

        await ptz.GotoPreset(request)
        self._moving = True

    async def continuous_move(self,x:float,y:float,zoom_if_able:bool,movement_speed_percent:float):
        if self.media_profile_token is None:
            await self.configure()

        if not self._can_update():
            return

        ptz = await self.ptz_service()

        #https://www.onvif.org/onvif/ver20/ptz/wsdl/ptz.wsdl#op.AbsoluteMove

        request = ptz.create_type('ContinuousMove')
        request.ProfileToken = self.media_profile_token

        request.Velocity = {
            'PanTilt': {
                'x': x, # TODO: should be % of velocity limits
                'y': y
            },
            'Zoom': {
                'x':1.0
            }
        }

        print(f"ptz continuous move update: {x},{y}")

        # Execute the movement
        await ptz.ContinuousMove(request)
        self.last_update_ms = now()
        self._moving = True

    # TODO: this doesn't work - need to move outside of event loop
    #def __del__(self):
    #    asyncio.run(self.stop_camera())


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
        movement_speed_percent:float
    ) -> BlockResult:

        if movement_type=="Follow":

            if len(predictions.xyxy)==0:
                asyncio.run(self.go_to_preset(camera_ip, camera_port, camera_username, camera_password, default_position_preset,camera_update_rate_limit))
                print(f"No predictions to move the camera to, moving to preset \"{default_position_preset}\"")
                return {OUTPUT_KEY:False}

            # get max confidence prediction
            max_confidence = predictions.confidence.max()
            max_confidence_prediction = predictions[predictions.confidence==max_confidence][0]
            asyncio.run(self.async_move(camera_ip, camera_port, camera_username, camera_password, max_confidence_prediction,zoom_if_able,center_tolerance,default_position_preset,camera_update_rate_limit,flip_x_movement,flip_y_movement,movement_speed_percent))

        elif movement_type=="Go To Preset":
            asyncio.run(self.go_to_preset(camera_ip, camera_port, camera_username, camera_password, default_position_preset,camera_update_rate_limit))

        return {OUTPUT_KEY:False}

    async def async_move(self,camera_ip:str,camera_port:int,camera_username:str,camera_password:str,prediction:OBJECT_DETECTION_PREDICTION_KIND,zoom_if_able:bool,center_tolerance:int,preset:str,max_update_rate:int,flip_x_movement:bool,flip_y_movement:bool,movement_speed_percent:float):

        camera = await get_camera(camera_ip,camera_port,camera_username,camera_password,max_update_rate)

        # use as stop command, more generic
        #await camera.continuous_move(0,0)

        # adjust speed so that camera moves proportionally towards bounding box
        image_dimensions = prediction.data['root_parent_dimensions']
        image_center = image_dimensions/2
        xyxy = prediction.xyxy
        centers = (xyxy[:, :2] + xyxy[:, 2:]) / 2
        center_point = centers[0]


        delta = image_center-center_point

        print(f"delta:{delta}")
        if abs(delta[0][0])<center_tolerance and abs(delta[0][1])<center_tolerance:
            camera.stop_camera()
        else:
            # larger axis moves at max speed, so normalize to 100%
            speeds = delta[0]/delta.max()


            print(f"movement speeds:{speeds}")

            # goal here will be to have the workflow do a continuous move & iterate to get the bounding box in position
            # might be necessary to start slow & calibrate x/y translation from pixel to speed
            # once it's in position, this speed should change to 0,0
            x_modifier = -1 if flip_x_movement else 1
            y_modifier = -1 if flip_y_movement else 1
            await camera.continuous_move(speeds[0]*x_modifier,speeds[1]*y_modifier,zoom_if_able,movement_speed_percent)


    async def go_to_preset(self,camera_ip:str,camera_port:int,camera_username:str,camera_password:str,preset:str,max_update_rate:int):
        camera = await get_camera(camera_ip,camera_port,camera_username,camera_password,max_update_rate)
        await camera.stop_camera()
        await camera.go_to_preset(preset)


