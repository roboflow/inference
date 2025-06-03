from typing import Dict, List, Literal, Optional, Type, Union, Tuple
import supervision as sv

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
    FLOAT_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    ROBOFLOW_API_KEY_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    ROBOFLOW_PROJECT_KIND,
    STRING_KIND,
    SECRET_KIND,
    TOP_CLASS_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
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

The camera can also move to a defined preset, which should be done when there are no objects within a field
of view.

The block returns booleans indicating success (it's communicating with the camera), whether or not it's
moving, and whether or not it's currently tracking an object.

Note that the tracking block uses the ONVIF continuous movement service. Tracking is adjusted on each successive
workflow execution. If workflow execution stops, and the camera is currently moving, the camera will continue
moving until it reaches the limits and will no longer be following an object.

"""

QUERY_PARAMS_KIND = [
    STRING_KIND,
    INTEGER_KIND,
    FLOAT_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    ROBOFLOW_PROJECT_KIND,
    ROBOFLOW_API_KEY_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    LIST_OF_VALUES_KIND,
    BOOLEAN_KIND,
    TOP_CLASS_KIND,
]
HEADER_KIND = [
    STRING_KIND,
    INTEGER_KIND,
    FLOAT_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    ROBOFLOW_PROJECT_KIND,
    ROBOFLOW_API_KEY_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    BOOLEAN_KIND,
    TOP_CLASS_KIND,
]


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
    #predictions: OBJECT_DETECTION_PREDICTION_KIND = Field(
    #    description="Object Detection Prediction",
    #)
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
    movement_type: Literal["Follow Tracker", "Go To Preset"] = Field(
        description="Zoom into object or reset",
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

async def get_camera(camera_ip:str,camera_port:int,camera_username:str,camera_password:str):
    global cameras
    mycam = None
    camera_key = (camera_ip,camera_port)
    if camera_key not in cameras:
        mycam = ONVIFCamera(camera_ip, camera_port, camera_username, camera_password, "/usr/local/lib/python3.9/site-packages/onvif/wsdl")
        cameras[camera_key] = CameraWrapper(mycam)
        await cameras[camera_key].configure()
    return cameras[camera_key]

def limits(s):
    return (s.XRange.Min,s.XRange.Max,s.YRange.Min,s.YRange.Max)

class CameraWrapper:
    camera:ONVIFCamera
    media_profile = None
    configuration_options = None
    _active:bool = False
    config_token = None
    media_profile_token = None
    velocity_limits: Union[Tuple[float,float,float,float],None] = None

    def __init__(self,camera:ONVIFCamera):
        self.camera = camera
        #self.media = media

    async def ptz_service(self):
        return await self.camera.create_ptz_service()

    async def media_service(self):
        return await self.camera.create_media_service()

    def activate(self):
        self._active = True

    def is_active(self):
        return self._active

    async def configure(self):
        await self.camera.update_xaddrs()
        capabilities = await self.camera.get_capabilities()
        #print(f"Camera capabilities: {capabilities}")
        media = await self.media_service()
        self.media_profile = (await media.GetProfiles())[0]
        ptz = await self.ptz_service()
        request = ptz.create_type('GetConfigurationOptions')
        self.media_profile_token = self.media_profile.token
        self.config_token = self.media_profile.PTZConfiguration.token
        request.ConfigurationToken = self.config_token
        self.configuration_options = ptz.GetConfigurationOptions(request)
        config_options = (await self.configuration_options)
        #self.abs_pan_tilt_position_space = config_options.Spaces.AbsolutePanTiltPositionSpace[0]
        self.velocity_limits = limits(config_options.Spaces.ContinuousPanTiltVelocitySpace[0])
        #self.pan_tilt_speed_space = config_options.Spaces.PanTiltSpeedSpace[0]


    async def continuous_move(self,x:float,y:float):
        if self.media_profile_token is None:
            await self.configure()

        ptz = await self.ptz_service()

        #nodes = await ptz.GetNodes()
        #print(nodes)


        #https://www.onvif.org/onvif/ver20/ptz/wsdl/ptz.wsdl#op.AbsoluteMove
        '''
        request = ptz.create_type('AbsoluteMove')
        request.ProfileToken = self.media_profile_token

        request.Position = {
            'ProfileToken': self.media_profile_token,
            'PanTilt': {
                'x': 0.5,
                'y': 0.5
            },
            'Zoom': {
                'x':1.0
            }
        }

        # Execute the movement
        print(await ptz.AbsoluteMove(request))
        '''

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

        # Execute the movement
        await ptz.ContinuousMove(request)


# pool of camera services can can be used across blocks
cameras: Dict[Tuple[str,int],CameraWrapper] = {}

class ONVIFSinkBlockV1(WorkflowBlock):

    camera_ip = None
    camera_port = None
    camera_username = None
    camera_password = None

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
    ) -> BlockResult:

        self.camera_ip = camera_ip
        self.camera_port = camera_port
        self.camera_username = camera_username
        self.camera_password = camera_password


        if movement_type=="Follow Tracker":

            if not predictions:
                # TODO: move to a preset here
                print("No predictions to move the camera to")
                return {OUTPUT_KEY:False}

            # get max confidence prediction
            max_confidence = predictions.confidence.max()
            max_confidence_prediction = any(predictions[predictions.confidence==max_confidence])

            if max_confidence_prediction:
                asyncio.run(self.async_move(max_confidence_prediction))
            else:
                # TODO: move to a preset here
                print("No max confidence prediction")
                return {OUTPUT_KEY:False}

        elif movement_type=="Go To Preset":
            asyncio.run(self.async_move(predictions.xyxy[0]))

        return {OUTPUT_KEY:False}

    async def async_move(self,xyxy):

        camera = await get_camera(self.camera_ip,self.camera_port,self.camera_username,self.camera_password)

        # use as stop command, more generic
        #await camera.continuous_move(0,0)

        # goal here will be to have the workflow do a continuous move & iterate to get the bounding box in position
        # might be necessary to start slow & calibrate x/y translation from pixel to speed
        # once it's in position, this speed should change to 0,0
        await camera.continuous_move(-0.5,-0.5)
