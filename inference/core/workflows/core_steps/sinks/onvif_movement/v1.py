from typing import Dict, List, Literal, Optional, Type, Union
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
This **ONVIF** block allows a workflow to control an ONVIF capable PTZ camera to zoom into a detected region
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
            "short_description": "Control a PTZ camera using workflow results",
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
    movement_type: Literal["To Bounding Box", "Follow Tracker", "Go To Position"] = Field(
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

class CameraWrapper:
    camera:ONVIFCamera
    media_profile = None
    configuration_options = None
    _active:bool = False
    x_min:float
    x_max:float
    y_min:float
    y_max:float
    config_token = None
    media_profile_token = None
    abs_pan_tilt_position_space = None

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

    async def get_limits(self):
        s = self.abs_pan_tilt_position_space
        self.x_min = s.XRange.Min
        self.x_max = s.XRange.Max
        self.y_min = s.YRange.Min
        self.y_max = s.YRange.Max

    async def configure(self):
        await self.camera.get_capabilities()
        media = await self.media_service()
        self.media_profile = (await media.GetProfiles())[0]
        ptz = await self.ptz_service()

        request = ptz.create_type('GetConfigurationOptions')
        self.media_profile_token = self.media_profile.token
        self.config_token = self.media_profile.PTZConfiguration.token
        request.ConfigurationToken = self.config_token
        self.configuration_options = ptz.GetConfigurationOptions(request)
        self.abs_pan_tilt_position_space = (await self.configuration_options).Spaces.AbsolutePanTiltPositionSpace[0]
        await self.get_limits()

    async def move_by_percent(self,x:float,y:float):
        ptz = await self.ptz_service()

        request = ptz.create_type('AbsoluteMove')

        request = {
            'ProfileToken':self.media_profile_token,
            #'ConfigurationToken':media_profile.PTZConfiguration.token,
            'Position': {
                #'Zoom': {
                #    'x': 2.0,
                #}
                'PanTilt': {
                    'space':self.abs_pan_tilt_position_space.URI,
                    'x': 0.5,
                    'y': 0.5
                }
            },
            #'Speed': {
            #    'Zoom': 1.0
            #}
        }

        print(f"move request {self.abs_pan_tilt_position_space.URI}")
        # Move the camera to the new zoom position
        await ptz.AbsoluteMove(request)


# pool of camera services can can be used across blocks
cameras: Dict[str,CameraWrapper] = {}

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


        if movement_type=="To Bounding Box" or movement_type=="Follow Tracker":

            if len(predictions)!=1:
                print("Prediction length for ONVIF input must be 1")
                return {OUTPUT_KEY:False}

            asyncio.run(self.async_zoom(predictions.xyxy[0]))
            #self.async_zoom(predictions.xyxy[0])


        return {OUTPUT_KEY:False}

    async def async_zoom(self,xyxy):

        global cameras
        mycam = None
        if self.camera_ip not in cameras:
            mycam = ONVIFCamera(self.camera_ip, self.camera_port, self.camera_username, self.camera_password, "/usr/local/lib/python3.9/site-packages/onvif/wsdl")
            cameras[self.camera_ip] = CameraWrapper(mycam)
            await cameras[self.camera_ip].configure()
        camera = cameras[self.camera_ip]

        try:
            await camera.move_by_percent(0.5,0.5)

        except Exception as e:
            print(f"ONVIF Error: {e}")
        #finally:
        #    if mycam is not None:
        #        await mycam.close()
