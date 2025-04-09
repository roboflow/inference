from typing import List, Literal, Optional, Type, Union

import cv2 as cv
import numpy as np
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    IMAGE_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_CALIBRATED_IMAGE_KEY: str = "calibrated_image"
LONG_DESCRIPTION = """
This block uses the OpenCV `calibrateCamera` function to remove lens distortions from an image.
Please refer to OpenCV documentation where camera calibration methodology is described:
https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

This block requires following parameters in order to perform the calibration:
Lens focal length along the x-axis and y-axis (fx, fy)
Lens optical centers along the x-axis and y-axis (cx, cy)
Radial distortion coefficients (k1, k2, k3)
Tangential distortion coefficients (p1, p2)

Based on above parameters, camera matrix will be built as follows:
[[fx 0  cx]
[ 0 fy cy]
[ 0  0  1 ]]

Distortions coefficient will be passed as 5-tuple (k1, k2, p1, p2, k3)
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Camera Calibration",
            "version": "v1",
            "short_description": "Remove camera lens distortions from an image using a calibration table.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "transformation",
                "icon": "far fa-crop-alt",
                "blockPriority": 1,
            },
        }
    )
    type: Literal["roboflow_core/camera-calibration@v1"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        description="Image to remove distortions from",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )
    fx: Union[
        Optional[float],
        Selector(kind=[FLOAT_KIND]),
    ] = Field(
        description="Focal length along the x-axis",
        examples=[0.123, "$inputs.fx"],
    )
    fy: Union[
        Optional[float],
        Selector(kind=[FLOAT_KIND]),
    ] = Field(
        description="Focal length along the y-axis",
        examples=[0.123, "$inputs.fy"],
    )
    cx: Union[
        Optional[float],
        Selector(kind=[FLOAT_KIND]),
    ] = Field(
        description="Optical center along the x-axis",
        examples=[0.123, "$inputs.cx"],
    )
    cy: Union[
        Optional[float],
        Selector(kind=[FLOAT_KIND]),
    ] = Field(
        description="Optical center along the y-axis",
        examples=[0.123, "$inputs.cy"],
    )
    k1: Union[
        Optional[float],
        Selector(kind=[FLOAT_KIND]),
    ] = Field(
        description="Radial distortion coefficient k1",
        examples=[0.123, "$inputs.k1"],
    )
    k2: Union[
        Optional[float],
        Selector(kind=[FLOAT_KIND]),
    ] = Field(
        description="Radial distortion coefficient k2",
        examples=[0.123, "$inputs.k2"],
    )
    k3: Union[
        Optional[float],
        Selector(kind=[FLOAT_KIND]),
    ] = Field(
        description="Radial distortion coefficient k3",
        examples=[0.123, "$inputs.k3"],
    )
    p1: Union[
        Optional[float],
        Selector(kind=[FLOAT_KIND]),
    ] = Field(
        description="Distortion coefficient p1",
        examples=[0.123, "$inputs.p1"],
    )
    p2: Union[
        Optional[float],
        Selector(kind=[FLOAT_KIND]),
    ] = Field(
        description="Distortion coefficient p2",
        examples=[0.123, "$inputs.p2"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=OUTPUT_CALIBRATED_IMAGE_KEY, kind=[IMAGE_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class CameraCalibrationBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        k1: float,
        k2: float,
        k3: float,
        p1: float,
        p2: float,
    ) -> BlockResult:
        return {
            OUTPUT_CALIBRATED_IMAGE_KEY: remove_distortions(
                image=image,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                k1=k1,
                k2=k2,
                k3=k3,
                p1=p1,
                p2=p2,
            )
        }


def remove_distortions(
    image: WorkflowImageData,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    k1: float,
    k2: float,
    k3: float,
    p1: float,
    p2: float,
) -> Optional[WorkflowImageData]:
    img = image.numpy_image
    h, w = img.shape[:2]

    cameraMatrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    distCoeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

    # https://docs.opencv.org/4.11.0/d9/d0c/group__calib3d.html#ga7a6c4e032c97f03ba747966e6ad862b1
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(
        cameraMatrix=cameraMatrix,
        distCoeffs=distCoeffs,
        imageSize=(w, h),
        alpha=1,
        newImgSize=(w, h),
    )
    # https://docs.opencv.org/4.11.0/d9/d0c/group__calib3d.html#ga69f2545a8b62a6b0fc2ee060dc30559d
    dst = cv.undistort(
        src=img,
        cameraMatrix=cameraMatrix,
        distCoeffs=distCoeffs,
        dst=None,
        newCameraMatrix=newcameramtx,
    )
    return WorkflowImageData(
        parent_metadata=image.parent_metadata,
        numpy_image=dst,
    )
