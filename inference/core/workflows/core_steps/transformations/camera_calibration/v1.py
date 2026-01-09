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
Remove lens distortions from images using camera calibration parameters (focal lengths, optical centers, and distortion coefficients) to correct radial and tangential distortions introduced by camera lenses, producing undistorted images suitable for accurate measurement, geometric analysis, and precision computer vision applications.

## How This Block Works

Camera lenses introduce distortions that cause straight lines to appear curved and objects near image edges to appear stretched or compressed. This block corrects these distortions using known camera calibration parameters. The block:

1. Receives input images and camera calibration parameters (focal lengths fx/fy, optical centers cx/cy, radial distortion coefficients k1/k2/k3, tangential distortion coefficients p1/p2)
2. Constructs a camera matrix from the intrinsic parameters (focal lengths and optical centers) in the standard OpenCV format: 3x3 matrix with fx, fy on the diagonal, cx, cy as the optical center, and 1 in the bottom-right corner
3. Assembles distortion coefficients into a 5-element array (k1, k2, p1, p2, k3) representing radial and tangential distortion parameters
4. Computes an optimal new camera matrix using OpenCV's `getOptimalNewCameraMatrix` to maximize the usable image area after correction (removes black borders that result from distortion correction)
5. Applies OpenCV's `undistort` function to correct both radial distortions (barrel and pincushion distortion causing curved lines) and tangential distortions (lens misalignment causing skewed images)
6. Returns the corrected, undistorted image with straight lines corrected, edge distortions removed, and geometric accuracy restored

The block uses OpenCV's camera calibration functions under the hood, following standard computer vision camera calibration methodology (see [OpenCV calibration tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) for details on obtaining calibration parameters). Radial distortion coefficients (k1, k2, k3) correct barrel/pincushion distortion where image points are displaced radially from the optical center. Tangential distortion coefficients (p1, p2) correct distortion caused by lens misalignment. The calibration parameters must be obtained beforehand through a camera calibration process (typically using checkerboard patterns) or provided by the camera manufacturer.

## Requirements

**Camera Calibration Parameters**: This block requires pre-computed camera calibration parameters obtained through camera calibration:
- **Focal lengths (fx, fy)**: Pixel focal lengths along x and y axes (may differ for non-square pixels)
- **Optical centers (cx, cy)**: Principal point coordinates (image center in ideal cameras)
- **Radial distortion coefficients (k1, k2, k3)**: Correct barrel and pincushion distortion
- **Tangential distortion coefficients (p1, p2)**: Correct lens misalignment distortion

These parameters are typically obtained using OpenCV's camera calibration process with a checkerboard pattern or similar calibration target. See [OpenCV camera calibration documentation](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) for calibration methodology.

## Common Use Cases

- **Measurement and Metrology Applications**: Correct lens distortions for accurate measurement workflows (e.g., remove distortions before measuring object sizes, correct geometric distortions for precision measurements, undistort images for dimensional analysis), enabling accurate measurements from camera images
- **Geometric Analysis Workflows**: Prepare images for geometric computer vision tasks (e.g., undistort images before line detection, correct distortions for geometric shape analysis, prepare images for accurate angle measurements), enabling precise geometric analysis with corrected images
- **Multi-Camera Systems**: Standardize images from multiple cameras with different lens characteristics (e.g., undistort images from different camera angles, correct wide-angle lens distortions, standardize images from multiple cameras for stereo vision), enabling consistent image geometry across camera setups
- **Pre-Processing for Precision Models**: Prepare images for models requiring high geometric accuracy (e.g., undistort images before running geometric models, correct distortions for accurate feature detection, prepare images for precise pose estimation), enabling better accuracy for geometric computer vision tasks
- **Wide-Angle and Fisheye Correction**: Correct severe distortions from wide-angle or fisheye lenses (e.g., correct barrel distortion from wide-angle lenses, remove fisheye distortion effects, straighten curved lines in wide-angle images), enabling use of wide-angle lenses with standard computer vision workflows
- **Video Stabilization Preparation**: Correct lens distortions as part of video stabilization pipelines (e.g., undistort video frames before stabilization, correct camera-specific distortions in video streams, prepare frames for motion analysis), enabling more accurate video processing

## Connecting to Other Blocks

This block receives images and produces undistorted images:

- **After image loading blocks** to correct lens distortions before processing, enabling accurate analysis with geometrically correct images
- **Before measurement and analysis blocks** that require geometric accuracy (e.g., size measurement, angle measurement, distance calculation, geometric shape analysis), enabling precise measurements from undistorted images
- **Before geometric computer vision blocks** that analyze lines, shapes, or spatial relationships (e.g., line detection, contour analysis, geometric pattern matching, pose estimation), enabling accurate geometric analysis with corrected images
- **In multi-camera workflows** to standardize images from different cameras before processing (e.g., undistort images from different camera angles, correct camera-specific distortions before comparison, standardize images for stereo vision), enabling consistent processing across camera setups
- **Before detection or classification blocks** in precision applications where geometric accuracy matters (e.g., detect objects in undistorted images for accurate localization, classify objects in geometrically correct images, run models requiring precise spatial relationships), enabling improved accuracy for detection and classification tasks
- **In video processing workflows** to correct distortions in video frames (e.g., undistort video frames for motion analysis, correct camera distortions in video streams, prepare frames for tracking algorithms), enabling accurate video analysis with corrected frames
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
        description="Input image to remove lens distortions from. The image will be corrected for radial and tangential distortions using the provided camera calibration parameters. Works with images from cameras with known calibration parameters. The undistorted output image will have corrected geometry with straight lines straightened and edge distortions removed.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )
    fx: Union[
        Optional[float],
        Selector(kind=[FLOAT_KIND]),
    ] = Field(
        description="Focal length along the x-axis in pixels. Part of the camera's intrinsic parameters. Typically obtained through camera calibration (e.g., using OpenCV calibration with a checkerboard pattern). Represents the camera's horizontal focal length. For square pixels, fx and fy are usually equal. Must be obtained from camera calibration or manufacturer specifications.",
        examples=[0.123, "$inputs.fx"],
    )
    fy: Union[
        Optional[float],
        Selector(kind=[FLOAT_KIND]),
    ] = Field(
        description="Focal length along the y-axis in pixels. Part of the camera's intrinsic parameters. Typically obtained through camera calibration (e.g., using OpenCV calibration with a checkerboard pattern). Represents the camera's vertical focal length. For square pixels, fx and fy are usually equal. Must be obtained from camera calibration or manufacturer specifications.",
        examples=[0.123, "$inputs.fy"],
    )
    cx: Union[
        Optional[float],
        Selector(kind=[FLOAT_KIND]),
    ] = Field(
        description="Optical center (principal point) x-coordinate in pixels. Part of the camera's intrinsic parameters representing the x-coordinate of the camera's principal point (image center in ideal cameras). Typically near half the image width. Obtained through camera calibration. Used with cy to define the optical center of the camera.",
        examples=[0.123, "$inputs.cx"],
    )
    cy: Union[
        Optional[float],
        Selector(kind=[FLOAT_KIND]),
    ] = Field(
        description="Optical center (principal point) y-coordinate in pixels. Part of the camera's intrinsic parameters representing the y-coordinate of the camera's principal point (image center in ideal cameras). Typically near half the image height. Obtained through camera calibration. Used with cx to define the optical center of the camera.",
        examples=[0.123, "$inputs.cy"],
    )
    k1: Union[
        Optional[float],
        Selector(kind=[FLOAT_KIND]),
    ] = Field(
        description="First radial distortion coefficient. Part of the camera's distortion parameters used to correct barrel and pincushion distortion (where straight lines appear curved). k1 is typically the dominant radial distortion term. Positive values often indicate barrel distortion, negative values indicate pincushion distortion. Obtained through camera calibration.",
        examples=[0.123, "$inputs.k1"],
    )
    k2: Union[
        Optional[float],
        Selector(kind=[FLOAT_KIND]),
    ] = Field(
        description="Second radial distortion coefficient. Part of the camera's distortion parameters used to correct higher-order radial distortion effects. k2 helps correct more complex radial distortion patterns beyond the first-order k1 term. Obtained through camera calibration. Often smaller in magnitude than k1.",
        examples=[0.123, "$inputs.k2"],
    )
    k3: Union[
        Optional[float],
        Selector(kind=[FLOAT_KIND]),
    ] = Field(
        description="Third radial distortion coefficient. Part of the camera's distortion parameters used to correct additional higher-order radial distortion effects. k3 is typically the smallest radial distortion term and is used for very precise distortion correction, especially for wide-angle lenses. Obtained through camera calibration. Often set to 0 for standard lenses.",
        examples=[0.123, "$inputs.k3"],
    )
    p1: Union[
        Optional[float],
        Selector(kind=[FLOAT_KIND]),
    ] = Field(
        description="First tangential distortion coefficient. Part of the camera's distortion parameters used to correct tangential distortion caused by lens misalignment. p1 corrects skew distortions where the lens is not perfectly aligned with the image sensor. Obtained through camera calibration. For well-aligned lenses, p1 and p2 are often close to zero.",
        examples=[0.123, "$inputs.p1"],
    )
    p2: Union[
        Optional[float],
        Selector(kind=[FLOAT_KIND]),
    ] = Field(
        description="Second tangential distortion coefficient. Part of the camera's distortion parameters used to correct additional tangential distortion effects. p2 works together with p1 to correct lens misalignment distortions. Obtained through camera calibration. For well-aligned lenses, p1 and p2 are often close to zero.",
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
