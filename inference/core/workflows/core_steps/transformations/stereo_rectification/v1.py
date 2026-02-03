"""
Stereo rectification: rectify two images so that epipolar lines are horizontal (same row).
Uses OpenCV stereoRectify + initUndistortRectifyMap + remap.
"""

from typing import Any, List, Literal, Optional, Type, Union

import cv2
import numpy as np
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    CameraIntrinsics,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    DICTIONARY_KIND,
    IMAGE_KIND,
    NUMPY_ARRAY_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


def _intrinsics_to_K_and_D(intrinsics: Any) -> tuple:
    """Build 3x3 K and 5-element dist (k1, k2, p1, p2, k3) from CameraIntrinsics or dict."""
    if hasattr(intrinsics, "to_calibration_matrix"):
        K = intrinsics.to_calibration_matrix()
        D = np.array(
            [
                intrinsics.k1,
                intrinsics.k2,
                intrinsics.p1,
                intrinsics.p2,
                intrinsics.k3,
            ],
            dtype=np.float64,
        )
        return K, D
    if isinstance(intrinsics, dict):
        K = np.array(
            [
                [float(intrinsics["fx"]), 0, float(intrinsics["cx"])],
                [0, float(intrinsics["fy"]), float(intrinsics["cy"])],
                [0, 0, 1],
            ]
        )
        D = np.array(
            [
                float(intrinsics.get("k1", 0)),
                float(intrinsics.get("k2", 0)),
                float(intrinsics.get("p1", 0)),
                float(intrinsics.get("p2", 0)),
                float(intrinsics.get("k3", 0)),
            ],
            dtype=np.float64,
        )
        return K, D
    raise TypeError(
        "camera_intrinsics must be CameraIntrinsics or dict with fx, fy, cx, cy"
    )


LONG_DESCRIPTION = """
Rectify two stereo images so that corresponding points lie on the same row (horizontal epipolar lines). Uses the relative pose (R, t) from the Essential Matrix block and camera intrinsics to compute rectification transforms with OpenCV stereoRectify, then remaps both images. Outputs are the two rectified images suitable for stereo matching (e.g. disparity).

## How This Block Works

1. Receives image_1, image_2, camera_intrinsics_1, camera_intrinsics_2, rotation and translation (from EssentialMatrixBlockV1).
2. Builds 3x3 K and distortion vectors from intrinsics; image size from image_1.
3. Calls cv2.stereoRectify to get R1, R2, P1, P2 (and Q); uses alpha=0 for valid pixels only (no black borders from scaling).
4. For each camera: initUndistortRectifyMap(K, D, R, P, size) → map1, map2; remap(image, map1, map2) → rectified image.
5. Returns rectified_image_1 and rectified_image_2 as WorkflowImageData.
"""

SHORT_DESCRIPTION = "Rectify two stereo images so epipolar lines are horizontal; returns rectified images."


class StereoRectificationBlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Stereo Rectification",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "transformations",
                "icon": "far fa-columns",
                "blockPriority": 5,
            },
        }
    )
    type: Literal["roboflow_core/stereo_rectification@v1"]

    image_1: Selector(kind=[IMAGE_KIND]) = Field(
        description="First (left) image of the stereo pair.",
        examples=["$inputs.image_1", "$steps.load_1.image"],
    )
    image_2: Selector(kind=[IMAGE_KIND]) = Field(
        description="Second (right) image of the stereo pair.",
        examples=["$inputs.image_2", "$steps.load_2.image"],
    )
    camera_intrinsics_1: Selector(kind=[DICTIONARY_KIND]) = Field(
        description="Camera intrinsics for the first image (fx, fy, cx, cy, k1, k2, p1, p2, k3).",
        examples=["$inputs.camera_intrinsics_1"],
    )
    camera_intrinsics_2: Selector(kind=[DICTIONARY_KIND]) = Field(
        description="Camera intrinsics for the second image.",
        examples=["$inputs.camera_intrinsics_2"],
    )
    rotation: Selector(kind=[NUMPY_ARRAY_KIND]) = Field(
        description="3x3 rotation matrix from EssentialMatrixBlockV1 (camera 2 w.r.t. camera 1).",
        examples=["$steps.essential_matrix.rotation"],
    )
    translation: Selector(kind=[NUMPY_ARRAY_KIND]) = Field(
        description="3-vector translation from EssentialMatrixBlockV1.",
        examples=["$steps.essential_matrix.translation"],
    )
    alpha: Union[float, Selector] = Field(
        default=0.0,
        description="Free scaling parameter (0=only valid pixels, 1=all pixels; may add black borders).",
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="rectified_image_1", kind=[IMAGE_KIND]),
            OutputDefinition(name="rectified_image_2", kind=[IMAGE_KIND]),
        ]


class StereoRectificationBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return StereoRectificationBlockManifest

    def run(
        self,
        image_1: WorkflowImageData,
        image_2: WorkflowImageData,
        camera_intrinsics_1: Union[dict, CameraIntrinsics],
        camera_intrinsics_2: Union[dict, CameraIntrinsics],
        rotation: np.ndarray,
        translation: np.ndarray,
        alpha: float = 0.0,
    ) -> BlockResult:
        img1 = image_1.numpy_image
        img2 = image_2.numpy_image
        if img1 is None or img2 is None:
            raise ValueError("Stereo rectification requires loaded numpy images.")
        h, w = img1.shape[:2]
        size = (w, h)

        K1, D1 = _intrinsics_to_K_and_D(camera_intrinsics_1)
        K2, D2 = _intrinsics_to_K_and_D(camera_intrinsics_2)
        R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        T = np.asarray(translation, dtype=np.float64).ravel()[:3]

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, D1, K2, D2, size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=float(alpha),
        )

        map1_1, map1_2 = cv2.initUndistortRectifyMap(
            K1, D1, R1, P1, size, cv2.CV_32FC1
        )
        map2_1, map2_2 = cv2.initUndistortRectifyMap(
            K2, D2, R2, P2, size, cv2.CV_32FC1
        )

        rect1 = cv2.remap(img1, map1_1, map1_2, cv2.INTER_LINEAR)
        rect2 = cv2.remap(img2, map2_1, map2_2, cv2.INTER_LINEAR)

        out1 = WorkflowImageData.copy_and_replace(
            origin_image_data=image_1,
            numpy_image=rect1,
        )
        out2 = WorkflowImageData.copy_and_replace(
            origin_image_data=image_2,
            numpy_image=rect2,
        )
        return {
            "rectified_image_1": out1,
            "rectified_image_2": out2,
        }
