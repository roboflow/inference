"""
Triangulate all matched keypoint pairs into 3D points (camera 1 frame) using
camera intrinsics and the relative pose (R, t) from the Essential Matrix block.
"""

from typing import Any, List, Literal, Optional, Type, Union

import numpy as np
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    CameraIntrinsics,
    OutputDefinition,
)
from inference.core.workflows.execution_engine.entities.types import (
    DICTIONARY_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    NUMPY_ARRAY_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


def _intrinsics_to_K(intrinsics: Any) -> np.ndarray:
    """Build 3x3 calibration matrix from CameraIntrinsics or dict."""
    if hasattr(intrinsics, "to_calibration_matrix"):
        return intrinsics.to_calibration_matrix()
    if isinstance(intrinsics, dict):
        return np.array(
            [
                [float(intrinsics["fx"]), 0, float(intrinsics["cx"])],
                [0, float(intrinsics["fy"]), float(intrinsics["cy"])],
                [0, 0, 1],
            ]
        )
    raise TypeError(
        "camera_intrinsics must be CameraIntrinsics or dict with fx, fy, cx, cy"
    )


def _extract_point_pairs(good_matches: List[Any]) -> List[tuple]:
    """Extract (pt1, pt2) from good_matches; pt1/pt2 are (x, y). Skip pairs with None."""
    pairs = []
    for m in good_matches or []:
        if not isinstance(m, dict):
            continue
        kp = m.get("keypoint_pairs")
        if not kp or len(kp) != 2:
            continue
        pt1, pt2 = kp[0], kp[1]
        if pt1 is None or pt2 is None:
            continue
        try:
            x1, y1 = float(pt1[0]), float(pt1[1])
            x2, y2 = float(pt2[0]), float(pt2[1])
        except (TypeError, IndexError):
            continue
        pairs.append(((x1, y1), (x2, y2)))
    return pairs


def _triangulate_one_point(
    n1: np.ndarray, n2: np.ndarray, R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Triangulate one point from normalized rays and (R, t). Returns 3D point in camera 1 frame."""
    A = np.column_stack([R @ n1, -n2])
    d1_d2 = np.linalg.lstsq(A, -t, rcond=None)[0]
    return d1_d2[0] * n1


LONG_DESCRIPTION = """
Triangulate all matched keypoint pairs into 3D points in the first camera's coordinate frame. Uses camera intrinsics to unproject 2D points to normalized rays, then solves for the 3D point that lies on both rays given the relative pose (R, t) from the Essential Matrix block.

## How This Block Works

1. Receives good_matches (from FeatureComparisonBlockV1), camera_intrinsics_1 and camera_intrinsics_2, and rotation and translation (from EssentialMatrixBlockV1).
2. Extracts 2D point pairs from good_matches.
3. For each pair: unprojects to normalized rays n1 = K1^{-1} @ [u,v,1], n2 = K2^{-1} @ [u,v,1]; solves for depths so that the 3D point in camera 1 frame lies on both rays; returns P = d1 * n1 in camera 1 frame.
4. Outputs points_3d as a list of [x, y, z] and as an Nx3 numpy array.
"""

SHORT_DESCRIPTION = "Triangulate matched keypoint pairs into 3D points (camera 1 frame) using intrinsics and R, t."


class TriangulationBlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Triangulation",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "transformations",
                "icon": "far fa-cubes",
                "blockPriority": 5,
            },
        }
    )
    type: Literal["roboflow_core/triangulation@v1"]

    good_matches: Selector(kind=[LIST_OF_VALUES_KIND]) = Field(
        description="Output good_matches from FeatureComparisonBlockV1.",
        examples=["$steps.feature_comparison.good_matches"],
    )
    camera_intrinsics_1: Selector(kind=[DICTIONARY_KIND]) = Field(
        description="Camera intrinsics for the first image.",
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
        description="3-vector translation from EssentialMatrixBlockV1 (up to scale).",
        examples=["$steps.essential_matrix.translation"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="points_3d", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="points_3d_array", kind=[NUMPY_ARRAY_KIND]),
            OutputDefinition(name="points_count", kind=[INTEGER_KIND]),
        ]


class TriangulationBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return TriangulationBlockManifest

    def run(
        self,
        good_matches: List[Any],
        camera_intrinsics_1: Union[dict, CameraIntrinsics],
        camera_intrinsics_2: Union[dict, CameraIntrinsics],
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> BlockResult:
        pairs = _extract_point_pairs(good_matches)
        if not pairs:
            return {
                "points_3d": [],
                "points_3d_array": np.zeros((0, 3), dtype=np.float64),
                "points_count": 0,
            }
        K1 = _intrinsics_to_K(camera_intrinsics_1)
        K2 = _intrinsics_to_K(camera_intrinsics_2)
        K1_inv = np.linalg.inv(K1)
        K2_inv = np.linalg.inv(K2)
        R = np.asarray(rotation, dtype=np.float64)
        if R.shape != (3, 3):
            R = R.reshape(3, 3)
        t = np.asarray(translation, dtype=np.float64).ravel()[:3]

        points = []
        for (x1, y1), (x2, y2) in pairs:
            p1 = np.array([x1, y1, 1.0])
            p2 = np.array([x2, y2, 1.0])
            n1 = (K1_inv @ p1).ravel()
            n2 = (K2_inv @ p2).ravel()
            P = _triangulate_one_point(n1, n2, R, t)
            points.append([float(P[0]), float(P[1]), float(P[2])])

        points_array = np.array(points, dtype=np.float64)
        return {
            "points_3d": points,
            "points_3d_array": points_array,
            "points_count": len(points),
        }
