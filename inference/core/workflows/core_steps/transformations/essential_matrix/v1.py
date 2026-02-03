"""
Essential Matrix block: compute E from keypoint correspondences and camera intrinsics
via the epipolar constraint (8-point algorithm + SVD), then recover R and t.
"""

from typing import Any, List, Literal, Optional, Type, Union

import numpy as np
from pydantic import ConfigDict, Field

from inference.core.logger import logger
from inference.core.workflows.execution_engine.entities.base import (
    CameraIntrinsics,
    OutputDefinition,
)
from inference.core.workflows.execution_engine.entities.types import (
    DICTIONARY_KIND,
    LIST_OF_VALUES_KIND,
    NUMPY_ARRAY_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

# Minimum point pairs for 8-point algorithm
MIN_POINT_PAIRS = 8

LONG_DESCRIPTION = """
Compute the Essential Matrix from keypoint correspondences and camera intrinsics using the epipolar constraint. The Essential Matrix encodes the rigid transformation (rotation and translation) between two camera views. The block converts 2D keypoint pairs to normalized rays using camera intrinsics, builds the linear system from the epipolar constraint (x2^T E x1 = 0), solves for E via SVD (8-point algorithm), enforces the rank-2 constraint, and recovers rotation R and translation t from E.

## How This Block Works

1. Receives good_matches from FeatureComparisonBlockV1 (list of {keypoint_pairs: [pt1, pt2], distance}) and camera intrinsics for both images.
2. Converts keypoint correspondences to homogeneous coordinates and unprojects to normalized bearing rays using K1^{-1} and K2^{-1}.
3. Builds the epipolar constraint matrix: for each pair (n1, n2), one row (n2 ⊗ n1) so that n2^T E n1 = 0.
4. Solves A @ vec(E) = 0 via SVD; the solution is the last right singular vector, reshaped to 3x3 E.
5. Enforces the essential matrix constraint (two equal singular values, one zero) by SVD of E and replacing with diag(1,1,0).
6. Recovers R and t from E: E = U diag(1,1,0) V^T; R = U W V^T or U W^T V^T, t = ±U[:,2]; chooses the solution with positive depth in both cameras.
7. Returns essential_matrix (3x3), rotation (3x3), and translation (3,) (translation is up to scale).
"""

SHORT_DESCRIPTION = "Compute Essential Matrix and rigid transformation (R, t) from keypoint pairs and camera intrinsics."


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
    raise TypeError("camera_intrinsics must be CameraIntrinsics or dict with fx, fy, cx, cy")


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


def _eight_point_essential(
    pairs: List[tuple], K1: np.ndarray, K2: np.ndarray
) -> np.ndarray:
    """Compute E from point pairs and calibration matrices (8-point + SVD cleanup)."""
    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)
    A = []
    for (x1, y1), (x2, y2) in pairs:
        p1 = np.array([x1, y1, 1.0])
        p2 = np.array([x2, y2, 1.0])
        n1 = (K1_inv @ p1).ravel()
        n2 = (K2_inv @ p2).ravel()
        # n2^T E n1 = 0 => row = [n2[0]*n1[0], n2[0]*n1[1], n2[0]*n1[2], ..., n2[2]*n1[2]]
        row = [
            n2[0] * n1[0],
            n2[0] * n1[1],
            n2[0] * n1[2],
            n2[1] * n1[0],
            n2[1] * n1[1],
            n2[1] * n1[2],
            n2[2] * n1[0],
            n2[2] * n1[1],
            n2[2] * n1[2],
        ]
        A.append(row)
    A = np.array(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    e = Vt[-1]
    E = e.reshape(3, 3)
    # Enforce rank-2: replace singular values with (1, 1, 0)
    U, S, Vt = np.linalg.svd(E)
    E = U @ np.diag([1.0, 1.0, 0.0]) @ Vt
    return E


def _triangulate_one_point(
    n1: np.ndarray, n2: np.ndarray, R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Triangulate one point from normalized rays and (R, t). Returns 3D point in camera 1 frame."""
    # n2 = R @ (lambda2 * n2_3d) + t  with n2_3d = n2 (normalized ray in cam2). So n2 ≈ R @ (d2 * n2) + t.
    # In cam1: ray1 = n1, point = d1 * n1. In cam2: point_cam2 = R @ (d1*n1) + t = d1 * R@n1 + t, and ray2 = n2.
    # So d1 * R@n1 + t = d2 * n2 => [R@n1, -n2] @ [d1; d2] = -t. Solve for d1, d2; take d1*n1.
    A = np.column_stack([R @ n1, -n2])
    d1_d2 = np.linalg.lstsq(A, -t, rcond=None)[0]
    return d1_d2[0] * n1


def _recover_pose_from_essential(
    E: np.ndarray,
    n1: Optional[np.ndarray] = None,
    n2: Optional[np.ndarray] = None,
) -> tuple:
    """Recover R and t from E. If n1, n2 given, choose solution with positive depth in both cameras."""
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    candidates = [
        (U @ W @ Vt, U[:, 2]),
        (U @ W @ Vt, -U[:, 2]),
        (U @ W.T @ Vt, U[:, 2]),
        (U @ W.T @ Vt, -U[:, 2]),
    ]
    for R, t in candidates:
        if np.linalg.det(R) < 0:
            R, t = -R, -t
        if n1 is not None and n2 is not None:
            P = _triangulate_one_point(n1, n2, R, t)
            depth1 = P[2]
            depth2 = (R @ P + t)[2]
            if depth1 > 0 and depth2 > 0:
                return R, t

    logger.warning(f"No solution found for R and t: {candidates}")
    return candidates[0][0], candidates[0][1]


class EssentialMatrixBlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Essential Matrix",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "transformations",
                "icon": "far fa-cube",
                "blockPriority": 5,
            },
        }
    )
    type: Literal["roboflow_core/essential_matrix@v1"]

    good_matches: Selector(kind=[LIST_OF_VALUES_KIND]) = Field(
        description="Output good_matches from FeatureComparisonBlockV1 (list of {keypoint_pairs: [pt1, pt2], distance}). At least 8 point pairs with valid coordinates are required.",
        examples=["$steps.feature_comparison.good_matches"],
    )
    camera_intrinsics_1: Selector(kind=[DICTIONARY_KIND]) = Field(
        description="Camera intrinsics for the first image (same as used for keypoints in image 1). Dict with fx, fy, cx, cy, k1, k2, p1, p2, k3; or pass image parent_metadata.camera_intrinsics. Only fx, fy, cx, cy are used for E.",
        examples=["$inputs.camera_intrinsics_1"],
    )
    camera_intrinsics_2: Selector(kind=[DICTIONARY_KIND]) = Field(
        description="Camera intrinsics for the second image.",
        examples=["$inputs.camera_intrinsics_2"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="essential_matrix", kind=[NUMPY_ARRAY_KIND]),
            OutputDefinition(name="rotation", kind=[NUMPY_ARRAY_KIND]),
            OutputDefinition(name="translation", kind=[NUMPY_ARRAY_KIND]),
        ]


class EssentialMatrixBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return EssentialMatrixBlockManifest

    def run(
        self,
        good_matches: List[Any],
        camera_intrinsics_1: Union[dict, CameraIntrinsics],
        camera_intrinsics_2: Union[dict, CameraIntrinsics],
    ) -> BlockResult:
        pairs = _extract_point_pairs(good_matches)
        if len(pairs) < MIN_POINT_PAIRS:
            return {
                "essential_matrix": np.eye(3, dtype=np.float64),
                "rotation": np.eye(3, dtype=np.float64),
                "translation": np.zeros(3, dtype=np.float64),
            }
        K1 = _intrinsics_to_K(camera_intrinsics_1)
        K2 = _intrinsics_to_K(camera_intrinsics_2)
        E = _eight_point_essential(pairs, K1, K2)
        K1_inv = np.linalg.inv(K1)
        K2_inv = np.linalg.inv(K2)
        (x1, y1), (x2, y2) = pairs[0]
        p1 = np.array([x1, y1, 1.0])
        p2 = np.array([x2, y2, 1.0])
        n1 = (K1_inv @ p1).ravel()
        n2 = (K2_inv @ p2).ravel()
        R, t = _recover_pose_from_essential(E, n1=n1, n2=n2)
        return {
            "essential_matrix": E,
            "rotation": R,
            "translation": t,
        }
