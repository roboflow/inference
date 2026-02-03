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

# RANSAC defaults
DEFAULT_RANSAC_THRESHOLD = 1e-4  # Sampson error in normalized coords
DEFAULT_RANSAC_MAX_ITERATIONS = 1000

DEFAULT_PARALLAX_COS_THRESHOLD = 0.995
DEFAULT_MIN_POSITIVE = 10

LONG_DESCRIPTION = """
Compute the Essential Matrix from keypoint correspondences and camera intrinsics using RANSAC with Sampson error in normalized coordinates, plus Hartley normalization for numerical stability. The Essential Matrix encodes the rigid transformation (rotation and translation) between two camera views. The block converts 2D point pairs to normalized coordinates, runs RANSAC (sample 8 points, fit E with 8-point + Hartley, score all points with Sampson error, keep best inliers), refits E on inliers, enforces rank-2, and recovers R and t.

## How This Block Works

1. Receives good_matches from FeatureComparisonBlockV1 and camera intrinsics for both images.
2. Converts point pairs to normalized homogeneous 2D using K1^{-1} and K2^{-1}.
3. RANSAC: repeatedly sample 8 point pairs, fit E using 8-point algorithm with Hartley normalization, compute Sampson error for all pairs in normalized coordinates, keep the E with the most inliers (error below ransac_threshold).
4. Refit E on all inliers with 8-point + Hartley; enforce rank-2 constraint.
5. Recovers R and t from E; chooses the solution with positive depth in both cameras.
6. Returns essential_matrix (3x3), rotation (3x3), and translation (3,) (translation is up to scale).
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


def _hartley_normalization_matrix(points_2d_hom: np.ndarray) -> np.ndarray:
    """
    Compute 3x3 Hartley normalization matrix for a set of 2D points in homogeneous coords.
    points_2d_hom: (N, 3) with last column 1. Translates centroid to origin and scales
    so mean distance from origin is sqrt(2). Returns T such that p_norm = T @ p.
    """
    # Inhomogeneous (x, y)
    x = points_2d_hom[:, 0] / points_2d_hom[:, 2]
    y = points_2d_hom[:, 1] / points_2d_hom[:, 2]

    cx = float(np.mean(x))
    cy = float(np.mean(y))
    dx = x - cx
    dy = y - cy

    scale = np.sqrt(2.0) / (np.mean(np.sqrt(dx * dx + dy * dy)) + 1e-12)

    T = np.array(
        [
            [scale, 0, -scale * cx],
            [0, scale, -scale * cy],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    return T


def _pairs_to_normalized_points(
    pairs: List[tuple], K1: np.ndarray, K2: np.ndarray
) -> tuple:
    """Convert 2D point pairs to normalized homogeneous 2D (N, 3) with last column 1."""
    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)

    n1_list = []
    n2_list = []

    for (x1, y1), (x2, y2) in pairs:
        p1 = np.array([x1, y1, 1.0])
        p2 = np.array([x2, y2, 1.0])
        n1 = (K1_inv @ p1).ravel()
        n2 = (K2_inv @ p2).ravel()

        # Store as homogeneous 2D: (nx/nz, ny/nz, 1) for Hartley normalization
        n1_list.append(np.array([n1[0] / n1[2], n1[1] / n1[2], 1.0]))
        n2_list.append(np.array([n2[0] / n2[2], n2[1] / n2[2], 1.0]))

    n1 = np.array(n1_list, dtype=np.float64)
    n2 = np.array(n2_list, dtype=np.float64)

    return n1, n2


def _eight_point_essential_from_normalized(
    pts1: np.ndarray, pts2: np.ndarray
) -> np.ndarray:
    """Compute E from normalized homogeneous 2D points (N, 3) with Hartley + SVD cleanup."""
    if len(pts1) < MIN_POINT_PAIRS or len(pts2) < MIN_POINT_PAIRS:
        raise ValueError(f"Need at least {MIN_POINT_PAIRS} points")
    T1 = _hartley_normalization_matrix(pts1)
    T2 = _hartley_normalization_matrix(pts2)

    pts1_norm = (T1 @ pts1.T).T
    pts2_norm = (T2 @ pts2.T).T
    pts1_norm /= pts1_norm[:, [2]]
    pts2_norm /= pts2_norm[:, [2]]

    # Build constraint matrix: for each (p1_norm, p2_norm), row = p2_norm ⊗ p1_norm so p2_norm^T E_norm p1_norm = 0
    A = []
    for i in range(len(pts1_norm)):
        p1 = pts1_norm[i]
        p2 = pts2_norm[i]
        row = [
            p2[0] * p1[0], p2[0] * p1[1], p2[0] * p1[2],
            p2[1] * p1[0], p2[1] * p1[1], p2[1] * p1[2],
            p2[2] * p1[0], p2[2] * p1[1], p2[2] * p1[2],
        ]
        A.append(row)
    A = np.array(A, dtype=np.float64)

    _, _, Vt = np.linalg.svd(A)
    E_norm = Vt[-1].reshape(3, 3)

    # Enforce rank-2
    U, _, Vt = np.linalg.svd(E_norm)
    E_norm = U @ np.diag([1.0, 1.0, 0.0]) @ Vt

    # Denormalize: E such that p2^T E p1 = 0 with p1,p2 original; p_norm = T p => E = T2^T E_norm T1
    E = T2.T @ E_norm @ T1

    U, _, Vt = np.linalg.svd(E)
    E = U @ np.diag([1.0, 1.0, 0.0]) @ Vt

    return E


def _sampson_errors(E: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Sampson error for Nx3 homogeneous correspondences.
    x1, x2 are row-vectors (N,3).
    Returns squared Sampson error (N,).
    """
    Ex1   = x1 @ E.T        # (E x1)^T
    Etx2  = x2 @ E          # (E^T x2)^T

    x2tEx1 = np.sum(x2 * Ex1, axis=1)  # x2^T E x1

    denom = Ex1[:,0]**2 + Ex1[:,1]**2 + Etx2[:,0]**2 + Etx2[:,1]**2
    error_values =  (x2tEx1**2) / (denom + 1e-12)

    return error_values


def _essential_matrix_ransac(
    pts1: np.ndarray,
    pts2: np.ndarray,
    threshold: float = DEFAULT_RANSAC_THRESHOLD,
    max_iterations: int = DEFAULT_RANSAC_MAX_ITERATIONS,
) -> tuple:
    """RANSAC for Essential Matrix using Sampson error in normalized coords. Returns E, inlier_mask."""
    rng = np.random.default_rng()
    n = len(pts1)

    if n < MIN_POINT_PAIRS:
        raise ValueError(f"Need at least {MIN_POINT_PAIRS} point pairs for RANSAC")

    best_inliers = np.zeros(n, dtype=bool)
    best_E = None
    best_cost = np.inf

    for _ in range(max_iterations):
        idx = rng.choice(n, size=MIN_POINT_PAIRS, replace=False)

        try:
            E = _eight_point_essential_from_normalized(pts1[idx], pts2[idx])
        except Exception:
            continue

        err = _sampson_errors(E, pts1, pts2)
        inliers = err < threshold

        num = int(inliers.sum())
        if num < MIN_POINT_PAIRS:
            continue

        cost = float(err[inliers].sum())

        if (num > best_inliers.sum()) or (num == best_inliers.sum() and cost < best_cost):
            best_inliers = inliers
            best_E = E
            best_cost = cost

    if best_E is None or best_inliers.sum() < MIN_POINT_PAIRS:
        raise ValueError("RANSAC failed to find a valid essential matrix")

    # Refit on inliers
    E_refined = _eight_point_essential_from_normalized(pts1[best_inliers], pts2[best_inliers])

    return E_refined, best_inliers


def _parallax_cos(x1: np.ndarray, x2: np.ndarray, R: np.ndarray) -> float:
    """Compute cosine of parallax angle between two points."""
    r1 = x1 / np.linalg.norm(x1)
    r2 = R.T @ x2
    r2 = r2 / np.linalg.norm(r2)
    return np.dot(r1, r2)


def triangulate_dlt(x1: np.ndarray, x2: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Linear triangulation (DLT).
    x1, x2: (3,) homogeneous normalized image points [u,v,1]
    Returns X in camera1 coordinates (3,)
    """
    P1 = np.hstack([np.eye(3), np.zeros((3,1))])
    P2 = np.hstack([R, t.reshape(3,1)])

    u1, v1 = x1[0]/x1[2], x1[1]/x1[2]
    u2, v2 = x2[0]/x2[2], x2[1]/x2[2]

    A = np.zeros((4,4), dtype=float)
    A[0] = u1 * P1[2] - P1[0]
    A[1] = v1 * P1[2] - P1[1]
    A[2] = u2 * P2[2] - P2[0]
    A[3] = v2 * P2[2] - P2[1]

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X[:3] / X[3]
    return X


def _choose_pose_by_cheirality(
    candidates: List[tuple],
    x1s: List[np.ndarray],
    x2s: List[np.ndarray],
    parallax_cos_threshold: float = DEFAULT_PARALLAX_COS_THRESHOLD,
    min_positive: int = DEFAULT_MIN_POSITIVE,
) -> tuple:
    """Choose pose by cheirality."""
    best = None
    best_count = -1

    for R, t in candidates:
        if np.linalg.det(R) < 0:
            R, t = -R, -t

        count = 0
        for x1, x2 in zip(x1s, x2s):
            cos_parallax = _parallax_cos(x1, x2, R)
            if cos_parallax > parallax_cos_threshold:
                continue

            X = triangulate_dlt(x1, x2, R, t)  # X in cam1

            z1 = X[2]
            z2 = (R @ X + t)[2]
            if z1 > 0 and z2 > 0:
                count += 1

        if count > best_count:
            best_count = count
            best = (R, t)

    if best is None or best_count < min_positive:
        raise ValueError("Cheirality failed / too few points in front")

    return best


def _recover_pose_from_essential(
    E: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
    parallax_cos_threshold: float = DEFAULT_PARALLAX_COS_THRESHOLD,
    min_positive: int = DEFAULT_MIN_POSITIVE,
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

    best = _choose_pose_by_cheirality(
        candidates,
        n1,
        n2,
        parallax_cos_threshold=parallax_cos_threshold,
        min_positive=min_positive,
    )

    return best


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
    ransac_threshold: Union[float, Selector] = Field(
        default=DEFAULT_RANSAC_THRESHOLD,
        description="Sampson error threshold in normalized coords for RANSAC inlier (smaller = stricter).",
    )
    ransac_max_iterations: Union[int, Selector] = Field(
        default=DEFAULT_RANSAC_MAX_ITERATIONS,
        description="Max RANSAC iterations.",
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
        ransac_threshold: float = DEFAULT_RANSAC_THRESHOLD,
        ransac_max_iterations: int = DEFAULT_RANSAC_MAX_ITERATIONS,
        parallax_cos_threshold: float = DEFAULT_PARALLAX_COS_THRESHOLD,
        min_positive: int = DEFAULT_MIN_POSITIVE,
    ) -> BlockResult:
        pairs = _extract_point_pairs(good_matches)

        if len(pairs) < MIN_POINT_PAIRS:
            raise ValueError(
                f"Not enough point pairs for essential matrix: {len(pairs)} < {MIN_POINT_PAIRS}"
            )

        K1 = _intrinsics_to_K(camera_intrinsics_1)
        K2 = _intrinsics_to_K(camera_intrinsics_2)

        pts1, pts2 = _pairs_to_normalized_points(pairs, K1, K2)

        E, inlier_mask = _essential_matrix_ransac(
            pts1, pts2,
            threshold=float(ransac_threshold),
            max_iterations=int(ransac_max_iterations),
        )

        n1 = pts1[inlier_mask, :]
        n2 = pts2[inlier_mask, :]

        R, t = _recover_pose_from_essential(
            E,
            n1=n1,
            n2=n2,
            parallax_cos_threshold=parallax_cos_threshold,
            min_positive=min_positive,
        )

        return {
            "essential_matrix": E,
            "rotation": R,
            "translation": t,
        }
