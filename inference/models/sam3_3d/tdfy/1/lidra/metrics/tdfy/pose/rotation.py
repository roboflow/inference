import itertools
import numpy as np

import torch
from pytorch3d.transforms import so3_relative_angle


def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-6, cos_bound=1e-6):
    """ """
    # rot_gt, rot_pred (B, 3, 3)
    assert rot_gt.shape == rot_pred.shape, f"{rot_gt.shape=} != {rot_pred.shape=}"
    assert rot_gt.ndim == 3, f"{rot_gt.ndim=} != 3"
    assert rot_pred.ndim == 3, f"{rot_pred.ndim=} != 3"
    rel_angle_cos = so3_relative_angle(
        rot_gt, rot_pred, eps=eps, cos_bound=cos_bound
    )  # cos_bound=1.0 to avoid NaN

    rel_rangle_deg = rel_angle_cos * 180 / torch.pi
    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg


def generate_90_deg_rotations():
    """
    Generate the 24 unique rotation matrices corresponding to 90° rotations.

    These are the proper rotations (determinant = 1) of a cube, which are given by
    all 3x3 matrices with one non-zero entry per row and per column, where that nonzero
    entry is either +1 or -1.

    Returns:
        rotations (list of np.ndarray): A list of 24 unique 3x3 rotation matrices.
    """
    rotations = []
    # Iterate over all permutations of the columns (positions for the nonzero entries)
    for perm in itertools.permutations([0, 1, 2]):
        # For each permutation, try all sign combinations for the 3 rows
        for signs in itertools.product([-1, 1], repeat=3):
            # Build the 3x3 matrix
            R = np.zeros((3, 3), dtype=int)
            for i in range(3):
                R[i, perm[i]] = signs[i]
            # Check if this matrix is a proper rotation (determinant == 1)
            if round(np.linalg.det(R)) == 1:
                rotations.append(R)
    # Optionally, remove any duplicates (there shouldn't be any in this method)
    unique_rotations = []
    for R in rotations:
        if not any(np.array_equal(R, Q) for Q in unique_rotations):
            unique_rotations.append(R)
    unique_rotations = [torch.from_numpy(R).float() for R in unique_rotations]
    return unique_rotations


def generate_candidate_rotations(n_angles):
    """
    Generate candidate rotations by uniformly sampling each Euler angle.
    n_angles: number of samples per Euler angle.
    Returns a list of 3x3 rotation matrices.
    """
    candidates = []
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    for alpha in angles:
        for beta in angles:
            for gamma in angles:
                R = euler_to_rotation(
                    torch.tensor(alpha, dtype=torch.float32),
                    torch.tensor(beta, dtype=torch.float32),
                    torch.tensor(gamma, dtype=torch.float32),
                )
                candidates.append(R)
    return candidates


def euler_to_rotation(alpha, beta, gamma):
    """Convert Euler angles (in radians) to a 3x3 rotation matrix.
    Here we use the ZYX convention."""
    Rx = torch.tensor(
        [
            [1, 0, 0],
            [0, torch.cos(alpha), -torch.sin(alpha)],
            [0, torch.sin(alpha), torch.cos(alpha)],
        ],
        dtype=torch.float32,
    )

    Ry = torch.tensor(
        [
            [torch.cos(beta), 0, torch.sin(beta)],
            [0, 1, 0],
            [-torch.sin(beta), 0, torch.cos(beta)],
        ],
        dtype=torch.float32,
    )

    Rz = torch.tensor(
        [
            [torch.cos(gamma), -torch.sin(gamma), 0],
            [torch.sin(gamma), torch.cos(gamma), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )

    # The overall rotation is R = Rz * Ry * Rx
    R = torch.mm(Rz, torch.mm(Ry, Rx))
    return R
