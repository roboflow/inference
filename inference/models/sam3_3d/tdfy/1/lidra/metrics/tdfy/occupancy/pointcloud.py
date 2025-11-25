import torch
from typing import Optional, Union
import warnings

from pytorch3d.ops import iterative_closest_point
from pytorch3d.loss import chamfer_distance


def create_occupancy_volume(
    points: torch.Tensor, n_voxels: int = 32, device: Optional[str] = None
) -> torch.Tensor:
    """Point cloud --> occupancy volume

    Args:
        points: [n, 3] Tensor with values in range [-0.5, 0.5]
        n_voxels: Number of voxels per side

    Returns:
        volume: Binary occupancy grid of shape (n_voxels, n_voxels, n_voxels)
    """
    device = device or points.device
    # Create empty volume grid on same device as points
    volume = torch.zeros((n_voxels, n_voxels, n_voxels), device=device)

    assert points.shape[1] == 3, "Points must be of shape (N, 3)"
    if points.numel() == 0:
        return volume

    assert (
        points.min() > -0.5 and points.max() < 0.5
    ), f"Points must be in the range [-0.5, 0.5] (got {points.min()}, {points.max()})"

    # Scale points to [0, n_voxels] range
    points = (points + 0.5) * (n_voxels)
    assert torch.all(points >= 0) and torch.all(
        points <= n_voxels
    ), f"Points must be in the range [0, {n_voxels}] (got {points.min()}, {points.max()})"
    indices = torch.clamp(points.long(), 0, n_voxels - 1)
    volume[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
    return volume


def occupancy_grid_to_local_points(
    occupancy_grid: torch.Tensor,
    threshold: float = 0.0,
    half_side_length: float = 0.5,
    origin: Union[float, torch.Tensor] = 0.0,
    return_logits: bool = False,
):
    """
    Convert a 3D occupancy grid to points in [-0.5, 0.5]^3.
    """
    assert occupancy_grid.ndim >= 3
    assert (
        occupancy_grid.shape[-1] == occupancy_grid.shape[-2] == occupancy_grid.shape[-3]
    )
    n_voxels_per_side = occupancy_grid.shape[-1]

    coords, logits = occupancy_grid_to_coords_logits(occupancy_grid)

    points_local = coords_to_local_points(
        coords, n_voxels_per_side, half_side_length, origin
    )
    # # Convert to local points
    # n_voxels_per_side = occupancy_grid.shape[-1]
    # points_local = (coords + 0.5) / n_voxels_per_side - 0.5
    # points_local = points_local * (2 * half_side_length) + origin
    if return_logits:
        return points_local, logits
    return points_local


def occupancy_grid_to_coords_logits(
    occupancy_grid: torch.Tensor,
    threshold: float = 0.0,
):
    assert occupancy_grid.ndim >= 3
    assert (
        occupancy_grid.shape[-1] == occupancy_grid.shape[-2] == occupancy_grid.shape[-3]
    )
    n_voxels_per_side = occupancy_grid.shape[-1]

    assert occupancy_grid.ndim == 5, f"Volume must be 5D, got {occupancy_grid.ndim}D"
    coords_and_logits = torch.argwhere(occupancy_grid > threshold)[:, [0, 2, 3, 4]]

    logits = coords_and_logits[:, 0]
    coords = coords_and_logits[:, 1:].int()
    return coords, logits


def coords_to_local_points(
    coords: torch.Tensor,
    n_voxels_per_side: int,
    half_side_length: float = 0.5,
    origin: Union[float, torch.Tensor] = 0.0,
):
    """
    Convert coordinates to local points.

    Note:
        Folks using our demo usually ask for this function.
        During release, we may want to expose the local_points directly.
    """
    # Convert to local points
    points_local = (coords + 0.5) / n_voxels_per_side - 0.5
    points_local = points_local * (2 * half_side_length) + origin
    return points_local


def chamfer_distance_icp_aligned(
    pc_pred: torch.Tensor,
    pc_gt: torch.Tensor,
    icp_align: bool = True,
    icp_max_iter: int = 100,
):
    """Calculate Chamfer distance between predicted and ground truth point clouds with optional ICP alignment.

    Args:
        pc_pred: Predicted point cloud tensor of shape (N, 3) or (B, N, 3)
        pc_gt: Ground truth point cloud tensor of shape (M, 3) or (B, M, 3)
        icp_align: Whether to align point clouds using iterative closest point algorithm before computing distance
        icp_max_iter: Maximum number of iterations for ICP algorithm

    Returns:
        loss_cd: Chamfer distance between the point clouds
        icp_solution: Solution from ICP algorithm if icp_align=True, otherwise None
    """
    if pc_pred.ndim == 2:
        pc_pred = pc_pred.unsqueeze(0)
    if pc_gt.ndim == 2:
        pc_gt = pc_gt.unsqueeze(0)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        if icp_align:
            icp_solution = iterative_closest_point(
                pc_pred.float(),
                pc_gt.float(),
                max_iterations=icp_max_iter,
            )
            pc_pred = icp_solution.Xt
        else:
            icp_solution = None

    loss_cd, _ = chamfer_distance(pc_pred, pc_gt)

    return loss_cd, icp_solution


def normalize_pcd(points: torch.Tensor) -> torch.Tensor:
    """
    Normalize a point cloud to be centered at the origin and fit within a cube from -0.5 to 0.5.
    Args:
        points (torch.Tensor): A tensor of shape (N, 3) representing the point cloud.
    Returns:
        torch.Tensor: The normalized point cloud.
    """
    if points.numel() == 0:
        return points
    # Calculate the bounding box
    min_vals, _ = points.min(dim=0)
    max_vals, _ = points.max(dim=0)
    # Calculate the center of the bounding box
    bbox_center = (min_vals + max_vals) / 2
    # Center the point cloud using the bounding box center
    centered_points = points - bbox_center
    # Scale the point cloud to fit within [-0.5, 0.5]
    max_abs_value = torch.max(torch.abs(centered_points))
    scaled_points = centered_points / (2 * max_abs_value)
    return scaled_points
