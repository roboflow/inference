"""
Utility functions for point map processing and intrinsics inference.
Extracted from moge library for use in lidra pipeline.
"""

from typing import Optional, Tuple, Union
import torch
import utils3d

# Import directly from moge for exact compatibility
from moge.utils.geometry_torch import (
    normalized_view_plane_uv,
    recover_focal_shift,
)
from moge.utils.geometry_numpy import (
    solve_optimal_focal_shift,
    solve_optimal_shift,
)


def infer_intrinsics_from_pointmap(
    points: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    fov_x: Optional[Union[float, torch.Tensor]] = None,
    mask_threshold: float = 0.5,
    force_projection: bool = False,
    apply_mask: bool = False,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Infer camera intrinsics from a point map.

    Exact implementation matching moge library's inference logic.

    Args:
        points: Point map tensor of shape (B, H, W, 3) or (H, W, 3)
        mask: Optional mask tensor of shape (B, H, W) or (H, W)
        fov_x: Optional horizontal field of view in degrees. If None, inferred from points
        mask_threshold: Threshold for binary mask creation
        force_projection: If True, recompute points using depth and intrinsics
        apply_mask: If True, apply mask to output points and depth
        device: Device for computation. If None, uses points.device

    Returns:
        Dictionary containing:
        - 'points': Camera-space points
        - 'intrinsics': Camera intrinsics matrix
        - 'depth': Depth map
        - 'mask': Binary mask
    """
    if device is None:
        device = points.device

    # Handle batch dimension
    squeeze_batch = False
    if points.dim() == 3:
        points = points.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)
        squeeze_batch = True

    height, width = points.shape[1:3]
    aspect_ratio = width / height

    # Always process the output in fp32 precision
    with torch.autocast(device_type=device.type, dtype=torch.float32):
        points, mask, fov_x = map(
            lambda x: x.float() if isinstance(x, torch.Tensor) else x,
            [points, mask, fov_x],
        )

        mask_binary = (
            mask > mask_threshold
            if mask is not None
            else torch.ones_like(points[..., 0], dtype=torch.bool)
        )

        # Add finite check to handle NaN and inf values
        finite_mask = torch.isfinite(points).all(dim=-1)
        mask_binary = mask_binary & finite_mask

        # Get camera-space point map. (Focal here is the focal length relative to half the image diagonal)
        if fov_x is None:
            # BUG: Recover focal shift numpy method has flipped outputs: https://github.com/microsoft/MoGe/issues/110
            shift, focal = recover_focal_shift(points, mask_binary)
        else:
            focal = (
                aspect_ratio
                / (1 + aspect_ratio**2) ** 0.5
                / torch.tan(
                    torch.deg2rad(
                        torch.as_tensor(fov_x, device=points.device, dtype=points.dtype)
                        / 2
                    )
                )
            )
            if focal.ndim == 0:
                focal = focal[None].expand(points.shape[0])
            _, shift = recover_focal_shift(points, mask_binary, focal=focal)
        fx = focal / 2 * (1 + aspect_ratio**2) ** 0.5 / aspect_ratio
        fy = focal / 2 * (1 + aspect_ratio**2) ** 0.5
        intrinsics = utils3d.torch.intrinsics_from_focal_center(fx, fy, 0.5, 0.5)
        depth = points[..., 2] + shift[..., None, None]

        # If projection constraint is forced, recompute the point map using the actual depth map
        if force_projection:
            points = utils3d.torch.depth_to_points(depth, intrinsics=intrinsics)
        else:
            shift_stacked = torch.stack(
                [torch.zeros_like(shift), torch.zeros_like(shift), shift], dim=-1
            )[..., None, None, :]
            points = points + shift_stacked

        # Apply mask if needed
        if apply_mask:
            points = torch.where(mask_binary[..., None], points, torch.inf)
            depth = torch.where(mask_binary, depth, torch.inf)

    return_dict = {
        "points": points.squeeze(0) if squeeze_batch else points,
        "intrinsics": intrinsics.squeeze(0) if squeeze_batch else intrinsics,
        "depth": depth.squeeze(0) if squeeze_batch else depth,
        "mask": mask_binary.squeeze(0) if squeeze_batch else mask_binary,
    }

    return return_dict
