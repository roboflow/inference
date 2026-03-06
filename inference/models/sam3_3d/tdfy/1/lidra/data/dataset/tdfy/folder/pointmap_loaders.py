"""
Pointmap loaders for FolderDataset
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from pytorch3d.io import load_ply
from pytorch3d.transforms import Transform3d
from loguru import logger
from typing import Optional

from lidra.data.dataset.tdfy.trellis.pose_loader import R3


def load_ply_pointmap(
    basename: str, pointmap_dir: str, sha256: str = "", image_fname: str = ""
) -> torch.Tensor:
    """
    Load pointmap from PLY file with simplified logic for tracking_dog format.

    Expected structure:
        pointmap_dir/
            frame_XXXXXX/
                frame_XXXXXX.ply
                frame_XXXXXX_metadata.json

    Args:
        basename: Frame identifier (e.g., "frame_000000")
        pointmap_dir: Directory containing frame subdirectories
        sha256: Unused (kept for compatibility)
        image_fname: Unused (kept for compatibility)

    Returns:
        torch.Tensor: Pointmap tensor of shape (3, H, W)
    """
    # Use subdirectory structure: frame_XXXXXX/frame_XXXXXX.ply
    subdir = Path(pointmap_dir) / basename
    ply_path = subdir / f"{basename}.ply"
    metadata_path = subdir / f"{basename}_metadata.json"

    # Check if files exist
    if not ply_path.exists():
        raise FileNotFoundError(
            f"PLY file not found: {ply_path}\n"
            f"Expected structure: {pointmap_dir}/{basename}/{basename}.ply"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            f"Expected structure: {pointmap_dir}/{basename}/{basename}_metadata.json"
        )

    # Load PLY file using pytorch3d
    verts, faces = load_ply(str(ply_path))

    # Load metadata to get image dimensions
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    if "image_size" not in metadata:
        raise ValueError(
            f"Metadata at {metadata_path} missing 'image_size' field. "
            f"Found keys: {list(metadata.keys())}"
        )

    # Get dimensions from metadata [width, height]
    width, height = metadata["image_size"]
    expected_points = width * height

    # Verify point count matches expected dimensions
    if verts.shape[0] != expected_points:
        raise ValueError(
            f"Shape mismatch: PLY has {verts.shape[0]} points, "
            f"but metadata specifies {width}x{height} = {expected_points} points"
        )

    # Reshape to (H, W, 3)
    pointmap = verts.reshape(height, width, 3)
    # pointmap[:, :, 2] *= -1
    # pointmap[:, :, 1] *= -1

    # Apply camera convention transform (R3 -> PyTorch3D coordinate system)
    camera_convention_transform = Transform3d().rotate(
        R3.r3_camera_to_pytorch3d_camera(device="cpu").rotation
    )
    pointmap = camera_convention_transform.transform_points(pointmap)

    # Replace non-finite values (inf, -inf, nan) with nan for proper handling
    pointmap = torch.where(torch.isfinite(pointmap), pointmap, torch.nan)

    # Transpose to channel-first format (3, H, W)
    return pointmap.permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)


def load_ply_pointmap_simple(
    basename: str, sha256: str = "", image_fname: str = ""
) -> torch.Tensor:
    """
    Simplified PLY loader that expects pointmap_dir to be bound via partial.
    This is the version typically used in configs with _partial_: True.

    Example config:
        pointmap_loader:
          _target_: lidra.data.dataset.tdfy.folder.pointmap_loaders.load_ply_pointmap_simple
          _partial_: True
          pointmap_dir: "/path/to/pointmaps"
    """
    # This will fail if pointmap_dir is not bound
    raise NotImplementedError(
        "This loader expects pointmap_dir to be bound via functools.partial. "
        "Use it with _partial_: True in the config."
    )
