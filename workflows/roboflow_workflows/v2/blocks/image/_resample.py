"""Resampling helpers shared by blocks that output a new root image.

Perspective correction, camera calibration and auto-rotate all follow one
path, on the image's device::

    source position of every output pixel   (block-specific, compute_dtype)
          │
          ▼
    remap_bilinear   ─▶ uint8 pixels, OpenCV pixel centres, black outside
          │
          ▼
    new_root_image   ─▶ fresh image id; keeps video_metadata and parent_row

``require_same_image`` guards blocks that pair an image with predictions.

``v2/image_preprocessing@v1`` rotate still uses its own copies of the
sampling and new-root steps. Its sampling neither rounds positions to 1/32
pixel nor maps non-finite positions to black, so it is not yet shared.
"""

from typing import Any

import torch
import torch.nn.functional as F
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.frames import frame_of
from roboflow_workflows.v2.data.image import ImageData

# Sub-pixel positions per pixel in OpenCV's bilinear remap.
REMAP_STEPS = 32


def compute_dtype(device: torch.device) -> torch.dtype:
    """Return the float type for source positions on ``device``.

    Args:
        device: Device the positions are computed on.

    Returns:
        ``torch.float64``, or ``torch.float32`` on MPS, which has no float64.
    """
    dtype = torch.float32 if device.type == "mps" else torch.float64

    return dtype


def remap_bilinear(
    pixels: torch.Tensor, *, source_x: torch.Tensor, source_y: torch.Tensor
) -> torch.Tensor:
    """Sample an image at given source positions, like ``cv2.remap``.

    Pixel centres sit at integer coordinates, as in OpenCV. Positions outside
    the image blend with black (``BORDER_CONSTANT`` 0); non-finite positions
    give black. Interpolation is float bilinear, rounded to the nearest
    level, so it may differ from OpenCV's fixed-point weights by 1 level.

    Args:
        pixels: ``(C, H, W)`` uint8 image.
        source_x: ``(H', W')`` float x position in ``pixels`` of each output
            pixel, on the same device.
        source_y: ``(H', W')`` float y position, same shape.

    Returns:
        ``(C, H', W')`` uint8 image on ``pixels``' device.
    """
    height, width = int(pixels.shape[-2]), int(pixels.shape[-1])
    outside = -2.0 * max(height, width)
    # OpenCV rounds positions to 1/32 pixel (INTER_BITS = 5) before blending.
    source_x = torch.round(source_x * REMAP_STEPS) / REMAP_STEPS
    source_y = torch.round(source_y * REMAP_STEPS) / REMAP_STEPS
    source_x = torch.nan_to_num(source_x, nan=outside, posinf=outside, neginf=outside)
    source_y = torch.nan_to_num(source_y, nan=outside, posinf=outside, neginf=outside)
    grid = torch.stack(
        ((2 * source_x + 1) / width - 1, (2 * source_y + 1) / height - 1), dim=-1
    ).to(torch.float32)
    sampled = F.grid_sample(
        pixels.unsqueeze(0).float(),
        grid.unsqueeze(0),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    remapped = sampled.squeeze(0).round().clamp(0, 255).to(torch.uint8)

    return remapped


def new_root_image(
    pixels: torch.Tensor, *, like: ImageData, image_id: str
) -> ImageData:
    """Wrap pixels whose geometry no provenance mapping can describe.

    Rotations, homographies and lens undistortion cannot be written as scale
    and offset, so the result is its own root, as in
    ``v2/image_preprocessing@v1`` rotate. It keeps ``like``'s source and time
    (``video_metadata``) and detection row (``parent_row``).

    Args:
        pixels: ``(C, H, W)`` uint8 image.
        like: Image the pixels were computed from.
        image_id: Identity of the new frame.

    Returns:
        A new root image.
    """
    image = ImageData.from_tensor(
        pixels, image_id=image_id, video_metadata=like.video_metadata
    ).with_parent_row(like.parent_row)

    return image


def require_same_image(prediction: Any, *, image: ImageData, consumer: str) -> None:
    """Refuse predictions made on another image than ``image``.

    Coordinates given with ``image`` (a polygon, the image's own pixels) are
    only comparable with predictions of that same image. A frame without an
    image id (imported data that cannot name its image) is not compared.

    Args:
        prediction: A native prediction carrying a frame.
        image: The image the caller pairs it with.
        consumer: Block type named in the error.

    Raises:
        ContractError: When the prediction names another image. Also when
            ``frame_of`` cannot read its frame: no ``image_metadata``, or
            missing or invalid frame geometry.
    """
    frame = frame_of(prediction)
    if frame.image_id is not None and frame.image_id != image.image_id:
        raise ContractError(
            f"{consumer}: the predictions describe image {frame.image_id!r}, but "
            f"the image given is {image.image_id!r}; connect predictions made on "
            "that same image"
        )
