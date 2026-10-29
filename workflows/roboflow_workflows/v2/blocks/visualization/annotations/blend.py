"""Color layers blended over an image on its device (mask and semantic blocks).

V1 ``sv.MaskAnnotator`` paints a solid copy of the frame, then blends it::

    colored = image; for each mask, largest first: colored[mask] = color
    output  = round(colored * opacity + image * (1 - opacity))   # cv2.addWeighted

``paint_instances`` and ``blend`` do the same with float32 weights and
round-half-to-even. Neither reads tensor values on the host. The painted
layer is exact. The blend equals OpenCV for opacity 0, 0.25, 0.5, 0.75 and 1;
at other opacities a pixel may differ by 1 level where the weighted sum is
an exact .5 tie, which OpenCV's fused arithmetic rounds differently.
"""

from typing import Optional

import numpy as np
import torch
from roboflow_workflows.v2.blocks.visualization.annotations.colors import UNKNOWN_COLOR


def device_colors(palette: np.ndarray, indices: torch.Tensor) -> torch.Tensor:
    """Look up palette colors on the indices' device; negative indices are grey.

    Args:
        palette: ``(P, 3)`` uint8 RGB colors on the host.
        indices: Integer palette indices of any shape, on any device.

    Returns:
        ``indices.shape + (3,)`` uint8 colors on the indices' device.
    """
    # A copy: cached palettes are read-only, which torch.from_numpy warns about.
    table = torch.tensor(palette, dtype=torch.uint8, device=indices.device)
    grey = torch.tensor(UNKNOWN_COLOR, dtype=torch.uint8, device=indices.device)
    indices = indices.to(torch.int64)
    colors = table[torch.remainder(indices, len(palette))]
    colors = torch.where((indices < 0).unsqueeze(-1), grey, colors)

    return colors


def paint_instances(
    canvas: torch.Tensor, masks: torch.Tensor, colors: torch.Tensor
) -> torch.Tensor:
    """Return a copy of ``canvas`` with masks painted largest first.

    Equal areas paint the later row first, as V1's ``np.flip(np.argsort())``
    does for up to 16 rows.

    Args:
        canvas: ``(3, H, W)`` uint8 image.
        masks: ``(N, H, W)`` bool masks on the same device.
        colors: ``(N, 3)`` uint8 colors on the same device.

    Returns:
        New ``(3, H, W)`` uint8 solid color layer.
    """
    colored = canvas.clone()
    if len(masks) == 0:
        return colored

    areas = masks.sum(dim=(1, 2))
    order = torch.sort(areas, stable=True).indices.flip(0)
    ordered_masks, ordered_colors = masks[order], colors[order]
    for index in range(len(masks)):
        colored = torch.where(
            ordered_masks[index], ordered_colors[index].view(3, 1, 1), colored
        )

    return colored


def blend(
    canvas: torch.Tensor,
    colored: torch.Tensor,
    *,
    opacity: float,
    region: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Blend a color layer over an image like ``cv2.addWeighted``.

    Exact for opacity 0, 0.25, 0.5, 0.75 and 1; otherwise within 1 level, at
    exact .5 ties only (see the module docstring).

    Args:
        canvas: ``(3, H, W)`` uint8 image.
        colored: ``(3, H, W)`` uint8 color layer.
        opacity: Weight of the layer, in [0, 1].
        region: Optional ``(H, W)`` bool; pixels outside keep ``canvas``.

    Returns:
        New contiguous ``(3, H, W)`` uint8 image.
    """
    alpha = torch.tensor(opacity, dtype=torch.float32)
    beta = torch.tensor(1.0 - opacity, dtype=torch.float32)
    mixed = colored.to(torch.float32) * alpha.to(canvas.device) + canvas.to(
        torch.float32
    ) * beta.to(canvas.device)
    blended = mixed.round().clamp(0, 255).to(torch.uint8)
    if region is not None:
        blended = torch.where(region, blended, canvas)
    # A permuted color layer gives permuted strides; images are contiguous.
    blended = blended.contiguous()

    return blended
