"""Consumers that read a ``SegmentationView`` grid without the dense upscale.

Exactness against ``view.full_res()``::

    exact = view.is_dense_fallback      the grid is the image itself
    exact = False                       a low-resolution grid: the dense
                                        reference interpolates scores, then
                                        thresholds; these functions threshold
                                        cells and spread them over the pixels
                                        each cell covers

``instance_areas`` and ``zone_fractions`` return a ``GridEstimate`` with an
``exact`` field. ``paint_masks`` returns ``(image, exact)``.
``nearest_full_res`` is a fidelity diagnostic; it is always approximate for a
low-resolution grid and returns no flag.

=====================  ==================================  ===================
function               low-resolution work                 dense equivalent
=====================  ==================================  ===================
instance_areas         count cells * cell area             ``mask.sum((1, 2))``
zone_fractions         cell centres inside the polygon     pixel centres inside
paint_masks            mean colour per cell, one nearest   same rule per pixel
                       expansion to the image, one blend
nearest_full_res       nearest expansion of cell masks     (fidelity only)
=====================  ==================================  ===================

Nothing here reads tensor values on the host: fixed op count, no ``.item()``.
"""

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
from roboflow_workflows.execution_engine.v2.blocks.segmentation_views import (
    SegmentationView,
)
from roboflow_workflows.execution_engine.v2.errors import ContractError


@dataclass(frozen=True)
class GridEstimate:
    """A per-instance value computed on a view's grid.

    Args:
        values: ``(N,)`` float tensor, on the view's device.
        exact: ``True`` only for dense-fallback views.
        method: How the value was computed, for experiment records.
    """

    values: torch.Tensor
    exact: bool
    method: str


def instance_areas(view: SegmentationView) -> GridEstimate:
    """Foreground area of each instance in original-image pixels.

    Args:
        view: The masks.

    Returns:
        ``(N,)`` areas: foreground cells times the pixel area of one cell.
    """
    scale_x, scale_y = view.geometry.grid_to_image().scale_xy
    cells = view.low_res_binary().sum(dim=(1, 2), dtype=torch.float32)
    estimate = GridEstimate(
        values=cells * (scale_x * scale_y),
        exact=view.is_dense_fallback,
        method="foreground cells x cell area",
    )

    return estimate


def zone_fractions(
    view: SegmentationView, polygon_xy: Sequence[Tuple[float, float]]
) -> GridEstimate:
    """Fraction of each instance's foreground inside a polygon zone.

    A cell (or pixel) is inside when its centre is, by the even-odd rule.
    Instances without foreground get 0.

    Args:
        view: The masks.
        polygon_xy: At least three ``(x, y)`` vertices in original-image pixels.

    Returns:
        ``(N,)`` fractions in ``[0, 1]``.

    Raises:
        ContractError: On fewer than three vertices.
    """
    if len(polygon_xy) < 3:
        raise ContractError(f"A zone needs at least 3 vertices, got {len(polygon_xy)}")

    binary = view.low_res_binary()
    zone = _cell_centres_inside(view, polygon_xy=polygon_xy)

    foreground = binary.sum(dim=(1, 2), dtype=torch.float32)
    inside = (binary & zone).sum(dim=(1, 2), dtype=torch.float32)
    fractions = torch.where(
        foreground > 0, inside / foreground.clamp(min=1.0), torch.zeros_like(foreground)
    )
    estimate = GridEstimate(
        values=fractions,
        exact=view.is_dense_fallback,
        method="cell centres in polygon (even-odd)",
    )

    return estimate


def paint_masks(
    image: torch.Tensor,
    view: SegmentationView,
    *,
    colors: torch.Tensor,
    opacity: float,
) -> Tuple[torch.Tensor, bool]:
    """Blend instance colours over a copy of the image.

    A covered pixel gets the mean colour of the instances covering it,
    blended with ``opacity``. The cost is ``O(N*h*w + H*W)``: the colour layer
    is built on the grid and expanded once.

    Args:
        image: ``(3, H, W)`` uint8 image the view's geometry belongs to.
        view: The masks.
        colors: ``(N, 3)`` uint8 colours, row-aligned with the view.
        opacity: Mask opacity, ``0..1``.

    Returns:
        ``(image, exact)``: a new ``(3, H, W)`` uint8 image, and ``True`` only
        for a dense fallback, where it equals this painting rule applied to
        ``full_res()``. This rule is not ``gpu_mask_composite``'s.

    Raises:
        ContractError: When the image or colours do not fit the view.
    """
    height, width = view.geometry.original_size_hw
    if (
        image.ndim != 3
        or image.shape[0] != 3
        or tuple(image.shape[1:]) != (height, width)
    ):
        raise ContractError(
            f"image must be (3, {height}, {width}) for this view, got {tuple(image.shape)}"
        )
    if tuple(colors.shape) != (len(view), 3):
        raise ContractError(
            f"colors must be ({len(view)}, 3), got {tuple(colors.shape)}"
        )

    binary = view.low_res_binary().to(torch.float32)
    color_sum = torch.einsum("nhw,nc->chw", binary, colors.to(binary))
    count = binary.sum(dim=0, keepdim=True)
    layer = torch.cat([color_sum / count.clamp(min=1.0), (count > 0).to(binary)])
    layer = _place_on_image(view, layer=layer)

    color, coverage = layer[:3], layer[3:] * opacity
    blended = image.to(torch.float32) * (1.0 - coverage) + color * coverage
    painted = blended.round_().clamp_(0, 255).to(torch.uint8)

    return painted, view.is_dense_fallback


def nearest_full_res(view: SegmentationView) -> torch.Tensor:
    """Spread each cell's foreground over the pixels it covers.

    A fidelity diagnostic for comparisons against ``full_res()``, not a
    replacement for it: approximate for a low-resolution grid, exact only for
    a dense fallback. It allocates ``N x H x W`` bools like the dense path.

    Args:
        view: The masks.

    Returns:
        ``(N, H, W)`` bool masks in original-image pixels.
    """
    placed = _place_on_image(view, layer=view.low_res_binary().to(torch.uint8))
    masks = placed.bool()

    return masks


def _place_on_image(view: SegmentationView, *, layer: torch.Tensor) -> torch.Tensor:
    """Expand a ``(C, h, w)`` grid layer to ``(C, H, W)`` original pixels.

    Each crop pixel takes the cell nearest to its centre; pixels outside the
    static crop are zero.
    """
    geometry = view.geometry
    height, width = geometry.original_size_hw
    crop_h, crop_w = geometry.crop_size_hw
    offset_x, offset_y = geometry.static_crop_xywh[:2]

    if tuple(layer.shape[1:]) == (crop_h, crop_w):
        expanded = layer
    else:
        source = layer if layer.is_floating_point() else layer.to(torch.float32)
        expanded = torch.nn.functional.interpolate(
            source[None], size=(crop_h, crop_w), mode="nearest-exact"
        )[0].to(layer.dtype)
    if (crop_h, crop_w) == (height, width):
        return expanded

    canvas = layer.new_zeros((layer.shape[0], height, width))
    canvas[:, offset_y : offset_y + crop_h, offset_x : offset_x + crop_w] = expanded

    return canvas


def _cell_centres_inside(
    view: SegmentationView, *, polygon_xy: Sequence[Tuple[float, float]]
) -> torch.Tensor:
    """``(h, w)`` bool: cell centres, in image pixels, inside the polygon."""
    mapping = view.geometry.grid_to_image()
    height, width = view.geometry.unpadded_size_hw
    device = view.scores.device

    xs = torch.arange(width, device=device, dtype=torch.float32) + 0.5
    ys = torch.arange(height, device=device, dtype=torch.float32) + 0.5
    xs = xs * mapping.scale_xy[0] + mapping.offset_xy[0]
    ys = ys * mapping.scale_xy[1] + mapping.offset_xy[1]
    px, py = xs[None, :].expand(height, width), ys[:, None].expand(height, width)

    inside = torch.zeros((height, width), dtype=torch.bool, device=device)
    vertices = [(float(x), float(y)) for x, y in polygon_xy]
    for (x1, y1), (x2, y2) in zip(vertices, vertices[1:] + vertices[:1]):
        if y1 == y2:
            continue
        crosses = (y1 > py) != (y2 > py)
        x_at_y = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (px < x_at_y)

    return inside


__all__ = [
    "GridEstimate",
    "instance_areas",
    "nearest_full_res",
    "paint_masks",
    "zone_fractions",
]
