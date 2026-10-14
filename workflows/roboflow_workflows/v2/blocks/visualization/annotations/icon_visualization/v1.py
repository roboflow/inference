"""Icons at fixed points or on detections: ``roboflow_core/icon_visualization@v1``.

Port of V1 ``roboflow_core/icon_visualization@v1`` (``sv.IconAnnotator``)::

    icon    letterboxed into icon_width x icon_height: resized keeping its
            aspect ratio (INTER_LINEAR), centred; the border is transparent.
            icon_alpha, when given, is resized with the color channels as
            one 4-channel image, as V1 resized BGRA
    anchor  static:  (x0 + icon_width // 2, y0 + icon_height // 2), where
                     x0 = x_position, or width + x_position - icon_width
                     when x_position < 0 (y0 likewise)
            dynamic: each row's anchor point at ``position``, truncated
    corner  int(anchor - icon_size / 2)    truncated toward zero, as V1
    output  the icon pasted at each corner in row order, clipped to the image;
            with icon_alpha a (0..255) blended in exact integers:
            (image * (255 - a) + icon * a) // 255

Transparency: ``ImageData`` has 1 or 3 channels, so it has no alpha. Pass
the icon's alpha channel as ``icon_alpha``, an ``(H, W)`` tensor of the
icon's size: uint8 0..255, or float 0..1 (rounded to 1/255 steps). Without
it the icon is opaque. To use an RGBA PNG, split it where it is loaded (see
example ``75-decorative-drawings``). V1 instead reloaded the icon's file or
base64 source here; V2 does not read files or URLs in ``run()``.

V1 blended in float64 and truncated. The integer blend equals it except
where the exact result is a whole number that float64 lands just below:
there V1 is 1 lower (0.074% of all image, icon, alpha triples).

    host    dynamic: one transfer of boxes (or mask centres); anchors.
            float icon_alpha: one range check
    device  icon resize (``raster.resize_linear``: OpenCV sample positions,
            float32 weights, so a pixel may differ by 1 from V1); one slice
            copy (opaque) or blend (icon_alpha) per icon

Box positions never read masks; ``CENTER_OF_MASS`` builds one full-image
mask per row at ``mask_quality``.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple
import torch
from pydantic import Field, StrictInt, model_validator
from roboflow_workflows.v2.blocks.visualization.annotations.anchors import (
    CENTER_OF_MASS,
    box_anchors,
    mass_centres,
)
from roboflow_workflows.v2.blocks.visualization.annotations.base import (
    annotated,
    canvas_of,
)
from roboflow_workflows.v2.blocks.visualization.annotations.params import (
    POSITIONS_WITH_MASS,
    check_literals,
    checked_choice,
    checked_int,
    checked_mask_quality,
    reject_copy_image,
)
from roboflow_workflows.v2.blocks.visualization.annotations.raster import resize_linear
from roboflow_workflows.v2.blocks.visualization.annotations.rows import (
    BOX_KINDS,
    MASK_QUALITIES,
    boxes_of,
    dense_masks_of,
    has_masks,
    placed_in,
    read_rows,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.data.predictions import TENSOR_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


MODES = ("static", "dynamic")

class IconVisualizationBlock(Block):
    """Paste an icon at a fixed point or on every detection, on a copy.

    Examples: ``mode="static", x_position=-10, y_position=10`` puts one icon
    10 px from the top-right corner (a watermark). ``mode="dynamic",
    position="TOP_CENTER"`` (defaults) centres an icon on the top edge of
    each detection's box.
    """

    type = "roboflow_core/icon_visualization@v1"
    prunable = True
    metadata = {
        "name": "Icon Visualization",
        "port_of": "roboflow_core/icon_visualization@v1",
        "section": "visualization",
    }
    outputs = {
        "image": Output(
            IMAGE_KIND,
            source="image",
            description=(
                "New annotated image with the input's identity, frame, source, "
                "time and parent row."
            ),
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description=(
                "RGB image to draw on; not modified. The output is a new image "
                "with the same identity, source and time."
            ),
            examples=["$inputs.image", "$steps.mask_visualization.image"],
        )
        icon: Ref(IMAGE_KIND) = Field(
            description=(
                "Icon image, 1 or 3 channels; resized to icon_width x "
                "icon_height keeping its aspect ratio. Images have no alpha "
                "channel: pass transparency as icon_alpha."
            ),
            examples=["$inputs.icon"],
        )
        icon_alpha: Optional[Ref(TENSOR_KIND)] = Field(
            default=None,
            description=(
                "Optional icon transparency: an (H, W) tensor of the icon's "
                "size, uint8 0..255 or float 0..1; 0 is transparent. Resized "
                "with the icon. Omitted: the icon is opaque. For an RGBA PNG, "
                "pass its RGB as icon and its alpha channel here."
            ),
            examples=["$inputs.icon_alpha"],
        )
        mode: Literal[MODES] | Ref(STRING_KIND) = Field(
            default="dynamic",
            description=(
                "static: one icon at x_position, y_position. dynamic: one icon "
                "per detection of predictions, at position."
            ),
            examples=["static", "dynamic", "$inputs.mode"],
        )
        predictions: Optional[Ref(*BOX_KINDS)] = Field(
            default=None,
            description=(
                "Detections to place icons on; required in dynamic mode, "
                "ignored in static mode. Predictions of another frame of the "
                "same workflow input are mapped into this image."
            ),
            examples=["$steps.object_detection.predictions"],
        )
        icon_width: StrictInt | Ref(INTEGER_KIND) = Field(
            default=64,
            description="Width of the icon box in pixels; >= 1.",
            examples=[64, "$inputs.icon_width"],
        )
        icon_height: StrictInt | Ref(INTEGER_KIND) = Field(
            default=64,
            description="Height of the icon box in pixels; >= 1.",
            examples=[64, "$inputs.icon_height"],
        )
        position: Literal[POSITIONS_WITH_MASS] | Ref(STRING_KIND) = Field(
            default="TOP_CENTER",
            description=(
                "Dynamic mode: point of each detection the icon is centred "
                "on: a box anchor, or CENTER_OF_MASS of its instance mask."
            ),
            examples=["TOP_CENTER", "CENTER", "$inputs.position"],
        )
        x_position: StrictInt | Ref(INTEGER_KIND) = Field(
            default=10,
            description=(
                "Static mode: left edge of the icon box; a negative value "
                "places its right edge that many pixels from the image's right "
                "edge."
            ),
            examples=[10, -10, "$inputs.x_position"],
        )
        y_position: StrictInt | Ref(INTEGER_KIND) = Field(
            default=10,
            description=(
                "Static mode: top edge of the icon box; a negative value "
                "places its bottom edge that many pixels from the image's "
                "bottom edge."
            ),
            examples=[10, -10, "$inputs.y_position"],
        )
        mask_quality: Literal[MASK_QUALITIES] | Ref(STRING_KIND) = Field(
            default="accurate",
            description=(
                "How instance masks are drawn; used when position is CENTER_OF_MASS. "
                "Each read builds one full-image mask per detection, at either "
                "quality. accurate: the masks the model defines (a low-resolution mask "
                "view is upscaled with the model's own interpolation). coarse: a mask "
                "view's grid cells are thresholded, then expanded, so edges can be "
                "blocky and small parts can be lost; it skips the accurate upscaling "
                "only. roboflow_core/roboflow_instance_segmentation_model@v4 currently "
                "returns full-resolution masks, so both give the same result for it; "
                "they differ only for producers of compact mask views."
            ),
            examples=["accurate", "coarse", "$inputs.mask_quality"],
        )

        @model_validator(mode="before")
        @classmethod
        def _reject_copy_image(cls, values: Any) -> Any:
            checked = reject_copy_image(values, title=cls.__name__)

            return checked

        @model_validator(mode="after")
        def _check_literals(self) -> "IconVisualizationBlock.Params":
            check_literals(self.__dict__, _CHECKS)
            if self.mode == "dynamic" and self.predictions is None:
                raise ValueError(_NO_PREDICTIONS)

            return self

    def run(
        self,
        *,
        image: ImageData,
        icon: ImageData,
        icon_alpha: Optional[torch.Tensor],
        mode: str,
        predictions: Any,
        icon_width: int,
        icon_height: int,
        position: str,
        x_position: int,
        y_position: int,
        mask_quality: str,
    ) -> Dict[str, Any]:
        """Paste ``icon`` onto a copy of ``image``.

        Args:
            image: RGB image; not modified.
            icon: Icon image, 1 or 3 channels; not modified.
            icon_alpha: ``(H, W)`` icon transparency, uint8 0..255 or float
                0..1, or ``None`` for an opaque icon; not modified.
            mode: ``static`` or ``dynamic``.
            predictions: A box or mask carrier, or ``None`` in static mode.
            icon_width: Icon box width.
            icon_height: Icon box height.
            position: Dynamic anchor position.
            x_position: Static left edge (negative: from the right).
            y_position: Static top edge (negative: from the bottom).
            mask_quality: ``accurate`` or ``coarse``.

        Returns:
            ``image``: the new annotated image.

        Raises:
            ValueError: On invalid parameters, an ``icon_alpha`` of another
                shape, dtype or range, dynamic mode without predictions,
                CENTER_OF_MASS without masks, keypoints without boxes or
                predictions of another workflow input.
        """
        values = dict(
            mode=mode,
            icon_width=icon_width,
            icon_height=icon_height,
            position=position,
            x_position=x_position,
            y_position=y_position,
            mask_quality=mask_quality,
        )
        for name, check in _CHECKS.items():
            check(values[name])
        canvas = canvas_of(image, block=self.type)
        alpha = None
        if icon_alpha is not None:
            alpha = _alpha_of(icon_alpha, icon=icon, device=canvas.device)
        if mode == "static":
            anchor = _static_anchor(
                image,
                x_position=x_position,
                y_position=y_position,
                width=icon_width,
                height=icon_height,
            )
            anchors = [anchor]
        elif predictions is None:
            raise ValueError(f"{self.type}: {_NO_PREDICTIONS}")
        else:
            anchors = self._anchors(
                predictions, image=image, position=position, mask_quality=mask_quality
            )
        if not anchors:
            result = annotated(image, canvas)
            return result

        pixels, alpha, (left, top) = _letterboxed(
            icon, alpha, (icon_height, icon_width), device=canvas.device
        )
        for anchor_x, anchor_y in anchors:
            corner_x = int(anchor_x - icon_width / 2)
            corner_y = int(anchor_y - icon_height / 2)
            _paste(canvas, pixels, alpha=alpha, x=corner_x + left, y=corner_y + top)
        result = annotated(image, canvas)

        return result

    def _anchors(
        self, predictions: Any, *, image: ImageData, position: str, mask_quality: str
    ) -> List[Tuple[int, int]]:
        """Return each row's integer anchor point in the image."""
        if position == CENTER_OF_MASS:
            if not has_masks(predictions):
                raise ValueError(
                    f"{self.type}: position CENTER_OF_MASS needs instance masks, "
                    f"got {type(predictions).__name__}; use a box position "
                    "such as CENTER"
                )
            dense = dense_masks_of(predictions, quality=mask_quality, block=self.type)
            dense = placed_in(dense, image, block=self.type)
            anchors = mass_centres(dense.mask)
        else:
            boxes = placed_in(
                boxes_of(predictions, block=self.type), image, block=self.type
            )
            anchors = box_anchors(read_rows(boxes).xyxy, position).astype(int)
        anchors = [(int(x), int(y)) for x, y in anchors.tolist()]

        return anchors

def _static_anchor(
    image: ImageData, *, x_position: int, y_position: int, width: int, height: int
) -> Tuple[int, int]:
    """Return V1's static anchor: the icon box centre, rounded down."""
    left = x_position if x_position >= 0 else image.width + x_position - width
    top = y_position if y_position >= 0 else image.height + y_position - height
    anchor = (left + width // 2, top + height // 2)

    return anchor

def _alpha_of(
    icon_alpha: Any, *, icon: ImageData, device: torch.device
) -> torch.Tensor:
    """Return ``icon_alpha`` as ``(1, H, W)`` uint8 0..255 on ``device``."""
    if not isinstance(icon_alpha, torch.Tensor):
        raise ValueError(
            f"icon_alpha must be a torch.Tensor, got {type(icon_alpha).__name__}"
        )
    if tuple(icon_alpha.shape) != icon.size_hw:
        raise ValueError(
            f"icon_alpha must have the icon's (height, width) {icon.size_hw}, "
            f"got shape {tuple(icon_alpha.shape)}"
        )
    alpha = icon_alpha.to(device)
    if alpha.dtype == torch.uint8:
        return alpha.unsqueeze(0)
    if not alpha.is_floating_point():
        raise ValueError(
            f"icon_alpha must be uint8 (0..255) or float (0..1), got {alpha.dtype}"
        )

    alpha = alpha.to(torch.float32)
    # One host sync. NaN fails both comparisons, so it is rejected too.
    if not bool(((alpha >= 0) & (alpha <= 1)).all()):
        raise ValueError("float icon_alpha must be finite and within 0..1")
    alpha = (alpha * 255).round().to(torch.uint8).unsqueeze(0)

    return alpha

def _letterboxed(
    icon: ImageData,
    alpha: Optional[torch.Tensor],
    size_hw: Tuple[int, int],
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[int, int]]:
    """Resize an icon into a box keeping its aspect ratio, as ``sv.letterbox_image``.

    Args:
        icon: Icon image, 1 or 3 channels.
        alpha: ``(1, H, W)`` uint8 alpha on ``device``, or ``None``.
        size_hw: Box ``(height, width)``.
        device: Device of the result.

    Returns:
        The resized ``(3, h, w)`` uint8 pixels, the resized ``(1, h, w)``
        alpha (or ``None``) and their ``(left, top)`` offset in the box; the
        rest of the box is transparent.
    """
    box_height, box_width = size_hw
    icon_height, icon_width = icon.size_hw
    ratio = icon_width / icon_height
    if ratio >= box_width / box_height:
        width, height = box_width, int(box_width / ratio)
    else:
        width, height = int(box_height * ratio), box_height
    # V1 fails on a zero-pixel side; a very thin icon keeps one pixel here.
    width, height = max(width, 1), max(height, 1)
    pixels = icon.tensor_image.to(device).expand(3, icon_height, icon_width)
    if alpha is not None:
        # One resize of color and alpha together: the same weights for both.
        pixels = torch.cat([pixels, alpha])
    resized = resize_linear(
        pixels,
        (height, width),
        scale_xy=(icon_width / width, icon_height / height),
    )
    resized_alpha = resized[3:] if alpha is not None else None
    resized = resized[:3]
    offset = ((box_width - width) // 2, (box_height - height) // 2)

    return resized, resized_alpha, offset

def _paste(
    canvas: torch.Tensor,
    pixels: torch.Tensor,
    *,
    alpha: Optional[torch.Tensor],
    x: int,
    y: int,
) -> None:
    """Write or blend ``pixels`` with their top-left corner at ``(x, y)``, clipped."""
    _, height, width = pixels.shape
    _, canvas_height, canvas_width = canvas.shape
    left, top = max(x, 0), max(y, 0)
    right, bottom = min(x + width, canvas_width), min(y + height, canvas_height)
    if left >= right or top >= bottom:
        return

    window = (slice(None), slice(top - y, bottom - y), slice(left - x, right - x))
    if alpha is None:
        canvas[:, top:bottom, left:right] = pixels[window]
        return

    weight = alpha[window].to(torch.int32)
    below = canvas[:, top:bottom, left:right].to(torch.int32)
    blended = (below * (255 - weight) + pixels[window].to(torch.int32) * weight) // 255
    canvas[:, top:bottom, left:right] = blended.to(torch.uint8)

def _checked_integer(value: Any, *, name: str) -> int:
    """Return ``value`` after checking it is an integer (negative allowed)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")

    return value

_NO_PREDICTIONS = (
    "mode 'dynamic' places one icon per detection and needs predictions; "
    "set predictions or use mode 'static'"
)

_CHECKS = {
    "mode": lambda value: checked_choice(value, name="mode", choices=MODES),
    "icon_width": lambda value: checked_int(value, name="icon_width", minimum=1),
    "icon_height": lambda value: checked_int(value, name="icon_height", minimum=1),
    "position": lambda value: checked_choice(
        value, name="position", choices=POSITIONS_WITH_MASS
    ),
    "x_position": lambda value: _checked_integer(value, name="x_position"),
    "y_position": lambda value: _checked_integer(value, name="y_position"),
    "mask_quality": checked_mask_quality,
}
