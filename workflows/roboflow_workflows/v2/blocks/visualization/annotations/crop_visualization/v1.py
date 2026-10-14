"""Magnified crops: ``roboflow_core/crop_visualization@v1``.

Port of V1 ``roboflow_core/crop_visualization@v1`` (``sv.CropAnnotator``).
Per row, in order::

    crop    = original image inside the integer box, clipped to the image
    scaled  = resize to int(w * scale_factor) x int(h * scale_factor), bilinear
    placed  at the anchor of the whole box (position), clipped to the frame
    border  cv2.rectangle around the placed crop, row color

Crops come from the input image, so a later crop never shows an earlier
one. Placement and borders equal V1; the bilinear pixels follow OpenCV's
sample positions with float weights and may differ from its fixed point by
1 level. Rows whose crop or scaled size is empty are skipped (V1 failed in
cv2.resize). Every row's resize tables travel to the device in one copy
per call.

Boxes mapped from another frame may extend past the left or top edge.
Only the part inside the image is copied, and the copy keeps the anchor of
the whole box. V1 sliced negative coordinates Python-style, which drops such
a crop or copies pixels from the opposite edge.
"""

from typing import Any, Dict, List, Literal, NamedTuple, Optional, Tuple
import cv2
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
from roboflow_workflows.v2.blocks.visualization.annotations.colors import (
    COLOR_AXES,
    PALETTE_NAMES,
)
from roboflow_workflows.v2.blocks.visualization.annotations.params import (
    Number,
    POSITIONS_WITH_MASS,
    check_literals,
    check_palette_literals,
    checked_choice,
    checked_int,
    checked_mask_quality,
    checked_number,
    reject_copy_image,
    row_colors,
)
from roboflow_workflows.v2.blocks.visualization.annotations.raster import (
    LinearTables,
    apply_linear,
    linear_tables,
    upload_tables,
)
from roboflow_workflows.v2.blocks.visualization.annotations.rows import (
    BOX_KINDS,
    MASK_QUALITIES,
    boxes_of,
    dense_masks_of,
    has_masks,
    placed_in,
    read_rows,
)
from roboflow_workflows.v2.blocks.visualization.annotations.strokes import (
    Strokes,
    rectangle_scan,
    stroke_reach,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


class _Crop(NamedTuple):
    # The visible part of a box (x1, y1, x2, y2) and its resize to the copy.
    window: Tuple[int, int, int, int]
    size_hw: Tuple[int, int]
    tables: LinearTables

Placement = Tuple[_Crop, int, int]

class CropVisualizationBlock(Block):
    """Paste a magnified copy of each detection next to it, with a border.

    Example: defaults show each box's contents at twice the size, centred
    above the box, framed in the class color.
    """

    type = "roboflow_core/crop_visualization@v1"
    prunable = True
    metadata = {
        "name": "Crop Visualization",
        "port_of": "roboflow_core/crop_visualization@v1",
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
        color_palette: Literal[PALETTE_NAMES] | Ref(STRING_KIND) = Field(
            default="DEFAULT",
            description="Palette of the drawn elements.",
            examples=["DEFAULT", "ROBOFLOW", "Matplotlib Viridis", "$inputs.palette"],
        )
        palette_size: StrictInt | Ref(INTEGER_KIND) = Field(
            default=10,
            description="Number of colors taken from a Matplotlib palette; >= 1.",
            examples=[10, "$inputs.palette_size"],
        )
        custom_colors: List[str] | Ref(LIST_OF_VALUES_KIND) = Field(
            default_factory=list,
            description=(
                "Colors of the CUSTOM palette: #RRGGBB, rgb(R, G, B), "
                "bgr(B, G, R) or a color name such as WHITE."
            ),
            examples=[["#FF0000", "#00FF00", "#0000FF"], "$inputs.custom_colors"],
        )
        color_axis: Literal[COLOR_AXES] | Ref(STRING_KIND) = Field(
            default="CLASS",
            description=(
                "How a row picks its palette color: CLASS (class id), INDEX (row "
                "position) or TRACK (bboxes_metadata tracker_id). Negative ids are "
                "grey."
            ),
            examples=["CLASS", "$inputs.color_axis"],
        )
        predictions: Ref(*BOX_KINDS) = Field(
            description=(
                "Detections, instance segmentation (dense, RLE or view) or keypoint "
                "predictions with boxes, of this image or of another frame of the "
                "same workflow input (for example a crop or its parent). Those are "
                "mapped into this image; parts outside it are clipped."
            ),
            examples=[
                "$steps.object_detection.predictions",
                "$steps.segmentation.segmentation",
            ],
        )
        position: Literal[POSITIONS_WITH_MASS] | Ref(STRING_KIND) = Field(
            default="TOP_CENTER",
            description=(
                "Anchor of the magnified crop on the box. CENTER_OF_MASS needs "
                "instance masks and builds one full-image mask per detection "
                "(see mask_quality)."
            ),
            examples=["TOP_CENTER", "CENTER"],
        )
        scale_factor: Number | Ref(FLOAT_KIND) = Field(
            default=2.0,
            description="Size of the crop copy relative to the box; > 0.",
            examples=[2.0, "$inputs.scale_factor"],
        )
        border_thickness: StrictInt | Ref(INTEGER_KIND) = Field(
            default=2,
            description="Border width around the copy; >= 1.",
            examples=[2],
        )
        mask_quality: Literal[MASK_QUALITIES] | Ref(STRING_KIND) = Field(
            default="accurate",
            description=(
                "How instance masks are drawn; used when position is CENTER_OF_MASS on "
                "instance masks. Each read builds one full-image mask per detection, "
                "at either quality. accurate: the masks the model defines (a "
                "low-resolution mask view is upscaled with the model's own "
                "interpolation). coarse: a mask view's grid cells are thresholded, "
                "then expanded, so edges can be blocky and small parts can be lost; it "
                "skips the accurate upscaling only. "
                "roboflow_core/roboflow_instance_segmentation_model@v4 currently "
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
        def _check_palette_literals(self) -> "CropVisualizationBlock.Params":
            check_palette_literals(
                palette_size=self.palette_size, custom_colors=self.custom_colors
            )

            return self

        @model_validator(mode="after")
        def _check_literals(self) -> "CropVisualizationBlock.Params":
            check_literals(self.__dict__, _CHECKS)

            return self

    def run(
        self,
        *,
        image: ImageData,
        predictions: Any,
        color_palette: str,
        palette_size: int,
        custom_colors: Any,
        color_axis: str,
        position: str,
        scale_factor: float,
        border_thickness: int,
        mask_quality: str,
    ) -> Dict[str, Any]:
        """Paste a magnified copy of every row onto a copy of ``image``.

        Args:
            image: RGB image; not modified.
            predictions: A box carrier (see ``rows``).
            color_palette: Palette name.
            palette_size: Matplotlib sample count.
            custom_colors: ``CUSTOM`` palette colors.
            color_axis: ``CLASS``, ``INDEX`` or ``TRACK``; border colors.
            position: Anchor of each copy.
            scale_factor: Copy size relative to the box.
            border_thickness: Border width.
            mask_quality: Mask quality for ``CENTER_OF_MASS``.

        Returns:
            ``image``: the new annotated image.

        Raises:
            ValueError: On invalid parameters, ``CENTER_OF_MASS`` without
                masks, keypoints without boxes or predictions of another
                workflow input.
        """
        _CHECKS["position"](position)
        _CHECKS["scale_factor"](scale_factor)
        _CHECKS["border_thickness"](border_thickness)
        _CHECKS["mask_quality"](mask_quality)
        canvas = canvas_of(image, block=self.type)
        boxes = placed_in(
            boxes_of(predictions, block=self.type), image, block=self.type
        )
        rows = read_rows(boxes)
        if len(rows) == 0:
            result = annotated(image, canvas)
            return result

        if position == CENTER_OF_MASS:
            if not has_masks(predictions):
                raise ValueError(
                    f"{self.type}: position CENTER_OF_MASS needs instance masks; "
                    "these predictions have none"
                )
            dense = dense_masks_of(predictions, quality=mask_quality, block=self.type)
            anchors = mass_centres(placed_in(dense, image, block=self.type).mask)
        else:
            anchors = box_anchors(rows.xyxy, position).astype(int)
        colors = row_colors(
            rows.class_id,
            rows.tracker_ids(),
            color_palette=color_palette,
            palette_size=palette_size,
            custom_colors=custom_colors,
            color_axis=color_axis,
        )

        strokes = Strokes(image.size_hw)
        placements: List[Placement] = []
        reach = stroke_reach(border_thickness)
        # Anchors come from the whole box; only its visible part is copied.
        for corners, anchor, color in zip(
            rows.xyxy.astype(int).tolist(), anchors, colors
        ):
            crop = _crop_of(image.size_hw, corners, scale_factor=scale_factor)
            if crop is None:
                continue
            height, width = crop.size_hw
            (x1, y1), (x2, y2) = _crop_corners(
                (int(anchor[0]), int(anchor[1])),
                width=width,
                height=height,
                position=position,
            )
            placements.append((crop, x1, y1))
            strokes.add(
                lambda mask, a=(x1, y1), b=(x2, y2): cv2.rectangle(
                    mask, a, b, 1, border_thickness
                ),
                color=color,
                scan=rectangle_scan((x1, y1, x2, y2), reach=reach),
            )

        pixels = image.tensor_image
        uploaded = upload_tables(
            [crop.tables for crop, _, _ in placements], device=pixels.device
        )
        borders = strokes.upload(canvas.device)
        for index, ((crop, x, y), tables) in enumerate(zip(placements, uploaded)):
            x1, y1, x2, y2 = crop.window
            scaled = apply_linear(pixels[:, y1:y2, x1:x2], tables)
            _overlay(canvas, scaled, x=x, y=y)
            borders.paint_one(canvas, index)
        result = annotated(image, canvas)

        return result

def _crop_of(
    image_hw: Tuple[int, int], corners: List[int], *, scale_factor: float
) -> Optional[_Crop]:
    # sv crop_image then scale_image (int sizes, INTER_LINEAR). The box is
    # clipped first: a negative bound must not wrap to the opposite edge.
    image_height, image_width = image_hw
    x1, y1, x2, y2 = corners
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, image_width), min(y2, image_height)
    if x2 <= x1 or y2 <= y1:
        return None

    height, width = y2 - y1, x2 - x1
    new_width, new_height = int(width * scale_factor), int(height * scale_factor)
    if min(height, width, new_height, new_width) <= 0:
        return None

    tables = linear_tables(
        (height, width),
        (new_height, new_width),
        scale_xy=(1.0 / (new_width / width), 1.0 / (new_height / height)),
    )
    crop = _Crop((x1, y1, x2, y2), (new_height, new_width), tables)

    return crop

def _crop_corners(
    anchor: Tuple[int, int], *, width: int, height: int, position: str
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    # sv.CropAnnotator.calculate_crop_coordinates
    x, y = anchor
    if position == "TOP_LEFT":
        return (x - width, y - height), (x, y)
    if position == "TOP_CENTER":
        return (x - width // 2, y - height), (x + width // 2, y)
    if position == "TOP_RIGHT":
        return (x, y - height), (x + width, y)
    if position == "CENTER_LEFT":
        return (x - width, y - height // 2), (x, y + height // 2)
    if position in ("CENTER", CENTER_OF_MASS):
        return (x - width // 2, y - height // 2), (x + width // 2, y + height // 2)
    if position == "CENTER_RIGHT":
        return (x, y - height // 2), (x + width, y + height // 2)
    if position == "BOTTOM_LEFT":
        return (x - width, y), (x, y + height)
    if position == "BOTTOM_CENTER":
        return (x - width // 2, y), (x + width // 2, y + height)

    return (x, y), (x + width, y + height)

def _overlay(canvas: torch.Tensor, overlay: torch.Tensor, *, x: int, y: int) -> None:
    # sv overlay_image: paste with clipping at the frame edges.
    frame_height, frame_width = int(canvas.shape[1]), int(canvas.shape[2])
    height, width = int(overlay.shape[1]), int(overlay.shape[2])
    if x + width <= 0 or x >= frame_width or y + height <= 0 or y >= frame_height:
        return

    left, top = max(x, 0), max(y, 0)
    right, bottom = min(frame_width, x + width), min(frame_height, y + height)
    canvas[:, top:bottom, left:right] = overlay[
        :, top - y : bottom - y, left - x : right - x
    ]

_CHECKS = {
    "position": lambda value: checked_choice(
        value, name="position", choices=POSITIONS_WITH_MASS
    ),
    "scale_factor": lambda value: checked_number(
        value, name="scale_factor", minimum=0.0, above_minimum=True
    ),
    "border_thickness": lambda value: checked_int(
        value, name="border_thickness", minimum=1
    ),
    "mask_quality": checked_mask_quality,
}
