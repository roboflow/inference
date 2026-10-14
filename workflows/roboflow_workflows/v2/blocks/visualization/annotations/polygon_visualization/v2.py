"""Mask outlines: ``roboflow_core/polygon_visualization@v2``.

Port of V1 ``roboflow_core/polygon_visualization@v2``
(``sv.PolygonAnnotator``). Pixels equal V1 for the same dense masks::

    per mask   cv2.findContours(RETR_TREE, CHAIN_APPROX_SIMPLE) of the whole
               mask; contours with >= 3 points; closed cv2.polylines

Only the window around each mask's pixels (plus one empty pixel, so OpenCV
sees the same borders) goes to the host, in one transfer; contour tracing is
a CPU algorithm here. Outline pixels are written on the device in one
indexed write (``strokes``). V1 drew nothing for predictions without masks;
V2 accepts only instance masks, so that mistake is a compile error.
"""

from typing import Any, Dict, List, Literal, Optional

import cv2
import numpy as np
import torch
from pydantic import Field, StrictInt, model_validator
from roboflow_workflows.v2.blocks.visualization.annotations.base import (
    annotated,
    canvas_of,
)
from roboflow_workflows.v2.blocks.visualization.annotations.colors import (
    COLOR_AXES,
    PALETTE_NAMES,
)
from roboflow_workflows.v2.blocks.visualization.annotations.params import (
    check_literals,
    check_palette_literals,
    checked_int,
    reject_copy_image,
    row_colors,
)
from roboflow_workflows.v2.blocks.visualization.annotations.rows import (
    MASK_KINDS,
    MASK_QUALITIES,
    dense_masks_of,
    mask_extents,
    placed_in,
    read_rows,
)
from roboflow_workflows.v2.blocks.visualization.annotations.strokes import (
    Strokes,
    bounds_scan,
    stroke_reach,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

MIN_POLYGON_POINTS = 3


class PolygonVisualizationBlock(Block):
    """Outline each instance mask on a copy of the image.

    Example: a person mask with a hole gets two closed outlines, the outer
    contour and the hole's, both in the person's color.
    """

    type = "roboflow_core/polygon_visualization@v2"
    prunable = True
    metadata = {
        "name": "Polygon Visualization",
        "port_of": "roboflow_core/polygon_visualization@v2",
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
        predictions: Ref(*MASK_KINDS) = Field(
            description=(
                "Instance segmentation: dense or RLE InstanceDetections, or a "
                "mask view (SegmentationView). Outlines are traced on one "
                "full-image mask per detection (see mask_quality)."
            ),
            examples=["$steps.instance_segmentation.segmentation"],
        )
        thickness: StrictInt | Ref(INTEGER_KIND) = Field(
            default=2,
            description="Outline width in pixels; >= 1.",
            examples=[2, "$inputs.thickness"],
        )
        mask_quality: Literal[MASK_QUALITIES] | Ref(STRING_KIND) = Field(
            default="accurate",
            description=(
                "How instance masks are drawn; used when the predictions are instance "
                "masks (always for this block). Each read builds one full-image mask "
                "per detection, at either quality. accurate: the masks the model "
                "defines (a low-resolution mask view is upscaled with the model's own "
                "interpolation). coarse: a mask view's grid cells are thresholded, "
                "then expanded, so edges can be blocky and small parts can be lost; it "
                "skips the accurate upscaling only. "
                "The model block's fast implementation retains compact scores when "
                "supported, so accurate and coarse can differ. Its accurate "
                "implementation and dense-only models already supply binary masks; "
                "both settings then use those same masks."
            ),
            examples=["accurate", "coarse", "$inputs.mask_quality"],
        )

        @model_validator(mode="before")
        @classmethod
        def _reject_copy_image(cls, values: Any) -> Any:
            checked = reject_copy_image(values, title=cls.__name__)

            return checked

        @model_validator(mode="after")
        def _check_palette_literals(self) -> "PolygonVisualizationBlock.Params":
            check_palette_literals(
                palette_size=self.palette_size, custom_colors=self.custom_colors
            )

            return self

        @model_validator(mode="after")
        def _check_literals(self) -> "PolygonVisualizationBlock.Params":
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
        thickness: int,
        mask_quality: str,
    ) -> Dict[str, Any]:
        """Outline every mask of ``predictions`` on a copy of ``image``.

        Args:
            image: RGB image; not modified.
            predictions: Instance masks (dense, RLE or view).
            color_palette: Palette name.
            palette_size: Matplotlib sample count.
            custom_colors: ``CUSTOM`` palette colors.
            color_axis: ``CLASS``, ``INDEX`` or ``TRACK``.
            thickness: Outline width.
            mask_quality: ``accurate`` or ``coarse`` masks.

        Returns:
            ``image``: the new annotated image.

        Raises:
            ValueError: On invalid parameters or predictions of another
                workflow input.
        """
        values = dict(thickness=thickness)
        for name, check in _CHECKS.items():
            check(values[name])
        canvas = canvas_of(image, block=self.type)
        dense = dense_masks_of(predictions, quality=mask_quality, block=self.type)
        dense = placed_in(dense, image, block=self.type)
        rows = read_rows(dense)
        colors = row_colors(
            rows.class_id,
            rows.tracker_ids(),
            color_palette=color_palette,
            palette_size=palette_size,
            custom_colors=custom_colors,
            color_axis=color_axis,
        )

        strokes = Strokes(image.size_hw)
        reach = stroke_reach(thickness)
        for polygons, color in zip(mask_polygons(dense.mask), colors):
            if not polygons:
                continue
            strokes.add(
                lambda mask, polygons=polygons: cv2.polylines(
                    mask, polygons, True, 1, thickness
                ),
                color=color,
                scan=bounds_scan(np.concatenate(polygons), reach=reach),
            )
        strokes.paint(canvas)
        result = annotated(image, canvas)

        return result


def mask_polygons(masks: torch.Tensor) -> List[List[np.ndarray]]:
    """Trace the contours of each mask as ``sv.mask_to_polygons`` does.

    Args:
        masks: ``(N, H, W)`` bool masks on any device.

    Returns:
        Per mask, its contours as ``(K, 2)`` int32 frame points, ``K >= 3``.
    """
    height, width = int(masks.shape[1]), int(masks.shape[2])
    windows: List[Optional[List[int]]] = []
    for extent in mask_extents(masks):
        if extent is None:
            windows.append(None)
            continue
        left, top, right, bottom = extent
        windows.append(
            [
                max(left - 1, 0),
                max(top - 1, 0),
                min(right + 1, width),
                min(bottom + 1, height),
            ]
        )

    crops = [
        masks[index, y0:y1, x0:x1].reshape(-1)
        for index, window in enumerate(windows)
        if window is not None
        for x0, y0, x1, y1 in [window]
    ]
    host = (
        torch.cat(crops).to(torch.uint8).cpu().numpy()
        if crops
        else np.zeros(0, np.uint8)
    )

    polygons: List[List[np.ndarray]] = []
    offset = 0
    for window in windows:
        if window is None:
            polygons.append([])
            continue
        x0, y0, x1, y1 = window
        size = (y1 - y0) * (x1 - x0)
        crop = host[offset : offset + size].reshape(y1 - y0, x1 - x0)
        offset += size
        contours, _ = cv2.findContours(crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        polygons.append(
            [
                (np.squeeze(contour, axis=1) + np.array([[x0, y0]])).astype(np.int32)
                for contour in contours
                if contour.shape[0] >= MIN_POLYGON_POINTS
            ]
        )

    return polygons


_CHECKS = {
    "thickness": lambda value: checked_int(value, name="thickness", minimum=1),
}
