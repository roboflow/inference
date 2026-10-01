from typing import Optional, TypeVar, Union

import cv2
import numpy as np
from PIL import Image
from supervision import (
    Color,
    ColorLookup,
    ColorPalette,
    CompactMask,
    Detections,
    draw_polygon,
    mask_to_polygons,
)
from supervision.annotators.base import BaseAnnotator
from supervision.annotators.utils import resolve_color

try:
    from supervision.utils.conversion import ensure_cv2_image_for_annotation
except ImportError:
    from supervision.utils.conversion import (
        ensure_cv2_image_for_class_method as ensure_cv2_image_for_annotation,
    )


ImageType = TypeVar("ImageType", np.ndarray, Image.Image)


class PolygonAnnotator(BaseAnnotator):
    """
    A class for drawing polygons on an image using provided detections.

    !!! warning

        This annotator uses `sv.Detections.mask`.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        thickness: int = 2,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            thickness (int): Thickness of the polygon lines.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.color_lookup: ColorLookup = color_lookup

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> ImageType:
        """
        Annotates the given scene with polygons based on the provided detections.

        Args:
            scene (ImageType): The image where polygons will be drawn.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import supervision as sv

            image = ...
            detections = sv.Detections(...)

            polygon_annotator = sv.PolygonAnnotator()
            annotated_frame = polygon_annotator.annotate(
                scene=image.copy(),
                detections=detections
            )
            ```

        ![polygon-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/polygon-annotator-example-purple.png)
        """
        assert isinstance(scene, np.ndarray)

        for detection_idx in range(len(detections)):
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=(
                    self.color_lookup
                    if custom_color_lookup is None
                    else custom_color_lookup
                ),
            )

            if detections.mask is None:
                x1, y1, x2, y2 = detections.xyxy[detection_idx].astype(int)
                cv2.rectangle(
                    img=scene,
                    pt1=(x1, y1),
                    pt2=(x2, y2),
                    color=color.as_bgr(),
                    thickness=self.thickness,
                )
            else:
                x1, y1, x2, y2 = detections.xyxy[detection_idx].astype(int)
                if isinstance(detections.mask, CompactMask):
                    # Preserve NumPy's bounding-box slicing semantics without
                    # materialising a full-frame mask for every instance.
                    _, height, width = detections.mask.shape
                    sx1, sx2, _ = slice(x1, x2).indices(width)
                    sy1, sy2, _ = slice(y1, y2).indices(height)
                    ox, oy = detections.mask.offsets[detection_idx]
                    crop = detections.mask.crop(detection_idx)
                    left, top = max(sx1, ox), max(sy1, oy)
                    right = min(sx2, ox + crop.shape[1])
                    bottom = min(sy2, oy + crop.shape[0])
                    if right <= left or bottom <= top:
                        continue

                    mask_crop = crop[top - oy : bottom - oy, left - ox : right - ox]
                    x1, y1 = x1 + left - sx1, y1 + top - sy1
                else:
                    mask = detections.mask[detection_idx]
                    mask_crop = mask[y1:y2, x1:x2]

                polygons = list(mask_to_polygons(mask=mask_crop))

                for polygon in polygons:
                    # Offset polygon points back to full-frame coordinates
                    full_polygon = (polygon + np.array([[x1, y1]])).astype(np.int32)
                    scene = draw_polygon(
                        scene=scene,
                        polygon=full_polygon,
                        color=color,
                        thickness=self.thickness,
                    )

        return scene
