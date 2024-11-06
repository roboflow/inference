from typing import Optional, Union

import cv2
import numpy as np
from supervision import Color, Detections
from supervision.annotators.base import BaseAnnotator, ImageType
from supervision.annotators.utils import ColorLookup, resolve_color
from supervision.draw.color import ColorPalette
from supervision.utils.conversion import ensure_cv2_image_for_annotation


class HaloAnnotator(BaseAnnotator):
    """
    A class for drawing Halos on an image using provided detections.

    !!! warning

        This annotator uses `sv.Detections.mask`.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        opacity: float = 0.8,
        kernel_size: int = 40,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            opacity (float): Opacity of the overlay mask. Must be between `0` and `1`.
            kernel_size (int): The size of the average pooling kernel used for creating
                the halo.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.opacity = opacity
        self.color_lookup: ColorLookup = color_lookup
        self.kernel_size: int = kernel_size

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> ImageType:
        """
        Annotates the given scene with halos based on the provided detections.

        Args:
            scene (ImageType): The image where masks will be drawn.
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

            halo_annotator = sv.HaloAnnotator()
            annotated_frame = halo_annotator.annotate(
                scene=image.copy(),
                detections=detections
            )
            ```

        ![halo-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/halo-annotator-example-purple.png)
        """
        assert isinstance(scene, np.ndarray)
        colored_mask = np.zeros_like(scene, dtype=np.uint8)
        fmask = np.array([False] * scene.shape[0] * scene.shape[1]).reshape(
            scene.shape[0], scene.shape[1]
        )

        for detection_idx in np.flip(np.argsort(detections.area)):
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
                mask = np.zeros(scene.shape[:2], dtype=bool)
                mask[y1:y2, x1:x2] = True
            else:
                mask = detections.mask[detection_idx]
            fmask = np.logical_or(fmask, mask)
            color_bgr = color.as_bgr()
            colored_mask[mask] = color_bgr

        colored_mask = cv2.blur(colored_mask, (self.kernel_size, self.kernel_size))
        colored_mask[fmask] = [0, 0, 0]
        gray = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2GRAY)
        alpha = self.opacity * gray / gray.max()
        alpha_mask = alpha[:, :, np.newaxis]
        blended_scene = np.uint8(scene * (1 - alpha_mask) + colored_mask * self.opacity)
        np.copyto(scene, blended_scene)
        return scene
