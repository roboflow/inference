import cv2
import numpy as np
from supervision import Color, Detections
from supervision.annotators.base import BaseAnnotator


class BackgroundColorAnnotator(BaseAnnotator):
    """
    A class for drawing background colors outside of detected box or mask regions.
    !!! warning
        This annotator uses `sv.Detections.mask`.
    """

    def __init__(
        self,
        color: Color = Color.BLACK,
        opacity: float = 0.5,
        force_box: bool = False,
    ):
        """
        Args:
            color (Color): The color to use for annotating detections.
            opacity (float): Opacity of the overlay mask. Must be between `0` and `1`.
        """
        self.color: Color = color
        self.opacity = opacity
        self.force_box = force_box

    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        """
        Annotates the given scene with masks based on the provided detections.
        Args:
            scene (ImageType): The image where masks will be drawn.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.
        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)
        Example:
            ```python
            import supervision as sv
            image = ...
            detections = sv.Detections(...)
            background_color_annotator = sv.BackgroundColorAnnotator()
            annotated_frame = background_color_annotator.annotate(
                scene=image.copy(),
                detections=detections
            )
            ```
        ![background-color-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/background-color-annotator-example-purple.png)
        """

        colored_mask = np.full_like(scene, self.color.as_bgr(), dtype=np.uint8)

        cv2.addWeighted(
            scene, 1 - self.opacity, colored_mask, self.opacity, 0, dst=colored_mask
        )

        if detections.mask is None or self.force_box:
            for detection_idx in range(len(detections)):
                x1, y1, x2, y2 = detections.xyxy[detection_idx].astype(int)
                colored_mask[y1:y2, x1:x2] = scene[y1:y2, x1:x2]
        else:
            for mask in detections.mask:
                colored_mask[mask] = scene[mask]

        return colored_mask
