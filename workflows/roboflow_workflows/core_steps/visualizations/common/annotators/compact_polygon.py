"""Polygon outlines from compact crops, without per-instance full-frame masks."""

import numpy as np
import supervision as sv
from roboflow_workflows.core_steps.visualizations.common.annotators.polygon import (
    ensure_cv2_image_for_annotation,
)
from supervision.annotators.utils import resolve_color
from supervision.detection.compact_mask import CompactMask


class CompactPolygonAnnotator(sv.PolygonAnnotator):
    """Draw Supervision polygon outlines from detection crops."""

    @ensure_cv2_image_for_annotation
    def annotate(self, scene, detections, custom_color_lookup=None):
        """Draw outlines while preserving the original annotation semantics.

        Args:
            scene (ImageType): NumPy or PIL image to annotate.
            detections (sv.Detections): Detections carrying dense or compact masks.
            custom_color_lookup (Optional[np.ndarray]): Per-detection color indices.

        Returns:
            ImageType: Annotated image with the same type as the input.
        """
        if not isinstance(detections.mask, CompactMask):
            annotated = super().annotate(scene, detections, custom_color_lookup)
            return annotated

        for index in range(len(detections)):
            crop = detections.mask.crop(index)
            offset = detections.mask.offsets[index]
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=index,
                color_lookup=(
                    self.color_lookup
                    if custom_color_lookup is None
                    else custom_color_lookup
                ),
            )
            for polygon in sv.mask_to_polygons(crop):
                scene = sv.draw_polygon(
                    scene=scene,
                    polygon=(polygon + offset).astype(np.int32),
                    color=color,
                    thickness=self.thickness,
                )
        return scene
