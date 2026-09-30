"""Polygon outlines from compact crops, without per-instance full-frame masks."""

import numpy as np
import supervision as sv
from supervision.annotators.utils import resolve_color
from supervision.detection.compact_mask import CompactMask

from inference.core.workflows.core_steps.visualizations.common.annotators.polygon import (
    ensure_cv2_image_for_annotation,
)


class CompactPolygonAnnotator(sv.PolygonAnnotator):
    @ensure_cv2_image_for_annotation
    def annotate(self, scene, detections, custom_color_lookup=None):
        if not isinstance(detections.mask, CompactMask):
            return super().annotate(scene, detections, custom_color_lookup)
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
