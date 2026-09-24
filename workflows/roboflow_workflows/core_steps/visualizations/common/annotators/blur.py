import cv2
import numpy as np
import supervision as sv
from supervision.annotators.utils import calculate_dynamic_kernel_size


class MaskAwareBlurAnnotator(sv.BlurAnnotator):
    """Blur annotator that follows segmentation masks when detections carry them.

    `sv.BlurAnnotator` blurs each detection's bounding box and never reads
    `Detections.mask`, so instance segmentation predictions were blurred as
    rectangles. With masks present, this annotator still blurs each bounding box
    region but copies the blurred pixels back only inside that detection's mask.
    Without masks it defers to `sv.BlurAnnotator` unchanged.
    """

    def annotate(self, scene: np.ndarray, detections: sv.Detections) -> np.ndarray:
        if detections.mask is None:
            return super().annotate(scene=scene, detections=detections)
        image_height, image_width = scene.shape[:2]
        clipped_xyxy = sv.clip_boxes(
            xyxy=detections.xyxy, resolution_wh=(image_width, image_height)
        ).astype(int)
        for (x1, y1, x2, y2), mask in zip(clipped_xyxy, detections.mask):
            if x2 <= x1 or y2 <= y1:
                continue
            inside = mask[y1:y2, x1:x2]
            if not inside.any():
                continue
            kernel_size = (
                self.kernel_size
                if self.kernel_size is not None
                else calculate_dynamic_kernel_size(x1, y1, x2, y2)
            )
            roi = scene[y1:y2, x1:x2]
            blurred = cv2.blur(roi, (kernel_size, kernel_size))
            roi[inside] = blurred[inside]
        return scene
