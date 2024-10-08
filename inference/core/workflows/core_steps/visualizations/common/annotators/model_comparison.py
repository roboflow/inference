import cv2
import numpy as np
from supervision import Color, Detections
from supervision.annotators.base import BaseAnnotator


class ModelComparisonAnnotator(BaseAnnotator):
    """
    A class for drawing background colors outside of detected box or mask regions.
    !!! warning
        This annotator uses `sv.Detections.mask`.
    """

    def __init__(
        self,
        color_a: Color = Color.GREEN,
        color_b: Color = Color.RED,
        background_color: Color = Color.BLACK,
        opacity: float = 0.7,
        force_box: bool = False,
    ):
        """
        Args:
            color_a (Color): Color of the predictions made only by Model A.
            color_b (Color): Color of the predictions made only by Model B.
            background_color (Color): Color of parts of the image not covered by any prediction.
            opacity (float): Opacity of the overlay mask. Must be between `0` and `1`.
        """
        self.color_a: Color = color_a
        self.color_b: Color = color_b
        self.background_color: Color = background_color
        self.opacity = opacity
        self.force_box = force_box

    def annotate(self, scene: np.ndarray, detections_a: Detections, detections_b: Detections) -> np.ndarray:
        """
        Annotates the given scene with masks based on the provided detections.
        
        Parameters:
        - scene: Original image as a NumPy array (H x W x C).
        - detections_a: Detections from Model A.
        - detections_b: Detections from Model B.
        
        Returns:
        - Annotated image as a NumPy array.
        """
        
        # Initialize single-channel masks
        neither_predicted = np.ones(scene.shape[:2], dtype=np.uint8)  # 1 where neither model predicts
        a_predicted = np.zeros(scene.shape[:2], dtype=np.uint8)
        b_predicted = np.zeros(scene.shape[:2], dtype=np.uint8)
        
        # Populate masks based on detections from Model A
        if detections_a.mask is None or self.force_box:
            for detection_idx in range(len(detections_a)):
                x1, y1, x2, y2 = detections_a.xyxy[detection_idx].astype(int)
                a_predicted[y1:y2, x1:x2] = 1
                neither_predicted[y1:y2, x1:x2] = 0
        else:
            for mask in detections_a.mask:
                # Assuming mask is a binary mask with 1s where predicted
                a_predicted[mask.astype(bool)] = 1
                neither_predicted[mask.astype(bool)] = 0
        
        # Populate masks based on detections from Model B
        if detections_b.mask is None or self.force_box:
            for detection_idx in range(len(detections_b)):
                x1, y1, x2, y2 = detections_b.xyxy[detection_idx].astype(int)
                b_predicted[y1:y2, x1:x2] = 1
                neither_predicted[y1:y2, x1:x2] = 0
        
        # Define combined masks
        only_a_predicted = a_predicted & (a_predicted ^ b_predicted)
        only_b_predicted = b_predicted & (b_predicted ^ a_predicted)
        
        # Prepare overlay colors
        background_color_bgr = self.background_color.as_bgr()  # Tuple like (B, G, R)
        color_a_bgr = self.color_a.as_bgr()
        color_b_bgr = self.color_b.as_bgr()
        
        # Create full-color overlay images
        overlay_background = np.full_like(scene, background_color_bgr, dtype=np.uint8)
        overlay_a = np.full_like(scene, color_a_bgr, dtype=np.uint8)
        overlay_b = np.full_like(scene, color_b_bgr, dtype=np.uint8)
        
        # Function to blend and apply overlay based on mask
        def apply_overlay(base_img, overlay_img, mask, opacity):
            """
            Blends the overlay with the base image where the mask is set.
            
            Parameters:
            - base_img: Original image.
            - overlay_img: Overlay color image.
            - mask: Single-channel mask where to apply the overlay.
            - opacity: Opacity of the overlay (0 to 1).
            
            Returns:
            - Image with overlay applied.
            """
            # Blend the entire images
            blended = cv2.addWeighted(base_img, 1 - opacity, overlay_img, opacity, 0)
            # Expand mask to three channels
            mask_3ch = np.stack([mask]*3, axis=-1)  # Shape: H x W x 3
            # Ensure mask is boolean
            mask_bool = mask_3ch.astype(bool)
            # Apply blended regions where mask is True
            base_img[mask_bool] = blended[mask_bool]
            return base_img
        
        # Apply background overlay where neither model predicted
        scene = apply_overlay(scene, overlay_background, neither_predicted, self.opacity)
        
        # Apply overlay for only Model A predictions
        scene = apply_overlay(scene, overlay_a, only_a_predicted, self.opacity)
        
        # Apply overlay for only Model B predictions
        scene = apply_overlay(scene, overlay_b, only_b_predicted, self.opacity)
        
        # Areas where both models predicted remain unchanged (no overlay)
        
        return scene
