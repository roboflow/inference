"""
This file contains the implementation of the Sam3Session class, which provides a stateful,
high-level interface for interactive segmentation using the stateless Sam3ImageModel.
"""
from typing import Union, Dict, Optional, List, Tuple
import numpy as np
import torch

from .sam3_image_model import Sam3ImageModel
from sam3.model.geometry_encoders import Prompt
from sam3.model.box_ops import box_xyxy_to_xywh, box_xywh_to_cxcywh


class Sam3Session:
    """
    Manages the state for an interactive segmentation session, providing a simple
    API to set an image, add prompts, and get predictions.
    """
    def __init__(self, model: Sam3ImageModel):
        self.model = model
        self._reset_session_state()

    def _reset_session_state(self):
        """Resets all state related to a specific session (image and prompts)."""
        self.image_features: Optional[Dict[str, torch.Tensor]] = None
        self.original_size: Optional[Tuple[int, int]] = None
        self.has_predicted: bool = False
        self.text_features: Optional[Dict[str, torch.Tensor]] = None
        self.prompts: Dict[str, Union[List, torch.Tensor]] = {
            "points": [], 
            "point_labels": [], 
            "boxes_cxcywh": torch.empty((0, 4))
        }

    @property
    def device(self) -> torch.device:
        return self.model.device

    def set_image(self, image: Union[np.ndarray, torch.Tensor]):
        """
        Sets a new image for the session, preprocessing and encoding it.
        This resets all existing prompts.
        """
        self._reset_session_state()
        processed_image, original_size = self.model.preprocess_image(image)
        self.original_size = original_size
        self.image_features = self.model.encode_image(processed_image)

    def set_text_prompt(self, text: Optional[str]):
        """Sets or clears the text prompt for the session."""
        if text:
            self.text_features = self.model.encode_text(text)
        else:
            self.text_features = None

    def add_point_prompt(self, points: List[List[float]], labels: List[int]):
        """Adds point prompts. Replaces existing point prompts."""
        self.prompts["points"] = points
        self.prompts["point_labels"] = labels

    def add_box_prompt(self, boxes: List[List[float]]):
        """
        Adds box prompts. Replaces existing box prompts.
        Args:
            boxes: A list of boxes, each in [xmin, ymin, xmax, ymax] 
                        format in pixel coordinates.
        """
        if self.original_size is None:
            raise RuntimeError("An image must be set before adding prompts. Call set_image first.")
        
        h, w = self.original_size
        
        boxes_tensor_xyxy = torch.tensor(boxes, dtype=torch.float32, device=self.device)
        # Normalize
        boxes_tensor_xyxy[:, [0, 2]] /= w
        boxes_tensor_xyxy[:, [1, 3]] /= h
        
        # Convert to normalized cxcywh
        boxes_tensor_cxcywh = box_xywh_to_cxcywh(box_xyxy_to_xywh(boxes_tensor_xyxy))
        self.prompts["boxes_cxcywh"] = boxes_tensor_cxcywh


    def reset_prompts(self):
        """Clears all prompts without clearing the image."""
        self.text_features = None
        self.prompts = {
            "points": [], 
            "point_labels": [], 
            "boxes_cxcywh": torch.empty((0, 4))
        }

    def predict(
        self,
        output_prob_thresh: float = 0.5,
        multimask_output: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Runs a prediction based on the current image and prompts.
        """
        if self.image_features is None or self.original_size is None:
            raise RuntimeError("An image must be set before running prediction. Call `set_image` first.")

        # Handle visual prompt logic
        is_visual_prompt = not self.has_predicted and self.prompts["boxes_cxcywh"].numel() > 0
        visual_prompt = None
        
        boxes_for_geo_prompt = self.prompts["boxes_cxcywh"]
        if is_visual_prompt:
            visual_prompt_box = boxes_for_geo_prompt[0:1]
            visual_prompt = Prompt(
                box_embeddings=visual_prompt_box.unsqueeze(0),
                box_labels=torch.ones(1, 1, device=self.device, dtype=torch.long),
                point_embeddings=torch.zeros((0, 1, 2), device=self.device),
                point_labels=torch.zeros((0, 1), device=self.device, dtype=torch.long),
            )
            boxes_for_geo_prompt = boxes_for_geo_prompt[1:]
        
        # Prepare geometric prompts
        points_tensor = torch.tensor(self.prompts["points"], device=self.device, dtype=torch.float32)
        point_labels_tensor = torch.tensor(self.prompts["point_labels"], device=self.device, dtype=torch.long)
        box_labels_tensor = torch.ones(boxes_for_geo_prompt.shape[0], device=self.device, dtype=torch.long)
        
        geometric_prompt = Prompt(
            point_embeddings=points_tensor.unsqueeze(1) if points_tensor.numel() > 0 else torch.zeros((0, 1, 2), device=self.device),
            point_labels=point_labels_tensor.unsqueeze(1) if point_labels_tensor.numel() > 0 else torch.zeros((0, 1), device=self.device, dtype=torch.long),
            box_embeddings=boxes_for_geo_prompt.unsqueeze(1) if boxes_for_geo_prompt.numel() > 0 else torch.zeros((0, 1, 4), device=self.device),
            box_labels=box_labels_tensor.unsqueeze(1) if box_labels_tensor.numel() > 0 else torch.zeros((0, 1), device=self.device, dtype=torch.long),
        )

        model_outputs = self.model.predict(
            image_features=self.image_features,
            text_features=self.text_features,
            geometric_prompt=geometric_prompt,
            visual_prompt=visual_prompt,
            multimask_output=multimask_output,
        )

        self.has_predicted = True

        return self.model.postprocess_outputs(
            model_outputs=model_outputs,
            original_size=self.original_size,
            output_prob_thresh=output_prob_thresh,
            multimask_output=multimask_output,
        )
