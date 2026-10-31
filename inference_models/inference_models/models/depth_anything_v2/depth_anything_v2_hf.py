from threading import Lock
from typing import List, Tuple, Union

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.entities import ImageDimensions
from inference_models.models.base.depth_estimation import DepthEstimationModel
from inference_models.models.common.roboflow.pre_processing import (
    extract_input_images_dimensions,
)


class DepthAnythingV2HF(
    DepthEstimationModel[torch.Tensor, List[ImageDimensions], torch.Tensor]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        local_files_only: bool = True,
        **kwargs,
    ) -> "DepthAnythingV2HF":
        model = AutoModelForDepthEstimation.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        ).to(device)
        processor = AutoImageProcessor.from_pretrained(
            model_name_or_path, local_files_only=local_files_only, use_fast=True
        )
        return cls(model=model, processor=processor, device=device)

    def __init__(
        self,
        model: AutoModelForDepthEstimation,
        processor: AutoImageProcessor,
        device: torch.device,
    ):
        self._model = model
        self._processor = processor
        self._device = device
        self._lock = Lock()

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> Tuple[torch.Tensor, List[ImageDimensions]]:
        image_dimensions = extract_input_images_dimensions(images=images)
        inputs = self._processor(images=images, return_tensors="pt")
        return inputs["pixel_values"].to(self._device), image_dimensions

    def forward(
        self,
        pre_processed_images: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        with self._lock, torch.inference_mode():
            return self._model(pre_processed_images)

    def post_process(
        self,
        model_results: torch.Tensor,
        pre_processing_meta: List[ImageDimensions],
        **kwargs,
    ) -> List[torch.Tensor]:
        target_sizes = [(dim.height, dim.width) for dim in pre_processing_meta]
        post_processed_outputs = self._processor.post_process_depth_estimation(
            model_results,
            target_sizes=target_sizes,
        )
        return [
            output["predicted_depth"].to(self._device)
            for output in post_processed_outputs
        ]
