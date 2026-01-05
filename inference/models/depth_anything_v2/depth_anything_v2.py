import os
import time
import warnings
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    DepthProForDepthEstimation,
    DepthProImageProcessorFast,
)

# Convert numpy array to WorkflowImageData
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from inference.models.transformers import TransformerModel


class DepthAnythingV2(TransformerModel):
    transformers_class = AutoModelForDepthEstimation
    processor_class = AutoImageProcessor
    load_base_from_roboflow = True
    needs_hf_token = False
    version_id = None
    default_dtype = torch.bfloat16
    load_weights_as_transformers = True
    endpoint = "depth-anything-v2/small"
    task_type = "depth-estimation"

    def __init__(self, *args, **kwargs):

        try:
            super().__init__(*args, **kwargs)
        except Exception as e:
            print(f"Error initializing depth estimation model: {str(e)}")
            raise

        # Set appropriate dtype based on device
        if self.model.device.type == "mps":
            self.model = self.model.to(torch.float32)  # MPS prefers float32
        elif self.model.device.type == "cpu":
            warnings.warn(
                "Running DepthPro on CPU. This may be very slow. Consider using GPU or MPS if available."
            )

    def predict(self, image_in: Image.Image, prompt="", history=None, **kwargs):
        try:
            # Process input image
            inputs = self.processor(images=image_in, return_tensors="pt")

            # Move inputs to device
            device = self.model.device
            if device.type == "mps":
                inputs = {
                    k: v.to(torch.float32).to(device) if torch.is_tensor(v) else v
                    for k, v in inputs.items()
                }
            else:
                inputs = {
                    k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in inputs.items()
                }

            # Run model inference
            with torch.inference_mode():
                outputs = self.model(**inputs)

                # Post-process depth estimation
                post_processed_outputs = self.processor.post_process_depth_estimation(
                    outputs, target_sizes=[(image_in.height, image_in.width)]
                )

                # Extract depth map
                depth_map = post_processed_outputs[0]["predicted_depth"]
                depth_map = depth_map.to(torch.float32).cpu().numpy()

                # Normalize depth values
                depth_min = depth_map.min()
                depth_max = depth_map.max()
                if depth_max == depth_min:
                    raise ValueError("Depth map has no variation (min equals max)")
                normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)

                # Create visualization
                depth_for_viz = (normalized_depth * 255.0).astype(np.uint8)
                cmap = plt.get_cmap("viridis")
                colored_depth = (cmap(depth_for_viz)[:, :, :3] * 255).astype(np.uint8)

                # Convert numpy array to WorkflowImageData
                parent_metadata = ImageParentMetadata(parent_id=f"{uuid4()}")
                colored_depth_image = WorkflowImageData(
                    numpy_image=colored_depth, parent_metadata=parent_metadata
                )

                # Create result dictionary
                result = {
                    "image": colored_depth_image,
                    "normalized_depth": normalized_depth,
                }

                return (result,)
        except Exception as e:
            raise
