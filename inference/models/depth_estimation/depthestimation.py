import torch
from PIL import Image
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation, AutoImageProcessor, AutoModelForDepthEstimation
# Convert numpy array to WorkflowImageData
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData, ImageParentMetadata
import psutil
import time
import warnings
import os
import numpy as np
import matplotlib.pyplot as plt
from uuid import uuid4

from inference.models.transformers import TransformerModel


class DepthEstimator(TransformerModel):
    transformers_class = AutoModelForDepthEstimation
    processor_class = AutoImageProcessor
    load_base_from_roboflow = False
    needs_hf_token = True
    version_id = None
    default_dtype = torch.bfloat16
    load_weights_as_transformers = True
    endpoint = "depth-anything/Depth-Anything-V2-Small-hf"
    task_type = "depth-estimation"
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(self, *args, **kwargs):
        # Handle model_id mapping
        input_model_id = kwargs.get('model_id', self.model_id)
        if input_model_id != self.model_id:
            kwargs["model_id"] = self.model_id
        
        # Check for Hugging Face token
        hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        hf_token_alt = os.environ.get("HUGGINGFACE_TOKEN")
        
        if not hf_token and not hf_token_alt:
            raise RuntimeError(
                "Hugging Face credentials required. Please set up your Hugging Face credentials by:\n"
                "1. Creating a Hugging Face account at https://huggingface.co/\n"
                "2. Getting your access token from https://huggingface.co/settings/tokens\n"
                "3. Setting the token in your environment: export HUGGING_FACE_HUB_TOKEN=your_token_here\n"
                "Or by logging in with: huggingface-cli login"
            )
        
        # Set the token in kwargs if it's not already there
        if "huggingface_token" not in kwargs:
            kwargs["huggingface_token"] = hf_token or hf_token_alt
        
        try:
            super().__init__(*args, **kwargs)
        except Exception as e:
            if "401" in str(e) or "Unauthorized" in str(e):
                raise RuntimeError(
                    "Hugging Face credentials required. Please set up your Hugging Face credentials by:\n"
                    "1. Creating a Hugging Face account at https://huggingface.co/\n"
                    "2. Getting your access token from https://huggingface.co/settings/tokens\n"
                    "3. Setting the token in your environment: export HUGGING_FACE_HUB_TOKEN=your_token_here\n"
                    "Or by logging in with: huggingface-cli login"
                ) from e
            raise
        
        # Set appropriate dtype based on device
        if self.model.device.type == 'mps':
            self.model = self.model.to(torch.float32)  # MPS prefers float32
        elif self.model.device.type == 'cpu':
            warnings.warn("Running DepthPro on CPU. This may be very slow. Consider using GPU or MPS if available.")

    def predict(self, image_in: Image.Image, prompt="", history=None, **kwargs):
        try:
            # Process input image
            inputs = self.processor(images=image_in, return_tensors="pt")
            
            # Move inputs to device
            device = self.model.device
            if device.type == 'mps':
                inputs = {k: v.to(torch.float32).to(device) if torch.is_tensor(v) else v 
                        for k, v in inputs.items()}
            else:
                inputs = {k: v.to(device) if torch.is_tensor(v) else v 
                        for k, v in inputs.items()}
            
            # Run model inference
            with torch.inference_mode():
                outputs = self.model(**inputs)
                
                # Post-process depth estimation
                post_processed_outputs = self.processor.post_process_depth_estimation(
                    outputs, target_sizes=[(image_in.height, image_in.width)]
                )
                
                # Extract depth map
                depth_map = post_processed_outputs[0]['predicted_depth']
                depth_map = depth_map.cpu().numpy()
                
                # Normalize depth values
                depth_min = depth_map.min()
                depth_max = depth_map.max()
                if depth_max == depth_min:
                    raise ValueError("Depth map has no variation (min equals max)")
                normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)
                
                # Create visualization
                depth_for_viz = (normalized_depth * 255.0).astype(np.uint8)
                cmap = plt.get_cmap('viridis')
                colored_depth = (cmap(depth_for_viz)[:, :, :3] * 255).astype(np.uint8)
                
                # Convert numpy array to WorkflowImageData
                parent_metadata = ImageParentMetadata(parent_id=f"{uuid4()}")
                colored_depth_image = WorkflowImageData(
                    numpy_image=colored_depth,
                    parent_metadata=parent_metadata
                )
                
                # Create result dictionary
                result = {
                    'normalized_depth': normalized_depth,
                    'image': colored_depth_image
                }
                
                return (result,)
        except Exception as e:
            raise