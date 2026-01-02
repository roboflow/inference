# Copyright (c) 2025 Roboflow, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import warnings
from pathlib import Path
from typing import Any, Tuple
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from transformers import AutoImageProcessor

from inference.models.depth_anything_v2.depth_anything_v2 import DepthAnythingV2
from inference.models.depth_anything_v3.architecture import DepthAnything3Net
from inference.models.transformers import TransformerModel


def convert_state_dict(state_dict: dict) -> dict:
    """
    Convert state dict from official DA3 format to our simplified format.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove 'model.' prefix if present
        new_key = key
        if new_key.startswith("model."):
            new_key = new_key[6:]

        # Map backbone paths
        new_key = new_key.replace("net.", "backbone.")

        # Skip camera encoder/decoder weights (not used for depth-only inference)
        if "cam_enc" in new_key or "cam_dec" in new_key:
            continue

        # Skip GS head weights (not used)
        if "gs_head" in new_key or "gs_adapter" in new_key:
            continue

        new_state_dict[new_key] = value

    return new_state_dict


def parse_config(config_path: str) -> dict:
    """
    Parse the config.json file from HuggingFace/official DA3 format.

    Args:
        config_path: Path to the config.json file

    Returns:
        Dictionary with model configuration parameters
    """
    with open(config_path, "r") as f:
        raw_config = json.load(f)

    config = raw_config.get("config", raw_config)

    # Extract backbone (net) configuration
    net_config = config.get("net", {})
    backbone_name = net_config.get("name", "vitb")
    out_layers = net_config.get("out_layers", [5, 7, 9, 11])
    alt_start = net_config.get("alt_start", 4)
    qknorm_start = net_config.get("qknorm_start", 4)
    rope_start = net_config.get("rope_start", 4)
    cat_token = net_config.get("cat_token", True)

    # Extract head configuration
    head_config = config.get("head", {})
    head_dim_in = head_config.get("dim_in", 1536)
    head_output_dim = head_config.get("output_dim", 2)
    head_features = head_config.get("features", 128)
    head_out_channels = head_config.get("out_channels", [96, 192, 384, 768])

    return {
        "backbone_name": backbone_name,
        "out_layers": out_layers,
        "alt_start": alt_start,
        "qknorm_start": qknorm_start,
        "rope_start": rope_start,
        "cat_token": cat_token,
        "head_dim_in": head_dim_in,
        "head_output_dim": head_output_dim,
        "head_features": head_features,
        "head_out_channels": head_out_channels,
    }


class DepthAnythingV3(DepthAnythingV2):
    """
    Depth Anything V3 model for monocular depth estimation.

    This model uses the Depth Anything V3 architecture with DinoV2 backbone
    and DualDPT head for dense depth prediction.

    Note: Unlike V2, V3 is not HuggingFace Transformers compatible, so the
    architecture is vendored in and model loading is custom. However, the
    external interface (inputs/outputs) matches V2.
    """

    endpoint = "depth-anything-v3/small"

    def __init__(self, *args, **kwargs):

        try:
            super().__init__(*args, **kwargs)
        except Exception as e:
            print(f"Error initializing depth estimation model: {str(e)}")
            raise

        # Set appropriate dtype based on device
        if self.device.type == "mps":
            self.model = self.model.to(torch.float32)  # MPS prefers float32
        elif self.device.type == "cpu":
            warnings.warn(
                "Running DepthAnythingV3 on CPU. This may be very slow. Consider using GPU or MPS if available."
            )

    def initialize_model(self, **kwargs):
        """Initialize the model with vendored architecture instead of HF Transformers."""
        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            warnings.warn(
                "Running DepthAnythingV3 on CPU. This may be slow. "
                "Consider using GPU or MPS if available."
            )

        # Determine dtype
        if self.device.type == "cuda":
            self.dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        elif self.device.type == "mps":
            self.dtype = torch.float32  # MPS works better with float32
        else:
            self.dtype = torch.float32

        # Load configuration from config.json
        config_path = self._get_config_path()
        self.config = parse_config(config_path)

        # Build model with vendored architecture
        self.model = DepthAnything3Net(**self.config)

        # Load weights
        self._load_weights()

        # Move model to device and set eval mode
        self.model = self.model.to(self.device, dtype=self.dtype)
        self.model.eval()

        # Load processor from cache dir (uses preprocessor_config.json)
        self.processor = AutoImageProcessor.from_pretrained(self.cache_dir)

    def _load_weights(self):
        """Load pretrained weights from the model cache."""
        weights_path = self._get_model_weights_path()

        if weights_path.endswith(".safetensors"):
            state_dict = load_safetensors(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location="cpu")

        # Convert state dict format
        state_dict = convert_state_dict(state_dict)

        # Load weights (strict=False to handle missing aux weights)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        # Filter out expected missing keys:
        # - cam_enc, cam_dec: Camera encoder/decoder (not used for depth-only)
        # - gs_head, gs_adapter: Gaussian splatting head (not used)
        # - output_conv2_aux: Auxiliary ray prediction heads (not used for depth-only)
        expected_missing = [
            "cam_enc",
            "cam_dec",
            "gs_head",
            "gs_adapter",
            "output_conv2_aux",
        ]
        unexpected_filtered = [
            k for k in unexpected if not any(skip in k for skip in expected_missing)
        ]
        missing_filtered = [
            k for k in missing if not any(skip in k for skip in expected_missing)
        ]

        if missing_filtered:
            warnings.warn(f"Missing keys when loading weights: {missing_filtered}")
        if unexpected_filtered:
            warnings.warn(
                f"Unexpected keys when loading weights: {unexpected_filtered}"
            )

    def _get_config_path(self) -> str:
        """Get path to model config file."""
        cache_dir = Path(self.cache_dir)
        config_file = cache_dir / "config.json"
        if config_file.exists():
            return str(config_file)
        raise FileNotFoundError(
            f"Could not find config.json in {cache_dir}. "
            f"Expected config.json to be downloaded alongside model weights."
        )

    def _get_model_weights_path(self) -> str:
        """Get path to model weights file."""
        cache_dir = Path(self.cache_dir)

        # Try weights.safetensors (common HF convention)
        weights_file = cache_dir / "model.safetensors"
        if weights_file.exists():
            return str(weights_file)
        else:
            raise FileNotFoundError(f"Could not find {weights_file} in {cache_dir}")

    def predict(self, image_in: Image.Image, prompt="", history=None, **kwargs):
        """
        Run depth prediction on an input image.

        Unlike V2, the vendored DepthAnything3Net expects a tensor directly
        with shape (B, N, 3, H, W) where N=1 for single-view inference.
        """
        from inference.core.workflows.execution_engine.entities.base import (
            ImageParentMetadata,
            WorkflowImageData,
        )

        # Process input image using the HF processor
        inputs = self.processor(images=image_in, return_tensors="pt")

        # Extract pixel_values and add the N dimension
        # Processor outputs: (B, C, H, W) -> Model expects: (B, N, C, H, W)
        pixel_values = inputs["pixel_values"]
        pixel_values = pixel_values.unsqueeze(1)  # Add N=1 dimension

        # Move to device and dtype
        pixel_values = pixel_values.to(self.device, dtype=self.dtype)

        # Run inference
        with torch.inference_mode():
            outputs = self.model(pixel_values)

            # Extract depth from model output
            # Model returns dict with 'depth' key containing (B, S, H, W) tensor
            # where S=1 for single-view, so we squeeze it to (B, H, W)
            depth_map = outputs["depth"].squeeze(1)

            # Resize back to original image size
            depth_map = torch.nn.functional.interpolate(
                depth_map.unsqueeze(1),
                size=(image_in.height, image_in.width),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

            depth_map = depth_map.to(torch.float32).cpu().numpy()

            # Normalize depth values
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            if depth_max == depth_min:
                raise ValueError("Depth map has no variation (min equals max)")
            normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)
            normalized_depth = 1 - normalized_depth

            # Create visualization
            depth_for_viz = (normalized_depth * 255.0).astype(np.uint8)
            cmap = plt.get_cmap("viridis")
            colored_depth = (cmap(depth_for_viz)[:, :, :3] * 255).astype(np.uint8)

            # Convert numpy array to WorkflowImageData
            parent_metadata = ImageParentMetadata(parent_id=f"{uuid4()}")
            colored_depth_image = WorkflowImageData(
                numpy_image=colored_depth, parent_metadata=parent_metadata
            )

            result = {
                "image": colored_depth_image,
                "normalized_depth": normalized_depth,
            }

            return (result,)
