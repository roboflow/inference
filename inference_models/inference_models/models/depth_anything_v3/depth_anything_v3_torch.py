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
from threading import Lock
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors
from transformers import AutoImageProcessor

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.entities import ImageDimensions
from inference_models.models.base.depth_estimation import DepthEstimationModel
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.roboflow.pre_processing import (
    extract_input_images_dimensions,
)
from inference_models.models.depth_anything_v3.architecture import DepthAnything3Net


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


class DepthAnythingV3Torch(
    DepthEstimationModel[torch.Tensor, List[ImageDimensions], torch.Tensor]
):
    """
    Depth Anything V3 model for monocular depth estimation.

    This model uses the Depth Anything V3 architecture with DinoV2 backbone
    and DualDPT head for dense depth prediction.

    Note: Unlike V2, V3 is not HuggingFace Transformers compatible, so the
    architecture is vendored in and model loading is custom. However, the
    external interface (inputs/outputs) matches V2.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        local_files_only: bool = True,
        **kwargs,
    ) -> "DepthAnythingV3Torch":
        if device.type == "cpu":
            warnings.warn(
                "Running DepthAnythingV3 on CPU. This may be slow. "
                "Consider using GPU or MPS if available."
            )

        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif device.type == "mps":
            dtype = torch.float32
        else:
            dtype = torch.float32

        model_package = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=["config.json", "model.safetensors", "preprocessor_config.json"],
        )

        config = parse_config(model_package["config.json"])
        model = DepthAnything3Net(**config)

        state_dict = load_safetensors(model_package["model.safetensors"])
        state_dict = convert_state_dict(state_dict)

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
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
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

        model = model.to(device, dtype=dtype)
        model.eval()

        processor = AutoImageProcessor.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )

        return cls(model=model, processor=processor, device=device, dtype=dtype)

    def __init__(
        self,
        model: DepthAnything3Net,
        processor: AutoImageProcessor,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self._model = model
        self._processor = processor
        self._device = device
        self._dtype = dtype
        self._lock = Lock()

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> Tuple[torch.Tensor, List[ImageDimensions]]:
        image_dimensions = extract_input_images_dimensions(images=images)
        inputs = self._processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        pixel_values = pixel_values.unsqueeze(1)
        pixel_values = pixel_values.to(self._device, dtype=self._dtype)
        return pixel_values, image_dimensions

    def forward(
        self,
        pre_processed_images: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        with self._lock, torch.inference_mode():
            outputs = self._model(pre_processed_images)
            depth_map = outputs["depth"].squeeze(1)
            depth_min = depth_map.min()
            depth_max = depth_map.max()

            # Flip to be consistent with V2
            depth_map = (depth_map * -1) + depth_min + depth_max
            return depth_map

    def post_process(
        self,
        model_results: torch.Tensor,
        pre_processing_meta: List[ImageDimensions],
        **kwargs,
    ) -> List[torch.Tensor]:
        result = []
        for i, dim in enumerate(pre_processing_meta):
            depth_map = model_results[i : i + 1]
            depth_map = torch.nn.functional.interpolate(
                depth_map.unsqueeze(1),
                size=(dim.height, dim.width),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            result.append(depth_map.to(torch.float32))
        return result
