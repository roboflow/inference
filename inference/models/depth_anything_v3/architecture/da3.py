# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
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

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from inference.models.depth_anything_v3.architecture.dinov2 import DinoV2
from inference.models.depth_anything_v3.architecture.dualdpt import DualDPT


class DepthAnything3Net(nn.Module):
    """
    Depth Anything 3 network for depth estimation.
    Simplified for single-view depth-only inference.

    This network consists of:
    - Backbone: DinoV2 feature extractor
    - Head: DualDPT for depth prediction

    Returns:
        Dictionary containing:
        - depth: Predicted depth map (B, H, W)
        - depth_conf: Depth confidence map (B, H, W)
    """

    PATCH_SIZE = 14

    def __init__(
        self,
        backbone_name: str,
        out_layers: list,
        alt_start: int,
        qknorm_start: int,
        rope_start: int,
        cat_token: bool,
        head_dim_in: int,
        head_output_dim: int,
        head_features: int,
        head_out_channels: list,
    ):
        """
        Initialize DepthAnything3Net.

        Args:
            backbone_name: DinoV2 backbone variant ("vits" or "vitb")
            out_layers: Layer indices to extract features from
            alt_start: Layer index to start alternating attention
            qknorm_start: Layer index to start QK normalization
            rope_start: Layer index to start RoPE
            cat_token: Whether to concatenate local and global tokens
            head_dim_in: Input dimension for the head
            head_output_dim: Output dimension for the head
            head_features: Feature dimension in the head
            head_out_channels: Output channel dimensions per stage
        """
        super().__init__()
        self.backbone = DinoV2(
            name=backbone_name,
            out_layers=out_layers,
            alt_start=alt_start,
            qknorm_start=qknorm_start,
            rope_start=rope_start,
            cat_token=cat_token,
        )
        self.head = DualDPT(
            dim_in=head_dim_in,
            output_dim=head_output_dim,
            features=head_features,
            out_channels=head_out_channels,
        )
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input images (B, N, 3, H, W) where N=1 for single-view

        Returns:
            Dictionary containing depth predictions
        """
        # Extract features using backbone
        feats, _ = self.backbone(x)
        H, W = x.shape[-2], x.shape[-1]

        # Process features through depth head
        with torch.autocast(device_type=x.device.type, enabled=False):
            output = self._process_depth_head(feats, H, W)

        return output

    def _process_depth_head(
        self, feats: list[torch.Tensor], H: int, W: int
    ) -> Dict[str, torch.Tensor]:
        """Process features through the depth prediction head."""
        return self.head(feats, H, W, patch_start_idx=0)

