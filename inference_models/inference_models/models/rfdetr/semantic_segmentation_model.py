"""RF-DETR semantic segmentation network, as trained by Roboflow.

The network is the RF-DETR encoder, the query-independent branch of the
instance segmentation head (its depthwise ``blocks`` and
``spatial_features_proj``) and a 1x1 class projection ``semantic_embed``. Its
state-dict keys match the ``weights.pth`` of Roboflow-trained ``rfdetr-sem-*``
model packages.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from inference_models.models.rfdetr.segmentation_head import DepthwiseConvBlock


class SpatialMaskFeatures(nn.Module):
    """Query-independent branch of the RF-DETR instance segmentation head.

    Args:
        hidden_dim: Feature channels.
        num_blocks: Number of depthwise blocks (``dec_layers`` of the model).
        downsample_ratio: Output stride relative to the input image.
    """

    def __init__(self, hidden_dim: int, num_blocks: int, downsample_ratio: int) -> None:
        super().__init__()
        self.downsample_ratio = downsample_ratio
        self.blocks = nn.ModuleList(
            [DepthwiseConvBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.spatial_features_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(
        self, features: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Upsample encoder features to the output stride and run every block.

        Args:
            features: Encoder features ``[B, C, h, w]``.
            image_size: Input ``(H, W)``.

        Returns:
            Projected features ``[B, C, H / stride, W / stride]``.
        """
        target_size = (
            image_size[0] // self.downsample_ratio,
            image_size[1] // self.downsample_ratio,
        )
        features = F.interpolate(
            features, size=target_size, mode="bilinear", align_corners=False
        )
        for block in self.blocks:
            features = block(features)

        projected_features = self.spatial_features_proj(features)
        return projected_features


class RFDetrSemanticSegmentationNetwork(nn.Module):
    """RF-DETR encoder, spatial mask blocks and a per-pixel class projection.

    Args:
        backbone: Backbone joiner; only its encoder (``backbone[0]``) is used.
        hidden_dim: Feature channels.
        num_blocks: Spatial blocks (``dec_layers``).
        num_classes: Semantic classes, background included.
        downsample_ratio: Output stride.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        hidden_dim: int,
        num_blocks: int,
        num_classes: int,
        downsample_ratio: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.segmentation_head = SpatialMaskFeatures(
            hidden_dim, num_blocks, downsample_ratio
        )
        self.semantic_embed = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Predict class logits for a batch of normalized images.

        Args:
            images: Images ``[B, 3, H, W]``; ``H`` and ``W`` divisible by
                ``patch_size * num_windows``.

        Returns:
            Class logits ``[B, num_classes, H / stride, W / stride]``.
        """
        features, _ = self.backbone[0].forward_export(images)
        spatial_features = self.segmentation_head(features[0], images.shape[-2:])

        logits = self.semantic_embed(spatial_features)
        return logits
