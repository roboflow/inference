"""V-JEPA encoder and frame-anchored, class-specific span predictions."""

import math

import torch
from torch import nn

from .vendor.modules import CrossAttentionBlock


class SpanHead(nn.Module):
    def __init__(self, frames, classes):
        super().__init__()
        self.frames, self.classes = frames, classes
        self.frame_queries = nn.Parameter(torch.empty(1, frames, 768))
        nn.init.trunc_normal_(self.frame_queries, std=0.02)
        self.pooler = CrossAttentionBlock(768, 12, mlp_ratio=4, qkv_bias=True)
        for module in self.pooler.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
        with torch.no_grad():
            self.pooler.mlp.fc2.weight.div_(math.sqrt(2))
        self.classifier = nn.Linear(768, 3 * classes)
        nn.init.zeros_(self.classifier.bias)
        self.register_buffer("anchors", torch.arange(frames).float() + 0.5)

    def forward(self, features):
        queries = self.frame_queries.expand(features.shape[0], -1, -1)
        output = self.classifier(self.pooler(queries, features))
        distances = (
            output[..., self.classes :]
            .float()
            .reshape(*output.shape[:2], self.classes, 2)
        )
        left, right = (distances.sigmoid() * self.frames).unbind(-1)
        anchors = self.anchors[None, :, None]
        return output[..., : self.classes], torch.stack(
            (anchors - left, anchors + right), -1
        )
