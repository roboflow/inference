"""V-JEPA encoder and frame-anchored, class-specific span predictions."""

import torch
from torch import nn

from .architecture import CrossAttentionBlock


class SpanHead(nn.Module):
    def __init__(self, frames, classes):
        super().__init__()
        self.frames, self.classes = frames, classes
        self.frame_queries = nn.Parameter(torch.empty(1, frames, 768))
        self.pooler = CrossAttentionBlock(768, 12)
        self.classifier = nn.Linear(768, 3 * classes)
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
