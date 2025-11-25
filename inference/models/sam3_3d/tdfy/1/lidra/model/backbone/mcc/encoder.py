import torch
from timm.models.vision_transformer import PatchEmbed

from lidra.model.backbone.mcc.common import (
    Transformer,
    PositionalEncoding2D,
)


class ImageEncoder(torch.nn.Module):
    def __init__(
        self,
        transformer: Transformer,
        image_size=224,
        image_channels=3,
        image_patch_size=16,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            image_size,
            image_patch_size,
            image_channels,
            transformer.embed_dim,
        )
        self.pe = PositionalEncoding2D(
            transformer.embed_dim,
            grid_size=int(self.patch_embed.num_patches**0.5),
        )
        self.transformer = transformer

    def forward(self, image):
        x = self.patch_embed(image)
        x = self.pe(x)
        x = self.transformer(x)
        return x
