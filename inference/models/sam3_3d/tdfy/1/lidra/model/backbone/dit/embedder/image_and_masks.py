import torch
import warnings
from loguru import logger
from torch import nn
from typing import Optional
from lidra.model.layers.llama3.ff import FeedForward


class ImageAndMasks(torch.nn.Module):
    def __init__(
        self,
        image_embedder: torch.nn.Module,
        mask_embedder: torch.nn.Module,
        projection_net_pre_norm: bool = True,
        projection_net_hidden_dim_multiplier: float = 4.0,
    ):
        super().__init__()

        # Provided embedders
        self.image_embedder = image_embedder  # e.g. dino
        self.image_embed_dim = image_embedder.embed_dim
        self.mask_embedder = mask_embedder
        self.mask_embed_dim = mask_embedder.embed_dim
        self.embed_dim = max(self.image_embed_dim, self.mask_embed_dim)

        # Projection nets
        self.projection_pre_norm = projection_net_pre_norm
        self.projection_net_hidden_dim_multiplier = projection_net_hidden_dim_multiplier
        self.image_modality_embed = self._make_projection_net(
            input_embed_dim=self.image_embed_dim,
            output_embed_dim=self.embed_dim,
        )
        self.mask_modality_embed = self._make_projection_net(
            input_embed_dim=self.mask_embed_dim,
            output_embed_dim=self.embed_dim,
        )
        # TODO(sasha): add mask index embed
        # TODO(sasha): generalize to more masks at test-time w/ Fast3R's randomized mask image index embedding technique
        # self.mask_index_embed = <TODO>

    def _make_projection_net(
        self,
        input_embed_dim,
        output_embed_dim: int,
    ):
        if self.projection_pre_norm:
            pre_norm = nn.LayerNorm(input_embed_dim)
        else:
            pre_norm = nn.Identity()

        # Per-token projection + gated activation
        ff_net = FeedForward(
            dim=input_embed_dim,
            hidden_dim=int(
                self.projection_net_hidden_dim_multiplier * output_embed_dim
            ),
            output_dim=output_embed_dim,
        )
        return nn.Sequential(pre_norm, ff_net)

    def forward(self, image: torch.Tensor, mask: torch.Tensor, **kwargs):
        """
        Args:
            image: (B, C, H, W)
            mask: (B, H, W)
        Returns:
            tokens: (B, T, D)
        """
        img_tokens = self.image_embedder(image)
        img_tokens = self.image_modality_embed(img_tokens)

        mask_tokens = self.mask_embedder(image, mask)
        mask_tokens = self.mask_modality_embed(mask_tokens)

        tokens = torch.cat([img_tokens, mask_tokens], dim=1)
        return tokens


class Imagex2AndMaskx2(ImageAndMasks):
    def __init__(
        self,
        image_embedder: torch.nn.Module,
        mask_embedder: torch.nn.Module,
        projection_net_pre_norm: bool = True,
        projection_net_hidden_dim_multiplier: float = 4.0,
        use_pos_embed=False,
    ):
        super().__init__(
            image_embedder,
            mask_embedder,
            projection_net_pre_norm,
            projection_net_hidden_dim_multiplier,
        )
        self.use_pos_embed = use_pos_embed
        if use_pos_embed:
            idx_emb = torch.randn(2, 1, image_embedder.embed_dim)

            self.register_buffer("idx_emb", idx_emb)

    def forward(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        image2: torch.Tensor,
        mask2: torch.Tensor,
        **kwargs,
    ):
        img_tokens = self.image_embedder(image)
        img_tokens = self.image_modality_embed(img_tokens)

        img2_tokens = self.image_embedder(image2)
        img2_tokens = self.image_modality_embed(img2_tokens)

        mask_tokens = self.mask_embedder(image, mask)
        mask_tokens = self.mask_modality_embed(mask_tokens)

        mask2_tokens = self.mask_embedder(image2, mask2)
        mask2_tokens = self.mask_modality_embed(mask2_tokens)

        if self.use_pos_embed:
            img_tokens += self.idx_emb[:1]
            img2_tokens += self.idx_emb[1:2]
            mask_tokens += self.idx_emb[:1]
            mask2_tokens += self.idx_emb[1:2]

        tokens = torch.cat([img_tokens, img2_tokens, mask_tokens, mask2_tokens], dim=1)
        return tokens


class Imagex2AndMaskx2AndPointmapx1(Imagex2AndMaskx2):
    def __init__(
        self,
        image_embedder: torch.nn.Module,
        mask_embedder: torch.nn.Module,
        pointmap_embedder: torch.nn.Module,
        projection_net_pre_norm: bool = True,
        projection_net_hidden_dim_multiplier: float = 4.0,
        use_pos_embed: bool = False,
    ):
        super().__init__(
            image_embedder,
            mask_embedder,
            projection_net_pre_norm,
            projection_net_hidden_dim_multiplier,
            use_pos_embed,
        )
        self.pointmap_embedder = pointmap_embedder
        self.pointmap_embed_dim = pointmap_embedder.embed_dim
        self.embed_dim = max(
            self.image_embed_dim, self.mask_embed_dim, self.pointmap_embed_dim
        )

        # Projection nets
        self.pointmap_modality_embed = self._make_projection_net(
            input_embed_dim=self.pointmap_embed_dim,
            output_embed_dim=self.embed_dim,
        )

    def forward(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        image2: torch.Tensor,
        mask2: torch.Tensor,
    ):
        toks = super().forward(image, mask, image2, mask2)
        img_tokens = self.pointmap_embedder(image)
        img_tokens = self.pointmap_modality_embed(img_tokens)

        img2_tokens = self.pointmap_embedder(image2)
        img2_tokens = self.pointmap_modality_embed(img2_tokens)

        if self.use_pos_embed:
            img_tokens += self.idx_emb[:1]
            img2_tokens += self.idx_emb[1:2]

        toks = torch.cat([toks, img_tokens, img2_tokens], dim=1)
        return toks
