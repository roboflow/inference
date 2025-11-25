import torch
from timm.models.vision_transformer import PatchEmbed
import numpy as np


class ImageToTokens(torch.nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        input_channels=4,
        hidden_size=1152,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_size = input_size
        self.patch_size = patch_size
        self.input_channels = input_channels

        self.embedder = PatchEmbed(
            input_size,
            patch_size,
            input_channels,
            hidden_size,
            bias=True,
        )

        num_patches = self.embedder.num_patches

        # Will use fixed sin-cos embedding:
        self.register_buffer(
            "positional_embedding",
            torch.zeros(1, num_patches, hidden_size),
            persistent=False,
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.positional_embedding.shape[-1], int(self.embedder.num_patches**0.5)
        )
        self.positional_embedding.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        # Initialize patch_embed like torch.nn.Linear (instead of torch.nn.Conv2d):
        w = self.embedder.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.constant_(self.embedder.proj.bias, 0)

    def forward(self, x):
        x = self.embedder(x)
        return x


class TokensToImage(torch.nn.Module):
    def __init__(self, channels, patch_size, **kwargs):
        super().__init__(**kwargs)

        self.channels = channels
        self.patch_size = patch_size

    def forward(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.channels
        p = self.patch_size

        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))

        return imgs


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
