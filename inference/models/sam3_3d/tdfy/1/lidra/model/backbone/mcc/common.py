import torch
from timm.models.vision_transformer import PatchEmbed, Block, Mlp, DropPath

##### <TODO> (Pierre) : clean / factorize


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, n_cls_token=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="ij")  # here w goes first
    grid = torch.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if n_cls_token > 0:
        pos_embed = torch.concatenate(
            [torch.zeros([n_cls_token, embed_dim]), pos_embed],
            axis=0,
        )
    return pos_embed


class PositionalEncoding2D(torch.nn.Module):
    def __init__(self, embed_dim: int, grid_size: int, n_cls_token: int = 0):
        super().__init__()

        self.register_buffer(
            "pe",
            get_2d_sincos_pos_embed(
                embed_dim,
                grid_size,
                n_cls_token=n_cls_token,
            ),
            persistent=False,
        )

    def forward(self, x):
        x += self.pe
        return x


##### </TODO>


class Transformer(torch.nn.Module):
    def __init__(
        self,
        embed_dim=1024,
        n_blocks=24,
        n_heads=16,
        n_cls_tokens=0,
        norm_layer=torch.nn.LayerNorm,
        drop_path=0.1,
        mlp_ratio=4.0,
        block_fn=Block,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.blocks = torch.nn.ModuleList(
            [
                block_fn(
                    embed_dim,
                    n_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    drop_path=drop_path,
                )
                for _ in range(n_blocks)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # init cls tokens
        if n_cls_tokens > 0:
            cls_tokens_init_tensor = torch.zeros(1, n_cls_tokens, embed_dim)
            torch.nn.init.normal_(cls_tokens_init_tensor, std=0.02)
            self.cls_tokens = torch.nn.Parameter(cls_tokens_init_tensor)
        else:
            self.cls_tokens = None

    @property
    def n_cls_tokens(self):
        return 0 if (self.cls_tokens is None) else self.cls_tokens.shape[1]

    def cls_extend(self, x):
        if self.cls_tokens is not None:
            # extend cls tolkens to match batch size
            cls_tokens = self.cls_tokens.expand(x.shape[0], -1, -1)

            # add tokens
            x = torch.cat((cls_tokens, x), dim=1)
        return x

    def forward(self, x, *block_args, **block_kwargs):
        x = self.cls_extend(x)

        # apply transformer blocks
        for blk in self.blocks:
            x = blk(x, *block_args, **block_kwargs)

        x = self.norm(x)

        return x
