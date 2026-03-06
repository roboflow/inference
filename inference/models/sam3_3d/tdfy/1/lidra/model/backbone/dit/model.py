# original : https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py

import torch
import torch.nn as nn
from copy import deepcopy
from typing import Callable, Optional
from functools import partial

from lidra.model.backbone.dit.block.modulation import (
    Block as ModulationBlock,
    FinalBlock as FinalModulationBlock,
)
from lidra.model.backbone.dit.block.cross_attn import (
    Block as CrossAttnBlock,
    FinalBlock as FinalCrossAttnBlock,
)
from lidra.model.backbone.dit.embedder.image import ImageToTokens, TokensToImage
from lidra.model.backbone.dit.embedder.dino import Dino
from lidra.model.backbone.dit.embedder.label import LabelEmbedder
from lidra.model.backbone.dit.embedder.time import TimestepEmbedder

VALID_INPUT_TYPES = {"image", "tokens", "tokens+linear"}
VALID_CONDITION_TYPES = {"none", "label", "image", "tokens", "dino"}


def unconditional(*args, **kwargs):
    return None


def returns_first_arg(x, *args, **kwargs):
    return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        hidden_size=1152,
        context_size=None,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        input_embedder: Optional[Callable] = None,
        condition_embedder: Optional[Callable] = unconditional,
        final_layer: Optional[Callable] = None,
        output_unembedder: Optional[Callable] = None,
        block_type: nn.Module = ModulationBlock,
    ):
        super().__init__()

        self.learn_sigma = learn_sigma
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.block_type = block_type
        self.context_size = self.hidden_size if context_size is None else context_size

        self.input_embedder = (
            nn.Identity() if input_embedder is None else input_embedder
        )
        self.condition_embedder = (
            nn.Identity() if condition_embedder is None else condition_embedder
        )
        self.time_embedder = TimestepEmbedder(self.context_size)
        self.blocks = nn.ModuleList(
            [
                block_type(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        self.final_layers_mean = (
            returns_first_arg if final_layer is None else final_layer
        )
        if self.learn_sigma:
            self.final_layers_sigma = (
                returns_first_arg if final_layer is None else deepcopy(final_layer)
            )

        self.output_unembedder = (
            nn.Identity() if output_unembedder is None else output_unembedder
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

    def forward(self, x, t, *condition_args, **condition_kwargs):
        # run optional input embedder (e.g. patch embadding for images)
        x = self.input_embedder(x)

        # embed timestep
        t = t[None] if t.ndim == 0 else t  # add batch dimension
        t = self.time_embedder(t)

        # embed conditions (could be None)
        y = self.condition_embedder(*condition_args, **condition_kwargs)

        if y is None:
            c = t
        else:
            c = t + y

        # x and c have shape (B, T, D)

        # TODO(Pierre) : add "t" as a argument to the block ? (proposed by Sasha)
        for block in self.blocks:
            x = block(x, c)

        mean = self.final_layers_mean(x, c)
        mean = self.output_unembedder(mean)

        if self.learn_sigma:
            sigma = self.final_layers_sigma(x, c)
            sigma = self.output_unembedder(sigma)
            return mean, sigma
        return mean


class Tagger(torch.nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)

    def forward(self, x):
        return x + self.weight


def _check_n_tokens(n_tokens):
    if n_tokens is None:
        raise RuntimeError(f"number of tokens should be specified (to 'tag' them)")


# TODO(Pierre) : Not the cleanest code, but works and will do for now ;~) (think impact / think PSC / think money $$$)
def create_dit_backbone(
    input_size=32,  # image channels if input_type is "image", input token dimension otherwise
    cond_image_size=None,
    patch_size=2,
    input_channels=3,
    hidden_size=1152,
    depth=28,
    num_heads=16,
    mlp_ratio=4.0,
    class_dropout_prob=0.1,
    num_classes=1000,
    learn_sigma=True,
    input_type: str = "image",
    condition_type: str = "none",
    n_tokens: int = None,
):
    assert (
        input_type in VALID_INPUT_TYPES
    ), f'invalid input type "{input_type}", should be one of {VALID_INPUT_TYPES}'
    assert (
        condition_type in VALID_CONDITION_TYPES
    ), f'invalid input type "{condition_type}", should be one of {VALID_CONDITION_TYPES}'

    # determine input / output type
    if input_type == "image":
        input_embedder = ImageToTokens(
            input_size,
            patch_size,
            input_channels,
            hidden_size,
        )
        output_unembedder = TokensToImage(input_channels, patch_size)
        use_final_layer = True
    elif input_type == "tokens":
        _check_n_tokens(n_tokens)
        input_embedder = Tagger(num_embeddings=n_tokens, embedding_dim=hidden_size)
        output_unembedder = None
        use_final_layer = False
    elif input_type == "tokens+linear":
        _check_n_tokens(n_tokens)
        input_embedder = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=input_size,
                out_features=hidden_size,
                bias=True,
            ),
            Tagger(num_embeddings=n_tokens, embedding_dim=hidden_size),
        )
        output_unembedder = torch.nn.Linear(
            in_features=hidden_size,
            out_features=input_size,
            bias=True,
        )
        use_final_layer = False

    if cond_image_size is None:
        cond_image_size = input_size

    # determine condition type
    context_dim = None
    if condition_type == "none":
        condition_embedder = unconditional
        modulation = True
    elif condition_type == "tokens":
        condition_embedder = None
        modulation = False
    elif condition_type == "image":
        condition_embedder = ImageToTokens(
            cond_image_size,
            patch_size,
            input_channels,
            hidden_size,
        )
        modulation = False
    elif condition_type == "dino":
        condition_embedder = Dino()
        context_dim = condition_embedder.embed_dim
        modulation = False
    elif condition_type == "label":
        condition_embedder = LabelEmbedder(
            num_classes,
            hidden_size,
            class_dropout_prob,
        )
        modulation = True

    # determine layer type (modulation vs cross-attn)
    final_layer = None
    if modulation:
        block_type = ModulationBlock
        if use_final_layer:
            final_layer = FinalModulationBlock(
                hidden_size,
                patch_size * patch_size * input_channels,
            )
    else:
        block_type = partial(CrossAttnBlock, context_dim=context_dim)
        if use_final_layer:
            final_layer = FinalCrossAttnBlock(
                hidden_size,
                patch_size * patch_size * input_channels,
            )

    model = DiT(
        hidden_size=hidden_size,
        context_size=context_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        learn_sigma=learn_sigma,
        input_embedder=input_embedder,
        condition_embedder=condition_embedder,
        final_layer=final_layer,
        output_unembedder=output_unembedder,
        block_type=block_type,
    )

    return model
