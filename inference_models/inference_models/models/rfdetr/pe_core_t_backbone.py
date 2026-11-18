"""PE-Core-T encoder for the RF-DETR Atto, Femto and Pico sizes (inference only).

A timm ``Eva`` trunk (``vit_pe_core_tiny_patch16_384``) whose blocks attend within
``num_windows x num_windows`` windows, with rotary position embeddings windowed the same
way, and attend globally in the blocks whose outputs feed the projector. Only the forward
pass needed for inference is implemented.
"""

import copy
from typing import List, Tuple

import timm
import torch
from timm.models._features import feature_take_indices
from torch import nn

from inference_models.errors import MissingDependencyError

PE_CORE_T_ARCHITECTURE = "vit_pe_core_tiny_patch16_384"


def _apply_windowing_with_rope(
    x: torch.Tensor,
    *,
    ropes: torch.Tensor,
    num_windows: int,
    patch_size: int,
    height: int,
    width: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # x is (B, HW, C) and ropes (HW, rope_dim). Windows come out batch, window row,
    # window column; batch and window row share one axis so no view exceeds rank 5,
    # which keeps the graph exportable to Core ML.
    num_h_patches, num_w_patches = height // patch_size, width // patch_size
    num_windows_squared = num_windows**2
    batch_size, num_patches, channels = x.shape
    _, rope_dim = ropes.shape
    h_per_window = num_h_patches // num_windows
    w_per_window = num_w_patches // num_windows
    patches_per_window = num_patches // num_windows_squared

    x = x.reshape(
        batch_size * num_windows, h_per_window, num_windows, w_per_window, channels
    )
    x = x.permute(0, 2, 1, 3, 4)
    x = x.reshape(batch_size * num_windows_squared, patches_per_window, channels)

    ropes = ropes.view(num_windows, h_per_window, num_windows, w_per_window, rope_dim)
    ropes = ropes.permute(0, 2, 1, 3, 4)
    ropes = ropes.reshape(num_windows_squared, patches_per_window, rope_dim)

    return x, ropes


def _undo_windowing(
    x: torch.Tensor,
    *,
    num_windows: int,
    patch_size: int,
    height: int,
    width: int,
) -> torch.Tensor:
    # Inverse of _apply_windowing_with_rope for x: (B * windows, HW / windows, C) -> (B, HW, C).
    num_h_patches, num_w_patches = height // patch_size, width // patch_size
    num_windows_squared = num_windows**2
    windowed_batch_size, patches_per_window, channels = x.shape
    batch_size = windowed_batch_size // num_windows_squared
    h_per_window = num_h_patches // num_windows
    w_per_window = num_w_patches // num_windows

    x = x.reshape(
        batch_size * num_windows, num_windows, h_per_window, w_per_window, channels
    )
    x = x.permute(0, 2, 1, 3, 4)
    x = x.reshape(batch_size, patches_per_window * num_windows_squared, channels)

    return x


class PECoreTEncoder(nn.Module):
    """Windowed PE-Core-T ViT that returns feature maps for the RF-DETR projector.

    The timm trunk is built without pretrained weights; the RF-DETR checkpoint provides
    them. Its parameters live under ``model.`` so they match the checkpoint keys
    (``backbone.0.encoder.model.*``).
    """

    def __init__(
        self,
        *,
        shape: Tuple[int, int],
        out_feature_indexes: List[int],
        patch_size: int,
        num_windows: int,
    ):
        """Build the encoder for a trained input size.

        Args:
            shape (tuple[int, int]): Input height and width the model was trained at; sizes the
                absolute position embedding to match the checkpoint.
            out_feature_indexes (list[int]): Blocks whose outputs feed the projector; these
                blocks attend over the whole image instead of within windows.
            patch_size (int): Patch size of the trained model (16 or 20).
            num_windows (int): Windows per side; 1 disables windowing.

        Raises:
            MissingDependencyError: If the installed timm predates PE-Core-T (added in 1.0.18).
            ValueError: If the input size is not divisible by ``patch_size * num_windows``.
        """
        super().__init__()
        if not timm.list_models(PE_CORE_T_ARCHITECTURE):
            raise MissingDependencyError(
                message=f"RF-DETR Atto, Femto and Pico (PyTorch backend) need timm>=1.0.18, which adds the "
                f"`{PE_CORE_T_ARCHITECTURE}` architecture; timm {timm.__version__} is installed.",
                help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
            )
        block_size = patch_size * num_windows
        if shape[0] % block_size or shape[1] % block_size:
            raise ValueError(
                f"PE-Core-T input size {shape} must be divisible by patch_size * num_windows = {block_size}"
            )

        self.model = timm.create_model(
            PE_CORE_T_ARCHITECTURE,
            pretrained=False,
            patch_size=patch_size,
            img_size=shape[0],
        )
        self.model.dynamic_img_size = True
        self.model.patch_embed.strict_img_size = False
        self.model.patch_embed.output_fmt = "NHWC"
        self.model.patch_embed.flatten = False
        self.model.set_input_size(img_size=shape)
        # The class token gets an identity rotary embedding instead (see forward).
        for block in self.model.blocks:
            block.attn.num_prefix_tokens = 0

        self.patch_size = patch_size
        self.num_windows = num_windows
        self.out_feature_indexes = out_feature_indexes
        self._out_feature_channels = [self.model.embed_dim] * len(out_feature_indexes)

    def _rope_for_grid(self, grid: Tuple[int, int]) -> torch.Tensor:
        # timm's RoPE caches the trained grid's embedding and ignores other shapes. Rebuild it on a
        # copy so the module, shared by concurrent requests, keeps the trained grid.
        rope_module = copy.deepcopy(self.model.rope)
        rope_module.update_feat_shape(list(grid))
        rope = rope_module.get_embed()

        return rope

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Run the trunk and return one feature map per ``out_feature_indexes`` entry.

        Inputs other than the trained size work as in the DINOv2 sizes: timm resamples the
        absolute position embedding to the new patch grid, and RoPE is rebuilt for that grid.

        Args:
            x (torch.Tensor): Normalized images, ``(B, 3, H, W)``, with H and W divisible by
                ``patch_size * num_windows``.

        Returns:
            list[torch.Tensor]: Feature maps of shape ``(B, embed_dim, H / patch_size, W / patch_size)``.

        Raises:
            ValueError: If H or W is not divisible by ``patch_size * num_windows``.
        """
        batch_size, _, height, width = x.shape
        block_size = self.patch_size * self.num_windows
        if height % block_size or width % block_size:
            raise ValueError(
                f"PE-Core-T input size {(height, width)} must be divisible by patch_size * num_windows = {block_size}"
            )

        model = self.model
        take_indices, max_index = feature_take_indices(
            len(model.blocks), self.out_feature_indexes
        )
        x = model.patch_embed(x)
        x, rope = model._pos_embed(x)
        grid = (height // self.patch_size, width // self.patch_size)
        if grid != tuple(model.rope.feat_shape):
            rope = self._rope_for_grid(grid)
        x = model.norm_pre(x)

        num_windows = self.num_windows
        num_windows_squared = num_windows**2
        rope_dim = rope.shape[-1]
        # Rotary embeddings are stored as [sin, cos]; sin=0, cos=1 leaves the class token unrotated.
        no_op_rope = torch.cat(
            [
                torch.zeros(rope_dim // 2, device=x.device, dtype=rope.dtype),
                torch.ones(rope_dim // 2, device=x.device, dtype=rope.dtype),
            ]
        )
        if num_windows > 1:
            cls_token, x = x[:, :1], x[:, 1:]
            x, windowed_rope = _apply_windowing_with_rope(
                x,
                ropes=rope,
                num_windows=num_windows,
                patch_size=self.patch_size,
                height=height,
                width=width,
            )
            x = torch.cat(
                (cls_token.repeat_interleave(num_windows_squared, dim=0), x), dim=1
            )
            windowed_rope = torch.cat(
                (
                    no_op_rope.view(1, 1, rope_dim).expand(
                        num_windows_squared, 1, rope_dim
                    ),
                    windowed_rope,
                ),
                dim=1,
            )
            global_rope = windowed_rope.reshape(-1, rope_dim)
            window_rope = windowed_rope.repeat(batch_size, 1, 1).unsqueeze(1)
        else:
            global_rope = torch.cat((no_op_rope.unsqueeze(0), rope), dim=0)
            window_rope = global_rope

        intermediates = []
        for index, block in enumerate(model.blocks[: max_index + 1]):
            attends_globally = num_windows > 1 and index in self.out_feature_indexes
            if attends_globally:
                _, tokens, channels = x.shape
                x = x.reshape(batch_size, num_windows_squared * tokens, channels)
            x = block(x, rope=global_rope if attends_globally else window_rope)
            if attends_globally:
                _, tokens, channels = x.shape
                x = x.reshape(
                    batch_size * num_windows_squared,
                    tokens // num_windows_squared,
                    channels,
                )
            if index in take_indices:
                intermediates.append(model.norm(x))

        grid_height, grid_width = grid
        feature_maps = []
        for features in intermediates:
            features = features[:, 1:]
            if num_windows > 1:
                features = _undo_windowing(
                    features,
                    num_windows=num_windows,
                    patch_size=self.patch_size,
                    height=height,
                    width=width,
                )
            features = features.reshape(batch_size, grid_height, grid_width, -1)
            feature_maps.append(features.permute(0, 3, 1, 2).contiguous())

        return feature_maps
