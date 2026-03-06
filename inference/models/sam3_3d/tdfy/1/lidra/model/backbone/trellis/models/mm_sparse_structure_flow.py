from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.utils import convert_module_to_f16, convert_module_to_f32
from collections import namedtuple
from ..modules.utils import FP16_TYPE
from ..modules.transformer import (
    ModulatedTransformerCrossBlock,
)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: a 1-D Tensor of N indices, one per batch element.
                These may be fractional.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        t = t[:, None].float()
        args = t * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        if t.ndim == 0:
            t = t.unsqueeze(0)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class SparseStructureFlowModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        is_shortcut_model: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = FP16_TYPE if use_fp16 else torch.float32
        self.is_shortcut_model = is_shortcut_model
        self.t_embedder = TimestepEmbedder(model_channels)
        if is_shortcut_model:
            self.d_embedder = TimestepEmbedder(model_channels)  # for shortcut model

        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        self.blocks = nn.ModuleList(
            [
                ModulatedTransformerCrossBlock(
                    model_channels,
                    cond_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    attn_mode="full",
                    use_checkpoint=self.use_checkpoint,
                    use_rope=(pe_mode == "rope"),
                    share_mod=share_mod,
                    qk_rms_norm=self.qk_rms_norm,
                    qk_rms_norm_cross=self.qk_rms_norm_cross,
                )
                for _ in range(num_blocks)
            ]
        )

        self.initialize_weights()

        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # zero init like controlnet, for MLP should only zero
        # the weight of the last layer only
        if self.is_shortcut_model:
            nn.init.constant_(self.d_embedder.mlp[2].weight, 0)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        h: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        d: torch.Tensor = None,
    ) -> torch.Tensor:
        t_emb = self.t_embedder(t)
        if d is not None:
            d_emb = self.d_embedder(d)
            t_emb = t_emb + d_emb
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)

        input_dtype = h.dtype
        t_emb = t_emb.type(self.dtype)
        h = h.type(self.dtype)
        cond = cond.type(self.dtype)

        for i, block in enumerate(self.blocks):
            h = block(h, t_emb, cond)
        h = h.type(input_dtype)

        return h


class SparseStructureFlowTdfyWrapper(SparseStructureFlowModel):
    def __init__(self, latent_mapping: dict, *args, **kwargs):
        condition_embedder = kwargs.pop("condition_embedder", None)
        # if enabled, model will record the condition_shape in one run and uses zeros for all that afterwards
        force_zeros_cond = kwargs.pop("force_zeros_cond", False)
        # backward compatible to models trained before PR #87
        kwargs.pop("shape_attend_pose", None)
        super().__init__(*args, **kwargs)
        if condition_embedder is not None:
            self.condition_embedder = condition_embedder
        else:
            self.condition_embedder = lambda x: x
        self.force_zeros_cond = force_zeros_cond
        self.latent_mapping = nn.ModuleDict(latent_mapping)
        self.input_latent_mappings = list(self.latent_mapping.keys())

    def forward(
        self,
        latents_dict: dict,
        t: torch.Tensor,
        *condition_args,
        **condition_kwargs,
    ) -> dict:
        d = condition_kwargs.pop("d", None)

        cfg_activate = condition_kwargs.pop("cfg", False)
        cfg_batch_mode = condition_kwargs.pop("cfg_batch_mode", False)
        use_unconditional_from_flow_matching = condition_kwargs.pop(
            "use_unconditional_from_flow_matching", False
        )

        if cfg_batch_mode:
            # Handle batched CFG: create batched inputs internally
            # Double the batch size for conditional + unconditional
            latents_dict_batched = {}
            for k, v in latents_dict.items():
                latents_dict_batched[k] = torch.cat([v, v], dim=0)
            latents_dict = latents_dict_batched

            if isinstance(t, torch.Tensor):
                t = torch.cat([t, t], dim=0)

            # Handle d parameter for batched mode
            if d is not None and use_unconditional_from_flow_matching:
                # Create separate d tensors: original for conditional, zeros for unconditional
                d_cond = d
                d_uncond = torch.zeros_like(
                    d
                )  # HACK: we use unconditional from flow matching mode by setting d to 0
                d = torch.cat([d_cond, d_uncond], dim=0)
            elif d is not None:
                # Both halves use the same d
                d = torch.cat([d, d], dim=0)

            # Embed conditions only once (for conditional part)
            cond = self.condition_embedder(*condition_args, **condition_kwargs)

            # Create batched condition and uncondition
            if self.force_zeros_cond:
                # first half conditional, second half zeros (unconditional)
                cond_batched = torch.cat([cond, torch.zeros_like(cond)], dim=0)
            else:
                # both halves are conditional
                cond_batched = torch.cat([cond, cond], dim=0)
            cond = cond_batched

        elif self.force_zeros_cond and cfg_activate:
            # TODO: @weiyaowang, refactor to read directly from embedder
            cond = self.condition_embedder(*condition_args, **condition_kwargs)
            cond = cond * 0
        else:
            cond = self.condition_embedder(*condition_args, **condition_kwargs)

        # concatenate input
        latent_list = self.concatenate_input(latents_dict)
        concatenated_input = torch.cat(latent_list, dim=1)
        output = super().forward(concatenated_input, t, cond, d)

        # split input to multiple output modalities
        output_latents = self.split_output(output)

        return output_latents

    def concatenate_input(
        self,
        latents_dict: dict,
    ) -> torch.Tensor:
        # concatenate input from multiple modalities
        latent_list = []
        for latent_name in self.input_latent_mappings:
            assert (
                latent_name in latents_dict
            ), f"'{latent_name}' not found in latents_dict"
            latent_input = latents_dict[latent_name]
            x = self.latent_mapping[latent_name].to_input(latent_input)
            latent_list.append(x)

        return latent_list

    def split_output(self, output: torch.Tensor) -> Dict:
        start = 0
        output_latents = {}
        for latent_name in self.input_latent_mappings:
            token_len = self.latent_mapping[latent_name].pos_emb.shape[0]
            latent = output[:, start : start + token_len]
            latent = self.latent_mapping[latent_name].to_output(latent)
            output_latents[latent_name] = latent
            start += token_len

        return output_latents
