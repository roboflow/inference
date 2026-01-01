# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict as TyDict
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
from addict import Dict

from inference.models.depth_anything_v3.architecture.dpt import (
    FeatureFusionBlock,
    _make_fusion_block,
    _make_scratch,
)
from inference.models.depth_anything_v3.architecture.head_utils import (
    Permute,
    create_uv_grid,
    custom_interpolate,
    position_grid_to_embed,
)


class DualDPT(nn.Module):
    """
    Dual-head DPT for dense prediction with an auxiliary head.
    Simplified for single-view depth estimation - only depth output is used.
    """

    def __init__(
        self,
        dim_in: int,
        *,
        patch_size: int = 14,
        output_dim: int = 2,
        activation: str = "exp",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        pos_embed: bool = True,
        down_ratio: int = 1,
        aux_pyramid_levels: int = 4,
        aux_out1_conv_num: int = 5,
        head_names: Tuple[str, str] = ("depth", "ray"),
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio

        self.aux_levels = aux_pyramid_levels
        self.aux_out1_conv_num = aux_out1_conv_num

        self.head_main, self.head_aux = head_names

        self.intermediate_layer_idx: Tuple[int, int, int, int] = (0, 1, 2, 3)

        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList(
            [nn.Conv2d(dim_in, oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )

        self.scratch = _make_scratch(list(out_channels), features, expand=False)

        # Main fusion chain
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        head_features_1 = features
        head_features_2 = 32
        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
        )
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
        )

        # Auxiliary fusion chain (for ray head - not used for inference but needed for weight loading)
        self.scratch.refinenet1_aux = _make_fusion_block(features)
        self.scratch.refinenet2_aux = _make_fusion_block(features)
        self.scratch.refinenet3_aux = _make_fusion_block(features)
        self.scratch.refinenet4_aux = _make_fusion_block(features, has_residual=False)

        self.scratch.output_conv1_aux = nn.ModuleList(
            [self._make_aux_out1_block(head_features_1) for _ in range(self.aux_levels)]
        )

        use_ln = True
        ln_seq = (
            [Permute((0, 2, 3, 1)), nn.LayerNorm(head_features_2), Permute((0, 3, 1, 2))]
            if use_ln
            else []
        )
        self.scratch.output_conv2_aux = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1
                    ),
                    *ln_seq,
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_features_2, 7, kernel_size=1, stride=1, padding=0),
                )
                for _ in range(self.aux_levels)
            ]
        )

    def forward(
        self,
        feats: List[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
        chunk_size: int = 8,
    ) -> Dict[str, torch.Tensor]:
        B, S, N, C = feats[0][0].shape
        feats = [feat[0].reshape(B * S, N, C) for feat in feats]
        if chunk_size is None or chunk_size >= S:
            out_dict = self._forward_impl(feats, H, W, patch_start_idx)
            out_dict = {k: v.reshape(B, S, *v.shape[1:]) for k, v in out_dict.items()}
            return Dict(out_dict)
        out_dicts = []
        for s0 in range(0, B * S, chunk_size):
            s1 = min(s0 + chunk_size, B * S)
            out_dict = self._forward_impl(
                [feat[s0:s1] for feat in feats],
                H,
                W,
                patch_start_idx,
            )
            out_dicts.append(out_dict)
        out_dict = {
            k: torch.cat([out_dict[k] for out_dict in out_dicts], dim=0)
            for k in out_dicts[0].keys()
        }
        out_dict = {k: v.view(B, S, *v.shape[1:]) for k, v in out_dict.items()}
        return Dict(out_dict)

    def _forward_impl(
        self,
        feats: List[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
    ) -> TyDict[str, torch.Tensor]:
        B, _, C = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size
        resized_feats = []
        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = feats[take_idx][:, patch_start_idx:]
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape(B, C, ph, pw)

            x = self.projects[stage_idx](x)
            if self.pos_embed:
                x = self._add_pos_embed(x, W, H)
            x = self.resize_layers[stage_idx](x)
            resized_feats.append(x)

        # Only compute main fusion for depth (skip aux for inference)
        fused_main, _ = self._fuse(resized_feats)

        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)

        fused_main = custom_interpolate(
            fused_main, (h_out, w_out), mode="bilinear", align_corners=True
        )
        if self.pos_embed:
            fused_main = self._add_pos_embed(fused_main, W, H)

        main_logits = self.scratch.output_conv2(fused_main)
        fmap = main_logits.permute(0, 2, 3, 1)
        main_pred = self._apply_activation_single(fmap[..., :-1], self.activation)
        main_conf = self._apply_activation_single(fmap[..., -1], self.conf_activation)

        return {
            self.head_main: main_pred.squeeze(-1),
            f"{self.head_main}_conf": main_conf,
        }

    def _fuse(self, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        l1, l2, l3, l4 = feats

        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)

        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        aux_out = self.scratch.refinenet4_aux(l4_rn, size=l3_rn.shape[2:])
        aux_list: List[torch.Tensor] = []
        if self.aux_levels >= 4:
            aux_list.append(aux_out)

        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        aux_out = self.scratch.refinenet3_aux(aux_out, l3_rn, size=l2_rn.shape[2:])
        if self.aux_levels >= 3:
            aux_list.append(aux_out)

        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        aux_out = self.scratch.refinenet2_aux(aux_out, l2_rn, size=l1_rn.shape[2:])
        if self.aux_levels >= 2:
            aux_list.append(aux_out)

        out = self.scratch.refinenet1(out, l1_rn)
        aux_out = self.scratch.refinenet1_aux(aux_out, l1_rn)
        aux_list.append(aux_out)

        out = self.scratch.output_conv1(out)
        aux_list = [self.scratch.output_conv1_aux[i](aux) for i, aux in enumerate(aux_list)]

        return out, aux_list

    def _add_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        pw, ph = x.shape[-1], x.shape[-2]
        pe = create_uv_grid(pw, ph, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pe = position_grid_to_embed(pe, x.shape[1]) * ratio
        pe = pe.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pe.to(x.dtype)

    def _make_aux_out1_block(self, in_ch: int) -> nn.Sequential:
        if self.aux_out1_conv_num == 5:
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
                nn.Conv2d(in_ch // 2, in_ch, 3, 1, 1),
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
                nn.Conv2d(in_ch // 2, in_ch, 3, 1, 1),
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
            )
        if self.aux_out1_conv_num == 3:
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
                nn.Conv2d(in_ch // 2, in_ch, 3, 1, 1),
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
            )
        if self.aux_out1_conv_num == 1:
            return nn.Sequential(nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1))
        raise ValueError(f"aux_out1_conv_num {self.aux_out1_conv_num} not supported")

    def _apply_activation_single(
        self, x: torch.Tensor, activation: str = "linear"
    ) -> torch.Tensor:
        act = activation.lower() if isinstance(activation, str) else activation
        if act == "exp":
            return torch.exp(x)
        if act == "expm1":
            return torch.expm1(x)
        if act == "expp1":
            return torch.exp(x) + 1
        if act == "relu":
            return torch.relu(x)
        if act == "sigmoid":
            return torch.sigmoid(x)
        if act == "softplus":
            return torch.nn.functional.softplus(x)
        if act == "tanh":
            return torch.tanh(x)
        return x

