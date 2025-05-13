# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------
"""
ms_deform_attn_func
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable


def ms_deform_attn_core_pytorch(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """ "for debug and test only, need to use cuda version instead"""
    # B, n_heads, head_dim, N
    B, n_heads, head_dim, _ = value.shape
    _, Len_q, n_heads, L, P, _ = sampling_locations.shape
    value_lens = [H * W for H, W in value_spatial_shapes]
    # Split efficiently
    value_list = value.split(value_lens, dim=3)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_all = []
    value_offset = 0
    # Precompute flattened sampling_grids for all levels (to avoid repeated transpose/flatten)
    sampling_grids_levels = sampling_grids.permute(
        3, 0, 2, 1, 4, 5
    ).contiguous()  # L, B, n_heads, Len_q, P, 2
    for lid_, (H, W) in enumerate(value_spatial_shapes):
        this_value = value_list[lid_]
        # B, n_heads, head_dim, H*W -> B*n_heads, head_dim, H, W
        value_l_ = this_value.reshape(B * n_heads, head_dim, H, W)
        # sampling_grids_levels[lid_] shape: B, n_heads, Len_q, P, 2
        grid_l_ = sampling_grids_levels[lid_].reshape(B * n_heads, Len_q, P, 2)
        # grid_sample expects [N, C, H, W] and [N, out_H, out_W, 2], but for 1D output:
        # Make out_H=Len_q, out_W=P
        # sampling_value_l_: [B*n_heads, head_dim, Len_q, P]
        sampling_value_l_ = F.grid_sample(
            value_l_,
            grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_all.append(sampling_value_l_)
    # Stack once, along new level-dimension (-2 so [-1= P, -2=Level])
    sampling_value_tensor = torch.stack(
        sampling_value_all, dim=-2
    )  # [B*n_heads, head_dim, Len_q, L, P]
    sampling_value_tensor = sampling_value_tensor.flatten(
        -2
    )  # [B*n_heads, head_dim, Len_q, L*P]
    attention_weights = attention_weights.transpose(1, 2).reshape(
        B * n_heads, 1, Len_q, L * P
    )
    output = (
        (sampling_value_tensor * attention_weights)
        .sum(-1)
        .view(B, n_heads * head_dim, Len_q)
    )
    return output.transpose(1, 2).contiguous()
