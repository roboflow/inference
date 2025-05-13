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
    B, n_heads, head_dim, _ = value.shape
    _, Len_q, n_heads, L, P, _ = sampling_locations.shape

    # Precompute flattened sizes for split/view
    spatial_areas = [int(H * W) for H, W in value_spatial_shapes]

    # Fast splitting, avoids list/genexpr overhead
    value_list = []
    start = 0
    for area, (H, W) in zip(spatial_areas, value_spatial_shapes):
        val = value[..., start : start + area]
        value_list.append(val.view(B * n_heads, head_dim, H, W))
        start += area

    # Vectorized normalize: Only do broadcast ops once
    sampling_grids = 2 * sampling_locations - 1

    # Pretranspose/flatten grids for all levels at once
    # (B, Len_q, n_heads, L, P, 2) -> (L, B*n_heads, Len_q, P, 2)
    sampling_grids = sampling_grids.permute(3, 0, 2, 1, 4, 5).contiguous()
    sampling_grids = sampling_grids.view(L, B * n_heads, Len_q, P, 2)

    # Use list comprehension for lesser Python overhead in append loop
    sampling_value_list = [
        F.grid_sample(
            value_l_,
            sampling_grids[lid_],  # (B * n_heads, Len_q, P, 2)
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        for lid_, value_l_ in enumerate(value_list)
    ]
    # Each is (B * n_heads, head_dim, Len_q, P)

    # Stack and flatten spatial dims in one step
    sampling_value = torch.cat(sampling_value_list, dim=3)  # concat spatial (L * P)
    # (B * n_heads, head_dim, Len_q, L * P)
    # See original: stack(sampling_value_list, -2).flatten(-2)

    # attention_weights: (N, Len_q, n_heads, L * P)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        B * n_heads, 1, Len_q, L * P
    )

    # Output: (B, n_heads * head_dim, Len_q)
    output = (
        (sampling_value * attention_weights).sum(-1).view(B, n_heads * head_dim, Len_q)
    )
    return output.transpose(1, 2).contiguous()
