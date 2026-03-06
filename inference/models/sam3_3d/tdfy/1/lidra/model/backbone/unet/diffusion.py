# DDPM version of the UNet architecture
# source : https://github.com/tqch/ddpm-torch/blob/b60eb8deee899900c28525fd77ff53cdd26d6558/ddpm_torch/models/unet.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable
from itertools import repeat

DEFAULT_NONLINEARITY = nn.SiLU()
DEFAULT_DTYPE = torch.float32


def ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, Iterable):
            return tuple(x)
        else:
            return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


pair = ntuple(2, "pair")


@torch.jit.script
def get_timestep_embedding(
    timesteps, embed_dim: int, dtype: torch.dtype = DEFAULT_DTYPE
):
    """
    Adapted from fairseq/fairseq/modules/sinusoidal_positional_embedding.py
    The implementation is slightly different from the decription in Section 3.5 of [1]
    [1] Vaswani, Ashish, et al. "Attention is all you need."
     Advances in neural information processing systems 30 (2017).
    """
    half_dim = embed_dim // 2
    embed = math.log(10000) / (half_dim - 1)
    embed = torch.exp(
        -torch.arange(half_dim, dtype=dtype, device=timesteps.device) * embed
    )
    embed = torch.outer(timesteps.ravel().to(dtype), embed)
    embed = torch.cat([torch.sin(embed), torch.cos(embed)], dim=1)
    if embed_dim % 2 == 1:
        embed = F.pad(embed, [0, 1])  # padding the last dimension
    assert embed.dtype == dtype
    return embed


@torch.jit.script
def approx_std_normal_cdf(x):
    """
    Reference:
    Page, E. “Approximations to the Cumulative Normal Function and Its Inverse for Use on a Pocket Calculator.”
     Applied Statistics 26.1 (1977): 75–76. Web.
    """
    return 0.5 * (
        1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))
    )


def DEFAULT_INITIALIZER(x, scale=1.0):
    """
    PyTorch Xavier uniform initialization: w ~ Uniform(-a, a), where a = gain * (6 / (fan_in + fan_out)) ** .5
    TensorFlow Variance-Scaling initialization (mode="fan_avg", distribution="uniform"):
    w ~ Uniform(-a, a), where a = (6 * scale / (fan_in + fan_out)) ** .5
    Therefore, gain = scale ** .5
    """
    return nn.init.xavier_uniform_(x, gain=math.sqrt(scale or 1e-10))


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_scale=1.0):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), dtype=DEFAULT_DTYPE)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features,), dtype=DEFAULT_DTYPE))
        else:
            self.register_parameter("bias", None)
        self.init_scale = init_scale
        self.reset_parameters()

    def reset_parameters(self):
        DEFAULT_INITIALIZER(self.weight, scale=self.init_scale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        init_scale=1.0,
    ):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size = pair(kernel_size)
        self.weight = nn.Parameter(
            torch.empty(
                (out_channels, in_channels // groups, kernel_size[0], kernel_size[1]),
                dtype=DEFAULT_DTYPE,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty((out_channels,), dtype=DEFAULT_DTYPE))
        else:
            self.register_parameter("bias", None)
        self.stride = pair(stride)
        self.padding = padding if isinstance(padding, str) else pair(padding)
        self.dilation = pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        self.init_scale = init_scale
        self.reset_parameter()

    def reset_parameter(self):
        DEFAULT_INITIALIZER(self.weight, scale=self.init_scale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)

    def forward(self, x):
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class SamePad2d(nn.Module):
    def __init__(self, kernel_size, stride, mode="constant", value=0.0):
        super(SamePad2d, self).__init__()
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.mode = mode
        self.value = value

    def forward(self, x):
        _, _, h, w = x.shape
        (k1, k2), (s1, s2) = self.kernel_size, self.stride
        h_pad, w_pad = (
            s1 * math.ceil(h / s1 - 1) + k1 - h,
            s2 * math.ceil(w / s2 - 1) + k2 - w,
        )
        top_pad, bottom_pad = (
            (math.floor(h_pad / 2), math.ceil(h_pad / 2)) if h_pad else (0, 0)
        )
        left_pad, right_pad = (
            (math.floor(w_pad / 2), math.ceil(w_pad / 2)) if w_pad else (0, 0)
        )
        x = F.pad(
            x,
            pad=(left_pad, right_pad, top_pad, bottom_pad),
            mode=self.mode,
            value=self.value,
        )
        return x


class Sequential(nn.Sequential):
    def forward(self, input, **kwargs):
        for module in self:
            input = module(input, **kwargs)
        return input


class DEFAULT_NORMALIZER(nn.GroupNorm):
    def __init__(self, num_channels, num_groups=32):
        super().__init__(
            num_groups=num_groups, num_channels=num_channels, eps=1e-6
        )  # PyTorch default eps is 1e-5


class AttentionBlock(nn.Module):
    normalize = DEFAULT_NORMALIZER

    def __init__(self, in_channels, mid_channels=None, out_channels=None):
        super(AttentionBlock, self).__init__()
        mid_channels = mid_channels or in_channels
        out_channels = out_channels or in_channels
        self.norm = self.normalize(in_channels)
        self.project_in = Conv2d(in_channels, 3 * mid_channels, 1)
        self.project_out = Conv2d(mid_channels, out_channels, 1, init_scale=0.0)
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else Conv2d(in_channels, out_channels, 1)
        )

    @staticmethod
    def qkv(q, k, v):
        B, C, H, W = q.shape
        w = torch.einsum("bchw, bcHW -> bhwHW", q, k)
        w = torch.softmax(w.reshape(B, H, W, H * W) / math.sqrt(C), dim=-1).reshape(
            B, H, W, H, W
        )
        out = torch.einsum(
            "bhwHW, bcHW -> bchw", w, v
        )  # this will break the contiguity -> impaired performance
        return out.contiguous()  # force to return a contiguous tensor

    def forward(self, x, **kwargs):
        skip = self.skip(x)
        C = x.shape[1]
        assert C == self.in_channels
        q, k, v = self.project_in(self.norm(x)).chunk(3, dim=1)
        x = self.qkv(q, k, v)
        x = self.project_out(x)
        return x + skip


class ResidualBlock(nn.Module):
    normalize = DEFAULT_NORMALIZER
    nonlinearity = DEFAULT_NONLINEARITY

    def __init__(self, in_channels, out_channels, embed_dim, drop_rate=0.0):
        super(ResidualBlock, self).__init__()
        self.norm1 = self.normalize(in_channels)
        self.conv1 = Conv2d(in_channels, out_channels, 3, 1, 1)
        self.fc = Linear(embed_dim, out_channels)
        self.norm2 = self.normalize(out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, 3, 1, 1, init_scale=0.0)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else Conv2d(in_channels, out_channels, 1)
        )
        self.dropout = nn.Dropout(p=drop_rate, inplace=True)

    def forward(self, x, t_emb):
        skip = self.skip(x)
        x = self.conv1(self.nonlinearity(self.norm1(x)))
        x += self.fc(self.nonlinearity(t_emb))[:, :, None, None]
        x = self.dropout(self.nonlinearity(self.norm2(x)))
        x = self.conv2(x)
        return x + skip


class UNet(nn.Module):
    normalize = DEFAULT_NORMALIZER
    nonlinearity = DEFAULT_NONLINEARITY

    def __init__(
        self,
        in_channels,
        hid_channels,
        out_channels,
        ch_multipliers,
        num_res_blocks,
        apply_attn,
        time_embedding_dim=None,
        drop_rate=0.0,
        resample_with_conv=True,
    ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.time_embedding_dim = time_embedding_dim or 4 * self.hid_channels
        levels = len(ch_multipliers)
        self.ch_multipliers = ch_multipliers
        if isinstance(apply_attn, bool):
            apply_attn = [apply_attn for _ in range(levels)]
        self.apply_attn = apply_attn
        self.num_res_blocks = num_res_blocks
        self.drop_rate = drop_rate
        self.resample_with_conv = resample_with_conv

        self.embed = Sequential(
            Linear(self.hid_channels, self.time_embedding_dim),
            self.nonlinearity,
            Linear(self.time_embedding_dim, self.time_embedding_dim),
        )
        self.in_conv = Conv2d(in_channels, hid_channels, 3, 1, 1)
        self.levels = levels
        self.downsamples = nn.ModuleDict(
            {f"level_{i}": self._get_downsample_by_level(i) for i in range(levels)}
        )
        mid_channels = ch_multipliers[-1] * hid_channels
        embed_dim = self.time_embedding_dim
        self.middle = Sequential(
            ResidualBlock(
                mid_channels, mid_channels, embed_dim=embed_dim, drop_rate=drop_rate
            ),
            AttentionBlock(mid_channels),
            ResidualBlock(
                mid_channels, mid_channels, embed_dim=embed_dim, drop_rate=drop_rate
            ),
        )
        self.upsamples = nn.ModuleDict(
            {f"level_{i}": self._get_upsample_by_level(i) for i in range(levels)}
        )
        self.out_conv = Sequential(
            self.normalize(hid_channels),
            self.nonlinearity,
            Conv2d(hid_channels, out_channels, 3, 1, 1, init_scale=0.0),
        )

    def _get_block_by_level(self, level):
        block_kwargs = {
            "embed_dim": self.time_embedding_dim,
            "drop_rate": self.drop_rate,
        }
        if self.apply_attn[level]:

            def block(in_chans, out_chans):
                return Sequential(
                    ResidualBlock(in_chans, out_chans, **block_kwargs),
                    AttentionBlock(out_chans),
                )

        else:

            def block(in_chans, out_chans):
                return ResidualBlock(in_chans, out_chans, **block_kwargs)

        return block

    def _get_downsample_by_level(self, level):
        block = self._get_block_by_level(level)
        prev_chans = (
            self.ch_multipliers[level - 1] if level else 1
        ) * self.hid_channels
        curr_chans = self.ch_multipliers[level] * self.hid_channels
        modules = nn.ModuleList([block(prev_chans, curr_chans)])
        for _ in range(self.num_res_blocks - 1):
            modules.append(block(curr_chans, curr_chans))
        if level != self.levels - 1:
            if self.resample_with_conv:
                downsample = Sequential(
                    SamePad2d(3, 2),  # custom same padding
                    Conv2d(curr_chans, curr_chans, 3, 2),
                )
            else:
                downsample = nn.AvgPool2d(2)
            modules.append(downsample)
        return modules

    def _get_upsample_by_level(self, level):
        block = self._get_block_by_level(level)
        ch = self.hid_channels
        chs = list(map(lambda x: ch * x, self.ch_multipliers))
        next_chans = ch if level == 0 else chs[level - 1]
        prev_chans = chs[-1] if level == self.levels - 1 else chs[level + 1]
        curr_chans = chs[level]
        modules = nn.ModuleList([block(prev_chans + curr_chans, curr_chans)])
        for _ in range(self.num_res_blocks - 1):
            modules.append(block(2 * curr_chans, curr_chans))
        modules.append(block(next_chans + curr_chans, curr_chans))
        if level != 0:
            """
            Note: the official TensorFlow implementation specifies `align_corners=True`
            However, PyTorch does not support align_corners for nearest interpolation
            to see the difference, run the following example:
            ---------------------------------------------------------------------------
            import numpy as np
            import torch
            import tensorflow as tf

            x = np.arange(9.).reshape(3, 3)
            print(torch.nn.functional.interpolate(torch.as_tensor(x).reshape(1, 1, 3, 3), size=7, mode="nearest"))  # asymmetric
            print(tf.squeeze(tf.compat.v1.image.resize(tf.reshape(tf.convert_to_tensor(x), shape=(3, 3, 1)), size=(7, 7), method="nearest", align_corners=True)))  # symmetric
            ---------------------------------------------------------------------------
            """  # noqa
            upsample = [nn.Upsample(scale_factor=2, mode="nearest")]
            if self.resample_with_conv:
                upsample.append(Conv2d(curr_chans, curr_chans, 3, 1, 1))
            modules.append(Sequential(*upsample))
        return modules

    def forward(self, x, t):
        t_emb = get_timestep_embedding(t, self.hid_channels)
        t_emb = self.embed(t_emb)

        # downsample
        hs = [self.in_conv(x)]
        for i in range(self.levels):
            downsample = self.downsamples[f"level_{i}"]
            for j, layer in enumerate(downsample):  # noqa
                h = hs[-1]
                if j != self.num_res_blocks:
                    hs.append(layer(h, t_emb=t_emb))
                else:
                    hs.append(layer(h))

        # middle
        h = self.middle(hs[-1], t_emb=t_emb)

        # upsample
        for i in range(self.levels - 1, -1, -1):
            upsample = self.upsamples[f"level_{i}"]
            for j, layer in enumerate(upsample):  # noqa
                if j != self.num_res_blocks + 1:
                    h = layer(torch.cat([h, hs.pop()], dim=1), t_emb=t_emb)
                else:
                    h = layer(h)

        h = self.out_conv(h)
        return h
