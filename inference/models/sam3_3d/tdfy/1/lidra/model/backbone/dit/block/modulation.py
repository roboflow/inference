import torch
from timm.models.vision_transformer import Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def check_if_modulator(x):
    assert (
        x.ndim == 3 and x.shape[1] == 1
    ), f"modulation only works with a conditional of one token (Bx1xD), found shape {tuple(x.shape)} instead"
    x = x[:, 0]
    return x


class Block(torch.nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            **block_kwargs,
        )
        self.norm2 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: torch.nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out adaLN modulation layers in DiT blocks:
        torch.nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        torch.nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        c = check_if_modulator(c)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalBlock(torch.nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = torch.nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.linear = torch.nn.Linear(
            hidden_size,
            out_size,
            bias=True,
        )
        self.adaLN_modulation = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out output layers:
        torch.nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        torch.nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        torch.nn.init.constant_(self.linear.weight, 0)
        torch.nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        c = check_if_modulator(c)

        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
