import torch
from inspect import isfunction
from einops import rearrange, repeat


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GEGLU(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = torch.nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gate)


class FeedForward(torch.nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            torch.nn.Sequential(torch.n.Linear(dim, inner_dim), torch.nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = torch.nn.Sequential(
            project_in,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(inner_dim, dim_out),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(torch.nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.scale = dim_head**-0.5

        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, query_dim),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            mask = repeat(
                mask, "b s -> b h l s", h=h, l=q.shape[2]
            )  # q.shape[2] is the length of the query
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask
            )  # (b h) n d
        else:
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask
            )  # (b h) n d
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)


class Block(torch.nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        mlp_ratio=4,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_sa=False,
    ):
        super().__init__()

        d_head = dim // n_heads

        if not disable_sa:
            self.attn1 = CrossAttention(
                query_dim=dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
            )  # is a self-attention

        self.ff = FeedForward(dim, mult=mlp_ratio, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )  # is self-attn if context is none

        if not disable_sa:
            self.norm1 = torch.nn.LayerNorm(dim)

        self.norm2 = torch.nn.LayerNorm(dim)
        self.norm3 = torch.nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        self.disable_sa = disable_sa

    def forward(self, x, context, mask=None):
        if not self.disable_sa:
            x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context, mask=mask) + x
        x = self.ff(self.norm3(x)) + x
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

        self.initialize_weights()

    def initialize_weights(self):
        # zero-out output layers
        torch.nn.init.constant_(self.linear.weight, 0)
        torch.nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, context):
        x = self.norm_final(x)
        x = self.linear(x)
        return x
