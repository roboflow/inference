from contextlib import nullcontext

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from inference_models.models.vjepa2_1.architecture import VJepaEncoder
from inference_models.models.vjepa2_1.head import SpanHead


def make_model():
    model = nn.Module()
    model.encoder = VJepaEncoder()
    model.head = SpanHead(frames=4, classes=2)
    with torch.no_grad():
        model.encoder.video_mod_embed.normal_(std=0.02)
        model.encoder.img_mod_embed.zero_()
        model.head.frame_queries.normal_(std=0.02)
        for layer in model.modules():
            if isinstance(layer, nn.LayerNorm):
                layer.weight.uniform_(0.8, 1.2)
                layer.bias.uniform_(-0.1, 0.1)
    return model.eval()


def test_state_layout_matches_existing_trainer_artifacts():
    model = make_model()
    expected = {
        "encoder.img_mod_embed": (1, 1, 768),
        "encoder.video_mod_embed": (1, 1, 768),
        "head.frame_queries": (1, 4, 768),
        "head.anchors": (4,),
    }
    layers = {
        "encoder.patch_embed.proj": (768, 3, 2, 16, 16),
        "encoder.patch_embed_img.proj": (768, 3, 1, 16, 16),
        "head.pooler.norm1": (768,),
        "head.pooler.norm2": (768,),
        "head.pooler.xattn.q": (768, 768),
        "head.pooler.xattn.kv": (1536, 768),
        "head.pooler.mlp.fc1": (3072, 768),
        "head.pooler.mlp.fc2": (768, 3072),
        "head.classifier": (6, 768),
    }
    for index in range(4):
        layers[f"encoder.norms_block.{index}"] = (768,)
    for index in range(12):
        for suffix, shape in {
            "norm1": (768,),
            "norm2": (768,),
            "attn.qkv": (2304, 768),
            "attn.proj": (768, 768),
            "mlp.fc1": (3072, 768),
            "mlp.fc2": (768, 3072),
        }.items():
            layers[f"encoder.blocks.{index}.{suffix}"] = shape
    for name, shape in layers.items():
        expected[f"{name}.weight"] = shape
        expected[f"{name}.bias"] = (shape[0],)
    assert {
        name: tuple(value.shape) for name, value in model.state_dict().items()
    } == expected


def reference_forward(video, weights):
    # Functional reference for the retained Meta path, independent of the
    # serving classes. Distinct norm parameters catch wrong output-layer reads.
    def linear(tokens, name):
        return F.linear(tokens, weights[f"{name}.weight"], weights[f"{name}.bias"])

    def norm(tokens, name, eps=1e-6):
        return F.layer_norm(
            tokens, (768,), weights[f"{name}.weight"], weights[f"{name}.bias"], eps
        )

    def mlp(tokens, name):
        return linear(F.gelu(linear(tokens, f"{name}.fc1")), f"{name}.fc2")

    depth, rows, columns = (
        video.shape[2] // 2,
        video.shape[3] // 16,
        video.shape[4] // 16,
    )
    coordinates = (
        torch.stack(
            torch.meshgrid(
                torch.arange(depth, device=video.device),
                torch.arange(rows, device=video.device),
                torch.arange(columns, device=video.device),
                indexing="ij",
            ),
            dim=-1,
        )
        .reshape(-1, 3)
        .float()
    )
    coordinates[:, 1] = coordinates[:, 1] * 15 / (rows - 1)
    coordinates[:, 2] = coordinates[:, 2] * 15 / (columns - 1)

    def rope(tensor):
        frequency = torch.arange(10, device=tensor.device, dtype=tensor.dtype)
        frequency /= 10.0
        frequency = 1.0 / 10000**frequency
        axes = []
        for axis in range(3):
            part = tensor[..., axis * 20 : (axis + 1) * 20]
            angles = torch.einsum("n,f->nf", coordinates[:, axis], frequency)
            sine, cosine = angles.sin(), angles.cos()
            even, odd = part[..., ::2], part[..., 1::2]
            axes.append(
                torch.stack(
                    (even * cosine - odd * sine, odd * cosine + even * sine), -1
                ).flatten(-2)
            )
        return torch.cat([*axes, tensor[..., 60:]], dim=-1)

    tokens = F.conv3d(
        video,
        weights["encoder.patch_embed.proj.weight"],
        weights["encoder.patch_embed.proj.bias"],
        stride=(2, 16, 16),
    )
    tokens = tokens.flatten(2).transpose(1, 2)
    tokens += weights["encoder.video_mod_embed"].repeat(tokens.shape[0], 1, 1)
    for index in range(12):
        prefix = f"encoder.blocks.{index}"
        queries, keys, values = (
            linear(norm(tokens, f"{prefix}.norm1"), f"{prefix}.attn.qkv")
            .reshape(video.shape[0], -1, 3, 12, 64)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        queries, keys = rope(queries), rope(keys)
        if values.dtype == torch.float16:
            queries, keys = queries.to(values.dtype), keys.to(values.dtype)
        attended = (
            F.scaled_dot_product_attention(queries, keys, values)
            .transpose(1, 2)
            .reshape_as(tokens)
        )
        tokens = tokens + linear(attended, f"{prefix}.attn.proj")
        tokens = tokens + mlp(norm(tokens, f"{prefix}.norm2"), f"{prefix}.mlp")
    features = norm(tokens, "encoder.norms_block.3")
    queries = weights["head.frame_queries"].expand(video.shape[0], -1, -1)
    projected = (
        linear(queries, "head.pooler.xattn.q")
        .reshape(video.shape[0], 4, 12, 64)
        .transpose(1, 2)
    )
    keys, values = (
        linear(norm(features, "head.pooler.norm1", eps=1e-5), "head.pooler.xattn.kv")
        .reshape(video.shape[0], -1, 2, 12, 64)
        .permute(2, 0, 3, 1, 4)
        .unbind(0)
    )
    queries = queries + F.scaled_dot_product_attention(
        projected, keys, values
    ).transpose(1, 2).reshape_as(queries)
    queries = queries + mlp(
        norm(queries, "head.pooler.norm2", eps=1e-5), "head.pooler.mlp"
    )
    output = linear(queries, "head.classifier")
    left, right = (
        output[..., 2:].float().reshape(video.shape[0], 4, 2, 2).sigmoid() * 4
    ).unbind(-1)
    anchors = weights["head.anchors"][None, :, None]
    return features, output[..., :2], torch.stack((anchors - left, anchors + right), -1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_encoder_and_head_match_reference_equations(dtype):
    if dtype != torch.float32 and not torch.cuda.is_available():
        pytest.skip("Autocast parity requires CUDA")
    elif dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("This GPU does not support BF16")
    device = "cpu" if dtype == torch.float32 else "cuda"
    torch.manual_seed(9)
    model = make_model().to(device)
    video = torch.randn(1, 3, 4, 32, 48, device=device)
    autocast = (
        nullcontext() if dtype == torch.float32 else torch.autocast("cuda", dtype=dtype)
    )
    with autocast, torch.backends.cuda.sdp_kernel():
        features = model.encoder(video)
        logits, spans = model.head(features)
        expected = reference_forward(video, model.state_dict())
    tolerance = 1e-5 if dtype == torch.float32 else 2e-2
    for actual, reference in zip((features, logits, spans), expected):
        torch.testing.assert_close(actual, reference, rtol=tolerance, atol=tolerance)
