"""Unit tests for Falcon Perception model architecture components."""

import pytest
import torch

from inference_models.models.falcon_perception.config import FalconPerceptionConfig
from inference_models.models.falcon_perception.model import (
    AnyUpUpsampler,
    CoordinateHead,
    FalconPerceptionModel,
    FeedForward,
    GoldenGateRoPE,
    HybridAttention,
    RMSNorm,
    SegmentationProjector,
    SizeHead,
    TransformerBlock,
    build_fourier_features,
)


@pytest.fixture
def small_config():
    """Small config for fast testing."""
    return FalconPerceptionConfig(
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        ffn_hidden_dim=128,
        vocab_size=256,
        patch_size=16,
        max_image_size=128,
        coord_bins=64,
        size_bins=64,
        seg_dim=32,
        anyup_levels=2,
        anyup_hidden_dim=32,
    )


class TestBuildFourierFeatures:
    def test_output_shape(self):
        values = torch.rand(2, 5)
        features = build_fourier_features(values, dim=32)
        assert features.shape == (2, 5, 32)

    def test_deterministic(self):
        values = torch.rand(2, 5)
        f1 = build_fourier_features(values, dim=32)
        f2 = build_fourier_features(values, dim=32)
        assert torch.allclose(f1, f2)


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalization(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64) * 100
        out = norm(x)
        # RMS norm should keep values bounded
        assert out.abs().max() < 200


class TestGoldenGateRoPE:
    def test_output_shape(self, small_config):
        rope = GoldenGateRoPE(small_config)
        cos, sin = rope.forward(
            seq_len=20,
            image_h_patches=2,
            image_w_patches=3,
            num_image_tokens=6,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        head_dim = small_config.hidden_dim // small_config.num_heads
        assert cos.shape == (1, 20, head_dim)
        assert sin.shape == (1, 20, head_dim)


class TestHybridAttention:
    def test_output_shape(self, small_config):
        attn = HybridAttention(small_config)
        x = torch.randn(2, 10, small_config.hidden_dim)
        head_dim = small_config.hidden_dim // small_config.num_heads
        cos = torch.ones(1, 10, head_dim)
        sin = torch.zeros(1, 10, head_dim)
        mask = torch.ones(2, 1, 10, 10, dtype=torch.bool)
        out, kv_cache = attn(x, cos, sin, mask)
        assert out.shape == x.shape
        assert kv_cache[0].shape[2] == 10  # key cache length

    def test_kv_cache_extension(self, small_config):
        attn = HybridAttention(small_config)
        x = torch.randn(1, 5, small_config.hidden_dim)
        head_dim = small_config.hidden_dim // small_config.num_heads
        cos = torch.ones(1, 5, head_dim)
        sin = torch.zeros(1, 5, head_dim)
        mask = torch.ones(1, 1, 5, 5, dtype=torch.bool)

        _, kv_cache = attn(x, cos, sin, mask)
        assert kv_cache[0].shape[2] == 5

        # Extend with 1 more token
        x2 = torch.randn(1, 1, small_config.hidden_dim)
        cos2 = torch.ones(1, 1, head_dim)
        sin2 = torch.zeros(1, 1, head_dim)
        mask2 = torch.ones(1, 1, 1, 6, dtype=torch.bool)
        _, kv_cache2 = attn(x2, cos2, sin2, mask2, kv_cache)
        assert kv_cache2[0].shape[2] == 6


class TestTransformerBlock:
    def test_output_shape(self, small_config):
        block = TransformerBlock(small_config)
        x = torch.randn(2, 10, small_config.hidden_dim)
        head_dim = small_config.hidden_dim // small_config.num_heads
        cos = torch.ones(1, 10, head_dim)
        sin = torch.zeros(1, 10, head_dim)
        mask = torch.ones(2, 1, 10, 10, dtype=torch.bool)
        out, _ = block(x, cos, sin, mask)
        assert out.shape == x.shape


class TestCoordinateHead:
    def test_output_shape(self, small_config):
        head = CoordinateHead(small_config)
        h = torch.randn(2, small_config.hidden_dim)
        x_logits, y_logits = head(h)
        assert x_logits.shape == (2, small_config.coord_bins)
        assert y_logits.shape == (2, small_config.coord_bins)


class TestSizeHead:
    def test_output_shape(self, small_config):
        head = SizeHead(small_config)
        h = torch.randn(2, small_config.hidden_dim)
        w_logits, h_logits = head(h)
        assert w_logits.shape == (2, small_config.size_bins)
        assert h_logits.shape == (2, small_config.size_bins)


class TestSegmentationProjector:
    def test_output_shape(self, small_config):
        proj = SegmentationProjector(small_config)
        h = torch.randn(2, small_config.hidden_dim)
        out = proj(h)
        assert out.shape == (2, small_config.anyup_hidden_dim)


class TestAnyUpUpsampler:
    def test_output_shape(self, small_config):
        upsampler = AnyUpUpsampler(small_config)
        # 4 patches in height, 4 in width
        features = torch.randn(1, 16, small_config.hidden_dim)
        out = upsampler(features, h_patches=4, w_patches=4)
        # 2 levels of 2x upsampling: 4 -> 8 -> 16
        expected_h = 4 * (2 ** small_config.anyup_levels)
        expected_w = 4 * (2 ** small_config.anyup_levels)
        assert out.shape == (1, small_config.anyup_hidden_dim, expected_h, expected_w)


class TestFalconPerceptionModel:
    def test_embed_image(self, small_config):
        model = FalconPerceptionModel(small_config)
        # Image must be divisible by patch_size
        image = torch.randn(1, 3, 64, 64)
        embeds, h, w = model.embed_image(image)
        expected_patches = (64 // small_config.patch_size) ** 2
        assert embeds.shape == (1, expected_patches, small_config.hidden_dim)
        assert h == 64 // small_config.patch_size
        assert w == 64 // small_config.patch_size

    def test_embed_tokens(self, small_config):
        model = FalconPerceptionModel(small_config)
        tokens = torch.randint(0, small_config.vocab_size, (1, 10))
        embeds = model.embed_tokens(tokens)
        assert embeds.shape == (1, 10, small_config.hidden_dim)

    def test_hybrid_attention_mask(self, small_config):
        model = FalconPerceptionModel(small_config)
        mask = model.build_hybrid_attention_mask(
            batch_size=1,
            seq_len=10,
            num_image_tokens=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert mask.shape == (1, 1, 10, 10)
        # Image tokens (0-3) should attend to all positions
        assert mask[0, 0, 0, :].all()
        assert mask[0, 0, 3, :].all()
        # Text token at position 5 should attend to positions 0-5 but not 6+
        assert mask[0, 0, 5, :6].all()
        assert not mask[0, 0, 5, 6]

    def test_forward_transformer(self, small_config):
        model = FalconPerceptionModel(small_config)
        model.eval()
        hidden = torch.randn(1, 10, small_config.hidden_dim)
        head_dim = small_config.hidden_dim // small_config.num_heads
        cos = torch.ones(1, 10, head_dim)
        sin = torch.zeros(1, 10, head_dim)
        mask = torch.ones(1, 1, 10, 10, dtype=torch.bool)

        with torch.inference_mode():
            out, kv_caches = model.forward_transformer(hidden, cos, sin, mask)
        assert out.shape == (1, 10, small_config.hidden_dim)
        assert len(kv_caches) == small_config.num_layers

    def test_compute_mask(self, small_config):
        model = FalconPerceptionModel(small_config)
        seg_proj = torch.randn(1, small_config.anyup_hidden_dim)
        features = torch.randn(1, small_config.anyup_hidden_dim, 32, 32)
        mask_logits = model.compute_mask(seg_proj, features)
        assert mask_logits.shape == (1, 32, 32)
