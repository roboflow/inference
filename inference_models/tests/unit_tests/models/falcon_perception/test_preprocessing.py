"""Unit tests for Falcon Perception image preprocessing."""

import numpy as np
import pytest
import torch

from inference_models.models.falcon_perception.config import FalconPerceptionConfig
from inference_models.models.falcon_perception.preprocessing import (
    normalize_image,
    pad_to_patch_multiple,
    preprocess_image,
    resize_image_preserve_aspect,
)


@pytest.fixture
def config():
    return FalconPerceptionConfig()


class TestResizeImagePreserveAspect:
    def test_image_within_max_size_unchanged(self):
        """Image already within bounds should not be resized."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        resized, h, w = resize_image_preserve_aspect(image, 1024)
        assert h == 480
        assert w == 640

    def test_large_landscape_image_resized(self):
        """Landscape image exceeding max should be scaled down."""
        image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        resized, h, w = resize_image_preserve_aspect(image, 1024)
        assert max(h, w) <= 1024
        # Aspect ratio preserved: 1920/1080 = 16/9
        assert abs(w / h - 1920 / 1080) < 0.05

    def test_large_portrait_image_resized(self):
        """Portrait image exceeding max should be scaled down."""
        image = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
        resized, h, w = resize_image_preserve_aspect(image, 1024)
        assert max(h, w) <= 1024
        assert abs(h / w - 1920 / 1080) < 0.05

    def test_square_image_at_max_size(self):
        """Square image at exactly max_size should not change."""
        image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        resized, h, w = resize_image_preserve_aspect(image, 1024)
        assert h == 1024
        assert w == 1024

    def test_output_shape_matches_reported_dimensions(self):
        """Returned h, w should match actual array dimensions."""
        image = np.random.randint(0, 255, (2000, 3000, 3), dtype=np.uint8)
        resized, h, w = resize_image_preserve_aspect(image, 1024)
        assert resized.shape[0] == h
        assert resized.shape[1] == w
        assert resized.shape[2] == 3


class TestPadToPatchMultiple:
    def test_already_aligned_no_padding(self):
        """Image divisible by patch_size needs no padding."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        padded, pad_h, pad_w = pad_to_patch_multiple(image, 16)
        assert pad_h == 0
        assert pad_w == 0
        assert padded.shape == (480, 640, 3)

    def test_needs_padding(self):
        """Image not divisible by patch_size gets padded."""
        image = np.zeros((481, 641, 3), dtype=np.uint8)
        padded, pad_h, pad_w = pad_to_patch_multiple(image, 16)
        assert padded.shape[0] % 16 == 0
        assert padded.shape[1] % 16 == 0
        assert pad_h == 15  # 481 + 15 = 496 (31 * 16)
        assert pad_w == 15  # 641 + 15 = 656 (41 * 16)

    def test_padding_is_zero(self):
        """Padded region should be filled with zeros."""
        image = np.ones((17, 17, 3), dtype=np.uint8) * 128
        padded, _, _ = pad_to_patch_multiple(image, 16)
        # Padded region should be 0
        assert padded[17:, :, :].sum() == 0
        assert padded[:, 17:, :].sum() == 0
        # Original region should be preserved
        assert (padded[:17, :17, :] == 128).all()


class TestNormalizeImage:
    def test_output_is_float_tensor(self):
        """Output should be float32 tensor."""
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        tensor = normalize_image(image)
        assert tensor.dtype == torch.float32

    def test_output_shape(self):
        """Output should be (3, H, W)."""
        image = np.random.randint(0, 255, (64, 48, 3), dtype=np.uint8)
        tensor = normalize_image(image)
        assert tensor.shape == (3, 64, 48)

    def test_normalized_range(self):
        """After ImageNet normalization, values should be roughly in [-3, 3]."""
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        tensor = normalize_image(image)
        assert tensor.min() > -4.0
        assert tensor.max() < 4.0


class TestPreprocessImage:
    def test_full_pipeline(self, config):
        """Full preprocessing pipeline produces valid output."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        pixel_values, metadata = preprocess_image(image, config)

        # Check output tensor
        assert pixel_values.ndim == 3  # (3, H, W)
        assert pixel_values.shape[0] == 3
        assert pixel_values.shape[1] % config.patch_size == 0
        assert pixel_values.shape[2] % config.patch_size == 0

        # Check metadata
        assert metadata.original_height == 480
        assert metadata.original_width == 640
        assert metadata.h_patches > 0
        assert metadata.w_patches > 0

    def test_large_image_resized(self, config):
        """Large image should be resized before patchification."""
        image = np.random.randint(0, 255, (2000, 3000, 3), dtype=np.uint8)
        pixel_values, metadata = preprocess_image(image, config)

        assert metadata.original_height == 2000
        assert metadata.original_width == 3000
        assert metadata.resized_height <= config.max_image_size
        assert metadata.resized_width <= config.max_image_size

    def test_patch_count_matches_dimensions(self, config):
        """h_patches * w_patches should match spatial dimensions / patch_size."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        pixel_values, metadata = preprocess_image(image, config)

        expected_h_patches = pixel_values.shape[1] // config.patch_size
        expected_w_patches = pixel_values.shape[2] // config.patch_size
        assert metadata.h_patches == expected_h_patches
        assert metadata.w_patches == expected_w_patches
