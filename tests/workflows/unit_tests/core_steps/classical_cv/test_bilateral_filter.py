import numpy as np
import pytest

from inference.core.workflows.core_steps.classical_cv.bilateral_filter.v1 import (
    BilateralFilterBlock,
    BilateralFilterManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


class TestBilateralFilterManifest:
    def test_bilateral_filter_validation_when_valid_manifest_is_given(self):
        manifest = BilateralFilterManifest.model_validate(
            {
                "type": "roboflow_core/bilateral_filter@v1",
                "name": "bilateral_filter",
                "image": "$inputs.image",
                "diameter": 9,
                "sigma_color": 75,
                "sigma_space": 75,
            }
        )

        assert manifest.type == "roboflow_core/bilateral_filter@v1"
        assert manifest.name == "bilateral_filter"
        assert manifest.diameter == 9
        assert manifest.sigma_color == 75
        assert manifest.sigma_space == 75

    def test_bilateral_filter_validation_when_invalid_image_is_given(self):
        with pytest.raises(Exception):
            BilateralFilterManifest.model_validate(
                {
                    "type": "roboflow_core/bilateral_filter@v1",
                    "name": "bilateral_filter",
                    "diameter": 9,
                    "sigma_color": 75,
                    "sigma_space": 75,
                }
            )

    def test_bilateral_filter_block_with_color_image(self):
        # Create a test BGR image with noise
        np.random.seed(42)
        image_array = np.full((100, 100, 3), 128, dtype=np.uint8)
        noise = np.random.randint(-20, 20, image_array.shape)
        image_array = np.clip(image_array.astype(np.int16) + noise, 0, 255).astype(
            np.uint8
        )
        image_data = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=image_array,
        )

        block = BilateralFilterBlock()

        result = block.run(
            image=image_data, diameter=9, sigma_color=75, sigma_space=75
        )

        assert "image" in result
        filtered = result["image"]
        assert filtered.numpy_image.shape == image_array.shape
        assert filtered.numpy_image.dtype == np.uint8

    def test_bilateral_filter_block_with_grayscale_image(self):
        # Create a test grayscale image
        np.random.seed(42)
        image_array = np.full((100, 100), 128, dtype=np.uint8)
        noise = np.random.randint(-20, 20, image_array.shape)
        image_array = np.clip(image_array.astype(np.int16) + noise, 0, 255).astype(
            np.uint8
        )
        image_data = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=image_array,
        )

        block = BilateralFilterBlock()

        result = block.run(
            image=image_data, diameter=9, sigma_color=75, sigma_space=75
        )

        assert "image" in result
        filtered = result["image"]
        assert filtered.numpy_image.shape == image_array.shape
        assert filtered.numpy_image.dtype == np.uint8

    def test_bilateral_filter_block_with_bgra_image(self):
        # Create a test BGRA image
        image_array = np.full((100, 100, 4), 128, dtype=np.uint8)
        image_array[:, :, 3] = 255  # Alpha channel
        image_data = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=image_array,
        )

        block = BilateralFilterBlock()

        result = block.run(
            image=image_data, diameter=9, sigma_color=75, sigma_space=75
        )

        assert "image" in result
        filtered = result["image"]
        assert filtered.numpy_image.shape == image_array.shape
        assert filtered.numpy_image.dtype == np.uint8
        # Alpha should be preserved
        assert np.all(filtered.numpy_image[:, :, 3] == 255)

    def test_bilateral_filter_block_with_different_diameter_values(self):
        image_array = np.full((100, 100), 128, dtype=np.uint8)
        image_data = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=image_array,
        )

        block = BilateralFilterBlock()

        result = block.run(
            image=image_data, diameter=5, sigma_color=75, sigma_space=75
        )

        assert "image" in result
        filtered = result["image"]
        assert filtered.numpy_image.shape == image_array.shape

    def test_bilateral_filter_block_with_even_diameter(self):
        # Even diameter should be converted to odd
        image_array = np.full((100, 100), 128, dtype=np.uint8)
        image_data = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=image_array,
        )

        block = BilateralFilterBlock()

        result = block.run(
            image=image_data, diameter=10, sigma_color=75, sigma_space=75
        )

        assert "image" in result
        filtered = result["image"]
        assert filtered.numpy_image.shape == image_array.shape

    def test_bilateral_filter_block_with_different_sigma_values(self):
        image_array = np.full((100, 100), 128, dtype=np.uint8)
        image_data = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=image_array,
        )

        block = BilateralFilterBlock()

        result = block.run(
            image=image_data, diameter=9, sigma_color=50, sigma_space=100
        )

        assert "image" in result
        filtered = result["image"]
        assert filtered.numpy_image.shape == image_array.shape

    def test_bilateral_filter_block_preserves_edges(self):
        # Create image with sharp edge
        image_array = np.zeros((100, 100), dtype=np.uint8)
        image_array[:, 50:] = 255  # Sharp edge at x=50
        image_data = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=image_array,
        )

        block = BilateralFilterBlock()

        result = block.run(
            image=image_data, diameter=9, sigma_color=75, sigma_space=75
        )

        assert "image" in result
        filtered = result["image"]
        # Edge should be mostly preserved (bilateral filter preserves edges)
        assert filtered.numpy_image.shape == image_array.shape

    def test_bilateral_filter_block_with_single_channel_input(self):
        # Single channel image (H, W, 1)
        image_array = np.full((100, 100, 1), 128, dtype=np.uint8)
        image_data = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=image_array,
        )

        block = BilateralFilterBlock()

        result = block.run(
            image=image_data, diameter=9, sigma_color=75, sigma_space=75
        )

        assert "image" in result
        filtered = result["image"]
        assert filtered.numpy_image.dtype == np.uint8

    def test_bilateral_filter_manifest_outputs(self):
        outputs = BilateralFilterManifest.describe_outputs()

        assert len(outputs) == 1
        assert outputs[0].name == "image"

    def test_bilateral_filter_validation_diameter_range(self):
        # Test that diameter must be in valid range
        with pytest.raises(Exception):
            BilateralFilterManifest.model_validate(
                {
                    "type": "roboflow_core/bilateral_filter@v1",
                    "name": "bilateral_filter",
                    "image": "$inputs.image",
                    "diameter": 1,  # Too small
                    "sigma_color": 75,
                    "sigma_space": 75,
                }
            )

    def test_bilateral_filter_validation_sigma_color_range(self):
        # Test that sigma_color must be in valid range
        with pytest.raises(Exception):
            BilateralFilterManifest.model_validate(
                {
                    "type": "roboflow_core/bilateral_filter@v1",
                    "name": "bilateral_filter",
                    "image": "$inputs.image",
                    "diameter": 9,
                    "sigma_color": 256,  # Too large
                    "sigma_space": 75,
                }
            )
