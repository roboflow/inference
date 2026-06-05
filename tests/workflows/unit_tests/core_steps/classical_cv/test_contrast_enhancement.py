import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.contrast_enhancement.v1 import (
    ContrastEnhancementBlock,
    ContrastEnhancementManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


class TestContrastEnhancementManifest:
    def test_contrast_enhancement_validation_when_valid_manifest_is_given(self):
        manifest = ContrastEnhancementManifest.model_validate(
            {
                "type": "roboflow_core/contrast_enhancement@v1",
                "name": "contrast_enhancement",
                "image": "$inputs.image",
                "clip_limit": 0,
            }
        )

        assert manifest.type == "roboflow_core/contrast_enhancement@v1"
        assert manifest.name == "contrast_enhancement"
        assert manifest.clip_limit == 0

    def test_contrast_enhancement_validation_when_invalid_image_is_given(self):
        with pytest.raises(ValidationError):
            ContrastEnhancementManifest.model_validate(
                {
                    "type": "roboflow_core/contrast_enhancement@v1",
                    "name": "contrast_enhancement",
                    "clip_limit": 0,
                }
            )

    def test_contrast_enhancement_block_with_color_image(self):
        # Create a test BGR image with low contrast
        image_array = np.ones((100, 100, 3), dtype=np.uint8) * 50
        image_array[:50, :50] = 60  # Slightly different region
        image_data = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=image_array,
        )

        block = ContrastEnhancementBlock()

        result = block.run(
            image=image_data,
            clip_limit=0,
            contrast_multiplier=1.0,
            normalize_brightness=False,
        )

        assert "image" in result
        enhanced = result["image"]
        assert enhanced.numpy_image.shape == image_array.shape
        assert enhanced.numpy_image.dtype == np.uint8
        # After enhancement, contrast should increase
        assert enhanced.numpy_image.max() > 60

    def test_contrast_enhancement_block_with_grayscale_image(self):
        # Create a test grayscale image
        image_array = np.ones((100, 100), dtype=np.uint8) * 50
        image_array[:50, :50] = 60
        image_data = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=image_array,
        )

        block = ContrastEnhancementBlock()

        result = block.run(
            image=image_data,
            clip_limit=0,
            contrast_multiplier=1.0,
            normalize_brightness=False,
        )

        assert "image" in result
        enhanced = result["image"]
        assert enhanced.numpy_image.shape == image_array.shape
        assert enhanced.numpy_image.dtype == np.uint8

    def test_contrast_enhancement_block_with_bgra_image(self):
        # Create a test BGRA image
        image_array = np.ones((100, 100, 4), dtype=np.uint8) * 50
        image_array[:50, :50, :3] = 60
        image_array[:, :, 3] = 255  # Alpha channel
        image_data = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=image_array,
        )

        block = ContrastEnhancementBlock()

        result = block.run(
            image=image_data,
            clip_limit=0,
            contrast_multiplier=1.0,
            normalize_brightness=False,
        )

        assert "image" in result
        enhanced = result["image"]
        assert enhanced.numpy_image.shape == image_array.shape
        assert enhanced.numpy_image.dtype == np.uint8
        # Alpha should be preserved
        assert np.all(enhanced.numpy_image[:, :, 3] == 255)

    def test_contrast_enhancement_block_with_clip_limit(self):
        # Create a test image with some outliers
        image_array = np.ones((100, 100), dtype=np.uint8) * 100
        image_array[0, 0] = 10  # Dark outlier
        image_array[99, 99] = 240  # Bright outlier
        image_data = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=image_array,
        )

        block = ContrastEnhancementBlock()

        result = block.run(
            image=image_data,
            clip_limit=2,
            contrast_multiplier=1.0,
            normalize_brightness=False,
        )

        assert "image" in result
        enhanced = result["image"]
        # With clipping, the range should not be stretched by outliers
        assert enhanced.numpy_image.shape == image_array.shape

    def test_contrast_enhancement_block_with_uniform_image(self):
        # Uniform images should not crash
        image_array = np.ones((100, 100), dtype=np.uint8) * 128
        image_data = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=image_array,
        )

        block = ContrastEnhancementBlock()

        result = block.run(
            image=image_data,
            clip_limit=0,
            contrast_multiplier=1.0,
            normalize_brightness=False,
        )

        assert "image" in result
        enhanced = result["image"]
        assert enhanced.numpy_image.shape == image_array.shape

    def test_contrast_enhancement_block_with_single_channel_input(self):
        # Single channel image (H, W, 1)
        image_array = np.ones((100, 100, 1), dtype=np.uint8) * 50
        image_array[:50, :50] = 60
        image_data = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=image_array,
        )

        block = ContrastEnhancementBlock()

        result = block.run(
            image=image_data,
            clip_limit=0,
            contrast_multiplier=1.0,
            normalize_brightness=False,
        )

        assert "image" in result
        enhanced = result["image"]
        assert enhanced.numpy_image.dtype == np.uint8

    def test_contrast_enhancement_manifest_outputs(self):
        outputs = ContrastEnhancementManifest.describe_outputs()

        assert len(outputs) == 1
        assert outputs[0].name == "image"
