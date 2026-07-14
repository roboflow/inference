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


# --- tensor-native sibling ---------------------------------------------------
# Parity contract: a tensor-born image through the v1_tensor block must produce
# the same pixels as the numpy block on the equivalent BGR image; numpy-born
# images delegate to the numpy implementation without forcing materialization.


def _tensor_contrast_imports():
    torch = pytest.importorskip("torch")
    from inference.core.workflows.core_steps.classical_cv.contrast_enhancement.v1_tensor import (
        ContrastEnhancementBlock as TensorContrastEnhancementBlock,
    )

    return torch, TensorContrastEnhancementBlock


def _paired_images(torch, bgr: np.ndarray):
    numpy_born = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=bgr,
    )
    if bgr.ndim == 2:
        chw = torch.from_numpy(bgr.copy()).unsqueeze(0)
    else:
        chw = torch.from_numpy(bgr[:, :, ::-1].copy()).permute(2, 0, 1).contiguous()
    tensor_born = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        tensor_image=chw,
    )
    return numpy_born, tensor_born


@pytest.mark.parametrize(
    "clip_limit, contrast_multiplier, normalize_brightness, exact",
    [
        (0, 1.0, False, True),
        (0, 1.5, False, True),
        (0, 1.0, True, True),
        (3, 1.0, False, False),  # np.percentile computes in float64 -> +-1
        (3, 1.8, True, False),
    ],
)
def test_tensor_contrast_enhancement_parity_with_numpy_block(
    clip_limit, contrast_multiplier, normalize_brightness, exact
) -> None:
    # given - the same pixels as numpy-born BGR and tensor-born RGB CHW
    torch, TensorContrastEnhancementBlock = _tensor_contrast_imports()
    rng = np.random.default_rng(42)
    bgr = rng.integers(20, 200, size=(24, 32, 3), dtype=np.uint8)
    numpy_born, tensor_born = _paired_images(torch, bgr)

    # when
    numpy_result = (
        ContrastEnhancementBlock()
        .run(
            image=numpy_born,
            clip_limit=clip_limit,
            contrast_multiplier=contrast_multiplier,
            normalize_brightness=normalize_brightness,
        )["image"]
        .numpy_image
    )
    tensor_result = (
        TensorContrastEnhancementBlock()
        .run(
            image=tensor_born,
            clip_limit=clip_limit,
            contrast_multiplier=contrast_multiplier,
            normalize_brightness=normalize_brightness,
        )["image"]
        .numpy_image
    )

    # then
    if exact:
        assert np.array_equal(tensor_result, numpy_result)
    else:
        diff = np.abs(tensor_result.astype(np.int16) - numpy_result.astype(np.int16))
        assert diff.max() <= 1, f"max deviation {diff.max()} exceeds float tolerance"


def test_tensor_contrast_enhancement_grayscale_parity() -> None:
    # given - a low-contrast grayscale ramp
    torch, TensorContrastEnhancementBlock = _tensor_contrast_imports()
    gray = np.tile(np.linspace(90, 160, 32, dtype=np.uint8), (24, 1))
    numpy_born, tensor_born = _paired_images(torch, gray)

    # when
    numpy_result = (
        ContrastEnhancementBlock()
        .run(
            image=numpy_born,
            clip_limit=0,
            contrast_multiplier=1.0,
            normalize_brightness=False,
        )["image"]
        .numpy_image
    )
    tensor_result = (
        TensorContrastEnhancementBlock()
        .run(
            image=tensor_born,
            clip_limit=0,
            contrast_multiplier=1.0,
            normalize_brightness=False,
        )["image"]
        .numpy_image
    )

    # then - tensor path emits (1, H, W) whose numpy view is the (H, W) shape
    assert tensor_result.shape == numpy_result.shape == (24, 32)
    assert np.array_equal(tensor_result, numpy_result)


def test_tensor_contrast_enhancement_flat_grayscale_returns_input() -> None:
    # given - a flat histogram: nothing to stretch (numpy grayscale parity)
    torch, TensorContrastEnhancementBlock = _tensor_contrast_imports()
    tensor_born = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        tensor_image=torch.full((1, 8, 8), 127, dtype=torch.uint8),
    )

    # when
    result = TensorContrastEnhancementBlock().run(
        image=tensor_born,
        clip_limit=0,
        contrast_multiplier=1.0,
        normalize_brightness=False,
    )["image"]

    # then
    assert result is tensor_born


def test_tensor_contrast_enhancement_delegates_for_numpy_born_images() -> None:
    # given
    torch, TensorContrastEnhancementBlock = _tensor_contrast_imports()
    rng = np.random.default_rng(7)
    bgr = rng.integers(40, 180, size=(16, 16, 3), dtype=np.uint8)
    numpy_born, _ = _paired_images(torch, bgr)
    reference = (
        ContrastEnhancementBlock()
        .run(
            image=WorkflowImageData(
                parent_metadata=ImageParentMetadata(parent_id="parent"),
                numpy_image=bgr,
            ),
            clip_limit=2,
            contrast_multiplier=1.2,
            normalize_brightness=True,
        )["image"]
        .numpy_image
    )

    # when
    result = TensorContrastEnhancementBlock().run(
        image=numpy_born,
        clip_limit=2,
        contrast_multiplier=1.2,
        normalize_brightness=True,
    )["image"]

    # then - identical output via the numpy delegate, and no forced H2D
    assert np.array_equal(result.numpy_image, reference)
    assert not numpy_born.is_tensor_materialised(), "delegate must not materialise"
