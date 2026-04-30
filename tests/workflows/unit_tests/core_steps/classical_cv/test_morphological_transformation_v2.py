import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.morphological_transformation.v2 import (
    MorphologicalTransformationBlockV2,
    MorphologicalTransformationV2Manifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_morphological_transformation_v2_validation_when_valid_manifest_is_given(
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/morphological_transformation@v2",
        "name": "morph2",
        images_field_alias: "$inputs.image",
        "operation": "Opening",
    }

    # when
    result = MorphologicalTransformationV2Manifest.model_validate(data)

    # then
    assert result == MorphologicalTransformationV2Manifest(
        type="roboflow_core/morphological_transformation@v2",
        name="morph2",
        image="$inputs.image",
        operation="Opening",
    )


def test_morphological_transformation_v2_validation_when_invalid_image_is_given() -> (
    None
):
    # given
    data = {
        "type": "roboflow_core/morphological_transformation@v2",
        "name": "morph2",
        "image": "invalid",
        "operation": "Closing",
    }

    # when
    with pytest.raises(ValidationError):
        _ = MorphologicalTransformationV2Manifest.model_validate(data)


def test_morphological_transformation_v2_block_with_color_image() -> None:
    """Test v2 block preserves color format (3 channels)."""
    # given
    block = MorphologicalTransformationBlockV2()
    start_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # when
    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=start_image,
        ),
        operation="Erosion",
    )

    # then
    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")
    # Output should be color (3 channels), not grayscale (1 channel)
    assert output.get("image").numpy_image.shape == (100, 100, 3)
    # Check image is modified
    assert not np.array_equal(output.get("image").numpy_image, start_image)


def test_morphological_transformation_v2_block_with_grayscale_input() -> None:
    """Test v2 block converts grayscale to color."""
    # given
    block = MorphologicalTransformationBlockV2()
    gray_image = np.random.randint(100, 200, (100, 100), dtype=np.uint8)

    # when
    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=gray_image,
        ),
        operation="Closing",
    )

    # then
    assert output is not None
    assert "image" in output
    result_image = output.get("image").numpy_image
    # Should output color (3 channels), not single channel
    assert result_image.shape == (100, 100, 3)


def test_morphological_transformation_v2_block_with_bgra_input() -> None:
    """Test v2 block handles BGRA (4-channel) input and outputs BGR (3-channel)."""
    # given
    block = MorphologicalTransformationBlockV2()
    bgra_image = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)

    # when
    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=bgra_image,
        ),
        operation="Dilation",
    )

    # then
    assert output is not None
    result_image = output.get("image").numpy_image
    # Should preserve BGRA format (4 channels), including alpha
    assert result_image.shape == (100, 100, 4)
    assert result_image.dtype == np.uint8


def test_morphological_transformation_v2_block_opening_operation() -> None:
    """Test opening operation."""
    block = MorphologicalTransformationBlockV2()
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=image,
        ),
        operation="Opening",
        kernel_size=5,
    )

    assert output is not None
    assert output.get("image").numpy_image.shape == (100, 100, 3)


def test_morphological_transformation_v2_block_closing_operation() -> None:
    """Test closing operation."""
    block = MorphologicalTransformationBlockV2()
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=image,
        ),
        operation="Closing",
        kernel_size=5,
    )

    assert output is not None
    assert output.get("image").numpy_image.shape == (100, 100, 3)


def test_morphological_transformation_v2_block_opening_then_closing_operation() -> None:
    """Test new opening+closing operation (specialized for edge refinement preprocessing)."""
    block = MorphologicalTransformationBlockV2()
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=image,
        ),
        operation="Opening then Closing",
        kernel_size=5,
    )

    assert output is not None
    assert output.get("image").numpy_image.shape == (100, 100, 3)
    # Image should be modified
    assert not np.array_equal(output.get("image").numpy_image, image)


def test_morphological_transformation_v2_block_gradient_operation() -> None:
    """Test gradient operation."""
    block = MorphologicalTransformationBlockV2()
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=image,
        ),
        operation="Gradient",
        kernel_size=5,
    )

    assert output is not None
    assert output.get("image").numpy_image.shape == (100, 100, 3)


def test_morphological_transformation_v2_block_top_hat_operation() -> None:
    """Test top hat operation."""
    block = MorphologicalTransformationBlockV2()
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=image,
        ),
        operation="Top Hat",
        kernel_size=5,
    )

    assert output is not None
    assert output.get("image").numpy_image.shape == (100, 100, 3)


def test_morphological_transformation_v2_block_black_hat_operation() -> None:
    """Test black hat operation."""
    block = MorphologicalTransformationBlockV2()
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=image,
        ),
        operation="Black Hat",
        kernel_size=5,
    )

    assert output is not None
    assert output.get("image").numpy_image.shape == (100, 100, 3)


def test_morphological_transformation_v2_all_operations(dogs_image: np.ndarray) -> None:
    """Test all supported operations on a real image."""
    block = MorphologicalTransformationBlockV2()
    operations = [
        "Erosion",
        "Dilation",
        "Opening",
        "Closing",
        "Opening then Closing",
        "Gradient",
        "Top Hat",
        "Black Hat",
    ]

    for operation in operations:
        output = block.run(
            image=WorkflowImageData(
                parent_metadata=ImageParentMetadata(parent_id="some"),
                numpy_image=dogs_image,
            ),
            operation=operation,
            kernel_size=5,
        )

        assert output is not None
        result_image = output.get("image").numpy_image
        # All outputs should be color (3 channels)
        assert result_image.shape == (dogs_image.shape[0], dogs_image.shape[1], 3)


def test_morphological_transformation_v2_with_different_kernel_sizes() -> None:
    """Test different kernel sizes produce different results."""
    block = MorphologicalTransformationBlockV2()
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    output_kernel_3 = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=image,
        ),
        operation="Closing",
        kernel_size=3,
    )

    output_kernel_7 = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=image,
        ),
        operation="Closing",
        kernel_size=7,
    )

    # Results should be different for different kernel sizes
    assert not np.array_equal(
        output_kernel_3.get("image").numpy_image,
        output_kernel_7.get("image").numpy_image,
    )


def test_morphological_transformation_v2_invalid_operation() -> None:
    """Test error handling for invalid operation."""
    block = MorphologicalTransformationBlockV2()
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        block.run(
            image=WorkflowImageData(
                parent_metadata=ImageParentMetadata(parent_id="some"),
                numpy_image=image,
            ),
            operation="InvalidOperation",
        )
