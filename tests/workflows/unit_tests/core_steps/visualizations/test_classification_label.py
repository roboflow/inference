import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.visualizations.classification_label.v1 import (
    ClassificationLabelManifest,
    ClassificationLabelVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize(
    "type_alias", ["roboflow_core/classification_label_visualization@v1"]
)
@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_label_validation_when_valid_manifest_is_given(
    type_alias: str, images_field_alias: str
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "label1",
        "predictions": "$steps.classification_model.predictions",
        images_field_alias: "$inputs.image",
        "text": "Class",
        "text_position": "TOP_LEFT",
        "text_color": "WHITE",
        "text_scale": 1.0,
        "text_thickness": 1,
        "text_padding": 10,
        "border_radius": 0,
    }

    # when
    result = ClassificationLabelManifest.model_validate(data)

    # then
    assert result == ClassificationLabelManifest(
        type=type_alias,
        name="label1",
        images="$inputs.image",
        predictions="$steps.classification_model.predictions",
        text="Class",
        text_position="TOP_LEFT",
        text_color="WHITE",
        text_scale=1.0,
        text_thickness=1,
        text_padding=10,
        border_radius=0,
    )


def test_label_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "ClassificationLabelVisualization",
        "name": "label1",
        "images": "invalid",
        "predictions": "$steps.classification_model.predictions",
        "text": "Class",
        "text_position": "TOP_LEFT",
        "text_color": "WHITE",
        "text_scale": 1.0,
        "text_thickness": 1,
        "text_padding": 10,
        "border_radius": 0,
    }

    # when
    with pytest.raises(ValidationError):
        _ = ClassificationLabelManifest.model_validate(data)


def test_classification_label_visualization_block_single_label() -> None:
    # given
    block = ClassificationLabelVisualizationBlockV1()

    # Single-label predictions format
    predictions = {
        "image": {"width": 1000, "height": 1000},
        "predictions": [
            {"class": "cat", "class_id": 0, "confidence": 0.95},
            {"class": "dog", "class_id": 1, "confidence": 0.85},
        ],
    }

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=np.zeros((1000, 1000, 3), dtype=np.uint8),
        ),
        predictions=predictions,
        copy_image=True,
        color_palette="DEFAULT",
        palette_size=10,
        custom_colors=None,
        color_axis="CLASS",
        text="Class and Confidence",
        text_position="TOP_LEFT",
        text_color="WHITE",
        text_scale=1.0,
        text_thickness=1,
        text_padding=10,
        border_radius=0,
    )

    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")

    # dimensions of output match input
    assert output.get("image").numpy_image.shape == (1000, 1000, 3)
    # check if the image is modified
    assert not np.array_equal(
        output.get("image").numpy_image, np.zeros((1000, 1000, 3), dtype=np.uint8)
    )


def test_classification_label_visualization_different_prediction_formats():
    block = ClassificationLabelVisualizationBlockV1()
    base_image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((1000, 1000, 3), dtype=np.uint8),
    )

    # Multi-label format
    multi_label_predictions = {
        "image": {"width": 1000, "height": 1000},
        "predicted_classes": ["cat", "dog"],
        "predictions": {
            "cat": {"class_id": 0, "confidence": 0.95},
            "dog": {"class_id": 1, "confidence": 0.85},
        },
    }

    # Single-label format
    single_label_predictions = {
        "image": {"width": 1000, "height": 1000},
        "predictions": [{"class": "cat", "class_id": 0, "confidence": 0.95}],
    }

    # Test both formats work without specifying task_type
    for predictions in [multi_label_predictions, single_label_predictions]:
        output = block.run(
            image=base_image,
            predictions=predictions,
            copy_image=True,
            color_palette="DEFAULT",
            palette_size=10,
            custom_colors=None,
            color_axis="CLASS",
            text="Class and Confidence",
            text_position="TOP_LEFT",
            text_color="WHITE",
            text_scale=1.0,
            text_thickness=1,
            text_padding=10,
            border_radius=0,
        )

        assert output is not None
        assert "image" in output
        assert not np.array_equal(
            output.get("image").numpy_image, np.zeros((1000, 1000, 3), dtype=np.uint8)
        )


def test_classification_label_visualization_empty_predictions():
    block = ClassificationLabelVisualizationBlockV1()
    base_image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((1000, 1000, 3), dtype=np.uint8),
    )

    # Empty multi-label predictions
    empty_multi_predictions = {
        "image": {"width": 1000, "height": 1000},
        "predicted_classes": [],
        "predictions": {},
    }

    output = block.run(
        image=base_image,
        predictions=empty_multi_predictions,
        copy_image=True,
        color_palette="DEFAULT",
        palette_size=10,
        custom_colors=None,
        color_axis="CLASS",
        text="Class and Confidence",
        text_position="TOP_LEFT",
        text_color="WHITE",
        text_scale=1.0,
        text_thickness=1,
        text_padding=10,
        border_radius=0,
    )

    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")

    # dimensions of output match input
    assert output.get("image").numpy_image.shape == (1000, 1000, 3)
    # check if the image is modified
    assert np.array_equal(
        output.get("image").numpy_image, np.zeros((1000, 1000, 3), dtype=np.uint8)
    )


def test_classification_label_visualization_block_multi_label():
    block = ClassificationLabelVisualizationBlockV1()
    base_image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((1000, 1000, 3), dtype=np.uint8),
    )

    # Multi-label with single prediction
    single_multi_predictions = {
        "image": {"width": 1000, "height": 1000},
        "predicted_classes": ["cat"],
        "predictions": {"cat": {"class_id": 0, "confidence": 0.95}},
    }

    output = block.run(
        image=base_image,
        predictions=single_multi_predictions,
        copy_image=True,
        color_palette="DEFAULT",
        palette_size=10,
        custom_colors=None,
        color_axis="CLASS",
        text="Class and Confidence",
        text_position="TOP_LEFT",
        text_color="WHITE",
        text_scale=1.0,
        text_thickness=1,
        text_padding=10,
        border_radius=0,
    )
    # Verify output has been modified (not equal to input)
    assert not np.array_equal(
        output.get("image").numpy_image, np.zeros((1000, 1000, 3), dtype=np.uint8)
    )


def test_classification_label_visualization_invalid_predictions():
    block = ClassificationLabelVisualizationBlockV1()
    base_image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((1000, 1000, 3), dtype=np.uint8),
    )

    # Invalid prediction format
    invalid_predictions = {
        "image": {"width": 1000, "height": 1000},
        "predictions": "invalid",  # Neither list nor dict
    }

    with pytest.raises(KeyError):
        block.run(
            image=base_image,
            predictions=invalid_predictions,
            copy_image=True,
            color_palette="DEFAULT",
            palette_size=10,
            custom_colors=None,
            color_axis="CLASS",
            text="Class and Confidence",
            text_position="TOP_LEFT",
            text_color="WHITE",
            text_scale=1.0,
            text_thickness=1,
            text_padding=10,
            border_radius=0,
        )


@pytest.mark.parametrize(
    "text_position,text_padding",
    [
        ("TOP_LEFT", 1),
        ("TOP_LEFT", 10),
        ("BOTTOM_RIGHT", 1),
        ("BOTTOM_RIGHT", 10),
        ("CENTER", 1),
        ("CENTER", 10),
    ],
)
def test_classification_label_visualization_position_combinations(
    text_position, text_padding
):
    block = ClassificationLabelVisualizationBlockV1()
    base_image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((1000, 1000, 3), dtype=np.uint8),
    )

    predictions = {
        "image": {"width": 1000, "height": 1000},
        "predictions": [
            {"class": "cat", "class_id": 0, "confidence": 0.95},
        ],
    }

    output = block.run(
        image=base_image,
        predictions=predictions,
        copy_image=True,
        color_palette="DEFAULT",
        palette_size=10,
        custom_colors=None,
        color_axis="CLASS",
        text="Class and Confidence",
        text_position=text_position,
        text_color="WHITE",
        text_scale=1.0,
        text_thickness=1,
        text_padding=text_padding,
        border_radius=0,
    )

    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")

    # dimensions of output match input
    assert output.get("image").numpy_image.shape == (1000, 1000, 3)
    # check if the image is modified
    assert not np.array_equal(
        output.get("image").numpy_image, np.zeros((1000, 1000, 3), dtype=np.uint8)
    )
