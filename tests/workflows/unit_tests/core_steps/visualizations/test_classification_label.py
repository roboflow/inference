import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.visualizations.classification_label.v1 import (
    ClassificationLabelManifest,
    ClassificationLabelVisualizationBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
)


@pytest.mark.parametrize(
    "type_alias", ["roboflow_core/classification_label_visualization@v1", "ClassificationLabelVisualization"]
)
@pytest.mark.parametrize("images_field_alias", ["images", "image"])
@pytest.mark.parametrize("task_type_alias", ["single-label", "multi-label"])
def test_label_validation_when_valid_manifest_is_given(
    type_alias: str, images_field_alias: str, task_type_alias: str
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "label1",
        "predictions": "$steps.classification_model.predictions",
        images_field_alias: "$inputs.image",
        "task_type": task_type_alias,
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
        task_type=task_type_alias,
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
        "task_type": "single-label",
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
        'image': {'width': 1000, 'height': 1000},
        'predictions': [
            {'class': 'cat', 'class_id': 0, 'confidence': 0.95},
            {'class': 'dog', 'class_id': 1, 'confidence': 0.85},
        ]
    }

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=np.zeros((1000, 1000, 3), dtype=np.uint8),
        ),
        predictions=predictions,
        task_type="single-label",
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



def test_classification_label_visualization_mismatched_task_types():
    block = ClassificationLabelVisualizationBlockV1()
    base_image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((1000, 1000, 3), dtype=np.uint8),
    )
    
    # Case 1: Multi-label predictions with single-label task type
    multi_label_predictions = {
        'image': {'width': 1000, 'height': 1000},
        'predicted_classes': ['cat', 'dog'],
        'predictions': {
            'cat': {'class_id': 0, 'confidence': 0.95},
            'dog': {'class_id': 1, 'confidence': 0.85}
        }
    }
    
    with pytest.raises(ValueError, match="Received multi-label predictions but task_type is set to 'single-label'"):
        block.run(
            image=base_image,
            predictions=multi_label_predictions,
            task_type="single-label",
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
    
    # Case 2: Single-label predictions with multi-label task type
    single_label_predictions = {
        'image': {'width': 1000, 'height': 1000},
        'predictions': [
            {'class': 'cat', 'class_id': 0, 'confidence': 0.95}
        ]
    }
    
    with pytest.raises(ValueError, match="Received single-label predictions but task_type is set to 'multi-label'"):
        block.run(
            image=base_image,
            predictions=single_label_predictions,
            task_type="multi-label",
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


def test_classification_label_visualization_empty_predictions():
    block = ClassificationLabelVisualizationBlockV1()
    base_image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((1000, 1000, 3), dtype=np.uint8),
    )
    
    # Empty multi-label predictions
    empty_multi_predictions = {
        'image': {'width': 1000, 'height': 1000},
        'predicted_classes': [],
        'predictions': {}
    }
    
    output = block.run(
        image=base_image,
        predictions=empty_multi_predictions,
        task_type="multi-label",
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
        'image': {'width': 1000, 'height': 1000},
        'predicted_classes': ['cat'],
        'predictions': {
            'cat': {'class_id': 0, 'confidence': 0.95}
        }
    }
    
    output = block.run(
        image=base_image,
        predictions=single_multi_predictions,
        task_type="multi-label",
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
        'image': {'width': 1000, 'height': 1000},
        'predictions': "invalid"  # Neither list nor dict
    }
    
    with pytest.raises(ValueError, match="Unknown prediction format"):
        block.run(
            image=base_image,
            predictions=invalid_predictions,
            task_type="multi-label",
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

@pytest.mark.parametrize("text_position,task_type,text_padding", [
    ("TOP_LEFT", "single-label", 1),
    ("TOP_LEFT", "single-label", 5),
    ("TOP_LEFT", "single-label", 10),
    ("TOP_LEFT", "single-label", 15),
    ("TOP_LEFT", "multi-label", 1),
    ("TOP_LEFT", "multi-label", 5),
    ("TOP_LEFT", "multi-label", 10),
    ("TOP_LEFT", "multi-label", 15),
    ("BOTTOM_RIGHT", "single-label", 1),
    ("BOTTOM_RIGHT", "single-label", 5),
    ("BOTTOM_RIGHT", "single-label", 10),
    ("BOTTOM_RIGHT", "single-label", 15),
    ("BOTTOM_RIGHT", "multi-label", 1),
    ("BOTTOM_RIGHT", "multi-label", 5),
    ("BOTTOM_RIGHT", "multi-label", 10),
    ("BOTTOM_RIGHT", "multi-label", 15),
    ("CENTER", "single-label", 1),
    ("CENTER", "single-label", 5),
    ("CENTER", "single-label", 10),
    ("CENTER", "single-label", 15),
    ("CENTER", "multi-label", 1),
    ("CENTER", "multi-label", 5),
    ("CENTER", "multi-label", 10),
    ("CENTER", "multi-label", 15),
])
def test_classification_label_visualization_position_and_type_combinations(text_position, task_type, text_padding):
    block = ClassificationLabelVisualizationBlockV1()
    base_image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((1000, 1000, 3), dtype=np.uint8),
    )
    
    predictions = {
        'single-label': {
            'image': {'width': 1000, 'height': 1000},
            'predictions': [
                {'class': 'cat', 'class_id': 0, 'confidence': 0.95},
            ]
        },
        'multi-label': {
            'image': {'width': 1000, 'height': 1000},
            'predicted_classes': ['cat', 'dog'],
            'predictions': {
                'cat': {'class_id': 0, 'confidence': 0.95},
                'dog': {'class_id': 1, 'confidence': 0.85}
            }
        }
    }
    
    output = block.run(
        image=base_image,
        predictions=predictions[task_type],
        task_type=task_type,
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



