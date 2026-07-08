"""Unit tests for SAM3 v3 block class_mapping feature."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.segment_anything3.v3 import (
    BlockManifest,
    SegmentAnything3BlockV3,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


def _make_detections(class_names: list[str]) -> sv.Detections:
    n = len(class_names)
    return sv.Detections(
        xyxy=np.array([[0, 0, 10, 10]] * n, dtype=np.float32),
        confidence=np.array([0.9] * n, dtype=np.float32),
        data={"class_name": np.array(class_names)},
    )


def _make_result(class_names: list[str]) -> list[dict]:
    return [{"predictions": _make_detections(class_names)}]


@pytest.fixture
def mock_workflow_image_data():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="test"),
        numpy_image=img,
    )


# --- Manifest tests ---


def test_manifest_parsing_with_class_mapping():
    """Test that BlockManifest accepts the class_mapping field."""
    data = {
        "type": "roboflow_core/sam3@v3",
        "name": "my_sam3_step",
        "images": "$inputs.image",
        "class_names": ["cat", "dog"],
        "class_mapping": {"cat": "gato", "dog": "perro"},
    }
    result = BlockManifest.model_validate(data)
    assert result.class_mapping == {"cat": "gato", "dog": "perro"}


def test_manifest_parsing_without_class_mapping():
    """Test that class_mapping is optional and defaults to None."""
    data = {
        "type": "roboflow_core/sam3@v3",
        "name": "my_sam3_step",
        "images": "$inputs.image",
        "class_names": ["cat", "dog"],
    }
    result = BlockManifest.model_validate(data)
    assert result.class_mapping is None


# --- _apply_class_mapping unit tests ---


def test_apply_class_mapping_full():
    """Test remapping all class names."""
    result = _make_result(["cat", "dog"])
    mapped = SegmentAnything3BlockV3._apply_class_mapping(
        result, {"cat": "gato", "dog": "perro"}
    )
    assert list(mapped[0]["predictions"].data["class_name"]) == ["gato", "perro"]


def test_apply_class_mapping_partial():
    """Test remapping only some class names, leaving others unchanged."""
    result = _make_result(["cat", "dog", "bird"])
    mapped = SegmentAnything3BlockV3._apply_class_mapping(result, {"cat": "gato"})
    assert list(mapped[0]["predictions"].data["class_name"]) == [
        "gato",
        "dog",
        "bird",
    ]


def test_apply_class_mapping_no_matching_keys():
    """Test that unmatched mapping keys leave predictions unchanged."""
    result = _make_result(["cat", "dog"])
    mapped = SegmentAnything3BlockV3._apply_class_mapping(result, {"fish": "pez"})
    assert list(mapped[0]["predictions"].data["class_name"]) == ["cat", "dog"]


def test_apply_class_mapping_multiple_images():
    """Test remapping across multiple images in a batch."""
    result = [
        {"predictions": _make_detections(["cat"])},
        {"predictions": _make_detections(["dog"])},
    ]
    mapped = SegmentAnything3BlockV3._apply_class_mapping(
        result, {"cat": "gato", "dog": "perro"}
    )
    assert list(mapped[0]["predictions"].data["class_name"]) == ["gato"]
    assert list(mapped[1]["predictions"].data["class_name"]) == ["perro"]


def test_apply_class_mapping_empty_result():
    """Test that an empty result list is handled gracefully."""
    result = []
    mapped = SegmentAnything3BlockV3._apply_class_mapping(result, {"cat": "gato"})
    assert mapped == []


def test_apply_class_mapping_empty_mapping():
    """Test that an empty mapping leaves predictions unchanged."""
    result = _make_result(["cat", "dog"])
    mapped = SegmentAnything3BlockV3._apply_class_mapping(result, {})
    assert list(mapped[0]["predictions"].data["class_name"]) == ["cat", "dog"]


# --- Block-level run() tests ---


@patch.object(SegmentAnything3BlockV3, "run_locally")
def test_run_with_class_mapping_remaps_predictions(
    mock_run_locally, mock_workflow_image_data
):
    """Test that block.run() applies class_mapping to predictions from run_locally."""
    mock_run_locally.return_value = _make_result(["cat", "dog"])
    block = SegmentAnything3BlockV3(
        model_manager=MagicMock(),
        api_key="test_key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        model_id="sam3/sam3_final",
        class_names=["cat", "dog"],
        confidence=0.5,
        class_mapping={"cat": "gato", "dog": "perro"},
    )

    assert list(result[0]["predictions"].data["class_name"]) == ["gato", "perro"]


@patch.object(SegmentAnything3BlockV3, "run_locally")
def test_run_without_class_mapping_leaves_predictions_unchanged(
    mock_run_locally, mock_workflow_image_data
):
    """Test that block.run() without class_mapping does not alter predictions."""
    mock_run_locally.return_value = _make_result(["cat", "dog"])
    block = SegmentAnything3BlockV3(
        model_manager=MagicMock(),
        api_key="test_key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        model_id="sam3/sam3_final",
        class_names=["cat", "dog"],
        confidence=0.5,
    )

    assert list(result[0]["predictions"].data["class_name"]) == ["cat", "dog"]


@patch.object(SegmentAnything3BlockV3, "run_locally")
def test_run_with_partial_class_mapping(mock_run_locally, mock_workflow_image_data):
    """Test that block.run() with partial class_mapping only remaps matched classes."""
    mock_run_locally.return_value = _make_result(["cat", "dog", "bird"])
    block = SegmentAnything3BlockV3(
        model_manager=MagicMock(),
        api_key="test_key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        model_id="sam3/sam3_final",
        class_names=["cat", "dog", "bird"],
        confidence=0.5,
        class_mapping={"cat": "gato"},
    )

    assert list(result[0]["predictions"].data["class_name"]) == [
        "gato",
        "dog",
        "bird",
    ]
