from typing import List, Union

import numpy as np
import pytest

from inference.core.workflows.core_steps.formatters.vlm_as_classifier.v1 import (
    BlockManifest,
    VLMAsClassifierBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("image", ["$inputs.image", "$steps.some.image"])
@pytest.mark.parametrize(
    "classes", ["$inputs.classes", "$steps.some.classes", ["a", "b"]]
)
def test_block_manifest_parsing_when_input_is_valid(
    image: str, classes: Union[str, List[str]]
) -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/vlm_as_classifier@v1",
        "image": image,
        "name": "parser",
        "vlm_output": "$steps.vlm.output",
        "classes": classes,
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result == BlockManifest(
        type="roboflow_core/vlm_as_classifier@v1",
        name="parser",
        image=image,
        vlm_output="$steps.vlm.output",
        classes=classes,
    )


def test_run_when_valid_json_given_for_multi_class_classification() -> None:
    # given
    image = WorkflowImageData(
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    vlm_output = """
```json
{"class_name": "car", "confidence": "0.7"}
```
    """
    block = VLMAsClassifierBlockV1()

    # when
    result = block.run(image=image, vlm_output=vlm_output, classes=["car", "cat"])

    # then
    assert result["error_status"] is False
    assert result["predictions"]["image"] == {"width": 168, "height": 192}
    assert result["predictions"]["predictions"] == [
        {"class": "car", "class_id": 0, "confidence": 0.7},
        {"class": "cat", "class_id": 1, "confidence": 0.0},
    ]
    assert result["predictions"]["top"] == "car"
    assert abs(result["predictions"]["confidence"] - 0.7) < 1e-5
    assert result["predictions"]["parent_id"] == "parent"
    assert len(result["inference_id"]) > 0
    assert result["inference_id"] == result["predictions"]["inference_id"]


def test_run_when_valid_json_given_for_multi_class_classification_when_unknown_class_predicted() -> (
    None
):
    # given
    image = WorkflowImageData(
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    vlm_output = """
```json
{"class_name": "my_class", "confidence": "0.7"}
```
    """
    block = VLMAsClassifierBlockV1()

    # when
    result = block.run(image=image, vlm_output=vlm_output, classes=["car", "cat"])

    # then
    assert result["error_status"] is False
    assert result["predictions"]["image"] == {"width": 168, "height": 192}
    assert result["predictions"]["predictions"] == [
        {"class": "my_class", "class_id": -1, "confidence": 0.7},
        {"class": "car", "class_id": 0, "confidence": 0.0},
        {"class": "cat", "class_id": 1, "confidence": 0.0},
    ]
    assert result["predictions"]["top"] == "my_class"
    assert abs(result["predictions"]["confidence"] - 0.7) < 1e-5
    assert result["predictions"]["parent_id"] == "parent"
    assert len(result["inference_id"]) > 0
    assert result["inference_id"] == result["predictions"]["inference_id"]


def test_run_when_valid_json_given_for_multi_label_classification() -> None:
    # given
    image = WorkflowImageData(
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    vlm_output = """
    {"predicted_classes": [
        {"class": "cat", "confidence": 0.3}, {"class": "dog", "confidence": 0.6},
        {"class": "cat", "confidence": "0.7"}
    ]}
    """
    block = VLMAsClassifierBlockV1()

    # when
    result = block.run(
        image=image, vlm_output=vlm_output, classes=["car", "cat", "dog"]
    )

    # then
    assert result["error_status"] is False
    assert result["predictions"]["image"] == {"width": 168, "height": 192}
    assert result["predictions"]["predictions"] == {
        "car": {"confidence": 0.0, "class_id": 0},
        "cat": {"confidence": 0.7, "class_id": 1},
        "dog": {"confidence": 0.6, "class_id": 2},
    }
    assert set(result["predictions"]["predicted_classes"]) == {"cat", "dog"}
    assert result["predictions"]["parent_id"] == "parent"
    assert len(result["inference_id"]) > 0
    assert result["inference_id"] == result["predictions"]["inference_id"]


def test_run_when_valid_json_given_for_multi_label_classification_when_unknown_class_provided() -> (
    None
):
    # given
    image = WorkflowImageData(
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    vlm_output = """
    {"predicted_classes": [
        {"class": "my_class_1", "confidence": 0.3}, {"class": "my_class_2", "confidence": 0.6},
        {"class": "my_class_1", "confidence": 0.7}
    ]}
    """
    block = VLMAsClassifierBlockV1()

    # when
    result = block.run(
        image=image, vlm_output=vlm_output, classes=["car", "cat", "dog"]
    )

    # then
    assert result["error_status"] is False
    assert result["predictions"]["image"] == {"width": 168, "height": 192}
    assert result["predictions"]["predictions"] == {
        "car": {"confidence": 0.0, "class_id": 0},
        "cat": {"confidence": 0.0, "class_id": 1},
        "dog": {"confidence": 0.0, "class_id": 2},
        "my_class_1": {"confidence": 0.7, "class_id": -1},
        "my_class_2": {"confidence": 0.6, "class_id": -1},
    }
    assert set(result["predictions"]["predicted_classes"]) == {
        "my_class_1",
        "my_class_2",
    }
    assert result["predictions"]["parent_id"] == "parent"
    assert len(result["inference_id"]) > 0
    assert result["inference_id"] == result["predictions"]["inference_id"]


def test_run_when_valid_json_of_unknown_structure_given() -> None:
    # given
    image = WorkflowImageData(
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    block = VLMAsClassifierBlockV1()

    # when
    result = block.run(
        image=image, vlm_output='{"some": "data"}', classes=["car", "cat"]
    )

    # then
    assert result["error_status"] is True
    assert result["predictions"] is None
    assert len(result["inference_id"]) > 0


def test_run_when_invalid_json_given() -> None:
    # given
    image = WorkflowImageData(
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    block = VLMAsClassifierBlockV1()

    # when
    result = block.run(image=image, vlm_output="invalid_json", classes=["car", "cat"])

    # then
    assert result["error_status"] is True
    assert result["predictions"] is None
    assert len(result["inference_id"]) > 0


def test_run_when_multiple_jsons_given() -> None:
    # given
    raw_json = """
    {"predicted_classes": [
        {"class": "cat", "confidence": 0.3}, {"class": "dog", "confidence": 0.6},
        {"class": "cat", "confidence": "0.7"}
    ]}
    {"predicted_classes": [
        {"class": "cat", "confidence": 0.4}, {"class": "dog", "confidence": 0.7},
        {"class": "cat", "confidence": "0.8"}
    ]}
    """
    image = WorkflowImageData(
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    block = VLMAsClassifierBlockV1()

    # when
    result = block.run(image=image, vlm_output=raw_json, classes=["car", "cat"])

    # then
    assert result["error_status"] is True
    assert result["predictions"] is None
    assert len(result["inference_id"]) > 0


def test_run_when_json_in_markdown_block_given() -> None:
    # given
    raw_json = """
```json
{"predicted_classes": [
    {"class": "cat", "confidence": 0.3}, {"class": "dog", "confidence": 0.6},
    {"class": "cat", "confidence": "0.7"}
]}
```
```
        """
    image = WorkflowImageData(
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    block = VLMAsClassifierBlockV1()

    # when
    result = block.run(image=image, vlm_output=raw_json, classes=["car", "cat", "dog"])

    # then
    assert result["error_status"] is False
    assert result["predictions"]["image"] == {"width": 168, "height": 192}
    assert result["predictions"]["predictions"] == {
        "car": {"confidence": 0.0, "class_id": 0},
        "cat": {"confidence": 0.7, "class_id": 1},
        "dog": {"confidence": 0.6, "class_id": 2},
    }
    assert set(result["predictions"]["predicted_classes"]) == {"cat", "dog"}
    assert result["predictions"]["parent_id"] == "parent"
    assert len(result["inference_id"]) > 0
    assert result["inference_id"] == result["predictions"]["inference_id"]


def test_run_when_json_in_markdown_block_without_new_lines_given() -> None:
    # given
    raw_json = """
```json{"predicted_classes": [{"class": "cat", "confidence": 0.3}, {"class": "dog", "confidence": 0.6}, {"class": "cat", "confidence": "0.7"}]}```
"""
    image = WorkflowImageData(
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    block = VLMAsClassifierBlockV1()

    # when
    result = block.run(image=image, vlm_output=raw_json, classes=["car", "cat", "dog"])

    # then
    assert result["error_status"] is False
    assert result["predictions"]["image"] == {"width": 168, "height": 192}
    assert result["predictions"]["predictions"] == {
        "car": {"confidence": 0.0, "class_id": 0},
        "cat": {"confidence": 0.7, "class_id": 1},
        "dog": {"confidence": 0.6, "class_id": 2},
    }
    assert set(result["predictions"]["predicted_classes"]) == {"cat", "dog"}
    assert result["predictions"]["parent_id"] == "parent"
    assert len(result["inference_id"]) > 0
    assert result["inference_id"] == result["predictions"]["inference_id"]


def test_run_when_multiple_jsons_in_markdown_block_given() -> None:
    # given
    raw_json = """
```json
{"predicted_classes": [
    {"class": "cat", "confidence": 0.3}, {"class": "dog", "confidence": 0.6},
    {"class": "cat", "confidence": "0.7"}
]}
```
```json
{"predicted_classes": [
    {"class": "cat", "confidence": 0.4}, {"class": "dog", "confidence": 0.7},
    {"class": "cat", "confidence": "0.8"}
]}
```
"""
    image = WorkflowImageData(
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    block = VLMAsClassifierBlockV1()

    # when
    result = block.run(image=image, vlm_output=raw_json, classes=["car", "cat", "dog"])

    # then
    assert result["error_status"] is False
    assert result["predictions"]["image"] == {"width": 168, "height": 192}
    assert result["predictions"]["predictions"] == {
        "car": {"confidence": 0.0, "class_id": 0},
        "cat": {"confidence": 0.7, "class_id": 1},
        "dog": {"confidence": 0.6, "class_id": 2},
    }
    assert set(result["predictions"]["predicted_classes"]) == {"cat", "dog"}
    assert result["predictions"]["parent_id"] == "parent"
    assert len(result["inference_id"]) > 0
    assert result["inference_id"] == result["predictions"]["inference_id"]
