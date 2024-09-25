from typing import List, Union

import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.foundation.florence2.v1 import (
    BlockManifest,
    prepare_detection_grounding_prompts,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize(
    "task",
    ["phrase-grounded-object-detection", "phrase-grounded-instance-segmentation"],
)
@pytest.mark.parametrize("image_field", ["image", "images"])
def test_florence2_manifest_for_prompt_requiring_tasks(
    task: str,
    image_field: str,
) -> None:
    # given
    manifest = {
        "type": "roboflow_core/florence_2@v1",
        "name": "model",
        image_field: "$inputs.image",
        "model_version": "florence-2-base",
        "task_type": task,
        "prompt": "my_prompt",
    }

    # when
    result = BlockManifest.model_validate(manifest)

    # then
    assert result == BlockManifest(
        type="roboflow_core/florence_2@v1",
        name="model",
        images="$inputs.image",
        model_version="florence-2-base",
        task_type=task,
        prompt="my_prompt",
    )


@pytest.mark.parametrize("task", ["open-vocabulary-object-detection"])
@pytest.mark.parametrize("image_field", ["image", "images"])
def test_florence2_manifest_for_classes_requiring_tasks(
    task: str,
    image_field: str,
) -> None:
    # given
    manifest = {
        "type": "roboflow_core/florence_2@v1",
        "name": "model",
        image_field: "$inputs.image",
        "model_version": "florence-2-base",
        "task_type": task,
        "classes": ["a", "b"],
    }

    # when
    result = BlockManifest.model_validate(manifest)

    # then
    assert result == BlockManifest(
        type="roboflow_core/florence_2@v1",
        name="model",
        images="$inputs.image",
        model_version="florence-2-base",
        task_type=task,
        classes=["a", "b"],
    )


@pytest.mark.parametrize(
    "task",
    [
        "detection-grounded-instance-segmentation",
        "detection-grounded-classification",
        "detection-grounded-caption",
        "detection-grounded-ocr",
    ],
)
@pytest.mark.parametrize("image_field", ["image", "images"])
@pytest.mark.parametrize(
    "grounding_detection",
    ["$inputs.bbox", "$steps.model.predictions", [0, 1, 2, 3], [0.0, 1.0, 0.0, 1.0]],
)
def test_florence2_manifest_for_classes_requiring_detection_grounding(
    task: str,
    image_field: str,
    grounding_detection: Union[List[int], List[float], str],
) -> None:
    # given
    manifest = {
        "type": "roboflow_core/florence_2@v1",
        "name": "model",
        image_field: "$inputs.image",
        "model_version": "florence-2-base",
        "task_type": task,
        "grounding_detection": grounding_detection,
    }

    # when
    result = BlockManifest.model_validate(manifest)

    # then
    assert result == BlockManifest(
        type="roboflow_core/florence_2@v1",
        name="model",
        images="$inputs.image",
        model_version="florence-2-base",
        task_type=task,
        grounding_detection=grounding_detection,
    )


def test_manifest_parsing_when_classes_not_given_but_should_have() -> None:
    # given
    manifest = {
        "type": "roboflow_core/florence_2@v1",
        "name": "model",
        "images": "$inputs.image",
        "model_version": "florence-2-base",
        "task_type": "open-vocabulary-object-detection",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(manifest)


def test_manifest_parsing_when_prompt_not_given_but_should_have() -> None:
    # given
    manifest = {
        "type": "roboflow_core/florence_2@v1",
        "name": "model",
        "images": "$inputs.image",
        "model_version": "florence-2-base",
        "task_type": "phrase-grounded-object-detection",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(manifest)


def test_manifest_parsing_when_detection_grounding_not_given_but_should_have() -> None:
    # given
    manifest = {
        "type": "roboflow_core/florence_2@v1",
        "name": "model",
        "images": "$inputs.image",
        "model_version": "florence-2-base",
        "task_type": "detection-grounded-instance-segmentation",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(manifest)


def test_prepare_detection_grounding_prompts_when_empty_sv_detections_given() -> None:
    # given
    image = WorkflowImageData(
        numpy_image=np.zeros((100, 200, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    detections = sv.Detections.empty()

    # when
    result = prepare_detection_grounding_prompts(
        images=Batch(content=[image], indices=[(0,)]),
        grounding_detection=Batch(content=[detections], indices=[(0,)]),
        grounding_selection_mode="most-confident",
    )

    # then
    assert result == [None]


def test_prepare_detection_grounding_prompts_when_non_empty_sv_detections_given() -> (
    None
):
    # given
    image = WorkflowImageData(
        numpy_image=np.zeros((100, 200, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    detections = sv.Detections(
        xyxy=np.array([[60, 30, 100, 50], [10, 20, 30, 40]]),
        confidence=np.array([0.7, 0.6]),
    )

    # when
    result = prepare_detection_grounding_prompts(
        images=Batch(content=[image], indices=[(0,)]),
        grounding_detection=Batch(content=[detections], indices=[(0,)]),
        grounding_selection_mode="most-confident",
    )

    # then
    assert result == ["<loc_300><loc_300><loc_500><loc_500>"]


def test_prepare_detection_grounding_prompts_when_batch_of_sv_detections_given() -> (
    None
):
    # given
    image = WorkflowImageData(
        numpy_image=np.zeros((100, 200, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    detections = sv.Detections(
        xyxy=np.array([[60, 30, 100, 50], [50, 10, 100, 40]]),
        confidence=np.array([0.7, 0.6]),
    )

    # when
    result = prepare_detection_grounding_prompts(
        images=Batch(content=[image, image], indices=[(0,), (1,)]),
        grounding_detection=Batch(
            content=[sv.Detections.empty(), detections],
            indices=[(0,), (1,)],
        ),
        grounding_selection_mode="least-confident",
    )

    # then
    assert result == [None, "<loc_250><loc_100><loc_500><loc_400>"]


def test_prepare_detection_grounding_prompts_list_of_int_given() -> None:
    # given
    image = WorkflowImageData(
        numpy_image=np.zeros((100, 200, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )

    # when
    result = prepare_detection_grounding_prompts(
        images=Batch(content=[image], indices=[(0,)]),
        grounding_detection=[60, 30, 100, 50],
        grounding_selection_mode="most-confident",
    )

    # then
    assert result == ["<loc_300><loc_300><loc_500><loc_500>"]


def test_prepare_detection_grounding_prompts_list_of_float_given() -> None:
    # given
    image = WorkflowImageData(
        numpy_image=np.zeros((100, 200, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )

    # when
    result = prepare_detection_grounding_prompts(
        images=Batch(content=[image], indices=[(0,)]),
        grounding_detection=[0.3, 0.3, 0.5, 0.5],
        grounding_selection_mode="most-confident",
    )

    # then
    assert result == ["<loc_300><loc_300><loc_500><loc_500>"]
