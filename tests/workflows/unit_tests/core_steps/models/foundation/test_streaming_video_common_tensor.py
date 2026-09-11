"""Unit tests for tensor-native streaming video tracker adapters."""

import numpy as np
import torch

from inference.core.workflows.core_steps.models.foundation.segment_anything_common.streaming_video import (
    BoxPromptMetadata,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything_common.visual_prompt import (
    SYNTHETIC_POINT_PROMPT_CLASS_ID,
    SYNTHETIC_POINT_PROMPT_CLASS_NAME,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything_common.streaming_video_tensor import (
    extract_box_prompts_tensor,
    masks_to_instance_detections,
)
from inference.core.workflows.execution_engine.constants import CLASS_NAMES_KEY
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks


def _image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((8, 8, 3), dtype=np.uint8),
    )


def _mask() -> np.ndarray:
    masks = np.zeros((1, 8, 8), dtype=bool)
    masks[0, 1:4, 2:6] = True
    return masks


def test_extract_box_prompts_tensor_returns_empty_lists_for_missing_or_empty_input():
    empty = Detections(
        xyxy=torch.zeros((0, 4), dtype=torch.float32),
        class_id=torch.zeros((0,), dtype=torch.int64),
        confidence=torch.zeros((0,), dtype=torch.float32),
    )

    assert extract_box_prompts_tensor(None) == ([], [])
    assert extract_box_prompts_tensor(empty) == ([], [])


def test_extract_box_prompts_tensor_resolves_class_names_and_parent_ids():
    detections = Detections(
        xyxy=torch.tensor(
            [[1, 2, 5, 6], [10, 20, 50, 60]], dtype=torch.float32
        ),
        class_id=torch.tensor([0, 1], dtype=torch.int64),
        confidence=torch.tensor([0.9, 0.8], dtype=torch.float32),
        image_metadata={CLASS_NAMES_KEY: {0: "person", 1: "vehicle"}},
        bboxes_metadata=[
            {"detection_id": "person-0"},
            {"class": "forklift", "detection_id": "vehicle-0"},
        ],
    )

    boxes, metadata = extract_box_prompts_tensor(detections)

    assert boxes == [(1.0, 2.0, 5.0, 6.0), (10.0, 20.0, 50.0, 60.0)]
    assert metadata[0].class_name == "person"
    assert metadata[0].parent_id == "person-0"
    assert metadata[1].class_name == "forklift"
    assert metadata[1].parent_id == "vehicle-0"


def test_masks_to_instance_detections_builds_dense_predictions():
    metadata = {
        3: BoxPromptMetadata(
            class_id=4,
            class_name="vehicle",
            confidence=0.75,
            parent_id="source-0",
        )
    }

    predictions = masks_to_instance_detections(
        masks=_mask(),
        obj_ids=np.array([3], dtype=np.int64),
        image=_image(),
        obj_id_metadata=metadata,
        threshold=0.5,
        mask_representation="dense",
    )

    assert predictions.xyxy.tolist() == [[2.0, 1.0, 5.0, 3.0]]
    assert predictions.class_id.tolist() == [4]
    assert predictions.confidence.tolist() == [0.75]
    assert isinstance(predictions.mask, torch.Tensor)
    assert predictions.mask.dtype == torch.bool
    assert predictions.bboxes_metadata[0]["class"] == "vehicle"
    assert predictions.bboxes_metadata[0]["tracker_id"] == 3
    assert predictions.image_metadata[CLASS_NAMES_KEY] == {4: "vehicle"}


def test_masks_to_instance_detections_uses_configurable_fallback_metadata():
    predictions = masks_to_instance_detections(
        masks=_mask(),
        obj_ids=np.array([7], dtype=np.int64),
        image=_image(),
        obj_id_metadata={},
        threshold=0.0,
        mask_representation="dense",
        fallback_class_id=SYNTHETIC_POINT_PROMPT_CLASS_ID,
        fallback_class_name=SYNTHETIC_POINT_PROMPT_CLASS_NAME,
    )

    assert predictions.class_id.tolist() == [-1]
    assert predictions.bboxes_metadata[0]["class"] == "foreground"
    assert predictions.image_metadata[CLASS_NAMES_KEY] == {-1: "foreground"}


def test_masks_to_instance_detections_builds_rle_predictions():
    predictions = masks_to_instance_detections(
        masks=_mask(),
        obj_ids=np.array([2], dtype=np.int64),
        image=_image(),
        obj_id_metadata={
            2: BoxPromptMetadata(
                class_id=0,
                class_name="person",
                confidence=1.0,
                parent_id=None,
            )
        },
        threshold=0.0,
        mask_representation="rle",
    )

    assert isinstance(predictions.mask, InstancesRLEMasks)
    assert predictions.mask.image_size == (8, 8)
    assert len(predictions.mask.masks) == 1
