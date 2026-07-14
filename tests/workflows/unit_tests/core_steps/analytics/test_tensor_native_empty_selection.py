import numpy as np
import pytest

torch = pytest.importorskip("torch")

from inference.core.workflows.core_steps.common.tensor_native import (  # noqa: E402
    take_prediction_by_mask,
)
from inference_models.models.base.instance_segmentation import (  # noqa: E402
    InstanceDetections,
)
from inference_models.models.base.object_detection import Detections  # noqa: E402
from inference_models.models.base.types import InstancesRLEMasks  # noqa: E402


def _detections() -> Detections:
    return Detections(
        xyxy=torch.tensor(
            [[0, 0, 10, 10], [10, 10, 20, 20], [20, 20, 30, 30]],
            dtype=torch.float32,
        ),
        class_id=torch.tensor([2, 3, 4], dtype=torch.int64),
        confidence=torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32),
        bboxes_metadata=[{"id": 0}, {"id": 1}, {"id": 2}],
    )


@pytest.mark.parametrize(
    "mask",
    [np.array([False, False, False]), torch.tensor([False, False, False])],
)
def test_all_false_mask_returns_empty_detection_views(mask) -> None:
    source = _detections()
    result = take_prediction_by_mask(source, mask)

    assert result.xyxy.shape == (0, 4)
    assert result.xyxy.dtype == source.xyxy.dtype
    assert result.xyxy.device == source.xyxy.device
    assert result.class_id.dtype == source.class_id.dtype
    assert result.confidence.dtype == source.confidence.dtype
    assert result.bboxes_metadata == []
    assert (
        result.xyxy.untyped_storage().data_ptr()
        == source.xyxy.untyped_storage().data_ptr()
    )


@pytest.mark.parametrize("rle", [False, True])
def test_all_false_mask_returns_empty_instance_detections(rle: bool) -> None:
    source = _detections()
    if rle:
        masks = InstancesRLEMasks(
            image_size=(30, 30),
            masks=[{"size": [30, 30], "counts": str(i)} for i in range(3)],
        )
    else:
        masks = torch.ones((3, 30, 30), dtype=torch.bool)
    instances = InstanceDetections(
        xyxy=source.xyxy,
        class_id=source.class_id,
        confidence=source.confidence,
        mask=masks,
        bboxes_metadata=source.bboxes_metadata,
    )

    result = take_prediction_by_mask(instances, np.zeros(3, dtype=bool))

    assert result.xyxy.shape == (0, 4)
    assert result.bboxes_metadata == []
    if rle:
        assert result.mask.masks == []
        assert result.mask.image_size == masks.image_size
    else:
        assert result.mask.shape == (0, 30, 30)
        assert result.mask.dtype == masks.dtype
        assert result.mask.device == masks.device


def test_identity_and_mixed_masks_preserve_selection_and_metadata_isolation() -> None:
    source = _detections()
    identity = take_prediction_by_mask(source, torch.ones(3, dtype=torch.bool))
    mixed = take_prediction_by_mask(source, np.array([True, False, True]))

    assert identity.xyxy is source.xyxy
    assert torch.equal(mixed.xyxy, source.xyxy[[0, 2]])
    assert mixed.bboxes_metadata == [{"id": 0}, {"id": 2}]
    mixed.bboxes_metadata[0]["changed"] = True
    assert "changed" not in source.bboxes_metadata[0]
