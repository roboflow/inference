import numpy as np
import supervision as sv
import torch

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections as TensorDetections
from inference_models.models.base.types import InstancesRLEMasks

from inference.core.workflows.core_steps.common.to_supervision import (
    detections_to_supervision_with_metadata,
    instance_detections_to_supervision_with_metadata,
    sv_detections_to_inference_models_detections,
    to_supervision_with_metadata,
)
from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    INFERENCE_ID_KEY,
    PARENT_ID_KEY,
)


def _make_tensor_detections(
    n: int = 2, image_metadata=None, bboxes_metadata=None
) -> TensorDetections:
    return TensorDetections(
        xyxy=torch.tensor([[0.0, 0.0, 10.0, 10.0]] * n, dtype=torch.float32),
        class_id=torch.tensor([0] * n, dtype=torch.int64),
        confidence=torch.tensor([0.5] * n, dtype=torch.float32),
        image_metadata=image_metadata,
        bboxes_metadata=bboxes_metadata,
    )


def test_detections_to_supervision_broadcasts_image_metadata() -> None:
    pred = _make_tensor_detections(
        n=3, image_metadata={INFERENCE_ID_KEY: "abc", "model_id": "m/1"}
    )
    sv_det = detections_to_supervision_with_metadata(pred)
    assert len(sv_det) == 3
    assert list(sv_det.data[INFERENCE_ID_KEY]) == ["abc"] * 3
    assert list(sv_det.data["model_id"]) == ["m/1"] * 3


def test_detections_bboxes_metadata_overrides_per_detection() -> None:
    pred = _make_tensor_detections(
        n=2,
        image_metadata={"model_id": "shared"},
        bboxes_metadata=[
            {DETECTION_ID_KEY: "d1", PARENT_ID_KEY: "p1"},
            {DETECTION_ID_KEY: "d2", PARENT_ID_KEY: "p2"},
        ],
    )
    sv_det = detections_to_supervision_with_metadata(pred)
    assert list(sv_det.data[DETECTION_ID_KEY]) == ["d1", "d2"]
    assert list(sv_det.data[PARENT_ID_KEY]) == ["p1", "p2"]
    # image_metadata still broadcasts in parallel
    assert list(sv_det.data["model_id"]) == ["shared", "shared"]


def test_instance_detections_rle_materialises_to_dense_at_boundary() -> None:
    # Build a valid 4x4 mask: COCO RLE encoding of all-zero mask of size 4x4
    # is "44" (16 zeros). pycocotools needs Fortran-order ravel.
    # Construct a mask manually and encode it via the pycocotools helper that
    # inference_models uses.
    from pycocotools import mask as mask_utils

    np_mask = np.zeros((4, 4), dtype=np.uint8)
    np_mask[1, 1] = 1
    rle = mask_utils.encode(np.asfortranarray(np_mask))
    pred = InstanceDetections(
        xyxy=torch.zeros((1, 4), dtype=torch.float32),
        class_id=torch.zeros((1,), dtype=torch.int64),
        confidence=torch.zeros((1,), dtype=torch.float32),
        mask=InstancesRLEMasks(image_size=(4, 4), masks=[rle["counts"]]),
    )

    sv_det = instance_detections_to_supervision_with_metadata(pred)

    assert sv_det.mask is not None
    assert sv_det.mask.shape == (1, 4, 4)
    # Recovered mask should have the pixel we set
    assert bool(sv_det.mask[0, 1, 1]) is True


def test_sv_to_inference_models_detections_roundtrip_preserves_xyxy_and_class() -> None:
    sv_det = sv.Detections(
        xyxy=np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
        class_id=np.array([0, 1]),
        confidence=np.array([0.9, 0.7]),
        data={DETECTION_ID_KEY: np.array(["d1", "d2"], dtype=object)},
    )
    tensor_det = sv_detections_to_inference_models_detections(
        sv_det, inherit_image_metadata={INFERENCE_ID_KEY: "carry"}
    )
    assert tensor_det.xyxy.shape == (2, 4)
    assert tensor_det.class_id.tolist() == [0, 1]
    assert tensor_det.image_metadata[INFERENCE_ID_KEY] == "carry"
    assert tensor_det.bboxes_metadata[0][DETECTION_ID_KEY] == "d1"


def test_to_supervision_with_metadata_passthrough_for_non_inference_models_values() -> None:
    arbitrary = {"not": "a prediction"}
    assert to_supervision_with_metadata(arbitrary) is arbitrary
    sv_det = sv.Detections.empty()
    assert to_supervision_with_metadata(sv_det) is sv_det
