"""``coordinates.prediction_to_frame``: native predictions into a non-root frame.

Images: root 12x16 (h x w); crop A at (4, 2) of size 8x10; crop B at (2, 1)
inside A, so B sits at (6, 3) in the root. Expected values follow
``frame_xy = local_xy * scale + offset``.
"""

import numpy as np
import pytest
import torch
from roboflow_workflows.v2.data.coordinates import (
    prediction_to_frame,
    prediction_to_root,
)
from roboflow_workflows.v2.data.frames import FRAME_RECORD_KEY, FrameRef, frame_of
from roboflow_workflows.v2.data.image import FrameMapping, ImageData

from inference_models.models.base.classification import ClassificationPrediction
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import (
    coco_rle_masks_to_numpy_mask,
    torch_mask_to_coco_rle,
)

ROOT = ImageData.from_tensor(
    torch.zeros((3, 12, 16), dtype=torch.uint8), image_id="root"
)
CROP_A = ROOT.crop((4, 2, 14, 10), image_id="a")
CROP_B = CROP_A.crop((2, 1, 8, 6), image_id="b")


def into(target, offset_xy, scale_xy=(1.0, 1.0)):
    mapping = FrameMapping(
        frame_id=target.image_id,
        frame_size_hw=target.size_hw,
        offset_xy=offset_xy,
        scale_xy=scale_xy,
    )

    return dict(mapping=mapping, frame=FrameRef.from_image(target))


def boxes_on(source, xyxy, *, mask=None, rows=None, extra=None):
    fields = dict(
        xyxy=torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4),
        class_id=torch.zeros(len(xyxy), dtype=torch.int64),
        confidence=torch.full((len(xyxy),), 0.5),
        image_metadata={**source.prediction_metadata(), **(extra or {})},
        bboxes_metadata=rows,
    )
    if mask is not None:
        return InstanceDetections(mask=mask, **fields)

    return Detections(**fields)


def as_rle(dense):
    return InstancesRLEMasks.from_coco_rle_masks(
        image_size=tuple(dense.shape[1:]),
        masks=[torch_mask_to_coco_rle(one) for one in dense],
    )


def dense_of(mask):
    if isinstance(mask, InstancesRLEMasks):
        return torch.from_numpy(coco_rle_masks_to_numpy_mask(mask).astype(bool))

    return mask


@pytest.mark.parametrize("encoding", ["dense", "rle"])
def test_paste_into_a_non_root_parent(encoding):
    local = torch.zeros((1, 5, 6), dtype=torch.bool)
    local[0, 0:2, 1:3] = True
    mask = as_rle(local) if encoding == "rle" else local
    prediction = boxes_on(CROP_B, [[1, 0, 3, 2]], mask=mask)

    moved = prediction_to_frame(prediction, **into(CROP_A, (2, 1)))

    expected = torch.zeros((1, 8, 10), dtype=torch.bool)
    expected[0, 1:3, 3:5] = True
    assert torch.equal(dense_of(moved.mask), expected)
    assert moved.xyxy.tolist() == [[3.0, 1.0, 5.0, 3.0]]
    assert frame_of(moved).image_id == "a"
    assert frame_of(moved).root.offset_xy == (4.0, 2.0)


@pytest.mark.parametrize("encoding", ["dense", "rle"])
def test_negative_offset_cuts_a_crop_window(encoding):
    full = torch.zeros((1, 12, 16), dtype=torch.bool)
    full[0, 1:4, 3:6] = True  # crosses A's top-left corner (4, 2)
    mask = as_rle(full) if encoding == "rle" else full
    prediction = boxes_on(ROOT, [[3, 1, 6, 4]], mask=mask)

    moved = prediction_to_frame(prediction, **into(CROP_A, (-4, -2)))

    expected = torch.zeros((1, 8, 10), dtype=torch.bool)
    expected[0, 0:2, 0:2] = True
    assert torch.equal(dense_of(moved.mask), expected)
    # Boxes are moved, not clipped.
    assert moved.xyxy.tolist() == [[-1.0, -1.0, 2.0, 2.0]]


def test_nested_moves_compose_with_root_conversion():
    prediction = boxes_on(CROP_B, [[1, 1, 4, 3]], rows=[{"polygon": [[1, 1], [4, 3]]}])

    in_a = prediction_to_frame(prediction, **into(CROP_A, (2, 1)))

    direct = prediction_to_root(prediction)
    via_a = prediction_to_root(in_a)
    assert direct.xyxy.tolist() == via_a.xyxy.tolist() == [[7.0, 4.0, 10.0, 6.0]]
    assert direct.bboxes_metadata == via_a.bboxes_metadata
    assert via_a.bboxes_metadata == [{"polygon": [[7, 4], [10, 6]]}]


def test_keypoint_tuples_and_scaled_mappings():
    key_points = KeyPoints(
        xy=torch.tensor([[[2.0, 4.0]]]),
        class_id=torch.tensor([0]),
        confidence=torch.ones((1, 1)),
        image_metadata=CROP_B.prediction_metadata(),
        covariance=torch.eye(2).reshape(1, 1, 2, 2),
    )
    pair = (key_points, boxes_on(CROP_B, [[0, 0, 2, 2]]))

    moved_points, moved_boxes = prediction_to_frame(
        pair, **into(CROP_A, (1, 1), scale_xy=(2.0, 0.5))
    )

    assert moved_points.xy.tolist() == [[[5.0, 3.0]]]
    assert moved_points.covariance[0, 0].tolist() == [[4.0, 0.0], [0.0, 0.25]]
    assert moved_boxes.xyxy.tolist() == [[1.0, 1.0, 5.0, 2.0]]
    assert frame_of(moved_points).image_id == "a"
    assert prediction_to_frame((key_points, None), **into(CROP_A, (2, 1)))[1] is None


def test_empty_rows_get_the_new_frame():
    prediction = boxes_on(
        CROP_B, [], mask=torch.zeros((0, 5, 6), dtype=torch.bool), rows=[]
    )

    moved = prediction_to_frame(prediction, **into(CROP_A, (2, 1)))

    assert moved.xyxy.shape == (0, 4) and moved.mask.shape == (0, 8, 10)
    assert frame_of(moved).image_id == "a"


def test_metadata_keeps_unrelated_keys_and_drops_only_host_mirrors():
    prediction = boxes_on(
        CROP_B,
        [[0, 0, 1, 1]],
        rows=[{"_host_xyxy": [0, 0, 1, 1], "tag": "t", "detection_id": "d"}],
        extra={"class_names": {0: "x"}, "model_id": "m/1"},
    )

    moved = prediction_to_frame(prediction, **into(CROP_A, (2, 1)))

    assert moved.bboxes_metadata == [{"tag": "t", "detection_id": "d"}]
    assert moved.image_metadata["class_names"] == {0: "x"}
    assert moved.image_metadata["model_id"] == "m/1"
    assert moved.image_metadata["parent_id"] == "a"
    assert moved.class_id is prediction.class_id
    assert prediction.bboxes_metadata[0]["_host_xyxy"] == [0, 0, 1, 1]


def test_typed_only_metadata_stays_typed_only():
    record_only = {FRAME_RECORD_KEY: CROP_B.prediction_metadata()[FRAME_RECORD_KEY]}
    prediction = Detections(
        xyxy=torch.zeros((1, 4)),
        class_id=torch.zeros(1, dtype=torch.int64),
        confidence=torch.ones(1),
        image_metadata=record_only,
    )

    moved = prediction_to_frame(prediction, **into(CROP_A, (2, 1)))

    assert set(moved.image_metadata) == {FRAME_RECORD_KEY}
    assert frame_of(moved).image_id == "a"


def test_mapping_must_name_the_frame_and_its_size():
    prediction = boxes_on(CROP_B, [[0, 0, 1, 1]])
    wrong_id = dict(into(CROP_A, (2, 1)), frame=FrameRef.from_image(ROOT))
    wrong_size = into(CROP_A, (2, 1))
    wrong_size["mapping"] = FrameMapping(
        frame_id="a", frame_size_hw=(9, 10), offset_xy=(2, 1)
    )

    with pytest.raises(ValueError, match="Mapping targets frame"):
        prediction_to_frame(prediction, **wrong_id)
    with pytest.raises(ValueError, match="Mapping targets frame"):
        prediction_to_frame(prediction, **wrong_size)


def test_classification_none_and_unsupported_payloads():
    classification = ClassificationPrediction(
        class_id=torch.tensor([0]), confidence=torch.ones((1, 1)), images_metadata=[{}]
    )

    assert prediction_to_frame(classification, **into(CROP_A, (0, 0))) is classification
    assert prediction_to_frame(None, **into(CROP_A, (0, 0))) is None
    with pytest.raises(TypeError, match="Cannot map"):
        prediction_to_frame(np.zeros(3), **into(CROP_A, (0, 0)))
