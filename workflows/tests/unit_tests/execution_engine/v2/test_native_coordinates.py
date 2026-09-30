"""Root-coordinate conversion of native predictions (``v2/blocks/coordinates.py``).

Expected values are hand-computed from ``root_xy = local_xy / s + o`` and the
documented pixel-centre mask sampling. V1 helpers serve as translation-only
parity oracles.
"""

import copy
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import supervision as sv
import torch
from pycocotools import mask as mask_utils
from roboflow_workflows.core_steps.common.tensor_native import (
    embed_rle_masks_in_larger_canvas,
    native_detections_to_root_coordinates,
)
from roboflow_workflows.core_steps.common.utils import sv_detections_to_root_coordinates
from roboflow_workflows.execution_engine.v2.blocks.coordinates import (
    convert_prediction_output,
    prediction_to_root,
)
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.declaration import Block, Output
from roboflow_workflows.execution_engine.v2.kinds import Kind
from supervision.config import ORIENTED_BOX_COORDINATES

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import coco_rle_masks_to_numpy_mask


def frame(
    *,
    offset_xy=(0, 0),
    scale: Any = 1.0,
    local_hw=(20, 40),
    root_hw=(20, 40),
    parent_id: str = "crop",
    **extra: Any,
) -> dict:
    """V1-keyed image metadata of a local frame inside a root frame."""
    metadata = {
        "class_names": {0: "a", 1: "b"},
        "prediction_type": "object-detection",
        "image_dimensions": list(local_hw),
        "parent_id": parent_id,
        "parent_coordinates": list(offset_xy),
        "parent_dimensions": list(root_hw),
        "root_parent_id": "root",
        "root_parent_coordinates": list(offset_xy),
        "root_parent_dimensions": list(root_hw),
        "scaling_relative_to_parent": scale,
        "scaling_relative_to_root_parent": scale,
    }
    metadata.update(extra)

    return metadata


def detections(
    xyxy: List[List[float]],
    image_metadata: Optional[dict],
    *,
    bboxes_metadata: Optional[List[dict]] = None,
    dtype: torch.dtype = torch.float32,
) -> Detections:
    rows = len(xyxy)
    prediction = Detections(
        xyxy=torch.tensor(xyxy, dtype=dtype).reshape(rows, 4),
        class_id=torch.arange(rows, dtype=torch.long) % 2,
        confidence=torch.linspace(0.5, 0.9, rows),
        image_metadata=image_metadata,
        bboxes_metadata=bboxes_metadata,
    )

    return prediction


def instances(
    masks: np.ndarray, image_metadata: dict, *, rle: bool
) -> InstanceDetections:
    """One row per mask; boxes are the tight mask bounds (exclusive max)."""
    boxes = []
    for mask in masks:
        ys, xs = np.nonzero(mask)
        boxes.append(
            [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
            if xs.size
            else [0, 0, 0, 0]
        )
    stored_masks = encode_rle(masks) if rle else torch.from_numpy(masks.astype(bool))
    prediction = InstanceDetections(
        xyxy=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        class_id=torch.zeros(len(masks), dtype=torch.long),
        confidence=torch.full((len(masks),), 0.8),
        mask=stored_masks,
        image_metadata=image_metadata,
        bboxes_metadata=[{"detection_id": f"d{row}"} for row in range(len(masks))],
    )

    return prediction


def encode_rle(masks: np.ndarray) -> InstancesRLEMasks:
    height, width = masks.shape[1:]
    encoded = [
        mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in masks
    ]
    rle_masks = InstancesRLEMasks.from_coco_rle_masks(
        image_size=(height, width), masks=encoded
    )

    return rle_masks


def dense(mask: Any) -> np.ndarray:
    if isinstance(mask, InstancesRLEMasks):
        return coco_rle_masks_to_numpy_mask(mask)

    return mask.numpy()


def snapshot(prediction: Any) -> Any:
    return copy.deepcopy(prediction)


def assert_same_prediction(actual: Any, expected: Any) -> None:
    """Field-wise equality of predictions, tuples and nested metadata."""
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple) and len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            assert_same_prediction(actual_item, expected_item)
        return
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
        return
    if expected is None or isinstance(expected, (str, int, float, bytes, list, dict)):
        assert actual == expected
        return

    assert type(actual) is type(expected)
    for name, expected_value in vars(expected).items():
        assert_same_prediction(getattr(actual, name), expected_value)


# The DX hard case: root 400x300 (w x h); crop at (30, 70); resize by
# (0.5, 0.25); crop at (5, 3) in resized pixels. Root offset
# (30 + 5 / 0.5, 70 + 3 / 0.25) = (40, 82); local box (2, 4, 6, 8) maps to
# (2 / 0.5 + 40, 4 / 0.25 + 82, 6 / 0.5 + 40, 8 / 0.25 + 82).
NESTED = frame(
    offset_xy=(40, 82), scale=[0.5, 0.25], local_hw=(20, 40), root_hw=(300, 400)
)
NESTED_BOX_ROOT = [44.0, 98.0, 52.0, 114.0]


def test_boxes_follow_affine_equation_with_asymmetric_scale_and_offset() -> None:
    prediction = detections([[2, 4, 6, 8], [0, 0, 40, 20]], NESTED)

    converted = prediction_to_root(prediction)

    assert converted.xyxy.tolist() == [NESTED_BOX_ROOT, [40.0, 82.0, 120.0, 162.0]]
    assert converted.class_id is prediction.class_id
    assert converted.confidence is prediction.confidence


def test_root_metadata_describes_the_root_frame_and_keeps_other_keys() -> None:
    prediction = detections(
        [[2, 4, 6, 8]],
        dict(NESTED, inference_id="inference-7", parent_frame_id="resized", custom=1),
    )

    metadata = prediction_to_root(prediction).image_metadata

    assert metadata == {
        "class_names": {0: "a", 1: "b"},
        "prediction_type": "object-detection",
        "inference_id": "inference-7",
        "custom": 1,
        "image_dimensions": [300, 400],
        "parent_id": "root",
        "parent_frame_id": "root",
        "parent_coordinates": [0, 0],
        "parent_dimensions": [300, 400],
        "root_parent_id": "root",
        "root_parent_coordinates": [0, 0],
        "root_parent_dimensions": [300, 400],
        "scaling_relative_to_parent": 1.0,
        "scaling_relative_to_root_parent": 1.0,
    }, "a root image is its own parent frame, as in ImageData"


def test_isotropic_scale_matches_v1_numpy_and_corrects_v1_native() -> None:
    metadata = frame(offset_xy=(7, 3), scale=2.0, local_hw=(8, 12), root_hw=(20, 40))
    local_mask = np.zeros((1, 8, 12), dtype=bool)
    local_mask[:, 2:6, 2:6] = True
    prediction = instances(local_mask, metadata, rle=False)
    numpy_prediction = sv.Detections(
        xyxy=prediction.xyxy.numpy().copy(),
        mask=local_mask.copy(),
        class_id=np.array([0]),
        data={
            key: np.array([value])
            for key, value in metadata.items()
            if key not in ("class_names", "prediction_type")
        },
    )

    converted = prediction_to_root(prediction)
    numpy_root = sv_detections_to_root_coordinates(numpy_prediction)
    native_v1_root = native_detections_to_root_coordinates(prediction)

    assert converted.xyxy.tolist() == [[8.0, 4.0, 10.0, 6.0]]
    assert converted.xyxy.tolist() == numpy_root.xyxy.tolist()
    assert np.array_equal(converted.mask.numpy(), numpy_root.mask)
    assert native_v1_root.xyxy.tolist() == [
        [9.0, 5.0, 13.0, 9.0]
    ], "V1 native conversion ignores scale; V2 must not reuse it for scaled frames"


def test_translation_matches_v1_native_for_every_geometry_field() -> None:
    metadata = frame(offset_xy=(7, 3), local_hw=(8, 12), root_hw=(20, 40))
    local_masks = np.zeros((2, 8, 12), dtype=bool)
    local_masks[0, 1:3, 2:5] = True
    local_masks[1, 5:8, 9:12] = True
    per_box = [
        {
            "detection_id": "d0",
            "polygon": [[2, 1], [5, 1], [5, 3]],
            "keypoints_xy": np.array([[2.5, 1.5]]),
            ORIENTED_BOX_COORDINATES: np.array([[2, 1], [5, 1], [5, 3], [2, 3]]),
        },
        {"detection_id": "d1", "polygon": [[9, 5], [12, 5], [12, 8]]},
    ]

    for rle in (False, True):
        prediction = instances(local_masks, metadata, rle=rle)
        prediction.bboxes_metadata = copy.deepcopy(per_box)

        converted = prediction_to_root(prediction)
        expected = native_detections_to_root_coordinates(snapshot(prediction))

        assert torch.equal(converted.xyxy, expected.xyxy)
        if rle:
            assert converted.mask.image_size == expected.mask.image_size
            assert converted.mask.masks == expected.mask.masks, "byte-identical RLE"
        else:
            assert torch.equal(converted.mask, expected.mask)
        for actual_entry, expected_entry in zip(
            converted.bboxes_metadata, expected.bboxes_metadata
        ):
            assert actual_entry.keys() == expected_entry.keys()
            for key in actual_entry:
                assert np.array_equal(actual_entry[key], expected_entry[key]), key
        for key in ("image_dimensions", "parent_id", "root_parent_coordinates"):
            assert converted.image_metadata[key] == expected.image_metadata[key]


def test_per_box_geometry_agrees_with_boxes_under_scale() -> None:
    corners = np.array([[2.0, 4.0], [6.0, 4.0], [6.0, 8.0], [2.0, 8.0]], np.float32)
    prediction = detections(
        [[2, 4, 6, 8]],
        NESTED,
        bboxes_metadata=[
            {
                "detection_id": "d0",
                "tracker_id": 11,
                "data": "qr payload",
                "polygon": [[2, 4], [6, 4], [6, 8], [2, 8]],
                "keypoints_xy": torch.tensor([[2.0, 4.0], [6.0, 8.0]]),
                ORIENTED_BOX_COORDINATES: corners,
                "_host_xyxy": [2.0, 4.0, 6.0, 8.0],
                "_host_class_id": 0,
                "_host_confidence": 0.5,
            }
        ],
    )

    converted = prediction_to_root(prediction)
    entry = converted.bboxes_metadata[0]

    x0, y0, x1, y1 = NESTED_BOX_ROOT
    root_corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    assert entry["polygon"] == root_corners
    assert entry["keypoints_xy"].tolist() == [[x0, y0], [x1, y1]]
    assert entry[ORIENTED_BOX_COORDINATES].tolist() == root_corners
    assert entry[ORIENTED_BOX_COORDINATES].dtype == np.float32
    assert {key: entry[key] for key in ("detection_id", "tracker_id", "data")} == {
        "detection_id": "d0",
        "tracker_id": 11,
        "data": "qr payload",
    }
    assert not any(key.startswith("_host") for key in entry), "stale mirrors dropped"
    assert list(converted)[0][4] == 11, "tracker id stays visible to row iteration"


def test_ragged_and_empty_point_payloads() -> None:
    prediction = detections(
        [[0, 0, 4, 4], [0, 0, 4, 4]],
        frame(offset_xy=(10, 20), scale=2.0, local_hw=(8, 8), root_hw=(40, 40)),
        bboxes_metadata=[
            {"polygon": [[[0, 0], [2, 0], [2, 2]], [[4, 4], [6, 6]]]},
            {"polygon": [], "keypoints_xy": np.zeros((0, 2))},
        ],
    )

    first, second = prediction_to_root(prediction).bboxes_metadata

    assert first["polygon"] == [
        [[10.0, 20.0], [11.0, 20.0], [11.0, 21.0]],
        [[12.0, 22.0], [13.0, 23.0]],
    ], "disconnected polygon parts are kept separately"
    assert second["polygon"] == []
    assert second["keypoints_xy"].shape == (0, 2)


def test_key_points_scale_positions_and_covariance() -> None:
    key_points = KeyPoints(
        xy=torch.tensor([[[2.0, 4.0], [0.0, 0.0]]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([[0.9, 0.0]]),
        image_metadata=NESTED,
        key_points_metadata=[{"skeleton": "person"}],
        covariance=torch.tensor([[[[4.0, 2.0], [2.0, 8.0]], [[1.0, 0.0], [0.0, 1.0]]]]),
        detection_confidence=torch.tensor([0.7]),
    )

    converted = prediction_to_root(key_points)

    assert converted.xy.tolist() == [
        [[44.0, 98.0], [40.0, 82.0]]
    ], "every slot uses the same map; the hidden slot stays hidden by confidence"
    # A = diag(1 / 0.5, 1 / 0.25) = diag(2, 4); A C A^T scales C_ij by a_i * a_j.
    assert converted.covariance.tolist() == [
        [[[16.0, 16.0], [16.0, 128.0]], [[4.0, 0.0], [0.0, 16.0]]]
    ]
    assert converted.confidence is key_points.confidence
    assert (converted.confidence > 0).tolist() == [[True, False]]
    assert converted.detection_confidence is key_points.detection_confidence
    assert converted.key_points_metadata == [{"skeleton": "person"}]
    assert converted.key_points_metadata[0] is not key_points.key_points_metadata[0]


def test_isotropic_covariance_matches_analytical_oracle() -> None:
    key_points = KeyPoints(
        xy=torch.tensor([[[2.0, 2.0]]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([[0.9]]),
        image_metadata=frame(offset_xy=(7, 3), scale=2.0, local_hw=(8, 12)),
        covariance=torch.tensor([[[[4.0, 2.0], [2.0, 8.0]]]]),
    )

    converted = prediction_to_root(key_points)

    assert converted.xy.tolist() == [[[8.0, 4.0]]]
    assert converted.covariance.tolist() == [[[[1.0, 0.5], [0.5, 2.0]]]]


def test_key_point_prediction_moves_both_components_together() -> None:
    key_points = KeyPoints(
        xy=torch.tensor([[[2.0, 4.0]]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([[0.9]]),
        image_metadata=NESTED,
    )
    boxes = detections([[2, 4, 6, 8]], NESTED)

    converted_points, converted_boxes = prediction_to_root((key_points, boxes))
    points_only, missing_boxes = prediction_to_root((key_points, None))

    assert converted_points.xy.tolist() == [[[44.0, 98.0]]]
    assert converted_boxes.xyxy.tolist() == [NESTED_BOX_ROOT]
    assert points_only.xy.tolist() == [[[44.0, 98.0]]]
    assert missing_boxes is None


def test_key_point_component_without_metadata_follows_its_sibling() -> None:
    key_points = KeyPoints(
        xy=torch.tensor([[[2.0, 4.0]]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([[0.9]]),
    )
    boxes = detections([[2, 4, 6, 8]], NESTED)

    converted_points, converted_boxes = prediction_to_root((key_points, boxes))

    assert converted_points.xy.tolist() == [[[44.0, 98.0]]]
    assert converted_points.image_metadata is None
    assert converted_boxes.xyxy.tolist() == [NESTED_BOX_ROOT]


def test_key_point_components_in_different_frames_are_rejected() -> None:
    key_points = KeyPoints(
        xy=torch.tensor([[[2.0, 4.0]]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([[0.9]]),
        image_metadata=NESTED,
    )
    boxes = detections([[2, 4, 6, 8]], dict(NESTED, root_parent_coordinates=[41, 82]))

    with pytest.raises(ValueError, match="different frames"):
        prediction_to_root((key_points, boxes))


def test_key_point_components_are_anchored_separately() -> None:
    # Same zero-offset map, but only the boxes come from a smaller crop.
    root_frame = frame(parent_id="root", local_hw=(20, 40), root_hw=(20, 40))
    key_points = KeyPoints(
        xy=torch.tensor([[[2.0, 4.0]]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([[0.9]]),
        image_metadata=root_frame,
    )
    boxes = detections([[2, 4, 6, 8]], dict(root_frame, image_dimensions=[10, 10]))

    converted_points, converted_boxes = prediction_to_root((key_points, boxes))
    unchanged = (key_points, detections([[2, 4, 6, 8]], root_frame))

    assert converted_points is key_points
    assert converted_boxes.image_metadata["image_dimensions"] == [20, 40]
    assert prediction_to_root(unchanged) is unchanged


def test_dense_mask_sampling_with_asymmetric_scale() -> None:
    # s = (2, 0.5), o = (3, 1): root X in 3..5 samples local columns 1, 3, 5;
    # root Y in 1..8 samples local rows 0, 0, 1, 1, 2, 2, 3, 3.
    local = np.zeros((1, 4, 6), dtype=bool)
    for row, column in ((0, 1), (2, 3), (3, 5), (1, 0)):
        local[0, row, column] = True
    metadata = frame(
        offset_xy=(3, 1), scale=[2, 0.5], local_hw=(4, 6), root_hw=(12, 10)
    )
    expected = np.zeros((1, 12, 10), dtype=bool)
    for row, column in ((1, 3), (2, 3), (5, 4), (6, 4), (7, 5), (8, 5)):
        expected[0, row, column] = True

    for rle in (False, True):
        converted = prediction_to_root(instances(local, metadata, rle=rle))

        assert np.array_equal(
            dense(converted.mask), expected
        ), f"rle={rle}: local column 0 is between sampled centres and is dropped"


def test_upsampled_rle_keeps_disconnected_semantic_regions() -> None:
    # s = 0.5: every local pixel becomes a 2x2 root block at offset (4, 2).
    local = np.zeros((1, 3, 5), dtype=bool)
    local[0, 0, 0] = True
    local[0, 2, 3:5] = True
    metadata = frame(
        offset_xy=(4, 2),
        scale=0.5,
        local_hw=(3, 5),
        root_hw=(9, 14),
        prediction_type="semantic-segmentation",
    )
    expected = np.zeros((1, 9, 14), dtype=bool)
    expected[0, 2:8, 4:14] = np.kron(local[0], np.ones((2, 2), dtype=bool))

    converted = prediction_to_root(instances(local, metadata, rle=True))

    assert isinstance(converted.mask, InstancesRLEMasks), "RLE stays compressed"
    assert converted.mask.image_size == (9, 14)
    assert np.array_equal(dense(converted.mask), expected)
    assert converted.xyxy.tolist() == [[4.0, 2.0, 14.0, 8.0]]


def test_fractional_offset_uses_the_documented_pixel_centre_rule() -> None:
    # o = (1.5, 0.5), s = 1: root X samples floor(X + 0.5 - 1.5) = X - 1,
    # root Y samples floor(Y + 0.5 - 0.5) = Y.
    local = np.array([[[True, False], [False, True]]])
    metadata = frame(offset_xy=(1.5, 0.5), local_hw=(2, 2), root_hw=(4, 4))
    expected = np.zeros((1, 4, 4), dtype=bool)
    expected[0, 0, 1] = expected[0, 1, 2] = True

    for rle in (False, True):
        converted = prediction_to_root(instances(local, metadata, rle=rle))

        assert np.array_equal(dense(converted.mask), expected)
        assert converted.xyxy.tolist() == [[1.5, 0.5, 3.5, 2.5]]


def test_frame_overhanging_the_root_by_one_pixel_is_clipped() -> None:
    local = np.ones((1, 2, 4), dtype=bool)
    metadata = frame(offset_xy=(3, 0), local_hw=(2, 4), root_hw=(2, 6))
    expected = np.zeros((1, 2, 6), dtype=bool)
    expected[0, :, 3:6] = True

    for rle in (False, True):
        converted = prediction_to_root(instances(local, metadata, rle=rle))

        assert np.array_equal(dense(converted.mask), expected)
        assert converted.xyxy.tolist() == [[3.0, 0.0, 7.0, 2.0]], "boxes not clipped"


def test_mask_below_root_resolution_can_become_empty() -> None:
    # s = 4: the only local pixel spans a quarter of a root pixel and holds
    # no root pixel centre.
    local = np.array([[[False, True]]])
    metadata = frame(offset_xy=(2, 0), scale=4.0, local_hw=(1, 2), root_hw=(3, 5))

    for rle in (False, True):
        converted = prediction_to_root(instances(local, metadata, rle=rle))

        assert dense(converted.mask).shape == (1, 3, 5)
        assert not dense(converted.mask).any()


def test_zero_offset_crop_still_moves_to_the_root_canvas() -> None:
    local = np.zeros((1, 5, 6), dtype=bool)
    local[0, 1:3, 1:4] = True
    metadata = frame(offset_xy=(0, 0), local_hw=(5, 6), root_hw=(10, 12))
    expected = np.zeros((1, 10, 12), dtype=bool)
    expected[0, 1:3, 1:4] = True

    for rle in (False, True):
        converted = prediction_to_root(instances(local, metadata, rle=rle))

        assert np.array_equal(dense(converted.mask), expected)
        assert converted.image_metadata["image_dimensions"] == [10, 12]
        assert converted.image_metadata["parent_id"] == "root"


def test_zero_offset_resize_still_rescales() -> None:
    prediction = detections(
        [[2, 4, 6, 8]], frame(offset_xy=(0, 0), scale=0.5, local_hw=(10, 20))
    )

    assert prediction_to_root(prediction).xyxy.tolist() == [[4.0, 8.0, 12.0, 16.0]]


@pytest.mark.parametrize("rle", [False, True, None])
def test_empty_predictions_still_change_frame(rle: Optional[bool]) -> None:
    if rle is None:
        prediction = detections([], NESTED)
    else:
        prediction = instances(np.zeros((0, 20, 40), dtype=bool), NESTED, rle=rle)

    converted = prediction_to_root(prediction)

    assert converted.xyxy.shape == (0, 4)
    assert converted.image_metadata["image_dimensions"] == [300, 400]
    assert converted.image_metadata["parent_id"] == "root"
    if rle is not None:
        assert dense(converted.mask).shape == (0, 300, 400)


def test_conversion_does_not_modify_the_input_or_its_siblings() -> None:
    shared_metadata = copy.deepcopy(NESTED)
    local = np.zeros((1, 20, 40), dtype=bool)
    local[0, 4:8, 2:6] = True
    prediction = instances(local, shared_metadata, rle=False)
    prediction.bboxes_metadata = [
        {"detection_id": "d0", "polygon": [[2, 4], [6, 8]], "_host_xyxy": [0, 0, 0, 0]}
    ]
    sibling = detections([[1, 1, 2, 2]], shared_metadata)
    before = snapshot(prediction)
    sibling_before = snapshot(sibling)

    converted = prediction_to_root(prediction)
    converted.image_metadata["custom"] = "changed"
    converted.bboxes_metadata[0]["detection_id"] = "changed"

    assert_same_prediction(prediction, before)
    assert_same_prediction(sibling, sibling_before)
    assert converted.image_metadata is not shared_metadata


def test_root_anchored_conversion_is_idempotent() -> None:
    local = np.zeros((1, 20, 40), dtype=bool)
    local[0, 4:8, 2:6] = True
    for rle in (False, True):
        converted = prediction_to_root(instances(local, NESTED, rle=rle))

        assert prediction_to_root(converted) is converted

    root_prediction = detections(
        [[1, 2, 3, 4]], frame(parent_id="root", local_hw=(20, 40), root_hw=(20, 40))
    )
    assert prediction_to_root(root_prediction) is root_prediction


def test_predictions_without_root_geometry_are_an_explicit_no_op() -> None:
    no_metadata = detections([[1, 2, 3, 4]], None)
    ids_only = detections([[1, 2, 3, 4]], {"parent_id": "a", "root_parent_id": "a"})

    assert prediction_to_root(no_metadata) is no_metadata
    assert prediction_to_root(ids_only) is ids_only


def test_v1_metadata_without_scale_means_unit_scale() -> None:
    metadata = frame(offset_xy=(7, 3), local_hw=(8, 12))
    del metadata["scaling_relative_to_parent"]
    del metadata["scaling_relative_to_root_parent"]

    converted = prediction_to_root(detections([[1, 2, 3, 4]], metadata))

    assert converted.xyxy.tolist() == [[8.0, 5.0, 10.0, 7.0]]


def test_detection_rows_convert_like_their_carrier() -> None:
    local = np.zeros((2, 20, 40), dtype=bool)
    local[0, 4:8, 2:6] = True
    local[1, 0:2, 0:2] = True
    for rle in (False, True):
        carrier = instances(local, NESTED, rle=rle)
        carrier.bboxes_metadata[0]["tracker_id"] = 5
        expected_rows = list(prediction_to_root(carrier))

        for row, expected_row in zip(carrier, expected_rows):
            converted = prediction_to_root(row)

            assert torch.equal(converted[0], expected_row[0])
            if rle:
                assert converted[1] == expected_row[1]
            else:
                assert torch.equal(converted[1], expected_row[1])
            assert converted[2] is row[2] and converted[3] is row[3]
            assert converted[4:] == expected_row[4:]

    bare_row = next(iter(detections([[1, 2, 3, 4]], None)))
    assert prediction_to_root(bare_row) is bare_row


def test_detection_row_keeps_its_own_tracker_field() -> None:
    carrier = detections(
        [[2, 4, 6, 8]],
        NESTED,
        bboxes_metadata=[{"detection_id": "d-1", "_host_xyxy": [2, 4, 6, 8]}],
    )
    xyxy, mask, class_id, confidence, _, data, metadata = next(iter(carrier))
    without_data_key = (xyxy, mask, class_id, confidence, 73, data, metadata)
    differing_data_key = (
        xyxy,
        mask,
        class_id,
        confidence,
        73,
        {"detection_id": "d-1", "tracker_id": 5},
        metadata,
    )
    without_data = (xyxy, mask, class_id, confidence, 73, None, metadata)

    converted = prediction_to_root(without_data_key)
    converted_differing = prediction_to_root(differing_data_key)
    converted_without_data = prediction_to_root(without_data)

    assert converted[0].tolist() == NESTED_BOX_ROOT
    assert converted[1] is None
    assert converted[2] is class_id and converted[3] is confidence
    assert converted[4] == 73, "the explicit tracker field survives conversion"
    assert converted[5] == {"detection_id": "d-1"}, "no tracker key is invented"
    assert converted[6]["parent_id"] == "root"
    assert converted_differing[4] == 73
    assert converted_differing[5] == {"detection_id": "d-1", "tracker_id": 5}
    assert converted_without_data[4] == 73
    assert converted_without_data[5] is None


def test_classification_predictions_carry_no_geometry() -> None:
    single = ClassificationPrediction(
        class_id=torch.tensor([1]),
        confidence=torch.tensor([[0.2, 0.8]]),
        images_metadata=[NESTED],
    )
    multi = MultiLabelClassificationPrediction(
        class_ids=torch.tensor([0]),
        confidence=torch.tensor([0.9, 0.1]),
        image_metadata=NESTED,
    )

    assert prediction_to_root(single) is single
    assert prediction_to_root(multi) is multi


@pytest.mark.parametrize(
    ("xyxy_dtype", "scale", "expected_dtype", "expected"),
    [
        (torch.float16, [0.5, 0.25], torch.float16, NESTED_BOX_ROOT),
        (torch.int64, 1.0, torch.int64, [42.0, 86.0, 46.0, 90.0]),
        (torch.int64, [0.5, 0.25], torch.float32, NESTED_BOX_ROOT),
    ],
)
def test_box_dtype_is_kept_when_the_map_allows_it(
    xyxy_dtype: torch.dtype, scale: Any, expected_dtype: torch.dtype, expected: list
) -> None:
    metadata = dict(NESTED, scaling_relative_to_root_parent=scale)
    prediction = detections([[2, 4, 6, 8]], metadata, dtype=xyxy_dtype)

    converted = prediction_to_root(prediction)

    assert converted.xyxy.dtype == expected_dtype
    assert converted.xyxy.float().tolist() == [expected]


ACCELERATORS = [
    device
    for device, available in (
        ("cuda", torch.cuda.is_available()),
        ("mps", torch.backends.mps.is_available()),
    )
    if available
]


@pytest.mark.parametrize("device", ["cpu", *ACCELERATORS])
def test_dense_paths_stay_on_device_without_host_copies(
    device: str, monkeypatch
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("host transfer on a native tensor path")

    local = torch.zeros((1, 20, 40), dtype=torch.bool, device=device)
    local[0, 4:8, 2:6] = True
    segmentation = InstanceDetections(
        xyxy=torch.tensor([[2.0, 4.0, 6.0, 8.0]], device=device),
        class_id=torch.tensor([0], device=device),
        confidence=torch.tensor([0.9], device=device),
        mask=local,
        image_metadata=NESTED,
    )
    key_points = KeyPoints(
        xy=torch.tensor([[[2.0, 4.0]]], device=device),
        class_id=torch.tensor([0], device=device),
        confidence=torch.tensor([[0.9]], device=device),
        image_metadata=NESTED,
        covariance=torch.eye(2, device=device).reshape(1, 1, 2, 2),
    )
    for method in ("cpu", "numpy", "tolist", "item"):
        monkeypatch.setattr(torch.Tensor, method, forbidden)

    converted_segmentation = prediction_to_root(segmentation)
    converted_key_points, _ = prediction_to_root((key_points, None))

    monkeypatch.undo()
    for tensor in (
        converted_segmentation.xyxy,
        converted_segmentation.mask,
        converted_key_points.xy,
        converted_key_points.covariance,
    ):
        assert tensor.device == local.device
    assert converted_segmentation.xyxy.cpu().tolist() == [NESTED_BOX_ROOT]
    # 4x4 local pixels, each 2 root pixels wide and 4 high.
    assert converted_segmentation.mask.sum().item() == 16 * 2 * 4
    assert converted_key_points.xy.cpu().tolist() == [[[44.0, 98.0]]]


# Narrow integer storage must not wrap: 20 + 250 exceeds uint8, 100 + 250
# exceeds int8 and 30010 + 5000 exceeds int16.
NARROW_INTEGER_CASES = {
    "uint8": (torch.uint8, [10, 2, 20, 4], (250, 5), (8, 30), (20, 400)),
    "int8": (torch.int8, [-5, 2, 100, 4], (250, 5), (8, 120), (20, 400)),
    "int16": (torch.int16, [30000, 0, 30010, 4], (5000, 5), (8, 30010), (20, 40000)),
}


@pytest.mark.parametrize("device", ["cpu", *ACCELERATORS])
@pytest.mark.parametrize("case", NARROW_INTEGER_CASES)
def test_narrow_integer_geometry_is_widened_not_wrapped(case: str, device: str) -> None:
    dtype, box, offset_xy, local_hw, root_hw = NARROW_INTEGER_CASES[case]
    metadata = frame(offset_xy=offset_xy, local_hw=local_hw, root_hw=root_hw)
    x0, y0, x1, y1 = box
    ox, oy = offset_xy
    expected_box = [x0 + ox, y0 + oy, x1 + ox, y1 + oy]
    corners = [[x0, y0], [x1, y1]]
    expected_corners = [[x0 + ox, y0 + oy], [x1 + ox, y1 + oy]]
    prediction = Detections(
        xyxy=torch.tensor([box], dtype=dtype, device=device),
        class_id=torch.tensor([0], device=device),
        confidence=torch.tensor([0.9], device=device),
        image_metadata=metadata,
        bboxes_metadata=[
            {
                "keypoints_xy": torch.tensor(corners, dtype=dtype, device=device),
                "polygon": np.array(corners, dtype=str(dtype).split(".")[-1]),
            }
        ],
    )
    key_points = KeyPoints(
        xy=torch.tensor([corners], dtype=dtype, device=device),
        class_id=torch.tensor([0], device=device),
        confidence=torch.tensor([[0.9, 0.9]], device=device),
        image_metadata=metadata,
    )

    converted = prediction_to_root(prediction)
    entry = converted.bboxes_metadata[0]
    converted_key_points = prediction_to_root(key_points)

    assert converted.xyxy.dtype == torch.int64
    assert converted.xyxy.device == prediction.xyxy.device
    assert converted.xyxy.cpu().tolist() == [expected_box]
    assert entry["keypoints_xy"].dtype == torch.int64
    assert entry["keypoints_xy"].cpu().tolist() == expected_corners
    assert entry["polygon"].dtype == np.int64
    assert entry["polygon"].tolist() == expected_corners
    assert converted_key_points.xy.dtype == torch.int64
    assert converted_key_points.xy.cpu().tolist() == [expected_corners]
    assert prediction.xyxy.dtype == dtype, "the input keeps its storage"


def test_narrow_integer_geometry_under_scale_becomes_float() -> None:
    metadata = frame(offset_xy=(250, 5), scale=0.5, local_hw=(8, 30), root_hw=(20, 400))
    prediction = detections([[10, 2, 20, 4]], metadata, dtype=torch.uint8)

    converted = prediction_to_root(prediction)

    assert converted.xyxy.dtype == torch.float32
    assert converted.xyxy.tolist() == [[270.0, 9.0, 290.0, 13.0]]


def test_rle_translation_is_byte_identical_to_the_v1_canvas_embedding() -> None:
    local = np.zeros((3, 7, 9), dtype=bool)
    local[0, 0, 0] = True
    local[1, 2:7, 5:9] = True
    local[2, 1:3, 1:3] = local[2, 5:7, 6:8] = True
    rle_masks = encode_rle(local)
    metadata = frame(offset_xy=(4, 6), local_hw=(7, 9), root_hw=(13, 13))

    converted = prediction_to_root(instances(local, metadata, rle=True))
    expected = embed_rle_masks_in_larger_canvas(
        masks=rle_masks, offset_xy=(4, 6), target_size_hw=(13, 13)
    )

    assert converted.mask.masks == expected.masks


class TestOutputOptions:
    PREDICTION = detections([[2, 4, 6, 8]], NESTED)

    def test_own_returns_the_local_payload(self) -> None:
        converted = convert_prediction_output(
            self.PREDICTION, {"coordinates_system": "own"}
        )

        assert converted is self.PREDICTION

    @pytest.mark.parametrize(
        "options",
        [{}, {"coordinates_system": "parent"}, {"coordinates_system": "root"}],
    )
    def test_parent_root_and_missing_mean_the_workflow_root(
        self, options: Dict[str, str]
    ) -> None:
        converted = convert_prediction_output(self.PREDICTION, options)

        assert converted.xyxy.tolist() == [NESTED_BOX_ROOT]

    @pytest.mark.parametrize("value", ["immediate_parent", "PARENT", None, 1])
    def test_unknown_coordinate_systems_are_rejected(self, value: Any) -> None:
        with pytest.raises(ValueError, match="Unknown coordinates_system"):
            convert_prediction_output(self.PREDICTION, {"coordinates_system": value})

    def test_options_must_be_a_mapping(self) -> None:
        with pytest.raises(TypeError, match="mapping"):
            convert_prediction_output(self.PREDICTION, None)

    def test_unsupported_payloads_are_rejected(self) -> None:
        with pytest.raises(TypeError, match="Cannot convert Tensor"):
            prediction_to_root(torch.zeros(3))


def replace(metadata: dict, **changes: Any) -> dict:
    changed = dict(metadata)
    for key, value in changes.items():
        if value is DELETE:
            changed.pop(key)
        else:
            changed[key] = value

    return changed


DELETE = object()
MALFORMED = {
    "negative offset": (dict(root_parent_coordinates=[-1, 0]), "outside the root"),
    "offset beyond root": (dict(root_parent_coordinates=[401, 0]), "outside the root"),
    "offset not a pair": (dict(root_parent_coordinates=[1]), r"\[x, y\] pair"),
    "offset not numeric": (dict(root_parent_coordinates=["1", 2]), "real numbers"),
    "offset nan": (dict(root_parent_coordinates=[float("nan"), 2]), "finite"),
    "zero scale": (dict(scaling_relative_to_root_parent=0), "positive"),
    "negative scale": (dict(scaling_relative_to_root_parent=[0.5, -1]), "positive"),
    "infinite scale": (dict(scaling_relative_to_root_parent=float("inf")), "finite"),
    "bool scale": (dict(scaling_relative_to_root_parent=True), "real numbers"),
    "scale triple": (dict(scaling_relative_to_root_parent=[1, 1, 1]), "pair"),
    "fractional dims": (dict(root_parent_dimensions=[300.5, 400]), "positive integers"),
    "zero dims": (dict(image_dimensions=[0, 40]), "positive integers"),
    "missing root dims": (
        dict(root_parent_dimensions=DELETE),
        "root_parent_dimensions",
    ),
    "missing root offset": (
        dict(root_parent_coordinates=DELETE),
        "root_parent_coordinates",
    ),
    "missing root id": (dict(root_parent_id=DELETE), "root_parent_id"),
    "frame exceeds root": (dict(image_dimensions=[60, 40]), "outside the root frame"),
    "mosaic": (dict(is_composite=True), "composite"),
    "empty mosaic": (dict(composite_sources=[]), "composite"),
}


@pytest.mark.parametrize("case", MALFORMED)
def test_malformed_frame_metadata_is_rejected(case: str) -> None:
    changes, message = MALFORMED[case]
    prediction = detections([[2, 4, 6, 8]], replace(NESTED, **changes))

    with pytest.raises(ValueError, match=message):
        prediction_to_root(prediction)


def test_masks_must_agree_with_image_dimensions_and_rows() -> None:
    local = np.zeros((1, 10, 40), dtype=bool)
    mismatched_canvas = instances(local, NESTED, rle=False)
    missing_mask_row = instances(np.zeros((1, 20, 40), dtype=bool), NESTED, rle=True)
    missing_mask_row.xyxy = torch.zeros((2, 4))

    with pytest.raises(ValueError, match="disagrees with 'image_dimensions'"):
        prediction_to_root(mismatched_canvas)
    with pytest.raises(ValueError, match="2 boxes but 1 masks"):
        prediction_to_root(missing_mask_row)


def test_composite_predictions_keep_canvas_coordinates() -> None:
    mosaic = detections([[2, 4, 6, 8]], replace(NESTED, composite_sources=[]))

    assert convert_prediction_output(mosaic, {"coordinates_system": "own"}) is mosaic


# One compiled workflow: the same payload read through three outputs.
DETECTION_KIND = Kind(
    name="test_native_detections",
    validate=lambda payload: isinstance(payload, Detections),
    convert_output=convert_prediction_output,
)
PRODUCED: List[Detections] = []


class EmitDetections(Block):
    type = "test/emit-native-detections@v1"
    outputs = {"predictions": Output(DETECTION_KIND)}

    def run(self) -> dict:
        prediction = detections([[2, 4, 6, 8]], copy.deepcopy(NESTED))
        PRODUCED.append(prediction)
        return {"predictions": prediction}


def test_workflow_outputs_convert_independently_per_declaration() -> None:
    definition = {
        "version": "2.0",
        "inputs": [],
        "steps": [{"type": "test/emit-native-detections@v1", "name": "model"}],
        "outputs": [
            {
                "type": "JsonField",
                "name": name,
                "selector": "$steps.model.predictions",
                **options,
            }
            for name, options in (
                ("own", {"coordinates_system": "own"}),
                ("parent", {"coordinates_system": "parent"}),
                ("default", {}),
            )
        ],
    }
    plan = compile_workflow(definition, catalogue=Catalogue([EmitDetections]))
    PRODUCED.clear()

    rows = plan.create_session().run({}).rows()

    (produced,) = PRODUCED
    (row,) = rows
    assert row["own"] is produced
    assert row["own"].xyxy.tolist() == [[2.0, 4.0, 6.0, 8.0]]
    assert row["parent"].xyxy.tolist() == [NESTED_BOX_ROOT]
    assert row["default"].xyxy.tolist() == [NESTED_BOX_ROOT]
    assert produced.image_metadata == NESTED
