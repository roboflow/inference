"""Partial legacy root metadata keeps the local mask canvas (E3-COR-003).

Legacy metadata may describe only the root: ``root_parent_dimensions``,
``root_parent_coordinates``, ``root_parent_id`` and the root scale, without
``image_dimensions`` or ``parent_id``. A mask or label grid then still tells
the local canvas. A 5x10 local canvas inside a 29x47 root at offset ``[0, 0]``
and scale 1 is not already in root placement: dense and RLE masks are pasted
onto the root canvas and a semantic grid is refused. Expected pixels are
computed by hand from ``root_xy = local_xy * S + o``.
"""

import numpy as np
import pytest
import torch
from pycocotools import mask as mask_utils
from roboflow_workflows.v2.data.coordinates import (
    convert_prediction_output,
    prediction_to_root,
)

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationResult,
)
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import coco_rle_masks_to_numpy_mask

ROOT_HW = (29, 47)
LOCAL_HW = (5, 10)


def _metadata(*, offset_xy=(0, 0), local_per_root=1.0, image_hw=None):
    metadata = {
        "root_parent_dimensions": list(ROOT_HW),
        "root_parent_coordinates": list(offset_xy),
        "root_parent_id": "root",
        "scaling_relative_to_root_parent": local_per_root,
    }
    if image_hw is not None:
        metadata["image_dimensions"] = list(image_hw)
    return metadata


def _dense(canvas_hw=LOCAL_HW, **metadata_options):
    detections = InstanceDetections(
        xyxy=torch.tensor([[0.0, 0.0, 10.0, 5.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.ones(1),
        mask=torch.ones((1, *canvas_hw), dtype=torch.bool),
        image_metadata=_metadata(**metadata_options),
    )
    return detections


def _rle(canvas_hw=LOCAL_HW, **metadata_options):
    local = np.asfortranarray(np.ones(canvas_hw, dtype=np.uint8))
    detections = InstanceDetections(
        xyxy=torch.tensor([[0.0, 0.0, 10.0, 5.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.ones(1),
        mask=InstancesRLEMasks(
            image_size=canvas_hw, masks=[mask_utils.encode(local)["counts"]]
        ),
        image_metadata=_metadata(**metadata_options),
    )
    return detections


def _semantic(canvas_hw=LOCAL_HW, **metadata_options):
    semantic = SemanticSegmentationResult(
        segmentation_map=torch.zeros(canvas_hw, dtype=torch.int64),
        confidence=torch.ones(canvas_hw),
        image_metadata=_metadata(**metadata_options),
    )
    return semantic


def _expected_root_mask(rows, columns):
    expected = np.zeros(ROOT_HW, dtype=bool)
    expected[rows, columns] = True
    return expected


def test_smaller_dense_mask_is_pasted_onto_the_root_canvas():
    converted = prediction_to_root(_dense())

    assert tuple(converted.mask.shape) == (1, *ROOT_HW)
    assert np.array_equal(
        converted.mask[0].numpy(), _expected_root_mask(slice(0, 5), slice(0, 10))
    )
    assert converted.image_metadata["image_dimensions"] == list(ROOT_HW)


def test_smaller_rle_mask_is_reencoded_on_the_root_canvas():
    converted = prediction_to_root(_rle())

    assert tuple(converted.mask.image_size) == ROOT_HW
    decoded = coco_rle_masks_to_numpy_mask(converted.mask)
    assert np.array_equal(
        decoded[0].astype(bool), _expected_root_mask(slice(0, 5), slice(0, 10))
    )


def test_smaller_semantic_grid_is_refused_for_root_output():
    with pytest.raises(ValueError, match="no lossless"):
        prediction_to_root(_semantic())


def test_root_sized_canvas_is_already_root():
    dense = _dense(canvas_hw=ROOT_HW)
    rle = _rle(canvas_hw=ROOT_HW)
    semantic = _semantic(canvas_hw=ROOT_HW)

    assert prediction_to_root(dense) is dense
    assert prediction_to_root(rle) is rle
    assert prediction_to_root(semantic) is semantic


def test_canvas_one_pixel_larger_than_root_is_clipped():
    converted = prediction_to_root(_dense(canvas_hw=(30, 48)))

    assert tuple(converted.mask.shape) == (1, *ROOT_HW)
    assert bool(converted.mask.all())


def test_canvas_far_larger_than_root_is_rejected():
    with pytest.raises(ValueError, match="outside the root frame"):
        prediction_to_root(_dense(canvas_hw=(40, 60)))


def test_plain_boxes_with_identity_metadata_are_unchanged():
    detections = Detections(
        xyxy=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.ones(1),
        image_metadata=_metadata(),
    )

    assert prediction_to_root(detections) is detections


def test_nonzero_offset_pastes_at_the_offset():
    converted = prediction_to_root(_dense(offset_xy=(7, 3)))

    assert np.array_equal(
        converted.mask[0].numpy(), _expected_root_mask(slice(3, 8), slice(7, 17))
    )
    assert converted.xyxy.tolist() == [[7.0, 3.0, 17.0, 8.0]]


def test_nonunit_scale_resamples_onto_the_root_canvas():
    # 0.5 local pixels per root pixel, so S = 2 root pixels per local pixel.
    converted = prediction_to_root(_dense(offset_xy=(7, 3), local_per_root=0.5))

    assert np.array_equal(
        converted.mask[0].numpy(), _expected_root_mask(slice(3, 13), slice(7, 27))
    )
    assert converted.xyxy.tolist() == [[7.0, 3.0, 27.0, 13.0]]


def test_semantic_grid_with_offset_is_refused():
    with pytest.raises(ValueError, match="no lossless"):
        prediction_to_root(_semantic(offset_xy=(7, 3)))


def test_explicit_image_dimensions_must_match_the_canvas():
    with pytest.raises(ValueError, match="disagrees"):
        prediction_to_root(_dense(image_hw=(6, 10)))

    converted = prediction_to_root(_dense(image_hw=LOCAL_HW))
    assert tuple(converted.mask.shape) == (1, *ROOT_HW)


def test_own_coordinates_keep_the_local_canvas():
    dense = _dense()
    semantic = _semantic()

    assert convert_prediction_output(dense, {"coordinates_system": "own"}) is dense
    assert (
        convert_prediction_output(semantic, {"coordinates_system": "own"}) is semantic
    )
