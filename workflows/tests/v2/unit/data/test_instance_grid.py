"""Native compact masks and encoded raster/image coordinate boundaries."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from pycocotools import mask as mask_utils
from roboflow_workflows.v2.blocks.geometry.mask_inputs import (
    boxes_of_view,
    instance_predictions_of,
)
from roboflow_workflows.v2.blocks.visualization.annotations.rows import boxes_of
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.masks import SegmentationView, mask_provider_of
from roboflow_workflows.v2.data.prediction_ops import select_predictions
from roboflow_workflows.v2.data.rle import decode_rle_masks

from inference_models.entities import ImageDimensions
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.types import (
    InstanceSegmentationGrid,
    InstancesRLEMasks,
)
from inference_models.models.common.roboflow.model_packages import StaticCropOffset
from inference_models.models.common.roboflow.post_processing import (
    align_instance_segmentation_results,
)


def grid_prediction(*, rows=2, padding=(0, 0, 0, 0), crop=(0, 0, 13, 9), image=(9, 13)):
    scores = torch.linspace(-2, 2, max(rows, 1) * 4 * 6).reshape(max(rows, 1), 4, 6)[
        :rows
    ]
    grid = InstanceSegmentationGrid(
        scores=scores,
        score_type="logits",
        threshold=0.15,
        grid_padding_ltrb=padding,
        image_crop_xywh=crop,
    )
    return InstanceDetections(
        xyxy=torch.tensor([[-1.2, 0.6, 12.8, 8.9], [2.2, 1.3, 7.9, 5.8]])[
            :rows
        ].float(),
        confidence=torch.ones(rows),
        class_id=torch.arange(rows),
        mask=grid,
        image_size=image,
        image_metadata={"name": "source"},
        bboxes_metadata=[{"row": i} for i in range(rows)],
    )


@pytest.mark.parametrize("factor", [0.0, 0.37, 1.0])
@pytest.mark.parametrize("rows", [0, 2])
@pytest.mark.parametrize(
    "padding,crop,image",
    [
        ((0, 0, 0, 0), (0, 0, 13, 9), (9, 13)),
        ((1, 1, 0, 0), (2, 1, 7, 5), (9, 13)),
        ((-1, -2, 0, 0), (0, 0, 7, 5), (9, 13)),
        ((0, 0, 0, 0), (0, 0, 3, 2), (2, 3)),
    ],
)
def test_grid_materialization_matches_native_mask_order(
    factor, rows, padding, crop, image
):
    native = grid_prediction(rows=rows, padding=padding, crop=crop, image=image)
    view = SegmentationView.from_predictions(native)
    x, y, width, height = crop
    _, expected = align_instance_segmentation_results(
        image_bboxes=torch.zeros((rows, 4)),
        masks=native.mask.scores,
        padding=padding,
        scale_width=1.0,
        scale_height=1.0,
        original_size=ImageDimensions(*image),
        size_after_pre_processing=ImageDimensions(height, width),
        inference_size=ImageDimensions(4, 6),
        static_crop_offset=StaticCropOffset(x, y, width, height),
        binarization_threshold=0.15,
        masks_resolution_factor=factor,
    )
    result = view.materialize(masks_resolution_factor=factor)
    assert torch.equal(result.mask, expected)
    assert result.mask_size == tuple(expected.shape[1:])
    assert result.image_size == result.mask_frame_size == image
    assert view.detections.xyxy is native.xyxy
    assert view.scores is native.mask.scores
    assert result.image_metadata is native.image_metadata
    assert result.bboxes_metadata is native.bboxes_metadata
    if factor == 1.0:
        assert result is view.full_res()
        expected_boxes = native.xyxy.clone()
        expected_boxes[:, 0::2].clamp_(0, image[1])
        expected_boxes[:, 1::2].clamp_(0, image[0])
        assert torch.equal(result.xyxy, expected_boxes.round().int())


def binary_prediction(*, rle=False, rows=2, reduced=True, extent=None):
    masks = torch.tensor([[[0, 1, 1, 0], [1, 0, 1, 0]]] * rows, dtype=torch.bool)
    if not rows:
        masks = torch.empty((0, 2, 4), dtype=torch.bool)
    mask = masks
    image = (8, 16) if reduced else (2, 4)
    if rle:
        counts = [
            mask_utils.encode(np.asfortranarray(row.numpy(), dtype=np.uint8))["counts"]
            for row in masks
        ]
        mask = InstancesRLEMasks(image_size=image, mask_size=(2, 4), masks=counts)
    return InstanceDetections(
        xyxy=torch.tensor([[1.0, 0.0, 3.0, 2.0]] * rows).reshape(rows, 4),
        confidence=torch.ones(rows),
        class_id=torch.arange(rows),
        mask=mask,
        mask_size=(2, 4),
        image_size=image,
        mask_frame_size=extent or image,
        image_metadata={"token": 1},
        bboxes_metadata=[{"row": i} for i in range(rows)],
    )


@pytest.mark.parametrize("rle", [False, True])
@pytest.mark.parametrize("rows", [0, 2])
def test_reduced_binary_carriers_are_lazy_and_fidelity_is_explicit(
    rle, rows, monkeypatch
):
    native = binary_prediction(rle=rle, rows=rows)
    from roboflow_workflows.v2.data import rle as rle_module

    original_decode = rle_module.decode_rle_tensor
    calls = []

    def decode(*args, **kwargs):
        calls.append(1)
        return original_decode(*args, **kwargs)

    monkeypatch.setattr(rle_module, "decode_rle_tensor", decode)
    view = SegmentationView.from_predictions(native)
    provider = mask_provider_of(view)
    expected_boxes = native.xyxy * 4
    assert torch.equal(view.detections.xyxy, expected_boxes)
    assert torch.equal(boxes_of(native, block="test").xyxy, expected_boxes)
    assert torch.equal(boxes_of_view(native).xyxy, expected_boxes)
    updated = view.with_image_metadata({"new": 2}).with_row_metadata(
        [{} for _ in range(rows)]
    )
    assert updated.binary_source.mask is native.mask
    selected = updated.select(torch.tensor([1, 0, 1] if rows else [], dtype=torch.long))
    assert selected.binary_source.mask_size == (2, 4)
    assert (
        selected.binary_source.image_size
        == selected.binary_source.mask_frame_size
        == (8, 16)
    )
    assert not calls
    with pytest.raises(ContractError, match="reduced binary"):
        provider.binary(fidelity="reference")
    with pytest.raises(ContractError, match="reduced binary"):
        instance_predictions_of(native)
    assert not calls
    coarse = provider.binary(fidelity="approximate")
    assert coarse.fidelity == "approximate"
    assert coarse.predictions.mask.shape == (rows, 8, 16)
    assert torch.equal(coarse.predictions.xyxy, expected_boxes)
    assert coarse.predictions is provider.binary(fidelity="approximate").predictions
    assert len(calls) == int(rle)


@pytest.mark.parametrize("rle", [False, True])
def test_selection_preserves_binary_sizes_and_counts(rle):
    native = binary_prediction(rle=rle)
    selected = select_predictions(native, indices=[1, 0, 1])
    assert selected.image_size == selected.mask_frame_size == (8, 16)
    assert selected.mask_size == (2, 4)
    assert selected.bboxes_metadata == [{"row": 1}, {"row": 0}, {"row": 1}]
    if rle:
        assert selected.mask.mask_size == (2, 4)
        assert selected.mask.image_size == (8, 16)
        assert selected.mask.masks[0] is native.mask.masks[1]
        assert decode_rle_masks(selected.mask).shape == (3, 2, 4)
        assert decode_rle_masks(native.mask, rows=[1]).shape == (1, 2, 4)


def test_grid_selection_metadata_and_threshold_changes_remain_lazy():
    native = grid_prediction()
    selected = select_predictions(native, indices=[1, 0, 1])
    assert isinstance(selected.mask, InstanceSegmentationGrid)
    assert (
        selected.image_size == selected.mask_frame_size == selected.mask_size == (9, 13)
    )
    assert torch.equal(selected.mask.scores, native.mask.scores[[1, 0, 1]])
    view = SegmentationView.from_predictions(native)
    updated = view.with_image_metadata({"new": 1}).with_row_metadata([{}, {}])
    assert updated.scores is view.scores
    assert updated.detections.image_metadata == {"new": 1}
    assert not updated.is_full_res_materialized()
    assert torch.equal(
        updated.select(torch.tensor([1, 0, 1])).scores, selected.mask.scores
    )
    assert view.with_threshold(0.8).threshold == 0.8
    assert view.threshold == 0.15


@pytest.mark.parametrize("rle", [False, True])
def test_full_res_binary_and_origin_extent_placement_are_exact(rle):
    native = binary_prediction(rle=rle, reduced=False)
    provider = mask_provider_of(native)
    result = provider.binary()
    assert result.fidelity == "reference"
    assert result.predictions.mask.shape == (2, 2, 4)
    crop = replace(native, image_size=(8, 16), mask_frame_size=(2, 4))
    if rle:
        crop = replace(crop, mask=replace(crop.mask, image_size=(8, 16)))
    full = SegmentationView.from_predictions(crop).full_res()
    assert full.mask.shape == (2, 8, 16)
    assert torch.equal(full.mask[:, :2, :4], result.predictions.mask)
    assert not full.mask[:, 2:].any()
    assert not full.mask[:, :, 4:].any()
    assert torch.equal(full.xyxy, native.xyxy)


@pytest.mark.parametrize("factor", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_materialization_factor_fails(factor):
    with pytest.raises(ContractError, match="finite"):
        SegmentationView.from_predictions(grid_prediction()).materialize(
            masks_resolution_factor=factor
        )


def test_factor_crop_rle_roundtrip_preserves_geometry_and_rows():
    view = SegmentationView.from_predictions(
        grid_prediction(padding=(1, 0, 0, 0), crop=(2, 1, 7, 5))
    ).select(torch.tensor([1, 0, 1]))
    dense = view.materialize(masks_resolution_factor=0.37)
    encoded = view.to_rle(masks_resolution_factor=0.37)
    assert encoded.mask_size == encoded.mask.mask_size == dense.mask_size
    assert encoded.image_size == encoded.mask.image_size == (9, 13)
    assert encoded.mask_frame_size == dense.mask_frame_size
    assert encoded.image_metadata is dense.image_metadata
    assert encoded.bboxes_metadata is dense.bboxes_metadata
    assert torch.equal(encoded.xyxy, dense.xyxy)
    assert np.array_equal(decode_rle_masks(encoded.mask), dense.mask.numpy())


def test_empty_reduced_rle_decodes_encoded_size_without_guessing_image_size():
    native = binary_prediction(rle=True, rows=0)
    view = SegmentationView.from_predictions(native)
    assert view.scores.shape == (0, 2, 4)
    assert decode_rle_masks(native.mask).shape == (0, 2, 4)


def test_grid_wire_boundary_requires_materialization_and_rle_keeps_sizes():
    from roboflow_workflows.v2.data.predictions import (
        INSTANCE_SEGMENTATION_PREDICTION_KIND,
    )

    native = grid_prediction()
    INSTANCE_SEGMENTATION_PREDICTION_KIND.check(native)
    with pytest.raises(ContractError, match="materialize.*to_rle"):
        INSTANCE_SEGMENTATION_PREDICTION_KIND.to_serialized(native)
    encoded = SegmentationView.from_predictions(native).to_rle(
        masks_resolution_factor=0.37
    )
    restored = INSTANCE_SEGMENTATION_PREDICTION_KIND.to_payload(
        INSTANCE_SEGMENTATION_PREDICTION_KIND.to_serialized(encoded)
    )
    assert restored.mask_size == encoded.mask_size
    assert restored.image_size == encoded.image_size
    assert restored.mask_frame_size == encoded.mask_frame_size
    assert restored.mask.mask_size == encoded.mask.mask_size
    assert np.array_equal(
        decode_rle_masks(restored.mask), decode_rle_masks(encoded.mask)
    )


def test_reduced_source_and_image_box_storage_both_join_stream_lifetime(monkeypatch):
    from roboflow_workflows.v2.data.masks import grid as grid_module

    native = binary_prediction()
    view = SegmentationView.from_predictions(native)
    ordered = []
    monkeypatch.setattr(
        grid_module,
        "order_current_stream",
        lambda ready, tensors: ordered.extend(tensors),
    )
    source = view.binary_source
    boxes = view.detections.xyxy
    assert source.xyxy is native.xyxy
    assert boxes.data_ptr() != native.xyxy.data_ptr()
    assert any(tensor is native.xyxy for tensor in ordered)
    assert any(tensor is boxes for tensor in ordered)
