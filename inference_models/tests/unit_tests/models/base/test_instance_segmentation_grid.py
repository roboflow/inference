"""Compact carrier ownership, coordinate semantics and compatibility."""

import pytest
import torch

from inference_models import InstanceDetections, InstanceSegmentationGrid
from inference_models.entities import ImageDimensions
from inference_models.models.common.roboflow.instance_segmentation_grid import (
    build_instance_segmentation_grid,
)
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)


def _metadata(**changes):
    metadata = PreProcessingMetadata(
        pad_left=4,
        pad_top=8,
        pad_right=4,
        pad_bottom=8,
        scale_width=0.5,
        scale_height=0.5,
        original_size=ImageDimensions(height=200, width=300),
        size_after_pre_processing=ImageDimensions(height=100, width=160),
        inference_size=ImageDimensions(height=66, width=88),
        static_crop_offset=StaticCropOffset(20, 30, 160, 100),
    )
    return metadata._replace(**changes)


def _grid(**changes):
    kwargs = dict(
        scores=torch.ones((2, 33, 44)),
        score_type="logits",
        threshold=0.0,
        grid_padding_ltrb=(2, 4, 2, 4),
        image_crop_xywh=(20, 30, 160, 100),
    )
    return InstanceSegmentationGrid(**(kwargs | changes))


def _result(**changes):
    kwargs = dict(
        xyxy=torch.tensor([[19.5, 29.5, 180.5, 130.5]] * 2),
        class_id=torch.tensor([0, 1]),
        confidence=torch.tensor([0.7, 0.8]),
        mask=_grid(),
        image_size=(200, 300),
    )
    return InstanceDetections(**(kwargs | changes))


def test_grid_uses_logical_image_sizes_and_preserves_subpixel_boxes():
    result = _result()
    assert result.mask_size == result.mask_frame_size == result.image_size == (200, 300)
    assert result.mask.scores.shape == (2, 33, 44)
    assert result.xyxy[0, 0] == 19.5


def test_grid_iteration_preserves_geometry_without_materialization():
    result = _result()
    rows = list(result)
    assert len(rows) == 2
    assert isinstance(rows[1][1], InstanceSegmentationGrid)
    assert rows[1][1].scores.shape == (1, 33, 44)
    assert rows[1][1].grid_padding_ltrb == result.mask.grid_padding_ltrb
    assert (
        rows[1][1].scores.untyped_storage().data_ptr()
        == result.mask.scores.untyped_storage().data_ptr()
    )
    assert rows[1][2] == 1


def test_supervision_rejects_grid_with_explicit_dense_guidance():
    with pytest.raises(ValueError, match="mask_format='dense'"):
        _result().to_supervision()


@pytest.mark.parametrize(
    "changes",
    [
        {"scores": torch.ones((2, 3, 4), dtype=torch.bool)},
        {"scores": torch.ones((2, 0, 4))},
        {"score_type": "binary"},
        {"threshold": float("nan")},
        {"score_type": "probabilities", "threshold": 1.1},
        {"grid_padding_ltrb": (44, 0, 0, 0)},
        {"grid_padding_ltrb": (0.5, 0, 0, 0)},
        {"image_crop_xywh": (-1, 0, 1, 1)},
    ],
)
def test_carrier_rejects_invalid_contract(changes):
    with pytest.raises(ValueError):
        _grid(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"image_size": None},
        {"mask_size": (33, 44)},
        {"mask_frame_size": (100, 160)},
        {"xyxy": torch.ones((2, 4), dtype=torch.int32)},
        {"class_id": torch.ones(1)},
        {"bboxes_metadata": [{}]},
        {"image_size": (100, 160)},
    ],
)
def test_result_rejects_inconsistent_grid_frame_or_rows(changes):
    with pytest.raises(ValueError):
        _result(**changes)


def test_builder_owns_scores_and_boxes_without_rounding_or_clipping():
    source_boxes = torch.tensor([[-10.25, -12.5, 200.25, 180.5]], dtype=torch.float16)
    source_scores = torch.arange(33 * 44, dtype=torch.float32).reshape(1, 33, 44)
    expected = (source_boxes.float() - torch.tensor([4, 8, 4, 8])) / 0.5 + torch.tensor(
        [20, 30, 20, 30]
    )
    boxes, grid = build_instance_segmentation_grid(
        image_bboxes=source_boxes,
        masks=source_scores,
        metadata=_metadata(),
        score_type="logits",
    )
    torch.testing.assert_close(boxes, expected)
    assert boxes.dtype == torch.float32
    assert boxes[0, 0] < 0 and boxes[0, 2] > 300
    assert grid.grid_padding_ltrb == (2, 4, 2, 4)
    source_boxes.zero_()
    source_scores.zero_()
    torch.testing.assert_close(boxes, expected)
    assert grid.scores[0, -1, -1] == 33 * 44 - 1


def test_builder_resolves_effective_non_square_grid_and_negative_padding():
    metadata = _metadata(
        nonsquare_intermediate_size=ImageDimensions(height=132, width=176),
        pad_left=-8,
        pad_top=-16,
        pad_right=-8,
        pad_bottom=-16,
    )
    _, grid = build_instance_segmentation_grid(
        image_bboxes=torch.zeros((1, 4)),
        masks=torch.zeros((1, 33, 44)),
        metadata=metadata,
        score_type="logits",
    )
    assert grid.grid_padding_ltrb == (-2, -4, -2, -4)


def test_builder_keeps_empty_grid_geometry_and_origin_crop():
    metadata = _metadata(static_crop_offset=StaticCropOffset(0, 0, 160, 100))
    boxes, grid = build_instance_segmentation_grid(
        image_bboxes=torch.empty((0, 4)),
        masks=torch.empty((0, 33, 44)),
        metadata=metadata,
        score_type="probabilities",
        binarization_threshold=0.5,
    )
    result = InstanceDetections(
        xyxy=boxes,
        class_id=torch.empty(0, dtype=torch.int64),
        confidence=torch.empty(0),
        mask=grid,
        image_size=(200, 300),
    )
    assert len(result) == 0
    assert grid.scores.shape == (0, 33, 44)
    assert grid.image_crop_xywh == (0, 0, 160, 100)
    assert result.mask_size == (200, 300)
