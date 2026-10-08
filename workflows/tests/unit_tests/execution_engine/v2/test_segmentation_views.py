"""SegmentationView: geometry, exact materialization, ownership, cache, ops."""

import threading

import pytest
import torch
from roboflow_workflows.execution_engine.v2 import blocks
from roboflow_workflows.execution_engine.v2.blocks import segmentation_view_ops as ops
from roboflow_workflows.execution_engine.v2.blocks import segmentation_views
from roboflow_workflows.execution_engine.v2.blocks.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.execution_engine.v2.blocks.segmentation_views import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    MaskGridGeometry,
    SegmentationView,
)
from roboflow_workflows.execution_engine.v2.errors import ContractError
from roboflow_workflows.execution_engine.v2.recording.codecs import type_name_of

from inference_models.entities import ImageDimensions
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)
from inference_models.models.common.roboflow.post_processing import (
    align_instance_segmentation_results,
)


def _metadata(
    *,
    original_hw,
    resized_hw,
    inference_hw,
    padding_ltrb,
    crop_xywh=None,
) -> PreProcessingMetadata:
    crop_xywh = crop_xywh or (0, 0, original_hw[1], original_hw[0])
    left, top, right, bottom = padding_ltrb
    metadata = PreProcessingMetadata(
        pad_left=left,
        pad_top=top,
        pad_right=right,
        pad_bottom=bottom,
        original_size=ImageDimensions(*original_hw),
        size_after_pre_processing=ImageDimensions(*resized_hw),
        inference_size=ImageDimensions(*inference_hw),
        scale_width=(inference_hw[1] - left - right) / resized_hw[1],
        scale_height=(inference_hw[0] - top - bottom) / resized_hw[0],
        static_crop_offset=StaticCropOffset(*crop_xywh),
    )

    return metadata


# Letterboxed 100 x 61 image in a 32 x 32 input: pad top 6, bottom 6 network
# pixels, rounded to 1.5 -> 2 grid cells (not a multiple of the stride).
LETTERBOX = _metadata(
    original_hw=(61, 100),
    resized_hw=(61, 100),
    inference_hw=(32, 32),
    padding_ltrb=(0, 6, 0, 6),
)
# Anisotropic stretch of a static crop: crop (10, 5, 50, 30) of a 60 x 80 image.
CROPPED_STRETCH = _metadata(
    original_hw=(60, 80),
    resized_hw=(30, 50),
    inference_hw=(16, 24),
    padding_ltrb=(0, 0, 0, 0),
    crop_xywh=(10, 5, 50, 30),
)
# Centre crop: negative padding.
CENTRE_CROP = _metadata(
    original_hw=(40, 40),
    resized_hw=(40, 40),
    inference_hw=(32, 32),
    padding_ltrb=(-4, -4, -4, -4),
)
GEOMETRY_CASES = {
    "letterbox": (LETTERBOX, (8, 8)),
    "cropped_stretch": (CROPPED_STRETCH, (4, 6)),
    "centre_crop": (CENTRE_CROP, (8, 8)),
}


def _detections(
    rows: int,
    *,
    device="cpu",
    class_rows=None,
    confidence_rows=None,
    metadata_rows=None,
    box_columns=4,
) -> Detections:
    """Row-aligned detections; the ``*_rows`` overrides build malformed ones."""
    xyxy = torch.arange(rows * box_columns, dtype=torch.float32, device=device)
    class_rows = rows if class_rows is None else class_rows
    confidence_rows = rows if confidence_rows is None else confidence_rows
    metadata_rows = rows if metadata_rows is None else metadata_rows
    detections = Detections(
        xyxy=xyxy.reshape(rows, box_columns).round().int(),
        class_id=torch.arange(class_rows, dtype=torch.int32, device=device),
        confidence=torch.linspace(0.9, 0.5, confidence_rows, device=device),
        bboxes_metadata=[{"row": row} for row in range(metadata_rows)],
    )

    return detections


def _view(metadata, grid_hw, *, rows=3, seed=0, score_type="logits", threshold=0.0):
    generator = torch.Generator().manual_seed(seed)
    scores = torch.randn((rows, *grid_hw), generator=generator)
    if score_type == "probabilities":
        scores = scores.sigmoid()
    view = SegmentationView(
        detections=_detections(rows),
        scores=scores,
        score_type=score_type,
        threshold=threshold,
        geometry=MaskGridGeometry.from_pre_processing(metadata, grid_size_hw=grid_hw),
    )

    return view


def _reference_mask(view: SegmentationView, metadata) -> torch.Tensor:
    _, mask = align_instance_segmentation_results(
        image_bboxes=torch.zeros((len(view), 4)),
        masks=view.scores.clone(),
        padding=(
            metadata.pad_left,
            metadata.pad_top,
            metadata.pad_right,
            metadata.pad_bottom,
        ),
        scale_width=metadata.scale_width,
        scale_height=metadata.scale_height,
        original_size=metadata.original_size,
        size_after_pre_processing=metadata.size_after_pre_processing,
        inference_size=metadata.inference_size,
        static_crop_offset=metadata.static_crop_offset,
        binarization_threshold=view.threshold,
    )

    return mask


# ---------------------------------------------------------------------------
# Geometry and exact materialization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(GEOMETRY_CASES))
@pytest.mark.parametrize(
    "score_type,threshold", [("logits", 0.0), ("probabilities", 0.5)]
)
def test_full_res_equals_reference_helper(case, score_type, threshold) -> None:
    metadata, grid_hw = GEOMETRY_CASES[case]
    view = _view(metadata, grid_hw, score_type=score_type, threshold=threshold)

    dense = view.full_res()

    assert isinstance(dense, InstanceDetections)
    assert dense.mask.dtype == torch.bool
    assert tuple(dense.mask.shape) == (3, *view.geometry.original_size_hw)
    assert torch.equal(dense.mask, _reference_mask(view, metadata))
    assert torch.equal(dense.xyxy, view.detections.xyxy)
    assert dense.bboxes_metadata == [{"row": 0}, {"row": 1}, {"row": 2}]
    INSTANCE_SEGMENTATION_PREDICTION_KIND.check(dense)


def test_letterbox_grid_padding_is_rounded_like_the_helper() -> None:
    geometry = MaskGridGeometry.from_pre_processing(LETTERBOX, grid_size_hw=(8, 8))

    assert geometry.grid_padding_tblr == (2, 2, 0, 0)
    assert geometry.unpadded_size_hw == (4, 8)
    mapping = geometry.grid_to_image(frame_id="frame")
    assert mapping.scale_xy == pytest.approx((100 / 8, 61 / 4))
    assert mapping.offset_xy == (0.0, 0.0)


def test_cropped_stretch_maps_cells_into_the_crop() -> None:
    geometry = MaskGridGeometry.from_pre_processing(
        CROPPED_STRETCH, grid_size_hw=(4, 6)
    )

    mapping = geometry.grid_to_image()

    assert mapping.scale_xy == pytest.approx((50 / 6, 30 / 4))
    assert mapping.offset_xy == (10.0, 5.0)
    assert mapping.frame_size_hw == (60, 80)


def test_partial_static_crop_at_the_origin_is_rejected() -> None:
    # The reference helper only places a crop on the full-image canvas at a
    # nonzero offset; at (0, 0) it would return crop-sized "full-res" masks.
    metadata = _metadata(
        original_hw=(60, 80),
        resized_hw=(30, 50),
        inference_hw=(16, 24),
        padding_ltrb=(0, 0, 0, 0),
        crop_xywh=(0, 0, 50, 30),
    )
    _, helper_mask = align_instance_segmentation_results(
        image_bboxes=torch.zeros((1, 4)),
        masks=torch.ones((1, 4, 6)),
        padding=(0, 0, 0, 0),
        scale_width=metadata.scale_width,
        scale_height=metadata.scale_height,
        original_size=metadata.original_size,
        size_after_pre_processing=metadata.size_after_pre_processing,
        inference_size=metadata.inference_size,
        static_crop_offset=metadata.static_crop_offset,
    )
    assert tuple(helper_mask.shape) == (1, 30, 50)

    with pytest.raises(ContractError, match=r"at \(0, 0\) is smaller"):
        MaskGridGeometry.from_pre_processing(metadata, grid_size_hw=(4, 6))


@pytest.mark.parametrize(
    "crop_xywh,crop_size_hw,message",
    [
        ((10, 5, 50, 30), (30, 49), "does not match crop_size_hw"),
        ((40, 5, 50, 30), (30, 50), "leaves the"),
    ],
)
def test_inconsistent_static_crop_is_rejected(crop_xywh, crop_size_hw, message) -> None:
    with pytest.raises(ContractError, match=message):
        MaskGridGeometry(
            grid_size_hw=(4, 6),
            inference_size_hw=(16, 24),
            padding_ltrb=(0, 0, 0, 0),
            scale_xy=(0.48, 0.5),
            crop_size_hw=crop_size_hw,
            original_size_hw=(60, 80),
            static_crop_xywh=crop_xywh,
        )


def test_centre_crop_unpad_zero_extends_like_the_helper() -> None:
    view = _view(CENTRE_CROP, (8, 8), rows=1)

    low_res = view.low_res()

    assert view.geometry.grid_padding_tblr == (-1, -1, -1, -1)
    assert tuple(low_res.shape) == (1, 10, 10)
    assert torch.equal(low_res[:, 1:-1, 1:-1], view.scores)
    assert low_res[:, 0].abs().sum() == 0


def test_zero_detections_materialize_an_empty_dense_mask() -> None:
    view = _view(LETTERBOX, (8, 8), rows=0)

    dense = view.full_res()

    assert tuple(dense.mask.shape) == (0, 61, 100)
    assert tuple(view.low_res_binary().shape) == (0, 4, 8)
    assert ops.instance_areas(view).values.shape == (0,)


def test_noncontiguous_scores_are_copied_contiguous() -> None:
    wide = torch.randn((3, 8, 16))
    scores = wide[:, :, ::2]
    assert not scores.is_contiguous()

    view = SegmentationView(
        detections=_detections(3),
        scores=scores,
        score_type="logits",
        threshold=0.0,
        geometry=MaskGridGeometry.from_pre_processing(LETTERBOX, grid_size_hw=(8, 8)),
    )

    assert view.scores.is_contiguous()
    assert torch.equal(view.full_res().mask, _reference_mask(view, LETTERBOX))


def test_reference_box_rescale_does_not_touch_view_boxes() -> None:
    view = _view(LETTERBOX, (8, 8))
    boxes_before = view.detections.xyxy.clone()

    view.compute_full_res()

    assert torch.equal(view.detections.xyxy, boxes_before)


# ---------------------------------------------------------------------------
# Ownership and immutability
# ---------------------------------------------------------------------------


def test_constructor_copies_unless_ownership_is_handed_over() -> None:
    scores = torch.randn((2, 8, 8))
    geometry = MaskGridGeometry.from_pre_processing(LETTERBOX, grid_size_hw=(8, 8))

    copied = SegmentationView(
        detections=_detections(2),
        scores=scores,
        score_type="logits",
        threshold=0.0,
        geometry=geometry,
    )

    assert copied.scores.data_ptr() != scores.data_ptr()


def test_gathered_rows_survive_reuse_of_the_model_output_buffer() -> None:
    reused_buffer = torch.randn((10, 8, 8))
    rows = torch.tensor([7, 2, 2])
    expected = reused_buffer[rows].clone()

    view = SegmentationView.from_selected_rows(
        reused_buffer,
        rows=rows,
        detections=_detections(3),
        score_type="logits",
        threshold=0.0,
        geometry=MaskGridGeometry.from_pre_processing(LETTERBOX, grid_size_hw=(8, 8)),
    )
    reused_buffer.fill_(-100.0)  # next frame overwrites the buffer

    assert torch.equal(view.scores, expected)


def test_attributes_cannot_be_reassigned() -> None:
    view = _view(LETTERBOX, (8, 8))

    with pytest.raises(AttributeError, match="read-only"):
        view._threshold = 0.3
    with pytest.raises(AttributeError, match="read-only"):
        view.threshold = 0.3


@pytest.mark.parametrize(
    "change,message",
    [
        ({"score_type": "heatmap"}, "score_type"),
        ({"scores": torch.randn((2, 8, 8))}, "2 rows for 3"),
        ({"scores": torch.randn((3, 4, 4))}, "does not match geometry"),
        ({"scores": torch.ones((3, 8, 8), dtype=torch.int32)}, "floating"),
        ({"score_type": "probabilities", "threshold": 1.5}, r"\[0, 1\]"),
        ({"score_type": "binary"}, "bool or uint8"),
        (
            {
                "score_type": "binary",
                "scores": torch.ones((3, 8, 8), dtype=torch.bool),
                "threshold": 0.5,
            },
            "fixed threshold 0.0",
        ),
        ({"detections": _detections(3, class_rows=2)}, r"class_id must be a \(3,\)"),
        ({"detections": _detections(3, confidence_rows=2)}, r"confidence must be"),
        ({"detections": _detections(3, metadata_rows=2)}, "2 rows for 3 boxes"),
        ({"detections": _detections(3, box_columns=5)}, r"xyxy must be an \(N, 4\)"),
    ],
)
def test_invalid_parts_are_rejected(change, message) -> None:
    parts = {
        "detections": _detections(3),
        "scores": torch.randn((3, 8, 8)),
        "score_type": "logits",
        "threshold": 0.0,
        "geometry": MaskGridGeometry.from_pre_processing(
            LETTERBOX, grid_size_hw=(8, 8)
        ),
    }
    parts.update(change)

    with pytest.raises(ContractError, match=message):
        SegmentationView(**parts)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_full_res_is_computed_once_and_shared() -> None:
    view = _view(LETTERBOX, (8, 8))
    assert not view.is_full_res_materialized()

    first, second = view.full_res(), view.full_res()

    assert first is second
    assert view.materialization_count == 1
    assert view.is_full_res_materialized()


def test_uncached_materialization_counts_every_call() -> None:
    view = _view(LETTERBOX, (8, 8))

    first, second = view.compute_full_res(), view.compute_full_res()

    assert first is not second
    assert torch.equal(first.mask, second.mask)
    assert view.materialization_count == 2
    assert not view.is_full_res_materialized()


def test_concurrent_consumers_share_one_materialization(monkeypatch) -> None:
    view = _view(LETTERBOX, (8, 8))
    entered = threading.Event()
    release = threading.Event()
    real_align = segmentation_views.align_instance_segmentation_results

    def slow_align(**kwargs):
        entered.set()
        release.wait(timeout=5)
        return real_align(**kwargs)

    monkeypatch.setattr(
        segmentation_views, "align_instance_segmentation_results", slow_align
    )
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(view.full_res()))
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    assert entered.wait(timeout=5)
    release.set()
    for thread in threads:
        thread.join(timeout=5)

    assert len(results) == 4
    assert all(result is results[0] for result in results)
    assert view.materialization_count == 1


def test_failed_materialization_publishes_nothing(monkeypatch) -> None:
    view = _view(LETTERBOX, (8, 8))
    real_align = segmentation_views.align_instance_segmentation_results
    calls = []

    def failing_once(**kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("out of memory")
        return real_align(**kwargs)

    monkeypatch.setattr(
        segmentation_views, "align_instance_segmentation_results", failing_once
    )

    with pytest.raises(RuntimeError, match="out of memory"):
        view.full_res()
    assert not view.is_full_res_materialized()
    assert view.materialization_count == 0

    dense = view.full_res()

    assert torch.equal(dense.mask, _reference_mask(view, LETTERBOX))
    assert view.materialization_count == 1


def test_reordered_and_repeated_rows_do_not_alias_caches() -> None:
    view = _view(LETTERBOX, (8, 8))
    full = view.full_res()

    reordered = view.select(torch.tensor([2, 0, 0]))
    other = view.select(torch.tensor([0, 2, 0]))

    assert torch.equal(reordered.full_res().mask, full.mask[[2, 0, 0]])
    assert torch.equal(other.full_res().mask, full.mask[[0, 2, 0]])
    assert reordered.detections.bboxes_metadata == [{"row": 2}, {"row": 0}, {"row": 0}]
    assert torch.equal(
        reordered.detections.class_id, torch.tensor([2, 0, 0], dtype=torch.int32)
    )
    assert view.materialization_count == 1


@pytest.mark.parametrize(
    "rows,message",
    [
        (torch.tensor([0, 3]), r"\[0, 3\), got \[0, 3\]"),
        (torch.tensor([-1, 0]), r"\[0, 3\), got \[-1, 0\]"),
        (torch.tensor([0.0, 1.0]), "1-D integer"),
        (torch.tensor([[0, 1]]), "1-D integer"),
        (torch.tensor([True, False]), "1-D integer"),
    ],
)
def test_invalid_cpu_rows_are_contract_errors(rows, message) -> None:
    view = _view(LETTERBOX, (8, 8))

    with pytest.raises(ContractError, match=message):
        view.select(rows)
    with pytest.raises(ContractError, match=message):
        SegmentationView.from_selected_rows(
            view.scores,
            rows=rows,
            detections=_detections(int(rows.numel())),
            score_type="logits",
            threshold=0.0,
            geometry=view.geometry,
        )


def test_threshold_change_builds_a_new_view_with_its_own_cache() -> None:
    view = _view(LETTERBOX, (8, 8))
    dense = view.full_res()

    stricter = view.with_threshold(1.0)

    assert stricter.threshold == 1.0
    assert not stricter.is_full_res_materialized()
    assert stricter.full_res().mask.sum() < dense.mask.sum()
    assert torch.equal(stricter.full_res().mask, _reference_mask(stricter, LETTERBOX))


# ---------------------------------------------------------------------------
# Dense fallback
# ---------------------------------------------------------------------------


def _dense_predictions() -> InstanceDetections:
    mask = torch.zeros((2, 20, 30), dtype=torch.bool)
    mask[0, 2:8, 3:9] = True
    mask[1, 10:20, 0:30] = True
    predictions = InstanceDetections(
        xyxy=torch.tensor([[3, 2, 9, 8], [0, 10, 30, 20]], dtype=torch.int32),
        class_id=torch.tensor([1, 2], dtype=torch.int32),
        confidence=torch.tensor([0.9, 0.8]),
        mask=mask,
    )

    return predictions


def test_dense_fallback_returns_the_dense_predictions_unchanged() -> None:
    predictions = _dense_predictions()

    view = SegmentationView.from_dense(predictions)

    assert view.is_dense_fallback
    assert view.full_res() is predictions
    assert view.compute_full_res() is predictions
    assert view.materialization_count == 0
    assert torch.equal(view.low_res_binary(), predictions.mask)
    with pytest.raises(ContractError, match="re-threshold"):
        view.with_threshold(0.5)


def test_dense_fallback_ops_are_exact() -> None:
    predictions = _dense_predictions()
    view = SegmentationView.from_dense(predictions)

    areas = ops.instance_areas(view)
    zones = ops.zone_fractions(view, [(0, 0), (30, 0), (30, 15), (0, 15)])

    assert areas.exact and zones.exact
    assert torch.equal(areas.values, predictions.mask.sum((1, 2)).float())
    assert zones.values.tolist() == pytest.approx([1.0, 0.5])


@pytest.mark.parametrize("rows", [[1, 0, 1], [1], []])
def test_selected_dense_fallback_keeps_fixed_binary_semantics(rows) -> None:
    predictions = _dense_predictions()
    predictions.bboxes_metadata = [{"row": 0}, {"row": 1}]
    view = SegmentationView.from_dense(predictions)

    selected = view.select(torch.tensor(rows, dtype=torch.int64))
    dense = selected.full_res()

    assert selected.is_dense_fallback
    assert selected.threshold == 0.0
    assert selected.materialization_count == 0
    assert view.materialization_count == 0
    assert torch.equal(dense.mask, predictions.mask[rows])
    assert torch.equal(selected.low_res_binary(), dense.mask)
    assert torch.equal(dense.xyxy, predictions.xyxy[rows])
    assert torch.equal(dense.class_id, predictions.class_id[rows])
    assert torch.equal(dense.confidence, predictions.confidence[rows])
    assert dense.bboxes_metadata == [{"row": row} for row in rows]
    assert ops.instance_areas(selected).exact
    assert torch.equal(
        ops.instance_areas(selected).values, dense.mask.sum((1, 2)).float()
    )
    with pytest.raises(ContractError, match="re-threshold"):
        selected.with_threshold(1.0)


def test_binary_scores_cannot_be_rethresholded() -> None:
    # Reviewer reproduction: from_dense(all True).select(...).with_threshold(1)
    # gave 48 foreground cells on the grid but an empty full_res mask.
    mask = torch.ones((2, 4, 6), dtype=torch.bool)
    predictions = InstanceDetections(
        xyxy=torch.zeros((2, 4)),
        class_id=torch.zeros(2, dtype=torch.int32),
        confidence=torch.ones(2),
        mask=mask,
    )
    selected = SegmentationView.from_dense(predictions).select(torch.tensor([1, 0, 1]))

    with pytest.raises(ContractError, match="re-threshold"):
        selected.with_threshold(1.0)
    assert int(selected.low_res_binary().sum()) == int(selected.full_res().mask.sum())


# ---------------------------------------------------------------------------
# Approximate low-resolution consumers
# ---------------------------------------------------------------------------


def _single_row_view(
    grid: torch.Tensor, metadata, *, threshold=0.0
) -> SegmentationView:
    view = SegmentationView(
        detections=_detections(grid.shape[0]),
        scores=grid,
        score_type="logits",
        threshold=threshold,
        geometry=MaskGridGeometry.from_pre_processing(
            metadata, grid_size_hw=tuple(grid.shape[1:])
        ),
    )

    return view


def test_one_cell_island_area_is_reported_as_an_approximation() -> None:
    metadata = _metadata(
        original_hw=(32, 32),
        resized_hw=(32, 32),
        inference_hw=(32, 32),
        padding_ltrb=(0, 0, 0, 0),
    )
    grid = torch.full((1, 8, 8), -4.0)
    grid[0, 3, 3] = 4.0
    view = _single_row_view(grid, metadata)

    estimate = ops.instance_areas(view)
    dense_area = view.full_res().mask.sum().item()

    assert not estimate.exact
    assert estimate.values.tolist() == [16.0]
    # Bilinear interpolation shrinks an isolated cell; the estimate differs.
    assert dense_area != 16


def test_thin_diagonal_and_scores_near_threshold_disagree_measurably() -> None:
    metadata = _metadata(
        original_hw=(64, 64),
        resized_hw=(64, 64),
        inference_hw=(64, 64),
        padding_ltrb=(0, 0, 0, 0),
    )
    grid = torch.full((1, 16, 16), -0.05)
    grid[0, torch.arange(16), torch.arange(16)] = 0.05
    view = _single_row_view(grid, metadata)

    approximate = ops.nearest_full_res(view)
    exact = view.full_res().mask

    assert approximate.shape == exact.shape
    disagreement = (approximate ^ exact).sum().item()
    assert disagreement > 0
    assert approximate.sum().item() == 16 * 16


def test_zone_fraction_uses_cell_centres_in_image_pixels() -> None:
    view = _single_row_view(torch.ones((1, 4, 6)), CROPPED_STRETCH)  # all foreground
    # Left half of the crop: x in [10, 35).
    estimate = ops.zone_fractions(view, [(10, 5), (35, 5), (35, 35), (10, 35)])

    assert not estimate.exact
    assert estimate.values.tolist() == pytest.approx([0.5])


def test_zone_needs_a_polygon() -> None:
    with pytest.raises(ContractError, match="3 vertices"):
        ops.zone_fractions(_view(LETTERBOX, (8, 8)), [(0, 0), (1, 1)])


def test_low_res_paint_leaves_input_intact_and_matches_dense_rule_on_cell_interiors() -> (
    None
):
    metadata = _metadata(
        original_hw=(32, 32),
        resized_hw=(32, 32),
        inference_hw=(32, 32),
        padding_ltrb=(0, 0, 0, 0),
    )
    grid = torch.full((1, 8, 8), -4.0)
    grid[0, 2:6, 2:6] = 4.0
    view = _single_row_view(grid, metadata)
    image = torch.full((3, 32, 32), 100, dtype=torch.uint8)
    original = image.clone()
    colors = torch.tensor([[255, 0, 0]], dtype=torch.uint8)

    painted, exact = ops.paint_masks(image, view, colors=colors, opacity=0.5)
    dense_painted, dense_exact = ops.paint_masks(
        image, SegmentationView.from_dense(view.full_res()), colors=colors, opacity=0.5
    )

    assert torch.equal(image, original)
    assert not exact and dense_exact
    assert painted[:, 12:20, 12:20].tolist() == dense_painted[:, 12:20, 12:20].tolist()
    assert painted[:, 0, 0].tolist() == [100, 100, 100]
    assert painted[:, 16, 16].tolist() == [178, 50, 50]


def test_paint_rejects_an_image_of_another_size() -> None:
    view = _view(LETTERBOX, (8, 8), rows=1)

    with pytest.raises(ContractError, match="image must be"):
        ops.paint_masks(
            torch.zeros((3, 10, 10), dtype=torch.uint8),
            view,
            colors=torch.zeros((1, 3), dtype=torch.uint8),
            opacity=0.5,
        )


# ---------------------------------------------------------------------------
# Kind
# ---------------------------------------------------------------------------


def test_view_kind_is_distinct_from_the_dense_kind() -> None:
    view = _view(LETTERBOX, (8, 8))

    INSTANCE_SEGMENTATION_VIEW_KIND.check(view)
    assert (
        INSTANCE_SEGMENTATION_VIEW_KIND.name
        != INSTANCE_SEGMENTATION_PREDICTION_KIND.name
    )
    with pytest.raises(ContractError, match="instance_segmentation_prediction"):
        INSTANCE_SEGMENTATION_VIEW_KIND.check(view.full_res())
    with pytest.raises(ContractError):
        INSTANCE_SEGMENTATION_PREDICTION_KIND.check(view)


def test_view_kind_refuses_hidden_serialization() -> None:
    view = _view(LETTERBOX, (8, 8))

    with pytest.raises(ContractError, match="full_res"):
        INSTANCE_SEGMENTATION_VIEW_KIND.to_serialized(view)
    assert INSTANCE_SEGMENTATION_VIEW_KIND.to_payload(view) is view
    assert view.materialization_count == 0


def test_catalogue_registers_the_view_kind_without_a_recording_codec() -> None:
    catalogue = blocks.create_catalogue()

    assert catalogue.kinds["instance_segmentation_view"] is (
        INSTANCE_SEGMENTATION_VIEW_KIND
    )
    assert blocks.SegmentationView is SegmentationView
    assert blocks.MaskGridGeometry is MaskGridGeometry
    codec_types = {codec.type_name for codec in catalogue.codecs.values()}
    assert type_name_of(SegmentationView) not in codec_types


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_view_materializes_on_its_device() -> None:
    scores = torch.randn((2, 8, 8), device="cuda")
    view = SegmentationView(
        detections=_detections(2, device="cuda"),
        scores=scores,
        score_type="logits",
        threshold=0.0,
        geometry=MaskGridGeometry.from_pre_processing(LETTERBOX, grid_size_hw=(8, 8)),
    )

    assert view.full_res().mask.device.type == "cuda"


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
# GPU clock cycles; long enough that unordered reads see the old buffer values.
_DELAY_CYCLES = 200_000_000


def _cuda_raw_output(device) -> tuple:
    """A model-like raw output and its detections, complete on the device."""
    generator = torch.Generator().manual_seed(7)
    raw = torch.randn((5, 8, 8), generator=generator).to(device)
    boxes = torch.tensor([[1, 2, 30, 40], [5, 5, 60, 50], [0, 0, 99, 60]]).to(device)
    torch.cuda.synchronize(device)

    return raw, boxes


def _cuda_detections(boxes: torch.Tensor) -> Detections:
    detections = Detections(
        xyxy=boxes.clone(),
        class_id=torch.arange(3, dtype=torch.int32, device=boxes.device),
        confidence=torch.full((3,), 0.5, device=boxes.device),
    )

    return detections


def _synchronous_reference(raw, boxes, rows) -> InstanceDetections:
    view = SegmentationView.from_selected_rows(
        raw,
        rows=rows,
        detections=_cuda_detections(boxes),
        score_type="logits",
        threshold=0.0,
        geometry=MaskGridGeometry.from_pre_processing(LETTERBOX, grid_size_hw=(8, 8)),
    )
    dense = view.full_res()
    torch.cuda.synchronize(raw.device)

    return dense


@needs_cuda
def test_cuda_view_orders_delayed_producer_materializer_and_consumer() -> None:
    device = torch.device("cuda")
    source, boxes = _cuda_raw_output(device)
    rows = torch.tensor([2, 0, 0], device=device)
    expected = _synchronous_reference(source, boxes, rows)
    producer, materializer, consumer = (torch.cuda.Stream(device) for _ in range(3))

    # The producer finishes last: an unordered materializer would read stale
    # scores, and an unordered consumer would read the mask before it exists.
    with torch.cuda.stream(producer):
        torch.cuda._sleep(2 * _DELAY_CYCLES)
        raw = source * 1.0
        view = SegmentationView.from_selected_rows(
            raw,
            rows=rows,
            detections=_cuda_detections(boxes),
            score_type="logits",
            threshold=0.0,
            geometry=MaskGridGeometry.from_pre_processing(
                LETTERBOX, grid_size_hw=(8, 8)
            ),
        )
        raw.fill_(-10.0)  # the model reuses its output buffer
    with torch.cuda.stream(materializer):
        torch.cuda._sleep(_DELAY_CYCLES)
        view.full_res()
    with torch.cuda.stream(consumer):
        dense = view.full_res()
        areas = dense.mask.sum(dim=(1, 2))
        cells = view.low_res_binary().sum(dim=(1, 2))
    torch.cuda.synchronize(device)

    assert view.materialization_count == 1
    assert torch.equal(dense.mask, expected.mask)
    assert torch.equal(areas, expected.mask.sum(dim=(1, 2)))
    assert torch.equal(cells, (source[[2, 0, 0], 2:6] > 0).sum(dim=(1, 2)))


@needs_cuda
def test_cuda_concurrent_callers_on_own_streams_share_one_result() -> None:
    device = torch.device("cuda")
    source, boxes = _cuda_raw_output(device)
    rows = torch.tensor([0, 1, 2], device=device)
    expected = _synchronous_reference(source, boxes, rows)
    producer = torch.cuda.Stream(device)
    with torch.cuda.stream(producer):
        torch.cuda._sleep(_DELAY_CYCLES)
        view = SegmentationView.from_selected_rows(
            source * 1.0,
            rows=rows,
            detections=_cuda_detections(boxes),
            score_type="logits",
            threshold=0.0,
            geometry=MaskGridGeometry.from_pre_processing(
                LETTERBOX, grid_size_hw=(8, 8)
            ),
        )
    results, areas = [], []

    def consume() -> None:
        with torch.cuda.stream(torch.cuda.Stream(device)):
            dense = view.full_res()
            results.append(dense)
            areas.append(dense.mask.sum(dim=(1, 2)))

    threads = [threading.Thread(target=consume) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    torch.cuda.synchronize(device)

    assert len(results) == 4
    assert all(result is results[0] for result in results)
    assert view.materialization_count == 1
    for area in areas:
        assert torch.equal(area, expected.mask.sum(dim=(1, 2)))


@needs_cuda
def test_cuda_dense_fallback_orders_its_consumer_after_the_producer() -> None:
    device = torch.device("cuda")
    source = torch.ones((2, 20, 30), dtype=torch.bool, device=device)
    torch.cuda.synchronize(device)
    producer, consumer = torch.cuda.Stream(device), torch.cuda.Stream(device)

    with torch.cuda.stream(producer):
        torch.cuda._sleep(_DELAY_CYCLES)
        predictions = InstanceDetections(
            xyxy=torch.zeros((2, 4), device=device),
            class_id=torch.zeros(2, dtype=torch.int32, device=device),
            confidence=torch.ones(2, device=device),
            mask=source.logical_not().logical_not(),
        )
        view = SegmentationView.from_dense(predictions)
    with torch.cuda.stream(consumer):
        selected = view.select(torch.tensor([1, 0], device=device))
        areas = ops.instance_areas(selected).values
    torch.cuda.synchronize(device)

    assert selected.is_dense_fallback
    assert areas.tolist() == [600.0, 600.0]


_BORROWED_ROWS = 65_536


def _prediction_layouts(predictions) -> list:
    """(shape, dtype) of every tensor field, to reallocate storage of each size."""
    tensors = [predictions.xyxy, predictions.class_id, predictions.confidence]
    if isinstance(predictions, InstanceDetections):
        tensors.append(predictions.mask)
    layouts = [(tensor.shape, tensor.dtype) for tensor in tensors]

    return layouts


def _borrowed_dense_mask_read(side: torch.cuda.Stream) -> tuple:
    """A dense fallback borrows a default-stream mask; the side stream reads it."""
    predictions = InstanceDetections(
        xyxy=torch.zeros((1, 4), device="cuda"),
        class_id=torch.zeros(1, dtype=torch.int32, device="cuda"),
        confidence=torch.ones(1, device="cuda"),
        mask=torch.ones((1, 1024, 1024), dtype=torch.bool, device="cuda"),
    )
    torch.cuda.synchronize()
    with torch.cuda.stream(side):
        torch.cuda._sleep(_DELAY_CYCLES)
        total = SegmentationView.from_dense(predictions).full_res().mask.sum()

    return total, _prediction_layouts(predictions), 1024 * 1024


def _borrowed_detections_read(side: torch.cuda.Stream) -> tuple:
    """A low-res view borrows default-stream detections; the side stream reads them."""
    detections = Detections(
        xyxy=torch.zeros((_BORROWED_ROWS, 4), device="cuda"),
        class_id=torch.zeros(_BORROWED_ROWS, dtype=torch.int32, device="cuda"),
        confidence=torch.ones(_BORROWED_ROWS, device="cuda"),
    )
    torch.cuda.synchronize()
    with torch.cuda.stream(side):
        torch.cuda._sleep(_DELAY_CYCLES)
        view = SegmentationView(
            detections=detections,
            scores=torch.zeros((_BORROWED_ROWS, 2, 2), device="cuda"),
            score_type="logits",
            threshold=0.0,
            geometry=MaskGridGeometry.identity((2, 2)),
        )
        total = view.detections.confidence.sum()

    return total, _prediction_layouts(detections), _BORROWED_ROWS


@needs_cuda
@pytest.mark.parametrize(
    "borrowed_read", [_borrowed_dense_mask_read, _borrowed_detections_read]
)
def test_cuda_borrowed_storage_outlives_its_owners_on_the_readiness_stream(
    borrowed_read,
) -> None:
    # Storage from the default stream, readiness and a delayed read on a side
    # stream. Every owner is gone when the helper returns; the default stream
    # then allocates and zeroes storage of every borrowed size, all alive at
    # once so that each takes its own block. Only the allocator's record of
    # the side stream's use keeps that storage from being reused before the
    # read completes.
    side = torch.cuda.Stream()
    for _ in range(3):
        total, layouts, expected = borrowed_read(side)
        replacements = [
            torch.empty(shape, dtype=dtype, device="cuda") for shape, dtype in layouts
        ]
        for replacement in replacements:
            replacement.zero_()
        torch.cuda.synchronize()

        assert int(total) == expected


@needs_cuda
def test_cuda_detections_on_another_device_are_rejected() -> None:
    with pytest.raises(ContractError, match="detections on cpu"):
        SegmentationView(
            detections=_detections(2),
            scores=torch.randn((2, 8, 8), device="cuda"),
            score_type="logits",
            threshold=0.0,
            geometry=MaskGridGeometry.from_pre_processing(
                LETTERBOX, grid_size_hw=(8, 8)
            ),
        )
