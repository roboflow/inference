"""SegmentationView full_res against the reference helper given the real boxes.

The view hands ``align_instance_segmentation_results`` host zero boxes, because
the helper's box output is discarded and no box value reaches its mask. These
tests call the unchanged helper with the view's real boxes and require the same
mask, so a helper whose mask starts to depend on box values fails here.
"""

import pytest
import torch
from roboflow_workflows.v2.data.masks import grid
from roboflow_workflows.v2.data.masks.grid import MaskGridGeometry, SegmentationView
from roboflow_workflows.v2.data.row_identity import row_record, with_row_identity

from inference_models.entities import ImageDimensions
from inference_models.models.base.object_detection import Detections
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)
from inference_models.models.common.roboflow.post_processing import (
    align_instance_segmentation_results,
)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


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


# Each case: metadata, grid size, and whether it has padding / a static crop.
GEOMETRY_CASES = {
    "no_padding": (
        _metadata(
            original_hw=(48, 64),
            resized_hw=(48, 64),
            inference_hw=(24, 32),
            padding_ltrb=(0, 0, 0, 0),
        ),
        (6, 8),
    ),
    "letterbox": (
        _metadata(
            original_hw=(61, 100),
            resized_hw=(61, 100),
            inference_hw=(32, 32),
            padding_ltrb=(0, 6, 0, 6),
        ),
        (8, 8),
    ),
    "static_crop": (
        _metadata(
            original_hw=(60, 80),
            resized_hw=(30, 50),
            inference_hw=(16, 24),
            padding_ltrb=(0, 0, 0, 0),
            crop_xywh=(10, 5, 50, 30),
        ),
        (4, 6),
    ),
    "letterboxed_static_crop": (
        _metadata(
            original_hw=(60, 80),
            resized_hw=(30, 50),
            inference_hw=(32, 32),
            padding_ltrb=(0, 6, 0, 6),
            crop_xywh=(10, 5, 50, 30),
        ),
        (8, 8),
    ),
}


def _real_boxes(rows: int, *, original_hw, seed: int, device) -> torch.Tensor:
    """Float boxes in image pixels, some beyond the image so the clamp acts."""
    generator = torch.Generator().manual_seed(seed)
    height, width = original_hw
    corners = torch.rand((rows, 2, 2), generator=generator)
    corners = corners * torch.tensor([width + 20.0, height + 20.0]) - 10.0
    top_left = corners.min(dim=1).values
    bottom_right = corners.max(dim=1).values
    boxes = torch.cat([top_left, bottom_right], dim=1).to(device)

    return boxes


def _view(
    metadata,
    grid_hw,
    *,
    rows=5,
    seed=0,
    score_type="logits",
    threshold=0.0,
    device="cpu",
):
    generator = torch.Generator().manual_seed(seed)
    scores = torch.randn((rows, *grid_hw), generator=generator)
    if score_type == "probabilities":
        scores = scores.sigmoid()
    original_hw = (metadata.original_size.height, metadata.original_size.width)
    detections = Detections(
        xyxy=_real_boxes(rows, original_hw=original_hw, seed=seed, device=device),
        class_id=torch.arange(rows, dtype=torch.int32, device=device),
        confidence=torch.linspace(0.9, 0.5, rows, device=device),
        image_metadata={"parent_id": "frame"},
        bboxes_metadata=[{"row": row} for row in range(rows)],
    )
    view = SegmentationView(
        detections=with_row_identity(detections, namespace="model"),
        scores=scores.to(device),
        score_type=score_type,
        threshold=threshold,
        geometry=MaskGridGeometry.from_pre_processing(metadata, grid_size_hw=grid_hw),
    )

    return view


def _helper_mask_with_real_boxes(view: SegmentationView, metadata) -> torch.Tensor:
    detections = view.detections
    _, mask = align_instance_segmentation_results(
        image_bboxes=detections.xyxy.to(dtype=torch.float32, copy=True),
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


def test_cases_cover_padding_and_static_crops() -> None:
    no_padding, _ = GEOMETRY_CASES["no_padding"]
    letterbox, _ = GEOMETRY_CASES["letterbox"]
    static_crop, _ = GEOMETRY_CASES["static_crop"]

    assert (no_padding.pad_left, no_padding.pad_top) == (0, 0)
    assert no_padding.static_crop_offset.offset_x == 0
    assert letterbox.pad_top > 0
    assert static_crop.static_crop_offset.offset_x > 0


@pytest.mark.parametrize("case", sorted(GEOMETRY_CASES))
@pytest.mark.parametrize(
    "score_type,threshold", [("logits", 0.0), ("probabilities", 0.5)]
)
def test_full_res_equals_helper_given_real_boxes(case, score_type, threshold) -> None:
    metadata, grid_hw = GEOMETRY_CASES[case]
    view = _view(metadata, grid_hw, score_type=score_type, threshold=threshold)

    expected = _helper_mask_with_real_boxes(view, metadata)

    assert expected.any() and not expected.all()
    assert torch.equal(view.full_res().mask, expected)
    assert torch.equal(view.compute_full_res().mask, expected)


@pytest.mark.parametrize("case", sorted(GEOMETRY_CASES))
def test_empty_view_materializes_an_empty_bool_mask(case) -> None:
    metadata, grid_hw = GEOMETRY_CASES[case]
    view = _view(metadata, grid_hw, rows=0)

    mask = view.full_res().mask

    expected = _helper_mask_with_real_boxes(view, metadata)
    assert mask.dtype == expected.dtype == torch.bool
    assert tuple(mask.shape) == tuple(expected.shape)
    assert tuple(mask.shape) == (0, *view.geometry.original_size_hw)
    assert mask.device == view.scores.device
    assert view.materialization_count == 1


def test_materialization_leaves_source_and_row_record_unchanged() -> None:
    metadata, grid_hw = GEOMETRY_CASES["letterboxed_static_crop"]
    view = _view(metadata, grid_hw)
    detections = view.detections
    image_metadata = detections.image_metadata
    record_before = row_record(detections)
    xyxy, class_id, confidence = (
        detections.xyxy.clone(),
        detections.class_id.clone(),
        detections.confidence.clone(),
    )
    scores = view.scores.clone()

    dense = view.full_res()

    assert torch.equal(detections.xyxy, xyxy)
    assert torch.equal(detections.class_id, class_id)
    assert torch.equal(detections.confidence, confidence)
    assert torch.equal(view.scores, scores)
    assert dense.xyxy is detections.xyxy
    assert dense.image_metadata is image_metadata
    assert dense.bboxes_metadata is detections.bboxes_metadata
    assert row_record(detections) == record_before
    assert row_record(dense) == record_before


def test_placeholder_boxes_stay_on_the_host_under_a_default_device(
    monkeypatch,
) -> None:
    metadata, grid_hw = GEOMETRY_CASES["letterboxed_static_crop"]
    view = _view(metadata, grid_hw, device="cpu")
    expected = _helper_mask_with_real_boxes(view, metadata)
    box_devices = []

    def recording_helper(*, image_bboxes, **kwargs):
        box_devices.append(image_bboxes.device)
        return align_instance_segmentation_results(image_bboxes=image_bboxes, **kwargs)

    monkeypatch.setattr(grid, "align_instance_segmentation_results", recording_helper)

    # A "meta" default device stands in for a CUDA one: a placeholder that
    # follows the default device would land off the host.
    with torch.device("meta"):
        mask = view.full_res().mask

    assert box_devices == [torch.device("cpu")]
    assert torch.equal(mask, expected)


@needs_cuda
@pytest.mark.parametrize("case", sorted(GEOMETRY_CASES))
def test_cuda_full_res_equals_helper_given_real_boxes(case) -> None:
    metadata, grid_hw = GEOMETRY_CASES[case]
    view = _view(metadata, grid_hw, device="cuda")

    mask = view.full_res().mask

    expected = _helper_mask_with_real_boxes(view, metadata)
    assert mask.device.type == "cuda"
    assert torch.equal(mask, expected)


@needs_cuda
def test_cuda_empty_view_mask_stays_on_the_scores_device() -> None:
    metadata, grid_hw = GEOMETRY_CASES["letterbox"]
    view = _view(metadata, grid_hw, rows=0, device="cuda")

    mask = view.full_res().mask

    assert mask.device == view.scores.device
    assert tuple(mask.shape) == (0, *view.geometry.original_size_hw)
    assert mask.dtype == torch.bool


@needs_cuda
@pytest.mark.parametrize("case", ["letterbox", "letterboxed_static_crop"])
def test_cuda_first_full_res_makes_no_synchronizing_call(case) -> None:
    metadata, grid_hw = GEOMETRY_CASES[case]
    view = _view(metadata, grid_hw, device="cuda")
    torch.cuda.synchronize()

    # "error" raises on any synchronizing CUDA call, such as a pageable
    # host-to-device copy or a stream synchronization.
    previous_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        dense = view.full_res()
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)

    assert dense.mask.device.type == "cuda"


@needs_cuda
def test_cuda_default_device_full_res_makes_no_synchronizing_call() -> None:
    metadata, grid_hw = GEOMETRY_CASES["letterboxed_static_crop"]
    view = _view(metadata, grid_hw, device="cuda")
    expected = _helper_mask_with_real_boxes(view, metadata)
    torch.cuda.synchronize()

    previous_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.device("cuda"):
            mask = view.full_res().mask
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)

    assert torch.equal(mask, expected)
