"""Analytic overlap ratios, record semantics and reference mask demands."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from inference_models.entities import ImageDimensions
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)
from pycocotools import mask as mask_utils
from pydantic import ValidationError
from roboflow_workflows.v2.blocks.fusion import overlap_analysis as module
from roboflow_workflows.v2.blocks.fusion.overlap_analysis import (
    DETECTIONS_OVERLAPS_KIND,
    OverlapAnalysisBlock,
)
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks.grid import MaskGridGeometry, SegmentationView


def _prediction(boxes, *, mask=None, ids=None, confidence=True):
    image = ImageData.from_tensor(
        torch.zeros((3, 16, 16), dtype=torch.uint8), image_id="frame"
    )
    fields = dict(
        xyxy=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        class_id=torch.arange(len(boxes)),
        confidence=torch.full((len(boxes),), 0.75) if confidence else None,
        image_metadata={**image.prediction_metadata(), "class_names": {0: "crate"}},
        bboxes_metadata=(
            None if ids is None else [{"detection_id": value} for value in ids]
        ),
    )
    result = (
        Detections(**fields)
        if mask is None
        else InstanceDetections(**fields, mask=mask)
    )
    return result


def _run(reference, candidate, *, threshold=0.1):
    result = OverlapAnalysisBlock().run(
        reference_predictions=reference,
        candidate_predictions=candidate,
        min_overlap=threshold,
    )["overlaps"]
    DETECTIONS_OVERLAPS_KIND.check(result)
    return result


def test_asymmetric_inclusive_ratio_record_schema_order_and_metadata():
    reference = _prediction([[0, 0, 8, 8], [0, 0, 4, 4]], ids=["ref", None])
    candidate = _prediction(
        [[0, 0, 4, 4], [4, 0, 8, 4]], confidence=False, ids=[None, "cand"]
    )
    records = _run(reference, candidate, threshold=0.25)
    assert records == [
        dict(
            reference_class="crate",
            reference_confidence=0.75,
            candidate_class="crate",
            candidate_confidence=None,
            overlap_ratio=0.25,
            reference_detection_id="ref",
            candidate_detection_id=None,
        ),
        dict(
            reference_class="crate",
            reference_confidence=0.75,
            candidate_class="class_1",
            candidate_confidence=None,
            overlap_ratio=0.25,
            reference_detection_id="ref",
            candidate_detection_id="cand",
        ),
        dict(
            reference_class="class_1",
            reference_confidence=0.75,
            candidate_class="crate",
            candidate_confidence=None,
            overlap_ratio=1.0,
            reference_detection_id=None,
            candidate_detection_id=None,
        ),
    ]
    assert _run(candidate, reference, threshold=1)[0]["overlap_ratio"] == 1
    assert (
        _run(reference, candidate, threshold=np.nextafter(0.25, 1))[0][
            "reference_class"
        ]
        == "class_1"
    )
    assert reference.bboxes_metadata == [
        {"detection_id": "ref"},
        {"detection_id": None},
    ]
    assert (
        "candidate_detection_id"
        not in _run(reference, _prediction([[0, 0, 4, 4]], ids=[None]))[0]
    )


@pytest.mark.parametrize("boxes", [[], [[8, 0, 12, 4]], [[4, 4, 4, 8]], [[4, 0, 8, 4]]])
def test_empty_disjoint_degenerate_and_touching_boxes_never_emit_at_zero(boxes):
    assert _run(_prediction([[0, 0, 4, 4]]), _prediction(boxes), threshold=0) == []
    assert _run(_prediction([]), _prediction(boxes), threshold=0) == []


@pytest.mark.parametrize("threshold", [-0.1, 1.01, float("nan"), float("inf")])
def test_literal_and_selected_threshold_bounds(threshold):
    with pytest.raises(ValidationError):
        OverlapAnalysisBlock.Params(
            reference_predictions="$inputs.a",
            candidate_predictions="$inputs.b",
            min_overlap=threshold,
        )
    with pytest.raises(ValueError, match="min_overlap"):
        _run(_prediction([]), _prediction([]), threshold=threshold)


def test_default_and_selector_contract():
    params = OverlapAnalysisBlock.Params(
        reference_predictions="$inputs.a", candidate_predictions="$inputs.b"
    )
    assert params.min_overlap == 0.1
    params = OverlapAnalysisBlock.Params(
        reference_predictions="$inputs.a",
        candidate_predictions="$inputs.b",
        min_overlap="$inputs.threshold",
    )
    assert str(params.min_overlap) == "$inputs.threshold"


def test_mask_uses_contour_area_instead_of_pixel_area_and_falls_back_to_box():
    # Full rectangle contour spans 0..7 (area 49), not 64 foreground pixels.
    masks = torch.zeros((1, 16, 16), dtype=torch.bool)
    masks[0, :8, :8] = True
    reference = _prediction([[0, 0, 8, 8]], mask=masks)
    candidate = _prediction([[0, 0, 4, 8]])
    assert _run(reference, candidate, threshold=0)[0]["overlap_ratio"] == pytest.approx(
        4 / 7
    )
    assert _run(candidate, reference, threshold=0)[0]["overlap_ratio"] == pytest.approx(
        7 / 8
    )
    assert (
        _run(replace(reference, mask=torch.zeros_like(masks)), candidate)[0][
            "overlap_ratio"
        ]
        == 0.5
    )
    one_pixel = torch.zeros_like(masks)
    one_pixel[0, 2, 2] = True
    assert (
        _run(replace(reference, mask=one_pixel), candidate)[0]["overlap_ratio"] == 0.5
    )
    assert torch.equal(reference.mask, masks)


def test_longest_contour_can_select_a_hole_boundary():
    masks = torch.zeros((1, 16, 16), dtype=torch.bool)
    masks[0, :8, :8] = True
    masks[0, 2:4, 2:4] = False
    masks[0, 10:13, 10:13] = True
    reference = _prediction([[0, 0, 16, 16]], mask=masks)
    # The hole has eight contour vertices; both filled rectangles have four.
    # V1 therefore selects the hole boundary, despite its smaller area.
    assert (
        _run(reference, _prediction([[1, 1, 4, 4]]), threshold=1)[0]["overlap_ratio"]
        == 1
    )
    assert (
        _run(reference, _prediction([[0, 0, 7, 7]]), threshold=1)[0]["overlap_ratio"]
        == 1
    )
    # Boxes overlap but precise polygons do not: zero threshold still emits.
    candidate = _prediction([[10, 10, 13, 13]])
    assert _run(reference, candidate, threshold=0)[0]["overlap_ratio"] == 0
    assert _run(reference, candidate, threshold=0.01) == []


def _grid_view():
    detections = _prediction([[0, 0, 8, 8], [10, 10, 14, 14]])
    metadata = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(16, 16),
        size_after_pre_processing=ImageDimensions(16, 16),
        inference_size=ImageDimensions(16, 16),
        scale_width=1,
        scale_height=1,
        static_crop_offset=StaticCropOffset(0, 0, 16, 16),
    )
    scores = torch.zeros((2, 8, 8))
    scores[0, :4, :4] = 1
    scores[1, 5:7, 5:7] = 1
    view = SegmentationView(
        detections=detections,
        scores=scores,
        score_type="probabilities",
        threshold=0.5,
        geometry=MaskGridGeometry.from_pre_processing(metadata, grid_size_hw=(8, 8)),
    )
    return view


@pytest.mark.parametrize("representation", ["dense", "rle", "grid"])
def test_only_participating_rows_demand_reference_masks(monkeypatch, representation):
    masks = torch.zeros((2, 16, 16), dtype=torch.bool)
    masks[0, :8, :8] = True
    masks[1, 10:14, 10:14] = True
    if representation == "grid":
        reference = _grid_view()
    else:
        mask = masks
        if representation == "rle":
            encoded = mask_utils.encode(
                np.asfortranarray(masks.numpy().transpose(1, 2, 0).astype(np.uint8))
            )
            mask = InstancesRLEMasks.from_coco_rle_masks((16, 16), encoded)
        reference = _prediction([[0, 0, 8, 8], [10, 10, 14, 14]], mask=mask)
    observed = []
    original = module.mask_provider_of

    class ObservedProvider:
        def select(self, *, indices):
            observed.append(indices)
            return original(reference).select(indices=indices)

    monkeypatch.setattr(module, "mask_provider_of", lambda payload: ObservedProvider())
    candidate = _prediction([[0, 0, 4, 8]])
    assert _run(reference, candidate, threshold=0)[0]["overlap_ratio"] == pytest.approx(
        4 / 7
    )
    assert observed == [[0]]
    assert _run(reference, _prediction([[30, 30, 32, 32]])) == []
    assert observed == [[0]]
    if representation == "grid":
        assert reference.materialization_count == 0


@pytest.mark.parametrize(
    "payload",
    [
        {},
        [dict(overlap_ratio=0.5)],
        [
            dict(
                reference_class="a",
                reference_confidence=None,
                candidate_class="b",
                candidate_confidence=None,
                overlap_ratio=2,
            )
        ],
    ],
)
def test_kind_rejects_malformed_records(payload):
    with pytest.raises(ContractError):
        DETECTIONS_OVERLAPS_KIND.check(payload)


def test_invalid_contour_falls_back_to_box(monkeypatch):
    reference = _prediction(
        [[0, 0, 8, 8]], mask=torch.ones((1, 16, 16), dtype=torch.bool)
    )
    monkeypatch.setattr(
        module.sv,
        "mask_to_polygons",
        lambda **kwargs: [np.array([[0, 0], [8, 8], [0, 8], [8, 0]])],
    )
    assert _run(reference, _prediction([[0, 0, 4, 8]]))[0]["overlap_ratio"] == 0.5


def test_numpy_detection_id_scalars_become_python_scalars():
    reference = _prediction([[0, 0, 4, 4]], ids=[np.int64(42)])
    record = _run(reference, reference)[0]
    assert record["reference_detection_id"] == 42
    assert type(record["reference_detection_id"]) is int


def test_seeded_box_geometry_matches_independent_shapely_rectangles():
    generator = torch.Generator().manual_seed(57)
    sets = []
    for rows in (7, 11):
        origins = torch.rand((rows, 2), generator=generator) * 10
        sizes = torch.rand((rows, 2), generator=generator) * 8 + 1
        sets.append(
            _prediction(
                torch.cat((origins, origins + sizes), dim=1).tolist(),
                ids=list(range(rows)),
            )
        )
    reference, candidate = sets
    expected = []
    for i, a in enumerate(reference.xyxy.tolist()):
        for j, b in enumerate(candidate.xyxy.tolist()):
            ref_poly, cand_poly = module.box(*a), module.box(*b)
            ratio = ref_poly.intersection(cand_poly).area / ref_poly.area
            if ratio >= 0.1:
                expected.append((i, j, ratio))
    actual = _run(reference, candidate)
    assert [
        (r["reference_detection_id"], r["candidate_detection_id"]) for r in actual
    ] == [(i, j) for i, j, _ in expected]
    assert [r["overlap_ratio"] for r in actual] == pytest.approx(
        [ratio for _, _, ratio in expected], abs=1e-12
    )
