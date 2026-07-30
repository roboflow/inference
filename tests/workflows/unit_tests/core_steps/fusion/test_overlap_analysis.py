"""
Unit tests for `OverlapAnalysisBlockV1` (`roboflow_core/overlap_analysis@v1`).

The test set mirrors the logical branches of the original Python-block code
this generic block replaces:

    for b in reference:
        for a in candidate:
            if not b.bbox.intersects(a.bbox):
                continue
            intersection_area = b.poly.intersection(a.poly).area
            percent_overlap = intersection_area / b.poly.area
            if percent_overlap >= 0.10:
                results.append({...})

so it exercises: bbox-only path, mask path, threshold boundary, empty
inputs, optional detection_id propagation, and an N x M cardinality check.
A parity test compares the new block against a verbatim port of the
original (bbox-only fixtures, where the polygon-construction paths of
both implementations are mathematically identical and bit-for-bit
agreement is meaningful).
"""

import json
from typing import List

import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError
from shapely.geometry import Polygon, box

from inference.core.workflows.core_steps import loader
from inference.core.workflows.core_steps.fusion.overlap_analysis.v1 import (
    BlockManifest,
    OverlapAnalysisBlockV1,
)
from inference.core.workflows.execution_engine.entities.types import (
    DETECTIONS_OVERLAPS_KIND,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _detections_from_xyxy(
    xyxy: np.ndarray,
    *,
    class_names: List[str] = None,
    confidences: List[float] = None,
    detection_ids: List[str] = None,
    masks: np.ndarray = None,
) -> sv.Detections:
    n = len(xyxy)
    data = {}
    if class_names is not None:
        data["class_name"] = np.array(class_names)
    if detection_ids is not None:
        data["detection_id"] = np.array(detection_ids)
    return sv.Detections(
        xyxy=np.asarray(xyxy, dtype=float),
        confidence=(
            np.asarray(confidences, dtype=float) if confidences is not None else None
        ),
        class_id=np.zeros(n, dtype=int),
        mask=masks,
        data=data,
    )


def _mask_from_polygon(polygon: np.ndarray, resolution_wh: tuple) -> np.ndarray:
    return sv.polygon_to_mask(polygon=polygon, resolution_wh=resolution_wh).astype(bool)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_manifest_validation_when_valid_input_provided() -> None:
    # given
    specification = {
        "type": "roboflow_core/overlap_analysis@v1",
        "name": "overlap",
        "reference_predictions": "$steps.model_a.predictions",
        "candidate_predictions": "$steps.model_b.predictions",
        "min_overlap": 0.2,
    }

    # when
    manifest = BlockManifest.model_validate(specification)

    # then
    assert manifest.type == "roboflow_core/overlap_analysis@v1"
    assert manifest.reference_predictions == "$steps.model_a.predictions"
    assert manifest.candidate_predictions == "$steps.model_b.predictions"
    assert manifest.min_overlap == 0.2


def test_manifest_accepts_short_type_alias() -> None:
    # given
    specification = {
        "type": "OverlapAnalysis",
        "name": "overlap",
        "reference_predictions": "$steps.model_a.predictions",
        "candidate_predictions": "$steps.model_b.predictions",
    }

    # when
    manifest = BlockManifest.model_validate(specification)

    # then
    assert manifest.type == "OverlapAnalysis"
    # default kicks in
    assert manifest.min_overlap == 0.1


def test_manifest_rejects_min_overlap_above_one() -> None:
    # given
    specification = {
        "type": "roboflow_core/overlap_analysis@v1",
        "name": "overlap",
        "reference_predictions": "$steps.model_a.predictions",
        "candidate_predictions": "$steps.model_b.predictions",
        "min_overlap": 1.5,
    }

    # when
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(specification)


# ---------------------------------------------------------------------------
# Block behaviour — bbox-only path
# ---------------------------------------------------------------------------


def test_run_with_bbox_only_inputs_full_containment() -> None:
    # given: reference fully inside candidate
    ref = _detections_from_xyxy(
        np.array([[10, 10, 30, 30]]),
        class_names=["R"],
        confidences=[0.9],
    )
    cand = _detections_from_xyxy(
        np.array([[0, 0, 100, 100]]),
        class_names=["C"],
        confidences=[0.7],
    )
    block = OverlapAnalysisBlockV1()

    # when
    result = block.run(
        reference_predictions=ref, candidate_predictions=cand, min_overlap=0.1
    )

    # then
    assert len(result["overlaps"]) == 1
    record = result["overlaps"][0]
    assert record["overlap_ratio"] == pytest.approx(1.0)
    assert record["reference_class"] == "R"
    assert record["candidate_class"] == "C"
    assert record["reference_confidence"] == pytest.approx(0.9)
    assert record["candidate_confidence"] == pytest.approx(0.7)
    # detection_id absent on input → absent on output
    assert "reference_detection_id" not in record
    assert "candidate_detection_id" not in record


def test_run_with_bbox_only_inputs_partial_overlap() -> None:
    # given: 50 % overlap on a 100x100 reference
    # reference: (0,0)-(100,100)  area = 10000
    # candidate: (50,0)-(150,100) intersection = (50,0)-(100,100) area = 5000
    # overlap_ratio = 5000 / 10000 = 0.5
    ref = _detections_from_xyxy(np.array([[0, 0, 100, 100]]), confidences=[0.5])
    cand = _detections_from_xyxy(np.array([[50, 0, 150, 100]]), confidences=[0.5])
    block = OverlapAnalysisBlockV1()

    # when
    result = block.run(
        reference_predictions=ref, candidate_predictions=cand, min_overlap=0.1
    )

    # then
    assert len(result["overlaps"]) == 1
    assert result["overlaps"][0]["overlap_ratio"] == pytest.approx(0.5)


def test_run_below_threshold_returns_empty() -> None:
    # given: only ~5% overlap, threshold 0.1
    ref = _detections_from_xyxy(np.array([[0, 0, 100, 100]]))
    cand = _detections_from_xyxy(np.array([[95, 0, 195, 100]]))  # 5% overlap
    block = OverlapAnalysisBlockV1()

    # when
    result = block.run(
        reference_predictions=ref, candidate_predictions=cand, min_overlap=0.1
    )

    # then
    assert result == {"overlaps": []}


def test_run_at_threshold_boundary_just_above_passes() -> None:
    # given: reference 100x100, candidate shifted so intersection = 11x100 = 1100,
    # overlap_ratio = 0.11 > 0.10 threshold
    ref = _detections_from_xyxy(np.array([[0, 0, 100, 100]]))
    cand = _detections_from_xyxy(np.array([[89, 0, 189, 100]]))
    block = OverlapAnalysisBlockV1()

    # when
    result = block.run(
        reference_predictions=ref, candidate_predictions=cand, min_overlap=0.1
    )

    # then
    assert len(result["overlaps"]) == 1
    assert result["overlaps"][0]["overlap_ratio"] == pytest.approx(0.11)


def test_run_at_threshold_boundary_just_below_is_dropped() -> None:
    # given: intersection = 9x100 = 900, overlap_ratio = 0.09 < 0.10
    ref = _detections_from_xyxy(np.array([[0, 0, 100, 100]]))
    cand = _detections_from_xyxy(np.array([[91, 0, 191, 100]]))
    block = OverlapAnalysisBlockV1()

    # when
    result = block.run(
        reference_predictions=ref, candidate_predictions=cand, min_overlap=0.1
    )

    # then
    assert result == {"overlaps": []}


def test_run_with_disjoint_detections_returns_empty() -> None:
    # given: bboxes don't touch
    ref = _detections_from_xyxy(np.array([[0, 0, 10, 10]]))
    cand = _detections_from_xyxy(np.array([[100, 100, 200, 200]]))
    block = OverlapAnalysisBlockV1()

    # when
    result = block.run(
        reference_predictions=ref, candidate_predictions=cand, min_overlap=0.1
    )

    # then
    assert result == {"overlaps": []}


# ---------------------------------------------------------------------------
# Block behaviour — mask path
# ---------------------------------------------------------------------------


def test_run_with_mask_inputs_uses_mask_path() -> None:
    # given: rectangular masks inside identical bboxes.
    # Both bboxes are 100x100; bbox-only IoU would give ratio = 1.0.
    # Mask shapes share only a known fraction of area, so the mask path
    # must produce a ratio noticeably less than 1.0.
    #   reference mask: left 60% of the bbox  -> (10,10)-(70,90), area ~ 4800
    #   candidate mask: right 60% of the bbox -> (40,10)-(100,90), area ~ 4800
    #   intersection: (40,10)-(70,90), area ~ 2400
    #   expected mask-based ratio ~ 2400 / 4800 = 0.5
    resolution = (200, 200)
    ref_poly = np.array([[10, 10], [70, 10], [70, 90], [10, 90]])
    cand_poly = np.array([[40, 10], [100, 10], [100, 90], [40, 90]])
    ref_mask = _mask_from_polygon(ref_poly, resolution)
    cand_mask = _mask_from_polygon(cand_poly, resolution)
    ref = sv.Detections(
        xyxy=np.array([[0, 0, 100, 100]], dtype=float),
        confidence=np.array([1.0]),
        class_id=np.array([0]),
        mask=np.array([ref_mask]),
        data={"class_name": np.array(["R"])},
    )
    cand = sv.Detections(
        xyxy=np.array([[0, 0, 100, 100]], dtype=float),
        confidence=np.array([1.0]),
        class_id=np.array([0]),
        mask=np.array([cand_mask]),
        data={"class_name": np.array(["C"])},
    )
    block = OverlapAnalysisBlockV1()

    # when
    result = block.run(
        reference_predictions=ref, candidate_predictions=cand, min_overlap=0.01
    )

    # then
    assert len(result["overlaps"]) == 1
    ratio = result["overlaps"][0]["overlap_ratio"]
    # Bbox-only IoU would be 1.0 (identical bboxes). Mask overlap is ~0.5;
    # allow a tolerance to absorb rasterisation rounding.
    assert 0.4 < ratio < 0.6, (
        f"Expected mask-based overlap ~0.5, got {ratio}. "
        "If this fails, the block is probably using bbox polygons even when "
        "masks are available."
    )


def test_run_with_invalid_polygon_falls_back_to_bbox() -> None:
    # given: mask whose longest contour is a bowtie (self-intersecting).
    # The shapely Polygon would be invalid and the helper must fall back to
    # the 4-corner bbox polygon.
    mask = np.zeros((100, 100), dtype=bool)
    # Two disjoint pixels in the mask means `mask_to_polygons` returns
    # multiple short polygons — the longest is just a single pixel which
    # cannot form a valid Polygon (<3 vertices). The helper falls back to
    # the bbox.
    mask[10, 10] = True
    mask[10, 11] = True
    ref = sv.Detections(
        xyxy=np.array([[0, 0, 100, 100]], dtype=float),
        confidence=np.array([1.0]),
        class_id=np.array([0]),
        mask=np.array([mask]),
        data={"class_name": np.array(["R"])},
    )
    cand = _detections_from_xyxy(
        np.array([[0, 0, 100, 100]]),
        class_names=["C"],
        confidences=[1.0],
    )
    block = OverlapAnalysisBlockV1()

    # when
    result = block.run(
        reference_predictions=ref, candidate_predictions=cand, min_overlap=0.1
    )

    # then
    # Fallback to bbox polygon → reference is a full 100x100 box → fully
    # contained in candidate → overlap_ratio = 1.0.
    assert len(result["overlaps"]) == 1
    assert result["overlaps"][0]["overlap_ratio"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# detection_id propagation
# ---------------------------------------------------------------------------


def test_run_propagates_detection_ids_when_present_on_both_sides() -> None:
    # given
    ref = _detections_from_xyxy(
        np.array([[0, 0, 100, 100]]),
        class_names=["R"],
        confidences=[1.0],
        detection_ids=["ref-1"],
    )
    cand = _detections_from_xyxy(
        np.array([[0, 0, 100, 100]]),
        class_names=["C"],
        confidences=[1.0],
        detection_ids=["cand-1"],
    )
    block = OverlapAnalysisBlockV1()

    # when
    result = block.run(
        reference_predictions=ref, candidate_predictions=cand, min_overlap=0.1
    )

    # then
    record = result["overlaps"][0]
    assert record["reference_detection_id"] == "ref-1"
    assert record["candidate_detection_id"] == "cand-1"


def test_run_propagates_only_present_detection_ids() -> None:
    # given: only candidate has detection_id
    ref = _detections_from_xyxy(
        np.array([[0, 0, 100, 100]]),
        class_names=["R"],
        confidences=[1.0],
    )
    cand = _detections_from_xyxy(
        np.array([[0, 0, 100, 100]]),
        class_names=["C"],
        confidences=[1.0],
        detection_ids=["cand-1"],
    )
    block = OverlapAnalysisBlockV1()

    # when
    result = block.run(
        reference_predictions=ref, candidate_predictions=cand, min_overlap=0.1
    )

    # then
    record = result["overlaps"][0]
    assert "reference_detection_id" not in record
    assert record["candidate_detection_id"] == "cand-1"


# ---------------------------------------------------------------------------
# Empty inputs
# ---------------------------------------------------------------------------


def test_run_with_empty_reference_returns_empty() -> None:
    # given
    ref = sv.Detections.empty()
    cand = _detections_from_xyxy(
        np.array([[0, 0, 100, 100]]),
        class_names=["C"],
        confidences=[1.0],
    )
    block = OverlapAnalysisBlockV1()

    # when
    result = block.run(
        reference_predictions=ref, candidate_predictions=cand, min_overlap=0.1
    )

    # then
    assert result == {"overlaps": []}


def test_run_with_empty_candidate_returns_empty() -> None:
    # given
    ref = _detections_from_xyxy(
        np.array([[0, 0, 100, 100]]),
        class_names=["R"],
        confidences=[1.0],
    )
    cand = sv.Detections.empty()
    block = OverlapAnalysisBlockV1()

    # when
    result = block.run(
        reference_predictions=ref, candidate_predictions=cand, min_overlap=0.1
    )

    # then
    assert result == {"overlaps": []}


# ---------------------------------------------------------------------------
# Cardinality
# ---------------------------------------------------------------------------


def test_run_n_to_m_pairs_emits_only_pairs_above_threshold() -> None:
    # given: 2 reference x 3 candidate; only some pairs overlap above 0.1.
    ref = _detections_from_xyxy(
        np.array(
            [
                [0, 0, 100, 100],  # ref 0
                [200, 200, 300, 300],  # ref 1 (disjoint from all candidates)
            ]
        ),
        class_names=["R0", "R1"],
        confidences=[1.0, 1.0],
        detection_ids=["r0", "r1"],
    )
    cand = _detections_from_xyxy(
        np.array(
            [
                [50, 0, 150, 100],  # cand 0 - 50% overlap with ref 0
                [80, 0, 180, 100],  # cand 1 - 20% overlap with ref 0
                [400, 400, 500, 500],  # cand 2 - no overlap with anything
            ]
        ),
        class_names=["C0", "C1", "C2"],
        confidences=[1.0, 1.0, 1.0],
        detection_ids=["c0", "c1", "c2"],
    )
    block = OverlapAnalysisBlockV1()

    # when
    result = block.run(
        reference_predictions=ref, candidate_predictions=cand, min_overlap=0.1
    )

    # then
    overlaps = result["overlaps"]
    pairs = {
        (o["reference_detection_id"], o["candidate_detection_id"]) for o in overlaps
    }
    assert pairs == {("r0", "c0"), ("r0", "c1")}
    ratios = {
        (o["reference_detection_id"], o["candidate_detection_id"]): o["overlap_ratio"]
        for o in overlaps
    }
    assert ratios[("r0", "c0")] == pytest.approx(0.5)
    assert ratios[("r0", "c1")] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Parity against verbatim port of the original user-supplied code
# ---------------------------------------------------------------------------


def _legacy_overlap_records(reference, candidate, *, threshold=0.10) -> list:
    """Verbatim logical port of the original Python-block code that this
    generic block replaces. Bbox-only path (no skimage dependency in the
    test), so comparison is meaningful for bbox inputs."""

    def get_shapely_poly(detections, idx):
        x1, y1, x2, y2 = detections.xyxy[idx]
        return box(x1, y1, x2, y2), box(x1, y1, x2, y2)

    refs = []
    for i in range(len(reference.xyxy)):
        poly, b_box = get_shapely_poly(reference, i)
        refs.append(
            {
                "poly": poly,
                "bbox": b_box,
                "conf": float(reference.confidence[i]),
                "class": reference.data.get("class_name", [])[i],
            }
        )
    cands = []
    for j in range(len(candidate.xyxy)):
        poly, b_box = get_shapely_poly(candidate, j)
        cands.append(
            {
                "poly": poly,
                "bbox": b_box,
                "conf": float(candidate.confidence[j]),
                "class": candidate.data.get("class_name", [])[j],
            }
        )

    out = []
    for b in refs:
        for a in cands:
            if not b["bbox"].intersects(a["bbox"]):
                continue
            intersection_area = b["poly"].intersection(a["poly"]).area
            percent_overlap = intersection_area / b["poly"].area
            if percent_overlap >= threshold:
                out.append(
                    {
                        "reference_class": b["class"],
                        "reference_confidence": b["conf"],
                        "candidate_class": a["class"],
                        "candidate_confidence": a["conf"],
                        "overlap_ratio": percent_overlap,
                    }
                )
    return out


def test_parity_against_original_code_on_bbox_only_inputs() -> None:
    # given: 3 x 4 grid with mixed overlaps
    ref = _detections_from_xyxy(
        np.array(
            [
                [0, 0, 100, 100],
                [200, 200, 300, 300],
                [50, 50, 150, 150],
            ]
        ),
        class_names=["alpha", "beta", "gamma"],
        confidences=[0.9, 0.8, 0.7],
    )
    cand = _detections_from_xyxy(
        np.array(
            [
                [50, 0, 150, 100],  # overlaps ref 0
                [80, 0, 180, 100],  # overlaps ref 0
                [400, 400, 500, 500],  # disjoint
                [100, 100, 200, 200],  # overlaps ref 2
            ]
        ),
        class_names=["x", "y", "z", "w"],
        confidences=[0.6, 0.5, 0.4, 0.3],
    )
    block = OverlapAnalysisBlockV1()

    # when
    new_records = block.run(
        reference_predictions=ref, candidate_predictions=cand, min_overlap=0.1
    )["overlaps"]
    legacy_records = _legacy_overlap_records(ref, cand, threshold=0.1)

    # then: same set of (class-pair, ratio) records
    def _record_key(r):
        return (
            r["reference_class"],
            r["candidate_class"],
            round(r["overlap_ratio"], 9),
        )

    assert sorted(map(_record_key, new_records)) == sorted(
        map(_record_key, legacy_records)
    )


def test_detections_overlaps_kind_has_registered_serializer_and_deserializer() -> None:
    # the kind is declared in load_kinds(), so it must also have both a
    # serializer and a deserializer registered, otherwise workflow output
    # construction silently falls back to raw pass-through.
    assert DETECTIONS_OVERLAPS_KIND.name in loader.KINDS_SERIALIZERS
    assert DETECTIONS_OVERLAPS_KIND.name in loader.KINDS_DESERIALIZERS


def test_overlap_records_survive_registered_serializer_round_trip() -> None:
    # given a real block output
    ref = _detections_from_xyxy(
        np.array([[0, 0, 100, 100]]),
        class_names=["alpha"],
        confidences=[0.9],
    )
    cand = _detections_from_xyxy(
        np.array([[50, 0, 150, 100]]),
        class_names=["x"],
        confidences=[0.6],
    )
    records = OverlapAnalysisBlockV1().run(
        reference_predictions=ref, candidate_predictions=cand, min_overlap=0.1
    )["overlaps"]
    assert records, "fixture must produce at least one overlap record"

    # when we push it through the serializer registered for this kind
    serializer = loader.KINDS_SERIALIZERS[DETECTIONS_OVERLAPS_KIND.name]
    serialized = serializer(records)

    # then the result is plain JSON serializable data
    dumped = json.dumps(serialized)
    assert json.loads(dumped) == serialized
