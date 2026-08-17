from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytest
import supervision as sv
import torch
from shapely.geometry import Polygon, box

import inference.core.workflows.core_steps.fusion.overlap_analysis.v1_tensor as overlap_module
from inference.core.workflows.core_steps.common.tensor_native import (
    instance_mask_to_numpy,
)
from inference.core.workflows.core_steps.fusion.overlap_analysis.v1_tensor import (
    OverlapAnalysisBlockV1,
    _class_names,
    _detection_ids,
    _safe_get,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import torch_mask_to_coco_rle

NativeDetections = Union[Detections, InstanceDetections]


def _oracle_detection_to_shapely(
    detections: NativeDetections, xyxy: np.ndarray, idx: int
) -> Polygon:
    x1, y1, x2, y2 = xyxy[idx]
    bbox_poly = box(float(x1), float(y1), float(x2), float(y2))
    if isinstance(detections, InstanceDetections) and detections.mask is not None:
        mask = instance_mask_to_numpy(detections, idx)
        if np.any(mask):
            polygons = sv.mask_to_polygons(mask=mask.astype(np.uint8))
            if polygons:
                longest = max(polygons, key=len)
                if len(longest) >= 3:
                    candidate = Polygon(
                        [(float(pt[0]), float(pt[1])) for pt in longest]
                    )
                    if candidate.is_valid and not candidate.is_empty:
                        return candidate
    return bbox_poly


def _oracle_run(
    reference_predictions: NativeDetections,
    candidate_predictions: NativeDetections,
    min_overlap: float,
) -> Dict[str, List[Dict[str, Any]]]:
    if len(reference_predictions) == 0 or len(candidate_predictions) == 0:
        return {"overlaps": []}

    ref_xyxy = reference_predictions.xyxy.detach().to("cpu").numpy()
    cand_xyxy = candidate_predictions.xyxy.detach().to("cpu").numpy()

    iou_matrix = sv.box_iou_batch(ref_xyxy, cand_xyxy)

    ref_ids = _detection_ids(reference_predictions)
    cand_ids = _detection_ids(candidate_predictions)
    ref_class_names = _class_names(reference_predictions)
    cand_class_names = _class_names(candidate_predictions)

    results: List[Dict[str, Any]] = []
    ref_polys: Dict[int, Polygon] = {}
    cand_polys: Dict[int, Polygon] = {}

    for i in range(len(reference_predictions)):
        for j in range(len(candidate_predictions)):
            if iou_matrix[i, j] <= 0.0:
                continue
            if i not in ref_polys:
                ref_polys[i] = _oracle_detection_to_shapely(
                    reference_predictions, ref_xyxy, i
                )
            if j not in cand_polys:
                cand_polys[j] = _oracle_detection_to_shapely(
                    candidate_predictions, cand_xyxy, j
                )
            ref_poly = ref_polys[i]
            cand_poly = cand_polys[j]
            if ref_poly.area <= 0:
                continue
            intersection_area = ref_poly.intersection(cand_poly).area
            overlap_ratio = intersection_area / ref_poly.area
            if overlap_ratio < min_overlap:
                continue
            record: Dict[str, Any] = {
                "reference_class": _safe_get(ref_class_names, i),
                "reference_confidence": (
                    float(reference_predictions.confidence[i])
                    if reference_predictions.confidence is not None
                    else None
                ),
                "candidate_class": _safe_get(cand_class_names, j),
                "candidate_confidence": (
                    float(candidate_predictions.confidence[j])
                    if candidate_predictions.confidence is not None
                    else None
                ),
                "overlap_ratio": float(overlap_ratio),
            }
            if ref_ids is not None:
                record["reference_detection_id"] = _safe_get(ref_ids, i)
            if cand_ids is not None:
                record["candidate_detection_id"] = _safe_get(cand_ids, j)
            results.append(record)
    return {"overlaps": results}


def _assert_results_equal(
    actual: Dict[str, List[Dict[str, Any]]],
    expected: Dict[str, List[Dict[str, Any]]],
    *,
    exact_ratio: bool,
) -> None:
    assert len(actual["overlaps"]) == len(expected["overlaps"])
    for actual_record, expected_record in zip(actual["overlaps"], expected["overlaps"]):
        assert list(actual_record) == list(expected_record)
        for key in actual_record:
            if key == "overlap_ratio" and not exact_ratio:
                assert actual_record[key] == pytest.approx(
                    expected_record[key], rel=1e-9, abs=0.0
                )
            else:
                assert actual_record[key] == expected_record[key]


def _random_boxes(rng: np.random.Generator, count: int) -> np.ndarray:
    centers = rng.normal(30.0, 8.0, size=(count, 2))
    sizes = rng.uniform(0.25, 35.0, size=(count, 2))
    boxes = np.column_stack(
        (
            centers[:, 0] - sizes[:, 0] / 2,
            centers[:, 1] - sizes[:, 1] / 2,
            centers[:, 0] + sizes[:, 0] / 2,
            centers[:, 1] + sizes[:, 1] / 2,
        )
    ).astype(np.float32)
    if count:
        reverse_x = rng.random(count) < 0.25
        reverse_y = rng.random(count) < 0.25
        boxes[reverse_x, 0], boxes[reverse_x, 2] = (
            boxes[reverse_x, 2].copy(),
            boxes[reverse_x, 0].copy(),
        )
        boxes[reverse_y, 1], boxes[reverse_y, 3] = (
            boxes[reverse_y, 3].copy(),
            boxes[reverse_y, 1].copy(),
        )
        degenerate = np.arange(count) % 7 == 0
        boxes[degenerate, 2] = boxes[degenerate, 0]
        horizontal = np.arange(count) % 11 == 0
        boxes[horizontal, 3] = boxes[horizontal, 1]
    return boxes


def _make_bbox_detections(
    boxes: np.ndarray,
    *,
    seed: int,
    instance_without_masks: bool,
    include_class_names: bool,
    include_ids: bool,
    confidence_is_none: bool,
) -> NativeDetections:
    count = len(boxes)
    class_ids = torch.arange(count, dtype=torch.int64) % 5
    confidence: Optional[torch.Tensor]
    confidence = (
        None
        if confidence_is_none
        else torch.linspace(0.05, 0.95, max(count, 1), dtype=torch.float32)[:count]
    )
    image_metadata = (
        {"class_names": {idx: f"seed_{seed}_class_{idx}" for idx in range(5)}}
        if include_class_names
        else None
    )
    bboxes_metadata = None
    if include_ids:
        bboxes_metadata = [
            {"detection_id": f"seed_{seed}_detection_{idx}"} if idx % 3 else {}
            for idx in range(count)
        ]
    kwargs = dict(
        xyxy=torch.from_numpy(boxes),
        class_id=class_ids,
        confidence=confidence,
        image_metadata=image_metadata,
        bboxes_metadata=bboxes_metadata,
    )
    if instance_without_masks:
        return InstanceDetections(mask=None, **kwargs)
    return Detections(**kwargs)


@pytest.mark.parametrize("seed", range(120))
def test_bbox_path_matches_shapely_oracle_across_seeded_scenarios(seed: int) -> None:
    rng = np.random.default_rng(seed)
    reference = _make_bbox_detections(
        _random_boxes(rng, int(rng.integers(0, 61))),
        seed=seed,
        instance_without_masks=seed % 3 == 0,
        include_class_names=seed % 4 != 0,
        include_ids=seed % 5 in {0, 1},
        confidence_is_none=seed % 13 == 0,
    )
    candidate = _make_bbox_detections(
        _random_boxes(rng, int(rng.integers(0, 61))),
        seed=seed + 1000,
        instance_without_masks=seed % 3 == 1,
        include_class_names=seed % 4 != 1,
        include_ids=seed % 5 in {1, 2},
        confidence_is_none=seed % 17 == 0,
    )
    min_overlap = (0.0, 0.1, 0.5, 1.0)[seed % 4]

    expected = _oracle_run(reference, candidate, min_overlap)
    actual = OverlapAnalysisBlockV1().run(reference, candidate, min_overlap)

    _assert_results_equal(actual, expected, exact_ratio=False)


def _dense_mask_detections() -> InstanceDetections:
    masks = torch.zeros((3, 32, 32), dtype=torch.bool)
    masks[0, 2:7, 2:7] = True
    masks[0, 12:29, 10:28] = True
    masks[2, 7:25, 4:23] = True
    return InstanceDetections(
        xyxy=torch.tensor(
            [[0, 0, 31, 31], [2, 2, 20, 20], [4, 5, 29, 29]],
            dtype=torch.float32,
        ),
        class_id=torch.tensor([0, 1, 2]),
        confidence=torch.tensor([0.25, 0.5, 0.75]),
        mask=masks,
        image_metadata={"class_names": {0: "zero", 1: "one", 2: "two"}},
        bboxes_metadata=[{"detection_id": "a"}, {}, {"detection_id": "c"}],
    )


def _as_rle(detections: InstanceDetections) -> InstanceDetections:
    encoded = [torch_mask_to_coco_rle(mask) for mask in detections.mask]
    return InstanceDetections(
        xyxy=detections.xyxy,
        class_id=detections.class_id,
        confidence=detections.confidence,
        mask=InstancesRLEMasks.from_coco_rle_masks((32, 32), encoded),
        image_metadata=detections.image_metadata,
        bboxes_metadata=detections.bboxes_metadata,
    )


def _as_bboxes(detections: InstanceDetections) -> Detections:
    return Detections(
        xyxy=detections.xyxy,
        class_id=detections.class_id,
        confidence=detections.confidence,
        image_metadata=detections.image_metadata,
        bboxes_metadata=detections.bboxes_metadata,
    )


@pytest.mark.parametrize(
    ("reference_kind", "candidate_kind"),
    [
        ("dense", "dense"),
        ("rle", "rle"),
        ("bbox", "dense"),
        ("dense", "bbox"),
    ],
)
def test_mask_paths_match_shapely_oracle(
    reference_kind: str, candidate_kind: str
) -> None:
    dense_reference = _dense_mask_detections()
    dense_candidate = _dense_mask_detections()
    dense_candidate.xyxy = dense_candidate.xyxy + torch.tensor([2.0, 0.0, 2.0, 0.0])
    variants = {
        "dense": lambda detections: detections,
        "rle": _as_rle,
        "bbox": _as_bboxes,
    }
    reference = variants[reference_kind](dense_reference)
    candidate = variants[candidate_kind](dense_candidate)

    expected = _oracle_run(reference, candidate, 0.1)
    actual = OverlapAnalysisBlockV1().run(reference, candidate, 0.1)

    _assert_results_equal(actual, expected, exact_ratio=True)


def test_dense_mask_path_does_not_call_instance_mask_to_numpy(monkeypatch) -> None:
    reference = _dense_mask_detections()
    candidate = _dense_mask_detections()
    expected = _oracle_run(reference, candidate, 0.1)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("instance_mask_to_numpy must not be called")

    monkeypatch.setattr(
        overlap_module, "instance_mask_to_numpy", fail_if_called, raising=False
    )

    actual = OverlapAnalysisBlockV1().run(reference, candidate, 0.1)

    _assert_results_equal(actual, expected, exact_ratio=True)
