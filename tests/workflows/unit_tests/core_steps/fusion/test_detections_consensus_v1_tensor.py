"""Fuzz parity between the vectorised tensor-native consensus matching and a
verbatim copy of the ORIGINAL per-pair logic (the oracle).

The optimisation replaced the O(N^2) `calculate_iou` / `_resolve_class_name`
per-pair device syncs with a single vectorised `sv.box_iou_batch` matrix and a
host download of `class_id` per source (see `_precompute_pair_data`). Everything
downstream of the matching (mask padding, merge, presence checks) is unchanged
and shared between the oracle and the new code, so this test drives hundreds of
random multi-source scenarios and asserts every output field is identical.

`uuid4` is patched to a deterministic counter (reset before each run) so the
merged-detection ``detection_id`` values are comparable: because the two paths
produce identical merges in identical order, the deterministic id stream lines
up exactly - any divergence in the matching would desync it.

Runs on CPU torch; no CUDA required.
"""

import random
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pytest
import torch

from inference.core.workflows.core_steps.fusion.detections_consensus import (
    v1_tensor as tv,
)
from inference.core.workflows.core_steps.fusion.detections_consensus.v1_tensor import (
    AggregationMode,
    MaskAggregationMode,
)
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAMES_KEY,
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY,
    PARENT_ID_KEY,
    ROOT_PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    ROOT_PARENT_ID_KEY,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections

TensorNativeDetections = Union[Detections, InstanceDetections]


# --------------------------------------------------------------------------- #
# Oracle: a verbatim copy of the ORIGINAL matching (per-pair calculate_iou /   #
# _resolve_class_name over freshly-built single-row predictions). Everything   #
# else reuses the unchanged shared helpers from the block module.             #
# --------------------------------------------------------------------------- #
def _oracle_enumerate_detections(
    detections_from_sources: List[TensorNativeDetections],
    excluded_source_id: Optional[int] = None,
):
    for source_id, detections in enumerate(detections_from_sources):
        if excluded_source_id == source_id:
            continue
        for i in range(len(detections)):
            yield source_id, tv.take_prediction_by_indices(detections, [i])


def _oracle_max_overlap(
    detection: TensorNativeDetections,
    source: int,
    detections_from_sources: List[TensorNativeDetections],
    iou_threshold: float,
    class_aware: bool,
    detections_already_considered: Set[str],
) -> Dict[int, Tuple[TensorNativeDetections, float]]:
    current_max_overlap: Dict[int, Tuple[TensorNativeDetections, float]] = {}
    for other_source, other_detection in _oracle_enumerate_detections(
        detections_from_sources=detections_from_sources,
        excluded_source_id=source,
    ):
        if tv._resolve_detection_id(other_detection) in detections_already_considered:
            continue
        if class_aware and tv._resolve_class_name(detection) != tv._resolve_class_name(
            other_detection
        ):
            continue
        iou_value = tv.calculate_iou(detection_a=detection, detection_b=other_detection)
        if iou_value <= iou_threshold:
            continue
        if current_max_overlap.get(other_source) is None:
            current_max_overlap[other_source] = (other_detection, iou_value)
        if current_max_overlap[other_source][1] < iou_value:
            current_max_overlap[other_source] = (other_detection, iou_value)
    return current_max_overlap


def _oracle_get_consensus_for_single_detection(
    detection,
    source_id,
    detections_from_sources,
    iou_threshold,
    class_aware,
    required_votes,
    confidence,
    detections_merge_confidence_aggregation,
    detections_merge_coordinates_aggregation,
    detections_merge_mask_aggregation,
    detections_already_considered,
):
    if (
        len(detection)
        and tv._resolve_detection_id(detection) in detections_already_considered
    ):
        return [], detections_already_considered
    consensus_detections = []
    detections_with_max_overlap = _oracle_max_overlap(
        detection=detection,
        source=source_id,
        detections_from_sources=detections_from_sources,
        iou_threshold=iou_threshold,
        class_aware=class_aware,
        detections_already_considered=detections_already_considered,
    )
    if len(detections_with_max_overlap) < (required_votes - 1):
        return consensus_detections, detections_already_considered
    detection_mask = tv._single_mask(detection)
    overlap_masks = {
        other_source: tv._single_mask(matched_value[0])
        for other_source, matched_value in detections_with_max_overlap.items()
    }
    if detection_mask is not None:
        for other_source, matched_mask in overlap_masks.items():
            if matched_mask is None:
                overlap_masks[other_source] = np.zeros(detection_mask.shape, dtype=bool)
    else:
        shape = None
        for matched_mask in overlap_masks.values():
            if matched_mask is not None:
                shape = matched_mask.shape
                break
        if shape:
            for other_source, matched_mask in overlap_masks.items():
                if matched_mask is None:
                    overlap_masks[other_source] = np.zeros(shape, dtype=bool)
            detection_mask = np.zeros(shape, dtype=bool)
    group_detections = [detection] + [
        matched_value[0] for matched_value in detections_with_max_overlap.values()
    ]
    group_masks = (
        [detection_mask]
        + [overlap_masks[other_source] for other_source in detections_with_max_overlap]
        if detection_mask is not None
        else None
    )
    merged_detection = tv.merge_detections(
        detections=group_detections,
        masks=group_masks,
        confidence_aggregation_mode=detections_merge_confidence_aggregation,
        boxes_aggregation_mode=detections_merge_coordinates_aggregation,
        mask_aggregation_mode=detections_merge_mask_aggregation,
    )
    if float(merged_detection.confidence[0]) < confidence:
        return consensus_detections, detections_already_considered
    consensus_detections.append(merged_detection)
    detections_already_considered.add(tv._resolve_detection_id(detection))
    for matched_value in detections_with_max_overlap.values():
        detections_already_considered.add(tv._resolve_detection_id(matched_value[0]))
    return consensus_detections, detections_already_considered


def oracle_agree_on_consensus(
    detections_from_sources,
    required_votes,
    class_aware,
    iou_threshold,
    confidence,
    classes_to_consider,
    required_objects,
    presence_confidence_aggregation,
    detections_merge_confidence_aggregation,
    detections_merge_coordinates_aggregation,
    detections_merge_mask_aggregation,
):
    if tv.does_not_detect_objects_in_any_source(
        detections_from_sources=detections_from_sources
    ):
        return (
            "undefined",
            False,
            {},
            tv._empty_native_detections(
                device=(
                    detections_from_sources[0].xyxy.device
                    if detections_from_sources
                    else None
                )
            ),
        )
    parent_id = tv.get_parent_id_of_detections_from_sources(
        detections_from_sources=detections_from_sources,
    )
    detections_from_sources = tv.filter_predictions(
        predictions=detections_from_sources,
        classes_to_consider=classes_to_consider,
    )
    detections_already_considered = set()
    consensus_detections = []
    for source_id, detection in _oracle_enumerate_detections(
        detections_from_sources=detections_from_sources
    ):
        (
            consensus_detections_update,
            detections_already_considered,
        ) = _oracle_get_consensus_for_single_detection(
            detection=detection,
            source_id=source_id,
            detections_from_sources=detections_from_sources,
            iou_threshold=iou_threshold,
            class_aware=class_aware,
            required_votes=required_votes,
            confidence=confidence,
            detections_merge_confidence_aggregation=detections_merge_confidence_aggregation,
            detections_merge_coordinates_aggregation=detections_merge_coordinates_aggregation,
            detections_merge_mask_aggregation=detections_merge_mask_aggregation,
            detections_already_considered=detections_already_considered,
        )
        consensus_detections += consensus_detections_update
    consensus_detections = tv._merge_native_detections(consensus_detections)
    (
        object_present,
        presence_confidence,
    ) = tv.check_objects_presence_in_consensus_detections(
        consensus_detections=consensus_detections,
        aggregation_mode=presence_confidence_aggregation,
        class_aware=class_aware,
        required_objects=required_objects,
    )
    return parent_id, object_present, presence_confidence, consensus_detections


# --------------------------------------------------------------------------- #
# Random native-detections scenario generation                                #
# --------------------------------------------------------------------------- #
_CLASS_POOL = ["a", "b", "c"]
_IMG_H, _IMG_W = 64, 64


class _DetSpec:
    """Plain-data description of one source, materialised fresh into a native
    prediction for each run so the two paths never share (or mutate) tensors."""

    def __init__(
        self,
        xyxy: np.ndarray,
        class_ids: List[int],
        confidences: List[float],
        detection_ids: List[str],
        class_names_map: Dict[int, str],
        parent_id: str,
        with_mask: bool,
    ) -> None:
        self.xyxy = xyxy
        self.class_ids = class_ids
        self.confidences = confidences
        self.detection_ids = detection_ids
        self.class_names_map = class_names_map
        self.parent_id = parent_id
        self.with_mask = with_mask

    def materialise(self) -> TensorNativeDetections:
        n = self.xyxy.shape[0]
        image_metadata = {
            CLASS_NAMES_KEY: dict(self.class_names_map),
            PARENT_ID_KEY: self.parent_id,
            PARENT_COORDINATES_KEY: [0, 0],
            PARENT_DIMENSIONS_KEY: [_IMG_H, _IMG_W],
            ROOT_PARENT_ID_KEY: self.parent_id,
            ROOT_PARENT_COORDINATES_KEY: [0, 0],
            ROOT_PARENT_DIMENSIONS_KEY: [_IMG_H, _IMG_W],
            IMAGE_DIMENSIONS_KEY: [_IMG_H, _IMG_W],
        }
        bboxes_metadata = [{DETECTION_ID_KEY: did} for did in self.detection_ids]
        xyxy = torch.tensor(self.xyxy, dtype=torch.float32)
        class_id = torch.tensor(self.class_ids, dtype=torch.long)
        confidence = torch.tensor(self.confidences, dtype=torch.float32)
        if not self.with_mask:
            return Detections(
                xyxy=xyxy,
                class_id=class_id,
                confidence=confidence,
                image_metadata=image_metadata,
                bboxes_metadata=bboxes_metadata,
            )
        mask = torch.zeros((n, _IMG_H, _IMG_W), dtype=torch.bool)
        for i in range(n):
            x1, y1, x2, y2 = self.xyxy[i].astype(int)
            x1 = max(0, min(_IMG_W, x1))
            x2 = max(0, min(_IMG_W, x2))
            y1 = max(0, min(_IMG_H, y1))
            y2 = max(0, min(_IMG_H, y2))
            if x2 > x1 and y2 > y1:
                mask[i, y1:y2, x1:x2] = True
        return InstanceDetections(
            xyxy=xyxy,
            class_id=class_id,
            confidence=confidence,
            mask=mask,
            image_metadata=image_metadata,
            bboxes_metadata=bboxes_metadata,
        )


def _random_box(rng: random.Random) -> np.ndarray:
    x1 = rng.uniform(0, _IMG_W - 8)
    y1 = rng.uniform(0, _IMG_H - 8)
    w = rng.uniform(4, _IMG_W - x1)
    h = rng.uniform(4, _IMG_H - y1)
    return np.array([x1, y1, x1 + w, y1 + h], dtype=np.float64)


def _jitter(box: np.ndarray, rng: random.Random) -> np.ndarray:
    d = np.array([rng.uniform(-3, 3) for _ in range(4)])
    out = box + d
    out[2] = max(out[2], out[0] + 2)
    out[3] = max(out[3], out[1] + 2)
    return np.clip(out, 0, max(_IMG_H, _IMG_W))


def _build_sources(scenario_index: int) -> Tuple[List[_DetSpec], Dict[str, Any]]:
    rng = random.Random(20260714 + scenario_index * 7919)
    parent_id = "parent-image"
    num_sources = rng.randint(2, 4)
    pool_size = rng.randint(1, 3)
    class_ids = list(range(pool_size))
    class_names_map = {cid: _CLASS_POOL[cid] for cid in class_ids}
    # A handful of shared cluster anchors so cross-source boxes actually overlap
    # (identical anchors exercise the equal-IoU tie rule).
    anchors = [_random_box(rng) for _ in range(rng.randint(1, 4))]
    homogeneous_mask = rng.random() < 0.5
    global_scenario_masks = rng.random() < 0.5

    specs: List[_DetSpec] = []
    for source_id in range(num_sources):
        count = rng.choice([0, 1, 1, 2, 3, 4, 5])
        boxes = []
        for _ in range(count):
            mode = rng.random()
            if anchors and mode < 0.4:
                boxes.append(_jitter(rng.choice(anchors), rng))
            elif anchors and mode < 0.6:
                boxes.append(np.array(rng.choice(anchors), dtype=np.float64))
            else:
                boxes.append(_random_box(rng))
        xyxy = np.stack(boxes, axis=0) if boxes else np.zeros((0, 4), dtype=np.float64)
        cids = [rng.choice(class_ids) for _ in range(count)]
        confs = [round(rng.uniform(0.05, 0.99), 4) for _ in range(count)]
        dids = [f"s{source_id}_d{i}" for i in range(count)]
        if homogeneous_mask:
            with_mask = global_scenario_masks
        else:
            with_mask = rng.random() < 0.5
        specs.append(
            _DetSpec(
                xyxy=xyxy,
                class_ids=cids,
                confidences=confs,
                detection_ids=dids,
                class_names_map=class_names_map,
                parent_id=parent_id,
                with_mask=with_mask,
            )
        )

    classes_to_consider = rng.choice(
        [None, None, [_CLASS_POOL[0]], _CLASS_POOL[:pool_size], ["nonexistent"]]
    )
    required_objects_choice = rng.choice([None, 1, 2, {"a": 1}, {"a": 1, "b": 1}])
    params = {
        "required_votes": rng.randint(1, 3),
        "class_aware": rng.random() < 0.5,
        "iou_threshold": rng.choice([0.0, 0.2, 0.3, 0.5, 0.7]),
        "confidence": rng.choice([0.0, 0.0, 0.3, 0.6]),
        "classes_to_consider": classes_to_consider,
        "required_objects": required_objects_choice,
        "presence_confidence_aggregation": rng.choice(list(AggregationMode)),
        "detections_merge_confidence_aggregation": rng.choice(list(AggregationMode)),
        "detections_merge_coordinates_aggregation": rng.choice(list(AggregationMode)),
        "detections_merge_mask_aggregation": rng.choice(list(MaskAggregationMode)),
    }
    return specs, params


# --------------------------------------------------------------------------- #
# Output comparison                                                            #
# --------------------------------------------------------------------------- #
def _assert_detections_equal(
    got: TensorNativeDetections, exp: TensorNativeDetections
) -> None:
    assert type(got) is type(exp), f"{type(got)} != {type(exp)}"
    np.testing.assert_array_equal(
        got.xyxy.detach().cpu().numpy(), exp.xyxy.detach().cpu().numpy()
    )
    np.testing.assert_array_equal(
        got.class_id.detach().cpu().numpy(), exp.class_id.detach().cpu().numpy()
    )
    np.testing.assert_array_equal(
        got.confidence.detach().cpu().numpy(),
        exp.confidence.detach().cpu().numpy(),
    )
    got_mask = getattr(got, "mask", None)
    exp_mask = getattr(exp, "mask", None)
    assert (got_mask is None) == (exp_mask is None)
    if got_mask is not None:
        np.testing.assert_array_equal(
            got_mask.detach().cpu().numpy(), exp_mask.detach().cpu().numpy()
        )
    got_meta = got.image_metadata or {}
    exp_meta = exp.image_metadata or {}
    assert got_meta.get(CLASS_NAMES_KEY) == exp_meta.get(CLASS_NAMES_KEY)
    got_ids = [m.get(DETECTION_ID_KEY) for m in (got.bboxes_metadata or [])]
    exp_ids = [m.get(DETECTION_ID_KEY) for m in (exp.bboxes_metadata or [])]
    assert got_ids == exp_ids


def _run_with_deterministic_uuid(monkeypatch, fn, specs, params):
    counter = {"n": 0}

    def _fake_uuid4():
        counter["n"] += 1
        return f"merged-{counter['n']:04d}"

    monkeypatch.setattr(tv, "uuid4", _fake_uuid4)
    sources = [spec.materialise() for spec in specs]
    return fn(
        detections_from_sources=sources,
        required_votes=params["required_votes"],
        class_aware=params["class_aware"],
        iou_threshold=params["iou_threshold"],
        confidence=params["confidence"],
        classes_to_consider=params["classes_to_consider"],
        required_objects=params["required_objects"],
        presence_confidence_aggregation=params["presence_confidence_aggregation"],
        detections_merge_confidence_aggregation=params[
            "detections_merge_confidence_aggregation"
        ],
        detections_merge_coordinates_aggregation=params[
            "detections_merge_coordinates_aggregation"
        ],
        detections_merge_mask_aggregation=params["detections_merge_mask_aggregation"],
    )


@pytest.mark.parametrize("scenario_index", list(range(160)))
def test_vectorised_consensus_matches_original_per_pair_logic(
    scenario_index: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # given
    specs, params = _build_sources(scenario_index)

    # when
    exp_parent, exp_present, exp_conf, exp_dets = _run_with_deterministic_uuid(
        monkeypatch, oracle_agree_on_consensus, specs, params
    )
    got_parent, got_present, got_conf, got_dets = _run_with_deterministic_uuid(
        monkeypatch,
        tv.agree_on_consensus_for_all_detections_sources,
        specs,
        params,
    )

    # then
    assert got_parent == exp_parent
    assert got_present == exp_present
    assert got_conf == pytest.approx(exp_conf)
    assert len(got_dets) == len(exp_dets)
    _assert_detections_equal(got_dets, exp_dets)


def test_scenarios_are_non_trivial() -> None:
    """Guard the fuzz corpus: across the seeded scenarios there must be runs
    that actually emit consensus detections (otherwise the parity assertions
    would be vacuously satisfied by everything returning empty)."""
    emitted = 0
    empty = 0
    for scenario_index in range(160):
        specs, params = _build_sources(scenario_index)
        sources = [spec.materialise() for spec in specs]
        _, _, _, dets = tv.agree_on_consensus_for_all_detections_sources(
            detections_from_sources=sources,
            required_votes=params["required_votes"],
            class_aware=params["class_aware"],
            iou_threshold=params["iou_threshold"],
            confidence=params["confidence"],
            classes_to_consider=params["classes_to_consider"],
            required_objects=params["required_objects"],
            presence_confidence_aggregation=params["presence_confidence_aggregation"],
            detections_merge_confidence_aggregation=params[
                "detections_merge_confidence_aggregation"
            ],
            detections_merge_coordinates_aggregation=params[
                "detections_merge_coordinates_aggregation"
            ],
            detections_merge_mask_aggregation=params[
                "detections_merge_mask_aggregation"
            ],
        )
        if len(dets) > 0:
            emitted += 1
        else:
            empty += 1
    assert emitted >= 20, f"too few non-empty consensus outputs: {emitted}"
    assert empty >= 5, f"expected some empty consensus outputs too: {empty}"


def test_single_d2h_per_tensor_field_per_source() -> None:
    """Exactly one host download per tensor field (xyxy, class_id) per source in
    `_precompute_pair_data`, and none left in the O(N^2) matching loop."""
    specs, _ = _build_sources(3)
    sources = [spec.materialise() for spec in specs if spec.xyxy.shape[0] > 0]
    # Make sure the corpus for this check has several detections to match.
    assert sum(len(s) for s in sources) >= 3

    calls = {"count": 0}
    original_to = torch.Tensor.to

    def _counting_to(self, *args, **kwargs):
        target = args[0] if args else kwargs.get("device")
        if isinstance(target, str) and target == "cpu":
            calls["count"] += 1
        return original_to(self, *args, **kwargs)

    # xyxy + class_id => 2 downloads per non-empty source.
    import unittest.mock as _mock

    with _mock.patch.object(torch.Tensor, "to", _counting_to):
        tv._precompute_pair_data(sources)
    assert calls["count"] == 2 * len(sources)
