"""Geometry, reference matching, bounds and row fidelity of detection changes."""

import ast
import math
from typing import List, Optional, Union

import numpy as np
import pytest
import supervision as sv
import torch
from pydantic import ValidationError
from roboflow_workflows.v2.blocks.fusion.detections_difference.v1 import DetectionsDifferenceBlock, _greedy_match, _pairwise_costs
from roboflow_workflows.v2.data.frames import frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.row_identity import row_record, with_row_identity
from roboflow_workflows.v2.engine.declaration import spec_of

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import torch_mask_to_coco_rle
from tests.v2.support.paths import WORKFLOWS_ROOT

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    ),
]
DEFAULTS = dict(
    spatial_weight=0.4,
    class_mismatch_penalty=0.15,
    reject_cost=0.75,
    class_strict=False,
    min_removed_to_verify=1,
)


def _prediction(
    boxes, *, device="cpu", ids=None, names=None, image="before", family="box"
):
    count = len(boxes)
    source = ImageData.from_tensor(
        torch.zeros((3, 64, 64), dtype=torch.uint8), image_id=image
    )
    fields = dict(
        xyxy=torch.tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4),
        class_id=torch.tensor(
            ids if ids is not None else [0] * count, dtype=torch.int64, device=device
        ),
        confidence=torch.linspace(0.5, 0.9, count, device=device),
        image_metadata={
            **source.prediction_metadata(),
            **({"class_names": names} if names is not None else {}),
        },
        bboxes_metadata=[
            {
                "detection_id": f"{image}-{i}",
                "parent_id": f"parent-{image}",
                "class": "row-override",
                "tag": i,
            }
            for i in range(count)
        ],
    )
    if family == "box":
        prediction = Detections(**fields)
    else:
        dense = torch.zeros((count, 64, 64), dtype=torch.bool, device=device)
        for i in range(count):
            dense[i, i, i] = True
        mask = (
            dense
            if family == "dense"
            else InstancesRLEMasks.from_coco_rle_masks(
                image_size=(64, 64),
                masks=[torch_mask_to_coco_rle(one.cpu()) for one in dense],
            )
        )
        prediction = InstanceDetections(mask=mask, **fields)
    prediction = with_row_identity(prediction)

    return prediction


def _run(reference, candidate, **options):
    result = DetectionsDifferenceBlock().run(
        reference_predictions=reference,
        candidate_predictions=candidate,
        **{**DEFAULTS, **options},
    )

    return result


def _tags(prediction):
    tags = [row["tag"] for row in prediction.bboxes_metadata]

    return tags


@pytest.fixture
def v1_functions():
    # Execute only the actual latest V1 pure matching functions, without
    # importing its engine/block module into the V2 runtime.
    path = (
        WORKFLOWS_ROOT
        / "roboflow_workflows/core_steps/fusion/detections_difference/v1_tensor.py"
    )
    names = {
        "compute_pairwise_costs",
        "greedy_match",
        "_xyxy_as_numpy",
        "_class_mismatch_matrix",
        "_class_name_labels",
        "_class_id_labels",
    }
    functions = [
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    scope = dict(
        np=np,
        sv=sv,
        List=List,
        Optional=Optional,
        TensorNativeDetections=Union[Detections, InstanceDetections],
        CLASS_NAMES_KEY="class_names",
    )
    scope["Tuple"] = tuple
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), scope)

    return scope


def test_contract_defaults_selectors_and_output_provenance():
    params = DetectionsDifferenceBlock.Params(
        reference_predictions="$inputs.before", candidate_predictions="$inputs.after"
    )
    assert {name: getattr(params, name) for name in DEFAULTS} == DEFAULTS
    spec = spec_of(DetectionsDifferenceBlock)
    assert spec.prunable and not spec.mutates
    assert spec.aliases == ("DetectionsDifference",)
    assert set(spec.outputs) == {
        "removed_detections",
        "persisted_detections",
        "new_detections",
        "removed_count",
        "new_count",
        "verified",
    }
    for output in ("removed_detections", "removed_count", "verified"):
        assert spec.outputs[output].source == "reference_predictions"
    for output in ("persisted_detections", "new_detections", "new_count"):
        assert spec.outputs[output].source == "candidate_predictions"


@pytest.mark.parametrize(
    "name,value",
    [
        ("spatial_weight", -0.1),
        ("spatial_weight", 1.1),
        ("class_mismatch_penalty", -1),
        ("reject_cost", -1),
        ("min_removed_to_verify", 0),
        ("reject_cost", float("nan")),
    ],
)
def test_invalid_literal_bounds(name, value):
    with pytest.raises(ValidationError):
        DetectionsDifferenceBlock.Params(
            reference_predictions="$inputs.before",
            candidate_predictions="$inputs.after",
            **{name: value},
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("before,after", [(0, 0), (2, 0), (0, 2)])
@pytest.mark.parametrize("family", ["box", "dense", "rle"])
def test_empty_sides_keep_frames_and_verification(device, before, after, family):
    a = _prediction([[0, 0, 2, 2]] * before, device=device, family=family)
    b = _prediction([[0, 0, 2, 2]] * after, device=device, image="after", family=family)
    out = _run(a, b)
    assert out["removed_count"] == before and out["new_count"] == after
    assert out["verified"] is (before > 0)
    assert len(out["persisted_detections"]) == 0
    for name, original in [
        ("removed_detections", a),
        ("persisted_detections", b),
        ("new_detections", b),
    ]:
        assert frame_of(out[name]) == frame_of(original)
        assert type(out[name]) is type(original)
        assert row_record(out[name]).namespace == row_record(original).namespace


@pytest.mark.parametrize("device", DEVICES)
def test_one_removed_one_persisted_one_new_and_verification_count(device):
    a = _prediction([[0, 0, 4, 4], [20, 20, 24, 24]], device=device)
    b = _prediction([[20, 20, 24, 24], [40, 40, 44, 44]], device=device, image="after")
    out = _run(a, b, reject_cost=0.1)
    assert _tags(out["removed_detections"]) == [0]
    assert _tags(out["persisted_detections"]) == [0]
    assert _tags(out["new_detections"]) == [1]
    assert (out["removed_count"], out["new_count"], out["verified"]) == (1, 1, True)
    assert not _run(a, b, reject_cost=0.1, min_removed_to_verify=2)["verified"]


@pytest.mark.parametrize("device", DEVICES)
def test_row_major_ties_ignore_confidence_and_are_one_to_one(device):
    a = _prediction([[0, 0, 4, 4]] * 3, device=device)
    b = _prediction([[0, 0, 4, 4]] * 2, device=device, image="after")
    out = _run(a, b)
    assert _tags(out["removed_detections"]) == [2]
    assert _tags(out["persisted_detections"]) == [0, 1]
    assert out["new_count"] == 0


@pytest.mark.parametrize("device", DEVICES)
def test_disjoint_proximity_and_persisted_candidate_order(device):
    a = _prediction([[0, 0, 2, 2]], device=device)
    b = _prediction([[30, 0, 32, 2], [3, 0, 5, 2]], device=device, image="after")
    out = _run(a, b)
    assert _tags(out["persisted_detections"]) == [1]
    assert _tags(out["new_detections"]) == [0]

    a = _prediction([[0, 0, 2, 2], [10, 0, 12, 2]], device=device)
    b = _prediction([[10, 0, 12, 2], [0, 0, 2, 2]], device=device, image="after")
    out = _run(a, b, reject_cost=0.1)
    # Greedy matches candidate1 first; outputs still keep candidate order.
    assert _tags(out["persisted_detections"]) == [0, 1]


def test_matching_is_greedy_not_hungarian():
    costs = torch.tensor([[0.1, 0.2], [0.11, 0.9]], dtype=torch.float64)
    assert _greedy_match(costs, reject_cost=0.75) == [(0, 0)]
    assert torch.equal(
        costs, torch.tensor([[0.1, 0.2], [0.11, 0.9]], dtype=torch.float64)
    )


@pytest.mark.parametrize("device", DEVICES)
def test_reject_cost_is_inclusive_and_disjoint_uses_enclosing_diagonal(device):
    a = _prediction([[0, 0, 3, 3]], device=device)
    b = _prediction([[1, 0, 4, 3]], device=device, image="after")
    exact = float(
        _pairwise_costs(
            reference=a,
            candidate=b,
            spatial_weight=0.4,
            class_mismatch_penalty=0.15,
            class_strict=False,
        )[0, 0]
    )
    assert _run(a, b, reject_cost=exact)["removed_count"] == 0
    assert _run(a, b, reject_cost=np.nextafter(exact, 0))["removed_count"] == 1
    far = _prediction([[6, 0, 9, 3]], device=device, image="after")
    cost = _pairwise_costs(
        reference=a,
        candidate=far,
        spatial_weight=1,
        class_mismatch_penalty=0,
        class_strict=False,
    )
    assert float(cost[0, 0]) == pytest.approx(1 + 6 / np.hypot(9, 3))


@pytest.mark.parametrize("device", DEVICES)
def test_class_name_precedence_unknown_fallback_and_strict_cost(device):
    a = _prediction([[0, 0, 3, 3]], device=device, ids=[3], names={3: "cup"})
    b = _prediction(
        [[0, 0, 3, 3]], device=device, ids=[7], names={7: "cup"}, image="after"
    )
    assert _run(a, b, class_strict=True)["removed_count"] == 0
    b.image_metadata["class_names"] = {7: "plate"}
    assert _run(a, b, reject_cost=0.15)["removed_count"] == 0
    assert _run(a, b, reject_cost=0.14)["removed_count"] == 1
    assert _run(a, b, class_strict=True, reject_cost=float("inf"))["removed_count"] == 1
    # Both row overrides say row-override; V1 map precedence still forbids it.
    b.image_metadata["class_names"] = {}
    assert _run(a, b, class_strict=True)["removed_count"] == 1
    a.image_metadata["class_names"] = {99: "unused"}
    b.image_metadata["class_names"] = {98: "unused"}
    b.class_id.fill_(3)
    assert _run(a, b, class_strict=True)["removed_count"] == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["box", "dense", "rle"])
def test_subsets_preserve_native_rows_masks_parents_and_different_frames(
    device, family
):
    a = _prediction([[0, 0, 2, 2], [10, 10, 12, 12]], device=device, family=family)
    b = _prediction(
        [[30, 30, 32, 32], [10, 10, 12, 12]],
        device=device,
        family=family,
        image="after",
    )
    out = _run(a, b, reject_cost=0.1)
    for name, original, indices in [
        ("removed_detections", a, [0]),
        ("persisted_detections", b, [1]),
        ("new_detections", b, [0]),
    ]:
        kept = out[name]
        assert type(kept) is type(original) and kept.xyxy.device == original.xyxy.device
        assert frame_of(kept) == frame_of(original)
        assert torch.equal(kept.xyxy, original.xyxy[indices])
        assert torch.equal(kept.confidence, original.confidence[indices])
        assert kept.bboxes_metadata == [original.bboxes_metadata[i] for i in indices]
        assert row_record(kept).namespace == row_record(original).namespace
        assert row_record(kept).host_ids() == indices
        if family == "dense":
            assert torch.equal(kept.mask, original.mask[indices])
        elif family == "rle":
            assert kept.mask.to_coco_rle_masks() == [
                original.mask.to_coco_rle_masks()[i] for i in indices
            ]


@pytest.mark.parametrize("device", DEVICES)
def test_actual_v1_matching_functions_agree_away_from_precision_boundary(
    device, v1_functions
):
    generator = torch.Generator().manual_seed(56)
    for _ in range(8):
        corners = torch.rand((6, 2), generator=generator) * 30
        boxes = torch.cat([corners, corners + 5], dim=1).tolist()
        a = _prediction(boxes, device=device, ids=[0, 1, 0, 1, 0, 1])
        b = _prediction(
            boxes[2:] + [[50, 50, 55, 55]],
            device=device,
            ids=[0, 1, 0, 1, 2],
            image="after",
        )
        options = dict(
            spatial_weight=0.4, class_mismatch_penalty=0.15, class_strict=False
        )
        cost = v1_functions["compute_pairwise_costs"](
            reference=a, candidate=b, **options
        )
        matches = v1_functions["greedy_match"](cost_matrix=cost, reject_cost=0.55)
        out = _run(a, b, reject_cost=0.55)
        assert _tags(out["removed_detections"]) == [
            i for i in range(len(a)) if i not in {r for r, _ in matches}
        ]
        assert _tags(out["persisted_detections"]) == sorted({c for _, c in matches})


def test_documented_float32_iou_boundary_differs_from_v1(v1_functions):
    a = _prediction([[0, 0, 3.1, 3.1]])
    b = _prediction([[1, 0, 4.1, 3.1]], image="after")
    cost = v1_functions["compute_pairwise_costs"](
        reference=a,
        candidate=b,
        spatial_weight=1,
        class_mismatch_penalty=0,
        class_strict=False,
    )
    boundary = float(cost[0, 0])
    assert boundary == 0.4878048896789551
    assert v1_functions["greedy_match"](cost_matrix=cost, reject_cost=boundary) == [
        (0, 0)
    ]
    assert (
        _run(a, b, spatial_weight=1, class_mismatch_penalty=0, reject_cost=boundary)[
            "removed_count"
        ]
        == 1
    )


@pytest.mark.parametrize("device", DEVICES)
def test_identical_zero_area_boxes_match(device):
    a = _prediction([[0, 0, 0, 0]], device=device)
    b = _prediction([[0, 0, 0, 0]], device=device, image="after")
    assert _run(a, b)["removed_count"] == 0


@pytest.mark.parametrize("device", DEVICES)
def test_infinite_soft_penalty_matches_nothing_as_in_v1(device, v1_functions):
    # Same class: mismatch 0, and 0 * inf is NaN in V1's NumPy cost. Greedy
    # stops at the first non-finite cheapest cost, so even identical boxes stay
    # unmatched. Keep this V1 parity; do not mask the NaN.
    a = _prediction([[0, 0, 3, 3]], device=device)
    b = _prediction([[0, 0, 3, 3]], device=device, image="after")
    options = dict(
        spatial_weight=0.4, class_mismatch_penalty=float("inf"), class_strict=False
    )
    cost = v1_functions["compute_pairwise_costs"](reference=a, candidate=b, **options)
    assert math.isnan(float(cost[0, 0]))
    assert v1_functions["greedy_match"](cost_matrix=cost, reject_cost=1) == []
    assert _run(a, b, class_mismatch_penalty=float("inf"))["removed_count"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_mixed_input_devices_preserve_each_output_device():
    a = _prediction([[0, 0, 3, 3]], device="cpu")
    b = _prediction([[0, 0, 3, 3]], device="cuda", image="after")
    out = _run(a, b)
    assert out["removed_detections"].xyxy.device.type == "cpu"
    assert out["persisted_detections"].xyxy.device.type == "cuda"
