import copy
import math

import numpy as np
import pytest
import supervision as sv
from roboflow_workflows.core_steps.common.oriented import (
    native_obb_from_inference,
    native_obb_to_tensor,
    serialise_native_obb,
)
from roboflow_workflows.core_steps.fusion.detections_consensus.v2 import (
    AggregationMode,
    DetectionsConsensusBlockV2,
    MaskAggregationMode,
)


def _response(
    *,
    identity,
    confidence=0.8,
    angle=0.0,
    sides=(12.0, 24.0),
    name="plane",
    heading=True
):
    return {
        "task_type": "oriented-object-detection",
        "image": {"width": 100, "height": 80},
        "class_heading": {name: heading, "unused": False},
        "predictions": [
            {
                "obb": [40.25, 30.75, *sides, angle],
                "class": name,
                "class_id": 7 if name == "plane" else 9,
                "heading": heading,
                "confidence": confidence,
                "detection_id": identity,
                "parent_id": "same-image",
            }
        ],
    }


def _run(sources, *, tensor=False, aggregation=AggregationMode.AVERAGE, **updates):
    predictions = [
        native_obb_from_inference(source) if isinstance(source, dict) else source
        for source in sources
    ]
    if tensor:
        predictions = [native_obb_to_tensor(source) for source in predictions]
    kwargs = dict(
        predictions_batches=[[source] for source in predictions],
        required_votes=len(sources),
        class_aware=False,
        iou_threshold=0.1,
        confidence=0.0,
        classes_to_consider=None,
        required_objects=None,
        presence_confidence_aggregation=AggregationMode.MAX,
        detections_merge_confidence_aggregation=aggregation,
        detections_merge_coordinates_aggregation=AggregationMode.AVERAGE,
        detections_merge_mask_aggregation=MaskAggregationMode.UNION,
    )
    kwargs.update(updates)
    return DetectionsConsensusBlockV2().run(**kwargs)[0]["predictions"]


@pytest.mark.parametrize("tensor", [False, True])
@pytest.mark.parametrize("sides", [(12.0, 24.0), (12.0, 12.0)])
@pytest.mark.parametrize("aggregation", list(AggregationMode))
def test_block_retains_opposite_front_contributor_atomically(
    tensor, sides, aggregation
):
    first = _response(identity="first", confidence=0.6, sides=sides)
    winner = _response(identity="winner", confidence=0.9, sides=sides, angle=math.pi)
    winner["class_heading"]["unobserved-from-winner"] = True
    result = serialise_native_obb(
        _run([first, winner], tensor=tensor, aggregation=aggregation)
    )
    row = result["predictions"][0]
    assert row["obb"] == pytest.approx(winner["predictions"][0]["obb"])
    assert row["detection_id"] == "winner"
    assert row["heading"] is True
    assert row["class_id"] == 7
    assert row["parent_id"] == "same-image"
    assert row["confidence"] == pytest.approx(
        {
            AggregationMode.AVERAGE: 0.75,
            AggregationMode.MAX: 0.9,
            AggregationMode.MIN: 0.6,
        }[aggregation]
    )
    assert result["class_heading"] == {
        **first["class_heading"],
        **winner["class_heading"],
    }


@pytest.mark.parametrize("tensor", [False, True])
@pytest.mark.parametrize(
    "aggregation,expected",
    [
        (AggregationMode.AVERAGE, "plane-best"),
        (AggregationMode.MIN, "plane-best"),
        (AggregationMode.MAX, "stone"),
    ],
)
def test_winner_is_highest_confidence_in_aggregated_class(
    tensor, aggregation, expected
):
    sources = [
        _response(identity="plane-first", confidence=0.3),
        _response(identity="plane-best", confidence=0.6, angle=math.pi),
        _response(identity="stone", confidence=0.99, name="stone", heading=False),
    ]
    row = serialise_native_obb(_run(sources, tensor=tensor, aggregation=aggregation))[
        "predictions"
    ][0]
    assert row["detection_id"] == expected
    assert row["heading"] is (expected != "stone")


@pytest.mark.parametrize("tensor", [False, True])
def test_ties_keep_source_order_and_inputs_are_unmodified(tensor):
    sources = [_response(identity="first"), _response(identity="second", angle=math.pi)]
    before = copy.deepcopy(sources)
    row = serialise_native_obb(_run(sources, tensor=tensor))["predictions"][0]
    assert row["detection_id"] == "first"
    assert sources == before


@pytest.mark.parametrize("tensor", [False, True])
@pytest.mark.parametrize("missing", ["xyxyxyxy", "heading"])
def test_block_rejects_missing_native_geometry_before_voting(tensor, missing):
    sources = [
        native_obb_from_inference(_response(identity=str(index))) for index in range(2)
    ]
    del sources[1].data[missing]
    with pytest.raises(ValueError, match="corners and heading"):
        _run(sources, tensor=tensor, required_votes=3)


@pytest.mark.parametrize("tensor", [False, True])
def test_block_rejects_conflicting_unobserved_frozen_policy(tensor):
    sources = [_response(identity="first"), _response(identity="second")]
    sources[1]["class_heading"]["unused"] = True
    with pytest.raises(ValueError, match="Conflicting native class heading"):
        _run(sources, tensor=tensor)


@pytest.mark.parametrize("tensor", [False, True])
def test_empty_consensus_retains_native_task_and_entire_frozen_policy(tensor):
    sources = [_response(identity="first"), _response(identity="second")]
    result = serialise_native_obb(_run(sources, tensor=tensor, required_votes=3))
    assert result["predictions"] == []
    assert result["class_heading"] == sources[0]["class_heading"]
    assert result["task_type"] == "oriented-object-detection"


def test_block_rejects_native_and_ordinary_even_when_classes_do_not_match():
    native = native_obb_from_inference(_response(identity="native"))
    ordinary = sv.Detections(
        xyxy=np.array([[0.0, 0.0, 10.0, 10.0]]),
        class_id=np.array([1]),
        confidence=np.array([0.9]),
        data={"class_name": np.array(["other"])},
    )
    with pytest.raises(ValueError, match="mix ordinary"):
        _run([native, ordinary], class_aware=True)


def test_v2_explicitly_rejects_ordinary_tensor_carrier():
    import torch

    from inference_models.models.base.object_detection import Detections

    ordinary = Detections(
        xyxy=torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        class_id=torch.tensor([1]),
        confidence=torch.tensor([0.9]),
    )
    with pytest.raises(ValueError, match="ordinary tensor consensus is unsupported"):
        _run([ordinary, ordinary])


@pytest.mark.parametrize("sides", [(12.0, 24.0), (12.0, 12.0)])
def test_serialized_workflow_native_consensus_and_combination_preserve_policy(sides):
    from roboflow_workflows.execution_engine.core import ExecutionEngine

    workflow = {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowBatchInput",
                "name": name,
                "kind": ["oriented_object_detection_prediction"],
            }
            for name in ("first", "second")
        ],
        "steps": [
            {
                "type": "roboflow_core/detections_consensus@v2",
                "name": "consensus",
                "predictions_batches": ["$inputs.first", "$inputs.second"],
                "required_votes": 2,
            },
            {
                "type": "roboflow_core/detections_combine@v1",
                "name": "combine",
                "prediction_one": "$steps.consensus.predictions",
                "prediction_two": "$inputs.first",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "result",
                "selector": "$steps.consensus.predictions",
            },
            {
                "type": "JsonField",
                "name": "combined",
                "selector": "$steps.combine.predictions",
            },
        ],
    }
    first = _response(identity="first", confidence=0.6, sides=sides)
    second = _response(identity="second", confidence=0.9, angle=math.pi, sides=sides)
    second["class_heading"]["additional"] = True
    engine = ExecutionEngine.init(workflow_definition=workflow)
    outputs = engine.run(
        runtime_parameters={"first": [first], "second": [second]},
        serialize_results=True,
    )[0]
    result = outputs["result"]
    assert len(outputs["combined"]["predictions"]) == 2
    assert outputs["combined"]["class_heading"] == second["class_heading"]
    assert result["predictions"][0]["detection_id"] == "second"
    assert result["predictions"][0]["obb"] == pytest.approx(
        second["predictions"][0]["obb"]
    )
    assert result["class_heading"] == second["class_heading"]
    engine.close()


@pytest.mark.parametrize("tensor", [False, True])
def test_all_empty_native_sources_keep_serializable_policy(tensor):
    sources = [_response(identity="first"), _response(identity="second")]
    for source in sources:
        source["predictions"] = []
    result = serialise_native_obb(_run(sources, tensor=tensor))
    assert result["predictions"] == []
    assert result["class_heading"] == sources[0]["class_heading"]


@pytest.mark.parametrize("tensor", [False, True])
def test_native_vote_compares_footprints_instead_of_enclosing_bounds(tensor):
    sources = [
        _response(identity="first", sides=(50.0, 3.0), angle=math.pi / 4),
        _response(identity="second", sides=(50.0, 3.0), angle=-math.pi / 4),
    ]
    result = serialise_native_obb(_run(sources, tensor=tensor, iou_threshold=0.5))
    assert result["predictions"] == []


@pytest.mark.parametrize("tensor", [False, True])
def test_native_block_keeps_source_carriers_unchanged(tensor):
    sources = [
        native_obb_from_inference(_response(identity="first", confidence=0.6)),
        native_obb_from_inference(_response(identity="second", confidence=0.9)),
    ]
    sources[1].metadata["class_heading"]["additional"] = True
    if tensor:
        sources = [native_obb_to_tensor(source) for source in sources]
    before = [serialise_native_obb(source) for source in sources]
    _run(sources)
    assert [serialise_native_obb(source) for source in sources] == before
    for source in sources:
        if isinstance(source, sv.Detections):
            assert "_consensus_source_order" not in source.data
        else:
            assert all(
                "_consensus_source_order" not in row for row in source.bboxes_metadata
            )


@pytest.mark.parametrize("tensor", [False, True])
def test_native_carrier_requires_frozen_policy_for_every_observed_class(tensor):
    sources = [
        native_obb_from_inference(_response(identity=str(index))) for index in range(2)
    ]
    del sources[1].metadata["class_heading"]["plane"]
    with pytest.raises(ValueError, match="frozen class semantics"):
        _run(sources, tensor=tensor)


@pytest.mark.parametrize("tensor", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_native_http_input_initializes_missing_lineage_before_consensus(
    monkeypatch, tensor, explicit
):
    from roboflow_workflows import environment
    from roboflow_workflows.core_steps.common.oriented import (
        _native_obb_to_sv,
        deserialize_native_obb,
    )

    payload = _response(identity="http")
    row = payload["predictions"][0]
    del row["parent_id"]
    if explicit:
        row.update(
            parent_id="crop",
            root_parent_id="root",
            parent_origin={"offset_x": 3, "offset_y": 5, "width": 100, "height": 80},
        )
    monkeypatch.setattr(environment, "ENABLE_TENSOR_DATA_REPRESENTATION", tensor)
    source = deserialize_native_obb("external", payload)
    lineage = _native_obb_to_sv(source).data
    assert lineage["parent_id"].tolist() == ["crop" if explicit else "external"]
    assert lineage["root_parent_id"].tolist() == ["root" if explicit else "external"]
    if explicit:
        np.testing.assert_array_equal(lineage["parent_coordinates"], [[3, 5]])
    result = serialise_native_obb(_run([source], required_votes=1))
    assert result["predictions"][0]["parent_id"] == ("crop" if explicit else "external")


@pytest.mark.parametrize(
    "empty_by", ["confidence", "votes", "per_class", "after_consensus", "after_combine"]
)
def test_compiled_empty_consensus_output_retains_crop_root_frame(empty_by):
    from roboflow_workflows.execution_engine.core import ExecutionEngine

    payload = _response(identity="crop-row")
    payload["image"] = {"width": 101, "height": 79}
    payload["predictions"][0].update(
        parent_id="crop",
        root_parent_id="root",
        parent_origin={"offset_x": 0, "offset_y": 0, "width": 101, "height": 79},
        root_parent_origin={
            "offset_x": 13,
            "offset_y": 17,
            "width": 300,
            "height": 200,
        },
    )
    workflow = {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowBatchInput",
                "name": "predictions",
                "kind": ["oriented_object_detection_prediction"],
            }
        ],
        "steps": [
            {
                "type": "roboflow_core/detections_consensus@v2",
                "name": "consensus",
                "predictions_batches": ["$inputs.predictions"],
                "required_votes": 2 if empty_by == "votes" else 1,
                "confidence": 0.99 if empty_by == "confidence" else 0,
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "root",
                "selector": "$steps.consensus.predictions",
                "coordinates_system": "parent",
            },
            {
                "type": "JsonField",
                "name": "own",
                "selector": "$steps.consensus.predictions",
                "coordinates_system": "own",
            },
        ],
    }
    if empty_by in {"per_class", "after_consensus", "after_combine"}:
        selector = "$inputs.predictions"
        if empty_by == "per_class":
            workflow["steps"] = []
        elif empty_by == "after_combine":
            workflow["steps"] = [
                {
                    "type": "roboflow_core/detections_combine@v1",
                    "name": "combined",
                    "prediction_one": selector,
                    "prediction_two": selector,
                }
            ]
            selector = "$steps.combined.predictions"
        else:
            selector = "$steps.consensus.predictions"
        workflow["steps"].append(
            {
                "type": "roboflow_core/per_class_confidence_filter@v1",
                "name": "filtered",
                "predictions": selector,
                "class_thresholds": {"plane": 0.99},
            }
        )
        for output in workflow["outputs"]:
            output["selector"] = "$steps.filtered.predictions"
    engine = ExecutionEngine.init(workflow_definition=workflow)
    try:
        result = engine.run(
            runtime_parameters={"predictions": [payload]}, serialize_results=True
        )[0]
    finally:
        engine.close()
    assert result["root"]["predictions"] == []
    assert result["root"]["image"] == {"height": 200, "width": 300}
    assert result["root"]["class_heading"] == payload["class_heading"]
    assert result["own"]["image"] == payload["image"]


@pytest.mark.parametrize("tensor", [False, True])
def test_native_consensus_accepts_identity_perspective_host_caches(tensor):
    from roboflow_workflows.core_steps.common.utils import (
        attach_parents_coordinates_to_sv_detections,
    )
    from roboflow_workflows.core_steps.transformations.perspective_correction import (
        v1,
        v1_tensor,
    )
    from roboflow_workflows.execution_engine.entities.base import (
        ImageParentMetadata,
        WorkflowImageData,
    )

    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata("same-image"),
        numpy_image=np.zeros((80, 100, 3), np.uint8),
    )
    source = attach_parents_coordinates_to_sv_detections(
        native_obb_from_inference(_response(identity="original")), image
    )
    if tensor:
        source = native_obb_to_tensor(source)
    before = serialise_native_obb(source)
    module = v1_tensor if tensor else v1
    warped = module.PerspectiveCorrectionBlockV1().run(
        images=[image],
        predictions=[source],
        perspective_polygons=[[0, 0], [99, 0], [99, 79], [0, 79]],
        transformed_rect_width=100,
        transformed_rect_height=80,
        extend_perspective_polygon_by_detections_anchor="CENTER",
        warp_image=False,
    )[0]["corrected_coordinates"]
    result = _run([source, warped], required_votes=2)
    assert len(result) == 1
    assert serialise_native_obb(result)["predictions"][0]["obb"] == pytest.approx(
        before["predictions"][0]["obb"]
    )
    assert serialise_native_obb(source) == before
