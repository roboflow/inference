from functools import partial
from importlib import import_module
from unittest.mock import patch

import numpy as np
import pytest
import supervision as sv
from roboflow_workflows.core_steps.common import keypoints
from roboflow_workflows.core_steps.common.keypoints import (
    COCO_KEYPOINT_NAMES,
    KEYPOINT_PADDING_CLASS_NAME,
    is_coco_skeleton,
    real_keypoints_count,
)


def test_real_keypoints_count_counts_only_named_slots() -> None:
    # given a padded detection: 2 real keypoints, 2 trailing padding slots
    class_names = np.array(
        ["nose", "eye", KEYPOINT_PADDING_CLASS_NAME, KEYPOINT_PADDING_CLASS_NAME],
        dtype=object,
    )

    # when / then
    assert real_keypoints_count(class_names, total=len(class_names)) == 2


def test_real_keypoints_count_when_no_padding() -> None:
    class_names = np.array(["nose", "eye", "ear"], dtype=object)
    assert real_keypoints_count(class_names, total=len(class_names)) == 3


def test_real_keypoints_count_when_all_padding() -> None:
    class_names = np.array([KEYPOINT_PADDING_CLASS_NAME] * 3, dtype=object)
    assert real_keypoints_count(class_names, total=3) == 0


def test_real_keypoints_count_falls_back_to_total_without_names() -> None:
    # No class-name metadata -> keypoints are emitted unchanged (fallback to total).
    assert real_keypoints_count(None, total=4) == 4


@pytest.mark.parametrize(
    "detections_count,max_keypoints",
    [(0, 0), (10, 0), (1, 1_000_000), (1000, 1000)],
)
def test_validate_keypoints_padding_accepts_bounded_shapes(
    detections_count: int, max_keypoints: int
) -> None:
    keypoints.validate_keypoints_padding(detections_count, max_keypoints)


def test_validate_keypoints_padding_rejects_above_limit() -> None:
    with pytest.raises(ValueError, match="exceeding the limit"):
        keypoints.validate_keypoints_padding(1, 1_000_001)


@pytest.mark.parametrize(
    "padding_path", ["numpy", "boundary", "native", "v1", "v2", "v3"]
)
@pytest.mark.parametrize("large_index", [0, 2])
def test_keypoint_padding_rejects_before_allocation(
    monkeypatch: pytest.MonkeyPatch, padding_path: str, large_index: int
) -> None:
    from roboflow_workflows.core_steps.common.tensor_native import (
        build_native_key_points,
    )
    from roboflow_workflows.core_steps.common.utils import (
        add_inference_keypoints_to_sv_detections,
    )
    from roboflow_workflows.execution_engine.v1.dynamic_blocks.representation_boundary import (
        _attach_padded_keypoint_columns,
    )

    # Only two real keypoints, but padding would require six slots.
    monkeypatch.setattr(keypoints, "MAX_KEYPOINTS_PADDING_CELLS", 4)
    predictions = [{"keypoints": []} for _ in range(3)]
    predictions[large_index]["keypoints"] = [
        {"x": 1, "y": 2, "class": "tip", "class_id": 0, "confidence": 0.9},
        {"x": 3, "y": 4, "class": "base", "class_id": 1, "confidence": 0.8},
    ]
    metadata = [
        {
            "keypoints_xy": [[k["x"], k["y"]] for k in p["keypoints"]],
            "keypoints_confidence": [k["confidence"] for k in p["keypoints"]],
            "keypoints_class_id": [k["class_id"] for k in p["keypoints"]],
            "keypoints_class_name": [k["class"] for k in p["keypoints"]],
        }
        for p in predictions
    ]
    if padding_path == "numpy":
        pad = partial(
            add_inference_keypoints_to_sv_detections,
            inference_prediction=predictions,
            detections=sv.Detections(xyxy=np.zeros((3, 4))),
        )
    elif padding_path == "boundary":
        pad = partial(
            _attach_padded_keypoint_columns,
            data={},
            bboxes_metadata=metadata,
            detections_number=3,
        )
    elif padding_path == "native":
        pad = partial(
            build_native_key_points,
            per_instance_xy=[m["keypoints_xy"] for m in metadata],
            per_instance_confidence=[m["keypoints_confidence"] for m in metadata],
            object_class_ids=[0] * 3,
            image_metadata={},
        )
    else:
        module = import_module(
            "roboflow_workflows.core_steps.models.roboflow."
            f"keypoint_detection.{padding_path}_tensor"
        )
        pad = partial(
            module._native_key_points_from_inference_predictions,
            detection_dicts=predictions,
            image_metadata={},
        )

    with (
        patch("numpy.zeros") as numpy_zeros,
        patch("numpy.full") as numpy_full,
        patch("torch.zeros") as torch_zeros,
        pytest.raises(ValueError, match="Keypoint padding requires 6 slots"),
    ):
        pad()
    numpy_zeros.assert_not_called()
    numpy_full.assert_not_called()
    torch_zeros.assert_not_called()


@pytest.mark.parametrize(
    "kind",
    [
        "object_detection_prediction",
        "instance_segmentation_prediction",
        "keypoint_detection_prediction",
    ],
)
def test_workflow_rejects_excessive_keypoint_padding(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    from roboflow_workflows.core_steps import loader
    from roboflow_workflows.core_steps.common.deserializers import (
        deserialize_detections_kind,
    )
    from roboflow_workflows.errors import RuntimeInputError
    from roboflow_workflows.execution_engine.core import ExecutionEngine

    monkeypatch.setattr(keypoints, "MAX_KEYPOINTS_PADDING_CELLS", 4)
    # Exercise the affected NumPy input route regardless of the suite's tensor flag.
    monkeypatch.setitem(loader.KINDS_DESERIALIZERS, kind, deserialize_detections_kind)
    engine = ExecutionEngine.init(
        workflow_definition={
            "version": "1.3.0",
            "inputs": [
                {"type": "WorkflowBatchInput", "name": "detections", "kind": [kind]}
            ],
            "steps": [],
            "outputs": [
                {
                    "type": "JsonField",
                    "name": "detections",
                    "selector": "$inputs.detections",
                }
            ],
        },
        init_parameters={},
    )
    prediction = {
        "x": 10,
        "y": 10,
        "width": 2,
        "height": 2,
        "confidence": 0.9,
        "class_id": 0,
        "class": "object",
    }
    predictions = [dict(prediction) for _ in range(3)]
    predictions[0]["keypoints"] = [
        {"x": 1, "y": 2, "class": "nose", "class_id": 0, "confidence": 0.9}
    ] * 2
    payload = {"image": {"width": 20, "height": 20}, "predictions": predictions}

    with pytest.raises(RuntimeInputError, match="Keypoint padding requires 6 slots"):
        engine.run(runtime_parameters={"detections": payload})

    # The exact limit still accepts a legitimate ragged prediction.
    payload["predictions"] = predictions[:2]
    result = engine.run(runtime_parameters={"detections": payload})
    assert result[0]["detections"].data["keypoints_xy"].shape == (2, 2, 2)


def test_is_coco_skeleton_requires_ids_at_coco_indices() -> None:
    assert is_coco_skeleton([0, 2, 4], ["nose", "right_eye", "right_ear"])
    assert is_coco_skeleton(range(17), COCO_KEYPOINT_NAMES)
    # COCO names numbered in another order are a different skeleton
    assert not is_coco_skeleton([0, 1, 2], ["left_eye", "right_eye", "nose"])
    assert not is_coco_skeleton([0, 1], ["nose", "tip"])
    assert not is_coco_skeleton([17], ["nose"])
    assert not is_coco_skeleton([], [])


def _remote_keypoint_dicts(per_detection: list) -> list:
    # per_detection: list of [(class_id, class_name), ...]; one dict per detection,
    # shaped like the remote keypoint model response after threshold filtering.
    return [
        {
            "x": 10,
            "y": 20,
            "width": 100,
            "height": 200,
            "confidence": 0.9,
            "class": "person",
            "class_id": 0,
            "keypoints": [
                {
                    "x": 100.0 + 10 * class_id,
                    "y": 100.0 + 20 * class_id,
                    "confidence": 0.5 + 0.01 * class_id,
                    "class": class_name,
                    "class_id": class_id,
                }
                for class_id, class_name in keypoints
            ],
        }
        for keypoints in per_detection
    ]


def _build_native_key_points(builder: str, predictions: list):
    if builder == "native":
        from roboflow_workflows.core_steps.common.tensor_native import (
            build_native_key_points,
        )

        return build_native_key_points(
            per_instance_xy=[
                [[k["x"], k["y"]] for k in p["keypoints"]] for p in predictions
            ],
            per_instance_confidence=[
                [k["confidence"] for k in p["keypoints"]] for p in predictions
            ],
            object_class_ids=[p["class_id"] for p in predictions],
            image_metadata={},
            per_instance_keypoint_class_ids=[
                [k["class_id"] for k in p["keypoints"]] for p in predictions
            ],
            per_instance_keypoint_class_names=[
                [k["class"] for k in p["keypoints"]] for p in predictions
            ],
        )
    module = import_module(
        "roboflow_workflows.core_steps.models.roboflow."
        f"keypoint_detection.{builder}_tensor"
    )
    return module._native_key_points_from_inference_predictions(
        detection_dicts=predictions, image_metadata={}
    )


@pytest.mark.parametrize("builder", ["native", "v1", "v2", "v3"])
def test_native_key_points_are_placed_at_class_id_slots(builder: str) -> None:
    # given: a custom skeleton, first instance missing slots 1 and 3, second missing 0, 2, 4
    predictions = _remote_keypoint_dicts(
        [[(0, "a"), (2, "c"), (4, "e")], [(1, "b"), (3, "d")]]
    )

    # when
    key_points = _build_native_key_points(builder, predictions)

    # then
    assert key_points.xy.shape == (2, 5, 2)
    for class_id in [0, 2, 4]:
        assert key_points.xy[0, class_id].tolist() == [
            100.0 + 10 * class_id,
            100.0 + 20 * class_id,
        ]
        assert key_points.confidence[0, class_id].item() == pytest.approx(
            0.5 + 0.01 * class_id
        )
    for class_id in [1, 3]:
        assert key_points.xy[1, class_id].tolist() == [
            100.0 + 10 * class_id,
            100.0 + 20 * class_id,
        ]
    assert key_points.xy[0, [1, 3]].abs().sum().item() == 0.0
    assert key_points.xy[1, [0, 2, 4]].abs().sum().item() == 0.0
    visible = key_points.to_supervision().visible
    assert visible[0].tolist() == [True, False, True, False, True]
    assert visible[1].tolist() == [False, True, False, True, False]
    assert key_points.class_id.tolist() == [0, 0]


@pytest.mark.parametrize("builder", ["native", "v1", "v2", "v3"])
def test_native_key_points_of_coco_skeleton_are_widened_to_full_skeleton(
    builder: str,
) -> None:
    # given: only the upper body (COCO keypoints 0-10) is in frame
    predictions = _remote_keypoint_dicts(
        [[(i, COCO_KEYPOINT_NAMES[i]) for i in range(11)]]
    )

    # when
    key_points = _build_native_key_points(builder, predictions)

    # then
    assert key_points.xy.shape == (1, 17, 2)
    visible = key_points.to_supervision().visible
    assert visible[0, :11].all()
    assert not visible[0, 11:].any()


@pytest.mark.parametrize("builder", ["native", "v1", "v2", "v3"])
def test_native_key_points_of_coco_names_in_other_order_are_not_widened(
    builder: str,
) -> None:
    # given: a custom skeleton that reuses COCO names but numbers them differently
    predictions = _remote_keypoint_dicts(
        [[(0, "left_eye"), (1, "right_eye"), (2, "nose")]]
    )

    # when
    key_points = _build_native_key_points(builder, predictions)

    # then
    assert key_points.xy.shape == (1, 3, 2)


def test_native_key_points_without_class_ids_are_packed_leading() -> None:
    from roboflow_workflows.core_steps.common.tensor_native import (
        build_native_key_points,
    )

    # when
    key_points = build_native_key_points(
        per_instance_xy=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0]]],
        per_instance_confidence=[[0.9, 0.8], [0.7]],
        object_class_ids=[0, 1],
        image_metadata={},
    )

    # then
    assert key_points.xy.tolist() == [
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [0.0, 0.0]],
    ]
    np.testing.assert_allclose(
        key_points.confidence.numpy(), [[0.9, 0.8], [0.7, 0.0]], rtol=1e-6
    )


@pytest.mark.parametrize("builder", ["v1", "v2", "v3"])
def test_remote_keypoints_without_class_ids_keep_their_positions(builder: str) -> None:
    # given: a response whose keypoints carry no class_id
    predictions = _remote_keypoint_dicts([[(0, "a"), (1, "b"), (2, "c")]])
    for keypoint in predictions[0]["keypoints"]:
        del keypoint["class_id"]

    # when
    key_points = _build_native_key_points(builder, predictions)

    # then
    assert key_points.xy.shape == (1, 3, 2)
    assert key_points.xy[0].tolist() == [[100.0, 100.0], [110.0, 120.0], [120.0, 140.0]]


def test_boundary_rebuilds_native_key_points_at_skeleton_slots() -> None:
    from roboflow_workflows.execution_engine.v1.dynamic_blocks.representation_boundary import (
        sv_detections_to_native_key_point_prediction,
    )

    # given: padded columns as `add_inference_keypoints_to_sv_detections` stores them;
    # the first detection has one real keypoint (right_eye) and two padding slots
    sv_detections = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10], [10, 10, 20, 20]], dtype=np.float64),
        class_id=np.array([0, 0]),
        confidence=np.array([0.9, 0.8], dtype=np.float32),
        data={
            "keypoints_xy": np.array(
                [
                    [[120.0, 140.0], [0.0, 0.0], [0.0, 0.0]],
                    [[150.0, 200.0], [160.0, 220.0], [170.0, 240.0]],
                ],
                dtype=np.float32,
            ),
            "keypoints_confidence": np.array(
                [[0.9, 0.0, 0.0], [0.8, 0.7, 0.6]], dtype=np.float32
            ),
            "keypoints_class_name": np.array(
                [
                    ["right_eye", "", ""],
                    ["left_shoulder", "right_shoulder", "left_elbow"],
                ],
                dtype=object,
            ),
            "keypoints_class_id": np.array([[2, 0, 0], [5, 6, 7]], dtype=int),
        },
    )

    # when
    key_points, _ = sv_detections_to_native_key_point_prediction(sv_detections)

    # then: padding did not land on slot 0 and did not block the COCO widening
    assert key_points.xy.shape == (2, 17, 2)
    assert key_points.xy[0, 2].tolist() == [120.0, 140.0]
    assert key_points.xy[0, 0].tolist() == [0.0, 0.0]
    assert key_points.xy[1, 5].tolist() == [150.0, 200.0]
    assert key_points.xy[1, 7].tolist() == [170.0, 240.0]
    visible = key_points.to_supervision().visible
    assert visible[0].tolist() == [i == 2 for i in range(17)]
    assert visible[1].tolist() == [i in (5, 6, 7) for i in range(17)]


@pytest.mark.parametrize("builder", ["native", "v1", "v2", "v3"])
def test_native_key_points_with_negative_class_id_keep_packed_layout(
    builder: str,
) -> None:
    # given: class ids reach the builders unchecked; a negative one cannot be a slot
    predictions = _remote_keypoint_dicts([[(0, "nose"), (-1, "right_eye")]])

    # when
    key_points = _build_native_key_points(builder, predictions)

    # then: packed as stored, no widening
    assert key_points.xy.shape == (1, 2, 2)
    assert key_points.xy[0].tolist() == [[100.0, 100.0], [90.0, 80.0]]


@pytest.mark.parametrize("builder", ["native", "v1", "v2", "v3"])
def test_native_key_points_with_oversized_class_id_keep_packed_layout(
    builder: str,
) -> None:
    # given: an id that would need a billion slots
    predictions = _remote_keypoint_dicts([[(0, "nose"), (1_000_000_000, "x")]])

    # when
    key_points = _build_native_key_points(builder, predictions)

    # then
    assert key_points.xy.shape == (1, 2, 2)
