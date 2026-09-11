from functools import partial
from importlib import import_module
from unittest.mock import patch

import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.common import keypoints
from inference.core.workflows.core_steps.common.keypoints import (
    KEYPOINT_PADDING_CLASS_NAME,
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
    from inference.core.workflows.core_steps.common.tensor_native import (
        build_native_key_points,
    )
    from inference.core.workflows.core_steps.common.utils import (
        add_inference_keypoints_to_sv_detections,
    )
    from inference.core.workflows.execution_engine.v1.dynamic_blocks.representation_boundary import (
        _attach_padded_keypoint_columns,
    )

    # Only two real keypoints, but padding would require six slots.
    monkeypatch.setattr(keypoints, "MAX_KEYPOINTS_PADDING_CELLS", 4)
    predictions = [{"keypoints": []} for _ in range(3)]
    predictions[large_index]["keypoints"] = [
        {"x": 1, "y": 2, "class": "nose", "class_id": 0, "confidence": 0.9}
    ] * 2
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
            "inference.core.workflows.core_steps.models.roboflow."
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
    from inference.core.workflows.core_steps import loader
    from inference.core.workflows.core_steps.common.deserializers import (
        deserialize_detections_kind,
    )
    from inference.core.workflows.errors import RuntimeInputError
    from inference.core.workflows.execution_engine.core import ExecutionEngine

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
