import json
import math

import numpy as np
import pytest
import supervision as sv
from roboflow_workflows.core_steps.common.oriented import (
    annotate_native_obb,
    native_obb_from_inference,
    native_obb_to_tensor,
    serialise_native_obb,
)
from roboflow_workflows.core_steps.common.tensor_native import (
    native_detections_to_root_coordinates,
    take_detections_by_indices,
)
from roboflow_workflows.core_steps.common.utils import (
    scale_sv_detections,
    sv_detections_to_root_coordinates,
)


def response():
    return dict(
        task_type="oriented-object-detection",
        image=dict(width=101, height=79),
        class_heading={"plane": True, "stone": False},
        predictions=[
            dict(
                obb=[40.25, 30.75, 12.5, 25.75, -2.9],
                confidence=0.9,
                class_id=7,
                heading=True,
                detection_id="plane0",
                **{"class": "plane"}
            ),
            dict(
                obb=[70.0, 50.0, 10.0, 10.0, math.pi / 2],
                confidence=0.8,
                class_id=9,
                heading=False,
                detection_id="stone1",
                **{"class": "stone"}
            ),
        ],
    )


def test_native_serialization_never_drops_heading_or_rounds_geometry():
    original = response()
    transport = native_obb_from_inference(original)
    result = serialise_native_obb(transport)
    for source, row in zip(original["predictions"], result["predictions"]):
        assert row["obb"] == pytest.approx(source["obb"])
        assert row["heading"] is source["heading"]
        assert row["class_id"] == source["class_id"]
        assert row["detection_id"] == source["detection_id"]
    from roboflow_workflows.core_steps.common.serializers import serialise_sv_detections

    assert serialise_sv_detections(transport) == result
    assert "x" not in result["predictions"][0]
    assert json.loads(json.dumps(result))["class_heading"] == original["class_heading"]


@pytest.mark.parametrize("tensor", [False, True])
def test_row_selection_and_combine_keep_front_alignment(tensor):
    original = native_obb_from_inference(response())
    if tensor:
        from roboflow_workflows.core_steps.transformations.detections_combine.v1_tensor import (
            DetectionsCombineBlockV1,
        )

        original = native_obb_to_tensor(original)
        first = take_detections_by_indices(original, [1])
        second = take_detections_by_indices(original, [0])
        combined = DetectionsCombineBlockV1().run(first, second)["predictions"]
    else:
        from roboflow_workflows.core_steps.transformations.detections_combine.v1 import (
            DetectionsCombineBlockV1,
        )

        first, second = original[[1]], original[[0]]
        combined = DetectionsCombineBlockV1().run(first, second)["predictions"]
    rows = serialise_native_obb(combined)["predictions"]
    assert [r["detection_id"] for r in rows] == ["stone1", "plane0"]
    assert [r["heading"] for r in rows] == [False, True]
    assert rows[1]["obb"] == pytest.approx(response()["predictions"][0]["obb"])


@pytest.mark.parametrize("tensor", [False, True])
def test_root_offset_moves_shape_and_front_together_in_float(tensor):
    transport = native_obb_from_inference(response())
    for key, value in {
        "root_parent_coordinates": [10, 20],
        "root_parent_dimensions": [200, 300],
        "root_parent_id": "root",
        "parent_id": "crop",
    }.items():
        transport.data[key] = np.asarray([value] * len(transport))
        transport.metadata[key] = value
    if tensor:
        transport = native_detections_to_root_coordinates(
            native_obb_to_tensor(transport)
        )
    else:
        transport = sv_detections_to_root_coordinates(transport)
    rows = serialise_native_obb(transport)
    assert rows["image"] == dict(width=300, height=200)
    assert rows["predictions"][0]["obb"] == pytest.approx(
        [50.25, 50.75, 12.5, 25.75, -2.9]
    )


def test_scale_preserves_order_and_anisotropy_is_explicit_quadrilateral():
    transport = native_obb_from_inference(response())
    scaled = scale_sv_detections(transport, scale=1.7)
    row = serialise_native_obb(scaled)["predictions"][0]
    assert row["obb"] == pytest.approx(
        [40.25 * 1.7, 30.75 * 1.7, 12.5 * 1.7, 25.75 * 1.7, -2.9]
    )
    warped = scale_sv_detections(
        transport, scale=(1.4, 0.7), update_scaling_metadata=False
    )
    row = serialise_native_obb(warped)["predictions"][0]
    assert "obb" not in row and row["geometry_type"] == "quadrilateral"
    corners = np.asarray(row["corners"])
    before = transport.data["xyxyxyxy"][0]
    np.testing.assert_allclose(corners, before * [1.4, 0.7], atol=1e-10)
    np.testing.assert_allclose(
        corners[1] - corners[0], (before[1] - before[0]) * [1.4, 0.7]
    )
    restored = serialise_native_obb(
        native_obb_from_inference(dict(warped=1, **serialise_native_obb(warped)))
    )
    np.testing.assert_allclose(restored["predictions"][0]["corners"], corners)


@pytest.mark.parametrize("field", ["heading", "detection_id", "obb"])
def test_missing_native_semantics_rejects(field):
    malformed = response()
    del malformed["predictions"][0][field]
    with pytest.raises((ValueError, KeyError)):
        native_obb_from_inference(malformed)


def test_conflicting_heading_rejects_instead_of_defaulting():
    malformed = response()
    malformed["predictions"][0]["heading"] = False
    with pytest.raises(ValueError, match="heading"):
        native_obb_from_inference(malformed)


@pytest.mark.parametrize("tensor", [False, True])
def test_empty_native_results_retain_frozen_schema(tensor):
    value = response()
    value["predictions"] = []
    transport = native_obb_from_inference(value)
    if tensor:
        transport = native_obb_to_tensor(transport)
    assert serialise_native_obb(transport) == value


def test_native_visualization_uses_rotated_footprint_and_front():
    value = response()
    value["predictions"] = value["predictions"][:1]
    transport = native_obb_from_inference(value)
    image = np.zeros((79, 101, 3), np.uint8)
    annotated = annotate_native_obb(image, transport)
    assert np.count_nonzero(annotated) > 0
    assert np.any(np.all(annotated == [51, 182, 255], axis=-1))
    # Center stays empty: this is a footprint/front-edge render, not an AABB fill.
    assert not annotated[31, 40].any()


@pytest.mark.parametrize("tensor", [False, True])
def test_dynamic_crop_and_root_restore_preserve_original_native_tuple(tensor):
    from roboflow_workflows.core_steps.common.utils import (
        attach_parents_coordinates_to_batch_of_sv_detections,
    )
    from roboflow_workflows.core_steps.transformations.dynamic_crop.v1 import crop_image
    from roboflow_workflows.execution_engine.entities.base import (
        Batch,
        ImageParentMetadata,
        WorkflowImageData,
    )

    original = response()
    original["predictions"] = original["predictions"][:1]
    transport = native_obb_from_inference(original)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="image0"),
        numpy_image=np.zeros((79, 101, 3), np.uint8),
    )
    transport = attach_parents_coordinates_to_batch_of_sv_detections(
        images=Batch([image], indices=[(0,)]), predictions=[transport]
    )[0]
    cropped = crop_image(image, transport, mask_opacity=0, background_color="black")[0]
    localized = attach_parents_coordinates_to_batch_of_sv_detections(
        images=Batch([cropped["crops"]], indices=[(0,)]),
        predictions=[cropped["predictions"]],
    )[0]
    restored = (
        native_detections_to_root_coordinates(native_obb_to_tensor(localized))
        if tensor
        else sv_detections_to_root_coordinates(localized)
    )
    assert serialise_native_obb(restored)["predictions"][0]["obb"] == pytest.approx(
        original["predictions"][0]["obb"], abs=1e-9
    )


def test_workflow_schema_accepts_explicit_quadrilateral_result():
    from roboflow_workflows.core_steps.common.oriented_entities import (
        OrientedWorkflowInferenceResponse,
    )

    transport = native_obb_from_inference(response())
    warped = scale_sv_detections(
        transport, scale=(1.4, 0.7), update_scaling_metadata=False
    )
    parsed = OrientedWorkflowInferenceResponse.model_validate(
        serialise_native_obb(warped)
    )
    assert parsed.predictions[0].geometry_type == "quadrilateral"


@pytest.mark.parametrize("tensor", [False, True])
def test_combination_preserves_full_frozen_maps_and_rejects_conflicts(tensor):
    from roboflow_workflows.core_steps.common.oriented import merge_native_semantics

    first = native_obb_from_inference(response())
    other = response()
    other["class_heading"]["unused"] = True
    second = native_obb_from_inference(other)
    if tensor:
        from roboflow_workflows.core_steps.transformations.detections_combine.v1_tensor import (
            DetectionsCombineBlockV1,
        )

        first, second = native_obb_to_tensor(first), native_obb_to_tensor(second)
    else:
        from roboflow_workflows.core_steps.transformations.detections_combine.v1 import (
            DetectionsCombineBlockV1,
        )
    combined = DetectionsCombineBlockV1().run(first, second)["predictions"]
    assert serialise_native_obb(combined)["class_heading"]["unused"] is True
    metadata = second.image_metadata if tensor else second.metadata
    metadata["class_heading"]["plane"] = False
    with pytest.raises(ValueError, match="Conflicting"):
        merge_native_semantics(first, second)


@pytest.mark.parametrize("tensor", [False, True])
def test_declared_kind_deserialization_uses_current_carrier(monkeypatch, tensor):
    from roboflow_workflows import environment
    from roboflow_workflows.core_steps.common.oriented import deserialize_native_obb

    monkeypatch.setattr(environment, "ENABLE_TENSOR_DATA_REPRESENTATION", tensor)
    value = deserialize_native_obb("input", response())
    assert isinstance(value, sv.Detections) is not tensor
    assert serialise_native_obb(value)["predictions"][0]["obb"] == pytest.approx(
        response()["predictions"][0]["obb"]
    )


@pytest.mark.parametrize("tensor", [False, True])
@pytest.mark.parametrize(
    "operation",
    [
        {"type": "DetectionsShift", "shift_x": 30, "shift_y": 20},
        {"type": "DetectionsOffset", "offset_x": 20, "offset_y": 10},
    ],
)
@pytest.mark.parametrize("marker", ["both", "image", "rows", "empty"])
def test_query_geometry_operations_reject_native_without_mutation(
    monkeypatch, tensor, operation, marker
):
    """Reject native geometry through the dispatcher before changing either view."""
    from roboflow_workflows.core_steps.common.query_language.errors import (
        InvalidInputTypeError,
    )
    from roboflow_workflows.core_steps.common.query_language.operations.core import (
        execute_operations,
    )
    from roboflow_workflows.core_steps.common.query_language.operations.detections import (
        base,
    )

    original = response()
    if marker == "empty":
        original["predictions"] = []
    transport = native_obb_from_inference(original)
    if marker == "rows":
        del transport.metadata["native_obb"]
    elif marker == "image":
        del transport.data["native_obb"]
    if tensor:
        transport = native_obb_to_tensor(transport)
    monkeypatch.setattr(base, "ENABLE_TENSOR_DATA_REPRESENTATION", tensor)
    before_bounds = np.asarray(transport.xyxy).copy()
    before_serialized = serialise_native_obb(transport)

    with pytest.raises(InvalidInputTypeError, match="native oriented detections"):
        execute_operations(transport, operations=[operation])

    np.testing.assert_array_equal(transport.xyxy, before_bounds)
    assert serialise_native_obb(transport) == before_serialized


@pytest.mark.parametrize("tensor", [False, True])
@pytest.mark.parametrize(
    "operation, expected",
    [
        (
            {"type": "DetectionsShift", "shift_x": 30, "shift_y": 20},
            [[40, 40, 60, 60]],
        ),
        (
            {"type": "DetectionsOffset", "offset_x": 20, "offset_y": 10},
            [[0, 15, 40, 45]],
        ),
    ],
)
def test_query_geometry_operations_preserve_ordinary_detection_behavior(
    monkeypatch, tensor, operation, expected
):
    """Keep ordinary AABB shift/offset results and input-copy behavior intact."""
    from roboflow_workflows.core_steps.common.query_language.operations.core import (
        execute_operations,
    )
    from roboflow_workflows.core_steps.common.query_language.operations.detections import (
        base,
    )

    transport = sv.Detections(
        xyxy=np.array([[10, 20, 30, 40]], dtype=float),
        class_id=np.array([7]),
        confidence=np.array([0.9]),
    )
    if tensor:
        import torch

        from inference_models.models.base.object_detection import Detections

        transport = Detections(
            xyxy=torch.as_tensor(transport.xyxy),
            class_id=torch.as_tensor(transport.class_id),
            confidence=torch.as_tensor(transport.confidence),
        )
    monkeypatch.setattr(base, "ENABLE_TENSOR_DATA_REPRESENTATION", tensor)

    result = execute_operations(transport, operations=[operation])

    np.testing.assert_array_equal(result.xyxy, expected)
    np.testing.assert_array_equal(transport.xyxy, [[10, 20, 30, 40]])
