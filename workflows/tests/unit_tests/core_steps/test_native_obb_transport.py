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
    from copy import deepcopy

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
    before = deepcopy(transport)
    if tensor:
        from roboflow_workflows.core_steps.transformations.dynamic_crop.v1_tensor import (
            crop_image as tensor_crop,
        )

        cropped = tensor_crop(image, native_obb_to_tensor(transport), 0, "black")[0]
    else:
        cropped = crop_image(
            image, transport, mask_opacity=0, background_color="black"
        )[0]
    localized = cropped["predictions"]
    assert serialise_native_obb(localized)["image"] == {
        "height": cropped["crops"].numpy_image.shape[0],
        "width": cropped["crops"].numpy_image.shape[1],
    }
    restored = (
        native_detections_to_root_coordinates(localized)
        if tensor
        else sv_detections_to_root_coordinates(localized)
    )
    assert serialise_native_obb(restored)["predictions"][0]["obb"] == pytest.approx(
        original["predictions"][0]["obb"], abs=1e-9
    )

    if tensor:
        from roboflow_workflows.core_steps.common.tensor_native import (
            take_detections_by_indices,
        )

        empty = take_detections_by_indices(localized, [])
        rooted_empty = native_detections_to_root_coordinates(empty)
    else:
        rooted_empty = sv_detections_to_root_coordinates(localized[:0])
    assert serialise_native_obb(rooted_empty)["image"] == original["image"]
    assert serialise_native_obb(transport)["predictions"][0]["obb"] == pytest.approx(
        original["predictions"][0]["obb"]
    )

    assert transport.metadata == before.metadata
    for key in transport.data:
        np.testing.assert_array_equal(transport.data[key], before.data[key])


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
@pytest.mark.parametrize("empty", ["neither", "first", "second", "both"])
def test_combination_preserves_full_frozen_maps_and_rejects_conflicts(tensor, empty):
    from copy import deepcopy

    from roboflow_workflows.core_steps.common.oriented import (
        _native_obb_to_sv,
        merge_native_semantics,
    )

    original = response()
    other = response()
    original["predictions"] = original["predictions"][:1]
    other["predictions"] = other["predictions"][1:]
    original["class_heading"] = {"plane": True}
    other["class_heading"] = {"stone": False}
    if empty in {"first", "both"}:
        original["predictions"] = []
    if empty in {"second", "both"}:
        other["predictions"] = []
    first = native_obb_from_inference(original)
    other["class_heading"]["unused"] = True
    second = native_obb_from_inference(other)
    if tensor:
        from roboflow_workflows.core_steps.transformations.detections_combine.v1_tensor import (
            DetectionsCombineBlockV1,
        )

    else:
        from roboflow_workflows.core_steps.transformations.detections_combine.v1 import (
            DetectionsCombineBlockV1,
        )
    first, second = native_obb_to_tensor(first), native_obb_to_tensor(second)
    if not tensor:
        first, second = _native_obb_to_sv(first), _native_obb_to_sv(second)
    headers = [
        deepcopy(value.image_metadata if tensor else value.metadata)
        for value in (first, second)
    ]
    before = [serialise_native_obb(value) for value in (first, second)]
    combined = DetectionsCombineBlockV1().run(first, second)["predictions"]
    assert serialise_native_obb(combined)["class_heading"]["unused"] is True
    assert serialise_native_obb(combined)["image"] == original["image"]
    assert [row["class"] for row in serialise_native_obb(combined)["predictions"]] == [
        row["class"] for value in before for row in value["predictions"]
    ]
    assert [serialise_native_obb(value) for value in (first, second)] == before
    assert [
        value.image_metadata if tensor else value.metadata for value in (first, second)
    ] == headers
    metadata = second.image_metadata if tensor else second.metadata
    metadata["class_heading"]["plane"] = False
    with pytest.raises(ValueError, match="Conflicting"):
        merge_native_semantics(first, second)


@pytest.mark.parametrize("tensor", [False, True])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("tensor_input", [False, True])
def test_declared_kind_deserialization_uses_current_carrier(
    monkeypatch, tensor, empty, tensor_input
):
    from roboflow_workflows import environment
    from roboflow_workflows.core_steps.common.oriented import deserialize_native_obb

    original = response()
    if empty:
        original["predictions"] = []
    supplied = (
        native_obb_to_tensor(native_obb_from_inference(original))
        if tensor_input
        else original
    )
    monkeypatch.setattr(environment, "ENABLE_TENSOR_DATA_REPRESENTATION", tensor)
    value = deserialize_native_obb("input", supplied)
    assert isinstance(value, sv.Detections) is not tensor
    serialized = serialise_native_obb(value)
    assert serialized["image"] == original["image"]
    assert serialized["class_heading"] == original["class_heading"]
    if empty:
        assert serialized["predictions"] == []
    else:
        assert serialized["predictions"][0]["obb"] == pytest.approx(
            original["predictions"][0]["obb"]
        )


@pytest.mark.parametrize("tensor", [False, True])
@pytest.mark.parametrize(
    "operation",
    [
        {"type": "DetectionsShift", "shift_x": 30, "shift_y": 20},
        {"type": "DetectionsOffset", "offset_x": 20, "offset_y": 10},
        {
            "type": "DetectionsRename",
            "class_map": {"plane": "aircraft", "stone": "rock"},
        },
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


def test_legacy_corner_scaling_keeps_existing_rounded_aabb_contract():
    """Native float geometry does not change unmarked legacy corner carriers."""
    corners = np.array([[[1.1, 2.2], [11.3, 2.2], [11.3, 12.4], [1.1, 12.4]]])
    source = sv.Detections(
        xyxy=np.array([[1.1, 2.2, 11.3, 12.4]]), data={"xyxyxyxy": corners}
    )

    result = scale_sv_detections(source, 2, update_scaling_metadata=False)

    np.testing.assert_array_equal(result.xyxy, [[2, 4, 23, 25]])
    np.testing.assert_array_equal(result.data["xyxyxyxy"], corners)


def test_legacy_perspective_corners_keep_rounding_and_dtype():
    """The existing oriented carrier retains its float32 rounded corners."""
    from roboflow_workflows.core_steps.transformations.perspective_correction.v1 import (
        correct_detections,
    )
    from roboflow_workflows.core_steps.transformations.perspective_correction.v1_tensor import (
        _correct_oriented_box_in_metadata,
    )

    corners = np.array([[[1.1, 2.2], [11.3, 2.2], [11.3, 12.4], [1.1, 12.4]]])
    matrix = np.array([[1.0, 0.0, 0.25], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
    source = sv.Detections(
        xyxy=np.array([[1.1, 2.2, 11.3, 12.4]]), data={"xyxyxyxy": corners}
    )
    expected = np.around(corners + [0.25, 0.5]).astype(np.float32)

    result = correct_detections(source, matrix, 100, 100)
    entry = {"xyxyxyxy": corners[0]}
    _correct_oriented_box_in_metadata(entry, matrix)

    np.testing.assert_array_equal(result.data["xyxyxyxy"], expected)
    assert result.data["xyxyxyxy"].dtype == np.float32
    np.testing.assert_array_equal(entry["xyxyxyxy"], expected[0])
    assert entry["xyxyxyxy"].dtype == np.float32


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize(
    "boundary",
    [
        "converter",
        "serialized",
        "carrier",
        "tensor_serialized",
        "tensor_carrier",
        "tensor_rle_serialized",
        "tensor_rle_carrier",
    ],
)
def test_ordinary_detection_boundaries_reject_native_payloads(empty, boundary):
    from roboflow_workflows.core_steps.common.deserializers import (
        deserialize_detections_kind,
    )
    from roboflow_workflows.core_steps.common.utils import (
        convert_inference_detections_batch_to_sv_detections,
    )
    from roboflow_workflows.errors import RuntimeInputError

    payload = response()
    if empty:
        payload["predictions"] = []
    error = ValueError if boundary == "converter" else RuntimeInputError
    with pytest.raises(error, match="oriented_object_detection_prediction"):
        if boundary == "converter":
            convert_inference_detections_batch_to_sv_detections([payload])
        elif boundary.startswith("tensor"):
            from roboflow_workflows.core_steps.common.deserializers_tensor import (
                deserialize_detections_kind as deserialize_tensor,
            )

            if "rle" in boundary:
                from roboflow_workflows.core_steps.common.deserializers_tensor import (
                    deserialize_rle_detections_kind as deserialize_tensor,
                )
            value = (
                native_obb_to_tensor(native_obb_from_inference(payload))
                if boundary.endswith("carrier")
                else payload
            )
            deserialize_tensor("detections", value)
        else:
            value = (
                native_obb_from_inference(payload) if boundary == "carrier" else payload
            )
            deserialize_detections_kind("detections", value)


def test_ordinary_detection_boundaries_preserve_aabb_predictions():
    from roboflow_workflows.core_steps.common.deserializers import (
        deserialize_detections_kind,
    )
    from roboflow_workflows.core_steps.common.utils import (
        convert_inference_detections_batch_to_sv_detections,
    )

    payload = dict(
        image=dict(width=100, height=80),
        predictions=[
            dict(
                x=20,
                y=30,
                width=10,
                height=20,
                confidence=0.9,
                class_id=7,
                detection_id="ordinary",
                **{"class": "plane"}
            )
        ],
    )
    for value in (
        convert_inference_detections_batch_to_sv_detections([payload])[0],
        deserialize_detections_kind("detections", payload),
    ):
        np.testing.assert_array_equal(value.xyxy, [[15, 20, 25, 40]])
        assert value["detection_id"].tolist() == ["ordinary"]
        assert deserialize_detections_kind("detections", value) is value

    from roboflow_workflows.core_steps.common.deserializers_tensor import (
        deserialize_detections_kind as deserialize_tensor,
    )

    tensor_value = deserialize_tensor("detections", payload)
    np.testing.assert_array_equal(tensor_value.xyxy.cpu(), [[15, 20, 25, 40]])
    assert tensor_value.bboxes_metadata[0]["detection_id"] == "ordinary"
    tensor_value.bboxes_metadata = [None]
    assert deserialize_tensor("detections", tensor_value) is tensor_value

    from roboflow_workflows.core_steps.common.deserializers_tensor import (
        deserialize_rle_detections_kind as deserialize_rle_tensor,
    )

    payload["predictions"][0]["rle_mask"] = {"size": [80, 100], "counts": [8000]}
    rle_value = deserialize_rle_tensor("detections", payload)
    np.testing.assert_array_equal(rle_value.xyxy.cpu(), [[15, 20, 25, 40]])
    assert rle_value.mask.to_coco_rle_masks() == [{"size": [80, 100], "counts": [8000]}]
    assert deserialize_rle_tensor("detections", rle_value) is rle_value


@pytest.mark.parametrize("tensor", [False, True])
@pytest.mark.parametrize(
    "obb, window",
    [
        ([5, 50, 30, 20, 0], (0, 40, 20, 60)),
        ([50, 5, 20, 30, 0], (40, 0, 60, 20)),
        ([95, 95, 30, 30, 0], (80, 80, 100, 100)),
        ([-50, 50, 30, 20, 0], None),
    ],
)
def test_native_crop_clips_pixels_and_origin_without_clipping_ordered_front(
    tensor, obb, window
):
    from roboflow_workflows.core_steps.transformations.dynamic_crop import v1, v1_tensor
    from roboflow_workflows.execution_engine.entities.base import (
        ImageParentMetadata,
        WorkflowImageData,
    )

    payload = response()
    payload["image"] = dict(width=100, height=100)
    payload["predictions"] = payload["predictions"][:1]
    payload["predictions"][0]["obb"] = obb
    prediction = native_obb_from_inference(payload)
    image_data = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="image0"), numpy_image=image_data
    )
    if tensor:
        result = v1_tensor.crop_image(
            image, native_obb_to_tensor(prediction), 0, "black"
        )[0]
    else:
        result = v1.crop_image(image, prediction, 0, "black")[0]
    if window is None:
        assert result == {"crops": None, "predictions": None}
        return

    x1, y1, x2, y2 = window
    crop = result["crops"]
    np.testing.assert_array_equal(crop.numpy_image, image_data[y1:y2, x1:x2])
    assert crop.parent_metadata.parent_id == "plane0"
    assert crop.parent_metadata.origin_coordinates.left_top_x == x1
    assert crop.parent_metadata.origin_coordinates.left_top_y == y1
    row = serialise_native_obb(result["predictions"])["predictions"][0]
    assert row["obb"] == pytest.approx([obb[0] - x1, obb[1] - y1, *obb[2:]])
    assert row["heading"] is True
    assert row["detection_id"] == "plane0"
    assert (
        serialise_native_obb(result["predictions"])["class_heading"]
        == payload["class_heading"]
    )


@pytest.mark.parametrize(
    "tensor, row_metadata", [(False, None), (True, None), (True, {"class": "plane"})]
)
def test_query_rename_preserves_ordinary_classes_and_ids(
    monkeypatch, tensor, row_metadata
):
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
        data={"class_name": np.array(["plane"])},
    )
    if tensor:
        import torch

        from inference_models.models.base.object_detection import Detections

        transport = Detections(
            xyxy=torch.as_tensor(transport.xyxy),
            class_id=torch.as_tensor(transport.class_id),
            confidence=torch.as_tensor(transport.confidence),
            bboxes_metadata=[row_metadata],
            image_metadata={"class_names": {7: "plane"}},
        )
    monkeypatch.setattr(base, "ENABLE_TENSOR_DATA_REPRESENTATION", tensor)
    result = execute_operations(
        transport,
        operations=[{"type": "DetectionsRename", "class_map": {"plane": "aircraft"}}],
    )
    names = (
        result.image_metadata["class_names"][int(result.class_id[0])]
        if tensor
        else result["class_name"][0]
    )
    assert names == "aircraft"
    assert int(result.class_id[0]) == 0


def test_query_rename_declaration_excludes_native_kind():
    from roboflow_workflows.core_steps.common.query_language.entities.operations import (
        DetectionsRename,
    )

    schema = DetectionsRename.model_json_schema()
    for kinds in (schema["input_kind"], schema["output_kind"]):
        names = [kind.name for kind in kinds]
        assert "object_detection_prediction" in names
        assert "oriented_object_detection_prediction" not in names


@pytest.mark.parametrize("tensor", [False, True])
@pytest.mark.parametrize("fusion", ["consensus", "combine"])
@pytest.mark.parametrize("consumer", ["per_class", "parent_class", "filter"])
def test_native_colliding_class_ids_keep_row_names_through_filter(
    monkeypatch, tensor, fusion, consumer
):
    from roboflow_workflows.core_steps.fusion.detections_consensus.v1 import (
        AggregationMode,
        MaskAggregationMode,
    )
    from roboflow_workflows.core_steps.fusion.detections_consensus.v2 import (
        DetectionsConsensusBlockV2,
    )
    from roboflow_workflows.core_steps.transformations.detections_combine import (
        v1 as combine,
    )
    from roboflow_workflows.core_steps.transformations.detections_combine import (
        v1_tensor as combine_tensor,
    )
    from roboflow_workflows.core_steps.transformations.per_class_confidence_filter import (
        v1,
        v1_tensor,
    )

    value = response()
    sources = []
    for row in value["predictions"]:
        row["class_id"] = 0
        row["confidence"] = 0.7
        row["parent_id"] = "image"
        source = native_obb_from_inference({**value, "predictions": [row]})
        sources.append(native_obb_to_tensor(source) if tensor else source)
    if fusion == "combine":
        module = combine_tensor if tensor else combine
        merged = module.DetectionsCombineBlockV1().run(*sources)["predictions"]
    else:
        merged = DetectionsConsensusBlockV2().run(
            predictions_batches=[[source] for source in sources],
            required_votes=1,
            class_aware=True,
            iou_threshold=0.5,
            confidence=0,
            classes_to_consider=None,
            required_objects=None,
            presence_confidence_aggregation=AggregationMode.MAX,
            detections_merge_confidence_aggregation=AggregationMode.AVERAGE,
            detections_merge_coordinates_aggregation=AggregationMode.AVERAGE,
            detections_merge_mask_aggregation=MaskAggregationMode.UNION,
        )[0]["predictions"]
    from roboflow_workflows.core_steps.common.query_language.entities.operations import (
        DetectionsProperty,
    )
    from roboflow_workflows.core_steps.common.query_language.operations.detections import (
        base,
    )

    monkeypatch.setattr(base, "ENABLE_TENSOR_DATA_REPRESENTATION", tensor)
    assert base.extract_detections_property(
        merged, DetectionsProperty.CLASS_NAME, "test"
    ) == ["plane", "stone"]
    if consumer == "per_class":
        module = v1_tensor if tensor else v1
        result = module.PerClassConfidenceFilterBlockV1().run(
            predictions=[merged],
            class_thresholds={"plane": 0.9, "stone": 0.1},
        )[0]["predictions"]
        expected = "stone"
    else:
        from roboflow_workflows.core_steps.common.query_language.operations.core import (
            execute_operations,
        )
        from roboflow_workflows.core_steps.common.query_language.operations.detection import (
            base as single_detection,
        )

        monkeypatch.setattr(
            single_detection, "ENABLE_TENSOR_DATA_REPRESENTATION", tensor
        )
        operation = (
            {"type": "PickDetectionsByParentClass", "parent_class": "plane"}
            if consumer == "parent_class"
            else {
                "type": "DetectionsFilter",
                "filter_operation": {
                    "type": "StatementGroup",
                    "operator": "and",
                    "statements": [
                        {
                            "type": "BinaryStatement",
                            "negate": False,
                            "left_operand": {
                                "type": "DynamicOperand",
                                "operand_name": "_",
                                "operations": [
                                    {
                                        "type": "ExtractDetectionProperty",
                                        "property_name": "class_name",
                                    }
                                ],
                            },
                            "comparator": {"type": "=="},
                            "right_operand": {
                                "type": "StaticOperand",
                                "value": "plane",
                            },
                        }
                    ],
                },
            }
        )
        result = execute_operations(merged, operations=[operation])
        expected = "plane"
    rows = serialise_native_obb(result)["predictions"]
    assert [(row["class"], row["class_id"]) for row in rows] == [(expected, 0)]


@pytest.mark.parametrize("tensor", [False, True])
def test_native_empty_visualization_block_returns_unchanged_image(tensor):
    from roboflow_workflows.core_steps.models.roboflow.oriented_object_detection.v1 import (
        NativeOBBVisualizationBlockV1,
    )
    from roboflow_workflows.execution_engine.entities.base import (
        ImageParentMetadata,
        WorkflowImageData,
    )

    value = response()
    value["predictions"] = []
    predictions = native_obb_from_inference(value)
    if tensor:
        predictions = native_obb_to_tensor(predictions)
    pixels = np.full((79, 101, 3), 123, dtype=np.uint8)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="image"), numpy_image=pixels
    )
    result = NativeOBBVisualizationBlockV1().run(image, predictions, 2)
    np.testing.assert_array_equal(result["image"].numpy_image, pixels)


@pytest.mark.parametrize("empty", [False, True])
def test_compiled_native_perspective_roundtrip_preserves_ordered_subpixel_quad(empty):
    from roboflow_workflows.execution_engine.core import ExecutionEngine

    workflow = {
        "version": "1.0",
        "inputs": [
            {"type": "InferenceImage", "name": "image"},
            {
                "type": "WorkflowBatchInput",
                "name": "predictions",
                "kind": ["oriented_object_detection_prediction"],
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/perspective_correction@v1",
                "name": "warp",
                "images": "$inputs.image",
                "predictions": "$inputs.predictions",
                "perspective_polygons": [[0, 0], [90, 7], [100, 78], [2, 75]],
                "transformed_rect_width": 103,
                "transformed_rect_height": 83,
                "extend_perspective_polygon_by_detections_anchor": "CENTER",
                "warp_image": False,
            },
            {
                "type": "roboflow_core/detections_consensus@v2",
                "name": "consumer",
                "predictions_batches": ["$steps.warp.corrected_coordinates"],
                "required_votes": 1,
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "result",
                "selector": "$steps.consumer.predictions",
                "coordinates_system": "own",
            }
        ],
    }
    payload = response()
    if empty:
        payload["predictions"] = []
    engine = ExecutionEngine.init(workflow_definition=workflow)
    try:
        result = engine.run(
            runtime_parameters={
                "image": np.zeros((79, 101, 3), dtype=np.uint8),
                "predictions": [payload],
            },
            serialize_results=True,
        )[0]["result"]
    finally:
        engine.close()
    assert result["image"] == {"height": 83, "width": 103}
    assert result["class_heading"] == payload["class_heading"]
    if empty:
        assert result["predictions"] == []
        return

    row = result["predictions"][0]
    assert row["geometry_type"] == "quadrilateral"
    corners = np.asarray(row["corners"])
    assert np.isfinite(corners).all()
    assert np.any(np.abs(corners - np.rint(corners)) > 0.01)
    restored = serialise_native_obb(native_obb_from_inference(result))
    np.testing.assert_allclose(restored["predictions"][0]["corners"], corners)
    assert row["heading"] is True
    assert row["detection_id"] == "plane0"


@pytest.mark.parametrize("initially_empty", [False, True])
def test_empty_tensor_bridge_retains_root_frame_for_serialization(initially_empty):
    from roboflow_workflows.core_steps.common.oriented import _native_obb_to_sv

    value = response()
    if initially_empty:
        value["predictions"] = []
    source = native_obb_from_inference(value)
    source.metadata.update(
        parent_id="crop",
        root_parent_id="root",
        root_parent_coordinates=[13, 17],
        root_parent_dimensions=[200, 300],
    )
    bridged = _native_obb_to_sv(native_obb_to_tensor(source))
    restored = sv_detections_to_root_coordinates(bridged[:0])
    assert restored.metadata["parent_id"] == "root"
    assert serialise_native_obb(restored)["image"] == {"height": 200, "width": 300}
    assert serialise_native_obb(restored)["predictions"] == []


@pytest.mark.parametrize("tensor", [False, True])
@pytest.mark.parametrize("initially_empty", [False, True])
def test_parent_class_query_empty_result_remains_native(
    monkeypatch, tensor, initially_empty
):
    from roboflow_workflows.core_steps.common.query_language.operations.core import (
        execute_operations,
    )
    from roboflow_workflows.core_steps.common.query_language.operations.detections import (
        base,
    )
    from roboflow_workflows.core_steps.transformations.detections_combine import (
        v1,
        v1_tensor,
    )

    payload = response()
    if initially_empty:
        payload["predictions"] = []
    source = native_obb_from_inference(payload)
    if tensor:
        source = native_obb_to_tensor(source)
    monkeypatch.setattr(base, "ENABLE_TENSOR_DATA_REPRESENTATION", tensor)
    selected = execute_operations(
        source,
        operations=[{"type": "PickDetectionsByParentClass", "parent_class": "absent"}],
    )
    serialized = serialise_native_obb(selected)
    assert serialized == {**payload, "predictions": []}
    module = v1_tensor if tensor else v1
    combined = module.DetectionsCombineBlockV1().run(selected, selected)["predictions"]
    assert serialise_native_obb(combined) == serialized


@pytest.mark.parametrize("tensor", [False, True])
def test_native_identity_perspective_keeps_tight_bounds_and_bbox_filter(
    monkeypatch, tensor
):
    from roboflow_workflows.core_steps.common.oriented import _native_obb_to_sv
    from roboflow_workflows.core_steps.common.query_language.operations.core import (
        execute_operations,
    )
    from roboflow_workflows.core_steps.common.query_language.operations.detection import (
        base as single,
    )
    from roboflow_workflows.core_steps.common.query_language.operations.detections import (
        base,
    )
    from roboflow_workflows.core_steps.transformations.perspective_correction import (
        v1,
        v1_tensor,
    )
    from roboflow_workflows.execution_engine.entities.base import (
        ImageParentMetadata,
        WorkflowImageData,
    )

    payload = response()
    payload["predictions"] = payload["predictions"][:1]
    source = native_obb_from_inference(payload)
    if tensor:
        source = native_obb_to_tensor(source)
    before = serialise_native_obb(source)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata("image"),
        numpy_image=np.zeros((79, 101, 3), np.uint8),
    )
    module = v1_tensor if tensor else v1
    warped = module.PerspectiveCorrectionBlockV1().run(
        images=[image],
        predictions=[source],
        perspective_polygons=[[0, 0], [100, 0], [100, 78], [0, 78]],
        transformed_rect_width=101,
        transformed_rect_height=79,
        extend_perspective_polygon_by_detections_anchor="CENTER",
        warp_image=False,
    )[0]["corrected_coordinates"]
    converted = _native_obb_to_sv(warped)
    corners = converted.data["xyxyxyxy"]
    np.testing.assert_allclose(
        converted.xyxy,
        np.concatenate((corners.min(1), corners.max(1)), axis=1),
        atol=1e-6,
    )
    assert serialise_native_obb(source) == before
    if tensor:
        from roboflow_workflows.core_steps.common.tensor_native import read_host_mirror

        mirror = read_host_mirror(warped.bboxes_metadata, len(warped))
        assert mirror is not None
        np.testing.assert_allclose(mirror[0], converted.xyxy, atol=1e-6)
    monkeypatch.setattr(base, "ENABLE_TENSOR_DATA_REPRESENTATION", tensor)
    monkeypatch.setattr(single, "ENABLE_TENSOR_DATA_REPRESENTATION", tensor)
    operation = {
        "type": "DetectionsFilter",
        "filter_operation": {
            "type": "StatementGroup",
            "operator": "and",
            "statements": [
                {
                    "type": "BinaryStatement",
                    "negate": False,
                    "left_operand": {
                        "type": "DynamicOperand",
                        "operand_name": "_",
                        "operations": [
                            {
                                "type": "ExtractDetectionProperty",
                                "property_name": "x_min",
                            }
                        ],
                    },
                    "comparator": {"type": "(Number) >"},
                    "right_operand": {"type": "StaticOperand", "value": 31.05},
                }
            ],
        },
    }
    assert len(execute_operations(source, operations=[operation])) == 1
    assert len(execute_operations(warped, operations=[operation])) == 1


@pytest.mark.parametrize("initially_empty", [False, True])
def test_native_scale_updates_header_before_rows_can_disappear(initially_empty):
    payload = response()
    if initially_empty:
        payload["predictions"] = []
    source = native_obb_from_inference(payload)
    for key, value in [
        ("scaling_relative_to_parent", 1.25),
        ("scaling_relative_to_root_parent", 0.5),
    ]:
        source.metadata[key] = value
        source.data[key] = np.full(len(source), value)
    scaled = scale_sv_detections(source, 2.0, target_size_wh=(203, 159))
    assert scaled.metadata["image_dimensions"] == [159, 203]
    assert scaled.metadata["scaling_relative_to_parent"] == 2.5
    assert scaled.metadata["scaling_relative_to_root_parent"] == 1.0
    if len(scaled):
        np.testing.assert_array_equal(
            scaled.data["image_dimensions"], [[159, 203]] * len(scaled)
        )
        np.testing.assert_array_equal(
            scaled.data["scaling_relative_to_parent"], [2.5] * len(scaled)
        )
    assert serialise_native_obb(native_obb_to_tensor(scaled[:0]))["image"] == {
        "height": 159,
        "width": 203,
    }
    assert source.metadata["image_dimensions"] == [79, 101]
    assert source.metadata["scaling_relative_to_parent"] == 1.25


def test_native_carrier_header_wins_without_mutating_input(monkeypatch):
    from copy import deepcopy

    from roboflow_workflows import environment
    from roboflow_workflows.core_steps.common.oriented import deserialize_native_obb

    source = native_obb_from_inference(response())
    source.metadata.update(
        root_parent_id="root",
        root_parent_coordinates=[13, 17],
        root_parent_dimensions=[200, 300],
    )
    source.data["image_dimensions"][:] = [333, 444]
    source.data["root_parent_id"] = np.array(["stale"] * len(source))
    source.data["root_parent_coordinates"] = np.tile([100, 100], (len(source), 1))
    source.data["root_parent_dimensions"] = np.tile([333, 444], (len(source), 1))
    before = deepcopy(source)
    monkeypatch.setattr(environment, "ENABLE_TENSOR_DATA_REPRESENTATION", False)
    normalized = deserialize_native_obb("input", source)
    assert serialise_native_obb(normalized)["image"] == {"width": 101, "height": 79}
    tensor = native_obb_to_tensor(normalized)
    assert tensor.image_metadata["image_dimensions"] == [79, 101]
    assert tensor.image_metadata["root_parent_dimensions"] == [200, 300]
    rooted = sv_detections_to_root_coordinates(normalized)
    bridged_empty = take_detections_by_indices(native_obb_to_tensor(rooted), [])
    assert serialise_native_obb(bridged_empty)["image"] == {"width": 300, "height": 200}
    assert rooted.metadata["root_parent_coordinates"] == [0, 0]
    assert rooted.metadata["parent_id"] == "root"
    assert source.metadata == before.metadata
    for key in source.data:
        np.testing.assert_array_equal(source.data[key], before.data[key])
