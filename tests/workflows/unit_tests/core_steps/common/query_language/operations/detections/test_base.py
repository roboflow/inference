import numpy as np
import pytest
import supervision as sv
import torch

from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION
from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
    OperationError,
)
from inference.core.workflows.core_steps.common.query_language.operations.detections.base import (
    rename_detections,
)
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAME_KEY,
    CLASS_NAMES_KEY,
)
from inference_models.models.base.instance_segmentation import (
    InstanceDetections as NativeInstanceDetections,
)
from inference_models.models.base.keypoints_detection import (
    KeyPoints as NativeKeyPoints,
)
from inference_models.models.base.object_detection import Detections as NativeDetections

# Under ENABLE_TENSOR_DATA_REPRESENTATION `rename_detections` is native-only: it rejects
# sv.Detections. The sv-based tests below are skipped when the flag is on; each has a
# `*_tensor_native` parity test (skipped when the flag is off) exercising the same
# scenario with an `inference_models.Detections` input. Native renaming rewrites the
# `class_id` tensor and the `image_metadata[CLASS_NAMES_KEY]` class_id -> name map
# (there is no per-box `data["class_name"]`); the parity tests therefore resolve each
# box's name through that map, the same way consumers do.
_NUMPY_ONLY = pytest.mark.skipif(
    ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="sv.Detections input; rename_detections is native-only under "
    "ENABLE_TENSOR_DATA_REPRESENTATION — see the *_tensor_native parity test",
)
_TENSOR_ONLY = pytest.mark.skipif(
    not ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="tensor-native variant; runs only with ENABLE_TENSOR_DATA_REPRESENTATION=True",
)


def _native_detections() -> NativeDetections:
    return NativeDetections(
        xyxy=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.float32),
        class_id=torch.tensor([10, 11], dtype=torch.long),
        confidence=torch.tensor([0.3, 0.4], dtype=torch.float32),
        image_metadata={CLASS_NAMES_KEY: {10: "a", 11: "b"}},
    )


def _resolved_class_names(detections: NativeDetections) -> list:
    class_names = detections.image_metadata[CLASS_NAMES_KEY]
    return [class_names[int(class_id)] for class_id in detections.class_id.tolist()]


def test_rename_detections_when_not_sv_detections_provided() -> None:
    # when
    with pytest.raises(InvalidInputTypeError):
        _ = rename_detections(
            detections="invalid",
            class_map={"a": "b"},
            strict=True,
            new_classes_id_offset=0,
            global_parameters={},
        )


@_NUMPY_ONLY
def test_rename_detections_when_strict_mode_enabled_and_all_classes_present() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[0, 1, 2, 3], [0, 1, 2, 3]]),
        class_id=np.array([10, 11]),
        confidence=np.array([0.3, 0.4]),
        data={"class_name": np.array(["a", "b"])},
    )

    # when
    result = rename_detections(
        detections=detections,
        class_map={"a": "A", "b": "B"},
        strict=True,
        new_classes_id_offset=1024,
        global_parameters={},
    )

    # then
    assert np.allclose(
        result.xyxy, np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    ), "Expected no to change"
    assert np.allclose(result.confidence, np.array([0.3, 0.4])), "Expected no to change"
    assert np.allclose(
        result.class_id, np.array([0, 1])
    ), "Expected to change with mapping"
    assert result.data["class_name"].tolist() == [
        "A",
        "B",
    ], "Expected to change with mapping"


@_NUMPY_ONLY
def test_rename_detections_when_strict_mode_enabled_and_not_all_classes_present() -> (
    None
):
    # given
    detections = sv.Detections(
        xyxy=np.array([[0, 1, 2, 3], [0, 1, 2, 3]]),
        class_id=np.array([10, 11]),
        confidence=np.array([0.3, 0.4]),
        data={"class_name": np.array(["a", "b"])},
    )

    # when
    with pytest.raises(OperationError):
        _ = rename_detections(
            detections=detections,
            class_map={"a": "A"},
            strict=True,
            new_classes_id_offset=1024,
            global_parameters={},
        )


@_NUMPY_ONLY
def test_rename_detections_when_non_strict_mode_enabled_and_all_classes_present() -> (
    None
):
    # given
    detections = sv.Detections(
        xyxy=np.array([[0, 1, 2, 3], [0, 1, 2, 3]]),
        class_id=np.array([10, 11]),
        confidence=np.array([0.3, 0.4]),
        data={"class_name": np.array(["a", "b"])},
    )

    # when
    result = rename_detections(
        detections=detections,
        class_map={"a": "A", "b": "B"},
        strict=False,
        new_classes_id_offset=1024,
        global_parameters={},
    )

    # then
    assert np.allclose(
        result.xyxy, np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    ), "Expected no to change"
    assert np.allclose(result.confidence, np.array([0.3, 0.4])), "Expected no to change"
    assert np.allclose(
        result.class_id, np.array([1024, 1025])
    ), "Expected to change with mapping"
    assert result.data["class_name"].tolist() == [
        "A",
        "B",
    ], "Expected to change with mapping"


@_NUMPY_ONLY
def test_rename_detections_when_non_strict_mode_enabled_and_not_all_classes_present() -> (
    None
):
    # given
    detections = sv.Detections(
        xyxy=np.array([[0, 1, 2, 3], [0, 1, 2, 3]]),
        class_id=np.array([10, 11]),
        confidence=np.array([0.3, 0.4]),
        data={"class_name": np.array(["a", "b"])},
    )

    # when
    result = rename_detections(
        detections=detections,
        class_map={"a": "A"},
        strict=False,
        new_classes_id_offset=1024,
        global_parameters={},
    )

    # then
    assert np.allclose(
        result.xyxy, np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    ), "Expected no to change"
    assert np.allclose(result.confidence, np.array([0.3, 0.4])), "Expected no to change"
    assert np.allclose(
        result.class_id, np.array([1024, 11])
    ), "Expected to change with mapping"
    assert result.data["class_name"].tolist() == [
        "A",
        "b",
    ], "Expected to change with mapping"


@_NUMPY_ONLY
def test_rename_detections_when_detections_have_no_class_name_data_and_not_strict() -> (
    None
):
    # given
    detections = sv.Detections(
        xyxy=np.array([[0, 1, 2, 3], [0, 1, 2, 3]]),
        class_id=np.array([10, 11]),
        confidence=np.array([0.3, 0.4]),
    )

    # when
    result = rename_detections(
        detections=detections,
        class_map={"a": "A"},
        strict=False,
        new_classes_id_offset=1024,
        global_parameters={},
    )

    # then - with no class names to rename, non-strict mode is a no-op: the
    # detections pass through length-consistent and unchanged, rather than having
    # class_id / class_name overwritten with zero-length arrays for the 2 real boxes.
    assert isinstance(result, sv.Detections)
    assert len(result) == 2, "Expected the 2 input boxes to be preserved"
    assert np.allclose(result.xyxy, np.array([[0, 1, 2, 3], [0, 1, 2, 3]]))
    assert np.allclose(
        result.class_id, np.array([10, 11])
    ), "Expected class_id unchanged in the no-op path"
    assert np.allclose(result.confidence, np.array([0.3, 0.4]))
    assert (
        "class_name" not in result.data
    ), "Expected no fabricated class_name field on the no-op path"


@_NUMPY_ONLY
def test_rename_detections_when_detections_are_empty_is_a_noop() -> None:
    # given
    detections = sv.Detections.empty()

    # when - empty detections have nothing to rename; both modes return a
    # consistent (empty) result without raising.
    non_strict = rename_detections(
        detections=detections,
        class_map={"a": "A"},
        strict=False,
        new_classes_id_offset=1024,
        global_parameters={},
    )
    strict = rename_detections(
        detections=detections,
        class_map={"a": "A"},
        strict=True,
        new_classes_id_offset=1024,
        global_parameters={},
    )

    # then
    assert isinstance(non_strict, sv.Detections) and len(non_strict) == 0
    assert isinstance(strict, sv.Detections) and len(strict) == 0


@_NUMPY_ONLY
def test_rename_detections_when_mapping_is_parametrised() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[0, 1, 2, 3], [0, 1, 2, 3]]),
        class_id=np.array([10, 11]),
        confidence=np.array([0.3, 0.4]),
        data={"class_name": np.array(["a", "b"])},
    )

    # when
    result = rename_detections(
        detections=detections,
        class_map="class_map_param",
        strict="strict_param",
        new_classes_id_offset=1024,
        global_parameters={
            "strict_param": True,
            "class_map_param": {"a": "A", "b": "B"},
        },
    )

    # then
    assert np.allclose(
        result.xyxy, np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    ), "Expected no to change"
    assert np.allclose(result.confidence, np.array([0.3, 0.4])), "Expected no to change"
    assert np.allclose(
        result.class_id, np.array([0, 1])
    ), "Expected to change with mapping"
    assert result.data["class_name"].tolist() == [
        "A",
        "B",
    ], "Expected to change with mapping"


@_TENSOR_ONLY
def test_rename_detections_when_strict_mode_enabled_and_all_classes_present_tensor_native() -> (
    None
):
    # given
    detections = _native_detections()

    # when
    result = rename_detections(
        detections=detections,
        class_map={"a": "A", "b": "B"},
        strict=True,
        new_classes_id_offset=1024,
        global_parameters={},
    )

    # then
    assert result.xyxy.tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]], "Expected no to change"
    assert torch.allclose(
        result.confidence, torch.tensor([0.3, 0.4])
    ), "Expected no to change"
    assert result.class_id.tolist() == [
        0,
        1,
    ], "Expected to change with mapping"
    assert _resolved_class_names(result) == [
        "A",
        "B",
    ], "Expected to change with mapping"


@_TENSOR_ONLY
def test_rename_detections_when_strict_mode_enabled_and_not_all_classes_present_tensor_native() -> (
    None
):
    # given
    detections = _native_detections()

    # when
    with pytest.raises(OperationError):
        _ = rename_detections(
            detections=detections,
            class_map={"a": "A"},
            strict=True,
            new_classes_id_offset=1024,
            global_parameters={},
        )


@_TENSOR_ONLY
def test_rename_detections_when_non_strict_mode_enabled_and_all_classes_present_tensor_native() -> (
    None
):
    # given
    detections = _native_detections()

    # when
    result = rename_detections(
        detections=detections,
        class_map={"a": "A", "b": "B"},
        strict=False,
        new_classes_id_offset=1024,
        global_parameters={},
    )

    # then
    assert result.xyxy.tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]], "Expected no to change"
    assert torch.allclose(
        result.confidence, torch.tensor([0.3, 0.4])
    ), "Expected no to change"
    assert result.class_id.tolist() == [
        1024,
        1025,
    ], "Expected to change with mapping"
    assert _resolved_class_names(result) == [
        "A",
        "B",
    ], "Expected to change with mapping"


@_TENSOR_ONLY
def test_rename_detections_when_non_strict_mode_enabled_and_not_all_classes_present_tensor_native() -> (
    None
):
    # given
    detections = _native_detections()

    # when
    result = rename_detections(
        detections=detections,
        class_map={"a": "A"},
        strict=False,
        new_classes_id_offset=1024,
        global_parameters={},
    )

    # then
    assert result.xyxy.tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]], "Expected no to change"
    assert torch.allclose(
        result.confidence, torch.tensor([0.3, 0.4])
    ), "Expected no to change"
    assert result.class_id.tolist() == [
        1024,
        11,
    ], "Expected to change with mapping"
    assert _resolved_class_names(result) == [
        "A",
        "b",
    ], "Expected to change with mapping"


@_TENSOR_ONLY
def test_rename_detections_when_detections_have_no_class_name_data_and_not_strict_tensor_native() -> (
    None
):
    # given - the native analog of sv detections without `data["class_name"]` is a
    # missing `image_metadata[CLASS_NAMES_KEY]` map
    detections = NativeDetections(
        xyxy=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.float32),
        class_id=torch.tensor([10, 11], dtype=torch.long),
        confidence=torch.tensor([0.3, 0.4], dtype=torch.float32),
        image_metadata={},
    )

    # when
    result = rename_detections(
        detections=detections,
        class_map={"a": "A"},
        strict=False,
        new_classes_id_offset=1024,
        global_parameters={},
    )

    # then - with no class names to rename, non-strict mode is a no-op copy: the
    # boxes pass through unchanged and no class_names map is fabricated
    assert isinstance(result, NativeDetections)
    assert result is not detections, "Expected a copy, not the input object"
    assert result.xyxy.tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert torch.allclose(result.confidence, torch.tensor([0.3, 0.4]))
    assert result.class_id.tolist() == [
        10,
        11,
    ], "Expected class_id unchanged in the no-op path"
    assert (result.image_metadata or {}).get(
        CLASS_NAMES_KEY
    ) is None, "Expected no fabricated class_names map on the no-op path"


@_TENSOR_ONLY
def test_rename_detections_when_detections_have_no_class_name_data_and_strict_tensor_native() -> (
    None
):
    # given
    detections = NativeDetections(
        xyxy=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.float32),
        class_id=torch.tensor([10, 11], dtype=torch.long),
        confidence=torch.tensor([0.3, 0.4], dtype=torch.float32),
        image_metadata={},
    )

    # when - strict mode cannot guarantee class_map coverage without class names,
    # so the shared helper raises (same OperationError as the numpy arm)
    with pytest.raises(OperationError):
        _ = rename_detections(
            detections=detections,
            class_map={"a": "A"},
            strict=True,
            new_classes_id_offset=1024,
            global_parameters={},
        )


@_TENSOR_ONLY
def test_rename_detections_when_detections_are_empty_is_a_noop_tensor_native() -> None:
    # given - empty native detections without a class_names map (the mirror of
    # sv.Detections.empty(), which carries no `class_name` data)
    def _empty_native_detections() -> NativeDetections:
        return NativeDetections(
            xyxy=torch.zeros((0, 4), dtype=torch.float32),
            class_id=torch.zeros((0,), dtype=torch.long),
            confidence=torch.zeros((0,), dtype=torch.float32),
            image_metadata={},
        )

    # when - empty detections have nothing to rename; both modes return a
    # consistent (empty) result without raising
    non_strict = rename_detections(
        detections=_empty_native_detections(),
        class_map={"a": "A"},
        strict=False,
        new_classes_id_offset=1024,
        global_parameters={},
    )
    strict = rename_detections(
        detections=_empty_native_detections(),
        class_map={"a": "A"},
        strict=True,
        new_classes_id_offset=1024,
        global_parameters={},
    )

    # then
    assert isinstance(non_strict, NativeDetections)
    assert int(non_strict.xyxy.shape[0]) == 0
    assert isinstance(strict, NativeDetections)
    assert int(strict.xyxy.shape[0]) == 0


def _native_detections_with_overrides() -> NativeDetections:
    # classes_replacement-style input: both rows share class_id 7 whose map name is
    # "cat"; row 1 carries a per-box CLASS_NAME_KEY override ("dog") — the label the
    # tensor serializer would prefer (C1), hence the label rename must operate on.
    return NativeDetections(
        xyxy=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.float32),
        class_id=torch.tensor([7, 7], dtype=torch.long),
        confidence=torch.tensor([0.3, 0.4], dtype=torch.float32),
        image_metadata={CLASS_NAMES_KEY: {7: "cat"}},
        bboxes_metadata=[
            {"detection_id": "d0"},
            {"detection_id": "d1", CLASS_NAME_KEY: "dog"},
        ],
    )


@_TENSOR_ONLY
def test_rename_detections_rewrites_per_box_class_overrides_tensor_native() -> None:
    # given
    detections = _native_detections_with_overrides()

    # when - only the override name is in the class_map; numpy on the equivalent
    # per-row class_names ["cat", "dog"] yields names ["cat", "canine"], ids [7, 1024]
    result = rename_detections(
        detections=detections,
        class_map={"dog": "canine"},
        strict=False,
        new_classes_id_offset=1024,
        global_parameters={},
    )

    # then
    assert result.class_id.tolist() == [7, 1024]
    assert (
        result.bboxes_metadata[1][CLASS_NAME_KEY] == "canine"
    ), "Expected the per-box override rewritten to the renamed label"
    assert (
        CLASS_NAME_KEY not in result.bboxes_metadata[0]
    ), "Expected no override fabricated on a row that had none"
    assert result.image_metadata[CLASS_NAMES_KEY][7] == "cat"
    assert (
        detections.bboxes_metadata[1][CLASS_NAME_KEY] == "dog"
    ), "Expected the source object untouched"


@_TENSOR_ONLY
def test_rename_detections_strict_mode_covers_per_box_overrides_tensor_native() -> None:
    # given
    detections = _native_detections_with_overrides()

    # when - class_map covers the map name but not the override; numpy raises
    # "Class 'dog' not found in class_map." for the equivalent per-row names
    with pytest.raises(OperationError):
        _ = rename_detections(
            detections=detections,
            class_map={"cat": "feline"},
            strict=True,
            new_classes_id_offset=1024,
            global_parameters={},
        )


@_TENSOR_ONLY
def test_rename_detections_splits_shared_class_id_on_divergent_overrides_tensor_native() -> (
    None
):
    # given
    detections = _native_detections_with_overrides()

    # when - both effective names renamed to NEW targets; numpy assigns offset ids
    # to the sorted new targets (canine=1024, feline=1025), splitting the shared id
    result = rename_detections(
        detections=detections,
        class_map={"cat": "feline", "dog": "canine"},
        strict=False,
        new_classes_id_offset=1024,
        global_parameters={},
    )

    # then
    assert result.class_id.tolist() == [1025, 1024]
    assert (
        result.image_metadata[CLASS_NAMES_KEY][1025] == "feline"
    ), "Expected the override-less row to resolve via the class_names map"
    assert result.bboxes_metadata[1][CLASS_NAME_KEY] == "canine"


@_TENSOR_ONLY
def test_rename_detections_keeps_uncovered_override_in_non_strict_mode_tensor_native() -> (
    None
):
    # given
    detections = _native_detections_with_overrides()

    # when - the override name is not in the class_map; numpy keeps the name and
    # its original id on the equivalent per-row input
    result = rename_detections(
        detections=detections,
        class_map={"cat": "feline"},
        strict=False,
        new_classes_id_offset=1024,
        global_parameters={},
    )

    # then
    assert result.class_id.tolist() == [1024, 7]
    assert (
        result.bboxes_metadata[1][CLASS_NAME_KEY] == "dog"
    ), "Expected the uncovered override kept intact"
    assert result.image_metadata[CLASS_NAMES_KEY][1024] == "feline"


@_TENSOR_ONLY
def test_rename_detections_when_mapping_is_parametrised_tensor_native() -> None:
    # given
    detections = _native_detections()

    # when
    result = rename_detections(
        detections=detections,
        class_map="class_map_param",
        strict="strict_param",
        new_classes_id_offset=1024,
        global_parameters={
            "strict_param": True,
            "class_map_param": {"a": "A", "b": "B"},
        },
    )

    # then
    assert result.xyxy.tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]], "Expected no to change"
    assert torch.allclose(
        result.confidence, torch.tensor([0.3, 0.4])
    ), "Expected no to change"
    assert result.class_id.tolist() == [
        0,
        1,
    ], "Expected to change with mapping"
    assert _resolved_class_names(result) == [
        "A",
        "B",
    ], "Expected to change with mapping"


# ---------------------------------------------------------------------------
# Parity for the other native detection representations: InstanceDetections
# (rename carries the per-instance mask through unchanged) and the KeyPoints
# prediction (a (KeyPoints, Detections) tuple; rename touches only the bbox
# component's class ids / names, the KeyPoints component is preserved as-is).
# The class-id / class-name rewrite logic itself is identical across reps and
# is already covered by the plain-`Detections` cases above, so these focus on
# the representation-specific carry-through.
# ---------------------------------------------------------------------------


def _native_instance_detections(mask: torch.Tensor) -> NativeInstanceDetections:
    return NativeInstanceDetections(
        xyxy=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.float32),
        class_id=torch.tensor([10, 11], dtype=torch.long),
        confidence=torch.tensor([0.3, 0.4], dtype=torch.float32),
        mask=mask,
        image_metadata={CLASS_NAMES_KEY: {10: "a", 11: "b"}},
    )


@_TENSOR_ONLY
def test_rename_detections_for_instance_segmentation_tensor_native() -> None:
    # given - distinct per-instance pixel counts (3 / 5) so we can assert the masks
    # ride through unchanged and in order.
    mask = torch.zeros((2, 20, 20), dtype=torch.bool)
    mask[0, 0, 0:3] = True  # 3 px
    mask[1, 0, 0:5] = True  # 5 px
    detections = _native_instance_detections(mask)

    # when
    result = rename_detections(
        detections=detections,
        class_map={"a": "A", "b": "B"},
        strict=True,
        new_classes_id_offset=1024,
        global_parameters={},
    )

    # then - same repr, class ids / names rewritten, mask carried through unchanged
    assert isinstance(result, NativeInstanceDetections)
    assert result.xyxy.tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]], "Expected no to change"
    assert result.class_id.tolist() == [0, 1], "Expected to change with mapping"
    assert _resolved_class_names(result) == [
        "A",
        "B",
    ], "Expected to change with mapping"
    assert result.mask.sum(dim=(1, 2)).cpu().numpy().tolist() == [
        3,
        5,
    ], "Expected mask to be carried through unchanged and in order"
    assert torch.equal(result.mask, mask)


@_TENSOR_ONLY
def test_rename_detections_for_instance_segmentation_when_empty_tensor_native() -> None:
    # given
    detections = NativeInstanceDetections(
        xyxy=torch.zeros((0, 4), dtype=torch.float32),
        class_id=torch.zeros((0,), dtype=torch.long),
        confidence=torch.zeros((0,), dtype=torch.float32),
        mask=torch.zeros((0, 20, 20), dtype=torch.bool),
        image_metadata={CLASS_NAMES_KEY: {}},
    )

    # when
    result = rename_detections(
        detections=detections,
        class_map={"a": "A", "b": "B"},
        strict=True,
        new_classes_id_offset=1024,
        global_parameters={},
    )

    # then - empty object of the SAME repr
    assert isinstance(result, NativeInstanceDetections)
    assert result.xyxy.shape[0] == 0
    assert result.mask.shape[0] == 0


@_TENSOR_ONLY
def test_rename_detections_for_keypoints_tensor_native() -> None:
    # given - a keypoint prediction is a (KeyPoints, Detections) tuple. Rename rewrites
    # the bbox component's class ids / names; the KeyPoints component is preserved.
    key_points = NativeKeyPoints(
        xy=torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        ),  # (instances, kpts, 2)
        class_id=torch.tensor([10, 11], dtype=torch.long),
        confidence=torch.tensor([[0.9, 0.8], [0.7, 0.6]]),  # (instances, kpts)
        image_metadata={CLASS_NAMES_KEY: {10: "a", 11: "b"}},
    )
    bboxes = NativeDetections(
        xyxy=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.float32),
        class_id=torch.tensor([10, 11], dtype=torch.long),
        confidence=torch.tensor([0.3, 0.4], dtype=torch.float32),
        image_metadata={CLASS_NAMES_KEY: {10: "a", 11: "b"}},
    )
    prediction = (key_points, bboxes)

    # when
    result = rename_detections(
        detections=prediction,
        class_map={"a": "A", "b": "B"},
        strict=True,
        new_classes_id_offset=1024,
        global_parameters={},
    )

    # then - a (KeyPoints, Detections) tuple; bbox renamed, keypoints carried through
    assert isinstance(result, tuple) and len(result) == 2
    result_key_points, result_bboxes = result
    assert isinstance(result_key_points, NativeKeyPoints)
    assert isinstance(result_bboxes, NativeDetections)
    assert result_bboxes.class_id.tolist() == [0, 1], "Expected to change with mapping"
    assert _resolved_class_names(result_bboxes) == [
        "A",
        "B",
    ], "Expected to change with mapping"
    # the KeyPoints component is unchanged
    assert torch.equal(result_key_points.xy, key_points.xy)
    assert torch.equal(result_key_points.class_id, key_points.class_id)
