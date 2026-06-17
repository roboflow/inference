import numpy as np
import pytest
import supervision as sv
import torch

from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION
from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
    OperationError,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    execute_operations,
)
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAMES_KEY,
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    POLYGON_KEY,
)
from inference_models.models.base.instance_segmentation import (
    InstanceDetections as NativeInstanceDetections,
)
from inference_models.models.base.keypoints_detection import (
    KeyPoints as NativeKeyPoints,
)
from inference_models.models.base.object_detection import Detections as NativeDetections

# Under ENABLE_TENSOR_DATA_REPRESENTATION the UQL detections ops are native-only:
# they reject sv.Detections / dict inputs. The sv-based tests below are skipped when
# the flag is on; each has a `*_tensor_native` parity test (skipped when the flag is
# off) that exercises the same scenario with an `inference_models.Detections` input.
_NUMPY_ONLY = pytest.mark.skipif(
    ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="sv.Detections input; UQL detections ops are native-only under "
    "ENABLE_TENSOR_DATA_REPRESENTATION — see the *_tensor_native parity test",
)
_TENSOR_ONLY = pytest.mark.skipif(
    not ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="tensor-native variant; runs only with ENABLE_TENSOR_DATA_REPRESENTATION=True",
)


def test_detections_to_dictionary_when_invalid_input_is_provided() -> None:
    # given
    operations = [{"type": "DetectionsToDictionary"}]
    # when
    with pytest.raises(InvalidInputTypeError):
        _ = execute_operations(value="invalid", operations=operations)


@_NUMPY_ONLY
def test_detections_to_dictionary_when_valid_input_is_provided() -> None:
    # given
    operations = [{"type": "DetectionsToDictionary"}]
    detections = sv.Detections(
        xyxy=np.array(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
            ]
        ),
        class_id=np.array([0, 1]),
        confidence=np.array([0.3, 0.4]),
        data={
            "class_name": np.array(["cat", "dog"]),
            "detection_id": np.array(["one", "two"]),
            "image_dimensions": np.array([[192, 168], [192, 168]]),
        },
    )

    # when
    result = execute_operations(value=detections, operations=operations)

    # then
    assert result == {
        "image": {"width": 168, "height": 192},
        "predictions": [
            {
                "width": 2.0,
                "height": 2.0,
                "x": 1.0,
                "y": 2.0,
                "confidence": 0.3,
                "class_id": 0,
                "class": "cat",
                "detection_id": "one",
            },
            {
                "width": 2.0,
                "height": 2.0,
                "x": 5.0,
                "y": 6.0,
                "confidence": 0.4,
                "class_id": 1,
                "class": "dog",
                "detection_id": "two",
            },
        ],
    }


@_NUMPY_ONLY
def test_detections_to_dictionary_when_malformed_input_is_provided() -> None:
    # given
    operations = [{"type": "DetectionsToDictionary"}]
    detections = sv.Detections(
        xyxy=np.array(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
            ]
        ),
        class_id=np.array([0, 1]),
        confidence=np.array([0.3, 0.4]),
    )

    # when
    with pytest.raises(OperationError):
        _ = execute_operations(value=detections, operations=operations)


def test_picking_detections_by_parent_class_when_invalid_input_provided() -> None:
    # given
    operations = [{"type": "PickDetectionsByParentClass", "parent_class": "a"}]

    # when
    with pytest.raises(InvalidInputTypeError):
        _ = execute_operations(value="FOR SURE NOT DETECTIONS", operations=operations)


@_NUMPY_ONLY
def test_picking_detections_by_parent_class_when_empty_detections_provided() -> None:
    # given
    operations = [{"type": "PickDetectionsByParentClass", "parent_class": "a"}]
    detections = sv.Detections.empty()

    # when
    result = execute_operations(value=detections, operations=operations)

    # then
    assert result is not detections
    assert len(result) == 0


@_NUMPY_ONLY
def test_picking_detections_by_parent_class_when_class_name_field_not_defined() -> None:
    # given
    operations = [{"type": "PickDetectionsByParentClass", "parent_class": "a"}]
    detections = sv.Detections(
        xyxy=np.array(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
            ]
        ),
        class_id=np.array([0, 1]),
        confidence=np.array([0.3, 0.4]),
    )

    # when
    result = execute_operations(value=detections, operations=operations)

    # then
    assert len(result) == 0


@_NUMPY_ONLY
def test_picking_detections_by_parent_class_when_parent_class_not_fond() -> None:
    # given
    operations = [{"type": "PickDetectionsByParentClass", "parent_class": "a"}]
    detections = sv.Detections(
        xyxy=np.array(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
            ]
        ),
        class_id=np.array([0, 1]),
        confidence=np.array([0.3, 0.4]),
        data={"class_name": np.array(["c", "d"])},
    )

    # when
    result = execute_operations(value=detections, operations=operations)

    # then
    assert len(result) == 0


@_NUMPY_ONLY
def test_picking_detections_by_parent_class_when_no_child_detections_matching() -> None:
    # given
    operations = [{"type": "PickDetectionsByParentClass", "parent_class": "a"}]
    detections = sv.Detections(
        xyxy=np.array(
            [
                [0, 0, 10, 10],
                [20, 20, 30, 30],
                [40, 40, 50, 50],
            ]
        ),
        class_id=np.array([0, 0, 1]),
        confidence=np.array([0.3, 0.4, 0.5]),
        data={"class_name": np.array(["a", "a", "b"])},
    )

    # when
    result = execute_operations(value=detections, operations=operations)

    # then
    assert len(result) == 2
    assert np.allclose(result.xyxy, np.array([[0, 0, 10, 10], [20, 20, 30, 30]]))
    assert np.allclose(result.confidence, [0.3, 0.4])


@_NUMPY_ONLY
def test_picking_detections_by_parent_class_when_there_are_child_detections_matching() -> (
    None
):
    # given
    operations = [{"type": "PickDetectionsByParentClass", "parent_class": "a"}]
    detections = sv.Detections(
        xyxy=np.array(
            [
                [0, 0, 50, 50],
                [20, 20, 30, 30],
                [40, 40, 50, 50],
            ]
        ),
        class_id=np.array([0, 1, 1]),
        confidence=np.array([0.3, 0.4, 0.5]),
        data={"class_name": np.array(["a", "b", "b"])},
    )

    # when
    result = execute_operations(value=detections, operations=operations)

    # then
    assert len(result) == 3
    assert np.allclose(
        result.xyxy, np.array([[0, 0, 50, 50], [20, 20, 30, 30], [40, 40, 50, 50]])
    )
    assert np.allclose(result.confidence, [0.3, 0.4, 0.5])


@_NUMPY_ONLY
def test_picking_detections_by_parent_class_when_there_are_child_detections_matching_different_parents() -> (
    None
):
    # given
    operations = [{"type": "PickDetectionsByParentClass", "parent_class": "a"}]
    detections = sv.Detections(
        xyxy=np.array(
            [
                [0, 0, 50, 50],
                [20, 20, 30, 30],
                [40, 40, 50, 50],
                [100, 100, 200, 200],
                [150, 100, 250, 200],
                [400, 400, 600, 600],
            ]
        ),
        class_id=np.array([0, 1, 1, 0, 2, 3]),
        confidence=np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.9]),
        data={"class_name": np.array(["a", "b", "b", "a", "c", "d"])},
    )

    # when
    result = execute_operations(value=detections, operations=operations)

    # then
    assert len(result) == 5
    assert np.allclose(
        result.xyxy,
        np.array(
            [
                [0, 0, 50, 50],
                [100, 100, 200, 200],
                [20, 20, 30, 30],
                [40, 40, 50, 50],
                [150, 100, 250, 200],
            ]
        ),
    )
    assert np.allclose(result.confidence, [0.3, 0.6, 0.4, 0.5, 0.7])


# ---------------------------------------------------------------------------
# Tensor-native parity variants (run only under ENABLE_TENSOR_DATA_REPRESENTATION).
# Same scenarios as the sv.Detections tests above, but with native
# `inference_models.Detections` inputs. Class names live in
# image_metadata[CLASS_NAMES_KEY]; per-box detection_id in bboxes_metadata.
# ---------------------------------------------------------------------------


@_TENSOR_ONLY
def test_detections_to_dictionary_when_valid_input_is_provided_tensor_native() -> None:
    # given
    operations = [{"type": "DetectionsToDictionary"}]
    detections = NativeDetections(
        xyxy=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.float32),
        class_id=torch.tensor([0, 1], dtype=torch.long),
        confidence=torch.tensor([0.3, 0.4], dtype=torch.float32),
        image_metadata={
            CLASS_NAMES_KEY: {0: "cat", 1: "dog"},
            IMAGE_DIMENSIONS_KEY: [192, 168],
        },
        bboxes_metadata=[{DETECTION_ID_KEY: "one"}, {DETECTION_ID_KEY: "two"}],
    )

    # when
    result = execute_operations(value=detections, operations=operations)

    # then
    assert result["image"] == {"width": 168, "height": 192}
    first, second = result["predictions"]
    assert (first["x"], first["y"], first["width"], first["height"]) == (
        1.0,
        2.0,
        2.0,
        2.0,
    )
    assert first["class_id"] == 0 and first["class"] == "cat"
    assert first["detection_id"] == "one"
    assert first["confidence"] == pytest.approx(0.3, abs=1e-6)
    assert (second["x"], second["y"], second["width"], second["height"]) == (
        5.0,
        6.0,
        2.0,
        2.0,
    )
    assert second["class_id"] == 1 and second["class"] == "dog"
    assert second["detection_id"] == "two"
    assert second["confidence"] == pytest.approx(0.4, abs=1e-6)


@_TENSOR_ONLY
def test_detections_to_dictionary_when_malformed_input_is_provided_tensor_native() -> (
    None
):
    # given - native detections without per-box detection_id -> serialiser raises
    operations = [{"type": "DetectionsToDictionary"}]
    detections = NativeDetections(
        xyxy=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.float32),
        class_id=torch.tensor([0, 1], dtype=torch.long),
        confidence=torch.tensor([0.3, 0.4], dtype=torch.float32),
        image_metadata={CLASS_NAMES_KEY: {0: "cat", 1: "dog"}},
        bboxes_metadata=None,
    )

    # when
    with pytest.raises(OperationError):
        _ = execute_operations(value=detections, operations=operations)


@_TENSOR_ONLY
def test_picking_detections_by_parent_class_when_empty_detections_provided_tensor_native() -> (
    None
):
    # given
    operations = [{"type": "PickDetectionsByParentClass", "parent_class": "a"}]
    detections = NativeDetections(
        xyxy=torch.zeros((0, 4), dtype=torch.float32),
        class_id=torch.zeros((0,), dtype=torch.long),
        confidence=torch.zeros((0,), dtype=torch.float32),
        image_metadata={CLASS_NAMES_KEY: {}},
    )

    # when
    result = execute_operations(value=detections, operations=operations)

    # then
    assert result is not detections
    assert result.xyxy.shape[0] == 0


@_TENSOR_ONLY
def test_picking_detections_by_parent_class_when_class_name_field_not_defined_tensor_native() -> (
    None
):
    # given - no CLASS_NAMES_KEY: the native op is stricter than the numpy sibling
    # (which yields []) and raises, since class names cannot be resolved.
    operations = [{"type": "PickDetectionsByParentClass", "parent_class": "a"}]
    detections = NativeDetections(
        xyxy=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.float32),
        class_id=torch.tensor([0, 1], dtype=torch.long),
        confidence=torch.tensor([0.3, 0.4], dtype=torch.float32),
        image_metadata={},
    )

    # when
    with pytest.raises(OperationError):
        _ = execute_operations(value=detections, operations=operations)


@_TENSOR_ONLY
def test_picking_detections_by_parent_class_when_parent_class_not_fond_tensor_native() -> (
    None
):
    # given
    operations = [{"type": "PickDetectionsByParentClass", "parent_class": "a"}]
    detections = NativeDetections(
        xyxy=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.float32),
        class_id=torch.tensor([0, 1], dtype=torch.long),
        confidence=torch.tensor([0.3, 0.4], dtype=torch.float32),
        image_metadata={CLASS_NAMES_KEY: {0: "c", 1: "d"}},
    )

    # when
    result = execute_operations(value=detections, operations=operations)

    # then
    assert result.xyxy.shape[0] == 0


@_TENSOR_ONLY
def test_picking_detections_by_parent_class_when_no_child_detections_matching_tensor_native() -> (
    None
):
    # given
    operations = [{"type": "PickDetectionsByParentClass", "parent_class": "a"}]
    detections = NativeDetections(
        xyxy=torch.tensor(
            [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], dtype=torch.float32
        ),
        class_id=torch.tensor([0, 0, 1], dtype=torch.long),
        confidence=torch.tensor([0.3, 0.4, 0.5], dtype=torch.float32),
        image_metadata={CLASS_NAMES_KEY: {0: "a", 1: "b"}},
    )

    # when
    result = execute_operations(value=detections, operations=operations)

    # then
    assert result.xyxy.shape[0] == 2
    assert np.allclose(
        result.xyxy.cpu().numpy(), np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
    )
    assert np.allclose(result.confidence.cpu().numpy(), [0.3, 0.4])


@_TENSOR_ONLY
def test_picking_detections_by_parent_class_when_there_are_child_detections_matching_tensor_native() -> (
    None
):
    # given
    operations = [{"type": "PickDetectionsByParentClass", "parent_class": "a"}]
    detections = NativeDetections(
        xyxy=torch.tensor(
            [[0, 0, 50, 50], [20, 20, 30, 30], [40, 40, 50, 50]], dtype=torch.float32
        ),
        class_id=torch.tensor([0, 1, 1], dtype=torch.long),
        confidence=torch.tensor([0.3, 0.4, 0.5], dtype=torch.float32),
        image_metadata={CLASS_NAMES_KEY: {0: "a", 1: "b"}},
    )

    # when
    result = execute_operations(value=detections, operations=operations)

    # then
    assert result.xyxy.shape[0] == 3
    assert np.allclose(
        result.xyxy.cpu().numpy(),
        np.array([[0, 0, 50, 50], [20, 20, 30, 30], [40, 40, 50, 50]]),
    )
    assert np.allclose(result.confidence.cpu().numpy(), [0.3, 0.4, 0.5])


@_TENSOR_ONLY
def test_picking_detections_by_parent_class_when_there_are_child_detections_matching_different_parents_tensor_native() -> (
    None
):
    # given
    operations = [{"type": "PickDetectionsByParentClass", "parent_class": "a"}]
    detections = NativeDetections(
        xyxy=torch.tensor(
            [
                [0, 0, 50, 50],
                [20, 20, 30, 30],
                [40, 40, 50, 50],
                [100, 100, 200, 200],
                [150, 100, 250, 200],
                [400, 400, 600, 600],
            ],
            dtype=torch.float32,
        ),
        class_id=torch.tensor([0, 1, 1, 0, 2, 3], dtype=torch.long),
        confidence=torch.tensor([0.3, 0.4, 0.5, 0.6, 0.7, 0.9], dtype=torch.float32),
        image_metadata={CLASS_NAMES_KEY: {0: "a", 1: "b", 2: "c", 3: "d"}},
    )

    # when
    result = execute_operations(value=detections, operations=operations)

    # then - parents first (original order), then contained dependents
    assert result.xyxy.shape[0] == 5
    assert np.allclose(
        result.xyxy.cpu().numpy(),
        np.array(
            [
                [0, 0, 50, 50],
                [100, 100, 200, 200],
                [20, 20, 30, 30],
                [40, 40, 50, 50],
                [150, 100, 250, 200],
            ]
        ),
    )
    assert np.allclose(result.confidence.cpu().numpy(), [0.3, 0.6, 0.4, 0.5, 0.7])


# ---------------------------------------------------------------------------
# Parity for the other native detection representations: InstanceDetections
# (masks -> polygons; pick preserves the mask) and the KeyPoints prediction
# (a (KeyPoints, Detections) tuple; pick slices both components together).
# ---------------------------------------------------------------------------


@_TENSOR_ONLY
def test_detections_to_dictionary_for_instance_segmentation_tensor_native() -> None:
    # given - InstanceDetections serialises like Detections but adds a per-box
    # polygon (POLYGON_KEY) derived from the mask. An instance whose mask has no
    # contour is silently dropped, so the masks here are filled blobs.
    operations = [{"type": "DetectionsToDictionary"}]
    mask = torch.zeros((2, 20, 20), dtype=torch.bool)
    mask[0, 2:12, 2:12] = True
    mask[1, 5:15, 5:15] = True
    detections = NativeInstanceDetections(
        xyxy=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.float32),
        class_id=torch.tensor([0, 1], dtype=torch.long),
        confidence=torch.tensor([0.3, 0.4], dtype=torch.float32),
        mask=mask,
        image_metadata={
            CLASS_NAMES_KEY: {0: "cat", 1: "dog"},
            IMAGE_DIMENSIONS_KEY: [192, 168],
        },
        bboxes_metadata=[{DETECTION_ID_KEY: "one"}, {DETECTION_ID_KEY: "two"}],
    )

    # when
    result = execute_operations(value=detections, operations=operations)

    # then
    assert result["image"] == {"width": 168, "height": 192}
    first, second = result["predictions"]
    assert first["class_id"] == 0 and first["class"] == "cat"
    assert first["detection_id"] == "one"
    assert first["confidence"] == pytest.approx(0.3, abs=1e-6)
    assert second["class_id"] == 1 and second["class"] == "dog"
    assert second["detection_id"] == "two"
    # the InstanceDetections-specific bit: a polygon rides on every prediction
    for prediction in (first, second):
        assert POLYGON_KEY in prediction
        assert len(prediction[POLYGON_KEY]) >= 3
        assert set(prediction[POLYGON_KEY][0].keys()) == {"x", "y"}


@_TENSOR_ONLY
def test_picking_detections_by_parent_class_for_instance_segmentation_tensor_native() -> (
    None
):
    # given - parent is NOT first, so the result re-orders boxes; the mask rows must
    # follow the same re-ordering. Each instance's mask carries a distinct pixel
    # count (1 / 2 / 3) so we can assert the rows move with their boxes.
    operations = [{"type": "PickDetectionsByParentClass", "parent_class": "a"}]
    mask = torch.zeros((3, 20, 20), dtype=torch.bool)
    mask[0, 0, 0:1] = True  # idx0 child -> 1 px
    mask[1, 0, 0:2] = True  # idx1 parent -> 2 px
    mask[2, 0, 0:3] = True  # idx2 child -> 3 px
    detections = NativeInstanceDetections(
        xyxy=torch.tensor(
            [[20, 20, 30, 30], [0, 0, 50, 50], [40, 40, 50, 50]], dtype=torch.float32
        ),
        class_id=torch.tensor([1, 0, 1], dtype=torch.long),
        confidence=torch.tensor([0.4, 0.3, 0.5], dtype=torch.float32),
        mask=mask,
        image_metadata={CLASS_NAMES_KEY: {0: "a", 1: "b"}},
    )

    # when
    result = execute_operations(value=detections, operations=operations)

    # then - same repr, parents-first ordering, mask sliced in lockstep
    assert isinstance(result, NativeInstanceDetections)
    assert result.xyxy.shape[0] == 3
    assert np.allclose(
        result.xyxy.cpu().numpy(),
        np.array([[0, 0, 50, 50], [20, 20, 30, 30], [40, 40, 50, 50]]),
    )
    assert np.allclose(result.confidence.cpu().numpy(), [0.3, 0.4, 0.5])
    mask_pixel_counts = result.mask.sum(dim=(1, 2)).cpu().numpy().tolist()
    assert mask_pixel_counts == [2, 1, 3]


@_TENSOR_ONLY
def test_picking_detections_by_parent_class_for_instance_segmentation_when_empty_tensor_native() -> (
    None
):
    # given
    operations = [{"type": "PickDetectionsByParentClass", "parent_class": "a"}]
    detections = NativeInstanceDetections(
        xyxy=torch.zeros((0, 4), dtype=torch.float32),
        class_id=torch.zeros((0,), dtype=torch.long),
        confidence=torch.zeros((0,), dtype=torch.float32),
        mask=torch.zeros((0, 20, 20), dtype=torch.bool),
        image_metadata={CLASS_NAMES_KEY: {}},
    )

    # when
    result = execute_operations(value=detections, operations=operations)

    # then - empty object of the SAME repr (not an sv / plain Detections)
    assert isinstance(result, NativeInstanceDetections)
    assert result.xyxy.shape[0] == 0
    assert result.mask.shape[0] == 0


@_TENSOR_ONLY
def test_detections_to_dictionary_for_keypoints_tensor_native() -> None:
    # given - a keypoint prediction is a (KeyPoints, Detections) tuple. The serialiser
    # uses only the bbox Detections component (class names / detection ids live there);
    # the raw keypoint payload is not part of the dictionary.
    operations = [{"type": "DetectionsToDictionary"}]
    key_points = NativeKeyPoints(
        xy=torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        ),  # (instances, kpts, 2)
        class_id=torch.tensor([0, 1], dtype=torch.long),
        confidence=torch.tensor([[0.9, 0.8], [0.7, 0.6]]),  # (instances, kpts)
        image_metadata={CLASS_NAMES_KEY: {0: "cat", 1: "dog"}},
    )
    bboxes = NativeDetections(
        xyxy=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.float32),
        class_id=torch.tensor([0, 1], dtype=torch.long),
        confidence=torch.tensor([0.3, 0.4], dtype=torch.float32),
        image_metadata={
            CLASS_NAMES_KEY: {0: "cat", 1: "dog"},
            IMAGE_DIMENSIONS_KEY: [192, 168],
        },
        bboxes_metadata=[{DETECTION_ID_KEY: "one"}, {DETECTION_ID_KEY: "two"}],
    )
    prediction = (key_points, bboxes)

    # when
    result = execute_operations(value=prediction, operations=operations)

    # then - serialised from the bbox component
    assert result["image"] == {"width": 168, "height": 192}
    first, second = result["predictions"]
    assert first["class_id"] == 0 and first["class"] == "cat"
    assert first["detection_id"] == "one"
    assert first["confidence"] == pytest.approx(0.3, abs=1e-6)
    assert second["class_id"] == 1 and second["class"] == "dog"
    assert second["detection_id"] == "two"


@_TENSOR_ONLY
def test_detections_to_dictionary_for_keypoints_when_bbox_missing_tensor_native() -> (
    None
):
    # given - a keypoint tuple with no bbox component is not a valid input
    operations = [{"type": "DetectionsToDictionary"}]
    key_points = NativeKeyPoints(
        xy=torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        class_id=torch.tensor([0], dtype=torch.long),
        confidence=torch.tensor([[0.9, 0.8]]),
        image_metadata={CLASS_NAMES_KEY: {0: "cat"}},
    )
    prediction = (key_points, None)

    # when
    with pytest.raises(InvalidInputTypeError):
        _ = execute_operations(value=prediction, operations=operations)


@_TENSOR_ONLY
def test_picking_detections_by_parent_class_for_keypoints_tensor_native() -> None:
    # given - pick on a keypoint tuple slices BOTH the KeyPoints and the bbox
    # Detections with the same parents-first index order. Parent (class "a") is at
    # index 1, so both components re-order to [parent, child, child]. Each instance's
    # first keypoint x is a distinct marker so we can assert the keypoints follow.
    operations = [{"type": "PickDetectionsByParentClass", "parent_class": "a"}]
    key_points = NativeKeyPoints(
        xy=torch.tensor(
            [
                [[10.0, 0.0], [11.0, 0.0]],  # idx0 child
                [[20.0, 0.0], [21.0, 0.0]],  # idx1 parent
                [[30.0, 0.0], [31.0, 0.0]],  # idx2 child
            ]
        ),
        class_id=torch.tensor([1, 0, 1], dtype=torch.long),
        confidence=torch.tensor([[0.9, 0.8], [0.7, 0.6], [0.5, 0.4]]),
        image_metadata={CLASS_NAMES_KEY: {0: "a", 1: "b"}},
    )
    bboxes = NativeDetections(
        xyxy=torch.tensor(
            [[20, 20, 30, 30], [0, 0, 50, 50], [40, 40, 50, 50]], dtype=torch.float32
        ),
        class_id=torch.tensor([1, 0, 1], dtype=torch.long),
        confidence=torch.tensor([0.4, 0.3, 0.5], dtype=torch.float32),
        image_metadata={CLASS_NAMES_KEY: {0: "a", 1: "b"}},
    )
    prediction = (key_points, bboxes)

    # when
    result = execute_operations(value=prediction, operations=operations)

    # then - a (KeyPoints, Detections) tuple, both sliced parents-first
    assert isinstance(result, tuple) and len(result) == 2
    result_key_points, result_bboxes = result
    assert isinstance(result_key_points, NativeKeyPoints)
    assert isinstance(result_bboxes, NativeDetections)
    assert np.allclose(
        result_bboxes.xyxy.cpu().numpy(),
        np.array([[0, 0, 50, 50], [20, 20, 30, 30], [40, 40, 50, 50]]),
    )
    assert np.allclose(result_bboxes.confidence.cpu().numpy(), [0.3, 0.4, 0.5])
    # the KeyPoints component is reduced to the same instances, in the same order
    # (len() here is the KeyPoints.__len__ added alongside these tests).
    assert len(result_key_points) == 3
    assert np.allclose(
        result_key_points.xy[:, 0, 0].cpu().numpy(), [20.0, 10.0, 30.0]
    )
