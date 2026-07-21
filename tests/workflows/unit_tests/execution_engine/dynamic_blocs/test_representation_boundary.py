"""Unit tests for the dynamic-block representation boundary (Step 2 - the IN
direction: native -> legacy). Pure converter/walker tests; engine wiring is
Step 4.

The boundary resolves ``ENABLE_TENSOR_DATA_REPRESENTATION`` once at import time
into ``representation_boundary._TENSOR_REPRESENTATION_ACTIVE``; tests patch that
module constant (the same technique block_assembler tests use for the env flag)
so both CI flag directions exercise every case deterministically.
"""

from collections import namedtuple
from typing import Optional
from unittest import mock
from uuid import UUID

import numpy as np
import pytest
import supervision as sv

torch = pytest.importorskip("torch")
pytest.importorskip("inference_models")

from pycocotools import mask as mask_utils

from inference.core.workflows.core_steps.common.utils import (
    sv_detections_to_root_coordinates,
)
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAME_KEY,
    CLASS_NAMES_KEY,
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY,
    PARENT_ID_KEY,
    POLYGON_KEY_IN_SV_DETECTIONS,
    PREDICTION_TYPE_KEY,
    ROOT_PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    ROOT_PARENT_ID_KEY,
    TRACKER_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks import (
    representation_boundary,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.entities import (
    DynamicInputDefinition,
    ManifestDescription,
    SelectorType,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.representation_boundary import (
    RepresentationBoundaryError,
    convert_kwargs_to_legacy,
    native_detections_to_sv,
)
from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks

CROP_OFFSET_X, CROP_OFFSET_Y = 100, 50
CROP_H, CROP_W = 40, 60
ROOT_H, ROOT_W = 480, 640

_boundary_on = mock.patch.object(
    representation_boundary, "_TENSOR_REPRESENTATION_ACTIVE", True
)
_boundary_off = mock.patch.object(
    representation_boundary, "_TENSOR_REPRESENTATION_ACTIVE", False
)


def _crop_local_dense_masks() -> np.ndarray:
    masks = np.zeros((2, CROP_H, CROP_W), dtype=bool)
    masks[0, 5:15, 10:30] = True
    masks[1, 20:35, 40:55] = True
    return masks


def _native_image_metadata() -> dict:
    return {
        CLASS_NAMES_KEY: {0: "a", 1: "b"},
        PREDICTION_TYPE_KEY: "instance-segmentation",
        IMAGE_DIMENSIONS_KEY: [CROP_H, CROP_W],
        INFERENCE_ID_KEY: "iid-1",
        PARENT_ID_KEY: "crop-1",
        PARENT_COORDINATES_KEY: [CROP_OFFSET_X, CROP_OFFSET_Y],
        PARENT_DIMENSIONS_KEY: [ROOT_H, ROOT_W],
        ROOT_PARENT_ID_KEY: "root-image",
        ROOT_PARENT_COORDINATES_KEY: [CROP_OFFSET_X, CROP_OFFSET_Y],
        ROOT_PARENT_DIMENSIONS_KEY: [ROOT_H, ROOT_W],
    }


def _native_instance_detections(
    mask: object,
    with_polygons: bool = False,
) -> InstanceDetections:
    bboxes_metadata = [
        {DETECTION_ID_KEY: "d0"},
        {DETECTION_ID_KEY: "d1"},
    ]
    if with_polygons:
        bboxes_metadata[0][POLYGON_KEY_IN_SV_DETECTIONS] = np.array(
            [[10, 5], [30, 5], [30, 15], [10, 15]], dtype=np.int64
        )
        bboxes_metadata[1][POLYGON_KEY_IN_SV_DETECTIONS] = np.array(
            [[40, 20], [55, 20], [55, 35], [40, 35]], dtype=np.int64
        )
    return InstanceDetections(
        xyxy=torch.tensor(
            [[10.0, 5.0, 30.0, 15.0], [40.0, 20.0, 55.0, 35.0]], dtype=torch.float32
        ),
        class_id=torch.tensor([0, 1], dtype=torch.long),
        confidence=torch.tensor([0.5, 0.25], dtype=torch.float32),
        mask=mask,
        image_metadata=_native_image_metadata(),
        bboxes_metadata=bboxes_metadata,
    )


def _native_object_detections() -> Detections:
    return Detections(
        xyxy=torch.tensor([[10.0, 5.0, 30.0, 15.0]], dtype=torch.float32),
        class_id=torch.tensor([1], dtype=torch.long),
        confidence=torch.tensor([0.75], dtype=torch.float32),
        image_metadata=_native_image_metadata(),
        bboxes_metadata=[{DETECTION_ID_KEY: "d0"}],
    )


def _dense_masks_to_rle(masks: np.ndarray) -> InstancesRLEMasks:
    encoded = [
        mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"]
        for mask in masks
    ]
    return InstancesRLEMasks(image_size=(CROP_H, CROP_W), masks=encoded)


def _manifest(inputs: dict) -> ManifestDescription:
    return ManifestDescription(
        type="ManifestDescription",
        block_type="MyBlock",
        inputs=inputs,
    )


def _input_with_kinds(*kind_names: str) -> DynamicInputDefinition:
    return DynamicInputDefinition(
        type="DynamicInputDefinition",
        selector_types=[SelectorType.STEP_OUTPUT],
        selector_data_kind={SelectorType.STEP_OUTPUT: list(kind_names)},
    )


def test_native_detections_to_sv_carries_full_lineage_key_set() -> None:
    # given
    native = _native_instance_detections(
        mask=torch.from_numpy(_crop_local_dense_masks())
    )

    # when
    converted = native_detections_to_sv(detections=native)

    # then - key-set superset of what attach_parents_coordinates_to_sv_detections
    # attaches, plus prediction_type / image_dimensions / inference_id
    expected_keys = {
        "class_name",
        DETECTION_ID_KEY,
        PARENT_ID_KEY,
        PARENT_COORDINATES_KEY,
        PARENT_DIMENSIONS_KEY,
        ROOT_PARENT_ID_KEY,
        ROOT_PARENT_COORDINATES_KEY,
        ROOT_PARENT_DIMENSIONS_KEY,
        PREDICTION_TYPE_KEY,
        IMAGE_DIMENSIONS_KEY,
        INFERENCE_ID_KEY,
    }
    assert expected_keys.issubset(set(converted.data.keys()))
    # shapes/values match the numpy convention exactly: (n, 2) [x, y] coordinates,
    # (n, 2) [h, w] dimensions, broadcast string ids
    assert np.array_equal(
        converted.data[PARENT_COORDINATES_KEY],
        np.array([[CROP_OFFSET_X, CROP_OFFSET_Y]] * 2),
    )
    assert np.array_equal(
        converted.data[ROOT_PARENT_DIMENSIONS_KEY], np.array([[ROOT_H, ROOT_W]] * 2)
    )
    assert converted.data[PARENT_ID_KEY].tolist() == ["crop-1", "crop-1"]
    assert converted.data[ROOT_PARENT_ID_KEY].tolist() == ["root-image", "root-image"]
    assert converted.data[PREDICTION_TYPE_KEY].tolist() == ["instance-segmentation"] * 2
    assert np.array_equal(
        converted.data[IMAGE_DIMENSIONS_KEY], np.array([[CROP_H, CROP_W]] * 2)
    )
    assert converted.data[INFERENCE_ID_KEY].tolist() == ["iid-1", "iid-1"]
    assert np.array_equal(
        converted.xyxy,
        np.array([[10.0, 5.0, 30.0, 15.0], [40.0, 20.0, 55.0, 35.0]], np.float32),
    )
    assert isinstance(converted.mask, torch.Tensor)
    assert converted.mask.dtype == torch.bool
    assert converted.mask.shape == (2, CROP_H, CROP_W)
    assert torch.equal(converted.mask, torch.from_numpy(_crop_local_dense_masks()))


def test_native_detections_to_sv_output_survives_numpy_root_coordinates_shift() -> None:
    # given - crop-local native prediction with dense masks and per-box polygons
    dense = _crop_local_dense_masks()
    native = _native_instance_detections(
        mask=torch.from_numpy(dense), with_polygons=True
    )
    converted = native_detections_to_sv(detections=native)

    # when - the NUMPY root-coordinates conversion runs on the converted object
    shifted = sv_detections_to_root_coordinates(detections=converted)

    # then - boxes, masks and polygon payloads land in root coordinates
    assert np.array_equal(
        shifted.xyxy,
        np.array(
            [
                [110.0, 55.0, 130.0, 65.0],
                [140.0, 70.0, 155.0, 85.0],
            ],
            dtype=np.float32,
        ),
    )
    expected_masks = np.zeros((2, ROOT_H, ROOT_W), dtype=bool)
    expected_masks[
        :,
        CROP_OFFSET_Y : CROP_OFFSET_Y + CROP_H,
        CROP_OFFSET_X : CROP_OFFSET_X + CROP_W,
    ] = dense
    assert np.array_equal(shifted.mask, expected_masks)
    assert np.array_equal(
        np.asarray(shifted.data[POLYGON_KEY_IN_SV_DETECTIONS][0]),
        np.array([[110, 55], [130, 55], [130, 65], [110, 65]]),
    )


def test_native_detections_to_sv_resolves_class_name_override_first() -> None:
    # given - classes_replacement-style per-box override on row 0 only
    native = _native_instance_detections(
        mask=torch.from_numpy(_crop_local_dense_masks())
    )
    native.bboxes_metadata[0][CLASS_NAME_KEY] = "replaced-label"

    # when
    converted = native_detections_to_sv(detections=native)

    # then - override wins on row 0, map resolves row 1; no shadow 'class' column
    assert converted.data["class_name"].tolist() == ["replaced-label", "b"]
    assert CLASS_NAME_KEY not in converted.data


def test_native_detections_to_sv_mints_uuid_detection_ids_when_missing() -> None:
    # given
    native = _native_instance_detections(
        mask=torch.from_numpy(_crop_local_dense_masks())
    )
    del native.bboxes_metadata[1][DETECTION_ID_KEY]

    # when
    converted = native_detections_to_sv(detections=native)

    # then
    detection_ids = converted.data[DETECTION_ID_KEY].tolist()
    assert detection_ids[0] == "d0"
    assert detection_ids[1] != "" and UUID(detection_ids[1])


def test_native_detections_to_sv_rle_and_dense_carriers_materialise_identically() -> (
    None
):
    # given
    dense = _crop_local_dense_masks()
    via_dense = _native_instance_detections(mask=torch.from_numpy(dense))
    via_rle = _native_instance_detections(mask=_dense_masks_to_rle(dense))

    # when
    converted_dense = native_detections_to_sv(detections=via_dense)
    converted_rle = native_detections_to_sv(detections=via_rle)

    # then - both carriers land as the same dense (N, H, W) bool tensor
    assert isinstance(converted_dense.mask, torch.Tensor)
    assert isinstance(converted_rle.mask, torch.Tensor)
    assert converted_dense.mask.dtype == torch.bool
    assert converted_rle.mask.dtype == torch.bool
    assert converted_dense.mask.shape == (2, CROP_H, CROP_W)
    assert converted_rle.mask.shape == (2, CROP_H, CROP_W)
    assert torch.equal(converted_dense.mask, converted_rle.mask)
    assert torch.equal(converted_dense.mask, torch.from_numpy(dense))


def test_native_detections_to_sv_pads_keypoint_payloads_like_numpy() -> None:
    # given - 2 keypoints on row 0, none on row 1 (ragged input)
    native = _native_instance_detections(
        mask=torch.from_numpy(_crop_local_dense_masks())
    )
    native.bboxes_metadata[0].update(
        {
            KEYPOINTS_XY_KEY_IN_SV_DETECTIONS: [[11.0, 6.0], [29.0, 14.0]],
            KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS: [0.875, 0.75],
            KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS: [0, 1],
            KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS: ["nose", "tail"],
        }
    )
    native.bboxes_metadata[1].update(
        {
            KEYPOINTS_XY_KEY_IN_SV_DETECTIONS: [],
            KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS: [],
            KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS: [],
            KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS: [],
        }
    )

    # when
    converted = native_detections_to_sv(detections=native)

    # then - numeric payloads are padded tensor columns; class names remain an
    # object column because strings have no tensor representation
    xy = converted.data[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS]
    confidence = converted.data[KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS]
    class_id = converted.data[KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS]

    assert isinstance(xy, torch.Tensor)
    assert xy.dtype == torch.float32
    assert xy.shape == (2, 2, 2)
    torch.testing.assert_close(
        xy,
        torch.tensor(
            [
                [[11.0, 6.0], [29.0, 14.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
    )

    assert isinstance(confidence, torch.Tensor)
    assert confidence.dtype == torch.float32
    assert confidence.shape == (2, 2)
    torch.testing.assert_close(
        confidence,
        torch.tensor([[0.875, 0.75], [0.0, 0.0]], dtype=torch.float32),
    )

    assert isinstance(class_id, torch.Tensor)
    assert class_id.dtype == torch.int64
    assert class_id.shape == (2, 2)
    assert torch.equal(class_id, torch.tensor([[0, 1], [0, 0]], dtype=torch.int64))

    assert converted.data[KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS][1].tolist() == [
        "",
        "",
    ]


def test_native_detections_to_sv_tracker_ids() -> None:
    # given
    native = _native_instance_detections(
        mask=torch.from_numpy(_crop_local_dense_masks())
    )
    native.bboxes_metadata[0][TRACKER_ID_KEY] = 7
    native.bboxes_metadata[1][TRACKER_ID_KEY] = 11

    # when
    converted = native_detections_to_sv(detections=native)

    # then - tracker ids populate the sv field, not a data column
    assert converted.tracker_id.tolist() == [7, 11]
    assert TRACKER_ID_KEY not in converted.data


def test_native_detections_to_sv_handles_plain_and_empty_detections() -> None:
    # given
    plain = _native_object_detections()
    empty = Detections(
        xyxy=torch.zeros((0, 4), dtype=torch.float32),
        class_id=torch.zeros((0,), dtype=torch.long),
        confidence=torch.zeros((0,), dtype=torch.float32),
        image_metadata=_native_image_metadata(),
        bboxes_metadata=[],
    )

    # when
    converted_plain = native_detections_to_sv(detections=plain)
    converted_empty = native_detections_to_sv(detections=empty)

    # then
    assert converted_plain.mask is None
    assert len(converted_plain) == 1
    assert len(converted_empty) == 0


def test_convert_kwargs_is_identity_when_tensor_representation_off() -> None:
    # given
    kwargs = {"predictions": _native_object_detections(), "threshold": 0.5}
    manifest = _manifest(inputs={})

    # when
    with _boundary_off:
        result = convert_kwargs_to_legacy(
            kwargs=kwargs, manifest_description=manifest, block_name="block"
        )

    # then - the very same object, untouched (flag-off byte-parity guarantee)
    assert result is kwargs
    assert isinstance(result["predictions"], Detections)


def test_convert_kwargs_is_identity_for_tensor_native_mode() -> None:
    # given
    kwargs = {"predictions": _native_object_detections()}
    manifest = ManifestDescription(
        type="ManifestDescription",
        block_type="MyBlock",
        inputs={},
        tensor_compatibility="tensor_native",
    )

    # when
    with _boundary_on:
        result = convert_kwargs_to_legacy(
            kwargs=kwargs, manifest_description=manifest, block_name="block"
        )

    # then
    assert result is kwargs
    assert isinstance(result["predictions"], Detections)


def test_convert_kwargs_declared_kind_converts_batch_preserving_indices() -> None:
    # given
    manifest = _manifest(
        inputs={"predictions": _input_with_kinds("object_detection_prediction")}
    )
    batch = Batch.init(
        content=[_native_object_detections(), None, _native_object_detections()],
        indices=[(0,), (1,), (2,)],
    )

    # when
    with _boundary_on:
        result = convert_kwargs_to_legacy(
            kwargs={"predictions": batch},
            manifest_description=manifest,
            block_name="block",
        )

    # then
    converted = result["predictions"]
    assert isinstance(converted, Batch)
    assert converted.indices == [(0,), (1,), (2,)]
    assert isinstance(converted[0], sv.Detections)
    assert converted[1] is None
    assert isinstance(converted[2], sv.Detections)
    assert converted[0].data["class_name"].tolist() == ["b"]


def test_convert_kwargs_nested_batches_walked() -> None:
    # given
    manifest = _manifest(
        inputs={"predictions": _input_with_kinds("object_detection_prediction")}
    )
    inner_one = Batch.init(content=[_native_object_detections()], indices=[(0, 0)])
    inner_two = Batch.init(content=[None], indices=[(1, 0)])
    outer = Batch.init(content=[inner_one, inner_two], indices=[(0,), (1,)])

    # when
    with _boundary_on:
        result = convert_kwargs_to_legacy(
            kwargs={"predictions": outer},
            manifest_description=manifest,
            block_name="block",
        )

    # then
    outer_converted = result["predictions"]
    assert outer_converted.indices == [(0,), (1,)]
    assert outer_converted[0].indices == [(0, 0)]
    assert isinstance(outer_converted[0][0], sv.Detections)
    assert outer_converted[1][0] is None


def test_convert_kwargs_declared_kind_with_wrong_type_raises_loudly() -> None:
    # given - declared classification receives a native detections object
    manifest = _manifest(
        inputs={"predictions": _input_with_kinds("classification_prediction")}
    )

    # when
    with _boundary_on, pytest.raises(RepresentationBoundaryError) as error:
        _ = convert_kwargs_to_legacy(
            kwargs={"predictions": _native_object_detections()},
            manifest_description=manifest,
            block_name="my_block",
        )

    # then
    assert "my_block" in str(error.value)
    assert "predictions" in str(error.value)
    assert "Detections" in str(error.value)


def test_convert_kwargs_wildcard_sniffs_known_native_types() -> None:
    # given
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="img"),
        numpy_image=np.zeros((6, 8, 3), dtype=np.uint8),
    )
    classification = ClassificationPrediction(
        class_id=torch.tensor([1]),
        confidence=torch.tensor([[0.25, 0.5]], dtype=torch.float32),
        images_metadata=[
            {
                CLASS_NAMES_KEY: {0: "cat", 1: "dog"},
                PREDICTION_TYPE_KEY: "classification",
                IMAGE_DIMENSIONS_KEY: [480, 640],
            }
        ],
    )
    key_points = KeyPoints(
        xy=torch.tensor([[[11.0, 6.0]]], dtype=torch.float32),
        class_id=torch.tensor([0], dtype=torch.long),
        confidence=torch.tensor([[0.9]], dtype=torch.float32),
    )
    kwargs = {
        "detections": _native_object_detections(),
        "keypoints": (key_points, _native_object_detections()),
        "classes": classification,
        "image": image,
        "threshold": 0.4,
        "label": "on",
        "nested": {"inner": [_native_object_detections(), 3]},
    }

    # when
    with _boundary_on:
        result = convert_kwargs_to_legacy(
            kwargs=kwargs, manifest_description=_manifest(inputs={}), block_name="b"
        )

    # then
    assert isinstance(result["detections"], sv.Detections)
    assert isinstance(result["keypoints"], sv.Detections)
    assert result["classes"]["top"] == "dog"
    assert result["image"] is image
    assert result["threshold"] == 0.4 and result["label"] == "on"
    assert isinstance(result["nested"]["inner"][0], sv.Detections)
    assert result["nested"]["inner"][1] == 3


def test_convert_kwargs_wildcard_bare_tensor_raises_actionable_error() -> None:
    # when
    with _boundary_on, pytest.raises(RepresentationBoundaryError) as error:
        _ = convert_kwargs_to_legacy(
            kwargs={"embedding": torch.zeros(4)},
            manifest_description=_manifest(inputs={}),
            block_name="my_block",
        )

    # then - block, input, offending type and remediation all present
    message = str(error.value)
    assert "my_block" in message
    assert "embedding" in message
    assert "torch.Tensor" in message
    assert "tensor_native" in message


def test_convert_kwargs_bare_keypoints_and_unknown_dataclass_raise() -> None:
    # given
    bare_key_points = KeyPoints(
        xy=torch.tensor([[[1.0, 1.0]]], dtype=torch.float32),
        class_id=torch.tensor([0], dtype=torch.long),
        confidence=torch.tensor([[0.9]], dtype=torch.float32),
    )
    unknown = InstancesRLEMasks(image_size=(4, 4), masks=[])

    # when / then
    with _boundary_on:
        with pytest.raises(RepresentationBoundaryError):
            _ = convert_kwargs_to_legacy(
                kwargs={"kp": bare_key_points},
                manifest_description=_manifest(inputs={}),
                block_name="b",
            )
        with pytest.raises(RepresentationBoundaryError):
            _ = convert_kwargs_to_legacy(
                kwargs={"masks": unknown},
                manifest_description=_manifest(inputs={}),
                block_name="b",
            )


def test_convert_kwargs_declared_embedding_and_tensor_kinds() -> None:
    # given
    manifest = _manifest(
        inputs={
            "embedding": _input_with_kinds("embedding"),
            "raw": _input_with_kinds("tensor"),
            "static_embedding": _input_with_kinds("embedding"),
        }
    )
    kwargs = {
        "embedding": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "raw": torch.ones((2, 2)),
        "static_embedding": [0.1, 0.2],
    }

    # when
    with _boundary_on:
        result = convert_kwargs_to_legacy(
            kwargs=kwargs, manifest_description=manifest, block_name="b"
        )

    # then - embedding flattens to List[float] (the numpy clip in-memory shape:
    # `predictions.embeddings[0]`); `tensor` kind lands as ndarray; static values
    # pass through untouched
    assert result["embedding"] == [1.0, 2.0, 3.0, 4.0]
    assert isinstance(result["raw"], np.ndarray)
    assert np.array_equal(result["raw"], np.ones((2, 2)))
    assert result["static_embedding"] == [0.1, 0.2]


def test_classification_conversion_matches_numpy_in_memory_dict() -> None:
    # given - the exact in-memory dict the numpy classification block emits:
    # response.model_dump(by_alias=True, exclude_none=True) + in-place
    # prediction_type / parent_id / root_parent_id writes
    from inference.core.entities.responses.inference import (
        ClassificationInferenceResponse,
    )
    from inference.core.entities.responses.inference import (
        ClassificationPrediction as ResponseClassificationPrediction,
    )
    from inference.core.entities.responses.inference import InferenceResponseImage

    response = ClassificationInferenceResponse(
        image=InferenceResponseImage(width=640, height=480),
        predictions=[
            ResponseClassificationPrediction(
                **{"class": "dog", "class_id": 1, "confidence": 0.5}
            ),
            ResponseClassificationPrediction(
                **{"class": "cat", "class_id": 0, "confidence": 0.25}
            ),
        ],
        top="dog",
        confidence=0.5,
        time=0.0123,
        inference_id="iid",
    )
    numpy_in_memory = response.model_dump(by_alias=True, exclude_none=True)
    numpy_in_memory[PREDICTION_TYPE_KEY] = "classification"
    numpy_in_memory[PARENT_ID_KEY] = "p1"
    numpy_in_memory[ROOT_PARENT_ID_KEY] = "r1"

    native = ClassificationPrediction(
        class_id=torch.tensor([1]),
        confidence=torch.tensor([[0.25, 0.5]], dtype=torch.float32),
        images_metadata=[
            {
                CLASS_NAMES_KEY: {0: "cat", 1: "dog"},
                PREDICTION_TYPE_KEY: "classification",
                IMAGE_DIMENSIONS_KEY: [480, 640],
                INFERENCE_ID_KEY: "iid",
                PARENT_ID_KEY: "p1",
                ROOT_PARENT_ID_KEY: "r1",
                "time": 0.0123,
            }
        ],
    )

    # when
    with _boundary_on:
        result = convert_kwargs_to_legacy(
            kwargs={"classes": native},
            manifest_description=_manifest(
                inputs={"classes": _input_with_kinds("classification_prediction")}
            ),
            block_name="b",
        )

    # then
    assert result["classes"] == numpy_in_memory


def test_convert_kwargs_multi_label_classification_sniffed() -> None:
    # given
    native = MultiLabelClassificationPrediction(
        class_ids=torch.tensor([0, 1]),
        confidence=torch.tensor([0.5, 0.25], dtype=torch.float32),
        image_metadata={
            CLASS_NAMES_KEY: {0: "cat", 1: "dog"},
            PREDICTION_TYPE_KEY: "classification",
            IMAGE_DIMENSIONS_KEY: [480, 640],
        },
    )

    # when
    with _boundary_on:
        result = convert_kwargs_to_legacy(
            kwargs={"classes": native},
            manifest_description=_manifest(inputs={}),
            block_name="b",
        )

    # then
    assert result["classes"]["predicted_classes"] == ["cat", "dog"]
    assert result["classes"]["predictions"]["cat"]["class_id"] == 0


def test_convert_kwargs_wildcard_keypoint_tuple_without_bbox_component_raises() -> None:
    # given - the native keypoint tuple is (KeyPoints, Optional[Detections]); a
    # None bbox component has no legacy sv.Detections equivalent BY DESIGN.
    key_points = KeyPoints(
        xy=torch.tensor([[[1.0, 2.0]]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([[0.9]]),
    )

    # when
    with _boundary_on, pytest.raises(RepresentationBoundaryError) as error:
        _ = convert_kwargs_to_legacy(
            kwargs={"prediction": (key_points, None)},
            manifest_description=_manifest(inputs={}),
            block_name="my_block",
        )

    # then
    assert "missing the bounding-box component" in str(error.value)
    assert "my_block" in str(error.value)


def test_convert_kwargs_empty_batch_passes_through_preserving_type() -> None:
    # given
    empty_batch = Batch.init(content=[], indices=[])

    # when
    with _boundary_on:
        result = convert_kwargs_to_legacy(
            kwargs={"predictions": empty_batch},
            manifest_description=_manifest(
                inputs={"predictions": _input_with_kinds("object_detection_prediction")}
            ),
            block_name="my_block",
        )

    # then
    converted = result["predictions"]
    assert isinstance(converted, Batch)
    assert len(converted) == 0
    assert list(converted.indices) == []


def test_namedtuple_is_preserved_across_both_boundary_walkers() -> None:
    boundary_value_type = namedtuple("BoundaryValue", ["label", "score"])
    boundary_value = boundary_value_type(label="ok", score=0.75)

    with _boundary_on:
        legacy_kwargs = convert_kwargs_to_legacy(
            kwargs={"value": boundary_value},
            manifest_description=_manifest(inputs={}),
            block_name="my_block",
        )
        native_result = convert_block_result_to_native(
            result={"value": boundary_value},
            manifest_description=_manifest_with_outputs(outputs={}),
            block_name="my_block",
        )

    assert type(legacy_kwargs["value"]) is boundary_value_type
    assert legacy_kwargs["value"] == boundary_value
    assert type(native_result["value"]) is boundary_value_type
    assert native_result["value"] == boundary_value


# --------------------------------------------------------------------------- #
# Step 3: OUT direction (legacy -> native) + result walker + round trips      #
# --------------------------------------------------------------------------- #

from inference.core.workflows.core_steps.common.serializers_tensor import (
    serialise_native_classification,
    serialise_sv_detections,
)
from inference.core.workflows.core_steps.common.tensor_native import (
    native_detections_to_root_coordinates,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.entities import (
    DynamicOutputDefinition,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.representation_boundary import (
    classification_dict_to_native,
    convert_block_result_to_native,
    sv_detections_to_native,
    sv_detections_to_native_key_point_prediction,
)
from inference.core.workflows.execution_engine.v1.entities import FlowControl


def _manifest_with_outputs(outputs: dict) -> ManifestDescription:
    return ManifestDescription(
        type="ManifestDescription",
        block_type="MyBlock",
        inputs={},
        outputs=outputs,
    )


def _output_with_kinds(*kind_names: str) -> DynamicOutputDefinition:
    return DynamicOutputDefinition(
        type="DynamicOutputDefinition", kind=list(kind_names)
    )


def _assert_native_detections_close(left: "Detections", right: "Detections") -> None:
    assert np.allclose(left.xyxy.cpu().numpy(), right.xyxy.cpu().numpy())
    assert left.class_id.cpu().tolist() == right.class_id.cpu().tolist()
    assert np.allclose(left.confidence.cpu().numpy(), right.confidence.cpu().numpy())


def test_sv_detections_to_native_round_trips_dense_instance_detections() -> None:
    # given - the crop-lineage native fixture with dense masks
    native = _native_instance_detections(
        mask=torch.from_numpy(_crop_local_dense_masks())
    )

    # when - native -> legacy -> native
    as_sv = native_detections_to_sv(detections=native)
    round_tripped = sv_detections_to_native(sv_detections=as_sv)

    # then - values, ids, lineage and masks survive
    assert isinstance(round_tripped, InstanceDetections)
    _assert_native_detections_close(native, round_tripped)
    assert [per_box[DETECTION_ID_KEY] for per_box in round_tripped.bboxes_metadata] == [
        "d0",
        "d1",
    ]
    original_metadata = _native_image_metadata()
    for key in (
        PARENT_ID_KEY,
        ROOT_PARENT_ID_KEY,
        PREDICTION_TYPE_KEY,
        INFERENCE_ID_KEY,
    ):
        assert str(original_metadata[key]) == str(round_tripped.image_metadata[key])
    for key in (
        PARENT_COORDINATES_KEY,
        PARENT_DIMENSIONS_KEY,
        ROOT_PARENT_COORDINATES_KEY,
        ROOT_PARENT_DIMENSIONS_KEY,
        IMAGE_DIMENSIONS_KEY,
    ):
        assert list(original_metadata[key]) == list(round_tripped.image_metadata[key])
    assert isinstance(round_tripped.mask, torch.Tensor)
    assert np.array_equal(
        round_tripped.mask.cpu().numpy().astype(bool), _crop_local_dense_masks()
    )
    # acceptance bar: the tensor serializer accepts it...
    serialized = serialise_sv_detections(round_tripped)
    assert [entry["class"] for entry in serialized["predictions"]] == ["a", "b"]
    # ...and root-coordinates conversion works on it (crop lineage present)
    at_root = native_detections_to_root_coordinates(prediction=round_tripped)
    assert np.allclose(
        at_root.xyxy.cpu().numpy()[:, 0],
        native.xyxy.cpu().numpy()[:, 0] + CROP_OFFSET_X,
    )


def test_sv_detections_to_native_round_trips_rle_carrier_under_prefer_rle() -> None:
    # given - native with an RLE carrier
    dense = _crop_local_dense_masks()
    native = _native_instance_detections(mask=_dense_masks_to_rle(dense))

    # when - IN densifies; OUT re-encodes under prefer_rle (declared RLE kind)
    as_sv = native_detections_to_sv(detections=native)
    round_tripped = sv_detections_to_native(sv_detections=as_sv, prefer_rle=True)

    # then - carrier is RLE again and decodes to the same masks
    assert isinstance(round_tripped, InstanceDetections)
    assert isinstance(round_tripped.mask, InstancesRLEMasks)
    from inference_models.models.common.rle_utils import coco_rle_masks_to_numpy_mask

    assert np.array_equal(coco_rle_masks_to_numpy_mask(round_tripped.mask), dense)


def test_sv_detections_to_native_round_trips_tracker_ids_and_class_overrides() -> None:
    # given - two rows SHARING class_id 0, one carrying a per-box override, plus
    # tracker ids (the classes_replacement -> rename regression shape)
    native = InstanceDetections(
        xyxy=torch.tensor(
            [[10.0, 5.0, 30.0, 15.0], [40.0, 20.0, 55.0, 35.0]], dtype=torch.float32
        ),
        class_id=torch.tensor([0, 0], dtype=torch.long),
        confidence=torch.tensor([0.5, 0.25], dtype=torch.float32),
        mask=torch.from_numpy(_crop_local_dense_masks()),
        image_metadata=_native_image_metadata(),
        bboxes_metadata=[
            {DETECTION_ID_KEY: "d0", TRACKER_ID_KEY: 11},
            {DETECTION_ID_KEY: "d1", TRACKER_ID_KEY: 14, CLASS_NAME_KEY: "override"},
        ],
    )

    # when
    as_sv = native_detections_to_sv(detections=native)
    round_tripped = sv_detections_to_native(sv_detections=as_sv)

    # then - serialized outputs are identical (the strongest effective-equality)
    original_serialized = serialise_sv_detections(native)
    round_tripped_serialized = serialise_sv_detections(round_tripped)
    assert original_serialized == round_tripped_serialized
    assert [per_box[TRACKER_ID_KEY] for per_box in round_tripped.bboxes_metadata] == [
        11,
        14,
    ]


def test_sv_detections_to_native_from_user_built_sv() -> None:
    # given - a bare sv.Detections the way legacy user code builds one (no
    # lineage, no detection ids)
    user_built = sv.Detections(
        xyxy=np.array([[1.0, 2.0, 11.0, 22.0]], dtype=np.float32),
        class_id=np.array([3]),
        confidence=np.array([0.9], dtype=np.float32),
        data={"class_name": np.array(["widget"], dtype=object)},
    )

    # when
    as_native = sv_detections_to_native(sv_detections=user_built)

    # then - serializer hard requirements are met out of the box
    assert isinstance(as_native, Detections)
    assert as_native.image_metadata[CLASS_NAMES_KEY] == {3: "widget"}
    minted = as_native.bboxes_metadata[0][DETECTION_ID_KEY]
    UUID(minted)  # parseable uuid
    serialized = serialise_sv_detections(as_native)
    assert serialized["predictions"][0]["class"] == "widget"
    # and converting back preserves the user's view
    back = native_detections_to_sv(detections=as_native)
    assert np.allclose(back.xyxy, user_built.xyxy)
    assert back.data["class_name"].tolist() == ["widget"]


def test_classification_single_label_round_trip() -> None:
    # given - a native single-label prediction with threshold + time in metadata
    metadata = {
        CLASS_NAMES_KEY: {0: "cat", 1: "dog"},
        PREDICTION_TYPE_KEY: "classification",
        IMAGE_DIMENSIONS_KEY: [480, 640],
        INFERENCE_ID_KEY: "iid-9",
        PARENT_ID_KEY: "img-1",
        ROOT_PARENT_ID_KEY: "img-1",
        "classification_confidence_threshold": 0.3,
        "time": 0.0123,
    }
    native = ClassificationPrediction(
        class_id=torch.tensor([1], dtype=torch.long),
        confidence=torch.tensor([[0.3, 0.7]], dtype=torch.float32),
        images_metadata=[metadata],
    )

    # when - native -> legacy dict -> native -> legacy dict
    legacy_dict = serialise_native_classification(native)
    rebuilt = classification_dict_to_native(
        prediction=legacy_dict, block_name="b", value_name="o"
    )
    re_serialized = serialise_native_classification(rebuilt)

    # then - byte-stable through the round trip, time included
    assert isinstance(rebuilt, ClassificationPrediction)
    assert legacy_dict == re_serialized
    assert re_serialized["time"] == 0.0123


def test_classification_multi_label_round_trip() -> None:
    # given - a native multi-label prediction with a gap class id
    metadata = {
        CLASS_NAMES_KEY: {0: "cat", 1: "1", 2: "dog"},
        PREDICTION_TYPE_KEY: "classification",
        IMAGE_DIMENSIONS_KEY: [480, 640],
        INFERENCE_ID_KEY: "iid-9",
        PARENT_ID_KEY: "img-1",
        ROOT_PARENT_ID_KEY: "img-1",
    }
    native = MultiLabelClassificationPrediction(
        class_ids=torch.tensor([0, 2], dtype=torch.long),
        confidence=torch.tensor([0.9, 0.0, 0.8], dtype=torch.float32),
        image_metadata=metadata,
    )

    # when
    legacy_dict = serialise_native_classification(native)
    rebuilt = classification_dict_to_native(
        prediction=legacy_dict, block_name="b", value_name="o"
    )
    re_serialized = serialise_native_classification(rebuilt)

    # then
    assert isinstance(rebuilt, MultiLabelClassificationPrediction)
    assert legacy_dict == re_serialized
    assert rebuilt.class_ids.cpu().tolist() == [0, 2]


def test_keypoint_tuple_round_trip() -> None:
    # given - a native keypoint prediction: bbox component carries per-box
    # keypoint payloads (the serializer convention)
    bboxes_metadata = [
        {
            DETECTION_ID_KEY: "d0",
            KEYPOINTS_XY_KEY_IN_SV_DETECTIONS: [[11.0, 6.0], [21.0, 9.0]],
            KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS: [0.875, 0.75],
            KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS: [0, 1],
            KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS: ["nose", "eye"],
        },
    ]
    bbox_component = Detections(
        xyxy=torch.tensor([[10.0, 5.0, 30.0, 15.0]], dtype=torch.float32),
        class_id=torch.tensor([0], dtype=torch.long),
        confidence=torch.tensor([0.5], dtype=torch.float32),
        image_metadata=_native_image_metadata(),
        bboxes_metadata=bboxes_metadata,
    )
    key_points = KeyPoints(
        xy=torch.tensor([[[11.0, 6.0], [21.0, 9.0]]], dtype=torch.float32),
        class_id=torch.tensor([0], dtype=torch.long),
        confidence=torch.tensor([[0.875, 0.75]], dtype=torch.float32),
        image_metadata=_native_image_metadata(),
    )

    # when - IN via the declared keypoint kind, then OUT
    with _boundary_on:
        as_legacy = convert_kwargs_to_legacy(
            kwargs={"prediction": (key_points, bbox_component)},
            manifest_description=_manifest(
                inputs={
                    "prediction": _input_with_kinds("keypoint_detection_prediction")
                }
            ),
            block_name="b",
        )["prediction"]
        result = convert_block_result_to_native(
            result={"prediction": as_legacy},
            manifest_description=_manifest_with_outputs(
                outputs={
                    "prediction": _output_with_kinds("keypoint_detection_prediction")
                }
            ),
            block_name="b",
        )

    # then - the tuple shape is rebuilt with matching keypoints
    rebuilt_key_points, rebuilt_bbox = result["prediction"]
    assert isinstance(rebuilt_key_points, KeyPoints)
    assert np.allclose(rebuilt_key_points.xy.cpu().numpy(), key_points.xy.cpu().numpy())
    assert np.allclose(
        rebuilt_key_points.confidence.cpu().numpy(),
        key_points.confidence.cpu().numpy(),
    )
    assert (
        serialise_sv_detections(rebuilt_bbox)["predictions"][0]["keypoints"]
        == serialise_sv_detections(bbox_component)["predictions"][0]["keypoints"]
    )


def test_convert_block_result_flow_control_passes_through() -> None:
    # given
    flow_control = FlowControl(context="$steps.a")

    # when
    with _boundary_on:
        scalar_result = convert_block_result_to_native(
            result=flow_control,
            manifest_description=_manifest_with_outputs(outputs={}),
            block_name="b",
        )
        listed_result = convert_block_result_to_native(
            result=[flow_control, {"output": "text"}],
            manifest_description=_manifest_with_outputs(outputs={}),
            block_name="b",
        )

    # then
    assert scalar_result is flow_control
    assert listed_result[0] is flow_control
    assert listed_result[1] == {"output": "text"}


def test_convert_block_result_walks_nested_lists_and_sniffs_wildcard() -> None:
    # given - List[List[dict]] result (dimensionality +1) with an sv.Detections
    # under a WILDCARD output plus a representation-invariant dict
    native = _native_instance_detections(
        mask=torch.from_numpy(_crop_local_dense_masks())
    )
    as_sv = native_detections_to_sv(detections=native)
    result = [[{"detections": as_sv, "meta": {"key": "value"}}]]

    # when
    with _boundary_on:
        converted = convert_block_result_to_native(
            result=result,
            manifest_description=_manifest_with_outputs(outputs={}),
            block_name="b",
        )

    # then - sv sniffed to native InstanceDetections; dict untouched
    leaf = converted[0][0]
    assert isinstance(leaf["detections"], InstanceDetections)
    assert leaf["meta"] == {"key": "value"}


def test_convert_block_result_wildcard_sniff_rebuilds_keypoint_tuple() -> None:
    # given - sv.Detections carrying all four keypoint payload columns
    bboxes_metadata = [
        {
            DETECTION_ID_KEY: "d0",
            KEYPOINTS_XY_KEY_IN_SV_DETECTIONS: [[11.0, 6.0]],
            KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS: [0.9],
            KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS: [0],
            KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS: ["nose"],
        },
    ]
    bbox_component = Detections(
        xyxy=torch.tensor([[10.0, 5.0, 30.0, 15.0]], dtype=torch.float32),
        class_id=torch.tensor([0], dtype=torch.long),
        confidence=torch.tensor([0.5], dtype=torch.float32),
        image_metadata=_native_image_metadata(),
        bboxes_metadata=bboxes_metadata,
    )
    as_sv = native_detections_to_sv(detections=bbox_component)

    # when
    with _boundary_on:
        converted = convert_block_result_to_native(
            result={"prediction": as_sv},
            manifest_description=_manifest_with_outputs(outputs={}),
            block_name="b",
        )

    # then
    rebuilt_key_points, rebuilt_bbox = converted["prediction"]
    assert isinstance(rebuilt_key_points, KeyPoints)
    assert isinstance(rebuilt_bbox, Detections)


def test_convert_block_result_declared_kind_wrong_type_raises_loudly() -> None:
    # given - a declared classification output receiving sv.Detections
    native = _native_object_detections()
    as_sv = native_detections_to_sv(detections=native)

    # when
    with _boundary_on, pytest.raises(RepresentationBoundaryError) as error:
        _ = convert_block_result_to_native(
            result={"output": as_sv},
            manifest_description=_manifest_with_outputs(
                outputs={"output": _output_with_kinds("classification_prediction")}
            ),
            block_name="my_block",
        )

    # then
    assert "my_block" in str(error.value)
    assert "output" in str(error.value)


def test_convert_block_result_embedding_and_tensor_kinds() -> None:
    # given
    result = {
        "embedding": [0.25, 0.5, 0.75],
        "raw": np.array([[1, 2], [3, 4]], dtype=np.int64),
    }

    # when
    with _boundary_on:
        converted = convert_block_result_to_native(
            result=result,
            manifest_description=_manifest_with_outputs(
                outputs={
                    "embedding": _output_with_kinds("embedding"),
                    "raw": _output_with_kinds("tensor"),
                }
            ),
            block_name="b",
        )

    # then - embedding is float32; tensor preserves dtype
    assert isinstance(converted["embedding"], torch.Tensor)
    assert converted["embedding"].dtype == torch.float32
    assert np.allclose(converted["embedding"].cpu().numpy(), [0.25, 0.5, 0.75])
    assert isinstance(converted["raw"], torch.Tensor)
    assert converted["raw"].dtype == torch.int64


def test_convert_block_result_is_identity_when_boundary_inactive() -> None:
    # given
    result = {"output": object()}
    manifest = _manifest_with_outputs(outputs={})

    # when - flag off
    with _boundary_off:
        flag_off = convert_block_result_to_native(
            result=result, manifest_description=manifest, block_name="b"
        )
    # and - tensor_native mode (validated at construction into the enum)
    manifest_native = ManifestDescription(
        type="ManifestDescription",
        block_type="MyBlock",
        inputs={},
        outputs={},
        tensor_compatibility="tensor_native",
    )
    with _boundary_on:
        native_mode = convert_block_result_to_native(
            result=result, manifest_description=manifest_native, block_name="b"
        )

    # then - the very same object, no copies
    assert flag_off is result
    assert native_mode is result


def test_sv_detections_to_native_takes_rle_mask_column_without_reencoding() -> None:
    # given - an sv carrying the `rle_mask` column exactly as the numpy
    # deserializer produces it (object array of COCO-RLE dicts)
    from inference_models.models.common.rle_utils import (
        coco_rle_masks_to_numpy_mask,
        torch_mask_to_coco_rle,
    )

    dense = _crop_local_dense_masks()
    coco_dicts = [
        torch_mask_to_coco_rle(torch.as_tensor(instance_mask))
        for instance_mask in dense
    ]
    sv_detections = sv.Detections(
        xyxy=np.asarray(
            [[1.0, 2.0, 10.0, 12.0], [3.0, 4.0, 20.0, 22.0]], dtype=np.float32
        ),
        class_id=np.asarray([0, 1]),
        confidence=np.asarray([0.5, 0.75], dtype=np.float32),
        mask=dense.astype(bool),
        data={"rle_mask": np.array(coco_dicts, dtype=object)},
    )

    # when
    converted = sv_detections_to_native(sv_detections=sv_detections)

    # then - the carried RLE wins, verbatim (no re-encode), and decodes back
    assert isinstance(converted, InstanceDetections)
    assert isinstance(converted.mask, InstancesRLEMasks)
    assert converted.mask.masks == [entry["counts"] for entry in coco_dicts]
    assert np.array_equal(
        coco_rle_masks_to_numpy_mask(converted.mask), dense.astype(bool)
    )


def test_sv_detections_to_native_malformed_rle_mask_column_raises_loudly() -> None:
    # given - an entry missing the `counts` key
    sv_detections = sv.Detections(
        xyxy=np.asarray([[1.0, 2.0, 10.0, 12.0]], dtype=np.float32),
        class_id=np.asarray([0]),
        confidence=np.asarray([0.5], dtype=np.float32),
        data={"rle_mask": np.array([{"size": [4, 4]}], dtype=object)},
    )

    # when
    with pytest.raises(ValueError) as error:
        _ = sv_detections_to_native(sv_detections=sv_detections)

    # then
    assert "rle_mask" in str(error.value)
    assert "COCO-RLE" in str(error.value)


@pytest.mark.parametrize(
    "declared_kind, expected_rle",
    [
        ("instance_segmentation_prediction", False),
        ("rle_instance_segmentation_prediction", True),
    ],
)
def test_convert_block_result_empty_sv_declared_mask_kind_stays_instance_shaped(
    declared_kind: str,
    expected_rle: bool,
) -> None:
    # given
    manifest = _manifest_with_outputs(
        outputs={"predictions": _output_with_kinds(declared_kind)}
    )

    # when
    with _boundary_on:
        converted = convert_block_result_to_native(
            result={"predictions": sv.Detections.empty()},
            manifest_description=manifest,
            block_name="my_block",
        )

    # then - empty output under a declared mask-carrying kind keeps the
    # InstanceDetections shape (native empty convention), metadata is None,
    # and the tensor serializer accepts it
    prediction = converted["predictions"]
    assert isinstance(prediction, InstanceDetections)
    assert prediction.bboxes_metadata is None
    assert isinstance(prediction.mask, InstancesRLEMasks) is expected_rle
    serialized = serialise_sv_detections(prediction)
    assert serialized["predictions"] == []


def test_convert_block_result_empty_sv_wildcard_stays_plain_detections() -> None:
    # when - wildcard output: intent unknowable, plain Detections is correct
    with _boundary_on:
        converted = convert_block_result_to_native(
            result={"anything": sv.Detections.empty()},
            manifest_description=_manifest_with_outputs(outputs={}),
            block_name="my_block",
        )

    # then
    prediction = converted["anything"]
    assert isinstance(prediction, Detections)
    assert not isinstance(prediction, InstanceDetections)
    assert prediction.bboxes_metadata is None
    assert serialise_sv_detections(prediction)["predictions"] == []


def test_convert_block_result_wildcard_bare_sv_keypoints_raises() -> None:
    # given - a bare sv.KeyPoints has no native keypoint-prediction equivalent
    bare_key_points = sv.KeyPoints(
        xy=np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    )

    # when
    with _boundary_on, pytest.raises(RepresentationBoundaryError) as error:
        _ = convert_block_result_to_native(
            result={"key_points": bare_key_points},
            manifest_description=_manifest_with_outputs(outputs={}),
            block_name="my_block",
        )

    # then - loud, actionable, mirrors the IN-side bare-KeyPoints rule
    assert "my_block" in str(error.value)
    assert "sv.KeyPoints" in str(error.value) or "detections component" in str(
        error.value
    )
