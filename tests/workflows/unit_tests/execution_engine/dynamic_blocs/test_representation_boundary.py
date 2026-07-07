"""Unit tests for the dynamic-block representation boundary (Step 2 - the IN
direction: native -> legacy). Pure converter/walker tests; engine wiring is
Step 4.

The boundary resolves ``ENABLE_TENSOR_DATA_REPRESENTATION`` once at import time
into ``representation_boundary._TENSOR_REPRESENTATION_ACTIVE``; tests patch that
module constant (the same technique block_assembler tests use for the env flag)
so both CI flag directions exercise every case deterministically.
"""

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
    assert converted.mask.dtype == bool and converted.mask.shape == (2, CROP_H, CROP_W)


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

    # then - both carriers land as the SAME dense (N, H, W) bool ndarray
    assert converted_dense.mask.dtype == bool and converted_rle.mask.dtype == bool
    assert np.array_equal(converted_dense.mask, converted_rle.mask)
    assert np.array_equal(converted_dense.mask, dense)


def test_native_detections_to_sv_pads_keypoint_payloads_like_numpy() -> None:
    # given - 2 keypoints on row 0, none on row 1 (ragged input)
    native = _native_instance_detections(
        mask=torch.from_numpy(_crop_local_dense_masks())
    )
    native.bboxes_metadata[0].update(
        {
            KEYPOINTS_XY_KEY_IN_SV_DETECTIONS: [[11.0, 6.0], [29.0, 14.0]],
            KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS: [0.9, 0.8],
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

    # then - proper padded N-d arrays, the add_inference_keypoints_to_sv_detections
    # convention (ragged object arrays break supervision's is_data_equal)
    assert converted.data[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS].shape == (2, 2, 2)
    assert converted.data[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS].dtype == np.float32
    assert np.array_equal(
        converted.data[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS][0],
        np.array([[11.0, 6.0], [29.0, 14.0]], dtype=np.float32),
    )
    assert np.array_equal(
        converted.data[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS][1],
        np.zeros((2, 2), dtype=np.float32),
    )
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
