"""Tests for the tensor-native root-coordinate conversion
(``native_detections_to_root_coordinates``) - especially the mask re-anchoring
and per-box geometry-payload shifts that mirror numpy's
``sv_detections_to_root_coordinates``."""

import numpy as np
import pytest
import supervision as sv
import torch
from pycocotools import mask as mask_utils
from supervision.config import ORIENTED_BOX_COORDINATES

from inference.core.workflows.core_steps.common.serializers import (
    serialise_sv_detections as numpy_serialise_sv_detections,
)
from inference.core.workflows.core_steps.common.serializers_tensor import (
    serialise_sv_detections as tensor_serialise_sv_detections,
)
from inference.core.workflows.core_steps.common.tensor_native import (
    native_detections_to_root_coordinates,
)
from inference.core.workflows.core_steps.common.utils import (
    sv_detections_to_root_coordinates,
)
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAMES_KEY,
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY,
    PARENT_ID_KEY,
    POLYGON_KEY_IN_SV_DETECTIONS,
    PREDICTION_TYPE_KEY,
    ROOT_PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    ROOT_PARENT_ID_KEY,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks

CROP_OFFSET_X, CROP_OFFSET_Y = 100, 50
CROP_H, CROP_W = 40, 60
ROOT_H, ROOT_W = 480, 640


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
        PARENT_ID_KEY: "crop-1",
        PARENT_COORDINATES_KEY: [CROP_OFFSET_X, CROP_OFFSET_Y],
        PARENT_DIMENSIONS_KEY: [ROOT_H, ROOT_W],
        ROOT_PARENT_ID_KEY: "root-image",
        ROOT_PARENT_COORDINATES_KEY: [CROP_OFFSET_X, CROP_OFFSET_Y],
        ROOT_PARENT_DIMENSIONS_KEY: [ROOT_H, ROOT_W],
    }


def _polygons() -> np.ndarray:
    # Uniform 4-point polygons so the numpy path's vectorized
    # `data[POLYGON] + shift` broadcast applies cleanly.
    return np.array(
        [
            [[10, 5], [30, 5], [30, 15], [10, 15]],
            [[40, 20], [55, 20], [55, 35], [40, 35]],
        ],
        dtype=np.int64,
    )


def _native_instance_detections(
    mask: object, with_geometry_payloads: bool = True
) -> InstanceDetections:
    polygons = _polygons()
    bboxes_metadata = []
    for index in range(2):
        entry = {DETECTION_ID_KEY: f"d{index}"}
        if with_geometry_payloads:
            entry[POLYGON_KEY_IN_SV_DETECTIONS] = polygons[index]
            entry[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS] = [
                [11.0, 6.0],
                [29.0, 14.0],
            ]
            entry[ORIENTED_BOX_COORDINATES] = np.array(
                [[10.0, 5.0], [30.0, 5.0], [30.0, 15.0], [10.0, 15.0]]
            )
        bboxes_metadata.append(entry)
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


def _dense_masks_to_rle(masks: np.ndarray) -> InstancesRLEMasks:
    encoded = [
        mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"]
        for mask in masks
    ]
    return InstancesRLEMasks(image_size=(CROP_H, CROP_W), masks=encoded)


def _expected_root_masks() -> np.ndarray:
    anchored = np.zeros((2, ROOT_H, ROOT_W), dtype=bool)
    anchored[
        :,
        CROP_OFFSET_Y : CROP_OFFSET_Y + CROP_H,
        CROP_OFFSET_X : CROP_OFFSET_X + CROP_W,
    ] = _crop_local_dense_masks()
    return anchored


def _decode_native_masks(mask: object) -> np.ndarray:
    if isinstance(mask, InstancesRLEMasks):
        return np.stack(
            [
                mask_utils.decode({"size": list(mask.image_size), "counts": counts})
                for counts in mask.masks
            ]
        ).astype(bool)
    return mask.detach().cpu().numpy().astype(bool)


def test_root_conversion_re_anchors_dense_masks() -> None:
    # given
    detections = _native_instance_detections(
        mask=torch.from_numpy(_crop_local_dense_masks())
    )

    # when
    result = native_detections_to_root_coordinates(prediction=detections)

    # then
    assert tuple(result.mask.shape) == (2, ROOT_H, ROOT_W)
    assert result.mask.dtype == torch.bool
    assert np.array_equal(_decode_native_masks(result.mask), _expected_root_masks())
    assert torch.allclose(
        result.xyxy,
        torch.tensor(
            [[110.0, 55.0, 130.0, 65.0], [140.0, 70.0, 155.0, 85.0]],
            dtype=torch.float32,
        ),
    )
    # source object is untouched (conversion returns a copy)
    assert tuple(detections.mask.shape) == (2, CROP_H, CROP_W)
    assert detections.image_metadata[ROOT_PARENT_COORDINATES_KEY] == [
        CROP_OFFSET_X,
        CROP_OFFSET_Y,
    ]


def test_root_conversion_re_anchors_rle_masks() -> None:
    # given
    detections = _native_instance_detections(
        mask=_dense_masks_to_rle(_crop_local_dense_masks())
    )

    # when
    result = native_detections_to_root_coordinates(prediction=detections)

    # then
    assert isinstance(result.mask, InstancesRLEMasks)
    assert tuple(result.mask.image_size) == (ROOT_H, ROOT_W)
    assert np.array_equal(_decode_native_masks(result.mask), _expected_root_masks())


def test_root_conversion_shifts_geometry_payloads_including_obb() -> None:
    # given
    detections = _native_instance_detections(
        mask=torch.from_numpy(_crop_local_dense_masks())
    )

    # when
    result = native_detections_to_root_coordinates(prediction=detections)

    # then
    entry = result.bboxes_metadata[0]
    shifted_polygon = entry[POLYGON_KEY_IN_SV_DETECTIONS]
    assert isinstance(shifted_polygon, np.ndarray)
    assert np.issubdtype(shifted_polygon.dtype, np.integer)
    assert np.array_equal(
        shifted_polygon,
        _polygons()[0] + np.array([CROP_OFFSET_X, CROP_OFFSET_Y]),
    )
    shifted_keypoints = entry[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS]
    assert isinstance(shifted_keypoints, list)
    assert shifted_keypoints == [
        [11.0 + CROP_OFFSET_X, 6.0 + CROP_OFFSET_Y],
        [29.0 + CROP_OFFSET_X, 14.0 + CROP_OFFSET_Y],
    ]
    # OBB corners shifted like every other geometry payload (the crop side
    # subtracts the crop origin from them, so root conversion adds it back)
    shifted_obb = entry[ORIENTED_BOX_COORDINATES]
    assert isinstance(shifted_obb, np.ndarray)
    assert np.issubdtype(shifted_obb.dtype, np.floating)
    assert np.array_equal(
        shifted_obb,
        np.array([[10.0, 5.0], [30.0, 5.0], [30.0, 15.0], [10.0, 15.0]])
        + np.array([CROP_OFFSET_X, CROP_OFFSET_Y]),
    )
    # source metadata untouched
    assert np.array_equal(
        detections.bboxes_metadata[0][POLYGON_KEY_IN_SV_DETECTIONS], _polygons()[0]
    )
    assert np.array_equal(
        detections.bboxes_metadata[0][ORIENTED_BOX_COORDINATES],
        np.array([[10.0, 5.0], [30.0, 5.0], [30.0, 15.0], [10.0, 15.0]]),
    )


def test_root_conversion_is_noop_without_shift() -> None:
    # given
    metadata = _native_image_metadata()
    metadata[ROOT_PARENT_COORDINATES_KEY] = [0, 0]
    detections = _native_instance_detections(
        mask=torch.from_numpy(_crop_local_dense_masks())
    )
    detections.image_metadata = metadata

    # when
    result = native_detections_to_root_coordinates(prediction=detections)

    # then
    assert result is detections


def test_root_conversion_raises_on_masks_without_root_dimensions() -> None:
    # given
    metadata = _native_image_metadata()
    del metadata[ROOT_PARENT_DIMENSIONS_KEY]
    detections = _native_instance_detections(
        mask=torch.from_numpy(_crop_local_dense_masks())
    )
    detections.image_metadata = metadata

    # when / then
    with pytest.raises(ValueError, match="root"):
        _ = native_detections_to_root_coordinates(prediction=detections)


def test_root_conversion_raises_when_masks_do_not_fit_root_canvas() -> None:
    # given
    metadata = _native_image_metadata()
    metadata[ROOT_PARENT_DIMENSIONS_KEY] = [60, 120]  # crop at (100, 50) cannot fit
    detections = _native_instance_detections(
        mask=torch.from_numpy(_crop_local_dense_masks())
    )
    detections.image_metadata = metadata

    # when / then
    with pytest.raises(ValueError, match="fit"):
        _ = native_detections_to_root_coordinates(prediction=detections)


def test_plain_detections_root_conversion_shifts_polygon_payloads() -> None:
    # given - OD (no masks) prediction carrying a declared polygon payload
    detections = Detections(
        xyxy=torch.tensor([[10.0, 5.0, 30.0, 15.0]], dtype=torch.float32),
        class_id=torch.tensor([0], dtype=torch.long),
        confidence=torch.tensor([0.5], dtype=torch.float32),
        image_metadata=_native_image_metadata(),
        bboxes_metadata=[
            {
                DETECTION_ID_KEY: "d0",
                POLYGON_KEY_IN_SV_DETECTIONS: [[10, 5], [30, 5], [30, 15]],
            }
        ],
    )

    # when
    result = native_detections_to_root_coordinates(prediction=detections)

    # then - list container and integer coordinates preserved
    assert result.bboxes_metadata[0][POLYGON_KEY_IN_SV_DETECTIONS] == [
        [110, 55],
        [130, 55],
        [130, 65],
    ]


def _sv_equivalent_detections() -> sv.Detections:
    polygons = _polygons()
    return sv.Detections(
        xyxy=np.array(
            [[10.0, 5.0, 30.0, 15.0], [40.0, 20.0, 55.0, 35.0]], dtype=np.float32
        ),
        class_id=np.array([0, 1]),
        confidence=np.array([0.5, 0.25], dtype=np.float32),
        mask=_crop_local_dense_masks(),
        data={
            "class_name": np.array(["a", "b"]),
            DETECTION_ID_KEY: np.array(["d0", "d1"]),
            PARENT_ID_KEY: np.array(["crop-1"] * 2),
            PARENT_COORDINATES_KEY: np.array([[CROP_OFFSET_X, CROP_OFFSET_Y]] * 2),
            PARENT_DIMENSIONS_KEY: np.array([[ROOT_H, ROOT_W]] * 2),
            ROOT_PARENT_ID_KEY: np.array(["root-image"] * 2),
            ROOT_PARENT_COORDINATES_KEY: np.array([[CROP_OFFSET_X, CROP_OFFSET_Y]] * 2),
            ROOT_PARENT_DIMENSIONS_KEY: np.array([[ROOT_H, ROOT_W]] * 2),
            IMAGE_DIMENSIONS_KEY: np.array([[CROP_H, CROP_W]] * 2),
            POLYGON_KEY_IN_SV_DETECTIONS: polygons,
            ORIENTED_BOX_COORDINATES: np.array(
                [[[10.0, 5.0], [30.0, 5.0], [30.0, 15.0], [10.0, 15.0]]] * 2
            ),
        },
    )


def test_root_conversion_parity_with_numpy_path() -> None:
    """End-to-end parity: the same crop-local prediction expressed as
    sv.Detections vs native InstanceDetections must root-convert to the same
    boxes, masks, polygons AND the same serialized response dict."""
    # given
    sv_detections = _sv_equivalent_detections()
    native_detections = _native_instance_detections(
        mask=torch.from_numpy(_crop_local_dense_masks()),
        with_geometry_payloads=True,
    )
    # numpy path has no keypoints in this scenario; strip them from the native
    # side too so both serialize the same payload set (OBB stays on BOTH sides -
    # both root conversions shift it now)
    for entry in native_detections.bboxes_metadata:
        del entry[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS]

    # when
    sv_root = sv_detections_to_root_coordinates(detections=sv_detections)
    native_root = native_detections_to_root_coordinates(prediction=native_detections)

    # then - geometry parity
    assert np.allclose(sv_root.xyxy, native_root.xyxy.detach().cpu().numpy())
    assert np.array_equal(sv_root.mask, _decode_native_masks(native_root.mask))
    for index in range(2):
        assert np.array_equal(
            sv_root.data[POLYGON_KEY_IN_SV_DETECTIONS][index],
            native_root.bboxes_metadata[index][POLYGON_KEY_IN_SV_DETECTIONS],
        )
        assert np.array_equal(
            sv_root.data[ORIENTED_BOX_COORDINATES][index],
            native_root.bboxes_metadata[index][ORIENTED_BOX_COORDINATES],
        )

    # then - serialized-output parity
    numpy_serialized = numpy_serialise_sv_detections(sv_root)
    tensor_serialized = tensor_serialise_sv_detections(native_root)
    assert numpy_serialized == tensor_serialized
