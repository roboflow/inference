"""Tests for the tensor-native root-coordinate conversion
(``native_detections_to_root_coordinates``) - especially the mask re-anchoring
and per-box geometry-payload shifts that mirror numpy's
``sv_detections_to_root_coordinates``."""

import numpy as np
import pytest
import supervision as sv
import torch
from pycocotools import mask as mask_utils
from roboflow_workflows.core_steps.common.serializers import (
    serialise_sv_detections as numpy_serialise_sv_detections,
)
from roboflow_workflows.core_steps.common.serializers_tensor import (
    serialise_sv_detections as tensor_serialise_sv_detections,
)
from roboflow_workflows.core_steps.common.tensor_native import (
    native_detections_to_root_coordinates,
)
from roboflow_workflows.core_steps.common.utils import sv_detections_to_root_coordinates
from roboflow_workflows.execution_engine.constants import (
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
    SCALING_RELATIVE_TO_PARENT_KEY,
    SCALING_RELATIVE_TO_ROOT_PARENT_KEY,
)
from supervision.config import ORIENTED_BOX_COORDINATES

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
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


def test_root_conversion_with_zero_offset_still_reanchors_crop_predictions() -> None:
    # given - a crop taken at (0, 0): the offset is zero, but the dimensions are
    # crop-sized and the parent is NOT the root. A zero offset alone must not be
    # treated as proof of root anchoring (PR review): masks still need
    # re-embedding onto the root canvas and the metadata still needs the root
    # rewrite.
    metadata = _native_image_metadata()
    metadata[PARENT_COORDINATES_KEY] = [0, 0]
    metadata[ROOT_PARENT_COORDINATES_KEY] = [0, 0]
    detections = _native_instance_detections(
        mask=torch.from_numpy(_crop_local_dense_masks())
    )
    detections.image_metadata = metadata

    # when
    result = native_detections_to_root_coordinates(prediction=detections)

    # then - boxes keep their values (the shift IS zero) but the masks are
    # re-anchored onto the root canvas and the lineage collapses to the root
    assert result is not detections
    assert torch.equal(result.xyxy, detections.xyxy)
    assert tuple(result.mask.shape[-2:]) == (ROOT_H, ROOT_W)
    assert list(result.image_metadata[IMAGE_DIMENSIONS_KEY]) == [ROOT_H, ROOT_W]
    assert result.image_metadata[PARENT_ID_KEY] == "root-image"


def test_root_conversion_is_noop_when_already_root_anchored() -> None:
    # given - genuinely root-anchored: zero offset AND root-sized dimensions AND
    # the parent IS the root
    metadata = _native_image_metadata()
    metadata[IMAGE_DIMENSIONS_KEY] = [ROOT_H, ROOT_W]
    metadata[PARENT_ID_KEY] = "root-image"
    metadata[PARENT_COORDINATES_KEY] = [0, 0]
    metadata[ROOT_PARENT_COORDINATES_KEY] = [0, 0]
    detections = _native_instance_detections(
        mask=torch.from_numpy(_crop_local_dense_masks())
    )
    detections.image_metadata = metadata

    # when
    result = native_detections_to_root_coordinates(prediction=detections)

    # then - identity: the input object is returned untouched
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


@pytest.mark.parametrize("encoding", ["dense", "rle"])
def test_root_scale_restores_masks_boxes_and_per_box_geometry(encoding: str) -> None:
    local_masks = _crop_local_dense_masks()
    mask = (
        torch.from_numpy(local_masks)
        if encoding == "dense"
        else _dense_masks_to_rle(local_masks)
    )
    mask_at_root_scale = local_masks.repeat(2, axis=1).repeat(2, axis=2)
    detections = _native_instance_detections(mask=mask)
    detections.image_metadata[SCALING_RELATIVE_TO_PARENT_KEY] = 0.5
    detections.image_metadata[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] = 0.5
    expected_mask = np.zeros((2, ROOT_H, ROOT_W), dtype=bool)
    expected_mask[:, 50:130, 100:220] = mask_at_root_scale

    result = native_detections_to_root_coordinates(detections)

    torch.testing.assert_close(
        result.xyxy,
        torch.tensor([[120.0, 60.0, 160.0, 80.0], [180.0, 90.0, 210.0, 120.0]]),
    )
    assert isinstance(result.mask, type(mask))
    np.testing.assert_array_equal(_decode_native_masks(result.mask), expected_mask)
    metadata = result.bboxes_metadata[0]
    expected_corners = [[120, 60], [160, 60], [160, 80], [120, 80]]
    np.testing.assert_array_equal(
        metadata[POLYGON_KEY_IN_SV_DETECTIONS], expected_corners
    )
    np.testing.assert_array_equal(metadata[ORIENTED_BOX_COORDINATES], expected_corners)
    assert metadata[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS] == [[122.0, 62.0], [158.0, 78.0]]
    assert result.image_metadata[SCALING_RELATIVE_TO_PARENT_KEY] == 1.0
    assert result.image_metadata[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] == 1.0
    assert detections.image_metadata[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] == 0.5
    assert detections.bboxes_metadata[0][KEYPOINTS_XY_KEY_IN_SV_DETECTIONS] == [
        [11.0, 6.0],
        [29.0, 14.0],
    ]
    assert native_detections_to_root_coordinates(result) is result


def test_root_scale_transforms_keypoint_tuple_and_covariance() -> None:
    metadata = _native_image_metadata()
    metadata[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] = 0.2
    box = Detections(
        xyxy=torch.tensor([[10.0, 5.0, 30.0, 15.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.5]),
        image_metadata=metadata,
        bboxes_metadata=[
            {KEYPOINTS_XY_KEY_IN_SV_DETECTIONS: [[2.3, 3.7], [29.0, 14.0]]}
        ],
    )
    points_metadata = dict(metadata)
    del points_metadata[SCALING_RELATIVE_TO_ROOT_PARENT_KEY]
    points = KeyPoints(
        xy=torch.tensor([[[2.3, 3.7], [29.0, 14.0]]]),
        class_id=box.class_id,
        confidence=torch.tensor([[0.9, 0.8]]),
        image_metadata=points_metadata,
        covariance=torch.tensor([[[[1.0, 0.0], [0.0, 2.0]], [[3.0, 0.0], [0.0, 4.0]]]]),
        detection_confidence=box.confidence,
    )

    rooted_points, rooted_box = native_detections_to_root_coordinates((points, box))

    torch.testing.assert_close(
        rooted_points.xy, torch.tensor([[[112.0, 68.0], [245.0, 120.0]]])
    )
    torch.testing.assert_close(
        rooted_box.xyxy, torch.tensor([[150.0, 75.0, 250.0, 125.0]])
    )
    torch.testing.assert_close(rooted_points.covariance, points.covariance * 25)
    assert (
        rooted_box.bboxes_metadata[0][KEYPOINTS_XY_KEY_IN_SV_DETECTIONS]
        == rooted_points.xy[0].tolist()
    )
    assert rooted_points.confidence is points.confidence
    assert rooted_points.detection_confidence is points.detection_confidence
    torch.testing.assert_close(points.xy, torch.tensor([[[2.3, 3.7], [29.0, 14.0]]]))


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_root_scale_rejects_invalid_factors(scale: float) -> None:
    detections = _native_instance_detections(mask=None)
    detections.image_metadata[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] = scale

    with pytest.raises(ValueError, match="root scale"):
        native_detections_to_root_coordinates(detections)


def test_root_scale_matches_numpy_rounding_for_box_and_polygon() -> None:
    metadata = _native_image_metadata()
    metadata[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] = 0.2
    corners = np.array(
        [[2.3, 2.3], [3.7, 2.3], [3.7, 3.7], [2.3, 3.7]], dtype=np.float32
    )
    detections = Detections(
        xyxy=torch.tensor([[2.3, 2.3, 3.7, 3.7]], dtype=torch.float32),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.5]),
        image_metadata=metadata,
        bboxes_metadata=[{POLYGON_KEY_IN_SV_DETECTIONS: corners}],
    )

    result = native_detections_to_root_coordinates(detections)

    torch.testing.assert_close(result.xyxy, torch.tensor([[111.0, 61.0, 119.0, 69.0]]))
    np.testing.assert_array_equal(
        result.bboxes_metadata[0][POLYGON_KEY_IN_SV_DETECTIONS],
        [[111.0, 61.0], [119.0, 61.0], [119.0, 69.0], [111.0, 69.0]],
    )


def test_root_scale_preserves_numpy_nearest_neighbor_sampling() -> None:
    metadata = _native_image_metadata()
    metadata[IMAGE_DIMENSIONS_KEY] = [4, 14]
    metadata[ROOT_PARENT_COORDINATES_KEY] = [1, 2]
    metadata[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] = 14 / 18
    local_mask = np.zeros((1, 4, 14), dtype=bool)
    local_mask[:, :, 6] = True
    local_box = [[6.0, 1.0, 7.0, 3.0]]
    legacy = sv.Detections(
        xyxy=np.array(local_box),
        mask=local_mask,
        data={key: np.array([value]) for key, value in metadata.items()},
    )
    encoded_mask = InstancesRLEMasks.from_coco_rle_masks(
        image_size=(4, 14),
        masks=[mask_utils.encode(np.asfortranarray(local_mask[0], dtype=np.uint8))],
    )
    expected = sv_detections_to_root_coordinates(legacy).mask

    for mask in (torch.from_numpy(local_mask), encoded_mask):
        detections = InstanceDetections(
            xyxy=torch.tensor(local_box),
            class_id=torch.tensor([0]),
            confidence=torch.tensor([0.5]),
            mask=mask,
            image_metadata=metadata,
        )

        result = native_detections_to_root_coordinates(detections)

        np.testing.assert_array_equal(_decode_native_masks(result.mask), expected)


@pytest.mark.parametrize("encoding", ["dense", "rle"])
@pytest.mark.parametrize(
    ("image_hw", "frame_hw", "scale"),
    [((4, 4), (4, 4), 0.5), ((5, 8), (4, 6), 0.5), ((5, 5), (5, 5), 5 / 3)],
)
def test_root_scale_normalizes_the_encoded_mask_grid(
    encoding: str, image_hw: tuple[int, int], frame_hw: tuple[int, int], scale: float
) -> None:
    encoded_mask = np.array(
        [[False, False, True], [False, True, True], [True, False, True]]
    )
    if encoding == "dense":
        mask = torch.from_numpy(encoded_mask[None, ...])
    else:
        mask = InstancesRLEMasks.from_coco_rle_masks(
            image_size=image_hw,
            mask_size=encoded_mask.shape,
            masks=[mask_utils.encode(np.asfortranarray(encoded_mask, dtype=np.uint8))],
        )
    metadata = _native_image_metadata()
    metadata[IMAGE_DIMENSIONS_KEY] = list(image_hw)
    metadata[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] = scale
    detections = InstanceDetections(
        xyxy=torch.tensor([[1.0, 0.0, 3.0, 3.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.5]),
        mask=mask,
        image_size=image_hw,
        mask_frame_size=frame_hw,
        image_metadata=metadata,
    )
    legacy = detections.to_supervision()
    for key, value in metadata.items():
        legacy[key] = np.array([value])
    expected = sv_detections_to_root_coordinates(legacy)

    result = native_detections_to_root_coordinates(detections)

    np.testing.assert_array_equal(result.xyxy.numpy(), expected.xyxy)
    np.testing.assert_array_equal(_decode_native_masks(result.mask), expected.mask)


@pytest.mark.parametrize("encoding", ["dense", "rle"])
def test_root_scale_keeps_empty_mask_batches_root_sized(encoding: str) -> None:
    metadata = _native_image_metadata()
    metadata[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] = 0.5
    if encoding == "dense":
        mask = torch.empty((0, 0, 0), dtype=torch.bool)
    else:
        mask = InstancesRLEMasks(image_size=(CROP_H, CROP_W), masks=[])
    detections = InstanceDetections(
        xyxy=torch.empty((0, 4)),
        class_id=torch.empty((0,), dtype=torch.long),
        confidence=torch.empty((0,)),
        mask=mask,
        image_metadata=metadata,
    )

    result = native_detections_to_root_coordinates(detections)

    assert result.xyxy.shape == (0, 4)
    if encoding == "dense":
        assert result.mask.shape == (0, ROOT_H, ROOT_W)
    else:
        assert result.mask.image_size == (ROOT_H, ROOT_W)
        assert result.mask.masks == []


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_root_scale_matches_box_and_polygon_frame_rounding(dtype: torch.dtype) -> None:
    metadata = _native_image_metadata()
    metadata[IMAGE_DIMENSIONS_KEY] = [4, 4]
    metadata[ROOT_PARENT_COORDINATES_KEY] = [1, 2]
    metadata[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] = 1.6
    box = [[0, 0, 3, 3]]
    polygon = np.array([[0, 0], [3, 0], [3, 3], [0, 3]])
    detections = InstanceDetections(
        xyxy=torch.tensor(box, dtype=dtype),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.5]),
        mask=torch.ones((1, 3, 3), dtype=torch.bool),
        image_size=(4, 4),
        image_metadata=metadata,
        bboxes_metadata=[{POLYGON_KEY_IN_SV_DETECTIONS: polygon}],
    )
    legacy = detections.to_supervision()
    for key, value in metadata.items():
        legacy[key] = np.array([value])
    expected = sv_detections_to_root_coordinates(legacy)

    result = native_detections_to_root_coordinates(detections)

    np.testing.assert_array_equal(result.xyxy.numpy(), expected.xyxy)
    restored = result.bboxes_metadata[0][POLYGON_KEY_IN_SV_DETECTIONS]
    np.testing.assert_array_equal(restored[0], expected.xyxy[0, :2])
    np.testing.assert_array_equal(restored[2], expected.xyxy[0, 2:])
