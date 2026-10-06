"""Dynamic Crop must describe the same coordinate frame as its forwarded geometry.

Run with ENABLE_TENSOR_DATA_REPRESENTATION=true to exercise native predictions
through the real output constructor, including its normal conversion gate.
"""

from copy import deepcopy
from dataclasses import is_dataclass

import numpy as np
import pytest
import supervision as sv
import torch
from pycocotools import mask as mask_utils
from roboflow_workflows.core_steps.common.tensor_native import (
    HOST_MIRROR_KEYS,
    attach_native_detection_metadata,
)
from roboflow_workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_sv_detections,
)
from roboflow_workflows.core_steps.transformations.detection_offset.v1 import (
    offset_detections as offset_numpy,
)
from roboflow_workflows.core_steps.transformations.detection_offset.v1_tensor import (
    offset_detections as offset_tensor,
)
from roboflow_workflows.core_steps.transformations.dynamic_crop.v1 import (
    crop_image as crop_numpy,
)
from roboflow_workflows.core_steps.transformations.dynamic_crop.v1_tensor import (
    crop_image as crop_tensor,
)
from roboflow_workflows.environment import (
    ENABLE_TENSOR_DATA_REPRESENTATION,
    WORKFLOWS_IMAGE_TENSOR_DEVICE,
)
from roboflow_workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY,
    PARENT_ID_KEY,
    POLYGON_KEY_IN_SV_DETECTIONS,
    ROOT_PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    ROOT_PARENT_ID_KEY,
    SCALING_RELATIVE_TO_PARENT_KEY,
    SCALING_RELATIVE_TO_ROOT_PARENT_KEY,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.v1.executor.output_constructor import (
    _prepare_data_piece_for_output,
)
from supervision.config import ORIENTED_BOX_COORDINATES

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks


def _image() -> WorkflowImageData:
    pixels = np.arange(480 * 640 * 3, dtype=np.uint32).reshape(480, 640, 3)
    result = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="root-image"),
        numpy_image=pixels.astype(np.uint8),
    )
    return result


def _predictions(image, *, backend, boxes, masks=None):
    if backend == "tensor" and not ENABLE_TENSOR_DATA_REPRESENTATION:
        pytest.skip("Native output conversion requires tensor representation mode")

    identifiers = [f"detection-{index}" for index in range(len(boxes))]
    height, width = image._read_shape_without_materialization()
    if backend == "numpy":
        result = sv.Detections(
            xyxy=np.asarray(boxes, dtype=np.float32),
            class_id=np.arange(len(boxes)),
            confidence=np.full(len(boxes), 0.75, dtype=np.float32),
            mask=masks,
            data={
                DETECTION_ID_KEY: np.array(identifiers),
                "class_name": np.array(["object"] * len(boxes)),
                IMAGE_DIMENSIONS_KEY: np.array([[height, width]] * len(boxes)),
            },
        )
        result = attach_parents_coordinates_to_sv_detections(result, image)
        return result

    arguments = {
        "xyxy": torch.tensor(boxes, dtype=torch.float32, device="cpu"),
        "class_id": torch.arange(len(boxes), device="cpu"),
        "confidence": torch.full((len(boxes),), 0.75, device="cpu"),
        "bboxes_metadata": [{DETECTION_ID_KEY: value} for value in identifiers],
    }
    if masks is None:
        result = Detections(**arguments)
    else:
        mask = (
            masks if isinstance(masks, InstancesRLEMasks) else torch.from_numpy(masks)
        )
        result = InstanceDetections(**arguments, mask=mask)

    result = attach_native_detection_metadata(
        detections=result,
        image=image,
        class_names={index: "object" for index in range(len(boxes))},
        prediction_type=(
            "instance-segmentation" if masks is not None else "object-detection"
        ),
    )
    return result


def _crop(image, *, predictions, backend):
    if backend == "numpy":
        result = crop_numpy(
            image=image,
            detections=predictions,
            mask_opacity=0,
            background_color=(0, 0, 0),
        )
    else:
        result = crop_tensor(
            image=image,
            predictions=predictions,
            mask_opacity=0,
            background_color=(0, 0, 0),
        )
    return result


def _output(predictions, *, root=True):
    result = _prepare_data_piece_for_output(
        data_piece=predictions,
        resolve_output_futures=True,
        convert_to_parent_coordinates=root,
    )
    return result


def _numpy(value):
    result = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
    return result


def _metadata(predictions, key):
    result = (
        predictions.data[key][0]
        if isinstance(predictions, sv.Detections)
        else predictions.image_metadata[key]
    )
    return result


def _assert_lineage(
    predictions, *, dimensions, origin, parent_dimensions, parent_origin
):
    expected = {
        IMAGE_DIMENSIONS_KEY: dimensions,
        PARENT_ID_KEY: "detection-0",
        PARENT_COORDINATES_KEY: parent_origin,
        PARENT_DIMENSIONS_KEY: parent_dimensions,
        ROOT_PARENT_ID_KEY: "root-image",
        ROOT_PARENT_COORDINATES_KEY: origin,
        ROOT_PARENT_DIMENSIONS_KEY: [480, 640],
    }
    for key, value in expected.items():
        np.testing.assert_array_equal(_metadata(predictions, key), value)


def _assert_unchanged(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.device == expected.device
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_unchanged(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected):
            _assert_unchanged(actual_value, expected_value)
    elif is_dataclass(expected):
        assert type(actual) is type(expected)
        _assert_unchanged(vars(actual), vars(expected))
    else:
        assert actual == expected


@pytest.mark.parametrize("backend", ["numpy", "tensor"])
@pytest.mark.parametrize("box", [[100, 50, 200, 150], [0, 0, 100, 100]])
def test_forwarded_crop_predictions_have_crop_lineage_and_root_output(backend, box):
    """Check nonzero and zero-origin crops through the actual output conversion gate.

    Args:
        backend: Prediction representation to exercise.
        box: Explicit source bounding box and expected root coordinates.
    """
    image = _image()
    predictions = _predictions(image, backend=backend, boxes=[box])
    before = deepcopy(predictions)

    result = _crop(image, predictions=predictions, backend=backend)[0]

    np.testing.assert_array_equal(
        _numpy(_output(result["predictions"], root=False).xyxy), [[0, 0, 100, 100]]
    )
    np.testing.assert_array_equal(
        result["crops"].numpy_image, image.numpy_image[box[1] : box[3], box[0] : box[2]]
    )
    _assert_lineage(
        result["predictions"],
        dimensions=[100, 100],
        origin=box[:2],
        parent_dimensions=[480, 640],
        parent_origin=box[:2],
    )
    np.testing.assert_array_equal(_numpy(_output(result["predictions"]).xyxy), [box])
    _assert_unchanged(predictions, before)


@pytest.mark.parametrize("backend", ["numpy", "tensor"])
@pytest.mark.parametrize("use_percentage", [False, True])
def test_forwarded_crop_offset_clips_at_crop_dimensions(backend, use_percentage):
    """Prevent expansion into pixels outside the generated crop.

    Args:
        backend: Prediction representation to exercise.
        use_percentage: Whether the offset is a percentage or pixels.
    """
    image = _image()
    predictions = _predictions(image, backend=backend, boxes=[[100, 50, 200, 150]])
    forwarded = _crop(image, predictions=predictions, backend=backend)[0]["predictions"]
    before = deepcopy(forwarded)
    offset = offset_numpy if backend == "numpy" else offset_tensor

    result = offset(
        forwarded, offset_width=20, offset_height=40, use_percentage=use_percentage
    )

    # The original box fills the crop, so either expansion must clip on all edges.
    np.testing.assert_array_equal(_numpy(result.xyxy), [[0, 0, 100, 100]])
    np.testing.assert_array_equal(_numpy(_output(result).xyxy), [[100, 50, 200, 150]])
    _assert_unchanged(forwarded, before)


@pytest.mark.parametrize("backend", ["numpy", "tensor"])
def test_nested_crop_uses_accumulated_origin_and_immediate_parent_dimensions(backend):
    """Keep both crop origins when forwarding a fresh second-stage prediction.

    Args:
        backend: Prediction representation to exercise.
    """
    image = _image()
    first = _crop(
        image,
        predictions=_predictions(image, backend=backend, boxes=[[100, 50, 200, 150]]),
        backend=backend,
    )[0]
    fresh = _predictions(first["crops"], backend=backend, boxes=[[20, 10, 70, 60]])
    before = deepcopy(fresh)

    second = _crop(first["crops"], predictions=fresh, backend=backend)[0]

    np.testing.assert_array_equal(_numpy(second["predictions"].xyxy), [[0, 0, 50, 50]])
    _assert_lineage(
        second["predictions"],
        dimensions=[50, 50],
        origin=[120, 60],
        parent_dimensions=[100, 100],
        parent_origin=[20, 10],
    )
    np.testing.assert_array_equal(
        _numpy(_output(second["predictions"]).xyxy), [[120, 60, 170, 110]]
    )
    _assert_unchanged(fresh, before)


@pytest.mark.parametrize("backend", ["numpy", "tensor"])
def test_each_crop_has_its_own_lineage_and_actual_image_dimensions(backend):
    """Use actual crop size when a box extends past the source image's right edge.

    Args:
        backend: Prediction representation to exercise.
    """
    image = _image()
    boxes = [[100, 50, 200, 150], [620, 450, 700, 530]]
    predictions = _predictions(image, backend=backend, boxes=boxes)
    before = deepcopy(predictions)

    results = _crop(image, predictions=predictions, backend=backend)

    assert len(results) == 2
    for index, (result, dimensions) in enumerate(zip(results, [[100, 100], [30, 20]])):
        forwarded = result["predictions"]
        np.testing.assert_array_equal(
            _metadata(forwarded, IMAGE_DIMENSIONS_KEY), dimensions
        )
        assert result["crops"].numpy_image.shape[:2] == tuple(dimensions)
        assert _metadata(forwarded, PARENT_ID_KEY) == f"detection-{index}"
        np.testing.assert_array_equal(
            _metadata(forwarded, ROOT_PARENT_COORDINATES_KEY), boxes[index][:2]
        )
        np.testing.assert_array_equal(_numpy(forwarded.class_id), [index])
        # Cropping changes the reference frame; it does not clip the prediction.
        np.testing.assert_array_equal(_numpy(_output(forwarded).xyxy), [boxes[index]])
    _assert_unchanged(predictions, before)


@pytest.mark.parametrize(
    "backend,mask_format", [("numpy", "dense"), ("tensor", "dense"), ("tensor", "rle")]
)
def test_crop_payloads_roundtrip_with_masks_keypoints_polygons_and_obb(
    backend, mask_format
):
    """Keep every geometry payload aligned with crop and root coordinates.

    Args:
        backend: Prediction representation to exercise.
        mask_format: Dense or native run-length encoded mask storage.
    """
    image = _image()
    masks = np.zeros((1, 480, 640), dtype=bool)
    masks[0, 60:90, 120:160] = True
    source_masks = masks
    if mask_format == "rle":
        source_masks = InstancesRLEMasks(
            image_size=(480, 640),
            masks=[
                mask_utils.encode(np.asfortranarray(masks[0].astype(np.uint8)))[
                    "counts"
                ]
            ],
        )
    predictions = _predictions(
        image, backend=backend, boxes=[[100, 50, 200, 150]], masks=source_masks
    )
    payloads = {
        KEYPOINTS_XY_KEY_IN_SV_DETECTIONS: np.array([[125, 65], [155, 85]]),
        POLYGON_KEY_IN_SV_DETECTIONS: np.array(
            [[120, 60], [160, 60], [160, 90], [120, 90]]
        ),
        ORIENTED_BOX_COORDINATES: np.array(
            [[120.0, 60.0], [160.0, 60.0], [160.0, 90.0], [120.0, 90.0]]
        ),
    }
    for key, value in payloads.items():
        if backend == "numpy":
            predictions[key] = value[np.newaxis]
        else:
            predictions.bboxes_metadata[0][key] = (
                value.tolist()
                if key == KEYPOINTS_XY_KEY_IN_SV_DETECTIONS
                else value.copy()
            )
    before = deepcopy(predictions)

    forwarded = _crop(image, predictions=predictions, backend=backend)[0]["predictions"]
    root = _output(forwarded)

    for key, expected in payloads.items():
        local_value = (
            forwarded.data[key][0]
            if backend == "numpy"
            else forwarded.bboxes_metadata[0][key]
        )
        root_value = (
            root.data[key][0] if backend == "numpy" else root.bboxes_metadata[0][key]
        )
        np.testing.assert_array_equal(local_value, expected - [100, 50])
        np.testing.assert_array_equal(root_value, expected)
    if mask_format == "rle":
        assert isinstance(forwarded.mask, InstancesRLEMasks)
        assert tuple(forwarded.mask.image_size) == (100, 100)
        assert tuple(root.mask.image_size) == (480, 640)
        actual_mask = mask_utils.decode(
            {"size": list(root.mask.image_size), "counts": root.mask.masks[0]}
        )[np.newaxis]
    else:
        np.testing.assert_array_equal(_numpy(forwarded.mask), masks[:, 50:150, 100:200])
        actual_mask = _numpy(root.mask)
    np.testing.assert_array_equal(actual_mask, masks)
    _assert_unchanged(predictions, before)


def test_keypoint_tuple_updates_both_carriers_without_changing_source():
    """Refresh keypoint and bbox lineage together for native tuple outputs."""
    image = _image()
    detections = _predictions(image, backend="tensor", boxes=[[100, 50, 200, 150]])
    keypoints = KeyPoints(
        xy=torch.tensor([[[125.0, 65.0], [155.0, 85.0]]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([[0.9, 0.8]]),
        image_metadata={
            **deepcopy(detections.image_metadata),
            "keypoint_extra": "keep",
        },
        key_points_metadata=[{"labels": ["left", "right"]}],
    )
    predictions = (keypoints, detections)
    before = deepcopy(predictions)

    forwarded = _crop(image, predictions=predictions, backend="tensor")[0][
        "predictions"
    ]

    for carrier in forwarded:
        _assert_lineage(
            carrier,
            dimensions=[100, 100],
            origin=[100, 50],
            parent_dimensions=[480, 640],
            parent_origin=[100, 50],
        )
    np.testing.assert_array_equal(_numpy(forwarded[0].xy), [[[25, 15], [55, 35]]])
    root_keypoints, root_detections = _output(forwarded)
    np.testing.assert_array_equal(_numpy(root_keypoints.xy), [[[125, 65], [155, 85]]])
    np.testing.assert_array_equal(_numpy(root_detections.xyxy), [[100, 50, 200, 150]])
    assert forwarded[0].image_metadata["keypoint_extra"] == "keep"
    assert forwarded[0].key_points_metadata == [{"labels": ["left", "right"]}]
    _assert_unchanged(predictions, before)


@pytest.mark.parametrize("backend", ["numpy", "tensor"])
def test_crop_preserves_unrelated_metadata_and_scale_fields(backend):
    """Restrict crop metadata updates to image dimensions and lineage.

    Args:
        backend: Prediction representation to exercise.
    """
    image = _image()
    predictions = _predictions(image, backend=backend, boxes=[[100, 50, 200, 150]])
    values = {
        "inference_id": "inference-123",
        "custom_text": "preserve me",
        SCALING_RELATIVE_TO_PARENT_KEY: 0.5,
        SCALING_RELATIVE_TO_ROOT_PARENT_KEY: 0.25,
    }
    if backend == "numpy":
        for key, value in values.items():
            predictions[key] = np.array([value])
        predictions.metadata["custom_image"] = {"label": "original"}
    else:
        predictions.image_metadata.update(values)
        predictions.bboxes_metadata[0]["custom_box"] = {"label": "original"}
    before = deepcopy(predictions)

    forwarded = _crop(image, predictions=predictions, backend=backend)[0]["predictions"]

    for key, value in values.items():
        np.testing.assert_array_equal(_metadata(forwarded, key), value)
    np.testing.assert_array_equal(_numpy(forwarded.class_id), [0])
    np.testing.assert_array_equal(_numpy(forwarded.confidence), [0.75])
    if backend == "numpy":
        assert forwarded.data[DETECTION_ID_KEY][0] == "detection-0"
        assert forwarded.data["class_name"][0] == "object"
        assert forwarded.metadata["custom_image"] == {"label": "original"}
    else:
        assert forwarded.bboxes_metadata[0][DETECTION_ID_KEY] == "detection-0"
        assert forwarded.bboxes_metadata[0]["custom_box"] == {"label": "original"}
        assert forwarded.image_metadata["class_names"] == {0: "object"}
        assert all(key not in forwarded.bboxes_metadata[0] for key in HOST_MIRROR_KEYS)
    _assert_unchanged(predictions, before)


@pytest.mark.skipif(
    str(WORKFLOWS_IMAGE_TENSOR_DEVICE) not in {"cpu", "None"},
    reason="CPU-only ownership regression",
)
def test_cpu_tensor_image_stays_tensor_backed_and_preserves_source():
    """Avoid materializing or mutating the source while refreshing crop metadata."""
    pixels = (
        torch.arange(3 * 480 * 640, device="cpu").reshape(3, 480, 640).to(torch.uint8)
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="root-image"), tensor_image=pixels
    )
    predictions = _predictions(image, backend="tensor", boxes=[[100, 50, 200, 150]])
    before = deepcopy(predictions)
    original_pixels = pixels.clone()

    result = _crop(image, predictions=predictions, backend="tensor")[0]

    assert image.is_tensor_materialised()
    assert image._numpy_image is None
    assert result["crops"].is_tensor_materialised()
    assert result["crops"]._numpy_image is None
    assert result["crops"].tensor_image.device.type == "cpu"
    assert image.tensor_image.data_ptr() == pixels.data_ptr()
    assert torch.equal(result["crops"].tensor_image, pixels[:, 50:150, 100:200])
    np.testing.assert_array_equal(
        _numpy(_output(result["predictions"]).xyxy), [[100, 50, 200, 150]]
    )
    assert torch.equal(pixels, original_pixels)
    _assert_unchanged(predictions, before)
