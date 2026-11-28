import cv2 as cv
import numpy as np
import pytest
import supervision as sv
import torch
from roboflow_workflows.core_steps.common.tensor_native import instance_mask_to_numpy
from roboflow_workflows.environment import ENABLE_TENSOR_DATA_REPRESENTATION
from roboflow_workflows.execution_engine.core import ExecutionEngine

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import torch_mask_to_coco_rle

SOURCE_SHAPE = (42, 44)
OUTPUT_SIZE = 40
DETECTION_IDS = ["first-instance", "second-instance"]
CLASS_IDS = [1, 2]
CONFIDENCES = [0.9, 0.6]


def _source_masks(case: str) -> np.ndarray:
    source_shape = (42, 32_767) if case == "wide" else SOURCE_SHAPE
    first = np.zeros(source_shape, dtype=np.uint8)
    if case != "empty":
        first[7:18, 4:11] = 1
    if case in {"rectangles", "upscale", "wide"}:
        first[7:18, 25:32] = 1
        if case == "wide":
            first[11:15, -8:-4] = 1
    elif case == "rectangle_and_triangle":
        triangle = np.array([[25, 7], [34, 12], [25, 17]], dtype=np.int32)
        cv.fillPoly(first, [triangle], color=1)
    elif case == "clipped_diagonal":
        first[20, 2] = 1
        first[21, 3] = 1
        assert cv.connectedComponents(first)[0] == 3
    elif case == "holes":
        first[19:37, 15:36] = 1
        first[22:34, 18:33] = 0
        first[25:30, 22:27] = 1
        first[27, 24] = 0
        first[27, 31] = 1
    second = np.zeros(source_shape, dtype=np.uint8)
    second[11:16, 8:15] = 1
    if case == "upscale":
        second[22:27, 24:30] = 1
    return np.stack([first, second]).astype(bool)


def _source_boxes(masks: np.ndarray) -> np.ndarray:
    boxes = []
    for mask in masks:
        y, x = np.nonzero(mask)
        if len(x) == 0:
            boxes.append([4, 7, 10, 17])
        else:
            boxes.append([x.min(), y.min(), x.max(), y.max()])
    return np.array(boxes, dtype=np.float32)


def _make_predictions(
    masks: np.ndarray, boxes: np.ndarray, *, storage: str
) -> sv.Detections | InstanceDetections:
    if not ENABLE_TENSOR_DATA_REPRESENTATION:
        return sv.Detections(
            xyxy=boxes.copy(),
            mask=masks.copy(),
            class_id=np.array(CLASS_IDS),
            confidence=np.array(CONFIDENCES, dtype=np.float32),
            data={"detection_id": np.array(DETECTION_IDS)},
        )

    native_masks = torch.from_numpy(masks.copy())
    if storage == "rle":
        native_masks = InstancesRLEMasks.from_coco_rle_masks(
            image_size=masks.shape[1:],
            masks=[torch_mask_to_coco_rle(mask) for mask in native_masks],
        )

    return InstanceDetections(
        xyxy=torch.from_numpy(boxes.copy()),
        class_id=torch.tensor(CLASS_IDS),
        confidence=torch.tensor(CONFIDENCES),
        mask=native_masks,
        image_size=masks.shape[1:],
        bboxes_metadata=[
            {"detection_id": detection_id} for detection_id in DETECTION_IDS
        ],
    )


@pytest.mark.parametrize(
    "storage", ["dense", "rle"] if ENABLE_TENSOR_DATA_REPRESENTATION else ["dense"]
)
@pytest.mark.parametrize(
    ("case", "offset_x", "offset_y", "scale"),
    [
        pytest.param(case, x, y, 1, id=f"{case}-{mapping}")
        for case in (
            "single",
            "rectangles",
            "rectangle_and_triangle",
            "holes",
            "empty",
            "clipped_diagonal",
        )
        for mapping, x, y in (("identity", 0, 0), ("translation", 3, 2))
    ]
    + [
        pytest.param("upscale", 0, 0, 3, id="rectangles-upscale"),
        pytest.param("wide", 0, 0, 1, id="wide-identity"),
    ],
)
def test_perspective_correction_preserves_mask_geometry(
    case: str, offset_x: int, offset_y: int, scale: int, storage: str
) -> None:
    """Run the real graph against exact mask and bounding-box oracles.

    Args:
        case (str): Geometry of the first instance mask.
        offset_x (int): Left edge of the source rectangle.
        offset_y (int): Top edge of the source rectangle.
        scale (int): Destination pixels per source pixel interval.
        storage (str): Instance-mask representation in tensor-native mode.
    """
    masks = _source_masks(case)
    boxes = _source_boxes(masks)
    predictions = _make_predictions(masks, boxes, storage=storage)
    destination_size = scale * (OUTPUT_SIZE - 1) + 1
    right = offset_x + OUTPUT_SIZE - 1
    bottom = offset_y + OUTPUT_SIZE - 1
    definition = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {
                "type": "WorkflowBatchInput",
                "name": "segments",
                "kind": ["instance_segmentation_prediction"],
                "dimensionality": 1,
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/perspective_correction@v1",
                "name": "correct",
                "images": "$inputs.image",
                "predictions": "$inputs.segments",
                "perspective_polygons": [
                    [offset_x, offset_y],
                    [right, offset_y],
                    [right, bottom],
                    [offset_x, bottom],
                ],
                "transformed_rect_width": destination_size,
                "transformed_rect_height": destination_size,
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "coordinates_system": "own",
                "selector": "$steps.correct.corrected_coordinates",
            }
        ],
    }
    engine = ExecutionEngine.init(workflow_definition=definition, init_parameters={})

    try:
        (result,) = engine.run(
            runtime_parameters={
                "image": np.zeros((*masks.shape[1:], 3), dtype=np.uint8),
                "segments": [predictions],
            }
        )
    finally:
        engine.close()

    corrected = result["predictions"]
    assert type(corrected) is type(predictions)
    assert len(corrected) == len(DETECTION_IDS)
    if isinstance(corrected, InstanceDetections):
        assert type(corrected.mask) is type(predictions.mask)
        if storage == "rle":
            assert corrected.mask.image_size == (destination_size, destination_size)
        corrected_masks = np.stack(
            [
                instance_mask_to_numpy(corrected, index)
                for index in range(len(corrected))
            ]
        )
        detection_ids = [entry["detection_id"] for entry in corrected.bboxes_metadata]
    else:
        corrected_masks = corrected.mask
        detection_ids = corrected.data["detection_id"].tolist()

    if scale == 1:
        expected_masks = masks[:, offset_y : bottom + 1, offset_x : right + 1]
        expected_boxes = boxes - [offset_x, offset_y, offset_x, offset_y]
    else:
        # Independent inverse mapping for the exact integer-scale homography.
        sample = np.rint(np.arange(destination_size) / scale).astype(int)
        expected_masks = masks[:, sample[:, np.newaxis], sample[np.newaxis, :]]
        expected_boxes = _source_boxes(expected_masks)
    np.testing.assert_array_equal(corrected_masks, expected_masks)
    assert corrected_masks.dtype == np.bool_
    np.testing.assert_array_equal(np.asarray(corrected.xyxy), expected_boxes)
    np.testing.assert_array_equal(np.asarray(corrected.class_id), CLASS_IDS)
    np.testing.assert_allclose(np.asarray(corrected.confidence), CONFIDENCES)
    assert detection_ids == DETECTION_IDS
