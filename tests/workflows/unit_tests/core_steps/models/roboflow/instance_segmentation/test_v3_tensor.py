from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.types import InstancesRLEMasks

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.tensor_prediction_metadata import (
    CLASS_NAMES_KEY,
    MODEL_ID_KEY,
)
from inference.core.workflows.core_steps.models.roboflow.instance_segmentation.v3_tensor import (
    RoboflowInstanceSegmentationModelBlockV3,
)
from inference.core.workflows.execution_engine.constants import (
    INFERENCE_ID_KEY,
    PREDICTION_TYPE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    OriginCoordinatesSystem,
    WorkflowImageData,
)


def _make_image(parent_id: str = "p", h: int = 64, w: int = 64) -> WorkflowImageData:
    numpy_image = np.zeros((h, w, 3), dtype=np.uint8)
    parent_metadata = ImageParentMetadata(
        parent_id=parent_id,
        origin_coordinates=OriginCoordinatesSystem(
            left_top_x=0,
            left_top_y=0,
            origin_width=w,
            origin_height=h,
        ),
    )
    return WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=numpy_image,
    )


def _make_instance_detections_with_rle(
    n: int = 2,
    image_size: tuple = (64, 64),
    image_metadata: Optional[dict] = None,
) -> InstanceDetections:
    rle_masks = InstancesRLEMasks(
        image_size=image_size,
        masks=[b"fake-rle-counts-" + str(i).encode() for i in range(n)],
    )
    return InstanceDetections(
        xyxy=torch.zeros((n, 4), dtype=torch.float32),
        class_id=torch.arange(n, dtype=torch.int64),
        confidence=torch.full((n,), 0.5, dtype=torch.float32),
        mask=rle_masks,
        image_metadata=image_metadata,
    )


def _make_instance_detections_with_dense_mask(
    n: int = 2, image_size: tuple = (16, 16)
) -> InstanceDetections:
    h, w = image_size
    return InstanceDetections(
        xyxy=torch.zeros((n, 4), dtype=torch.float32),
        class_id=torch.arange(n, dtype=torch.int64),
        confidence=torch.full((n,), 0.5, dtype=torch.float32),
        mask=torch.zeros((n, h, w), dtype=torch.bool),
    )


def _make_block(
    model_manager: MagicMock,
    *,
    step_execution_mode: StepExecutionMode = StepExecutionMode.LOCAL,
) -> RoboflowInstanceSegmentationModelBlockV3:
    return RoboflowInstanceSegmentationModelBlockV3(
        model_manager=model_manager,
        api_key="test-key",
        step_execution_mode=step_execution_mode,
    )


def _local_kwargs(**overrides):
    base = {
        "model_id": "m/1",
        "class_agnostic_nms": False,
        "class_filter": None,
        "confidence": 0.5,
        "iou_threshold": 0.3,
        "max_detections": 300,
        "max_candidates": 3000,
        "mask_decode_mode": "accurate",
        "tradeoff_factor": 0.0,
        "disable_active_learning": True,
        "active_learning_target_dataset": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# run_locally
# ---------------------------------------------------------------------------


def test_run_locally_preserves_rle_masks_from_adapter() -> None:
    # given
    image = _make_image()
    images = Batch(content=[image], indices=[(0,)])
    raw = [_make_instance_detections_with_rle(n=2, image_size=(64, 64))]
    model_manager = MagicMock()
    model_manager.run_tensor_native_inference.return_value = raw
    model_manager.get_class_names.return_value = ["cat", "dog"]
    block = _make_block(model_manager)

    # when
    result = block.run_locally(images=images, **_local_kwargs())

    # then
    prediction = result[0]["predictions"]
    assert isinstance(prediction, InstanceDetections)
    assert isinstance(prediction.mask, InstancesRLEMasks)
    assert prediction.mask.image_size == (64, 64)
    assert len(prediction.mask.masks) == 2


def test_run_locally_passes_dense_mask_through_when_adapter_returns_dense() -> None:
    # given
    image = _make_image()
    images = Batch(content=[image], indices=[(0,)])
    raw = [_make_instance_detections_with_dense_mask(n=1, image_size=(8, 8))]
    model_manager = MagicMock()
    model_manager.run_tensor_native_inference.return_value = raw
    model_manager.get_class_names.return_value = ["a"]
    block = _make_block(model_manager)

    # when
    result = block.run_locally(images=images, **_local_kwargs())

    # then
    prediction = result[0]["predictions"]
    assert isinstance(prediction.mask, torch.Tensor)
    assert prediction.mask.shape == (1, 8, 8)


def test_run_locally_attaches_metadata_with_full_class_names_dict() -> None:
    # given
    image = _make_image()
    images = Batch(content=[image], indices=[(0,)])
    model_manager = MagicMock()
    model_manager.run_tensor_native_inference.return_value = [
        _make_instance_detections_with_rle(n=1)
    ]
    model_manager.get_class_names.return_value = ["cat", "dog", "bird"]
    block = _make_block(model_manager)

    # when
    result = block.run_locally(images=images, **_local_kwargs())

    # then
    meta = result[0]["predictions"].image_metadata
    assert meta[CLASS_NAMES_KEY] == {0: "cat", 1: "dog", 2: "bird"}
    assert meta[PREDICTION_TYPE_KEY] == "instance-segmentation"
    assert meta[MODEL_ID_KEY] == "m/1"


def test_run_locally_passes_input_color_format_rgb_to_adapter() -> None:
    # given
    image = _make_image()
    images = Batch(content=[image], indices=[(0,)])
    model_manager = MagicMock()
    model_manager.run_tensor_native_inference.return_value = [
        _make_instance_detections_with_rle(n=0)
    ]
    model_manager.get_class_names.return_value = []
    block = _make_block(model_manager)

    # when
    block.run_locally(images=images, **_local_kwargs())

    # then
    _, kwargs = model_manager.run_tensor_native_inference.call_args
    assert kwargs["input_color_format"] == "rgb"
    assert kwargs["mask_decode_mode"] == "accurate"


def test_run_locally_handles_empty_predictions_batch_member() -> None:
    # given
    image = _make_image()
    images = Batch(content=[image], indices=[(0,)])
    model_manager = MagicMock()
    model_manager.run_tensor_native_inference.return_value = [
        _make_instance_detections_with_rle(n=0)
    ]
    model_manager.get_class_names.return_value = ["a"]
    block = _make_block(model_manager)

    # when
    result = block.run_locally(images=images, **_local_kwargs())

    # then
    pred = result[0]["predictions"]
    assert pred.xyxy.shape == (0, 4)
    assert isinstance(pred.mask, InstancesRLEMasks)
    assert pred.image_metadata[PREDICTION_TYPE_KEY] == "instance-segmentation"


# ---------------------------------------------------------------------------
# run_remotely — RLE path
# ---------------------------------------------------------------------------


def _patch_inference_http_client(client: MagicMock):
    return patch(
        "inference.core.workflows.core_steps.models.roboflow.instance_segmentation.v3_tensor.InferenceHTTPClient",
        return_value=client,
    )


def _patch_inference_configuration():
    return patch(
        "inference.core.workflows.core_steps.models.roboflow.instance_segmentation.v3_tensor.InferenceConfiguration"
    )


def test_run_remotely_requests_rle_mask_format_on_inference_configuration() -> None:
    # given
    image = _make_image()
    images = Batch(content=[image], indices=[(0,)])
    http_client = MagicMock()
    http_client.infer.return_value = [
        {"image": {"width": 64, "height": 64}, "predictions": []},
    ]
    model_manager = MagicMock()
    block = _make_block(model_manager, step_execution_mode=StepExecutionMode.REMOTE)

    # when
    with _patch_inference_http_client(http_client), _patch_inference_configuration() as Cfg:
        block.run_remotely(images=images, **_local_kwargs())

    # then
    _, kwargs = Cfg.call_args
    assert kwargs["response_mask_format"] == "rle"


def test_run_remotely_builds_instance_detections_with_rle_masks() -> None:
    # given
    image = _make_image()
    images = Batch(content=[image], indices=[(0,)])
    http_client = MagicMock()
    http_client.infer.return_value = [
        {
            "image": {"width": 64, "height": 64},
            "predictions": [
                {
                    "x": 32.0,
                    "y": 32.0,
                    "width": 10.0,
                    "height": 10.0,
                    "class": "cat",
                    "class_id": 0,
                    "confidence": 0.9,
                    "rle": {"size": [64, 64], "counts": "fake-rle-bytes"},
                },
            ],
            INFERENCE_ID_KEY: "remote-inf-1",
        },
    ]
    model_manager = MagicMock()
    block = _make_block(model_manager, step_execution_mode=StepExecutionMode.REMOTE)

    # when
    with _patch_inference_http_client(http_client):
        result = block.run_remotely(images=images, **_local_kwargs())

    # then
    prediction = result[0]["predictions"]
    assert isinstance(prediction, InstanceDetections)
    assert isinstance(prediction.mask, InstancesRLEMasks)
    assert prediction.mask.image_size == (64, 64)
    assert prediction.mask.masks == ["fake-rle-bytes"]
    assert result[0]["inference_id"] == "remote-inf-1"


def test_run_remotely_attaches_sparse_class_names_from_response() -> None:
    # given
    image = _make_image()
    images = Batch(content=[image], indices=[(0,)])
    http_client = MagicMock()
    http_client.infer.return_value = [
        {
            "image": {"width": 64, "height": 64},
            "predictions": [
                {
                    "x": 32.0,
                    "y": 32.0,
                    "width": 10.0,
                    "height": 10.0,
                    "class": "dog",
                    "class_id": 5,
                    "confidence": 0.7,
                    "rle": {"size": [64, 64], "counts": "x"},
                },
            ],
        },
    ]
    model_manager = MagicMock()
    block = _make_block(model_manager, step_execution_mode=StepExecutionMode.REMOTE)

    # when
    with _patch_inference_http_client(http_client):
        result = block.run_remotely(images=images, **_local_kwargs())

    # then
    metadata = result[0]["predictions"].image_metadata
    assert metadata[CLASS_NAMES_KEY] == {5: "dog"}
