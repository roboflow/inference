from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from inference_models.models.base.object_detection import Detections as TensorDetections

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.tensor_prediction_metadata import (
    CLASS_NAMES_KEY,
    MODEL_ID_KEY,
)
from inference.core.workflows.core_steps.models.roboflow.object_detection.v3_tensor import (
    RoboflowObjectDetectionModelBlockV3,
)
from inference.core.workflows.execution_engine.constants import (
    IMAGE_DIMENSIONS_KEY,
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


def _make_tensor_detections(
    n: int, image_metadata: Optional[dict] = None
) -> TensorDetections:
    return TensorDetections(
        xyxy=torch.zeros((n, 4), dtype=torch.float32),
        class_id=torch.arange(n, dtype=torch.int64),
        confidence=torch.full((n,), 0.5, dtype=torch.float32),
        image_metadata=image_metadata,
    )


def _make_block(
    model_manager: MagicMock,
    *,
    step_execution_mode: StepExecutionMode = StepExecutionMode.LOCAL,
) -> RoboflowObjectDetectionModelBlockV3:
    return RoboflowObjectDetectionModelBlockV3(
        model_manager=model_manager,
        api_key="test-key",
        step_execution_mode=step_execution_mode,
    )


# ---------------------------------------------------------------------------
# run_locally
# ---------------------------------------------------------------------------


def test_run_locally_returns_predictions_with_metadata_attached() -> None:
    # given
    image_1 = _make_image(parent_id="p1")
    image_2 = _make_image(parent_id="p2")
    images = Batch(content=[image_1, image_2], indices=[(0,), (1,)])
    raw_predictions = [_make_tensor_detections(n=2), _make_tensor_detections(n=1)]
    model_manager = MagicMock()
    model_manager.run_tensor_native_inference.return_value = raw_predictions
    model_manager.get_class_names.return_value = ["cat", "dog"]
    block = _make_block(model_manager)

    # when
    result = block.run_locally(
        images=images,
        model_id="model/1",
        class_agnostic_nms=False,
        class_filter=None,
        confidence=0.5,
        iou_threshold=0.3,
        max_detections=300,
        max_candidates=3000,
        disable_active_learning=True,
        active_learning_target_dataset=None,
    )

    # then
    assert len(result) == 2
    for item in result:
        assert set(item.keys()) == {"inference_id", "predictions", "model_id"}
        assert item["model_id"] == "model/1"
        assert isinstance(item["predictions"], TensorDetections)
        meta = item["predictions"].image_metadata
        assert meta is not None
        assert meta[MODEL_ID_KEY] == "model/1"
        assert meta[PREDICTION_TYPE_KEY] == "object-detection"
        assert meta[CLASS_NAMES_KEY] == {0: "cat", 1: "dog"}


def test_run_locally_passes_input_color_format_rgb_to_adapter() -> None:
    # given
    image = _make_image()
    images = Batch(content=[image], indices=[(0,)])
    model_manager = MagicMock()
    model_manager.run_tensor_native_inference.return_value = [_make_tensor_detections(n=0)]
    model_manager.get_class_names.return_value = []
    block = _make_block(model_manager)

    # when
    block.run_locally(
        images=images,
        model_id="model/1",
        class_agnostic_nms=False,
        class_filter=None,
        confidence=0.5,
        iou_threshold=0.3,
        max_detections=300,
        max_candidates=3000,
        disable_active_learning=True,
        active_learning_target_dataset=None,
    )

    # then
    _, kwargs = model_manager.run_tensor_native_inference.call_args
    assert kwargs["input_color_format"] == "rgb"
    assert kwargs["model_id"] == "model/1"


def test_run_locally_passes_class_filter_through_to_adapter() -> None:
    # given
    image = _make_image()
    images = Batch(content=[image], indices=[(0,)])
    model_manager = MagicMock()
    model_manager.run_tensor_native_inference.return_value = [_make_tensor_detections(n=0)]
    model_manager.get_class_names.return_value = ["a", "b"]
    block = _make_block(model_manager)

    # when
    block.run_locally(
        images=images,
        model_id="model/1",
        class_agnostic_nms=False,
        class_filter=["b"],
        confidence=0.5,
        iou_threshold=0.3,
        max_detections=300,
        max_candidates=3000,
        disable_active_learning=True,
        active_learning_target_dataset=None,
    )

    # then
    _, kwargs = model_manager.run_tensor_native_inference.call_args
    assert kwargs["class_filter"] == ["b"]


def test_run_locally_handles_empty_predictions_batch_member() -> None:
    # given
    image = _make_image()
    images = Batch(content=[image], indices=[(0,)])
    model_manager = MagicMock()
    model_manager.run_tensor_native_inference.return_value = [
        _make_tensor_detections(n=0),
    ]
    model_manager.get_class_names.return_value = ["a"]
    block = _make_block(model_manager)

    # when
    result = block.run_locally(
        images=images,
        model_id="m/1",
        class_agnostic_nms=False,
        class_filter=None,
        confidence=0.5,
        iou_threshold=0.3,
        max_detections=300,
        max_candidates=3000,
        disable_active_learning=True,
        active_learning_target_dataset=None,
    )

    # then
    assert len(result) == 1
    pred = result[0]["predictions"]
    assert isinstance(pred, TensorDetections)
    assert pred.xyxy.shape == (0, 4)
    assert pred.image_metadata is not None
    assert pred.image_metadata[PREDICTION_TYPE_KEY] == "object-detection"


def test_run_locally_preserves_inference_id_from_adapter_metadata() -> None:
    # given
    image = _make_image()
    images = Batch(content=[image], indices=[(0,)])
    preset = _make_tensor_detections(
        n=1, image_metadata={INFERENCE_ID_KEY: "preset-from-adapter"}
    )
    model_manager = MagicMock()
    model_manager.run_tensor_native_inference.return_value = [preset]
    model_manager.get_class_names.return_value = ["a"]
    block = _make_block(model_manager)

    # when
    result = block.run_locally(
        images=images,
        model_id="m/1",
        class_agnostic_nms=False,
        class_filter=None,
        confidence=0.5,
        iou_threshold=0.3,
        max_detections=300,
        max_candidates=3000,
        disable_active_learning=True,
        active_learning_target_dataset=None,
    )

    # then
    assert result[0]["inference_id"] == "preset-from-adapter"


def test_run_locally_mints_inference_id_when_adapter_did_not_supply_one() -> None:
    # given
    image = _make_image()
    images = Batch(content=[image], indices=[(0,)])
    model_manager = MagicMock()
    model_manager.run_tensor_native_inference.return_value = [
        _make_tensor_detections(n=2),
    ]
    model_manager.get_class_names.return_value = ["a"]
    block = _make_block(model_manager)

    # when
    result = block.run_locally(
        images=images,
        model_id="m/1",
        class_agnostic_nms=False,
        class_filter=None,
        confidence=0.5,
        iou_threshold=0.3,
        max_detections=300,
        max_candidates=3000,
        disable_active_learning=True,
        active_learning_target_dataset=None,
    )

    # then
    inference_id = result[0]["inference_id"]
    assert isinstance(inference_id, str)
    assert len(inference_id) > 0


def test_run_locally_writes_image_dimensions_into_metadata() -> None:
    # given
    image = _make_image(h=128, w=256)
    images = Batch(content=[image], indices=[(0,)])
    model_manager = MagicMock()
    model_manager.run_tensor_native_inference.return_value = [
        _make_tensor_detections(n=1),
    ]
    model_manager.get_class_names.return_value = ["a"]
    block = _make_block(model_manager)

    # when
    result = block.run_locally(
        images=images,
        model_id="m/1",
        class_agnostic_nms=False,
        class_filter=None,
        confidence=0.5,
        iou_threshold=0.3,
        max_detections=300,
        max_candidates=3000,
        disable_active_learning=True,
        active_learning_target_dataset=None,
    )

    # then
    assert result[0]["predictions"].image_metadata[IMAGE_DIMENSIONS_KEY] == (128, 256)


# ---------------------------------------------------------------------------
# run_remotely
# ---------------------------------------------------------------------------


def _patch_inference_http_client(client: MagicMock):
    return patch(
        "inference.core.workflows.core_steps.models.roboflow.object_detection.v3_tensor.InferenceHTTPClient",
        return_value=client,
    )


def test_run_remotely_converts_response_dict_to_inference_models_detections() -> None:
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
                },
            ],
            INFERENCE_ID_KEY: "remote-inf-1",
        },
    ]
    model_manager = MagicMock()
    block = _make_block(model_manager, step_execution_mode=StepExecutionMode.REMOTE)

    # when
    with _patch_inference_http_client(http_client):
        result = block.run_remotely(
            images=images,
            model_id="m/1",
            class_agnostic_nms=False,
            class_filter=None,
            confidence=0.5,
            iou_threshold=0.3,
            max_detections=300,
            max_candidates=3000,
            disable_active_learning=True,
            active_learning_target_dataset=None,
        )

    # then
    assert len(result) == 1
    prediction = result[0]["predictions"]
    assert isinstance(prediction, TensorDetections)
    assert prediction.xyxy.shape == (1, 4)
    assert torch.allclose(
        prediction.xyxy[0], torch.tensor([27.0, 27.0, 37.0, 37.0])
    )
    assert result[0]["inference_id"] == "remote-inf-1"


def test_run_remotely_builds_sparse_class_names_dict_from_response() -> None:
    # given
    image = _make_image()
    images = Batch(content=[image], indices=[(0,)])
    http_client = MagicMock()
    http_client.infer.return_value = [
        {
            "image": {"width": 64, "height": 64},
            "predictions": [
                {
                    "x": 30.0,
                    "y": 30.0,
                    "width": 4.0,
                    "height": 4.0,
                    "class": "cat",
                    "class_id": 0,
                    "confidence": 0.5,
                },
                {
                    "x": 40.0,
                    "y": 40.0,
                    "width": 4.0,
                    "height": 4.0,
                    "class": "dog",
                    "class_id": 1,
                    "confidence": 0.5,
                },
            ],
        },
    ]
    model_manager = MagicMock()
    block = _make_block(model_manager, step_execution_mode=StepExecutionMode.REMOTE)

    # when
    with _patch_inference_http_client(http_client):
        result = block.run_remotely(
            images=images,
            model_id="m/1",
            class_agnostic_nms=False,
            class_filter=None,
            confidence=0.5,
            iou_threshold=0.3,
            max_detections=300,
            max_candidates=3000,
            disable_active_learning=True,
            active_learning_target_dataset=None,
        )

    # then
    metadata = result[0]["predictions"].image_metadata
    assert metadata is not None
    assert metadata[CLASS_NAMES_KEY] == {0: "cat", 1: "dog"}
    assert metadata[MODEL_ID_KEY] == "m/1"
    assert metadata[PREDICTION_TYPE_KEY] == "object-detection"


def test_run_remotely_class_names_dict_is_empty_when_response_has_no_predictions() -> None:
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
    with _patch_inference_http_client(http_client):
        result = block.run_remotely(
            images=images,
            model_id="m/1",
            class_agnostic_nms=False,
            class_filter=None,
            confidence=0.5,
            iou_threshold=0.3,
            max_detections=300,
            max_candidates=3000,
            disable_active_learning=True,
            active_learning_target_dataset=None,
        )

    # then
    metadata = result[0]["predictions"].image_metadata
    assert metadata is not None
    assert metadata[CLASS_NAMES_KEY] == {}


def test_run_remotely_mints_inference_id_when_response_lacks_one() -> None:
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
    with _patch_inference_http_client(http_client):
        result = block.run_remotely(
            images=images,
            model_id="m/1",
            class_agnostic_nms=False,
            class_filter=None,
            confidence=0.5,
            iou_threshold=0.3,
            max_detections=300,
            max_candidates=3000,
            disable_active_learning=True,
            active_learning_target_dataset=None,
        )

    # then
    inference_id = result[0]["inference_id"]
    assert isinstance(inference_id, str)
    assert len(inference_id) > 0


def test_run_remotely_handles_singleton_response_wrapping() -> None:
    # given
    image = _make_image()
    images = Batch(content=[image], indices=[(0,)])
    http_client = MagicMock()
    # When inference_input is len 1 the client returns a bare dict, not a list.
    http_client.infer.return_value = {
        "image": {"width": 64, "height": 64},
        "predictions": [],
    }
    model_manager = MagicMock()
    block = _make_block(model_manager, step_execution_mode=StepExecutionMode.REMOTE)

    # when
    with _patch_inference_http_client(http_client):
        result = block.run_remotely(
            images=images,
            model_id="m/1",
            class_agnostic_nms=False,
            class_filter=None,
            confidence=0.5,
            iou_threshold=0.3,
            max_detections=300,
            max_candidates=3000,
            disable_active_learning=True,
            active_learning_target_dataset=None,
        )

    # then
    assert len(result) == 1
    assert isinstance(result[0]["predictions"], TensorDetections)
