from unittest.mock import MagicMock

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.roboflow.instance_segmentation.v3 import (
    RoboflowInstanceSegmentationModelBlockV3,
)


def _block(model_manager) -> RoboflowInstanceSegmentationModelBlockV3:
    block = RoboflowInstanceSegmentationModelBlockV3(
        model_manager=model_manager,
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    block._last_model_id = "m/1"
    return block


def _manager() -> MagicMock:
    manager = MagicMock()
    manager.__contains__.return_value = True  # MagicMock's default is False
    return manager


def test_is_stream_pipelined_uses_the_port_not_item_access() -> None:
    manager = _manager()
    manager.model_supports_stream_pipeline.return_value = True
    assert _block(manager).is_stream_pipelined() is True
    manager.model_supports_stream_pipeline.assert_called_once_with("m/1")
    manager.__getitem__.assert_not_called()


def test_stream_pipeline_depth_subtracts_one() -> None:
    manager = _manager()
    manager.model_supports_stream_pipeline.return_value = True
    manager.get_model_pipeline_depth.return_value = 4
    assert _block(manager).stream_pipeline_depth() == 3
    manager.__getitem__.assert_not_called()


def test_flush_stream_pipeline_outputs_clears_contexts_when_flush_unavailable() -> None:
    manager = _manager()
    manager.flush_model_stream_pipeline.return_value = None
    block = _block(manager)
    block._pending_stream_prediction_contexts.append(object())
    assert block.flush_stream_pipeline_outputs() == []
    assert len(block._pending_stream_prediction_contexts) == 0
    manager.__getitem__.assert_not_called()


def test_close_stream_pipeline_delegates_shutdown() -> None:
    manager = _manager()
    _block(manager).close_stream_pipeline()
    manager.shutdown_model_stream_pipeline.assert_called_once_with("m/1")
    manager.__getitem__.assert_not_called()


def test_dc_and_pydantic_responses_normalise_to_the_same_dict() -> None:
    from inference.core.entities.responses.inference import (
        InferenceResponseImage,
        InferenceResponseImageDC,
        InstanceSegmentationInferenceResponse,
        InstanceSegmentationInferenceResponseDC,
        InstanceSegmentationPrediction,
        InstanceSegmentationPredictionDC,
        Point,
        PointDC,
    )

    dc = InstanceSegmentationInferenceResponseDC(
        image=InferenceResponseImageDC(width=10, height=20),
        predictions=[
            InstanceSegmentationPredictionDC(
                x=1.0,
                y=2.0,
                width=3.0,
                height=4.0,
                confidence=0.5,
                class_name="a",
                class_id=0,
                points=[
                    PointDC(x=0.0, y=0.0),
                    PointDC(x=1.0, y=1.0),
                    PointDC(x=2.0, y=0.0),
                ],
                detection_id="fixed-id",
            )
        ],
    )
    pydantic_equivalent = InstanceSegmentationInferenceResponse(
        image=InferenceResponseImage(width=10, height=20),
        predictions=[
            InstanceSegmentationPrediction(
                **{
                    "x": 1.0,
                    "y": 2.0,
                    "width": 3.0,
                    "height": 4.0,
                    "confidence": 0.5,
                    "class": "a",
                    "class_id": 0,
                    "detection_id": "fixed-id",
                    "points": [
                        Point(x=0.0, y=0.0),
                        Point(x=1.0, y=1.0),
                        Point(x=2.0, y=0.0),
                    ],
                }
            )
        ],
    )
    assert dc.to_dict() == pydantic_equivalent.model_dump(
        by_alias=True, exclude_none=True
    )
