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


from concurrent.futures import Future

import numpy as np
import supervision as sv

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)
from inference.core.workflows.prototypes.models_provider import InferenceResultsDC


class _StreamResponse:
    """Shaped like the rfdetr adapter's workflow-execution responses: `to_dict()`
    plus, when the adapter attached a handoff, the two private attributes
    `attach_async_response_future` sets
    (inference_models/models/base/async_handoff.py:106-114)."""

    def __init__(self, image, predictions=(), future=None, context_id=None):
        self._payload = {
            "inference_id": "inf",
            "image": image,
            "predictions": list(predictions),
        }
        if future is not None:
            self._async_response_future = future
        if context_id is not None:
            self._async_response_context_id = context_id

    def to_dict(self):
        return self._payload


IMAGE_10x20 = {"width": 10, "height": 20}
ONE_PREDICTION = [
    {
        "x": 5.0,
        "y": 10.0,
        "width": 4.0,
        "height": 6.0,
        "confidence": 0.9,
        "class": "a",
        "class_id": 0,
        "detection_id": "d0",
        "points": [{"x": 3.0, "y": 7.0}, {"x": 7.0, "y": 7.0}, {"x": 7.0, "y": 13.0}],
    }
]


def _one_image_batch(parent_id):
    return Batch(
        content=[
            WorkflowImageData(
                parent_metadata=ImageParentMetadata(parent_id=parent_id),
                numpy_image=np.zeros((20, 10, 3), dtype=np.uint8),
            )
        ],
        indices=[(0,)],
    )


def _run_locally(block, images):
    return block.run_locally(
        images=images,
        model_id="m/1",
        class_agnostic_nms=False,
        class_filter=None,
        confidence=0.4,
        iou_threshold=0.3,
        max_detections=300,
        max_candidates=3000,
        mask_decode_mode="accurate",
        tradeoff_factor=0.0,
        disable_active_learning=False,
        active_learning_target_dataset=None,
        enforce_dense_masks_in_inference_models=False,
    )


def test_cold_model_first_frame_is_queued_before_inference_and_the_pipeline_pairs_and_flushes() -> (
    None
):
    """Round-1 defect 1 + round-3 defect 5. Registration must happen BEFORE
    the depth check, so the FIRST frame - on a cold model - is queued by the
    time the provider is called. A later frame's response then carries the
    first frame's future and context id, which pairs with (and removes) that
    queued context; the frame that carried it stays queued until flush."""
    loaded = {"value": False}
    registered = []
    at_provider_call = []  # (context id passed, pending ids at that moment)

    manager = MagicMock()
    manager.__contains__.side_effect = lambda model_id: loaded["value"]

    def _add_model(**kwargs):
        registered.append(kwargs)
        loaded["value"] = True

    manager.add_model.side_effect = _add_model
    manager.model_supports_stream_pipeline.side_effect = lambda _: loaded["value"]
    manager.get_model_pipeline_depth.side_effect = lambda _: 3 if loaded["value"] else 1

    frame_0_future = Future()

    def _run_instance_segmentation(**kwargs):
        context_id = kwargs["stream_pipeline_context_id"]
        assert kwargs["return_raw_responses"] is True
        at_provider_call.append(
            (
                context_id,
                [c.context_id for c in block._pending_stream_prediction_contexts],
            )
        )
        if len(at_provider_call) == 1:
            # Cold pipeline: the adapter has no finished response yet, so it
            # hands back an empty response with NO future attached.
            response = _StreamResponse(IMAGE_10x20)
        else:
            # Steady state: frame N's call returns the finished result of an
            # OLDER frame - here frame 0's - as a future carrying frame 0's id.
            response = _StreamResponse(
                IMAGE_10x20, future=frame_0_future, context_id=at_provider_call[0][0]
            )
        return InferenceResultsDC(predictions=[], raw_responses=[response])

    manager.run_instance_segmentation.side_effect = _run_instance_segmentation

    block = RoboflowInstanceSegmentationModelBlockV3(
        model_manager=manager,
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    try:
        first = _run_locally(block, _one_image_batch("p0"))
        assert registered == [{"model_id": "m/1", "api_key": "k"}]
        c0 = at_provider_call[0][0]
        # Registration preceded the depth check: the COLD first frame was
        # already queued when the provider was called.
        assert at_provider_call[0][1] == [c0]
        # No handoff on the cold response -> finalised immediately, c0 stays queued.
        assert [c.context_id for c in block._pending_stream_prediction_contexts] == [c0]
        assert len(first) == 1 and len(first[0]["predictions"]) == 0

        second = _run_locally(block, _one_image_batch("p1"))
        c1 = at_provider_call[1][0]
        assert at_provider_call[1][1] == [c0, c1]
        # Frame 1's call returned frame 0's future: c0 moved to deferred
        # processing, c1 is the genuinely outstanding context.
        assert [c.context_id for c in block._pending_stream_prediction_contexts] == [c1]
        assert isinstance(second[0]["predictions"], Future)
        assert second[0]["model_id"] == "m/1"

        # Resolve frame 0's delayed result: it must be finalised against frame
        # 0's image (parent "p0"), not the frame that carried it.
        frame_0_future.set_result(
            [_StreamResponse(IMAGE_10x20, predictions=ONE_PREDICTION)]
        )
        resolved = second[0]["predictions"].result(timeout=5)
        assert isinstance(resolved, sv.Detections) and len(resolved) == 1
        assert resolved["parent_id"].tolist() == ["p0"]

        # Flush drains exactly the outstanding context (c1) and pairs it with frame 1.
        manager.flush_model_stream_pipeline.return_value = [
            _StreamResponse(IMAGE_10x20, predictions=ONE_PREDICTION)
        ]
        flushed = block.flush_stream_pipeline_outputs()
        assert len(flushed) == 1
        indices, outputs = flushed[0]
        assert indices == [(0,)]
        assert outputs[0]["predictions"]["parent_id"].tolist() == ["p1"]
        assert len(block._pending_stream_prediction_contexts) == 0
    finally:
        # Round-4 defect 7: if an assertion above fails before `set_result`, the
        # worker is still blocked in `_finalize_async_prediction_value`
        # (`v3.py:763`, `future.result(timeout=WORKFLOWS_ASYNC_FUTURE_RESULT_TIMEOUT)`,
        # 60 s by default, env.py:1240). Cancel the source future first - the
        # waiter gets CancelledError immediately - then drain the executor with
        # a blocking shutdown, then close the pipeline.
        if not frame_0_future.done():
            frame_0_future.cancel()
        executor = block._stream_response_executor
        if executor is not None:
            executor.shutdown(wait=True)
        block.close_stream_pipeline()


from inference.core.entities.requests.inference import (
    InstanceSegmentationInferenceRequest,
)
from inference.core.interfaces.workflows_models_provider import (
    ModelManagerModelsProvider,
)

# A payload `InferenceRequestImage` accepts - distinct from `IMAGE_10x20`, which
# is shaped like a *response* image and would fail request validation.
REQUEST_IMAGE = {"type": "base64", "value": "aGVsbG8="}


def test_adapter_run_instance_segmentation_forwards_explicit_none_and_defaults_omitted_fields() -> (
    None
):
    """Round-1 review finding: an explicitly-passed `None` (e.g. `class_filter`)
    must reach the built request as `None`, while a field that is simply never
    passed (the UNSET-defaulted ones) must keep the pydantic default rather than
    becoming `None`."""
    manager = MagicMock()
    manager.infer_from_request_sync.return_value = []
    provider = ModelManagerModelsProvider(manager)

    provider.run_instance_segmentation(
        model_id="m/1",
        images=[REQUEST_IMAGE],
        api_key="k",
        confidence=0.4,
        class_filter=None,
        enforce_dense_masks_in_inference_models=None,
    )

    request = manager.infer_from_request_sync.call_args.kwargs["request"]
    assert isinstance(request, InstanceSegmentationInferenceRequest)
    assert request.class_filter is None
    default = InstanceSegmentationInferenceRequest(
        api_key="k", model_id="m/1", image=[REQUEST_IMAGE]
    )
    # An explicit None overrides a non-None request default (the `_passed`
    # rule must forward None, not drop it)...
    assert default.enforce_dense_masks_in_inference_models is not None
    assert request.enforce_dense_masks_in_inference_models is None
    # ...while omitted (UNSET) fields keep the request's defaults.
    assert request.response_mask_format == default.response_mask_format
    assert request.stream_pipeline_context_id == default.stream_pipeline_context_id


def test_adapter_run_instance_segmentation_wraps_a_single_raw_response_like_v3_used_to() -> (
    None
):
    """`return_raw_responses=True` over a SINGLE (non-list) manager response must
    get the same `[predictions]` wrapping the blocks used to apply inline, with
    the same object identity preserved inside."""
    manager = MagicMock()
    raw = object()
    manager.infer_from_request_sync.return_value = raw  # not a list
    provider = ModelManagerModelsProvider(manager)

    result = provider.run_instance_segmentation(
        model_id="m/1",
        images=[REQUEST_IMAGE],
        api_key="k",
        confidence=0.4,
        return_raw_responses=True,
    )

    assert isinstance(result, InferenceResultsDC)
    assert result.predictions == []
    assert result.raw_responses == [raw]
    assert result.raw_responses[0] is raw


def test_adapter_run_instance_segmentation_normalises_a_list_response_like_the_inlined_dump_used_to() -> (
    None
):
    """The default (non-raw) output must equal what v1/v2/v3/v4 produced inline
    before Task 11.9: `[e.model_dump(by_alias=True, exclude_none=True) for e in
    predictions]`."""
    from inference.core.entities.responses.inference import (
        InferenceResponseImage,
        InstanceSegmentationInferenceResponse,
    )

    response = InstanceSegmentationInferenceResponse(
        image=InferenceResponseImage(width=10, height=20),
        predictions=[],
    )
    manager = MagicMock()
    manager.infer_from_request_sync.return_value = [response]
    provider = ModelManagerModelsProvider(manager)

    result = provider.run_instance_segmentation(
        model_id="m/1",
        images=[REQUEST_IMAGE],
        api_key="k",
        confidence=0.4,
    )

    assert result == [response.model_dump(by_alias=True, exclude_none=True)]
