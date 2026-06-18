from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    InferenceResponseImageDC,
    InstanceSegmentationInferenceResponse,
    InstanceSegmentationInferenceResponseDC,
    InstanceSegmentationPrediction,
    InstanceSegmentationPredictionDC,
    Point,
    PointDC,
    _is_response_dc_to_dict,
)
from inference_models.models.base.async_handoff import (
    attach_async_response_future,
    get_async_response_context_id,
    get_async_response_future,
)


def test_instance_segmentation_dc_dump_matches_pydantic_model_dump() -> None:
    dc_response = InstanceSegmentationInferenceResponseDC(
        predictions=[
            InstanceSegmentationPredictionDC(
                x=1.0,
                y=2.0,
                width=3.0,
                height=4.0,
                confidence=0.9,
                class_name="car",
                class_id=1,
                points=[PointDC(x=1.0, y=2.0), PointDC(x=3.0, y=4.0)],
                detection_id="detection-1",
                parent_id="parent-1",
                class_confidence=0.8,
            )
        ],
        image=InferenceResponseImageDC(width=640, height=480),
        inference_id="inference-1",
        frame_id=1,
        time=0.01,
    )
    pydantic_response = InstanceSegmentationInferenceResponse(
        predictions=[
            InstanceSegmentationPrediction(
                x=1.0,
                y=2.0,
                width=3.0,
                height=4.0,
                confidence=0.9,
                class_id=1,
                points=[Point(x=1.0, y=2.0), Point(x=3.0, y=4.0)],
                detection_id="detection-1",
                parent_id="parent-1",
                class_confidence=0.8,
                **{"class": "car"},
            )
        ],
        image=InferenceResponseImage(width=640, height=480),
        inference_id="inference-1",
        frame_id=1,
        time=0.01,
    )

    assert _is_response_dc_to_dict(dc_response) == pydantic_response.model_dump(
        by_alias=True,
        exclude_none=True,
    )


def test_instance_segmentation_dc_can_carry_async_response_context() -> None:
    response = InstanceSegmentationInferenceResponseDC(
        predictions=[],
        image=InferenceResponseImageDC(width=640, height=480),
    )
    response_future = object()

    attach_async_response_future(
        response=response,
        response_future=response_future,
        context_id="context-1",
    )

    assert get_async_response_future(response) is response_future
    assert get_async_response_context_id(response) == "context-1"
