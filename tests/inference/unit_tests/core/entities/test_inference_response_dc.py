import pytest

from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    InferenceResponseImageDC,
    InstanceSegmentationInferenceResponse,
    InstanceSegmentationInferenceResponseDC,
    InstanceSegmentationPrediction,
    InstanceSegmentationPredictionDC,
    Point,
    PointDC,
    _is_pred_dc_to_dict,
    _is_response_dc_to_dict,
)


def _pydantic_prediction(
    *,
    class_name: str = "car",
    detection_id: str = "detection-1",
    optional_fields: dict | None = None,
) -> InstanceSegmentationPrediction:
    fields = {
        "x": 1.0,
        "y": 2.0,
        "width": 3.0,
        "height": 4.0,
        "confidence": 0.9,
        "class_id": 1,
        "points": [Point(x=1.0, y=2.0), Point(x=3.0, y=4.0)],
        "detection_id": detection_id,
        **{"class": class_name},
    }
    if optional_fields:
        fields.update(optional_fields)
    return InstanceSegmentationPrediction(**fields)


def _dc_prediction(
    *,
    class_name: str = "car",
    detection_id: str = "detection-1",
    optional_fields: dict | None = None,
) -> InstanceSegmentationPredictionDC:
    fields = {
        "x": 1.0,
        "y": 2.0,
        "width": 3.0,
        "height": 4.0,
        "confidence": 0.9,
        "class_name": class_name,
        "class_id": 1,
        "points": [PointDC(x=1.0, y=2.0), PointDC(x=3.0, y=4.0)],
        "detection_id": detection_id,
    }
    if optional_fields:
        fields.update(optional_fields)
    return InstanceSegmentationPredictionDC(**fields)


def _pydantic_response(
    predictions: list[InstanceSegmentationPrediction],
    *,
    inference_id: str | None = "inference-1",
    frame_id: int | None = 1,
    time: float | None = 0.01,
) -> InstanceSegmentationInferenceResponse:
    return InstanceSegmentationInferenceResponse(
        predictions=predictions,
        image=InferenceResponseImage(width=640, height=480),
        inference_id=inference_id,
        frame_id=frame_id,
        time=time,
    )


def _dc_response(
    predictions: list[InstanceSegmentationPredictionDC],
    *,
    inference_id: str | None = "inference-1",
    frame_id: int | None = 1,
    time: float | None = 0.01,
) -> InstanceSegmentationInferenceResponseDC:
    return InstanceSegmentationInferenceResponseDC(
        predictions=predictions,
        image=InferenceResponseImageDC(width=640, height=480),
        inference_id=inference_id,
        frame_id=frame_id,
        time=time,
    )


@pytest.mark.parametrize(
    "optional_prediction_fields",
    [
        {},
        {"parent_id": "parent-1", "class_confidence": 0.8},
    ],
    ids=["minimal", "with_optional_fields"],
)
def test_instance_segmentation_dc_dump_matches_pydantic_model_dump(
    optional_prediction_fields: dict,
) -> None:
    dc_response = _dc_response(
        predictions=[_dc_prediction(optional_fields=optional_prediction_fields)]
    )
    pydantic_response = _pydantic_response(
        predictions=[_pydantic_prediction(optional_fields=optional_prediction_fields)]
    )

    assert _is_response_dc_to_dict(dc_response) == pydantic_response.model_dump(
        by_alias=True,
        exclude_none=True,
    )


def test_instance_segmentation_dc_dump_matches_pydantic_empty_predictions() -> None:
    dc_response = _dc_response(predictions=[])
    pydantic_response = _pydantic_response(predictions=[])

    assert _is_response_dc_to_dict(dc_response) == pydantic_response.model_dump(
        by_alias=True,
        exclude_none=True,
    )


def test_is_pred_dc_to_dict_includes_mask_format_even_when_default() -> None:
    prediction = _dc_prediction()

    result = _is_pred_dc_to_dict(prediction)

    assert result["mask_format"] == "polygon"


def test_workflow_dc_path_emits_mask_format_in_prediction_dicts() -> None:
    # Mirrors InstanceSegmentationPredictionDC construction in
    # InferenceModelsInstanceSegmentationAdapter._build_responses_from_detections.
    response = InstanceSegmentationInferenceResponseDC(
        predictions=[
            InstanceSegmentationPredictionDC(
                x=10.0,
                y=20.0,
                width=30.0,
                height=40.0,
                confidence=0.95,
                class_name="person",
                class_id=0,
                points=[PointDC(x=1.0, y=2.0), PointDC(x=3.0, y=4.0)],
            )
        ],
        image=InferenceResponseImageDC(width=640, height=480),
    )

    dumped = _is_response_dc_to_dict(response)

    assert all(
        prediction["mask_format"] == "polygon"
        for prediction in dumped["predictions"]
    )
