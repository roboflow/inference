import numpy as np
import pytest

from inference.core.entities.responses.inference import ObjectDetectionInferenceResponse
from inference.core.env import MAX_BATCH_SIZE
from inference.models import YOLOv10ObjectDetection


@pytest.mark.slow
def test_yolov10_detection_single_image_inference(
    yolov10_det_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    model = YOLOv10ObjectDetection(model_id=yolov10_det_model, api_key="DUMMY")

    # when
    result = model.infer(example_image)

    # then
    assert len(result) == 1, "Batch size=1 hence 1 result expected"
    assert_yolov10_detection_prediction_matches_reference(prediction=result[0])


@pytest.mark.slow
def test_yolov10_detection_batch_inference_when_batch_size_smaller_than_max_batch_size(
    yolov10_det_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    batch_size = min(4, MAX_BATCH_SIZE)
    model = YOLOv10ObjectDetection(model_id=yolov10_det_model, api_key="DUMMY")

    # when
    result = model.infer([example_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_yolov10_detection_prediction_matches_reference(prediction=prediction)


@pytest.mark.slow
@pytest.mark.skipif(
    MAX_BATCH_SIZE > 8,
    reason="This test requires reasonably small MAX_BATCH_SIZE set via environment variable",
)
def test_yolov10_detection_batch_inference_when_batch_size_larger_than_max_batch_size(
    yolov10_det_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    batch_size = MAX_BATCH_SIZE + 2
    model = YOLOv10ObjectDetection(model_id=yolov10_det_model, api_key="DUMMY")

    # when
    result = model.infer([example_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_yolov10_detection_prediction_matches_reference(prediction=prediction)


def assert_yolov10_detection_prediction_matches_reference(
    prediction: ObjectDetectionInferenceResponse,
) -> None:
    assert (
        len(prediction.predictions) == 1
    ), "Example model is expected to predict 1 bbox, as this is the result obtained while test creation"
    assert (
        prediction.predictions[0].class_name == "dog"
    ), "Dog class was predicted by exported model"
    assert (
        abs(prediction.predictions[0].confidence - 0.903276) < 1e-3
    ), "Confidence while test creation was 0.903276"
    xywh = [
        prediction.predictions[0].x,
        prediction.predictions[0].y,
        prediction.predictions[0].width,
        prediction.predictions[0].height,
    ]
    assert np.allclose(
        xywh, [314.5, 217.0, 597.0, 414.0], atol=0.6
    ), "while test creation, box coordinates was [314.5, 217.0, 597.0, 414.0]"
