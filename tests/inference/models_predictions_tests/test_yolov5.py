import numpy as np
import pytest

from inference.core.entities.responses.inference import (
    InstanceSegmentationInferenceResponse,
    ObjectDetectionInferenceResponse,
)
from inference.core.env import MAX_BATCH_SIZE
from inference.models import YOLOv5InstanceSegmentation, YOLOv5ObjectDetection


@pytest.mark.slow
def test_yolov5_detection_single_image_inference(
    yolov5_det_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    model = YOLOv5ObjectDetection(model_id=yolov5_det_model, api_key="DUMMY")

    # when
    result = model.infer(example_image)

    # then
    assert len(result) == 1, "Batch size=1 hence 1 result expected"
    assert_yolov5_detection_prediction_matches_reference(prediction=result[0])


@pytest.mark.slow
def test_yolov5_detection_batch_inference_when_batch_size_smaller_than_max_batch_size(
    yolov5_det_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    batch_size = min(4, MAX_BATCH_SIZE)
    model = YOLOv5ObjectDetection(model_id=yolov5_det_model, api_key="DUMMY")

    # when
    result = model.infer([example_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_yolov5_detection_prediction_matches_reference(prediction=prediction)


@pytest.mark.slow
@pytest.mark.skipif(
    MAX_BATCH_SIZE > 8,
    reason="This test requires reasonably small MAX_BATCH_SIZE set via environment variable",
)
def test_yolov5_detection_batch_inference_when_batch_size_larger_then_max_batch_size(
    yolov5_det_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    batch_size = MAX_BATCH_SIZE + 2
    model = YOLOv5ObjectDetection(model_id=yolov5_det_model, api_key="DUMMY")

    # when
    result = model.infer([example_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_yolov5_detection_prediction_matches_reference(prediction=prediction)


def assert_yolov5_detection_prediction_matches_reference(
    prediction: ObjectDetectionInferenceResponse,
) -> None:
    assert (
        len(prediction.predictions) == 1
    ), "Only one bbox predicted by the model while test creation"
    assert (
        prediction.predictions[0].class_name == "dog"
    ), "while test creation, dog was the bbox class"
    assert (
        abs(prediction.predictions[0].confidence - 0.58558) < 1e-4
    ), "while test creation, confidence was 0.58558"
    xywh = [
        prediction.predictions[0].x,
        prediction.predictions[0].y,
        prediction.predictions[0].width,
        prediction.predictions[0].height,
    ]
    assert np.allclose(
        xywh, [309.5, 220.5, 619.0, 407.0], atol=0.6
    ), "while test creation, box coordinates was [309.5, 220.5, 619.0, 407.0]"


@pytest.mark.slow
def test_yolov5_segmentation_single_image_inference(
    yolov5_seg_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    model = YOLOv5InstanceSegmentation(model_id=yolov5_seg_model, api_key="DUMMY")

    # when
    result = model.infer(
        example_image[:, :, ::-1]
    )  # TODO: investigate why RGB is needed

    # then
    assert len(result) == 1, "Batch size=1 hence 1 result expected"
    assert_yolov5_segmentation_prediction_matches_reference(prediction=result[0])


@pytest.mark.slow
def test_yolov5_segmentation_batch_inference_when_batch_size_smaller_than_max_batch_size(
    yolov5_seg_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    batch_size = min(4, MAX_BATCH_SIZE)
    model = YOLOv5InstanceSegmentation(model_id=yolov5_seg_model, api_key="DUMMY")

    # when
    result = model.infer(
        [example_image[:, :, ::-1]] * batch_size
    )  # TODO: investigate why RGB is needed

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_yolov5_segmentation_prediction_matches_reference(prediction=prediction)


@pytest.mark.slow
@pytest.mark.skipif(
    MAX_BATCH_SIZE > 8,
    reason="This test requires reasonably small MAX_BATCH_SIZE set via environment variable",
)
def test_yolov5_segmentation_batch_inference_when_batch_size_larger_then_max_batch_size(
    yolov5_seg_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    batch_size = MAX_BATCH_SIZE + 2
    model = YOLOv5InstanceSegmentation(model_id=yolov5_seg_model, api_key="DUMMY")

    # when
    result = model.infer(
        [example_image[:, :, ::-1]] * batch_size
    )  # TODO: investigate why RGB is needed

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_yolov5_segmentation_prediction_matches_reference(prediction=prediction)


def assert_yolov5_segmentation_prediction_matches_reference(
    prediction: InstanceSegmentationInferenceResponse,
) -> None:
    assert (
        len(prediction.predictions) == 1
    ), "Only one instance predicted by the model while test creation"
    assert (
        prediction.predictions[0].class_name == "dog"
    ), "while test creation, dog was the bbox class"
    assert (
        abs(prediction.predictions[0].confidence - 0.53377) < 1e-4
    ), "while test creation, confidence was 0.53377"
    xywh = [
        prediction.predictions[0].x,
        prediction.predictions[0].y,
        prediction.predictions[0].width,
        prediction.predictions[0].height,
    ]
    assert np.allclose(
        xywh, [365.5, 319.0, 527.0, 412.0], atol=0.6
    ), "while test creation, box coordinates was [365.5, 212.0, 527.0, 412.0]"
    assert (
        len(prediction.predictions[0].points) == 579
    ), "while test creation, mask had 579 points"
