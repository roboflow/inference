import numpy as np
import pytest

from inference.core.entities.responses.inference import (
    InstanceSegmentationInferenceResponse,
)
from inference.core.env import MAX_BATCH_SIZE
from inference.models import YOLOv7InstanceSegmentation


@pytest.mark.slow
def test_yolov7_segmentation_single_image_inference(
    yolov7_seg_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    model = YOLOv7InstanceSegmentation(model_id=yolov7_seg_model, api_key="DUMMY")

    # when
    result = model.infer(
        example_image,
        confidence=0.5,
        iou_threshold=0.5,
    )

    # then
    assert len(result) == 1, "Batch size=1 hence 1 result expected"
    assert_yolov7_segmentation_prediction_matches_reference(prediction=result[0])


@pytest.mark.slow
def test_yolov7_segmentation_batch_inference_when_batch_size_smaller_than_max_batch_size(
    yolov7_seg_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    batch_size = min(4, MAX_BATCH_SIZE)
    model = YOLOv7InstanceSegmentation(model_id=yolov7_seg_model, api_key="DUMMY")

    # when
    result = model.infer(
        [example_image] * batch_size,
        confidence=0.5,
        iou_threshold=0.5,
    )

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_yolov7_segmentation_prediction_matches_reference(prediction=prediction)


@pytest.mark.slow
@pytest.mark.skipif(
    MAX_BATCH_SIZE > 8,
    reason="This test requires reasonably small MAX_BATCH_SIZE set via environment variable",
)
def test_yolov7_segmentation_batch_inference_when_batch_size_larger_than_max_batch_size(
    yolov7_seg_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    batch_size = MAX_BATCH_SIZE + 2
    model = YOLOv7InstanceSegmentation(model_id=yolov7_seg_model, api_key="DUMMY")

    # when
    result = model.infer(
        [example_image] * batch_size,
        confidence=0.5,
        iou_threshold=0.5,
    )

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_yolov7_segmentation_prediction_matches_reference(prediction=prediction)


def assert_yolov7_segmentation_prediction_matches_reference(
    prediction: InstanceSegmentationInferenceResponse,
) -> None:
    assert (
        len(prediction.predictions) == 4
    ), "Four instances predicted by the model while test creation"
    assert (
        prediction.predictions[0].class_name == "dog"
    ), "while test creation, dog was first bbox class"
    assert (
        abs(prediction.predictions[0].confidence - 0.771583) < 1e-4
    ), "while test creation, confidence was 0.771583"
    xywh = [
        prediction.predictions[0].x,
        prediction.predictions[0].y,
        prediction.predictions[0].width,
        prediction.predictions[0].height,
    ]
    assert np.allclose(
        xywh, [312.0, 321.5, 616.0, 411.5], atol=0.6
    ), "while test creation, box coordinates was [312.0, 321.5, 616.0, 411.5]"
    assert (
        len(prediction.predictions[0].points) == 618
    ), "while test creation, mask had 618 points"
