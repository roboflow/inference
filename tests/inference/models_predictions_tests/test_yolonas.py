import numpy as np
import pytest

from inference.core.entities.responses.inference import ObjectDetectionInferenceResponse
from inference.core.env import MAX_BATCH_SIZE
from inference.models import YOLONASObjectDetection
from tests.common import assert_localized_predictions_match


@pytest.mark.slow
def test_yolonas_detection_single_image_inference(
    yolonas_det_model: str,
    beer_image: np.ndarray,
    yolonas_det_reference_prediction: ObjectDetectionInferenceResponse,
) -> None:
    # given
    model = YOLONASObjectDetection(model_id=yolonas_det_model, api_key="DUMMY")

    # when
    result = model.infer(beer_image)

    # then
    assert len(result) == 1, "Batch size=1 hence 1 result expected"
    assert_localized_predictions_match(
        result_prediction=result[0].model_dump(by_alias=True, exclude_none=True),
        reference_prediction=yolonas_det_reference_prediction.model_dump(
            by_alias=True, exclude_none=True
        ),
    )


@pytest.mark.slow
def test_yolonas_detection_batch_inference_when_batch_size_smaller_than_max_batch_size(
    yolonas_det_model: str,
    beer_image: np.ndarray,
    yolonas_det_reference_prediction: ObjectDetectionInferenceResponse,
) -> None:
    # given
    batch_size = min(4, MAX_BATCH_SIZE)
    model = YOLONASObjectDetection(model_id=yolonas_det_model, api_key="DUMMY")

    # when
    result = model.infer([beer_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_localized_predictions_match(
            result_prediction=prediction.model_dump(by_alias=True, exclude_none=True),
            reference_prediction=yolonas_det_reference_prediction.model_dump(
                by_alias=True, exclude_none=True
            ),
        )


@pytest.mark.slow
@pytest.mark.skipif(
    MAX_BATCH_SIZE > 8,
    reason="This test requires reasonably small MAX_BATCH_SIZE set via environment variable",
)
def test_yolonas_detection_batch_inference_when_batch_size_larger_than_max_batch_size(
    yolonas_det_model: str,
    beer_image: np.ndarray,
    yolonas_det_reference_prediction: ObjectDetectionInferenceResponse,
) -> None:
    # given
    batch_size = MAX_BATCH_SIZE + 2
    model = YOLONASObjectDetection(model_id=yolonas_det_model, api_key="DUMMY")

    # when
    result = model.infer([beer_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_localized_predictions_match(
            result_prediction=prediction.model_dump(by_alias=True, exclude_none=True),
            reference_prediction=yolonas_det_reference_prediction.model_dump(
                by_alias=True, exclude_none=True
            ),
        )
