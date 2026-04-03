import numpy as np
import pytest

from inference.core.entities.responses.inference import (
    InstanceSegmentationInferenceResponse,
    KeypointsDetectionInferenceResponse,
    ObjectDetectionInferenceResponse,
)
from inference.core.env import MAX_BATCH_SIZE
from inference.models import (
    YOLO26InstanceSegmentation,
    YOLO26KeypointsDetection,
    YOLO26ObjectDetection,
)
from tests.common import assert_localized_predictions_match


@pytest.mark.slow
def test_yolo26_detection_single_image_inference(
    yolo26_det_model: str,
    example_image: np.ndarray,
    yolo26_det_reference_prediction: ObjectDetectionInferenceResponse,
) -> None:
    # given
    model = YOLO26ObjectDetection(model_id=yolo26_det_model, api_key="DUMMY")

    # when
    result = model.infer(example_image)

    # then
    assert len(result) == 1, "Batch size=1 hence 1 result expected"
    assert_localized_predictions_match(
        result_prediction=result[0].model_dump(by_alias=True, exclude_none=True),
        reference_prediction=yolo26_det_reference_prediction.model_dump(
            by_alias=True, exclude_none=True
        ),
    )


@pytest.mark.slow
def test_yolo26_detection_batch_inference_when_batch_size_smaller_than_max_batch_size(
    yolo26_det_model: str,
    example_image: np.ndarray,
    yolo26_det_reference_prediction: ObjectDetectionInferenceResponse,
) -> None:
    # given
    batch_size = min(4, MAX_BATCH_SIZE)
    model = YOLO26ObjectDetection(model_id=yolo26_det_model, api_key="DUMMY")

    # when
    result = model.infer([example_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_localized_predictions_match(
            result_prediction=prediction.model_dump(by_alias=True, exclude_none=True),
            reference_prediction=yolo26_det_reference_prediction.model_dump(
                by_alias=True, exclude_none=True
            ),
        )


@pytest.mark.slow
@pytest.mark.skipif(
    MAX_BATCH_SIZE > 8,
    reason="This test requires reasonably small MAX_BATCH_SIZE set via environment variable",
)
def test_yolo26_detection_batch_inference_when_batch_size_larger_than_max_batch_size(
    yolo26_det_model: str,
    example_image: np.ndarray,
    yolo26_det_reference_prediction: ObjectDetectionInferenceResponse,
) -> None:
    # given
    batch_size = MAX_BATCH_SIZE + 2
    model = YOLO26ObjectDetection(model_id=yolo26_det_model, api_key="DUMMY")

    # when
    result = model.infer([example_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_localized_predictions_match(
            result_prediction=prediction.model_dump(by_alias=True, exclude_none=True),
            reference_prediction=yolo26_det_reference_prediction.model_dump(
                by_alias=True, exclude_none=True
            ),
        )


@pytest.mark.slow
def test_yolo26_segmentation_single_image_inference(
    yolo26_seg_model: str,
    example_image: np.ndarray,
    yolo26_seg_reference_prediction: InstanceSegmentationInferenceResponse,
) -> None:
    # given
    model = YOLO26InstanceSegmentation(model_id=yolo26_seg_model, api_key="DUMMY")

    # when
    result = model.infer(example_image)

    # then
    assert len(result) == 1, "Batch size=1 hence 1 result expected"
    assert_localized_predictions_match(
        result_prediction=result[0].model_dump(by_alias=True, exclude_none=True),
        reference_prediction=yolo26_seg_reference_prediction.model_dump(
            by_alias=True, exclude_none=True
        ),
    )


@pytest.mark.slow
def test_yolo26_segmentation_batch_inference_when_batch_size_smaller_than_max_batch_size(
    yolo26_seg_model: str,
    example_image: np.ndarray,
    yolo26_seg_reference_prediction: InstanceSegmentationInferenceResponse,
) -> None:
    # given
    batch_size = min(4, MAX_BATCH_SIZE)
    model = YOLO26InstanceSegmentation(model_id=yolo26_seg_model, api_key="DUMMY")

    # when
    result = model.infer([example_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_localized_predictions_match(
            result_prediction=prediction.model_dump(by_alias=True, exclude_none=True),
            reference_prediction=yolo26_seg_reference_prediction.model_dump(
                by_alias=True, exclude_none=True
            ),
        )


@pytest.mark.slow
@pytest.mark.skipif(
    MAX_BATCH_SIZE > 8,
    reason="This test requires reasonably small MAX_BATCH_SIZE set via environment variable",
)
def test_yolo26_segmentation_batch_inference_when_batch_size_larger_than_max_batch_size(
    yolo26_seg_model: str,
    example_image: np.ndarray,
    yolo26_seg_reference_prediction: InstanceSegmentationInferenceResponse,
) -> None:
    # given
    batch_size = MAX_BATCH_SIZE + 2
    model = YOLO26InstanceSegmentation(model_id=yolo26_seg_model, api_key="DUMMY")

    # when
    result = model.infer([example_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_localized_predictions_match(
            result_prediction=prediction.model_dump(by_alias=True, exclude_none=True),
            reference_prediction=yolo26_seg_reference_prediction.model_dump(
                by_alias=True, exclude_none=True
            ),
        )


@pytest.mark.slow
def test_yolo26_pose_single_image_inference(
    yolo26_pose_model: str,
    person_image: np.ndarray,
    yolo26_pose_reference_prediction: KeypointsDetectionInferenceResponse,
) -> None:
    # given
    model = YOLO26KeypointsDetection(model_id=yolo26_pose_model, api_key="DUMMY")

    # when
    result = model.infer(person_image)

    # then
    assert len(result) == 1, "Batch size=1 hence 1 result expected"
    assert_localized_predictions_match(
        result_prediction=result[0].model_dump(by_alias=True, exclude_none=True),
        reference_prediction=yolo26_pose_reference_prediction.model_dump(
            by_alias=True, exclude_none=True
        ),
    )


@pytest.mark.slow
def test_yolo26_pose_batch_inference_when_batch_size_smaller_than_max_batch_size(
    yolo26_pose_model: str,
    person_image: np.ndarray,
    yolo26_pose_reference_prediction: KeypointsDetectionInferenceResponse,
) -> None:
    # given
    batch_size = min(4, MAX_BATCH_SIZE)
    model = YOLO26KeypointsDetection(model_id=yolo26_pose_model, api_key="DUMMY")

    # when
    result = model.infer([person_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_localized_predictions_match(
            result_prediction=prediction.model_dump(by_alias=True, exclude_none=True),
            reference_prediction=yolo26_pose_reference_prediction.model_dump(
                by_alias=True, exclude_none=True
            ),
        )


@pytest.mark.slow
@pytest.mark.skipif(
    MAX_BATCH_SIZE > 8,
    reason="This test requires reasonably small MAX_BATCH_SIZE set via environment variable",
)
def test_yolo26_pose_batch_inference_when_batch_size_larger_than_max_batch_size(
    yolo26_pose_model: str,
    person_image: np.ndarray,
    yolo26_pose_reference_prediction: KeypointsDetectionInferenceResponse,
) -> None:
    # given
    batch_size = MAX_BATCH_SIZE + 2
    model = YOLO26KeypointsDetection(model_id=yolo26_pose_model, api_key="DUMMY")

    # when
    result = model.infer([person_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_localized_predictions_match(
            result_prediction=prediction.model_dump(by_alias=True, exclude_none=True),
            reference_prediction=yolo26_pose_reference_prediction.model_dump(
                by_alias=True, exclude_none=True
            ),
        )
