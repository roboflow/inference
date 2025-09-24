import json

import numpy as np
import pytest
import supervision as sv

from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    InstanceSegmentationInferenceResponse,
    KeypointsDetectionInferenceResponse,
    ObjectDetectionInferenceResponse,
)
from inference.core.env import MAX_BATCH_SIZE
from inference.models import (
    YOLOv8Classification,
    YOLOv8InstanceSegmentation,
    YOLOv8KeypointsDetection,
    YOLOv8ObjectDetection,
)
from tests.common import (
    assert_classification_predictions_match,
    assert_localized_predictions_match,
)


@pytest.mark.slow
def test_yolov8_classification_single_image_inference(
    yolov8_cls_model: str,
    example_image: np.ndarray,
    yolov8_cls_reference_prediction: ClassificationInferenceResponse,
) -> None:
    # given
    model = YOLOv8Classification(model_id=yolov8_cls_model, api_key="DUMMY")

    # when
    result = model.infer(example_image, confidence=0.0009)

    # then
    assert len(result) == 1, "Batch size=1 hence 1 result expected"
    assert_classification_predictions_match(
        result_prediction=result[0].model_dump(by_alias=True, exclude_none=True),
        reference_prediction=yolov8_cls_reference_prediction.model_dump(
            by_alias=True, exclude_none=True
        ),
    )


@pytest.mark.slow
def test_yolov8_classification_batch_inference_when_batch_size_smaller_than_max_batch_size(
    yolov8_cls_model: str,
    example_image: np.ndarray,
    yolov8_cls_reference_prediction: ClassificationInferenceResponse,
) -> None:
    # given
    batch_size = min(4, MAX_BATCH_SIZE)
    model = YOLOv8Classification(model_id=yolov8_cls_model, api_key="DUMMY")

    # when
    result = model.infer([example_image] * batch_size, confidence=0.0009)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_classification_predictions_match(
            result_prediction=prediction.model_dump(by_alias=True, exclude_none=True),
            reference_prediction=yolov8_cls_reference_prediction.model_dump(
                by_alias=True, exclude_none=True
            ),
        )


@pytest.mark.slow
@pytest.mark.skipif(
    MAX_BATCH_SIZE > 8,
    reason="This test requires reasonably small MAX_BATCH_SIZE set via environment variable",
)
def test_yolov8_classification_batch_inference_when_batch_size_larger_than_max_batch_size(
    yolov8_cls_model: str,
    example_image: np.ndarray,
    yolov8_cls_reference_prediction: ClassificationInferenceResponse,
) -> None:
    # given
    batch_size = MAX_BATCH_SIZE + 2
    model = YOLOv8Classification(model_id=yolov8_cls_model, api_key="DUMMY")

    # when
    result = model.infer([example_image] * batch_size, confidence=0.0009)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_classification_predictions_match(
            result_prediction=prediction.model_dump(by_alias=True, exclude_none=True),
            reference_prediction=yolov8_cls_reference_prediction.model_dump(
                by_alias=True, exclude_none=True
            ),
        )


@pytest.mark.slow
def test_yolov8_detection_single_image_inference(
    yolov8_det_model: str,
    example_image: np.ndarray,
    yolov8_det_reference_prediction: ObjectDetectionInferenceResponse,
) -> None:
    # given
    model = YOLOv8ObjectDetection(model_id=yolov8_det_model, api_key="DUMMY")

    # when
    result = model.infer(example_image)

    # then
    assert len(result) == 1, "Batch size=1 hence 1 result expected"
    assert_localized_predictions_match(
        result_prediction=result[0].model_dump(by_alias=True, exclude_none=True),
        reference_prediction=yolov8_det_reference_prediction.model_dump(
            by_alias=True, exclude_none=True
        ),
    )


@pytest.mark.slow
def test_yolov8_detection_batch_inference_when_batch_size_smaller_than_max_batch_size(
    yolov8_det_model: str,
    example_image: np.ndarray,
    yolov8_det_reference_prediction: ObjectDetectionInferenceResponse,
) -> None:
    # given
    batch_size = min(4, MAX_BATCH_SIZE)
    model = YOLOv8ObjectDetection(model_id=yolov8_det_model, api_key="DUMMY")

    # when
    result = model.infer([example_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_localized_predictions_match(
            result_prediction=prediction.model_dump(by_alias=True, exclude_none=True),
            reference_prediction=yolov8_det_reference_prediction.model_dump(
                by_alias=True, exclude_none=True
            ),
        )


@pytest.mark.slow
@pytest.mark.skipif(
    MAX_BATCH_SIZE > 8,
    reason="This test requires reasonably small MAX_BATCH_SIZE set via environment variable",
)
def test_yolov8_detection_batch_inference_when_batch_size_larger_than_max_batch_size(
    yolov8_det_model: str,
    example_image: np.ndarray,
    yolov8_det_reference_prediction: ObjectDetectionInferenceResponse,
) -> None:
    # given
    batch_size = MAX_BATCH_SIZE + 2
    model = YOLOv8ObjectDetection(model_id=yolov8_det_model, api_key="DUMMY")

    # when
    result = model.infer([example_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_localized_predictions_match(
            result_prediction=prediction.model_dump(by_alias=True, exclude_none=True),
            reference_prediction=yolov8_det_reference_prediction.model_dump(
                by_alias=True, exclude_none=True
            ),
        )


@pytest.mark.slow
def test_yolov8_segmentation_single_image_inference(
    yolov8_seg_model: str,
    example_image: np.ndarray,
    yolov8_seg_reference_prediction: InstanceSegmentationInferenceResponse,
) -> None:
    # given
    model = YOLOv8InstanceSegmentation(model_id=yolov8_seg_model, api_key="DUMMY")

    # when
    result = model.infer(example_image)

    # then
    assert len(result) == 1, "Batch size=1 hence 1 result expected"
    assert_localized_predictions_match(
        result_prediction=result[0].model_dump(by_alias=True, exclude_none=True),
        reference_prediction=yolov8_seg_reference_prediction.model_dump(
            by_alias=True, exclude_none=True
        ),
    )


@pytest.mark.slow
def test_yolov8_segmentation_batch_inference_when_batch_size_smaller_than_max_batch_size(
    yolov8_seg_model: str,
    example_image: np.ndarray,
    yolov8_seg_reference_prediction: InstanceSegmentationInferenceResponse,
) -> None:
    # given
    batch_size = min(4, MAX_BATCH_SIZE)
    model = YOLOv8InstanceSegmentation(model_id=yolov8_seg_model, api_key="DUMMY")

    # when
    result = model.infer([example_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_localized_predictions_match(
            result_prediction=prediction.model_dump(by_alias=True, exclude_none=True),
            reference_prediction=yolov8_seg_reference_prediction.model_dump(
                by_alias=True, exclude_none=True
            ),
        )


@pytest.mark.slow
@pytest.mark.skipif(
    MAX_BATCH_SIZE > 8,
    reason="This test requires reasonably small MAX_BATCH_SIZE set via environment variable",
)
def test_yolov8_segmentation_batch_inference_when_batch_size_larger_than_max_batch_size(
    yolov8_seg_model: str,
    example_image: np.ndarray,
    yolov8_seg_reference_prediction: InstanceSegmentationInferenceResponse,
) -> None:
    # given
    batch_size = MAX_BATCH_SIZE + 2
    model = YOLOv8InstanceSegmentation(model_id=yolov8_seg_model, api_key="DUMMY")

    # when
    result = model.infer([example_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_localized_predictions_match(
            result_prediction=prediction.model_dump(by_alias=True, exclude_none=True),
            reference_prediction=yolov8_seg_reference_prediction.model_dump(
                by_alias=True, exclude_none=True
            ),
        )


@pytest.mark.slow
def test_yolov8_pose_single_image_inference(
    yolov8_pose_model: str,
    person_image: np.ndarray,
    yolov8_pose_reference_prediction: KeypointsDetectionInferenceResponse,
) -> None:
    # given
    model = YOLOv8KeypointsDetection(model_id=yolov8_pose_model, api_key="DUMMY")

    # when
    result = model.infer(person_image)

    # then
    assert len(result) == 1, "Batch size=1 hence 1 result expected"
    assert_localized_predictions_match(
        result_prediction=result[0].model_dump(by_alias=True, exclude_none=True),
        reference_prediction=yolov8_pose_reference_prediction.model_dump(
            by_alias=True, exclude_none=True
        ),
    )


@pytest.mark.slow
def test_yolov8_pose_batch_inference_when_batch_size_smaller_than_max_batch_size(
    yolov8_pose_model: str,
    person_image: np.ndarray,
    yolov8_pose_reference_prediction: KeypointsDetectionInferenceResponse,
) -> None:
    # given
    batch_size = min(4, MAX_BATCH_SIZE)
    model = YOLOv8KeypointsDetection(model_id=yolov8_pose_model, api_key="DUMMY")

    # when
    result = model.infer([person_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_localized_predictions_match(
            result_prediction=prediction.model_dump(by_alias=True, exclude_none=True),
            reference_prediction=yolov8_pose_reference_prediction.model_dump(
                by_alias=True, exclude_none=True
            ),
        )


@pytest.mark.slow
@pytest.mark.skipif(
    MAX_BATCH_SIZE > 8,
    reason="This test requires reasonably small MAX_BATCH_SIZE set via environment variable",
)
def test_yolov8_pose_batch_inference_when_batch_size_larger_than_max_batch_size(
    yolov8_pose_model: str,
    person_image: np.ndarray,
    yolov8_pose_reference_prediction: KeypointsDetectionInferenceResponse,
) -> None:
    # given
    batch_size = MAX_BATCH_SIZE + 2
    model = YOLOv8KeypointsDetection(model_id=yolov8_pose_model, api_key="DUMMY")

    # when
    result = model.infer([person_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_localized_predictions_match(
            result_prediction=prediction.model_dump(by_alias=True, exclude_none=True),
            reference_prediction=yolov8_pose_reference_prediction.model_dump(
                by_alias=True, exclude_none=True
            ),
        )
