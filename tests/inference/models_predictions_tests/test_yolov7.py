import numpy as np
import pytest

from inference.core.entities.responses.inference import (
    InstanceSegmentationInferenceResponse,
)
from inference.core.env import MAX_BATCH_SIZE
from inference.models import YOLOv7InstanceSegmentation
from tests.common import assert_localized_predictions_match


@pytest.mark.slow
def test_yolov7_segmentation_single_image_inference(
    yolov7_seg_model: str,
    example_image: np.ndarray,
    yolov7_seg_reference_prediction: InstanceSegmentationInferenceResponse,
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
    assert_localized_predictions_match(
        result_prediction=result[0].model_dump(by_alias=True, exclude_none=True),
        reference_prediction=yolov7_seg_reference_prediction.model_dump(
            by_alias=True, exclude_none=True
        ),
        box_confidence_tolerance=5e-3,
    )


@pytest.mark.slow
def test_yolov7_segmentation_batch_inference_when_batch_size_smaller_than_max_batch_size(
    yolov7_seg_model: str,
    example_image: np.ndarray,
    yolov7_seg_reference_prediction: InstanceSegmentationInferenceResponse,
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
        assert_localized_predictions_match(
            result_prediction=prediction.model_dump(by_alias=True, exclude_none=True),
            reference_prediction=yolov7_seg_reference_prediction.model_dump(
                by_alias=True, exclude_none=True
            ),
            box_confidence_tolerance=5e-3,
        )


@pytest.mark.slow
@pytest.mark.skipif(
    MAX_BATCH_SIZE > 8,
    reason="This test requires reasonably small MAX_BATCH_SIZE set via environment variable",
)
def test_yolov7_segmentation_batch_inference_when_batch_size_larger_than_max_batch_size(
    yolov7_seg_model: str,
    example_image: np.ndarray,
    yolov7_seg_reference_prediction: InstanceSegmentationInferenceResponse,
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
        assert_localized_predictions_match(
            result_prediction=prediction.model_dump(by_alias=True, exclude_none=True),
            reference_prediction=yolov7_seg_reference_prediction.model_dump(
                by_alias=True, exclude_none=True
            ),
            box_confidence_tolerance=5e-3,
        )
