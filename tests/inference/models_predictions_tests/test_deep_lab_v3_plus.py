import numpy as np
import pytest

from inference.core.entities.responses.inference import (
    SemanticSegmentationInferenceResponse,
)
from inference.core.env import MAX_BATCH_SIZE
from inference.models import DeepLabV3PlusSemanticSegmentation
from tests.common import assert_semantic_segmentation_predictions_match


@pytest.mark.slow
def test_deep_lab_v3_plus_single_image_inference(
    deep_lab_v3_plus_seg_model: str,
    example_image: np.ndarray,
    deep_lab_v3_plus_seg_reference_prediction: SemanticSegmentationInferenceResponse,
) -> None:
    # given
    model = DeepLabV3PlusSemanticSegmentation(
        model_id=deep_lab_v3_plus_seg_model, api_key="DUMMY"
    )

    # when
    result = model.infer(example_image)

    # then
    assert len(result) == 1, "Batch size=1 hence 1 result expected"
    assert_semantic_segmentation_predictions_match(
        result_prediction=result[0].model_dump(by_alias=True, exclude_none=True),
        reference_prediction=deep_lab_v3_plus_seg_reference_prediction.model_dump(
            by_alias=True, exclude_none=True
        ),
    )


@pytest.mark.slow
def test_deep_lab_v3_plus_batch_inference_when_batch_size_smaller_than_max_batch_size(
    deep_lab_v3_plus_seg_model: str,
    example_image: np.ndarray,
    deep_lab_v3_plus_seg_reference_prediction: SemanticSegmentationInferenceResponse,
) -> None:
    # given
    batch_size = min(4, MAX_BATCH_SIZE)
    model = DeepLabV3PlusSemanticSegmentation(
        model_id=deep_lab_v3_plus_seg_model, api_key="DUMMY"
    )

    # when
    result = model.infer([example_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_semantic_segmentation_predictions_match(
            result_prediction=prediction.model_dump(
                by_alias=True, exclude_none=True
            ),
            reference_prediction=deep_lab_v3_plus_seg_reference_prediction.model_dump(
                by_alias=True, exclude_none=True
            ),
        )


@pytest.mark.slow
@pytest.mark.skipif(
    MAX_BATCH_SIZE > 8,
    reason="This test requires reasonably small MAX_BATCH_SIZE set via environment variable",
)
def test_deep_lab_v3_plus_batch_inference_when_batch_size_larger_than_max_batch_size(
    deep_lab_v3_plus_seg_model: str,
    example_image: np.ndarray,
    deep_lab_v3_plus_seg_reference_prediction: SemanticSegmentationInferenceResponse,
) -> None:
    # given
    batch_size = MAX_BATCH_SIZE + 2
    model = DeepLabV3PlusSemanticSegmentation(
        model_id=deep_lab_v3_plus_seg_model, api_key="DUMMY"
    )

    # when
    result = model.infer([example_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_semantic_segmentation_predictions_match(
            result_prediction=prediction.model_dump(
                by_alias=True, exclude_none=True
            ),
            reference_prediction=deep_lab_v3_plus_seg_reference_prediction.model_dump(
                by_alias=True, exclude_none=True
            ),
        )
