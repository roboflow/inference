import numpy as np
import pytest

from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    MultiLabelClassificationInferenceResponse,
)
from inference.core.env import MAX_BATCH_SIZE
from inference.models import VitClassification


@pytest.mark.slow
def test_vit_multi_class_single_image_inference(
    vit_multi_class_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    model = VitClassification(model_id=vit_multi_class_model, api_key="DUMMY")

    # when
    result = model.infer(example_image)

    # then
    assert len(result) == 1, "Batch size=1 hence 1 result expected"
    assert_vit_multi_class_prediction_matches_reference(prediction=result[0])


@pytest.mark.slow
def test_vit_multi_class_batch_inference_when_batch_size_smaller_than_max_batch_size(
    vit_multi_class_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    batch_size = min(4, MAX_BATCH_SIZE)
    model = VitClassification(model_id=vit_multi_class_model, api_key="DUMMY")

    # when
    result = model.infer([example_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    reference_prediction = result[0]
    assert all(
        p == reference_prediction for p in result
    ), "All predictions must be the same as input was re-used"
    assert_vit_multi_class_prediction_matches_reference(prediction=reference_prediction)


@pytest.mark.slow
@pytest.mark.skipif(
    MAX_BATCH_SIZE > 8,
    reason="This test requires reasonably small MAX_BATCH_SIZE set via environment variable",
)
def test_vit_multi_class_batch_inference_when_batch_size_smaller_larger_max_batch_size(
    vit_multi_class_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    batch_size = MAX_BATCH_SIZE + 2
    model = VitClassification(model_id=vit_multi_class_model, api_key="DUMMY")

    # when
    result = model.infer([example_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    reference_prediction = result[0]
    assert all(
        p == reference_prediction for p in result
    ), "All predictions must be the same as input was re-used"
    assert_vit_multi_class_prediction_matches_reference(prediction=reference_prediction)


def assert_vit_multi_class_prediction_matches_reference(
    prediction: ClassificationInferenceResponse,
) -> None:
    assert (
        prediction.top == "bird"
    ), "This is assertion for model from random weights, it was checked to be bird while model was created"
    assert (
        abs(prediction.confidence - 0.0386) < 1e-5
    ), "This is assertion for model from random weights, it was checked to be 0.038 while model was created"
    assert (
        len(prediction.predictions) == 32
    ), "This random model was created with 32 classes - all must be in prediction"


@pytest.mark.slow
def test_vit_multi_label_single_image_inference(
    vit_multi_label_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    model = VitClassification(model_id=vit_multi_label_model, api_key="DUMMY")

    # when
    result = model.infer(example_image)

    # then
    assert len(result) == 1, "Batch size=1 hence 1 result expected"
    assert_vit_multi_label_prediction_matches_reference(prediction=result[0])


@pytest.mark.slow
def test_vit_multi_label_batch_inference_when_batch_size_smaller_than_max_batch_size(
    vit_multi_label_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    batch_size = min(4, MAX_BATCH_SIZE)
    model = VitClassification(model_id=vit_multi_label_model, api_key="DUMMY")

    # when
    result = model.infer([example_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    reference_prediction = result[0]
    assert all(
        p == reference_prediction for p in result
    ), "All predictions must be the same as input was re-used"
    assert_vit_multi_label_prediction_matches_reference(prediction=reference_prediction)


@pytest.mark.slow
@pytest.mark.skipif(
    MAX_BATCH_SIZE > 8,
    reason="This test requires reasonably small MAX_BATCH_SIZE set via environment variable",
)
def test_vit_multi_label_batch_inference_when_batch_size_smaller_larger_max_batch_size(
    vit_multi_label_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    batch_size = MAX_BATCH_SIZE + 2
    model = VitClassification(model_id=vit_multi_label_model, api_key="DUMMY")

    # when
    result = model.infer([example_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    reference_prediction = result[0]
    assert all(
        p == reference_prediction for p in result
    ), "All predictions must be the same as input was re-used"
    assert_vit_multi_label_prediction_matches_reference(prediction=reference_prediction)


def assert_vit_multi_label_prediction_matches_reference(
    prediction: MultiLabelClassificationInferenceResponse,
) -> None:
    assert sorted(prediction.predicted_classes) == sorted(
        [
            "parking meter",
            "horse",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "frisbee",
            "airplane",
            "truck",
            "boat",
        ]
    ), "This is assertion for model from random weights, it was checked while model was created"
    assert (
        abs(prediction.predictions["person"].confidence - 0.469358) < 1e-4
    ), "This is assertion for model from random weights, it was checked to be 0.469358 while model was created"
    assert (
        len(prediction.predictions) == 32
    ), "This random model was created with 32 classes - all must be in prediction"
