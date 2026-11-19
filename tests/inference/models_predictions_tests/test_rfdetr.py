import numpy as np
import pytest

from inference import get_model
from inference.core.entities.responses.inference import ObjectDetectionInferenceResponse
from inference.models import RFDETRObjectDetection
from tests.common import assert_localized_predictions_match


def test_rfdetr_base(example_image: np.ndarray) -> None:
    model = get_model("rfdetr-base")

    predictions = model.infer(example_image, confidence=0.5)[0]

    assert predictions is not None, "Predictions should not be None"


def test_rfdetr_base_with_background_class(
    melee_image: np.ndarray, rfdetr_base_model: str
) -> None:
    model = get_model(rfdetr_base_model)

    predictions = model.infer(melee_image, confidence=0.5)[0]

    assert predictions is not None, "Predictions should not be None"
    classes = [p.class_name for p in predictions.predictions]
    assert not any("background" in class_name for class_name in classes)
    assert set([p.class_id for p in predictions.predictions]) == set([0, 1])


def _sorted_by_x(prediction: dict) -> dict:
    # Two Atto detections differ in confidence by ~1e-4, below the comparison
    # tolerance, so compare detections in x order rather than confidence order.
    return {
        **prediction,
        "predictions": sorted(prediction["predictions"], key=lambda p: p["x"]),
    }


@pytest.mark.slow
@pytest.mark.parametrize("size", ["atto", "femto", "pico"])
def test_rfdetr_pe_core_t_single_image_inference(
    size: str, request, coin_counting_image: np.ndarray
) -> None:
    # given
    model_id = request.getfixturevalue(f"rfdetr_{size}_coin_counting_model")
    reference: ObjectDetectionInferenceResponse = request.getfixturevalue(
        f"rfdetr_{size}_coin_counting_reference_prediction"
    )
    model = get_model(model_id, api_key="DUMMY")

    # when
    result = model.infer(coin_counting_image)

    # then
    assert isinstance(model, RFDETRObjectDetection)
    assert len(result) == 1, "Batch size=1 hence 1 result expected"
    assert_localized_predictions_match(
        result_prediction=_sorted_by_x(
            result[0].model_dump(by_alias=True, exclude_none=True)
        ),
        reference_prediction=_sorted_by_x(
            reference.model_dump(by_alias=True, exclude_none=True)
        ),
    )


@pytest.mark.slow
@pytest.mark.parametrize("size", ["atto", "femto", "pico"])
def test_rfdetr_pe_core_t_batch_inference(
    size: str, request, coin_counting_image: np.ndarray
) -> None:
    # given
    model_id = request.getfixturevalue(f"rfdetr_{size}_coin_counting_model")
    reference: ObjectDetectionInferenceResponse = request.getfixturevalue(
        f"rfdetr_{size}_coin_counting_reference_prediction"
    )
    model = get_model(model_id, api_key="DUMMY")

    # when
    result = model.infer([coin_counting_image, coin_counting_image])

    # then
    assert len(result) == 2, "Number of results must match batch size"
    for prediction in result:
        assert_localized_predictions_match(
            result_prediction=_sorted_by_x(
                prediction.model_dump(by_alias=True, exclude_none=True)
            ),
            reference_prediction=_sorted_by_x(
                reference.model_dump(by_alias=True, exclude_none=True)
            ),
        )
