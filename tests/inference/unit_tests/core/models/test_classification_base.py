import numpy as np

from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    MultiLabelClassificationInferenceResponse,
)
from inference.core.models.classification_base import (
    ClassificationBaseOnnxRoboflowInferenceModel,
)


def _make_model(multiclass: bool) -> ClassificationBaseOnnxRoboflowInferenceModel:
    # Bypass the weight-loading __init__ and set only the attributes
    # make_response depends on.
    model = ClassificationBaseOnnxRoboflowInferenceModel.__new__(
        ClassificationBaseOnnxRoboflowInferenceModel
    )
    model.multiclass = multiclass
    model.class_names = ["cat", "dog"]
    return model


def test_make_response_multilabel_reports_width_and_height_for_non_square_image() -> (
    None
):
    # given
    model = _make_model(multiclass=True)
    predictions = [np.array([[0.9, 0.2]])]
    # img_dims come from prepare() as (height, width); here 640 wide x 480 tall.
    img_dims = [(480, 640)]

    # when
    responses = model.make_response(predictions, img_dims, confidence=0.5)

    # then
    assert isinstance(responses[0], MultiLabelClassificationInferenceResponse)
    assert responses[0].image.width == 640
    assert responses[0].image.height == 480


def test_make_response_single_label_reports_width_and_height_for_non_square_image() -> (
    None
):
    # given
    model = _make_model(multiclass=False)
    predictions = [np.array([[0.9, 0.2]])]
    img_dims = [(480, 640)]

    # when
    responses = model.make_response(predictions, img_dims, confidence=0.0)

    # then
    assert isinstance(responses[0], ClassificationInferenceResponse)
    assert responses[0].image.width == 640
    assert responses[0].image.height == 480
