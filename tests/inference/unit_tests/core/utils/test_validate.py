from inference.core.utils.validate import get_num_classes_from_model_prediction_shape


def test_get_num_classes_from_model_prediction_shape():
    prediction_len = 10
    num_classes = get_num_classes_from_model_prediction_shape(prediction_len)
    assert num_classes == 5


def test_get_num_classes_from_model_prediction_shape_with_masks():
    prediction_len = 42
    num_masks = 32
    num_classes = get_num_classes_from_model_prediction_shape(prediction_len, num_masks)
    assert num_classes == 5
