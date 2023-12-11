from inference.core.models.utils.validate import (
    get_num_classes_from_model_prediction_shape,
)


def test_get_num_classes_from_model_prediction_shape() -> None:
    # given
    prediction_len = 10
    # when
    num_classes = get_num_classes_from_model_prediction_shape(prediction_len)
    # then
    assert num_classes == 5


def test_get_num_classes_from_model_prediction_shape_with_masks() -> None:
    # given
    prediction_len = 42
    num_masks = 32
    # when
    num_classes = get_num_classes_from_model_prediction_shape(prediction_len, num_masks)
    # then
    assert num_classes == 5


def test_get_num_classes_from_model_prediction_shape_with_keypoints() -> None:
    # given
    prediction_len = 57
    num_keypoints = 17
    # when
    num_classes = get_num_classes_from_model_prediction_shape(
        prediction_len, keypoints=num_keypoints
    )
    # then
    assert num_classes == 1


def test_get_num_classes_from_model_prediction_shape_with_keypoints_more_classes() -> (
    None
):
    # given
    prediction_len = 46
    num_keypoints = 12
    # when
    num_classes = get_num_classes_from_model_prediction_shape(
        prediction_len, keypoints=num_keypoints
    )
    # then
    assert num_classes == 5
