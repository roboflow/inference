import base64
from datetime import datetime

import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.common.deserializers import (
    deserialize_boolean_kind,
    deserialize_bytes_kind,
    deserialize_classification_prediction_kind,
    deserialize_detections_kind,
    deserialize_float_zero_to_one_kind,
    deserialize_integer_kind,
    deserialize_list_of_values_kind,
    deserialize_numpy_array,
    deserialize_optional_string_kind,
    deserialize_point_kind,
    deserialize_rgb_color_kind,
    deserialize_timestamp,
    deserialize_zone_kind,
)
from inference.core.workflows.errors import RuntimeInputError


def test_deserialize_detections_kind_when_sv_detections_given() -> None:
    # given
    detections = sv.Detections.empty()

    # when
    result = deserialize_detections_kind(
        parameter="my_param",
        detections=detections,
    )

    # then
    assert result is detections, "Expected object not to be touched"


def test_deserialize_detections_kind_when_invalid_data_type_given() -> None:
    # when
    with pytest.raises(RuntimeInputError):
        _ = deserialize_detections_kind(
            parameter="my_param",
            detections="INVALID",
        )


def test_deserialize_detections_kind_when_malformed_data_type_given() -> None:
    # when
    with pytest.raises(RuntimeInputError):
        _ = deserialize_detections_kind(
            parameter="my_param",
            detections={
                "image": {"height": 100, "width": 300},
                # lack of predictions
            },
        )


def test_deserialize_detections_kind_when_serialized_empty_detections_given() -> None:
    # given
    detections = {
        "image": {"height": 100, "width": 300},
        "predictions": [],
    }

    # when
    result = deserialize_detections_kind(
        parameter="my_param",
        detections=detections,
    )

    # then
    assert isinstance(result, sv.Detections)
    assert len(result) == 0


def test_deserialize_detections_kind_when_serialized_non_empty_object_detections_given() -> (
    None
):
    # given
    detections = {
        "image": {
            "width": 168,
            "height": 192,
        },
        "predictions": [
            {
                "data": "some",
                "width": 1.0,
                "height": 1.0,
                "x": 1.5,
                "y": 1.5,
                "confidence": 0.1,
                "class_id": 1,
                "tracker_id": 1,
                "class": "cat",
                "detection_id": "first",
                "parent_id": "image",
            },
        ],
    }

    # when
    result = deserialize_detections_kind(
        parameter="my_param",
        detections=detections,
    )

    # then
    assert isinstance(result, sv.Detections)
    assert len(result) == 1
    assert np.allclose(result.xyxy, np.array([[1, 1, 2, 2]]))
    assert result.data["class_name"] == np.array(["cat"])
    assert result.data["detection_id"] == np.array(["first"])
    assert result.data["parent_id"] == np.array(["image"])
    assert result.data["detection_id"] == np.array(["first"])
    assert np.allclose(result.data["image_dimensions"], np.array([[192, 168]]))


def test_deserialize_detections_kind_when_serialized_non_empty_instance_segmentations_given() -> (
    None
):
    # given
    detections = {
        "image": {
            "width": 168,
            "height": 192,
        },
        "predictions": [
            {
                "data": "some",
                "width": 1.0,
                "height": 1.0,
                "x": 1.5,
                "y": 1.5,
                "confidence": 0.1,
                "class_id": 1,
                "tracker_id": 1,
                "class": "cat",
                "detection_id": "first",
                "parent_id": "image",
                "points": [
                    {"x": 1.0, "y": 1.0},
                    {"x": 1.0, "y": 10.0},
                    {"x": 10.0, "y": 10.0},
                    {"x": 10.0, "y": 1.0},
                ],
            },
        ],
    }

    # when
    result = deserialize_detections_kind(
        parameter="my_param",
        detections=detections,
    )

    # then
    assert isinstance(result, sv.Detections)
    assert len(result) == 1
    assert np.allclose(result.xyxy, np.array([[1, 1, 2, 2]]))
    assert result.data["class_name"] == np.array(["cat"])
    assert result.data["detection_id"] == np.array(["first"])
    assert result.data["parent_id"] == np.array(["image"])
    assert result.data["detection_id"] == np.array(["first"])
    assert np.allclose(result.data["image_dimensions"], np.array([[192, 168]]))
    assert result.mask.shape == (1, 192, 168)


def test_deserialize_detections_kind_when_serialized_non_empty_keypoints_detections_given() -> (
    None
):
    # given
    detections = {
        "image": {
            "width": 168,
            "height": 192,
        },
        "predictions": [
            {
                "data": "some",
                "width": 1.0,
                "height": 1.0,
                "x": 1.5,
                "y": 1.5,
                "confidence": 0.1,
                "class_id": 1,
                "tracker_id": 1,
                "class": "cat",
                "detection_id": "first",
                "parent_id": "image",
                "keypoints": [
                    {
                        "class_id": 1,
                        "class": "nose",
                        "confidence": 0.1,
                        "x": 11.0,
                        "y": 11.0,
                    },
                    {
                        "class_id": 2,
                        "class": "ear",
                        "confidence": 0.2,
                        "x": 12.0,
                        "y": 13.0,
                    },
                    {
                        "class_id": 3,
                        "class": "eye",
                        "confidence": 0.3,
                        "x": 14.0,
                        "y": 15.0,
                    },
                ],
            },
        ],
    }

    # when
    result = deserialize_detections_kind(
        parameter="my_param",
        detections=detections,
    )

    # then
    assert isinstance(result, sv.Detections)
    assert len(result) == 1
    assert np.allclose(result.xyxy, np.array([[1, 1, 2, 2]]))
    assert result.data["class_name"] == np.array(["cat"])
    assert result.data["detection_id"] == np.array(["first"])
    assert result.data["parent_id"] == np.array(["image"])
    assert result.data["detection_id"] == np.array(["first"])
    assert np.allclose(result.data["image_dimensions"], np.array([[192, 168]]))
    assert (
        result.data["keypoints_class_id"]
        == np.array(
            [np.array([1, 2, 3])],
            dtype="object",
        )
    ).all()
    assert (
        result.data["keypoints_class_name"]
        == np.array(
            np.array(["nose", "ear", "eye"]),
            dtype="object",
        )
    ).all()
    assert np.allclose(
        result.data["keypoints_confidence"].astype(np.float64),
        np.array([[0.1, 0.2, 0.3]], dtype=np.float64),
    )


def test_deserialize_numpy_array_when_numpy_array_is_given() -> None:
    # given
    raw_array = np.array([1, 2, 3])

    # when
    result = deserialize_numpy_array(parameter="some", raw_array=raw_array)

    # then
    assert result is raw_array


def test_deserialize_numpy_array_when_serialized_array_is_given() -> None:
    # given
    raw_array = [1, 2, 3]

    # when
    result = deserialize_numpy_array(parameter="some", raw_array=raw_array)

    # then
    assert np.allclose(result, np.array([1, 2, 3]))


def test_deserialize_numpy_array_when_invalid_value_given() -> None:
    # when
    with pytest.raises(RuntimeInputError):
        _ = deserialize_numpy_array(parameter="some", raw_array="invalid")


def test_deserialize_optional_string_kind_when_empty_value_given() -> None:
    # when
    result = deserialize_optional_string_kind(parameter="some", value=None)

    # then
    assert result is None


def test_deserialize_optional_string_kind_when_string_given() -> None:
    # when
    result = deserialize_optional_string_kind(parameter="some", value="some")

    # then
    assert result == "some"


def test_deserialize_optional_string_kind_when_invalid_value_given() -> None:
    # when
    with pytest.raises(RuntimeInputError):
        _ = deserialize_optional_string_kind(parameter="some", value=b"some")


def test_deserialize_float_zero_to_one_kind_when_not_a_number_given() -> None:
    # when
    with pytest.raises(RuntimeInputError):
        _ = deserialize_float_zero_to_one_kind(parameter="some", value="some")


def test_deserialize_float_zero_to_one_kind_when_integer_given() -> None:
    # when
    result = deserialize_float_zero_to_one_kind(parameter="some", value=1)

    # then
    assert abs(result - 1.0) < 1e-5
    assert isinstance(result, float)


def test_deserialize_float_zero_to_one_kind_when_float_given() -> None:
    # when
    result = deserialize_float_zero_to_one_kind(parameter="some", value=0.5)

    # then
    assert abs(result - 0.5) < 1e-5
    assert isinstance(result, float)


def test_deserialize_float_zero_to_one_kind_when_value_out_of_range_given() -> None:
    # when
    with pytest.raises(RuntimeInputError):
        _ = deserialize_float_zero_to_one_kind(parameter="some", value=1.5)


def test_deserialize_list_of_values_kind_when_invalid_value_given() -> None:
    # when
    with pytest.raises(RuntimeInputError):
        _ = deserialize_list_of_values_kind(parameter="some", value=1.5)


def test_deserialize_list_of_values_kind_when_list_given() -> None:
    # when
    result = deserialize_list_of_values_kind(parameter="some", value=[1, 2, 3])

    # then
    assert result == [1, 2, 3]


def test_deserialize_list_of_values_kind_when_tuple_given() -> None:
    # when
    result = deserialize_list_of_values_kind(parameter="some", value=(1, 2, 3))

    # then
    assert result == [1, 2, 3]


def test_deserialize_boolean_kind_when_boolean_given() -> None:
    # when
    result = deserialize_boolean_kind(parameter="some", value=True)

    # then
    assert result is True


def test_deserialize_boolean_kind_when_invalid_value_given() -> None:
    # when
    with pytest.raises(RuntimeInputError):
        _ = deserialize_boolean_kind(parameter="some", value="True")


def test_deserialize_integer_kind_when_integer_given() -> None:
    # when
    result = deserialize_integer_kind(parameter="some", value=3)

    # then
    assert result == 3


def test_deserialize_integer_kind_when_invalid_value_given() -> None:
    # when
    with pytest.raises(RuntimeInputError):
        _ = deserialize_integer_kind(parameter="some", value=3.0)


def test_deserialize_classification_prediction_kind_when_valid_multi_class_prediction_given() -> (
    None
):
    # given
    prediction = {
        "image": {"height": 128, "width": 256},
        "predictions": [{"class_name": "A", "class_id": 0, "confidence": 0.3}],
        "top": "A",
        "confidence": 0.3,
        "parent_id": "some",
        "prediction_type": "classification",
        "inference_id": "some",
        "root_parent_id": "some",
    }

    # when
    result = deserialize_classification_prediction_kind(
        parameter="some",
        value=prediction,
    )

    # then
    assert result is prediction


def test_deserialize_classification_prediction_kind_when_valid_multi_label_prediction_given() -> (
    None
):
    # given
    prediction = {
        "image": {"height": 128, "width": 256},
        "predictions": {
            "a": {"confidence": 0.3, "class_id": 0},
            "b": {"confidence": 0.3, "class_id": 1},
        },
        "predicted_classes": ["a", "b"],
        "parent_id": "some",
        "prediction_type": "classification",
        "inference_id": "some",
        "root_parent_id": "some",
    }

    # when
    result = deserialize_classification_prediction_kind(
        parameter="some",
        value=prediction,
    )

    # then
    assert result is prediction


def test_deserialize_classification_prediction_kind_when_not_a_dictionary_given() -> (
    None
):
    # when
    with pytest.raises(RuntimeInputError):
        _ = deserialize_classification_prediction_kind(
            parameter="some",
            value="invalid",
        )


@pytest.mark.parametrize(
    "to_delete",
    [
        ["image"],
        ["predictions"],
        ["top", "predicted_classes"],
        ["confidence", "predicted_classes"],
    ],
)
def test_deserialize_classification_prediction_kind_when_required_keys_not_given(
    to_delete: list,
) -> None:
    # given
    prediction = {
        "image": {"height": 128, "width": 256},
        "predictions": [{"class_name": "A", "class_id": 0, "confidence": 0.3}],
        "top": "A",
        "confidence": 0.3,
        "predicted_classes": ["a", "b"],
        "parent_id": "some",
        "prediction_type": "classification",
        "inference_id": "some",
        "root_parent_id": "some",
    }
    for field in to_delete:
        del prediction[field]

    with pytest.raises(RuntimeInputError):
        _ = deserialize_classification_prediction_kind(
            parameter="some",
            value=prediction,
        )


def test_deserialize_zone_kind_when_valid_input_given() -> None:
    # given
    zone = [
        (1, 2),
        [3, 4],
        (5, 6),
    ]

    # when
    result = deserialize_zone_kind(parameter="some", value=zone)

    # then
    assert result == [
        (1, 2),
        [3, 4],
        (5, 6),
    ]


def test_deserialize_zone_kind_when_zone_misses_points() -> None:
    # given
    zone = [
        [3, 4],
        (5, 6),
    ]

    # when
    with pytest.raises(RuntimeInputError):
        _ = deserialize_zone_kind(parameter="some", value=zone)


def test_deserialize_zone_kind_when_zone_has_invalid_elements() -> None:
    # given
    zone = [[3, 4], (5, 6), "invalid"]

    # when
    with pytest.raises(RuntimeInputError):
        _ = deserialize_zone_kind(parameter="some", value=zone)


def test_deserialize_zone_kind_when_zone_defines_invalid_points() -> None:
    # given
    zone = [
        [3, 4],
        (5, 6, 3),
        (1, 2),
    ]

    # when
    with pytest.raises(RuntimeInputError):
        _ = deserialize_zone_kind(parameter="some", value=zone)


def test_deserialize_zone_kind_when_zone_defines_points_not_being_numbers() -> None:
    # given
    zone = [
        [3, 4],
        (5, 6),
        (1, "invalid"),
    ]

    # when
    with pytest.raises(RuntimeInputError):
        _ = deserialize_zone_kind(parameter="some", value=zone)


def test_deserialize_rgb_color_kind_when_invalid_value_given() -> None:
    # when
    with pytest.raises(RuntimeInputError):
        _ = deserialize_rgb_color_kind(parameter="some", value=1)


def test_deserialize_rgb_color_kind_when_string_given() -> None:
    # when
    result = deserialize_rgb_color_kind(parameter="some", value="#fff")

    # then
    assert result == "#fff"


def test_deserialize_rgb_color_kind_when_valid_tuple_given() -> None:
    # when
    result = deserialize_rgb_color_kind(parameter="some", value=(1, 2, 3))

    # then
    assert result == (1, 2, 3)


def test_deserialize_rgb_color_kind_when_to_short_tuple_given() -> None:
    # when
    with pytest.raises(RuntimeInputError):
        _ = deserialize_rgb_color_kind(parameter="some", value=(1, 2))


def test_deserialize_rgb_color_kind_when_to_long_tuple_given() -> None:
    # when
    result = deserialize_rgb_color_kind(parameter="some", value=(1, 2, 3, 4))

    # then
    assert result == (1, 2, 3)


def test_deserialize_rgb_color_kind_when_valid_list_given() -> None:
    # when
    result = deserialize_rgb_color_kind(parameter="some", value=[1, 2, 3])

    # then
    assert result == (1, 2, 3)


def test_deserialize_rgb_color_kind_when_to_short_list_given() -> None:
    # when
    with pytest.raises(RuntimeInputError):
        _ = deserialize_rgb_color_kind(parameter="some", value=[1, 2])


def test_deserialize_rgb_color_kind_when_to_long_list_given() -> None:
    # when
    result = deserialize_rgb_color_kind(parameter="some", value=[1, 2, 3, 4])

    # then
    assert result == (1, 2, 3)


def test_deserialize_point_kind_when_invalid_value_given() -> None:
    # when
    with pytest.raises(RuntimeInputError):
        _ = deserialize_point_kind(parameter="some", value=1)


def test_deserialize_point_kind_when_valid_tuple_given() -> None:
    # when
    result = deserialize_point_kind(parameter="some", value=(1, 2))

    # then
    assert result == (1, 2)


def test_deserialize_point_kind_when_to_short_tuple_given() -> None:
    # when
    with pytest.raises(RuntimeInputError):
        _ = deserialize_point_kind(parameter="some", value=(1,))


def test_deserialize_point_kind_when_to_long_tuple_given() -> None:
    # when
    result = deserialize_point_kind(parameter="some", value=(1, 2, 3, 4))

    # then
    assert result == (1, 2)


def test_deserialize_point_kind_when_valid_list_given() -> None:
    # when
    result = deserialize_point_kind(parameter="some", value=[1, 2])

    # then
    assert result == (1, 2)


def test_deserialize_point_kind_when_to_short_list_given() -> None:
    # when
    with pytest.raises(RuntimeInputError):
        _ = deserialize_point_kind(parameter="some", value=[1])


def test_deserialize_point_kind_when_to_long_list_given() -> None:
    # when
    result = deserialize_point_kind(parameter="some", value=[1, 2, 3, 4])

    # then
    assert result == (1, 2)


def test_deserialize_point_kind_when_point_element_is_not_number() -> None:
    # when
    with pytest.raises(RuntimeInputError):
        _ = deserialize_point_kind(parameter="some", value=[1, "invalid"])


def test_deserialize_bytes_kind_when_invalid_value_given() -> None:
    # when
    with pytest.raises(RuntimeInputError):
        _ = deserialize_bytes_kind(parameter="some", value=1)


def test_deserialize_bytes_kind_when_bytes_given() -> None:
    # when
    result = deserialize_bytes_kind(parameter="some", value=b"abcd")

    # then
    assert result == b"abcd"


def test_deserialize_bytes_kind_when_base64_string_given() -> None:
    # given
    data = base64.b64encode(b"data").decode("utf-8")

    # when
    result = deserialize_bytes_kind(parameter="some", value=data)

    # then
    assert result == b"data"


def test_deserialize_timestamp_when_valid_input_provided() -> None:
    # given
    timestamp = datetime.now()

    # when
    result = deserialize_timestamp(parameter="some", value=timestamp)

    # then
    assert result == timestamp


def test_deserialize_timestamp_when_invalid_input_provided() -> None:
    # when
    with pytest.raises(RuntimeInputError):
        _ = deserialize_timestamp(parameter="some", value="invalid")
