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
    deserialize_image_kind,
    deserialize_integer_kind,
    deserialize_list_of_values_kind,
    deserialize_numpy_array,
    deserialize_optional_string_kind,
    deserialize_parent_origin,
    deserialize_point_kind,
    deserialize_rgb_color_kind,
    deserialize_timestamp,
    deserialize_zone_kind,
)
from inference.core.workflows.errors import RuntimeInputError
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    OriginCoordinatesSystem,
    ParentOrigin,
    WorkflowImageData,
)


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


def test_deserialize_image_kind_when_workflow_image_data_given() -> None:
    # given
    workflow_image_data = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="original_parent"),
        numpy_image=np.zeros((100, 100, 3), dtype=np.uint8),
    )

    # when
    result = deserialize_image_kind(parameter="test_param", image=workflow_image_data)

    # then
    assert result is workflow_image_data, "Expected object not to be touched"


def test_deserialize_image_kind_when_numpy_array_given() -> None:
    # given
    image_array = np.zeros((100, 100, 3), dtype=np.uint8)

    # when
    result = deserialize_image_kind(parameter="test_param", image=image_array)

    # then
    assert isinstance(result, WorkflowImageData)
    assert np.array_equal(result.numpy_image, image_array)


def test_deserialize_image_kind_when_dict_with_numpy_value_given() -> None:
    # given
    image_array = np.zeros((100, 100, 3), dtype=np.uint8)
    image_dict = {"value": image_array}

    # when
    result = deserialize_image_kind(parameter="test_param", image=image_dict)

    # then
    assert isinstance(result, WorkflowImageData)
    assert np.array_equal(result.numpy_image, image_array)


def test_deserialize_image_kind_when_dict_with_base64_value_given() -> None:
    # given
    # simple 1x1 pixel base64 encoded image
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    image_dict = {"value": base64_image}

    # when
    result = deserialize_image_kind(parameter="test_param", image=image_dict)

    # then
    assert isinstance(result, WorkflowImageData)
    assert result.base64_image == base64_image


def test_deserialize_image_kind_when_base64_string_given() -> None:
    # given
    # simple 1x1 pixel base64 encoded image
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    # when
    result = deserialize_image_kind(parameter="test_param", image=base64_image)

    # then
    assert isinstance(result, WorkflowImageData)
    assert result.base64_image == base64_image


def test_deserialize_image_kind_when_parent_id_given() -> None:
    # given
    image_array = np.zeros((100, 100, 3), dtype=np.uint8)
    image_dict = {"value": image_array, "parent_id": "custom_parent"}

    # when
    result = deserialize_image_kind(parameter="test_param", image=image_dict)

    # then
    assert isinstance(result, WorkflowImageData)
    assert result.parent_metadata.parent_id == "custom_parent"
    assert np.array_equal(result.numpy_image, image_array)


def test_deserialize_image_kind_when_parent_id_not_given() -> None:
    # given
    image_array = np.zeros((100, 100, 3), dtype=np.uint8)
    image_dict = {"value": image_array}

    # when
    result = deserialize_image_kind(parameter="test_param", image=image_dict)

    # then
    assert isinstance(result, WorkflowImageData)
    assert result.parent_metadata.parent_id == "test_param"
    assert np.array_equal(result.numpy_image, image_array)


def test_deserialize_parent_origin_when_parent_origin_instance_given() -> None:
    # given
    parent_origin = ParentOrigin(
        offset_x=100,
        offset_y=200,
        width=800,
        height=600,
    )

    # when
    result = deserialize_parent_origin(
        parameter="test_param", parent_origin=parent_origin
    )

    # then
    assert result is parent_origin


def test_deserialize_parent_origin_when_valid_dict_given() -> None:
    # given
    parent_origin_dict = {
        "offset_x": 100,
        "offset_y": 200,
        "width": 800,
        "height": 600,
    }

    # when
    result = deserialize_parent_origin(
        parameter="test_param", parent_origin=parent_origin_dict
    )

    # then
    assert isinstance(result, ParentOrigin)
    assert result.offset_x == 100
    assert result.offset_y == 200
    assert result.width == 800
    assert result.height == 600


def test_deserialize_parent_origin_when_invalid_type_given() -> None:
    # when
    with pytest.raises(RuntimeInputError):
        _ = deserialize_parent_origin(parameter="test_param", parent_origin="invalid")


def test_deserialize_parent_origin_when_malformed_dict_given() -> None:
    # given
    malformed_dict = {
        "offset_x": 100,
        # missing offset_y
        "width": 800,
        "height": 600,
    }

    # when
    with pytest.raises(RuntimeInputError):
        _ = deserialize_parent_origin(
            parameter="test_param", parent_origin=malformed_dict
        )


def test_deserialize_parent_origin_when_dict_with_invalid_values_given() -> None:
    # given
    invalid_dict = {
        "offset_x": 100,
        "offset_y": 200,
        "width": 0,  # invalid: must be > 0
        "height": 600,
    }

    # when
    with pytest.raises(RuntimeInputError):
        _ = deserialize_parent_origin(
            parameter="test_param", parent_origin=invalid_dict
        )


def test_deserialize_image_kind_with_parent_origin_metadata() -> None:
    # given
    image_array = np.zeros((100, 100, 3), dtype=np.uint8)
    image_dict = {
        "value": image_array,
        "parent_id": "crop_id",
        "parent_origin": {
            "offset_x": 50,
            "offset_y": 75,
            "width": 800,
            "height": 600,
        },
    }

    # when
    result = deserialize_image_kind(parameter="test_param", image=image_dict)

    # then
    assert isinstance(result, WorkflowImageData)
    assert result.parent_metadata.parent_id == "crop_id"
    assert result.parent_metadata.origin_coordinates == OriginCoordinatesSystem(
        left_top_x=50,
        left_top_y=75,
        origin_width=800,
        origin_height=600,
    )
    assert np.array_equal(result.numpy_image, image_array)


def test_deserialize_image_kind_with_root_parent_origin_metadata() -> None:
    # given
    image_array = np.zeros((100, 100, 3), dtype=np.uint8)
    image_dict = {
        "value": image_array,
        "parent_id": "crop_id",
        "parent_origin": {
            "offset_x": 50,
            "offset_y": 75,
            "width": 800,
            "height": 600,
        },
        "root_parent_id": "original_image",
        "root_parent_origin": {
            "offset_x": 150,
            "offset_y": 200,
            "width": 1920,
            "height": 1080,
        },
    }

    # when
    result = deserialize_image_kind(parameter="test_param", image=image_dict)

    # then
    assert isinstance(result, WorkflowImageData)
    assert result.parent_metadata.parent_id == "crop_id"
    assert result.parent_metadata.origin_coordinates == OriginCoordinatesSystem(
        left_top_x=50,
        left_top_y=75,
        origin_width=800,
        origin_height=600,
    )
    assert result.workflow_root_ancestor_metadata.parent_id == "original_image"
    assert (
        result.workflow_root_ancestor_metadata.origin_coordinates
        == OriginCoordinatesSystem(
            left_top_x=150,
            left_top_y=200,
            origin_width=1920,
            origin_height=1080,
        )
    )
    assert np.array_equal(result.numpy_image, image_array)


def test_deserialize_detections_kind_with_parent_origin_metadata() -> None:
    # given
    detections = {
        "image": {
            "width": 1200,
            "height": 900,
        },
        "predictions": [
            {
                "width": 10,
                "height": 20,
                "x": 150,
                "y": 200,
                "confidence": 0.1,
                "class_id": 1,
                "class": "cat",
                "detection_id": "first",
                "parent_id": "crop_id",
                "parent_origin": {
                    "offset_x": 50,
                    "offset_y": 75,
                    "width": 800,
                    "height": 600,
                },
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
    assert result.data["parent_id"] == np.array(["crop_id"])
    assert "parent_coordinates" in result.data
    assert np.allclose(result.data["parent_coordinates"], np.array([[50, 75]]))
    assert "parent_dimensions" in result.data
    assert np.allclose(result.data["parent_dimensions"], np.array([[600, 800]]))


def test_deserialize_detections_kind_with_root_parent_origin_metadata() -> None:
    # given
    detections = {
        "image": {
            "width": 1200,
            "height": 900,
        },
        "predictions": [
            {
                "width": 10,
                "height": 20,
                "x": 150,
                "y": 200,
                "confidence": 0.1,
                "class_id": 1,
                "class": "cat",
                "detection_id": "first",
                "parent_id": "crop_id",
                "parent_origin": {
                    "offset_x": 50,
                    "offset_y": 75,
                    "width": 800,
                    "height": 600,
                },
                "root_parent_id": "original_image",
                "root_parent_origin": {
                    "offset_x": 150,
                    "offset_y": 200,
                    "width": 900,
                    "height": 400,
                },
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
    assert result.data["parent_id"] == np.array(["crop_id"])
    assert "parent_coordinates" in result.data
    assert np.allclose(result.data["parent_coordinates"], np.array([[50, 75]]))
    assert "parent_dimensions" in result.data
    assert np.allclose(result.data["parent_dimensions"], np.array([[600, 800]]))
    assert result.data["root_parent_id"] == np.array(["original_image"])
    assert "root_parent_coordinates" in result.data
    assert np.allclose(result.data["root_parent_coordinates"], np.array([[150, 200]]))
    assert "root_parent_dimensions" in result.data
    assert np.allclose(result.data["root_parent_dimensions"], np.array([[400, 900]]))
