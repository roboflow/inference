import numpy as np
import supervision as sv

from inference.core.workflows.core_steps.common.utils import (
    extract_origin_size_from_images_batch,
    filter_out_unwanted_classes_from_predictions_detections,
)


def test_filter_out_unwanted_classes_from_predictions_detections_when_empty_results_provided() -> (
    None
):
    # when
    result = filter_out_unwanted_classes_from_predictions_detections(
        predictions=[], classes_to_accept=["a", "b"]
    )

    # then
    assert result == []


def test_filter_out_unwanted_classes_from_predictions_detections_when_no_class_filter_provided() -> (
    None
):
    # given
    predictions = [
        {
            "predictions": sv.Detections(
                    xyxy=np.array([[4.5, 5, 17.5, 19], [6.5, 7, 23.5, 25]], dtype=np.float64),
                    class_id=np.array([1, 1]),
                    confidence=np.array([0.5, 0.6], dtype=np.float64),
                    data={
                        "class_name" : np.array(["a", "a"]),
                        "field": np.array(["a", "a"])
                    }
                )
        }
    ]

    # when
    result = filter_out_unwanted_classes_from_predictions_detections(
        predictions=predictions,
        classes_to_accept=None,
    )

    # then
    assert result == [
        {
            "predictions": sv.Detections(
                    xyxy=np.array([[4.5, 5, 17.5, 19], [6.5, 7, 23.5, 25]], dtype=np.float64),
                    class_id=np.array([1, 1]),
                    confidence=np.array([0.5, 0.6], dtype=np.float64),
                    data={
                        "class_name" : np.array(["a", "a"]),
                        "field": np.array(["a", "a"])
                    }
                )
        }
    ]


def test_filter_out_unwanted_classes_from_predictions_detections_when_there_are_classes_to_be_filtered_out():
    # given
    predictions = [
        {
            "predictions": sv.Detections(
                    xyxy=np.array([[4.5, 5, 17.5, 19], [6.5, 7, 23.5, 25]], dtype=np.float64),
                    class_id=np.array([1, 2]),
                    confidence=np.array([0.5, 0.6], dtype=np.float64),
                    data={
                        "class_name" : np.array(["a", "b"]),
                        "field": np.array(["a", "a"])
                    }
                )
        },
        {
            "predictions": sv.Detections(
                    xyxy=np.array([[9.5, 10, 32.5, 34]], dtype=np.float64),
                    class_id=np.array([1]),
                    confidence=np.array([0.7], dtype=np.float64),
                    data={
                        "class_name" : np.array(["a"]),
                        "field": np.array(["a"])
                    }
                )
        }
    ]

    # when
    result = filter_out_unwanted_classes_from_predictions_detections(
        predictions=predictions,
        classes_to_accept=["b"],
    )

    # then
    expected_result = [
        {
            "predictions": sv.Detections(
                    xyxy=np.array([[6.5, 7, 23.5, 25]], dtype=np.float64),
                    class_id=np.array([2]),
                    confidence=np.array([0.6], dtype=np.float64),
                    data={
                        "class_name" : np.array(["b"]),
                        "field": np.array(["a"])
                    }
                )
        },
        {
            "predictions": sv.Detections(
                    xyxy=np.array([[9.5, 10, 32.5, 34]], dtype=np.float64),
                    class_id=np.array([1]),
                    confidence=np.array([0.7], dtype=np.float64),
                    data={
                        "class_name" : np.array(["a"]),
                        "field": np.array(["a"])
                    }
                )[[]]
        }
    ]

    assert result == expected_result


def test_extract_origin_size_from_images() -> None:
    # given
    input_images = [
        {
            "type": "url",
            "value": "https://some/image.jpg",
            "origin_coordinates": {
                "origin_image_size": {"height": 1000, "width": 1000}
            },
        },
        {
            "type": "numpy_object",
            "value": np.zeros((192, 168, 3), dtype=np.uint8),
        },
    ]
    decoded_image = [
        np.zeros((200, 200, 3), dtype=np.uint8),
        np.zeros((192, 168, 3), dtype=np.uint8),
    ]

    # when
    result = extract_origin_size_from_images_batch(
        input_images=input_images,
        decoded_images=decoded_image,
    )

    # then
    assert result == [{"height": 1000, "width": 1000}, {"height": 192, "width": 168}]
