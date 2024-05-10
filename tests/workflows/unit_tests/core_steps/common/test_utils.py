import numpy as np

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
            "image": {"height": 100, "width": 200},
            "predictions": [
                {"class": "a", "field": "b"},
                {"class": "b", "field": "b"},
            ],
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
            "image": {"height": 100, "width": 200},
            "predictions": [
                {"class": "a", "field": "b"},
                {"class": "b", "field": "b"},
            ],
        }
    ]


def test_filter_out_unwanted_classes_from_predictions_detections_when_there_are_classes_to_be_filtered_out() -> (
    None
):
    # given
    predictions = [
        {
            "image": {"height": 100, "width": 200},
            "predictions": [
                {"class": "a", "field": "b"},
                {"class": "b", "field": "b"},
            ],
        },
        {
            "image": {"height": 100, "width": 200},
            "predictions": [
                {"class": "c", "field": "b"},
            ],
        },
    ]

    # when
    result = filter_out_unwanted_classes_from_predictions_detections(
        predictions=predictions,
        classes_to_accept=["b"],
    )

    # then
    assert result == [
        {
            "image": {"height": 100, "width": 200},
            "predictions": [
                {"class": "b", "field": "b"},
            ],
        },
        {"image": {"height": 100, "width": 200}, "predictions": []},
    ]


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
