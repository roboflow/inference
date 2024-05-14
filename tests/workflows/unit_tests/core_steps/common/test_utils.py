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


def test_filter_out_unwanted_classes_from_predictions_detections_when_there_are_classes_to_be_filtered_out():
    # given
    predictions = [
        {
            "predictions": sv.Detections.from_inference(
                {
                    "image": {"height": 100, "width": 200},
                    "predictions": [
                        {"x": 11, "y": 12, "width": 13, "height": 14, "class_id": 1, "class": "a", "confidence": 0.5},
                        {"x": 15, "y": 16, "width": 17, "height": 18, "class_id": 2, "class": "b", "confidence": 0.6},
                    ],
                },
            )
        },
        {
            "predictions": sv.Detections.from_inference(
                {
                    "image": {"height": 100, "width": 200},
                    "predictions": [
                        {"x": 21, "y": 22, "width": 23, "height": 24, "class_id": 1, "class": "a", "confidence": 0.7},
                    ],
                },
            )
        }
    ]
    for p in predictions:
        p["predictions"]["field"] = ["b"] * len(p["predictions"])

    # when
    result = filter_out_unwanted_classes_from_predictions_detections(
        predictions=predictions,
        classes_to_accept=["b"],
    )

    # then
    expected_result = [
        {
            "predictions": sv.Detections.from_inference(
                {
                    "image": {"height": 100, "width": 200},
                    "predictions": [
                        {"x": 15, "y": 16, "width": 17, "height": 18, "class_id": 2, "class": "b", "confidence": 0.6},
                    ],
                },
            )
        },
        {"predictions": sv.Detections.from_inference({"image": {"height": 100, "width": 200}, "predictions": []})},
    ]
    for p in expected_result:
        p["predictions"]["field"] = ["b"] * len(p["predictions"])
        if not p["predictions"]:
            empty_detections = p["predictions"]
            empty_detections.xyxy.dtype = np.float64
            empty_detections.xyxy.shape = (0, 4)
            empty_detections.confidence.dtype = np.float64
            empty_detections.data["field"].dtype = "<U1"
            empty_detections.data["class_name"].dtype = "<U1"

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
