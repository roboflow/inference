from inference.enterprise.workflows.core_steps.common.utils import (
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
