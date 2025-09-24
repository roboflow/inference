from inference.core.entities.responses.inference import ObjectDetectionPrediction
from inference.core.utils.visualisation import bbox_to_points


def test_bbox_to_points() -> None:
    # given
    bbox = ObjectDetectionPrediction(
        **{
            "x": 10.3,
            "y": 20.3,
            "width": 10.0,
            "height": 8.0,
            "confidence": 0.9,
            "class": "a",
            "class_confidence": None,
            "class_id": 1,
            "tracker_id": None,
        }
    )

    # when
    result = bbox_to_points(box=bbox)

    # then
    assert result == ((5, 16), (15, 24))
