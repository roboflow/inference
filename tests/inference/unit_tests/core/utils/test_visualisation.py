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


def test_polygon_visualization_uses_image_coordinates_without_mutation(
    monkeypatch,
):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import numpy as np

    from inference.core.entities.responses.inference import (
        InstanceSegmentationPrediction,
        Point,
    )
    from inference.core.utils import visualisation

    box = InstanceSegmentationPrediction(
        x=50,
        y=50,
        width=40,
        height=40,
        confidence=0.9,
        class_id=0,
        points=[Point(x=25, y=32)],
        **{"class": "car"},
    )
    response = SimpleNamespace(predictions=[box])
    scene = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_polygon = MagicMock(return_value=scene)
    monkeypatch.setattr(visualisation, "load_image_rgb", lambda _: scene)
    monkeypatch.setattr(
        visualisation, "draw_instance_segmentation_points", draw_polygon
    )

    result = visualisation.draw_detection_predictions(
        inference_request=SimpleNamespace(
            image=None, visualization_stroke_width=1, visualization_labels=False
        ),
        inference_response=response,
        colors={"car": "#ffffff"},
    )

    assert isinstance(result, bytes)
    assert draw_polygon.call_args.kwargs["points"] == [Point(x=25, y=32)]
    assert box.points == [Point(x=25, y=32)]
