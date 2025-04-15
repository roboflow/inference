import numpy as np
import supervision as sv
from inference import get_model
from tests.google_colab.conftest import PLAYER_DETECTION_MODEL_ID, PLAYER_CLASS_ID, FOOTBALL_FIELD_DETECTOR_MODEL_ID


def test_cropping_players(
    reference_video: str,
    roboflow_api_key: str
) -> None:
    # given
    player_detection_model = get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key=roboflow_api_key)
    frame_generator = sv.get_video_frames_generator(source_path=reference_video, stride=30)

    # when
    crops = []
    for frame in frame_generator:
        result = player_detection_model.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        detections = detections[detections.class_id == PLAYER_CLASS_ID]
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
        crops += players_crops

    # then
    assert len(crops) >= 470


def test_detecting_football_field(
    reference_video: str,
    roboflow_api_key: str
) -> None:
    # given
    field_detector_model = get_model(FOOTBALL_FIELD_DETECTOR_MODEL_ID)
    frame_generator = sv.get_video_frames_generator(reference_video)
    frame = next(frame_generator)
    vertex_annotator = sv.VertexAnnotator(
        color=sv.Color.from_hex("#FF1493"),
        radius=8
    )
    result = field_detector_model.infer(frame, confidence=0.3)[0]
    key_points = sv.KeyPoints.from_inference(result)
    filtered_key_points = key_points.confidence[0] > 0.5
    frame_reference_points = key_points.xy[0][filtered_key_points]
    frame_reference_key_points = sv.KeyPoints(
        xy=frame_reference_points[np.newaxis, ...]
    )

    annotated_frame = frame.copy()
    annotated_frame = vertex_annotator.annotate(
        scene=annotated_frame,
        key_points=frame_reference_key_points,
    )

    assert not np.allclose(frame, annotated_frame)
    assert frame_reference_key_points.xy.shape == (1, 16, 2)
