import matplotlib.pyplot as plt
import numpy as np
import supervision as sv

from inference import get_model


def test_basic_object_detection_visualization(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov8n-640")
    box_annotator = sv.BoxAnnotator(
        color=sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700']),
        thickness=2
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700']),
        text_color=sv.Color.from_hex('#000000')
    )

    # when
    result_raw = model.infer(reference_image)[0]
    detections = sv.Detections.from_inference(result_raw)
    annotated_frame = reference_image.copy()
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(detections["class_name"], detections.confidence)
    ]
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )

    # then
    assert isinstance(annotated_frame, np.ndarray)
    assert not np.allclose(reference_image, annotated_frame)


def test_basic_instance_segmentation_visualization(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov8n-seg-640")
    mask_annotator = sv.MaskAnnotator(
        color=sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700']),
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700']),
        text_color=sv.Color.from_hex('#000000')
    )

    # when
    result_raw = model.infer(reference_image)[0]
    detections = sv.Detections.from_inference(result_raw)
    annotated_frame = reference_image.copy()
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(detections["class_name"], detections.confidence)
    ]
    annotated_frame = mask_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )

    # then
    assert isinstance(annotated_frame, np.ndarray)
    assert not np.allclose(reference_image, annotated_frame)


def test_basic_pose_estimation_visualization(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov8n-pose-640")
    vertex_annotator = sv.VertexAnnotator()
    label_annotator = sv.VertexLabelAnnotator()

    # when
    result_raw = model.infer(reference_image)[0]
    key_points = sv.KeyPoints.from_inference(result_raw)
    annotated_frame = reference_image.copy()
    annotated_frame = vertex_annotator.annotate(
        scene=annotated_frame,
        key_points=key_points
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        key_points=key_points,
    )

    # then
    assert isinstance(annotated_frame, np.ndarray)
    assert not np.allclose(reference_image, annotated_frame)


def test_basic_tracking(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov8n-pose-640")
    tracker = sv.ByteTrack()
    tracker.reset()
    tracker_annotator = sv.TraceAnnotator()
    box_annotator = sv.BoxAnnotator(
        color=sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700']),
        thickness=2
    )

    # when
    for _ in range(10):
        result_raw = model.infer(reference_image)[0]
        detections = sv.Detections.from_inference(result_raw)
        detections = tracker.update_with_detections(detections=detections)

        # then
        annotated_frame = tracker_annotator.annotate(reference_image.copy(), detections)
        annotated_frame = box_annotator.annotate(annotated_frame, detections)
        assert detections.tracker_id is not None
        assert not np.allclose(reference_image, annotated_frame)
