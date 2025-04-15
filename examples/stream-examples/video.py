import cv2
import supervision as sv

import inference

box_annotator = sv.BoxAnnotator(color=sv.Color(103, 6, 206))
trace_annotator = sv.TraceAnnotator(color=sv.Color(163, 81, 251), thickness=6)
byte_tracker = sv.ByteTrack()


def render(detections, image):
    detections = sv.Detections.from_inference(detections)
    detections = byte_tracker.update_with_detections(detections)
    image = trace_annotator.annotate(scene=image, detections=detections)
    image = box_annotator.annotate(scene=image, detections=detections)

    cv2.imshow("Prediction", image)
    cv2.waitKey(1)


inference.Stream(
    source="people-walking-bw.mp4",
    model="microsoft-coco/9",
    output_channel_order="BGR",
    use_main_thread=True,
    on_prediction=render,
)
