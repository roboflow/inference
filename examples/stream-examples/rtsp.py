import cv2
import supervision as sv

import inference

annotator = sv.BoxAnnotator()


def render(predictions, image):
    print(predictions)
    image = annotator.annotate(
        scene=image, detections=sv.Detections.from_inference(predictions)
    )

    cv2.imshow("Prediction", image)
    cv2.waitKey(1)

# replace with your own RTSP stream
inference.Stream(
    source="rtsp://127.0.0.1:8000/test",
    model="microsoft-coco/9",
    output_channel_order="BGR",
    use_main_thread=True,
    on_prediction=render,
)
