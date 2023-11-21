import os

import cv2
import supervision as sv

from inference.core.interfaces.stream.inference_pipeline import InferencePipeline

annotator = sv.BoxAnnotator()


def on_prediction(image, predictions):
    labels = [p["class"] for p in predictions["predictions"]]
    detections = sv.Detections.from_roboflow(predictions)
    cv2.imshow(
        "Prediction",
        annotator.annotate(scene=image, detections=detections, labels=labels),
    ),
    cv2.waitKey(1)


def main() -> None:
    pipeline = InferencePipeline.init(
        api_key=os.environ["API_KEY"],
        model_id="rock-paper-scissors-sxsw/11",
        video_reference=0,
        on_prediction=on_prediction,
        max_fps=None,
    )
    pipeline.start()
    pipeline.join()


if __name__ == "__main__":
    main()
