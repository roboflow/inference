import os
from threading import Thread

import cv2
import supervision as sv

from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
from inference.core.interfaces.stream.watchdog import (
    BasePipelineWatchDog,
    PipelineWatchDog,
)

annotator = sv.BoxAnnotator()

STOP = False


def on_prediction(image, predictions):
    labels = [p["class"] for p in predictions["predictions"]]
    detections = sv.Detections.from_roboflow(predictions)
    image = annotator.annotate(scene=image, detections=detections, labels=labels)
    cv2.imshow("Prediction", image)
    cv2.waitKey(1)


def main() -> None:
    global STOP
    watchdog = BasePipelineWatchDog()
    pipeline = InferencePipeline.init(
        api_key=os.environ["API_KEY"],
        model_id="microsoft-coco/9",
        video_reference="rtsp://localhost:8554/live0.stream",
        on_prediction=on_prediction,
        max_fps=None,
        watchdog=watchdog,
    )
    control_thread = Thread(target=command_thread, args=(pipeline, watchdog))
    control_thread.start()
    pipeline.start()
    STOP = True
    pipeline.join()
    cv2.destroyAllWindows()


def command_thread(pipeline: InferencePipeline, watchdog: PipelineWatchDog) -> None:
    global STOP
    while not STOP:
        key = input()
        if key == "i":
            print(watchdog.get_report())
        if key == "t":
            pipeline.terminate()
            STOP = True
        elif key == "p":
            pipeline.pause_stream()
        elif key == "m":
            pipeline.mute_stream()
        elif key == "r":
            pipeline.resume_stream()


if __name__ == "__main__":
    main()
