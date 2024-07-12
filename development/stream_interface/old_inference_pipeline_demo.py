import argparse
import os
import subprocess
from datetime import datetime
from functools import partial
from threading import Thread
from typing import Optional, Union

import cv2
import numpy as np
import supervision as sv

from inference import Stream
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes, display_image
from inference.core.interfaces.stream.watchdog import (
    BasePipelineWatchDog,
    PipelineWatchDog,
)
from inference.core.utils.environment import str2bool
from inference.core.utils.preprocess import letterbox_image


STOP = False

MODELS = {
    "a": "microsoft-coco/8",
    "b": "microsoft-coco/9",
    "c": "microsoft-coco/10",
    "d": "microsoft-coco/11",
    "e": "microsoft-coco/12",
}

STREAM_SERVER_URL = os.getenv("STREAM_SERVER", "rtsp://localhost:8554")
ANNOTATOR = sv.BoxAnnotator()
FPS_MONITOR = sv.FPSMonitor()


def render(predictions: dict, image: np.ndarray) -> None:
    FPS_MONITOR.tick()
    fps = FPS_MONITOR()
    image = ANNOTATOR.annotate(
        scene=image, detections=sv.Detections.from_roboflow(predictions)
    )
    image = letterbox_image(image, desired_size=(1280, 720))
    fps = round(fps, 2)
    image = cv2.putText(
        image, f"THROUGHPUT: {fps}", (10, image.shape[0] - 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    )
    cv2.imshow("Prediction", image)
    cv2.waitKey(1)


def main(
    model_type: str,
    stream_id: int,
) -> None:
    Stream(
        source=f"{STREAM_SERVER_URL}/live{stream_id}.stream",
        model=MODELS[model_type.lower()],
        output_channel_order="BGR",
        use_main_thread=True,
        on_prediction=render,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference pipeline demo")
    parser.add_argument(
        "--model_type",
        help=f"Type of a model from {list(MODELS.keys())}",
        required=False,
        type=str,
        default="a",
    )
    parser.add_argument(
        "--stream_id",
        help=f"Id of a stream",
        required=True,
        type=int,
    )
    args = parser.parse_args()
    main(
        model_type=args.model_type,
        stream_id=args.stream_id,
    )
