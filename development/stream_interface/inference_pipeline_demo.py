import argparse
import os
import subprocess
from functools import partial
from threading import Thread
from typing import Optional, Union

import numpy as np
import supervision as sv

from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
from inference.core.interfaces.stream.watchdog import (
    BasePipelineWatchDog,
    PipelineWatchDog,
)
from inference.core.utils.preprocess import letterbox_image

annotator = sv.BoxAnnotator()

STOP = False

MODELS = {
    "a": "microsoft-coco/8",
    "b": "microsoft-coco/9",
    "c": "microsoft-coco/10",
    "d": "microsoft-coco/11",
    "e": "microsoft-coco/12",
}

STREAM_SERVER_URL = os.getenv("STREAM_SERVER", "rtsp://localhost:8554")


def main(
    model_type: str,
    stream_id: int,
    max_fps: Optional[Union[int, float]],
) -> None:
    global STOP
    watchdog = BasePipelineWatchDog()
    ffmpeg_process = open_ffmpeg_stream_process(stream_id=stream_id)
    pipeline = InferencePipeline.init(
        api_key=os.environ["API_KEY"],
        model_id=MODELS[model_type.lower()],
        video_reference=f"{STREAM_SERVER_URL}/live{stream_id}.stream",
        on_prediction=partial(on_prediction, ffmpeg_process=ffmpeg_process),
        max_fps=max_fps,
        watchdog=watchdog,
    )
    control_thread = Thread(target=command_thread, args=(pipeline, watchdog))
    control_thread.start()
    pipeline.start()
    STOP = True
    pipeline.join()
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()


def on_prediction(image, predictions, ffmpeg_process):
    labels = [p["class"] for p in predictions["predictions"]]
    detections = sv.Detections.from_roboflow(predictions)
    image = annotator.annotate(scene=image, detections=detections, labels=labels)
    image = letterbox_image(image, desired_size=(640, 480))[:, :, ::-1]
    ffmpeg_process.stdin.write(image.astype(np.uint8).tobytes())


def open_ffmpeg_stream_process(stream_id: int) -> subprocess.Popen:
    args = (
        "ffmpeg -re -stream_loop -1 -f rawvideo -pix_fmt "
        "rgb24 -s 640x480 -i pipe:0 -pix_fmt yuv420p "
        f"-f rtsp -rtsp_transport tcp {STREAM_SERVER_URL}/predictions{stream_id}.stream"
    ).split()
    return subprocess.Popen(args, stdin=subprocess.PIPE)


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
    parser.add_argument(
        "--max_fps",
        help=f"Limit on FPS",
        required=False,
        type=int,
        default=None,
    )
    args = parser.parse_args()
    main(
        model_type=args.model_type,
        stream_id=args.stream_id,
        max_fps=args.max_fps,
    )
