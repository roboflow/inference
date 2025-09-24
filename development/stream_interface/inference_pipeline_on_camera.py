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

from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
from inference.core.interfaces.stream.sinks import (
    UDPSink,
    display_image,
    multi_sink,
    render_boxes,
)
from inference.core.interfaces.stream.watchdog import (
    BasePipelineWatchDog,
    PipelineWatchDog,
)
from inference.core.utils.environment import str2bool
from inference.core.utils.preprocess import letterbox_image

STOP = False


def main(
    model_id: str,
    stream_id: int,
) -> None:
    global STOP
    watchdog = BasePipelineWatchDog()
    on_prediction = partial(
        render_boxes,
        display_statistics=True,
    )
    pipeline = InferencePipeline.init(
        model_id=model_id,
        video_reference=stream_id,
        on_prediction=on_prediction,
        watchdog=watchdog,
    )
    control_thread = Thread(target=command_thread, args=(pipeline, watchdog))
    control_thread.start()
    pipeline.start()
    STOP = True
    pipeline.join()


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
        "--model_id",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--stream_id",
        help=f"Id of a stream",
        required=False,
        type=int,
        default=0,
    )
    args = parser.parse_args()
    main(
        model_id=args.model_id,
        stream_id=args.stream_id,
    )
