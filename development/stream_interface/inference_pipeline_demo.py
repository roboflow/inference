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
from inference.core.interfaces.stream.sinks import render_boxes, display_image, UDPSink, multi_sink
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
UDP_SERVER_HOST = os.getenv("UDP_SERVER_HOST", "127.0.0.1")
UDP_SERVER_PORT = int(os.getenv("UDP_SERVER_PORT", "9999"))


def main(
    model_type: str,
    stream_id: int,
    max_fps: Optional[Union[int, float]],
    enable_stats: bool,
    output_type: str,
) -> None:
    global STOP
    watchdog = BasePipelineWatchDog()
    ffmpeg_process = None
    sinks = []
    if "video_stream" in output_type:
        ffmpeg_process = open_ffmpeg_stream_process(stream_id=stream_id)
        on_frame_rendered = partial(stream_prediction, ffmpeg_process=ffmpeg_process)
        on_prediction = partial(
            render_boxes,
            display_statistics=enable_stats,
            on_frame_rendered=on_frame_rendered,
        )
        sinks.append(on_prediction)
    if "udp_stream" in output_type:
        udp_sink = UDPSink.init(ip_address=UDP_SERVER_HOST, port=UDP_SERVER_PORT)
        on_prediction = udp_sink.send_predictions
        sinks.append(on_prediction)
    if "display" in output_type:
        on_prediction = partial(
            render_boxes,
            display_statistics=enable_stats,
        )
        sinks.append(on_prediction)
    on_prediction = partial(multi_sink, sinks=sinks)
    pipeline = InferencePipeline.init(
        model_id=MODELS[model_type.lower()],
        video_reference=f"{STREAM_SERVER_URL}/live{stream_id}.stream",
        on_prediction=on_prediction,
        max_fps=max_fps,
        watchdog=watchdog,
    )
    control_thread = Thread(target=command_thread, args=(pipeline, watchdog))
    control_thread.start()
    pipeline.start()
    STOP = True
    pipeline.join()
    if ffmpeg_process is not None:
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()


def stream_prediction(image: np.ndarray, ffmpeg_process: subprocess.Popen) -> None:
    ffmpeg_process.stdin.write(image[:, :, ::-1].astype(np.uint8).tobytes())


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
    parser.add_argument(
        "--enable_stats",
        help=f"Flag to decide if stats to be displayed - pass 0 to disable",
        required=False,
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--output_type",
        help=f"Flag to decide if output to be streamed or displayed on screen",
        required=False,
        type=str,
        default="screen",
    )
    args = parser.parse_args()
    main(
        model_type=args.model_type,
        stream_id=args.stream_id,
        max_fps=args.max_fps,
        enable_stats=args.enable_stats,
        output_type=args.output_type,
    )
