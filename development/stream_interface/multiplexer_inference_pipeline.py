import argparse
import json
import os
from dataclasses import asdict
from datetime import date, datetime
from enum import Enum
from functools import partial
from threading import Thread
from typing import Any

import cv2
import numpy as np

from inference import InferencePipeline
from inference.core.interfaces.stream.inference_pipeline import SinkMode
from inference.core.interfaces.stream.sinks import render_boxes
from inference.core.interfaces.stream.watchdog import BasePipelineWatchDog

STREAM_SERVER_URL = os.getenv("STREAM_SERVER", "rtsp://localhost:8554")

STOP = False

BLACK_FRAME = np.zeros((348, 348, 3), dtype=np.uint8)


def main(n: int) -> None:
    stream_uris = []
    for i in range(n):
        stream_uris.append(f"{STREAM_SERVER_URL}/live{i}.stream")
    watchdog = BasePipelineWatchDog()
    pipeline = InferencePipeline.init(
        video_reference=stream_uris,
        model_id="yolov8n-640",
        on_prediction=partial(render_boxes, display_statistics=True),
        watchdog=watchdog,
    )
    control_thread = Thread(target=command_thread, args=(pipeline, watchdog, ))
    control_thread.start()
    pipeline.start()
    pipeline.join()
    cv2.destroyAllWindows()


def command_thread(pipeline: InferencePipeline, watchdog: BasePipelineWatchDog) -> None:
    global STOP
    while not STOP:
        try:
            payload = input()
            if payload == "s":
                pipeline.terminate()
                STOP = True
            elif payload.startswith("p"):
                idx = payload.split(",")[1]
                pipeline.mute_stream(source_id=idx)
            elif payload.startswith("r"):
                idx = payload.split(",")[1]
                pipeline.resume_stream(source_id=idx)
            elif payload == "i":
                print(json.dumps(asdict(watchdog.get_report()), default=serialise_to_json,))
        except Exception as e:
            print(e)
    print("END CMD THREAD")


def serialise_to_json(obj: Any) -> Any:
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if issubclass(type(obj), Enum):
        return obj.value
    raise TypeError(f"Type {type(obj)} not serializable")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DEMUX demo")
    parser.add_argument(
        "--n", type=int, help="Number of streams", required=False, default=6
    )
    args = parser.parse_args()
    main(n=args.n)
