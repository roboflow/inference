import argparse
import os
import time
from datetime import datetime
from functools import partial
from threading import Thread
from typing import List

import cv2
import numpy as np
import supervision as sv

from inference import InferencePipeline
from inference.core.interfaces.camera.entities import StatusUpdate
from inference.core.interfaces.camera.utils import multiplex_videos
from inference.core.interfaces.camera.video_source import VideoSource
from inference.core.interfaces.stream.inference_pipeline import SinkMode
from inference.core.interfaces.stream.sinks import render_boxes
from inference.core.models.utils.batching import create_batches
from inference.core.utils.preprocess import letterbox_image

STREAM_SERVER_URL = os.getenv("STREAM_SERVER", "rtsp://localhost:8554")

STOP = False

BLACK_FRAME = np.zeros((348, 348, 3), dtype=np.uint8)


def main(n: int) -> None:
    stream_uris = []
    for i in range(n):
        stream_uris.append(f"{STREAM_SERVER_URL}/live{i}.stream")
    pipeline = InferencePipeline.init(
        video_reference=stream_uris,
        model_id="yolov8n-640",
        on_prediction=partial(render_boxes, display_statistics=True),
    )
    try:
        pipeline.start()
    except KeyboardInterrupt:
        print("Terminating")
        pipeline.terminate()
    finally:
        pipeline.join()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DEMUX demo")
    parser.add_argument(
        "--n", type=int, help="Number of streams", required=False, default=6
    )
    args = parser.parse_args()
    main(n=args.n)
