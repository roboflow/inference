import argparse
import os
import time
from datetime import datetime
from threading import Thread
from typing import List

import cv2
import numpy as np
import supervision as sv

from inference.core.interfaces.camera.entities import StatusUpdate
from inference.core.interfaces.camera.utils import multiplex_videos
from inference.core.interfaces.camera.video_source import VideoSource
from inference.core.models.utils.batching import create_batches
from inference.core.utils.preprocess import letterbox_image

STREAM_SERVER_URL = os.getenv("STREAM_SERVER", "rtsp://localhost:8554")

STOP = False

BLACK_FRAME = np.zeros((348, 348, 3), dtype=np.uint8)


def main(n: int) -> None:
    stream_uris = []
    for i in range(n):
        stream_uris.append(f"{STREAM_SERVER_URL}/live{i}.stream")
    cameras = [VideoSource.init(uri, source_id=i) for i, uri in enumerate(stream_uris)]
    control_thread = Thread(target=command_thread, args=(cameras,))
    control_thread.start()
    for camera in cameras:
        camera.start()
    multiplexer = multiplex_videos(
        videos=cameras,
        batch_collection_timeout=None,
        should_stop=lambda: STOP,
    )
    fps_monitor = sv.FPSMonitor(sample_size=8192)
    monitors = {}
    for i in range(n):
        monitors[i] = sv.FPSMonitor(sample_size=8192)
    for frames in multiplexer:
        registered_frames = {}
        for f in frames:
            monitors[f.source_id].tick()
            fps = monitors[f.source_id].fps if hasattr(monitors[f.source_id], "fps") else monitors[f.source_id]()
            i = cv2.putText(
                f.image,
                f"LATENCY: {round((datetime.now() - f.frame_timestamp).total_seconds() * 1000, 2)} ms",
                (10, f.image.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (0, 255, 0),
                4,
            )
            i = cv2.putText(
                i,
                f"THROUGHPUT: {round(fps, 2)}",
                (10, f.image.shape[0] - 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (0, 255, 0),
                4,
            )
            registered_frames[f.source_id] = i
        for _ in range(len(frames)):
            fps_monitor.tick()
        if hasattr(fps_monitor, "fps"):
            fps_value = fps_monitor.fps
        else:
            fps_value = fps_monitor()
        images = [letterbox_image(registered_frames.get(i, BLACK_FRAME), (348, 348)) for i in range(n)]
        rows = list(create_batches(sequence=images, batch_size=4))
        while len(rows[-1]) < 4:
            rows[-1].append(BLACK_FRAME)
        rows_merged = [np.concatenate(r, axis=1) for r in rows]
        merged = np.concatenate(rows_merged, axis=0)
        fps = round(fps_value, 2)
        merged = cv2.putText(
            merged,
            f"THROUGHPUT: {fps}",
            (10, merged.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imshow("playback", merged)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    global STOP
    STOP = True
    control_thread.join()
    for c in cameras:
        c.terminate(wait_on_frames_consumption=False, purge_frames_buffer=True)
    print("JOINED")


def command_thread(cameras: List[VideoSource]) -> None:
    global STOP
    while not STOP:
        try:
            payload = input().split(",")
            idx = None
            if len(payload) > 1:
                idx = int(payload[1])
            if payload[0] == "i":
                print(cameras[idx].describe_source())
            elif payload[0] == "s":
                STOP = True
            elif payload[0] == "p":
                cameras[idx].pause()
            elif payload[0] == "m":
                cameras[idx].mute()
            elif payload[0] == "r":
                cameras[idx].resume()
            elif payload[0] == "re":
                cameras[idx].restart(wait_on_frames_consumption=False)
        except Exception as e:
            print(e)
    print("END CMD THREAD")


def dump_status_update(status_update: StatusUpdate) -> None:
    print(status_update)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DEMUX demo")
    parser.add_argument(
        "--n", type=int, help="Number of streams", required=False, default=6
    )
    args = parser.parse_args()
    main(n=args.n)
