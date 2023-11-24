import argparse
import time
from datetime import datetime
from threading import Thread
from typing import Optional

import cv2
import supervision as sv

from inference.core.interfaces.camera.camera import WebcamStream
from inference.core.interfaces.camera.entities import StatusUpdate
from inference.core.interfaces.camera.utils import get_video_frames_generator
from inference.core.interfaces.camera.video_source import VideoSource
from inference.core.utils.preprocess import letterbox_image

STOP = False


def main(stream_uri: str, max_fps: Optional[int] = None, processing_time: float = 0.0, enable_stats: int = 1) -> None:
    global STOP
    camera = WebcamStream(stream_id=stream_uri)
    camera.start()
    camera.max_fps = max_fps
    control_thread = Thread(target=command_thread, args=(camera,))
    control_thread.start()
    fps_monitor = sv.FPSMonitor()
    previous_frame_id = 0
    while True:
        if STOP:
            break
        frame, frame_id = camera.read_opencv()
        fps_monitor.tick()
        fps_value = fps_monitor()
        resized_frame = letterbox_image(frame, desired_size=(1280, 720))
        if enable_stats:
            dropped_frames = max(frame_id - previous_frame_id - 1, 0)
            resized_frame = cv2.putText(
                resized_frame, f"DROPPED FRAMES: {dropped_frames}", (10, 670),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
            )
            resized_frame = cv2.putText(
                resized_frame, f"FPS: {round(fps_value, 2)}", (10, 710),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
            )
        cv2.imshow("Stream", resized_frame)
        _ = cv2.waitKey(1)
        previous_frame_id = frame_id
        time.sleep(processing_time)
    STOP = True
    print("DONE")
    cv2.destroyAllWindows()
    control_thread.join()


def command_thread(camera: WebcamStream) -> None:
    global STOP
    while not STOP:
        key = input()
        if key == "s":
            camera.stop()
            STOP = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Camera demo parser")
    parser.add_argument(
        "--stream_uri", help="URI of video stream", type=str, required=True
    )
    parser.add_argument(
        "--max_fps", help="Limit FPS", type=int, required=False, default=None
    )
    parser.add_argument(
        "--processing_time",
        help="Time of processing of each frame to be emulated (in fraction of seconds)",
        type=float,
        required=False,
        default=0.0,
    )
    parser.add_argument(
        "--enable_stats",
        help=f"Flag to decide if stats to be displayed - pass 0 to disable",
        required=False,
        type=int,
        default=1,
    )
    args = parser.parse_args()
    main(
        stream_uri=args.stream_uri,
        max_fps=args.max_fps,
        processing_time=args.processing_time,
        enable_stats=args.enable_stats,
    )
