import argparse
import os
from threading import Thread
from typing import List

import cv2
import numpy as np

from inference.core.interfaces.camera.entities import StatusUpdate
from development.stream_interface.multiplexer import StreamMultiplexer
from inference.core.interfaces.camera.video_source import VideoSource
from inference.core.utils.preprocess import letterbox_image

STREAM_SERVER_URL = os.getenv("STREAM_SERVER", "rtsp://localhost:8554")

STOP = False


def main(n: int) -> None:
    stream_uris = []
    for i in range(8):
        stream_uris.append(f"{STREAM_SERVER_URL}/live{i % n}.stream")
        stream_uris.append(f"{STREAM_SERVER_URL}/predictions{i % n}.stream")
    cameras = [VideoSource.init(uri, status_update_handlers=[]) for uri in stream_uris]
    for camera in cameras:
        camera.start()
    multiplexer = StreamMultiplexer(sources=cameras)
    control_thread = Thread(target=command_thread, args=(cameras,))
    control_thread.start()
    previous_frames = [
        np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(len(cameras))
    ]
    while not STOP:
        new_frames = multiplexer.get_frames()
        for i in range(len(new_frames)):
            if new_frames[i] is not None:
                previous_frames[i] = letterbox_image(
                    image=new_frames[i].image, desired_size=(640, 480)
                )
        first_row = np.concatenate(
            [
                previous_frames[0],
                np.zeros((480, 10, 3), dtype=np.uint8),
                previous_frames[1],
                np.zeros((480, 10, 3), dtype=np.uint8),
                previous_frames[2],
                np.zeros((480, 10, 3), dtype=np.uint8),
                previous_frames[3],
            ],
            axis=1,
        )
        second_row = np.concatenate(
            [
                previous_frames[4],
                np.zeros((480, 10, 3), dtype=np.uint8),
                previous_frames[5],
                np.zeros((480, 10, 3), dtype=np.uint8),
                previous_frames[6],
                np.zeros((480, 10, 3), dtype=np.uint8),
                previous_frames[7],
            ],
            axis=1,
        )
        third_row = np.concatenate(
            [
                previous_frames[8],
                np.zeros((480, 10, 3), dtype=np.uint8),
                previous_frames[9],
                np.zeros((480, 10, 3), dtype=np.uint8),
                previous_frames[10],
                np.zeros((480, 10, 3), dtype=np.uint8),
                previous_frames[11],
            ],
            axis=1,
        )
        forth_row = np.concatenate(
            [
                previous_frames[12],
                np.zeros((480, 10, 3), dtype=np.uint8),
                previous_frames[13],
                np.zeros((480, 10, 3), dtype=np.uint8),
                previous_frames[14],
                np.zeros((480, 10, 3), dtype=np.uint8),
                previous_frames[15],
            ],
            axis=1,
        )
        image = np.concatenate(
            [
                first_row,
                np.zeros((10, 2590, 3), dtype=np.uint8),
                second_row,
                np.zeros((10, 2590, 3), dtype=np.uint8),
                third_row,
                np.zeros((10, 2590, 3), dtype=np.uint8),
                forth_row,
            ],
            axis=0,
        )
        cv2.imshow("Stream", image)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    control_thread.join()


def command_thread(cameras: List[VideoSource]) -> None:
    global STOP
    while not STOP:
        idx, key = input().split(",")
        idx = int(idx)
        if key == "q":
            continue
        elif key == "i":
            print(cameras[idx].describe_source())
        elif key == "t":
            for c in cameras:
                c.terminate()
            STOP = True
        elif key == "p":
            cameras[idx].pause()
        elif key == "m":
            cameras[idx].mute()
        elif key == "r":
            cameras[idx].resume()
        elif key == "re":
            cameras[idx].restart()


def dump_status_update(status_update: StatusUpdate) -> None:
    print(status_update)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DEMUX demo")
    parser.add_argument(
        "--n", type=int, help="Number of streams", required=False, default=6
    )
    args = parser.parse_args()
    main(n=args.n)
