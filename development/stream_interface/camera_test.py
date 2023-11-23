import argparse
from typing import Union

import supervision as sv
import cv2

from inference.core.utils.preprocess import letterbox_image


def main(video: Union[str, int], display: int, each_nth: int) -> None:
    stream = cv2.VideoCapture(video)
    fps_monitor = sv.FPSMonitor()
    frames_counter = 0
    while stream.isOpened():
        print(frames_counter, frames_counter % each_nth)
        status = stream.grab()
        fps_monitor.tick()
        fps_value = fps_monitor()
        print("GRABBING FPS: ", fps_value)
        if not status:
            print("EOS")
            break
        if frames_counter % each_nth != 0 and display > 0:
            frames_counter += 1
            continue
        status, image = stream.retrieve()
        if not status:
            print("EOS")
            break
        image = letterbox_image(image=image, desired_size=(1280, 720))
        cv2.imshow("stream", image)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        frames_counter += 1
    print("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Camera test script")
    parser.add_argument("--video", required=False, default=0)
    parser.add_argument("--display", type=int, required=False, default=1)
    parser.add_argument("--each_nth", type=int, required=False, default=1)
    args = parser.parse_args()
    main(video=args.video, display=args.display, each_nth=args.each_nth)
