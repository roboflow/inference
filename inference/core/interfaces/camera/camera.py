import time
from threading import Thread

import cv2
from PIL import Image

from inference.core.logger import logger


class WebcamStream:
    """Class to handle webcam streaming using a separate thread.

    Attributes:
        stream_id (int): The ID of the webcam stream.
        frame_id (int): A counter for the current frame.
        vcap (VideoCapture): OpenCV video capture object.
        width (int): The width of the video frame.
        height (int): The height of the video frame.
        fps_input_stream (int): Frames per second of the input stream.
        grabbed (bool): A flag indicating if a frame was successfully grabbed.
        frame (array): The current frame as a NumPy array.
        pil_image (Image): The current frame as a PIL image.
        stopped (bool): A flag indicating if the stream is stopped.
        t (Thread): The thread used to update the stream.
    """

    def __init__(self, stream_id=0, enforce_fps=False):
        """Initialize the webcam stream.

        Args:
            stream_id (int, optional): The ID of the webcam stream. Defaults to 0.
        """
        self.stream_id = stream_id
        self.enforce_fps = enforce_fps
        self.frame_id = 0
        self.vcap = cv2.VideoCapture(self.stream_id)
        self.width = int(self.vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.max_fps = 30
        if self.vcap.isOpened() is False:
            logger.debug("[Exiting]: Error accessing webcam stream.")
            exit(0)
        self.fps_input_stream = int(self.vcap.get(cv2.CAP_PROP_FPS))
        logger.debug(
            "FPS of webcam hardware/input stream: {}".format(self.fps_input_stream)
        )
        self.grabbed, self.frame = self.vcap.read()
        self.pil_image = Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
        if self.grabbed is False:
            logger.debug("[Exiting] No more frames to read")
            exit(0)
        self.stopped = True
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True

    def start(self):
        """Start the thread for reading frames."""
        self.stopped = False
        self.t.start()

    def update(self):
        """Update the frame by reading from the webcam."""
        frame_id = 0
        skip_seconds = 0
        last_frame_position = time.perf_counter()
        t0 = time.perf_counter()
        while True:
            t1 = time.perf_counter()
            if self.stopped is True:
                break

            self.grabbed = self.vcap.grab()
            if self.grabbed:
                frame_id += 1
                if (
                    self.enforce_fps != "skip"
                    or t1 >= last_frame_position + skip_seconds
                ):
                    ret, frame = self.vcap.retrieve()
                    logger.debug("video capture FPS: %s", frame_id / (t1 - t0))
                    if frame is not None:
                        last_frame_position = t1
                        self.frame_id = frame_id
                        self.frame = frame
                    else:
                        logger.debug("[Exiting] Frame not available to retrieve")
                        self.stopped = True
                        break

            if self.grabbed is False:
                logger.debug("[Exiting] No more frames to read")
                self.stopped = True
                break
            if self.enforce_fps:
                t2 = time.perf_counter()
                next_frame = max(
                    1 / self.max_fps + 0.02, 1 / self.fps_input_stream - (t2 - t1)
                )
                if self.enforce_fps == "skip":
                    skip_seconds = next_frame
                else:
                    time.sleep(next_frame)
        self.vcap.release()

    def read_opencv(self):
        """Read the current frame using OpenCV.

        Returns:
            array, int: The current frame as a NumPy array, and the frame ID.
        """
        return self.frame, self.frame_id

    def stop(self):
        """Stop the webcam stream."""
        self.stopped = True
