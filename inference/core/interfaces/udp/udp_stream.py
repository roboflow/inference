import json
import socket
import sys
import threading
import time
from typing import Union

import supervision as sv
from PIL import Image

from inference.core import data_models as M
from inference.core.env import (
    API_KEY,
    CLASS_AGNOSTIC_NMS,
    CONFIDENCE,
    ENABLE_BYTE_TRACK,
    ENFORCE_FPS,
    IOU_THRESHOLD,
    IP_BROADCAST_ADDR,
    IP_BROADCAST_PORT,
    JSON_RESPONSE,
    MAX_CANDIDATES,
    MAX_DETECTIONS,
    MODEL_ID,
    STREAM_ID,
)
from inference.core.interfaces.base import BaseInterface
from inference.core.interfaces.camera.camera import WebcamStream
from inference.core.logger import logger
from inference.core.version import __version__
from inference.models.utils import get_roboflow_model


class UdpStream(BaseInterface):
    """Roboflow defined UDP interface for a general-purpose inference server.

    Attributes:
        model_manager (ModelManager): The manager that handles model inference tasks.
        model_registry (RoboflowModelRegistry): The registry to fetch model instances.
        api_key (str): The API key for accessing models.
        class_agnostic_nms (bool): Flag for class-agnostic non-maximum suppression.
        confidence (float): Confidence threshold for inference.
        ip_broadcast_addr (str): The IP address to broadcast to.
        ip_broadcast_port (int): The port to broadcast on.
        iou_threshold (float): The intersection-over-union threshold for detection.
        json_response (bool): Flag to toggle JSON response format.
        max_candidates (float): The maximum number of candidates for detection.
        max_detections (float): The maximum number of detections.
        model_id (str): The ID of the model to be used.
        stream_id (str): The ID of the stream to be used.
        use_bytetrack (bool): Flag to use bytetrack,

    Methods:
        init_infer: Initialize the inference with a test frame.
        preprocess_thread: Preprocess incoming frames for inference.
        inference_request_thread: Manage the inference requests.
        run_thread: Run the preprocessing and inference threads.
    """

    def __init__(
        self,
        api_key: str = API_KEY,
        class_agnostic_nms: bool = CLASS_AGNOSTIC_NMS,
        confidence: float = CONFIDENCE,
        enforce_fps: bool = ENFORCE_FPS,
        ip_broadcast_addr: str = IP_BROADCAST_ADDR,
        ip_broadcast_port: int = IP_BROADCAST_PORT,
        iou_threshold: float = IOU_THRESHOLD,
        json_response: bool = JSON_RESPONSE,
        max_candidates: float = MAX_CANDIDATES,
        max_detections: float = MAX_DETECTIONS,
        model_id: str = MODEL_ID,
        stream_id: Union[int, str] = STREAM_ID,
        use_bytetrack: bool = ENABLE_BYTE_TRACK,
    ):
        """Initialize the UDP stream with the given parameters.
        Prints the server settings and initializes the inference with a test frame.
        """
        logger.info("Initializing server")

        self.frame_count = 0
        self.byte_tracker = sv.ByteTrack() if use_bytetrack else None
        self.use_bytetrack = use_bytetrack

        self.stream_id = stream_id
        if self.stream_id is None:
            raise ValueError("STREAM_ID is not defined")
        self.model_id = model_id
        if not self.model_id:
            raise ValueError("MODEL_ID is not defined")
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("API_KEY is not defined")

        self.model = get_roboflow_model(self.model_id, self.api_key)

        self.class_agnostic_nms = class_agnostic_nms
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.max_candidates = max_candidates
        self.max_detections = max_detections
        self.ip_broadcast_addr = ip_broadcast_addr
        self.ip_broadcast_port = ip_broadcast_port
        self.json_response = json_response

        self.inference_request_type = M.ObjectDetectionInferenceRequest

        self.UDPServerSocket = socket.socket(
            family=socket.AF_INET, type=socket.SOCK_DGRAM
        )
        self.UDPServerSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.UDPServerSocket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.UDPServerSocket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1)
        self.UDPServerSocket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)

        self.webcam_stream = WebcamStream(
            stream_id=self.stream_id, enforce_fps=enforce_fps
        )
        logger.info(
            f"Streaming from device with resolution: {self.webcam_stream.width} x {self.webcam_stream.height}"
        )

        self.init_infer()
        self.preproc_result = None
        self.inference_request_obj = None
        self.queue_control = False
        self.inference_response = None
        self.stop = False

        self.frame = None
        self.frame_cv = None
        self.frame_id = None
        logger.info("Server initialized with settings:")
        logger.info(f"Stream ID: {self.stream_id}")
        logger.info(f"Model ID: {self.model_id}")
        logger.info(f"Confidence: {self.confidence}")
        logger.info(f"Class Agnostic NMS: {self.class_agnostic_nms}")
        logger.info(f"IOU Threshold: {self.iou_threshold}")
        logger.info(f"Max Candidates: {self.max_candidates}")
        logger.info(f"Max Detections: {self.max_detections}")

    def init_infer(self):
        """Initialize the inference with a test frame.

        Creates a test frame and runs it through the entire inference process to ensure everything is working.
        """
        frame = Image.new("RGB", (640, 640), color="black")
        self.model.infer(
            frame, confidence=self.confidence, iou_threshold=self.iou_threshold
        )

    def preprocess_thread(self):
        """Preprocess incoming frames for inference.

        Reads frames from the webcam stream, converts them into the proper format, and preprocesses them for
        inference.
        """
        webcam_stream = self.webcam_stream
        webcam_stream.start()
        # processing frames in input stream
        try:
            while True:
                if webcam_stream.stopped is True or self.stop:
                    break
                else:
                    self.frame, self.frame_cv, frame_id = webcam_stream.read()
                    if frame_id != self.frame_id:
                        self.frame_id = frame_id
                        self.preproc_result = self.model.preprocess(self.frame)
                        self.img_in, self.img_dims = self.preproc_result
                        self.queue_control = True

        except Exception as e:
            logger.error(e)

    def inference_request_thread(self):
        """Manage the inference requests.

        Processes preprocessed frames for inference, post-processes the predictions, and sends the results
        as a UDP broadcast.
        """
        last_print = time.perf_counter()
        print_ind = 0
        print_chars = ["|", "/", "-", "\\"]
        while True:
            if self.stop:
                break
            if self.queue_control:
                self.queue_control = False
                frame_id = self.frame_id
                predictions = self.model.predict(
                    self.img_in,
                )
                predictions = self.model.postprocess(
                    predictions,
                    self.img_dims,
                    class_agnostic_nms=self.class_agnostic_nms,
                    confidence=self.confidence,
                    iou_threshold=self.iou_threshold,
                    max_candidates=self.max_candidates,
                    max_detections=self.max_detections,
                )
                if self.json_response:
                    predictions = self.model.make_response(
                        predictions,
                        self.img_dims,
                    )[0]
                    if self.use_bytetrack:
                        detections = sv.Detections.from_roboflow(
                            predictions.dict(by_alias=True), self.model.class_names
                        )
                        detections = self.byte_tracker.update_with_detections(
                            detections
                        )
                        for pred, detect in zip(predictions.predictions, detections):
                            pred.tracker_id = int(detect[4])
                    predictions.frame_id = frame_id
                    predictions = predictions.json(exclude_none=True, by_alias=True)
                else:
                    predictions = json.dumps(predictions)

                self.inference_response = predictions
                self.frame_count += 1

                bytesToSend = predictions.encode("utf-8")
                self.UDPServerSocket.sendto(
                    bytesToSend,
                    (
                        self.ip_broadcast_addr,
                        self.ip_broadcast_port,
                    ),
                )
                if time.perf_counter() - last_print > 1:
                    print(f"Streaming {print_chars[print_ind]}", end="\r")
                    print_ind = (print_ind + 1) % 4
                    last_print = time.perf_counter()

    def run_thread(self):
        """Run the preprocessing and inference threads.

        Starts the preprocessing and inference threads, and handles graceful shutdown on KeyboardInterrupt.
        """
        preprocess_thread = threading.Thread(target=self.preprocess_thread)
        inference_request_thread = threading.Thread(
            target=self.inference_request_thread
        )

        preprocess_thread.start()
        inference_request_thread.start()

        while True:
            try:
                time.sleep(10)
            except KeyboardInterrupt:
                logger.info("Stopping server...")
                self.stop = True
                time.sleep(3)
                sys.exit(0)
