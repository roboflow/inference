import json
import socket
import sys
import threading
import time

from PIL import Image

from inference.core import data_models as M
from inference.core.env import (
    API_KEY,
    CLASS_AGNOSTIC_NMS,
    CONFIDENCE,
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
from inference.core.managers.base import ModelManager
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.core.version import __version__


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

    Methods:
        init_infer: Initialize the inference with a test frame.
        preprocess_thread: Preprocess incoming frames for inference.
        inference_request_thread: Manage the inference requests.
        run_thread: Run the preprocessing and inference threads.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        model_registry: RoboflowModelRegistry,
        api_key: str = API_KEY,
        class_agnostic_nms: bool = CLASS_AGNOSTIC_NMS,
        confidence: float = CONFIDENCE,
        ip_broadcast_addr: str = IP_BROADCAST_ADDR,
        ip_broadcast_port: int = IP_BROADCAST_PORT,
        iou_threshold: float = IOU_THRESHOLD,
        json_response: bool = JSON_RESPONSE,
        max_candidates: float = MAX_CANDIDATES,
        max_detections: float = MAX_DETECTIONS,
        model_id: str = MODEL_ID,
        stream_id: str = STREAM_ID,
    ):
        """Initialize the UDP stream with the given parameters.
        Prints the server settings and initializes the inference with a test frame.
        """
        print("Initializing server")

        self.model_manager = model_manager
        self.model_registry = model_registry
        self.frame_count = 0

        self.stream_id = stream_id
        if self.stream_id is None:
            raise ValueError("STREAM_ID is not defined")
        self.model_id = model_id
        if not self.model_id:
            raise ValueError("MODEL_ID is not defined")
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("API_KEY is not defined")

        model = self.model_registry.get_model(self.model_id, self.api_key)(
            model_id=self.model_id,
            api_key=self.api_key,
        )
        self.model_manager.add_model(self.model_id, model)

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
        self.UDPServerSocket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1)

        self.webcam_stream = WebcamStream(stream_id=self.stream_id)
        print(
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
        print("Server initialized with settings:")
        print(f"Stream ID: {self.stream_id}")
        print(f"Model ID: {self.model_id}")
        print(f"Confidence: {self.confidence}")
        print(f"Class Agnostic NMS: {self.class_agnostic_nms}")
        print(f"IOU Threshold: {self.iou_threshold}")
        print(f"Max Candidates: {self.max_candidates}")
        print(f"Max Detections: {self.max_detections}")

    def init_infer(self):
        """Initialize the inference with a test frame.

        Creates a test frame and runs it through the entire inference process to ensure everything is working.
        """
        frame = Image.new("RGB", (640, 640), color="black")
        request_image = M.InferenceRequestImage(type="pil", value=frame)
        inference_request_obj = self.inference_request_type(
            model_id=self.model_id,
            image=request_image,
            api_key=self.api_key,
        )
        preproc_result = self.model_manager.preprocess(
            inference_request_obj.model_id, inference_request_obj
        )
        img_in, img_dims = preproc_result
        predictions = self.model_manager.predict(
            inference_request_obj.model_id,
            img_in,
        )
        predictions = self.model_manager.postprocess(
            inference_request_obj.model_id,
            predictions,
            img_dims,
            class_agnostic_nms=self.class_agnostic_nms,
            confidence=self.confidence,
            iou_threshold=self.iou_threshold,
            max_candidates=self.max_candidates,
            max_detections=self.max_detections,
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
                        request_image = M.InferenceRequestImage(
                            type="pil", value=self.frame
                        )
                        self.inference_request_obj = self.inference_request_type(
                            model_id=self.model_id,
                            image=request_image,
                            api_key=self.api_key,
                        )
                        self.preproc_result = self.model_manager.preprocess(
                            self.inference_request_obj.model_id,
                            self.inference_request_obj,
                        )
                        self.img_in, self.img_dims = self.preproc_result
                        self.queue_control = True

        except Exception as e:
            print(e)

    def inference_request_thread(self):
        """Manage the inference requests.

        Processes preprocessed frames for inference, post-processes the predictions, and sends the results
        as a UDP broadcast.
        """
        start_time = time.time()
        while True:
            if self.stop:
                break
            if self.queue_control:
                self.queue_control = False
                frame_id = self.frame_id
                predictions = self.model_manager.predict(
                    self.inference_request_obj.model_id,
                    self.img_in,
                )
                predictions = self.model_manager.postprocess(
                    self.inference_request_obj.model_id,
                    predictions,
                    self.img_dims,
                    class_agnostic_nms=self.class_agnostic_nms,
                    confidence=self.confidence,
                    iou_threshold=self.iou_threshold,
                    max_candidates=self.max_candidates,
                    max_detections=self.max_detections,
                )
                if self.json_response:
                    predictions = self.model_manager.make_response(
                        self.inference_request_obj.model_id,
                        predictions,
                        self.img_dims,
                    )[0]
                    predictions.frame_id = frame_id
                    predictions = predictions.json(exclude_none=True)
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
                print("Stopping server...")
                self.stop = True
                time.sleep(3)
                sys.exit(0)
