import threading
from copy import copy
from typing import Callable, Union, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from inference.core.active_learning.middlewares import (
    NullActiveLearningMiddleware,
    ThreadingActiveLearningMiddleware,
)
from inference.core.cache import cache
from inference.core.env import (
    API_KEY,
    CLASS_AGNOSTIC_NMS,
    CONFIDENCE,
    IOU_THRESHOLD,
    MAX_CANDIDATES,
    MAX_DETECTIONS,
    MODEL_ID,
    STREAM_ID,
    MAX_FPS,
)
from inference.core.interfaces.camera.entities import (
    StatusUpdate,
    FrameTimestamp,
    FrameID,
)
from inference.core.interfaces.camera.utils import get_video_frames_generator
from inference.core.interfaces.camera.video_stream import VideoStream
from inference.core.interfaces.stream.utils import translate_stream_reference
from inference.core.logger import logger
from inference.core.models.base import BaseInference
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.registries.roboflow import get_model_type
from inference.models.utils import get_roboflow_model


def log_video_stream_status(status_update: StatusUpdate) -> None:
    logger.log(
        level=status_update.severity.value,
        msg=f"[{status_update.event_type}] {status_update.payload}",
    )


class Stream:
    """Roboflow defined stream interface for a general-purpose inference server.

    Attributes:
        model_manager (ModelManager): The manager that handles model inference tasks.
        model_registry (RoboflowModelRegistry): The registry to fetch model instances.
        api_key (str): The API key for accessing models.
        class_agnostic_nms (bool): Flag for class-agnostic non-maximum suppression.
        confidence (float): Confidence threshold for inference.
        iou_threshold (float): The intersection-over-union threshold for detection.
        json_response (bool): Flag to toggle JSON response format.
        max_candidates (float): The maximum number of candidates for detection.
        max_detections (float): The maximum number of detections.
        model (str|Callable): The model to be used.
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
        api_key: Optional[str] = API_KEY,
        class_agnostic_nms: bool = CLASS_AGNOSTIC_NMS,
        confidence: float = CONFIDENCE,
        max_fps: Optional[int] = MAX_FPS,
        iou_threshold: float = IOU_THRESHOLD,
        max_candidates: float = MAX_CANDIDATES,
        max_detections: float = MAX_DETECTIONS,
        model: Optional[Union[str, BaseInference]] = MODEL_ID,
        source: Optional[Union[int, str]] = STREAM_ID,
        use_main_thread: bool = False,
        output_channel_order: str = "RGB",
        on_prediction: Callable = None,
        on_start: Callable = None,
        on_stop: Callable = None,
    ):
        """Initialize the stream with the given parameters.
        Prints the server settings and initializes the inference with a test frame.
        """
        logger.info("Initializing server")
        self.frame_count = 0
        self.stream_id = translate_stream_reference(stream_reference=source)
        if model is None:
            raise ValueError("MODEL_ID is not defined.")
        if api_key is None:
            raise ValueError("API_KEY is not defined.")
        self.model_id = model
        self.api_key = api_key
        if isinstance(model, str):
            self.model = get_roboflow_model(model, self.api_key)
            self.active_learning_middleware = ThreadingActiveLearningMiddleware.init(
                api_key=self.api_key,
                model_id=self.model_id,
                cache=cache,
            )
            self.task_type = get_model_type(
                model_id=self.model_id, api_key=self.api_key
            )[0]
        else:
            self.model = model
            self.active_learning_middleware = NullActiveLearningMiddleware()
            self.task_type = "unknown"
        self.class_agnostic_nms = class_agnostic_nms
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.max_candidates = max_candidates
        self.max_detections = max_detections
        self.json_response = json_response
        self.use_main_thread = use_main_thread
        self.output_channel_order = output_channel_order
        self.video_stream = VideoStream.init(
            stream_reference=self.stream_id,
            status_update_handlers=[log_video_stream_status],
        )
        self.max_fps = max_fps
        logger.info(
            f"Streaming from device with resolution: {self.video_stream.width} x {self.video_stream.height}"
        )

        self.on_start_callbacks = []
        self.on_stop_callbacks = [
            lambda: self.active_learning_middleware.stop_registration_thread()
        ]
        self.on_prediction_callbacks = []

        if on_prediction:
            self.on_prediction_callbacks.append(on_prediction)

        if on_start:
            self.on_start_callbacks.append(on_start)

        if on_stop:
            self.on_stop_callbacks.append(on_stop)
        self._new_frame_captured = threading.Event()
        self.init_infer()
        self.preproc_result = None
        self.inference_request_obj = None
        self.inference_response = None
        self.stop = False

        self.frame = None
        self.frame_cv = None
        self.frame_id = None
        self._frame_data_lock = threading.Lock()
        self._frame_data: Optional[Tuple[FrameTimestamp, FrameID, np.ndarray]] = None
        self._preprocessing_result = Optional[
            Tuple[np.ndarray, PreprocessReturnMetadata]
        ] = None
        logger.info("Server initialized with settings:")
        logger.info(f"Stream ID: {self.stream_id}")
        logger.info(f"Model ID: {self.model_id}")
        logger.info(f"Enforce FPS: {enforce_fps}")
        logger.info(f"JSON Response: {self.json_response}")
        logger.info(f"Confidence: {self.confidence}")
        logger.info(f"Class Agnostic NMS: {self.class_agnostic_nms}")
        logger.info(f"IOU Threshold: {self.iou_threshold}")
        logger.info(f"Max Candidates: {self.max_candidates}")
        logger.info(f"Max Detections: {self.max_detections}")

        self.run_thread()

    def on_start(self, callback):
        self.on_start_callbacks.append(callback)

        unsubscribe = lambda: self.on_start_callbacks.remove(callback)
        return unsubscribe

    def on_stop(self, callback):
        self.on_stop_callbacks.append(callback)

        unsubscribe = lambda: self.on_stop_callbacks.remove(callback)
        return unsubscribe

    def on_prediction(self, callback):
        self.on_prediction_callbacks.append(callback)

        unsubscribe = lambda: self.on_prediction_callbacks.remove(callback)
        return unsubscribe

    def init_infer(self):
        """Initialize the inference with a test frame.

        Creates a test frame and runs it through the entire inference process to ensure everything is working.
        """
        frame = Image.new("RGB", (640, 640), color="black")
        self.model.infer(
            frame, confidence=self.confidence, iou_threshold=self.iou_threshold
        )
        self.active_learning_middleware.start_registration_thread()

    def preprocess_thread(self):
        """Preprocess incoming frames for inference.

        Reads frames from the webcam stream, converts them into the proper format, and preprocesses them for
        inference.
        """
        try:
            for frame_data in get_video_frames_generator(
                stream=self.video_stream,
                max_fps=self.max_fps,
            ):
                if self.stop:
                    break
                preprocessing_result = self.model.preprocess(frame_data[2])
                with self._frame_data_lock:
                    self._frame_data = frame_data
                    self._preprocessing_result = preprocessing_result
                self._new_frame_captured.set()
        except Exception as error:
            self.stop = True
            logger.exception(error)

    def inference_request_thread(self):
        """Manage the inference requests.

        Processes preprocessed frames for inference, post-processes the predictions, and sends the results
        to registered callbacks.
        """
        while True:
            if self.stop:
                while len(self.on_stop_callbacks) > 0:
                    # run each onStop callback only once from this thread
                    cb = self.on_stop_callbacks.pop()
                    cb()
                break
            self._new_frame_captured.wait()
            self._new_frame_captured.clear()
            while len(self.on_start_callbacks) > 0:
                # run each onStart callback only once from this thread
                cb = self.on_start_callbacks.pop()
                cb()
            with self._frame_data_lock:
                frame_id = self._frame_data[1]
                raw_inference_input = np.copy(self._frame_data[2])
                preprocessed_inference_input = np.copy(self._preprocessing_result[0])
                img_dims = copy(self._preprocessing_result[1])
            predictions = self.model.predict(preprocessed_inference_input)
            predictions = self.model.postprocess(
                predictions,
                img_dims,
                class_agnostic_nms=self.class_agnostic_nms,
                confidence=self.confidence,
                iou_threshold=self.iou_threshold,
                max_candidates=self.max_candidates,
                max_detections=self.max_detections,
            )
            predictions = self.model.make_response(predictions, img_dims)[0]
            predictions.frame_id = frame_id
            predictions = predictions.dict(by_alias=True, exclude_none=True)
            self.active_learning_middleware.register(
                inference_input=raw_inference_input,
                prediction=predictions,
                prediction_type=self.task_type,
            )
            self.inference_response = predictions
            self.frame_count += 1
            for cb in self.on_prediction_callbacks:
                if self.output_channel_order == "BGR":
                    cb(predictions, self._frame_data[2])
                else:
                    cb(
                        predictions,
                        cv2.cvtColor(self._frame_data[2], cv2.COLOR_BGR2RGB),
                    )

    def run_thread(self):
        """Run the preprocessing and inference threads.

        Starts the preprocessing and inference threads, and handles graceful shutdown on KeyboardInterrupt.
        """
        preprocess_thread = threading.Thread(target=self.preprocess_thread)
        preprocess_thread.start()

        if self.use_main_thread:
            self.inference_request_thread()
        else:
            # start a thread that looks for the predictions
            # and call the callbacks
            inference_request_thread = threading.Thread(
                target=self.inference_request_thread
            )
            inference_request_thread.start()
