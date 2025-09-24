import os
import signal
from dataclasses import asdict
from multiprocessing import Process, Queue
from types import FrameType
from typing import Callable, Optional, Tuple

from inference.core import logger
from inference.core.exceptions import (
    MissingApiKeyError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
)
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.camera.exceptions import StreamOperationNotAllowedError
from inference.core.interfaces.camera.video_source import (
    BufferConsumptionStrategy,
    BufferFillingStrategy,
)
from inference.core.interfaces.stream.entities import ObjectDetectionPrediction
from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
from inference.core.interfaces.stream.sinks import UDPSink
from inference.core.interfaces.stream.watchdog import (
    BasePipelineWatchDog,
    PipelineWatchDog,
)
from inference.enterprise.stream_management.manager.entities import (
    STATUS_KEY,
    TYPE_KEY,
    CommandType,
    ErrorType,
    OperationStatus,
)
from inference.enterprise.stream_management.manager.serialisation import describe_error


def ignore_signal(signal_number: int, frame: FrameType) -> None:
    pid = os.getpid()
    logger.info(
        f"Ignoring signal {signal_number} in InferencePipelineManager in process:{pid}"
    )


class InferencePipelineManager(Process):
    @classmethod
    def init(
        cls, command_queue: Queue, responses_queue: Queue
    ) -> "InferencePipelineManager":
        return cls(command_queue=command_queue, responses_queue=responses_queue)

    def __init__(self, command_queue: Queue, responses_queue: Queue):
        super().__init__()
        self._command_queue = command_queue
        self._responses_queue = responses_queue
        self._inference_pipeline: Optional[InferencePipeline] = None
        self._watchdog: Optional[PipelineWatchDog] = None
        self._stop = False

    def run(self) -> None:
        signal.signal(signal.SIGINT, ignore_signal)
        signal.signal(signal.SIGTERM, self._handle_termination_signal)
        while not self._stop:
            command: Optional[Tuple[str, dict]] = self._command_queue.get()
            if command is None:
                break
            request_id, payload = command
            self._handle_command(request_id=request_id, payload=payload)

    def _handle_command(self, request_id: str, payload: dict) -> None:
        try:
            logger.info(f"Processing request={request_id}...")
            command_type = payload[TYPE_KEY]
            if command_type is CommandType.INIT:
                return self._initialise_pipeline(request_id=request_id, payload=payload)
            if command_type is CommandType.TERMINATE:
                return self._terminate_pipeline(request_id=request_id)
            if command_type is CommandType.MUTE:
                return self._mute_pipeline(request_id=request_id)
            if command_type is CommandType.RESUME:
                return self._resume_pipeline(request_id=request_id)
            if command_type is CommandType.STATUS:
                return self._get_pipeline_status(request_id=request_id)
            raise NotImplementedError(
                f"Command type `{command_type}` cannot be handled"
            )
        except (KeyError, NotImplementedError) as error:
            self._handle_error(
                request_id=request_id, error=error, error_type=ErrorType.INVALID_PAYLOAD
            )
        except Exception as error:
            self._handle_error(
                request_id=request_id, error=error, error_type=ErrorType.INTERNAL_ERROR
            )

    def _initialise_pipeline(self, request_id: str, payload: dict) -> None:
        try:
            watchdog = BasePipelineWatchDog()
            sink = assembly_pipeline_sink(sink_config=payload["sink_configuration"])
            source_buffer_filling_strategy, source_buffer_consumption_strategy = (
                None,
                None,
            )
            if "source_buffer_filling_strategy" in payload:
                source_buffer_filling_strategy = BufferFillingStrategy(
                    payload["source_buffer_filling_strategy"].upper()
                )
            if "source_buffer_consumption_strategy" in payload:
                source_buffer_consumption_strategy = BufferConsumptionStrategy(
                    payload["source_buffer_consumption_strategy"].upper()
                )
            model_configuration = payload["model_configuration"]
            if model_configuration["type"] != "object-detection":
                raise NotImplementedError("Only object-detection models are supported")
            self._inference_pipeline = InferencePipeline.init(
                model_id=payload["model_id"],
                video_reference=payload["video_reference"],
                on_prediction=sink,
                api_key=payload.get("api_key"),
                max_fps=payload.get("max_fps"),
                watchdog=watchdog,
                source_buffer_filling_strategy=source_buffer_filling_strategy,
                source_buffer_consumption_strategy=source_buffer_consumption_strategy,
                class_agnostic_nms=model_configuration.get("class_agnostic_nms"),
                confidence=model_configuration.get("confidence"),
                iou_threshold=model_configuration.get("iou_threshold"),
                max_candidates=model_configuration.get("max_candidates"),
                max_detections=model_configuration.get("max_detections"),
                active_learning_enabled=payload.get("active_learning_enabled"),
                video_source_properties=payload.get("video_source_properties"),
                active_learning_target_dataset=payload.get(
                    "active_learning_target_dataset"
                ),
                batch_collection_timeout=payload.get("batch_collection_timeout"),
            )
            self._watchdog = watchdog
            self._inference_pipeline.start(use_main_thread=False)
            self._responses_queue.put(
                (request_id, {STATUS_KEY: OperationStatus.SUCCESS})
            )
            logger.info(f"Pipeline initialised. request_id={request_id}...")
        except (MissingApiKeyError, KeyError, NotImplementedError) as error:
            self._handle_error(
                request_id=request_id, error=error, error_type=ErrorType.INVALID_PAYLOAD
            )
        except RoboflowAPINotAuthorizedError as error:
            self._handle_error(
                request_id=request_id,
                error=error,
                error_type=ErrorType.AUTHORISATION_ERROR,
            )
        except RoboflowAPINotNotFoundError as error:
            self._handle_error(
                request_id=request_id, error=error, error_type=ErrorType.NOT_FOUND
            )

    def _terminate_pipeline(self, request_id: str) -> None:
        if self._inference_pipeline is None:
            self._responses_queue.put(
                (request_id, {STATUS_KEY: OperationStatus.SUCCESS})
            )
            self._stop = True
            return None
        try:
            self._execute_termination()
            logger.info(f"Pipeline terminated. request_id={request_id}...")
            self._responses_queue.put(
                (request_id, {STATUS_KEY: OperationStatus.SUCCESS})
            )
        except StreamOperationNotAllowedError as error:
            self._handle_error(
                request_id=request_id, error=error, error_type=ErrorType.OPERATION_ERROR
            )

    def _handle_termination_signal(self, signal_number: int, frame: FrameType) -> None:
        try:
            pid = os.getpid()
            logger.info(f"Terminating pipeline in process:{pid}...")
            if self._inference_pipeline is not None:
                self._execute_termination()
            self._command_queue.put(None)
            logger.info(f"Termination successful in process:{pid}...")
        except Exception as error:
            logger.warning(f"Could not terminate pipeline gracefully. Error: {error}")

    def _execute_termination(self) -> None:
        self._inference_pipeline.terminate()
        self._inference_pipeline.join()
        self._stop = True

    def _mute_pipeline(self, request_id: str) -> None:
        if self._inference_pipeline is None:
            return self._handle_error(
                request_id=request_id, error_type=ErrorType.OPERATION_ERROR
            )
        try:
            self._inference_pipeline.mute_stream()
            logger.info(f"Pipeline muted. request_id={request_id}...")
            self._responses_queue.put(
                (request_id, {STATUS_KEY: OperationStatus.SUCCESS})
            )
        except StreamOperationNotAllowedError as error:
            self._handle_error(
                request_id=request_id, error=error, error_type=ErrorType.OPERATION_ERROR
            )

    def _resume_pipeline(self, request_id: str) -> None:
        if self._inference_pipeline is None:
            return self._handle_error(
                request_id=request_id, error_type=ErrorType.OPERATION_ERROR
            )
        try:
            self._inference_pipeline.resume_stream()
            logger.info(f"Pipeline resumed. request_id={request_id}...")
            self._responses_queue.put(
                (request_id, {STATUS_KEY: OperationStatus.SUCCESS})
            )
        except StreamOperationNotAllowedError as error:
            self._handle_error(
                request_id=request_id, error=error, error_type=ErrorType.OPERATION_ERROR
            )

    def _get_pipeline_status(self, request_id: str) -> None:
        if self._watchdog is None:
            return self._handle_error(
                request_id=request_id, error_type=ErrorType.OPERATION_ERROR
            )
        try:
            report = self._watchdog.get_report()
            if report is None:
                return self._handle_error(
                    request_id=request_id, error_type=ErrorType.OPERATION_ERROR
                )
            response_payload = {
                STATUS_KEY: OperationStatus.SUCCESS,
                "report": asdict(report),
            }
            self._responses_queue.put((request_id, response_payload))
            logger.info(f"Pipeline status returned. request_id={request_id}...")
        except StreamOperationNotAllowedError as error:
            self._handle_error(
                request_id=request_id, error=error, error_type=ErrorType.OPERATION_ERROR
            )

    def _handle_error(
        self,
        request_id: str,
        error: Optional[Exception] = None,
        error_type: ErrorType = ErrorType.INTERNAL_ERROR,
    ):
        logger.error(
            f"Could not handle Command. request_id={request_id}, error={error}, error_type={error_type}"
        )
        response_payload = describe_error(error, error_type=error_type)
        self._responses_queue.put((request_id, response_payload))


def assembly_pipeline_sink(
    sink_config: dict,
) -> Callable[[ObjectDetectionPrediction, VideoFrame], None]:
    if sink_config["type"] != "udp_sink":
        raise NotImplementedError("Only `udp_socket` sink type is supported")
    sink = UDPSink.init(ip_address=sink_config["host"], port=sink_config["port"])
    return sink.send_predictions
