import asyncio
import os
import signal
import threading
import time
from dataclasses import asdict
from functools import partial
from multiprocessing import Process, Queue
from queue import Empty
from threading import Event
from types import FrameType
from typing import Dict, Optional, Tuple

import cv2 as cv
from pydantic import ValidationError

from inference.core import logger
from inference.core.exceptions import (
    MissingApiKeyError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
)
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.camera.exceptions import StreamOperationNotAllowedError
from inference.core.interfaces.http.orjson_utils import (
    serialise_single_workflow_result_element,
)
from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
from inference.core.interfaces.stream.sinks import InMemoryBufferSink, multi_sink
from inference.core.interfaces.stream.watchdog import (
    BasePipelineWatchDog,
    PipelineWatchDog,
)
from inference.core.interfaces.stream_manager.manager_app.entities import (
    STATUS_KEY,
    TYPE_KEY,
    CommandType,
    ErrorType,
    InitialisePipelinePayload,
    InitialiseWebRTCPipelinePayload,
    OperationStatus,
)
from inference.core.interfaces.stream_manager.manager_app.serialisation import (
    describe_error,
)
from inference.core.interfaces.stream_manager.manager_app.webrtc import (
    RTCPeerConnectionWithFPS,
    WebRTCVideoFrameProducer,
    init_rtc_peer_connection,
)
from inference.core.utils.async_utils import Queue as SyncAsyncQueue
from inference.core.workflows.errors import WorkflowSyntaxError
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData


def ignore_signal(signal_number: int, frame: FrameType) -> None:
    pid = os.getpid()
    logger.info(
        f"Ignoring signal {signal_number} in InferencePipelineManager in process:{pid}"
    )


class InferencePipelineManager(Process):
    @classmethod
    def init(
        cls, pipeline_id: str, command_queue: Queue, responses_queue: Queue
    ) -> "InferencePipelineManager":
        return cls(
            pipeline_id=pipeline_id,
            command_queue=command_queue,
            responses_queue=responses_queue,
        )

    def __init__(self, pipeline_id: str, command_queue: Queue, responses_queue: Queue):
        super().__init__()
        self._pipeline_id = pipeline_id
        self._command_queue = command_queue
        self._responses_queue = responses_queue
        self._inference_pipeline: Optional[InferencePipeline] = None
        self._watchdog: Optional[PipelineWatchDog] = None
        self._stop = False
        self._buffer_sink: Optional[InMemoryBufferSink] = None
        self._last_consume_time = (
            time.monotonic()
        )  # Track last consume time for the pipeline
        self._consumption_timeout: Optional[float] = (
            None  # Track zero consume timeout for the pipeline
        )

    def run(self) -> None:
        signal.signal(signal.SIGINT, ignore_signal)
        signal.signal(signal.SIGTERM, self._handle_termination_signal)

        while not self._stop:
            self._check_pipeline_timeout()
            # Handle commands from the queue
            try:
                command: Optional[Tuple[str, dict]] = self._command_queue.get(timeout=1)
            except Empty:
                continue
            if command is None:
                break
            request_id, payload = command
            self._handle_command(request_id=request_id, payload=payload)

    def _check_pipeline_timeout(self) -> None:
        if self._inference_pipeline and self._consumption_timeout is not None:
            time_since_last_consume = time.monotonic() - self._last_consume_time
            if time_since_last_consume > self._consumption_timeout:
                logger.info("Terminating pipeline due to zero consume timeout...")
                try:
                    pid = os.getpid()
                    logger.info(
                        f"Terminating pipeline due to timeout (no consumption):{pid}..."
                    )
                    if self._inference_pipeline is not None:
                        self._execute_termination()
                    self._command_queue.put(None)
                    logger.info(f"Timeout Termination successful in process:{pid}...")
                except Exception as error:
                    logger.warning(
                        f"Could not terminate pipeline gracefully. Error: {error}"
                    )

    def _handle_command(self, request_id: str, payload: dict) -> None:
        try:
            logger.info(f"Processing request={request_id}...")
            command_type = CommandType(payload[TYPE_KEY])
            if command_type is CommandType.INIT:
                return self._initialise_pipeline(request_id=request_id, payload=payload)
            if command_type is CommandType.WEBRTC:
                return self._start_webrtc(request_id=request_id, payload=payload)
            if command_type is CommandType.TERMINATE:
                return self._terminate_pipeline(request_id=request_id)
            if command_type is CommandType.MUTE:
                return self._mute_pipeline(request_id=request_id)
            if command_type is CommandType.RESUME:
                return self._resume_pipeline(request_id=request_id)
            if command_type is CommandType.STATUS:
                return self._get_pipeline_status(request_id=request_id)
            if command_type is CommandType.CONSUME_RESULT:
                return self._consume_results(request_id=request_id, payload=payload)
            raise NotImplementedError(
                f"Command type `{command_type}` cannot be handled"
            )
        except KeyError as error:
            self._handle_error(
                request_id=request_id,
                error=error,
                public_error_message="Invalid command sent to InferencePipeline manager - malformed payload",
                error_type=ErrorType.INVALID_PAYLOAD,
            )
        except NotImplementedError as error:
            self._handle_error(
                request_id=request_id,
                error=error,
                public_error_message=f"Invalid command sent to InferencePipeline manager - {error}",
                error_type=ErrorType.INVALID_PAYLOAD,
            )
        except Exception as error:
            self._handle_error(
                request_id=request_id,
                error=error,
                public_error_message="Unknown internal error. Raise this issue providing as "
                "much of a context as possible: https://github.com/roboflow/inference/issues",
                error_type=ErrorType.INTERNAL_ERROR,
            )

    def _initialise_pipeline(self, request_id: str, payload: dict) -> None:
        try:
            self._watchdog = BasePipelineWatchDog()
            parsed_payload = InitialisePipelinePayload.model_validate(payload)
            buffer_sink = InMemoryBufferSink.init(
                queue_size=parsed_payload.sink_configuration.results_buffer_size,
            )
            self._buffer_sink = buffer_sink
            self._inference_pipeline = InferencePipeline.init_with_workflow(
                video_reference=parsed_payload.video_configuration.video_reference,
                workflow_specification=parsed_payload.processing_configuration.workflow_specification,
                workspace_name=parsed_payload.processing_configuration.workspace_name,
                workflow_id=parsed_payload.processing_configuration.workflow_id,
                api_key=parsed_payload.api_key,
                image_input_name=parsed_payload.processing_configuration.image_input_name,
                workflows_parameters=parsed_payload.processing_configuration.workflows_parameters,
                on_prediction=self._buffer_sink.on_prediction,
                max_fps=parsed_payload.video_configuration.max_fps,
                watchdog=self._watchdog,
                source_buffer_filling_strategy=parsed_payload.video_configuration.source_buffer_filling_strategy,
                source_buffer_consumption_strategy=parsed_payload.video_configuration.source_buffer_consumption_strategy,
                video_source_properties=parsed_payload.video_configuration.video_source_properties,
                workflows_thread_pool_workers=parsed_payload.processing_configuration.workflows_thread_pool_workers,
                cancel_thread_pool_tasks_on_exit=parsed_payload.processing_configuration.cancel_thread_pool_tasks_on_exit,
                video_metadata_input_name=parsed_payload.processing_configuration.video_metadata_input_name,
                batch_collection_timeout=parsed_payload.video_configuration.batch_collection_timeout,
                decoding_buffer_size=parsed_payload.decoding_buffer_size,
                predictions_queue_size=parsed_payload.predictions_queue_size,
            )
            self._consumption_timeout = parsed_payload.consumption_timeout
            self._last_consume_time = time.monotonic()
            self._inference_pipeline.start(use_main_thread=False)
            self._responses_queue.put(
                (request_id, {STATUS_KEY: OperationStatus.SUCCESS})
            )
            logger.info(f"Pipeline initialised. request_id={request_id}...")
        except (
            ValidationError,
            MissingApiKeyError,
            KeyError,
            NotImplementedError,
        ) as error:
            self._handle_error(
                request_id=request_id,
                error=error,
                public_error_message="Could not decode InferencePipeline initialisation command payload.",
                error_type=ErrorType.INVALID_PAYLOAD,
            )
        except RoboflowAPINotAuthorizedError as error:
            self._handle_error(
                request_id=request_id,
                error=error,
                public_error_message="Invalid API key used or API key is missing. "
                "Visit https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key",
                error_type=ErrorType.AUTHORISATION_ERROR,
            )
        except RoboflowAPINotNotFoundError as error:
            self._handle_error(
                request_id=request_id,
                error=error,
                public_error_message="Requested Roboflow resources (models / workflows etc.) not available or "
                "wrong API key used.",
                error_type=ErrorType.NOT_FOUND,
            )
        except WorkflowSyntaxError as error:
            self._handle_error(
                request_id=request_id,
                error=error,
                public_error_message="Provided workflow configuration is not valid.",
                error_type=ErrorType.INVALID_PAYLOAD,
            )

    def _start_webrtc(self, request_id: str, payload: dict):
        try:
            self._watchdog = BasePipelineWatchDog()
            parsed_payload = InitialiseWebRTCPipelinePayload.model_validate(payload)

            def start_loop(loop: asyncio.AbstractEventLoop):
                asyncio.set_event_loop(loop)
                loop.run_forever()

            loop = asyncio.new_event_loop()
            t = threading.Thread(target=start_loop, args=(loop,), daemon=True)
            t.start()

            webrtc_offer = parsed_payload.webrtc_offer
            webrtc_turn_config = parsed_payload.webrtc_turn_config
            webcam_fps = parsed_payload.webcam_fps
            to_inference_queue = SyncAsyncQueue(loop=loop)
            from_inference_queue = SyncAsyncQueue(loop=loop)

            stop_event = Event()

            future = asyncio.run_coroutine_threadsafe(
                init_rtc_peer_connection(
                    webrtc_offer=webrtc_offer,
                    webrtc_turn_config=webrtc_turn_config,
                    to_inference_queue=to_inference_queue,
                    from_inference_queue=from_inference_queue,
                    webrtc_peer_timeout=parsed_payload.webrtc_peer_timeout,
                    feedback_stop_event=stop_event,
                    asyncio_loop=loop,
                    webcam_fps=webcam_fps,
                ),
                loop,
            )
            peer_connection: RTCPeerConnectionWithFPS = future.result()

            webrtc_producer = partial(
                WebRTCVideoFrameProducer,
                to_inference_queue=to_inference_queue,
                stop_event=stop_event,
                webrtc_video_transform_track=peer_connection.video_transform_track,
                webrtc_peer_timeout=parsed_payload.webrtc_peer_timeout,
            )

            def webrtc_sink(
                prediction: Dict[str, WorkflowImageData], video_frame: VideoFrame
            ) -> None:
                errors = []
                if not any(
                    isinstance(v, WorkflowImageData) for v in prediction.values()
                ):
                    errors.append("Visualisation blocks were not executed")
                    errors.append("or workflow was not configured to output visuals.")
                    errors.append(
                        "Please try to adjust the scene so models detect objects"
                    )
                    errors.append("or stop preview, update workflow and try again.")
                    result_frame = video_frame.image.copy()
                    for row, error in enumerate(errors):
                        result_frame = cv.putText(
                            result_frame,
                            error,
                            (10, 20 + 30 * row),
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )
                    from_inference_queue.sync_put(result_frame)
                    return
                if parsed_payload.stream_output[0] not in prediction or not isinstance(
                    prediction[parsed_payload.stream_output[0]], WorkflowImageData
                ):
                    for output in prediction.values():
                        if isinstance(output, WorkflowImageData):
                            from_inference_queue.sync_put(output.numpy_image)
                            return
                from_inference_queue.sync_put(
                    prediction[parsed_payload.stream_output[0]].numpy_image
                )

            buffer_sink = InMemoryBufferSink.init(
                queue_size=parsed_payload.sink_configuration.results_buffer_size,
            )
            self._buffer_sink = buffer_sink
            chained_sink = partial(
                multi_sink, sinks=[buffer_sink.on_prediction, webrtc_sink]
            )

            self._inference_pipeline = InferencePipeline.init_with_workflow(
                video_reference=webrtc_producer,
                workflow_specification=parsed_payload.processing_configuration.workflow_specification,
                workspace_name=parsed_payload.processing_configuration.workspace_name,
                workflow_id=parsed_payload.processing_configuration.workflow_id,
                api_key=parsed_payload.api_key,
                image_input_name=parsed_payload.processing_configuration.image_input_name,
                workflows_parameters=parsed_payload.processing_configuration.workflows_parameters,
                on_prediction=chained_sink,
                max_fps=parsed_payload.video_configuration.max_fps,
                watchdog=self._watchdog,
                source_buffer_filling_strategy=parsed_payload.video_configuration.source_buffer_filling_strategy,
                source_buffer_consumption_strategy=parsed_payload.video_configuration.source_buffer_consumption_strategy,
                video_source_properties=parsed_payload.video_configuration.video_source_properties,
                workflows_thread_pool_workers=parsed_payload.processing_configuration.workflows_thread_pool_workers,
                cancel_thread_pool_tasks_on_exit=parsed_payload.processing_configuration.cancel_thread_pool_tasks_on_exit,
                video_metadata_input_name=parsed_payload.processing_configuration.video_metadata_input_name,
                batch_collection_timeout=parsed_payload.video_configuration.batch_collection_timeout,
                predictions_queue_size=parsed_payload.predictions_queue_size,
                decoding_buffer_size=parsed_payload.decoding_buffer_size,
            )
            self._inference_pipeline.start(use_main_thread=False)
            self._responses_queue.put(
                (
                    request_id,
                    {
                        STATUS_KEY: OperationStatus.SUCCESS,
                        "sdp": peer_connection.localDescription.sdp,
                        "type": peer_connection.localDescription.type,
                    },
                )
            )
            logger.info(f"WebRTC pipeline initialised. request_id={request_id}...")
        except (
            ValidationError,
            MissingApiKeyError,
            KeyError,
            NotImplementedError,
        ) as error:
            self._handle_error(
                request_id=request_id,
                error=error,
                public_error_message="Could not decode InferencePipeline initialisation command payload.",
                error_type=ErrorType.INVALID_PAYLOAD,
            )
        except RoboflowAPINotAuthorizedError as error:
            self._handle_error(
                request_id=request_id,
                error=error,
                public_error_message="Invalid API key used or API key is missing. "
                "Visit https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key",
                error_type=ErrorType.AUTHORISATION_ERROR,
            )
        except RoboflowAPINotNotFoundError as error:
            self._handle_error(
                request_id=request_id,
                error=error,
                public_error_message="Requested Roboflow resources (models / workflows etc.) not available or "
                "wrong API key used.",
                error_type=ErrorType.NOT_FOUND,
            )
        except WorkflowSyntaxError as error:
            self._handle_error(
                request_id=request_id,
                error=error,
                public_error_message="Provided workflow configuration is not valid.",
                error_type=ErrorType.INVALID_PAYLOAD,
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
                request_id=request_id,
                error=error,
                public_error_message="Cannot get pipeline status in the current state of InferencePipeline.",
                error_type=ErrorType.OPERATION_ERROR,
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
                request_id=request_id,
                public_error_message="Cannot retrieve InferencePipeline status. Internal Error. Service misconfigured.",
                error_type=ErrorType.OPERATION_ERROR,
            )
        try:
            self._inference_pipeline.mute_stream()
            logger.info(f"Pipeline muted. request_id={request_id}...")
            self._responses_queue.put(
                (request_id, {STATUS_KEY: OperationStatus.SUCCESS})
            )
        except StreamOperationNotAllowedError as error:
            self._handle_error(
                request_id=request_id,
                error=error,
                public_error_message="Cannot get pipeline status in the current state of InferencePipeline.",
                error_type=ErrorType.OPERATION_ERROR,
            )

    def _resume_pipeline(self, request_id: str) -> None:
        if self._inference_pipeline is None:
            return self._handle_error(
                request_id=request_id,
                public_error_message="Cannot retrieve InferencePipeline status. Internal Error. Service misconfigured.",
                error_type=ErrorType.OPERATION_ERROR,
            )
        try:
            self._inference_pipeline.resume_stream()
            logger.info(f"Pipeline resumed. request_id={request_id}...")
            self._responses_queue.put(
                (request_id, {STATUS_KEY: OperationStatus.SUCCESS})
            )
        except StreamOperationNotAllowedError as error:
            self._handle_error(
                request_id=request_id,
                error=error,
                public_error_message="Cannot get pipeline status in the current state of InferencePipeline.",
                error_type=ErrorType.OPERATION_ERROR,
            )

    def _get_pipeline_status(self, request_id: str) -> None:
        if self._watchdog is None:
            return self._handle_error(
                request_id=request_id,
                error_type=ErrorType.OPERATION_ERROR,
                public_error_message="Cannot retrieve InferencePipeline status. "
                "Try again later - Inference Pipeline not initialised.",
            )
        try:
            report = self._watchdog.get_report()
            if report is None:
                return self._handle_error(
                    request_id=request_id,
                    error_type=ErrorType.OPERATION_ERROR,
                    public_error_message="Cannot retrieve InferencePipeline status. Try again later.",
                )
            response_payload = {
                STATUS_KEY: OperationStatus.SUCCESS,
                "report": asdict(report),
            }
            self._responses_queue.put((request_id, response_payload))
            logger.info(f"Pipeline status returned. request_id={request_id}...")
        except StreamOperationNotAllowedError as error:
            self._handle_error(
                request_id=request_id,
                error=error,
                public_error_message="Cannot get pipeline status in the current state of InferencePipeline.",
                error_type=ErrorType.OPERATION_ERROR,
            )

    def _consume_results(self, request_id: str, payload: dict) -> None:
        try:
            if self._buffer_sink.empty():
                response_payload = {
                    STATUS_KEY: OperationStatus.SUCCESS,
                    "outputs": [],
                    "frames_metadata": [],
                }
                self._responses_queue.put((request_id, response_payload))
                return None
            excluded_fields = payload.get("excluded_fields")
            predictions, frames = self._buffer_sink.consume_prediction()
            self._last_consume_time = time.monotonic()
            predictions = [
                (
                    serialise_single_workflow_result_element(
                        result_element=result_element,
                        excluded_fields=excluded_fields,
                    )
                    if result_element is not None
                    else None
                )
                for result_element in predictions
            ]
            frames_metadata = []
            for frame in frames:
                if frame is None:
                    frames_metadata.append(None)
                else:
                    frames_metadata.append(
                        {
                            "frame_timestamp": frame.frame_timestamp.isoformat(),
                            "frame_id": frame.frame_id,
                            "source_id": frame.source_id,
                        }
                    )
            response_payload = {
                STATUS_KEY: OperationStatus.SUCCESS,
                "outputs": predictions,
                "frames_metadata": frames_metadata,
            }
            self._responses_queue.put((request_id, response_payload))
        except Exception as error:
            self._handle_error(
                request_id=request_id,
                error=error,
                public_error_message="Unexpected error with InferencePipeline results consumption.",
                error_type=ErrorType.OPERATION_ERROR,
            )

    def _handle_error(
        self,
        request_id: str,
        error: Optional[Exception] = None,
        public_error_message: Optional[str] = None,
        error_type: ErrorType = ErrorType.INTERNAL_ERROR,
    ):
        logger.exception(
            f"Could not handle Command. request_id={request_id}, "
            f"error={error}, error_type={error_type}, public_error_message={public_error_message}"
        )
        response_payload = describe_error(
            error, error_type=error_type, public_error_message=public_error_message
        )
        self._responses_queue.put((request_id, response_payload))
