import os
import signal
from dataclasses import asdict
from multiprocessing import Process, Queue
from types import FrameType
from typing import Optional, Tuple

from pydantic import ValidationError

from inference.core import logger
from inference.core.exceptions import (
    MissingApiKeyError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
)
from inference.core.interfaces.camera.exceptions import StreamOperationNotAllowedError
from inference.core.interfaces.http.orjson_utils import serialise_workflow_result
from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
from inference.core.interfaces.stream.sinks import InMemoryBufferSink
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
    OperationStatus,
)
from inference.core.interfaces.stream_manager.manager_app.serialisation import (
    describe_error,
)


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
            command_type = CommandType(payload[TYPE_KEY])
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
            if command_type is CommandType.CONSUME_RESULT:
                return self._consume_results(request_id=request_id)
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
            parsed_payload = InitialisePipelinePayload.model_validate(payload)
            watchdog = BasePipelineWatchDog()
            buffer_sink = InMemoryBufferSink(
                queue_size=parsed_payload.sink_configuration.results_buffer_size,
            )
            self._buffer_sink = buffer_sink
            self._inference_pipeline = InferencePipeline.init_with_workflow(
                video_reference=parsed_payload.video_reference,
                workflow_specification=parsed_payload.workflow_specification,
                workspace_name=parsed_payload.workspace_name,
                workflow_id=parsed_payload.workflow_id,
                api_key=parsed_payload.api_key,
                image_input_name=parsed_payload.image_input_name,
                workflows_parameters=parsed_payload.workflows_parameters,
                on_prediction=self._buffer_sink.on_prediction,
                max_fps=parsed_payload.max_fps,
                watchdog=watchdog,
                source_buffer_filling_strategy=parsed_payload.source_buffer_filling_strategy,
                source_buffer_consumption_strategy=parsed_payload.source_buffer_consumption_strategy,
                video_source_properties=parsed_payload.video_source_properties,
                workflow_init_parameters=parsed_payload.workflow_init_parameters,
                workflows_thread_pool_workers=parsed_payload.workflows_thread_pool_workers,
                cancel_thread_pool_tasks_on_exit=parsed_payload.cancel_thread_pool_tasks_on_exit,
                video_metadata_input_name=parsed_payload.video_metadata_input_name,
            )
            self._watchdog = watchdog
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

    def _consume_results(self, request_id: str) -> None:
        try:
            if self._buffer_sink.empty():
                response_payload = {
                    STATUS_KEY: OperationStatus.SUCCESS,
                    "outputs": [],
                    "frames_metadata": [],
                }
            else:
                predictions, frames = [], []
                while not self._buffer_sink.empty():
                    predictions, frames = self._buffer_sink.consume_prediction()
                predictions = serialise_workflow_result(
                    result=predictions,
                )
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
        except ValidationError as error:
            self._handle_error(
                request_id=request_id, error=error, error_type=ErrorType.INVALID_PAYLOAD
            )
        except Exception as error:
            self._handle_error(
                request_id=request_id, error=error, error_type=ErrorType.INTERNAL_ERROR
            )

    def _handle_error(
        self,
        request_id: str,
        error: Optional[Exception] = None,
        error_type: ErrorType = ErrorType.INTERNAL_ERROR,
    ):
        logger.exception(
            f"Could not handle Command. request_id={request_id}, error={error}, error_type={error_type}"
        )
        response_payload = describe_error(error, error_type=error_type)
        self._responses_queue.put((request_id, response_payload))
