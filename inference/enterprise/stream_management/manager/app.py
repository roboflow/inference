import os
import signal
import sys
from multiprocessing import Process, Queue
from socketserver import BaseRequestHandler
from types import FrameType
from typing import Dict, Optional, Tuple
from uuid import uuid4

from inference.core import logger
from inference.enterprise.stream_management.manager.communication import (
    receive_socket_data,
    send_data_trough_socket,
)
from inference.enterprise.stream_management.manager.entities import (
    PIPELINE_ID_KEY,
    STATUS_KEY,
    TYPE_KEY,
    CommandType,
    ErrorType,
    OperationStatus,
)
from inference.enterprise.stream_management.manager.inference_pipeline import (
    InferencePipelineManager,
)
from inference.enterprise.stream_management.manager.tcp_server import RoboflowTCPServer
from inference.enterprise.stream_management.manager.serialisation import (
    describe_error,
    prepare_error_response,
    prepare_response,
)

PROCESSES_TABLE: Dict[str, Tuple[Process, Queue, Queue]] = {}
HEADER_SIZE = 4
SOCKET_BUFFER_SIZE = 16384
HOST = os.getenv("STREAM_MANAGER_HOST", "127.0.0.1")
PORT = int(os.getenv("STREAM_MANAGER_PORT", "7070"))
SOCKET_TIMEOUT = float(os.getenv("STREAM_MANAGER_SOCKET_TIMEOUT", "5.0"))


class InferencePipelinesManagerHandler(BaseRequestHandler):
    def handle(self) -> None:
        pipeline_id: Optional[str] = None
        request_id = str(uuid4())
        try:
            data = receive_socket_data(
                source=self.request,
                header_size=HEADER_SIZE,
                buffer_size=SOCKET_BUFFER_SIZE,
            )
            data[TYPE_KEY] = CommandType(data[TYPE_KEY])
            if data[TYPE_KEY] is CommandType.LIST_PIPELINES:
                return self._list_pipelines(request_id=request_id)
            if data[TYPE_KEY] is CommandType.INIT:
                return self._initialise_pipeline(request_id=request_id, command=data)
            pipeline_id = data[PIPELINE_ID_KEY]
            if data[TYPE_KEY] is CommandType.TERMINATE:
                self._terminate_pipeline(
                    request_id=request_id, pipeline_id=pipeline_id, command=data
                )
            else:
                response = handle_command(
                    request_id=request_id, pipeline_id=pipeline_id, command=data
                )
                serialised_response = prepare_response(
                    request_id=request_id, response=response, pipeline_id=pipeline_id
                )
                send_data_trough_socket(
                    target=self.request,
                    header_size=HEADER_SIZE,
                    data=serialised_response,
                    request_id=request_id,
                    pipeline_id=pipeline_id,
                )
        except KeyError as error:
            logger.error(
                f"Invalid payload in processes manager. error={error} request_id={request_id}..."
            )
            payload = prepare_error_response(
                request_id=request_id,
                error=error,
                error_type=ErrorType.INVALID_PAYLOAD,
                pipeline_id=pipeline_id,
            )
            send_data_trough_socket(
                target=self.request,
                header_size=HEADER_SIZE,
                data=payload,
                request_id=request_id,
                pipeline_id=pipeline_id,
            )
        except Exception as error:
            logger.error(
                f"Internal error in processes manager. error={error} request_id={request_id}..."
            )
            payload = prepare_error_response(
                request_id=request_id,
                error=error,
                error_type=ErrorType.INTERNAL_ERROR,
                pipeline_id=pipeline_id,
            )
            send_data_trough_socket(
                target=self.request,
                header_size=HEADER_SIZE,
                data=payload,
                request_id=request_id,
                pipeline_id=pipeline_id,
            )

    def _list_pipelines(self, request_id: str) -> None:
        global PROCESSES_TABLE
        serialised_response = prepare_response(
            request_id=request_id,
            response={
                "pipelines": list(PROCESSES_TABLE.keys()),
                STATUS_KEY: OperationStatus.SUCCESS,
            },
            pipeline_id=None,
        )
        send_data_trough_socket(
            target=self.request,
            header_size=HEADER_SIZE,
            data=serialised_response,
            request_id=request_id,
        )

    def _initialise_pipeline(self, request_id: str, command: dict) -> None:
        global PROCESSES_TABLE
        pipeline_id = str(uuid4())
        command_queue = Queue()
        responses_queue = Queue()
        inference_pipeline_manager = InferencePipelineManager(
            command_queue=command_queue,
            responses_queue=responses_queue,
        )
        inference_pipeline_manager.start()
        PROCESSES_TABLE[pipeline_id] = (
            inference_pipeline_manager,
            command_queue,
            responses_queue,
        )
        command_queue.put((request_id, command))
        response = get_response_ignoring_thrash(
            responses_queue=responses_queue, matching_request_id=request_id
        )
        serialised_response = prepare_response(
            request_id=request_id, response=response, pipeline_id=pipeline_id
        )
        send_data_trough_socket(
            target=self.request,
            header_size=HEADER_SIZE,
            data=serialised_response,
            request_id=request_id,
            pipeline_id=pipeline_id,
        )

    def _terminate_pipeline(
        self, request_id: str, pipeline_id: str, command: dict
    ) -> None:
        response = handle_command(
            request_id=request_id, pipeline_id=pipeline_id, command=command
        )
        if response[STATUS_KEY] is OperationStatus.SUCCESS:
            logger.info(
                f"Joining inference pipeline. pipeline_id={pipeline_id} request_id={request_id}"
            )
            join_inference_pipeline(pipeline_id=pipeline_id)
            logger.info(
                f"Joined inference pipeline. pipeline_id={pipeline_id} request_id={request_id}"
            )
        serialised_response = prepare_response(
            request_id=request_id, response=response, pipeline_id=pipeline_id
        )
        send_data_trough_socket(
            target=self.request,
            header_size=HEADER_SIZE,
            data=serialised_response,
            request_id=request_id,
            pipeline_id=pipeline_id,
        )


def get_response_ignoring_thrash(
    responses_queue: Queue, matching_request_id: str
) -> dict:
    while True:
        response = responses_queue.get()
        if response[0] == matching_request_id:
            return response[1]
        logger.warning(
            f"Dropping response for request_id={response[0]} with payload={response[1]}"
        )


def handle_command(request_id: str, pipeline_id: str, command: dict) -> dict:
    global PROCESSES_TABLE
    if pipeline_id not in PROCESSES_TABLE:
        return describe_error(exception=None, error_type=ErrorType.NOT_FOUND)
    _, command_queue, responses_queue = PROCESSES_TABLE[pipeline_id]
    command_queue.put((request_id, command))
    return get_response_ignoring_thrash(
        responses_queue=responses_queue, matching_request_id=request_id
    )


def execute_termination(signal_number: int, frame: FrameType) -> None:
    global PROCESSES_TABLE
    pipeline_ids = list(PROCESSES_TABLE.keys())
    for pipeline_id in pipeline_ids:
        logger.info(f"Terminating pipeline: {pipeline_id}")
        PROCESSES_TABLE[pipeline_id][0].terminate()
        logger.info(f"Pipeline: {pipeline_id} terminated.")
        logger.info(f"Joining pipeline: {pipeline_id}")
        PROCESSES_TABLE[pipeline_id][0].join()
        logger.info(f"Pipeline: {pipeline_id} joined.")
    logger.info(f"Termination handler completed.")
    sys.exit(0)


def join_inference_pipeline(pipeline_id: str) -> None:
    global PROCESSES_TABLE
    inference_pipeline_manager, command_queue, responses_queue = PROCESSES_TABLE[
        pipeline_id
    ]
    inference_pipeline_manager.join()
    del PROCESSES_TABLE[pipeline_id]


if __name__ == "__main__":
    signal.signal(signal.SIGINT, execute_termination)
    signal.signal(signal.SIGTERM, execute_termination)
    with RoboflowTCPServer(
        server_address=(HOST, PORT),
        handler_class=InferencePipelinesManagerHandler,
        socket_operations_timeout=SOCKET_TIMEOUT,
    ) as server:
        logger.info(
            f"Inference Pipeline Processes Manager is ready to accept connections at {(HOST, PORT)}"
        )
        server.serve_forever()
