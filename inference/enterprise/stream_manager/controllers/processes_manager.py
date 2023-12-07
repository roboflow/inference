import argparse
import json
import signal
import sys
from multiprocessing import Process, Queue
from socketserver import TCPServer, BaseRequestHandler
from types import FrameType
from typing import Dict, Optional, Tuple
from uuid import uuid4

from inference.core import logger
from inference.enterprise.stream_manager.controllers.entities import REQUEST_ID_KEY, RESPONSE_KEY, ErrorType, ENCODING, \
    COMMAND_KEY, TYPE_KEY, CommandType, PIPELINE_ID_KEY, STATUS_KEY, OperationStatus
from inference.enterprise.stream_manager.controllers.inference_pipeline import describe_error, InferencePipelineManager
from inference.enterprise.stream_manager.controllers.serialisation import serialise_to_json

PROCESSES_TABLE: Dict[str, Tuple[Process, Queue, Queue]] = {}
MAX_MESSAGE_SIZE = 16384


class InferencePipelinesManagerHandler(BaseRequestHandler):

    def handle(self) -> None:
        pipeline_id: Optional[str] = None
        request_id = str(uuid4())
        try:
            data = json.loads(self.request.recv(MAX_MESSAGE_SIZE).decode(ENCODING))
            data[TYPE_KEY] = CommandType(data[TYPE_KEY])
            if data[TYPE_KEY] is CommandType.LIST_PIPELINES:
                return self._list_pipelines(request_id=request_id)
            if data[TYPE_KEY] is CommandType.INIT:
                return self._initialise_pipeline(request_id=request_id, command=data)
            pipeline_id = data[PIPELINE_ID_KEY]
            if data[TYPE_KEY] is CommandType.TERMINATE:
                self._terminate_pipeline(request_id=request_id, pipeline_id=pipeline_id, command=data)
            else:
                response = handle_command(request_id=request_id, pipeline_id=pipeline_id, command=data)
                serialised_response = prepare_response(request_id=request_id, response=response,
                                                       pipeline_id=pipeline_id)
                self.request.sendall(serialised_response)
        except KeyError as error:
            logger.error(f"Invalid payload in processes manager. error={error} request_id={request_id}...")
            payload = prepare_error_response(request_id=request_id, error=error, error_type=ErrorType.INVALID_PAYLOAD, pipeline_id=pipeline_id)
            self.request.sendall(payload)
        except Exception as error:
            logger.error(f"Internal error in processes manager. error={error} request_id={request_id}...")
            payload = prepare_error_response(request_id=request_id, error=error, error_type=ErrorType.INTERNAL_ERROR, pipeline_id=pipeline_id)
            self.request.sendall(payload)

    def _list_pipelines(self, request_id: str) -> None:
        global PROCESSES_TABLE
        serialised_response = prepare_response(
            request_id=request_id, response={"pipelines": list(PROCESSES_TABLE.keys())}, pipeline_id=None
        )
        self.request.sendall(serialised_response)

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
        PROCESSES_TABLE[pipeline_id] = (inference_pipeline_manager, command_queue, responses_queue)
        command_queue.put((request_id, command))
        response = get_response_ignoring_thrash(responses_queue=responses_queue, matching_request_id=request_id)
        serialised_response = prepare_response(request_id=request_id, response=response, pipeline_id=pipeline_id)
        self.request.sendall(serialised_response)

    def _terminate_pipeline(self, request_id: str, pipeline_id: str, command: dict) -> None:
        response = handle_command(request_id=request_id, pipeline_id=pipeline_id, command=command)
        if response[STATUS_KEY] is OperationStatus.SUCCESS:
            logger.info(f"Joining inference pipeline. pipeline_id={pipeline_id} request_id={request_id}")
            join_inference_pipeline(pipeline_id=pipeline_id)
            logger.info(f"Joined inference pipeline. pipeline_id={pipeline_id} request_id={request_id}")
        serialised_response = prepare_response(request_id=request_id, response=response, pipeline_id=pipeline_id)
        self.request.sendall(serialised_response)

def get_response_ignoring_thrash(responses_queue: Queue, matching_request_id: str) -> dict:
    while True:
        response = responses_queue.get()
        if response[0] == matching_request_id:
            return response[1]
        logger.warning(f"Dropping response for request_id={response[0]} with payload={response[1]}")


def prepare_error_response(request_id: str, error: Exception, error_type: ErrorType, pipeline_id: Optional[str]) -> bytes:
    error_description = describe_error(exception=error, error_type=error_type)
    return prepare_response(request_id=request_id, response=error_description, pipeline_id=pipeline_id)


def prepare_response(request_id: str, response: dict, pipeline_id: Optional[str]) -> bytes:
    payload = json.dumps(
        {REQUEST_ID_KEY: request_id, RESPONSE_KEY: response, PIPELINE_ID_KEY: pipeline_id},
        default=serialise_to_json,
    )
    return payload.encode(ENCODING)


def handle_command(request_id: str, pipeline_id: str, command: dict) -> dict:
    global PROCESSES_TABLE
    if pipeline_id not in PROCESSES_TABLE:
        return describe_error(exception=None, error_type=ErrorType.NOT_FOUND)
    _, command_queue, responses_queue = PROCESSES_TABLE[pipeline_id]
    command_queue.put((request_id, command))
    return get_response_ignoring_thrash(responses_queue=responses_queue, matching_request_id=request_id)


def join_inference_pipeline(pipeline_id: str) -> None:
    global PROCESSES_TABLE
    inference_pipeline_manager, command_queue, responses_queue = PROCESSES_TABLE[pipeline_id]
    inference_pipeline_manager.join()
    del PROCESSES_TABLE[pipeline_id]


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Inference Pipeline Processes Manager")
    parser.add_argument("--host", type=str, required=True, help="IP address to bind the server")
    parser.add_argument("--port", type=int, required=True, help="Port to bind the server")
    args = parser.parse_args()
    with TCPServer((args.host, args.port), InferencePipelinesManagerHandler) as server:
        server.serve_forever()
