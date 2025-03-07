import os
import signal
import socket
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Process, Queue
from socketserver import BaseRequestHandler, BaseServer
from threading import Lock, Thread
from types import FrameType
from typing import Any, Deque, Dict, List, Optional, Tuple
from uuid import uuid4

import psutil

from inference.core import logger
from inference.core.env import (
    STREAM_MANAGER_MAX_RAM_MB,
    STREAM_MANAGER_RAM_USAGE_QUEUE_SIZE,
)
from inference.core.interfaces.camera.video_source import StreamState
from inference.core.interfaces.stream_manager.manager_app.communication import (
    receive_socket_data,
    send_data_trough_socket,
)
from inference.core.interfaces.stream_manager.manager_app.entities import (
    PIPELINE_ID_KEY,
    REPORT_KEY,
    SOURCES_METADATA_KEY,
    STATE_KEY,
    STATUS_KEY,
    TYPE_KEY,
    CommandType,
    ErrorType,
    OperationStatus,
)
from inference.core.interfaces.stream_manager.manager_app.errors import (
    MalformedPayloadError,
)
from inference.core.interfaces.stream_manager.manager_app.inference_pipeline_manager import (
    InferencePipelineManager,
)
from inference.core.interfaces.stream_manager.manager_app.serialisation import (
    describe_error,
    prepare_error_response,
    prepare_response,
)
from inference.core.interfaces.stream_manager.manager_app.tcp_server import (
    RoboflowTCPServer,
)


@dataclass
class ManagedInferencePipeline:
    pipeline_id: str
    pipeline_manager: InferencePipelineManager
    command_queue: Queue
    responses_queue: Queue
    operation_lock: Lock
    is_idle: bool
    ram_usage_queue: Deque = field(
        default_factory=lambda: deque(
            maxlen=min(max(STREAM_MANAGER_RAM_USAGE_QUEUE_SIZE, 10), 10)
        )
    )


PROCESSES_TABLE: Dict[str, ManagedInferencePipeline] = {}
PROCESSES_TABLE_LOCK = Lock()
HEADER_SIZE = 4
SOCKET_BUFFER_SIZE = 16384
HOST = os.getenv("STREAM_MANAGER_HOST", "127.0.0.1")
PORT = int(os.getenv("STREAM_MANAGER_PORT", "7070"))
SOCKET_TIMEOUT = float(os.getenv("STREAM_MANAGER_SOCKET_TIMEOUT", "5.0"))


class InferencePipelinesManagerHandler(BaseRequestHandler):
    def __init__(
        self,
        request: socket.socket,
        client_address: Any,
        server: BaseServer,
        processes_table: Dict[str, ManagedInferencePipeline],
    ):
        self._processes_table = processes_table  # in this case it's required to set the state of class before superclass init - as it invokes ()
        super().__init__(request, client_address, server)

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
            if data[TYPE_KEY] is CommandType.WEBRTC:
                return self._start_webrtc(request_id=request_id, command=data)
            pipeline_id = data[PIPELINE_ID_KEY]
            if data[TYPE_KEY] is CommandType.TERMINATE:
                self._terminate_pipeline(
                    request_id=request_id, pipeline_id=pipeline_id, command=data
                )
            else:
                response = handle_command(
                    processes_table=self._processes_table,
                    request_id=request_id,
                    pipeline_id=pipeline_id,
                    command=data,
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
        except (KeyError, ValueError, MalformedPayloadError) as error:
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
        serialised_response = prepare_response(
            request_id=request_id,
            response={
                "pipelines": [
                    k for k, v in self._processes_table.items() if not v.is_idle
                ],
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
        managed_pipeline = get_or_spawn_pipeline_process(
            processes_table=self._processes_table,
        )
        managed_pipeline.command_queue.put((request_id, command))
        response = get_response_ignoring_thrash(
            responses_queue=managed_pipeline.responses_queue,
            matching_request_id=request_id,
        )
        serialised_response = prepare_response(
            request_id=request_id,
            response=response,
            pipeline_id=managed_pipeline.pipeline_id,
        )
        send_data_trough_socket(
            target=self.request,
            header_size=HEADER_SIZE,
            data=serialised_response,
            request_id=request_id,
            pipeline_id=managed_pipeline.pipeline_id,
        )

    def _start_webrtc(self, request_id: str, command: dict):
        managed_pipeline = get_or_spawn_pipeline_process(
            processes_table=self._processes_table,
        )
        managed_pipeline.command_queue.put((request_id, command))
        response = get_response_ignoring_thrash(
            responses_queue=managed_pipeline.responses_queue,
            matching_request_id=request_id,
        )
        serialised_response = prepare_response(
            request_id=request_id,
            response=response,
            pipeline_id=managed_pipeline.pipeline_id,
        )
        send_data_trough_socket(
            target=self.request,
            header_size=HEADER_SIZE,
            data=serialised_response,
            request_id=request_id,
            pipeline_id=managed_pipeline.pipeline_id,
        )

    def _terminate_pipeline(
        self, request_id: str, pipeline_id: str, command: dict
    ) -> None:
        response = handle_command(
            processes_table=self._processes_table,
            request_id=request_id,
            pipeline_id=pipeline_id,
            command=command,
        )
        if response[STATUS_KEY] is OperationStatus.SUCCESS:
            logger.info(
                f"Joining inference pipeline. pipeline_id={pipeline_id} request_id={request_id}"
            )
            join_inference_pipeline(
                processes_table=self._processes_table, pipeline_id=pipeline_id
            )
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


def handle_command(
    processes_table: Dict[str, ManagedInferencePipeline],
    request_id: str,
    pipeline_id: str,
    command: dict,
) -> dict:
    if pipeline_id not in processes_table:
        return describe_error(
            exception=None,
            error_type=ErrorType.NOT_FOUND,
            public_error_message=f"Could not find InferencePipeline with id={pipeline_id}.",
        )
    managed_pipeline = processes_table[pipeline_id]
    with managed_pipeline.operation_lock:
        managed_pipeline.command_queue.put((request_id, command))
        return get_response_ignoring_thrash(
            responses_queue=managed_pipeline.responses_queue,
            matching_request_id=request_id,
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


def execute_termination(
    signal_number: int,
    frame: FrameType,
    processes_table: Dict[str, Tuple[Process, Queue, Queue, Lock]],
) -> None:
    with PROCESSES_TABLE_LOCK:
        pipeline_ids = list(processes_table.keys())
        for pipeline_id in pipeline_ids:
            logger.info(f"Terminating pipeline: {pipeline_id}")
            processes_table[pipeline_id][0].terminate()
            logger.info(f"Pipeline: {pipeline_id} terminated.")
            logger.info(f"Joining pipeline: {pipeline_id}")
            processes_table[pipeline_id][0].join()
            logger.info(f"Pipeline: {pipeline_id} joined.")
        logger.info(f"Termination handler completed.")
        sys.exit(0)


def join_inference_pipeline(
    processes_table: Dict[str, ManagedInferencePipeline], pipeline_id: str
) -> None:
    inference_pipeline_manager = processes_table[pipeline_id].pipeline_manager
    inference_pipeline_manager.join()
    del processes_table[pipeline_id]


def check_process_health() -> None:
    while True:
        total_ram_usage: int = _get_current_process_ram_usage_mb()
        with PROCESSES_TABLE_LOCK:
            for pipeline_id, managed_pipeline in list(PROCESSES_TABLE.items()):
                process = managed_pipeline.pipeline_manager
                process_ram_usage_mb = _get_process_memory_usage_mb(process=process)
                managed_pipeline.ram_usage_queue.append(process_ram_usage_mb)

                if not process.is_alive():
                    logger.warning(
                        "Process for pipeline_id=%s is not alive. Terminating...",
                        pipeline_id,
                    )
                    process.terminate()
                    process.join()
                    del PROCESSES_TABLE[pipeline_id]
                    continue

                total_ram_usage += process_ram_usage_mb
                if (
                    STREAM_MANAGER_MAX_RAM_MB is not None
                    and total_ram_usage > STREAM_MANAGER_MAX_RAM_MB
                ):
                    logger.warning(
                        "Process for pipeline_id=%s is above RAM limit.", pipeline_id
                    )

                if managed_pipeline.is_idle:
                    # skipping idle pipelines in status probing
                    continue
                command = {
                    TYPE_KEY: CommandType.STATUS,
                    PIPELINE_ID_KEY: pipeline_id,
                }
                response = handle_command(
                    processes_table=PROCESSES_TABLE,
                    request_id=uuid.uuid4().hex,
                    pipeline_id=pipeline_id,
                    command=command,
                )
                if (
                    REPORT_KEY not in response
                    or SOURCES_METADATA_KEY not in response[REPORT_KEY]
                ):
                    continue
                all_sources_statues = set(
                    source_metadata[STATE_KEY]
                    for source_metadata in response[REPORT_KEY][SOURCES_METADATA_KEY]
                    if STATE_KEY in source_metadata
                )
                if not all_sources_statues:
                    continue
                if all_sources_statues.issubset({StreamState.ENDED, StreamState.ERROR}):
                    total_ram_usage -= process_ram_usage_mb
                    logger.info(
                        "All sources depleted in pipeline %s, terminating", pipeline_id
                    )
                    command = {
                        TYPE_KEY: CommandType.TERMINATE,
                        PIPELINE_ID_KEY: pipeline_id,
                    }
                    response = handle_command(
                        processes_table=PROCESSES_TABLE,
                        request_id=uuid.uuid4().hex,
                        pipeline_id=pipeline_id,
                        command=command,
                    )
                    if not response.get(STATUS_KEY) == "success":
                        logger.error(
                            "Malformed response returned by termination command, '%s'",
                            response,
                        )
                        continue
                    process.join()
                    del PROCESSES_TABLE[pipeline_id]
        time.sleep(1)


def _get_current_process_ram_usage_mb() -> int:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def get_or_spawn_pipeline_process(
    processes_table: Dict[str, ManagedInferencePipeline],
) -> ManagedInferencePipeline:
    with PROCESSES_TABLE_LOCK:
        idle_pipelines = get_idle_pipelines_id(processes_table=processes_table)
        if len(idle_pipelines) > 0:
            chosen_pipeline = processes_table[idle_pipelines[0]]
            chosen_pipeline.is_idle = False
            return chosen_pipeline

        current_ram_usage = (
            sum(
                (
                    managed_pipeline.ram_usage_queue[-1]
                    if managed_pipeline.ram_usage_queue
                    else 0
                )
                for managed_pipeline in processes_table.values()
            )
            + _get_current_process_ram_usage_mb()
        )
        highest_pipeline_ram_usage = 0
        if processes_table:
            highest_pipeline_ram_usage = max(
                max(
                    (
                        managed_pipeline.ram_usage_queue
                        if managed_pipeline.ram_usage_queue
                        else 0
                    )
                    for managed_pipeline in processes_table.values()
                )
            )

        if (
            STREAM_MANAGER_MAX_RAM_MB is not None
            and current_ram_usage + highest_pipeline_ram_usage
            > STREAM_MANAGER_MAX_RAM_MB
        ):
            raise Exception(
                "Cannot spawn new pipeline due to insufficient RAM,"
                f" current RAM usage: {current_ram_usage}MB,"
                f" predicted RAM required to spawn new pipeline: {highest_pipeline_ram_usage}MB,"
                f" max: {STREAM_MANAGER_MAX_RAM_MB}MB"
            )
        new_pipeline_id = spawn_managed_pipeline_process(
            processes_table=processes_table,
            mark_as_idle=False,
        )
        return processes_table[new_pipeline_id]


def ensure_idle_pipelines_warmed_up(expected_warmed_up_pipelines: int) -> None:
    while True:
        with PROCESSES_TABLE_LOCK:
            idle_pipelines = len(get_idle_pipelines_id(processes_table=PROCESSES_TABLE))
            if idle_pipelines < expected_warmed_up_pipelines:
                _ = spawn_managed_pipeline_process(processes_table=PROCESSES_TABLE)
        time.sleep(5)


def get_idle_pipelines_id(
    processes_table: Dict[str, ManagedInferencePipeline],
) -> List[str]:
    return [
        pipeline_id
        for pipeline_id, managed_pipeline in processes_table.items()
        if managed_pipeline.is_idle
    ]


def spawn_managed_pipeline_process(
    processes_table: Dict[str, ManagedInferencePipeline],
    mark_as_idle: bool = True,
) -> str:
    logger.info(
        f"Spawning new managed InferencePipeline process. Idle flag: {mark_as_idle}"
    )
    pipeline_id = str(uuid4())
    command_queue = Queue()
    responses_queue = Queue()
    inference_pipeline_manager = InferencePipelineManager.init(
        pipeline_id=pipeline_id,
        command_queue=command_queue,
        responses_queue=responses_queue,
    )
    inference_pipeline_manager.start()
    processes_table[pipeline_id] = ManagedInferencePipeline(
        pipeline_id=pipeline_id,
        pipeline_manager=inference_pipeline_manager,
        command_queue=command_queue,
        responses_queue=responses_queue,
        operation_lock=Lock(),
        is_idle=mark_as_idle,
    )
    processes_table[pipeline_id].ram_usage_queue.append(
        _get_process_memory_usage_mb(process=inference_pipeline_manager)
    )
    logger.info(f"Spawned new InferencePipeline process with id: {pipeline_id}")
    return pipeline_id


def _get_process_memory_usage_mb(process: Process) -> int:
    return psutil.Process(process.pid).memory_info().rss / (1024 * 1024)


def start(expected_warmed_up_pipelines: int = 0) -> None:
    signal.signal(
        signal.SIGINT, partial(execute_termination, processes_table=PROCESSES_TABLE)
    )
    signal.signal(
        signal.SIGTERM, partial(execute_termination, processes_table=PROCESSES_TABLE)
    )

    # check process health in daemon thread
    Thread(target=check_process_health, daemon=True).start()

    # keep expected number of processes ready for processing
    Thread(
        target=partial(
            ensure_idle_pipelines_warmed_up,
            expected_warmed_up_pipelines=expected_warmed_up_pipelines,
        ),
        daemon=True,
    ).start()

    with RoboflowTCPServer(
        server_address=(HOST, PORT),
        handler_class=partial(
            InferencePipelinesManagerHandler, processes_table=PROCESSES_TABLE
        ),
        socket_operations_timeout=SOCKET_TIMEOUT,
    ) as tcp_server:
        logger.info(
            f"Inference Pipeline Processes Manager is ready to accept connections at {(HOST, PORT)}"
        )
        tcp_server.serve_forever()


if __name__ == "__main__":
    start()
