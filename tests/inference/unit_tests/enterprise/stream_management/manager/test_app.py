"""
Unit tests in this module are realised using `InferencePipelineManager` mock - and within single process, submitting
command queues upfront, and then handling one-by-one in the same process.
"""

import json
from multiprocessing import Process, Queue
from unittest import mock
from unittest.mock import MagicMock

import pytest

from inference.enterprise.stream_management.manager import app
from inference.enterprise.stream_management.manager.app import (
    InferencePipelinesManagerHandler,
    execute_termination,
    get_response_ignoring_thrash,
    handle_command,
    join_inference_pipeline,
)
from inference.enterprise.stream_management.manager.entities import (
    CommandType,
    ErrorType,
    OperationStatus,
)
from inference.enterprise.stream_management.manager.inference_pipeline_manager import (
    InferencePipelineManager,
)


def test_get_response_ignoring_thrash_when_nothing_is_to_ignore() -> None:
    # given
    responses_queue = Queue()
    responses_queue.put(("my_request", {"some": "data"}))

    # when
    result = get_response_ignoring_thrash(
        responses_queue=responses_queue, matching_request_id="my_request"
    )

    # then
    assert result == {"some": "data"}


def test_get_response_ignoring_thrash_when_trash_message_is_to_be_ignored() -> None:
    # given
    responses_queue = Queue()
    responses_queue.put(("thrash", {"other": "data"}))
    responses_queue.put(("my_request", {"some": "data"}))

    # when
    result = get_response_ignoring_thrash(
        responses_queue=responses_queue, matching_request_id="my_request"
    )

    # then
    assert result == {"some": "data"}


def test_handle_command_when_pipeline_id_is_not_registered_in_the_table() -> None:
    # when
    result = handle_command(
        processes_table={},
        request_id="my_request",
        pipeline_id="unknown",
        command={"type": CommandType.RESUME},
    )

    # then
    assert result == {
        "status": OperationStatus.FAILURE,
        "error_type": ErrorType.NOT_FOUND,
    }


class DummyPipelineManager(Process):
    def __init__(self, input_queue: Queue, output_queue: Queue):
        super().__init__()
        self._input_queue = input_queue
        self._output_queue = output_queue

    def run(self) -> None:
        input_data = self._input_queue.get()
        self._output_queue.put(input_data)


@pytest.mark.timeout(30)
@pytest.mark.slow
def test_handle_command_when_pipeline_id_is_registered_in_the_table() -> None:
    # given
    input_queue, output_queue = Queue(), Queue()
    pipeline_manager = DummyPipelineManager(
        input_queue=input_queue, output_queue=output_queue
    )
    pipeline_manager.start()
    processes_table = {"my_pipeline": (pipeline_manager, input_queue, output_queue)}

    try:
        # when
        result = handle_command(
            processes_table=processes_table,
            request_id="my_request",
            pipeline_id="my_pipeline",
            command={"type": CommandType.RESUME},
        )

        # then
        assert result == {"type": CommandType.RESUME}
    finally:
        pipeline_manager.join(timeout=1.0)


@pytest.mark.timeout(30)
@pytest.mark.slow
def test_join_inference_pipeline() -> None:
    # given
    input_queue, output_queue = Queue(), Queue()
    pipeline_manager = DummyPipelineManager(
        input_queue=input_queue, output_queue=output_queue
    )
    pipeline_manager.start()
    processes_table = {"my_pipeline": (pipeline_manager, input_queue, output_queue)}

    # when
    input_queue.put(None)
    _ = output_queue.get()
    join_inference_pipeline(processes_table=processes_table, pipeline_id="my_pipeline")

    # then
    assert "my_pipeline" not in processes_table
    assert pipeline_manager.is_alive() is False


@pytest.mark.timeout(30)
@pytest.mark.slow
@mock.patch.object(app.sys, "exit")
def test_execute_termination(exit_mock: MagicMock) -> None:
    # given
    command_queue, responses_queue = Queue(), Queue()
    inference_pipeline_manager = InferencePipelineManager(
        command_queue=command_queue,
        responses_queue=responses_queue,
    )
    inference_pipeline_manager.start()
    processes_table = {
        "my_pipeline": (inference_pipeline_manager, command_queue, responses_queue)
    }

    # when
    # initial command makes sure the error handling is set and there is no time-hazard with execute_termination(...)
    command_queue.put(("unknown", {}))
    _ = responses_queue.get()
    execute_termination(9, MagicMock(), processes_table=processes_table)

    # then
    exit_mock.assert_called_once_with(0)


class DummySocket:
    def __init__(self):
        self._buffer = b""
        self._sent = b""

    def get_data_that_was_sent(self) -> bytes:
        return self._sent

    def fill(self, data: bytes) -> None:
        self._buffer = data

    def recv(self, __bufsize: int) -> bytes:
        chunk = self._buffer[:__bufsize]
        self._buffer = self._buffer[__bufsize:]
        return chunk

    def sendall(self, __data: bytes) -> None:
        self._sent += __data


@pytest.mark.timeout(30)
def test_pipeline_manager_handler_when_wrong_input_format_is_sent() -> None:
    # given
    socket = DummySocket()
    payload = "FOR SURE NOT A JSON".encode("utf-8")
    header = len(payload).to_bytes(length=4, byteorder="big")
    socket.fill(header + payload)

    # when
    _ = InferencePipelinesManagerHandler(
        request=socket,
        client_address=MagicMock(),
        server=MagicMock(),
        processes_table={},
    )
    response = json.loads(socket.get_data_that_was_sent()[4:].decode("utf-8"))

    # then
    assert (
        response["pipeline_id"] is None
    ), "Pipeline ID cannot be associated to this request"
    assert response["response"]["status"] == "failure", "Operation should failed"
    assert (
        response["response"]["error_type"] == "invalid_payload"
    ), "Wrong payload should be denoted as error cause"


@pytest.mark.timeout(30)
def test_pipeline_manager_handler_when_malformed_input_is_sent() -> None:
    # given
    socket = DummySocket()
    payload = json.dumps({"invalid": "data"}).encode("utf-8")
    header = len(payload).to_bytes(length=4, byteorder="big")
    socket.fill(header + payload)

    # when
    _ = InferencePipelinesManagerHandler(
        request=socket,
        client_address=MagicMock(),
        server=MagicMock(),
        processes_table={},
    )
    response = json.loads(socket.get_data_that_was_sent()[4:].decode("utf-8"))

    # then
    assert (
        response["pipeline_id"] is None
    ), "Pipeline ID cannot be associated to this request"
    assert response["response"]["status"] == "failure", "Operation should failed"
    assert (
        response["response"]["error_type"] == "invalid_payload"
    ), "Wrong payload should be denoted as error cause"


@pytest.mark.timeout(30)
def test_pipeline_manager_handler_when_unknown_command_is_sent() -> None:
    # given
    socket = DummySocket()
    payload = json.dumps({"type": "unknown"}).encode("utf-8")
    header = len(payload).to_bytes(length=4, byteorder="big")
    socket.fill(header + payload)

    # when
    _ = InferencePipelinesManagerHandler(
        request=socket,
        client_address=MagicMock(),
        server=MagicMock(),
        processes_table={},
    )
    response = json.loads(socket.get_data_that_was_sent()[4:].decode("utf-8"))

    # then
    assert (
        response["pipeline_id"] is None
    ), "Pipeline ID cannot be associated to this request"
    assert response["response"]["status"] == "failure", "Operation should failed"
    assert (
        response["response"]["error_type"] == "invalid_payload"
    ), "Wrong payload should be denoted as error cause"


@pytest.mark.timeout(30)
def test_pipeline_manager_handler_when_command_requested_for_unknown_pipeline() -> None:
    # given
    socket = DummySocket()
    payload = json.dumps({"type": "terminate", "pipeline_id": "unknown"}).encode(
        "utf-8"
    )
    header = len(payload).to_bytes(length=4, byteorder="big")
    socket.fill(header + payload)

    # when
    _ = InferencePipelinesManagerHandler(
        request=socket,
        client_address=MagicMock(),
        server=MagicMock(),
        processes_table={},
    )
    response = json.loads(socket.get_data_that_was_sent()[4:].decode("utf-8"))

    # then
    assert (
        response["pipeline_id"] == "unknown"
    ), "Pipeline ID must be assigned to request"
    assert response["response"]["status"] == "failure", "Operation should failed"
    assert (
        response["response"]["error_type"] == "not_found"
    ), "Pipeline not found should be denoted as error cause"


@pytest.mark.timeout(30)
@pytest.mark.slow
def test_pipeline_manager_handler_when_pipeline_initialisation_triggered_with_malformed_payload() -> (
    None
):
    # given
    socket = DummySocket()
    payload = json.dumps({"type": "init"}).encode("utf-8")
    header = len(payload).to_bytes(length=4, byteorder="big")
    socket.fill(header + payload)
    processes_table = {}

    try:
        # when
        _ = InferencePipelinesManagerHandler(
            request=socket,
            client_address=MagicMock(),
            server=MagicMock(),
            processes_table=processes_table,
        )
        response = json.loads(socket.get_data_that_was_sent()[4:].decode("utf-8"))

        # then
        assert (
            len(processes_table) == 1
        ), "Pipeline table should be filled with manager process"
        assert (
            type(response["pipeline_id"]) is str
        ), "Pipeline ID must be set to random string"
        assert response["response"]["status"] == "failure", "Operation should failed"
        assert (
            response["response"]["error_type"] == "invalid_payload"
        ), "Pipeline could not be initialised due to invalid payload"
    finally:
        process = processes_table[list(processes_table.keys())[0]]
        process[0].terminate()


@pytest.mark.timeout(30)
@pytest.mark.slow
def test_pipeline_manager_handler_when_termination_requested_after_failed_initialisation() -> (
    None
):
    # given
    socket = DummySocket()
    payload = json.dumps({"type": "init"}).encode("utf-8")
    header = len(payload).to_bytes(length=4, byteorder="big")
    socket.fill(header + payload)
    processes_table = {}

    try:
        # when
        _ = InferencePipelinesManagerHandler(
            request=socket,
            client_address=MagicMock(),
            server=MagicMock(),
            processes_table=processes_table,
        )
        init_response = json.loads(socket.get_data_that_was_sent()[4:].decode("utf-8"))
        socket = DummySocket()
        payload = json.dumps(
            {"type": "terminate", "pipeline_id": init_response["pipeline_id"]}
        ).encode("utf-8")
        header = len(payload).to_bytes(length=4, byteorder="big")
        socket.fill(header + payload)
        _ = InferencePipelinesManagerHandler(
            request=socket,
            client_address=MagicMock(),
            server=MagicMock(),
            processes_table=processes_table,
        )
        terminate_response = json.loads(
            socket.get_data_that_was_sent()[4:].decode("utf-8")
        )

        # then
        assert (
            len(processes_table) == 0
        ), "Pipeline should be removed from table after termination"
        assert (
            type(init_response["pipeline_id"]) is str
        ), "Pipeline ID must be set to random string"
        assert (
            init_response["response"]["status"] == "failure"
        ), "Operation should failed"
        assert (
            init_response["response"]["error_type"] == "invalid_payload"
        ), "Pipeline could not be initialised due to invalid payload"
        assert (
            terminate_response["pipeline_id"] == init_response["pipeline_id"]
        ), "Terminate request must refer the same pipeline that was created"
        assert (
            terminate_response["response"]["status"] == "success"
        ), "Termination operation should succeed"
    finally:
        if len(processes_table) > 0:
            process = processes_table[list(processes_table.keys())]
            process[0].terminate()


@pytest.mark.timeout(30)
@pytest.mark.slow
def test_pipeline_manager_handler_when_list_of_pipelines_requested_after_unsuccessful_initialisation() -> (
    None
):
    # given
    socket = DummySocket()
    payload = json.dumps({"type": "init"}).encode("utf-8")
    header = len(payload).to_bytes(length=4, byteorder="big")
    socket.fill(header + payload)
    processes_table = {}

    try:
        # when
        _ = InferencePipelinesManagerHandler(
            request=socket,
            client_address=MagicMock(),
            server=MagicMock(),
            processes_table=processes_table,
        )
        init_response = json.loads(socket.get_data_that_was_sent()[4:].decode("utf-8"))
        socket = DummySocket()
        payload = json.dumps({"type": "list_pipelines"}).encode("utf-8")
        header = len(payload).to_bytes(length=4, byteorder="big")
        socket.fill(header + payload)
        _ = InferencePipelinesManagerHandler(
            request=socket,
            client_address=MagicMock(),
            server=MagicMock(),
            processes_table=processes_table,
        )
        listing_response = json.loads(
            socket.get_data_that_was_sent()[4:].decode("utf-8")
        )

        # then
        assert (
            len(processes_table) == 1
        ), "Pipeline table should be filled with manager process"
        assert (
            type(init_response["pipeline_id"]) is str
        ), "Pipeline ID must be set to random string"
        assert (
            init_response["response"]["status"] == "failure"
        ), "Operation should failed"
        assert (
            init_response["response"]["error_type"] == "invalid_payload"
        ), "Pipeline could not be initialised due to invalid payload"
        assert (
            listing_response["response"]["status"] == "success"
        ), "Listing operation should succeed"
        assert listing_response["response"]["pipelines"] == [
            init_response["pipeline_id"]
        ]
    finally:
        process = processes_table[list(processes_table.keys())[0]]
        process[0].terminate()
