from multiprocessing import Queue, Process
from unittest import mock
from unittest.mock import MagicMock

import pytest

from inference.enterprise.stream_management.manager.app import (
    get_response_ignoring_thrash,
    handle_command,
    join_inference_pipeline,
    execute_termination,
)
from inference.enterprise.stream_management.manager import app
from inference.enterprise.stream_management.manager.entities import (
    CommandType,
    OperationStatus,
    ErrorType,
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
    execute_termination(9, MagicMock(), processes_table=processes_table)

    # then
    exit_mock.assert_called_once_with(0)
