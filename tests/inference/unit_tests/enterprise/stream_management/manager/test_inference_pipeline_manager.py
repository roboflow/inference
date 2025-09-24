"""
Unit tests in this module are realised using `InferencePipeline` mock - and within single process, submitting
command queues upfront, and then handling one-by-one in the same process.
"""

from multiprocessing import Queue
from unittest import mock
from unittest.mock import MagicMock

import pytest

from inference.core.exceptions import (
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
)
from inference.core.interfaces.camera.exceptions import StreamOperationNotAllowedError
from inference.enterprise.stream_management.manager import inference_pipeline_manager
from inference.enterprise.stream_management.manager.entities import (
    CommandType,
    ErrorType,
    OperationStatus,
)
from inference.enterprise.stream_management.manager.inference_pipeline_manager import (
    InferencePipelineManager,
)


@pytest.mark.timeout(30)
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init")
def test_inference_pipeline_manager_when_init_pipeline_operation_is_requested(
    pipeline_init_mock: MagicMock,
) -> None:
    # given
    pipeline_init_mock.return_value = MagicMock()
    command_queue, responses_queue = Queue(), Queue()
    manager = InferencePipelineManager(
        command_queue=command_queue, responses_queue=responses_queue
    )
    init_payload = assembly_valid_init_payload()

    # when
    command_queue.put(("1", init_payload))
    command_queue.put(("2", {"type": CommandType.TERMINATE}))

    manager.run()

    status_1 = responses_queue.get()
    status_2 = responses_queue.get()

    # then
    assert status_1 == (
        "1",
        {"status": OperationStatus.SUCCESS},
    ), "Initialisation operation must succeed"
    assert status_2 == (
        "2",
        {"status": OperationStatus.SUCCESS},
    ), "Termination operation must succeed"

    actual_video_source_properties = pipeline_init_mock.call_args[1][
        "video_source_properties"
    ]
    assert actual_video_source_properties == {
        "fps": 30,
        "frame_height": 1080,
        "frame_width": 1920,
    }


@pytest.mark.timeout(30)
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init")
def test_inference_pipeline_manager_when_init_pipeline_operation_is_requested_without_api_key(
    pipeline_init_mock: MagicMock,
) -> None:
    # given
    pipeline_init_mock.return_value = MagicMock()
    command_queue, responses_queue = Queue(), Queue()
    manager = InferencePipelineManager(
        command_queue=command_queue, responses_queue=responses_queue
    )
    init_payload = assembly_valid_init_payload()
    del init_payload["api_key"]

    # when
    command_queue.put(("1", init_payload))
    command_queue.put(("2", {"type": CommandType.TERMINATE}))

    manager.run()

    status_1 = responses_queue.get()
    status_2 = responses_queue.get()

    # then
    assert status_1 == (
        "1",
        {"status": OperationStatus.SUCCESS},
    ), "Initialisation operation must succeed"
    assert status_2 == (
        "2",
        {"status": OperationStatus.SUCCESS},
    ), "Termination operation must succeed"


@pytest.mark.timeout(30)
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init")
def test_inference_pipeline_manager_when_init_pipeline_operation_is_requested_but_invalid_payload_sent(
    pipeline_init_mock: MagicMock,
) -> None:
    # given
    pipeline_init_mock.return_value = MagicMock()
    command_queue, responses_queue = Queue(), Queue()
    manager = InferencePipelineManager(
        command_queue=command_queue, responses_queue=responses_queue
    )
    init_payload = assembly_valid_init_payload()
    del init_payload["model_configuration"]

    # when
    command_queue.put(("1", init_payload))
    command_queue.put(("2", {"type": CommandType.TERMINATE}))

    manager.run()

    status_1 = responses_queue.get()
    status_2 = responses_queue.get()

    # then
    assert (
        status_1[0] == "1"
    ), "First request should be reported in responses_queue at first"
    assert (
        status_1[1]["status"] == OperationStatus.FAILURE
    ), "Init operation should fail"
    assert (
        status_1[1]["error_type"] == ErrorType.INVALID_PAYLOAD
    ), "Invalid Payload error is expected"
    assert status_2 == (
        "2",
        {"status": OperationStatus.SUCCESS},
    ), "Termination of pipeline must happen"


@pytest.mark.timeout(30)
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init")
def test_inference_pipeline_manager_when_init_pipeline_operation_is_requested_but_roboflow_operation_not_authorised(
    pipeline_init_mock: MagicMock,
) -> None:
    # given
    pipeline_init_mock.side_effect = RoboflowAPINotAuthorizedError()
    command_queue, responses_queue = Queue(), Queue()
    manager = InferencePipelineManager(
        command_queue=command_queue, responses_queue=responses_queue
    )
    init_payload = assembly_valid_init_payload()

    # when
    command_queue.put(("1", init_payload))
    command_queue.put(("2", {"type": CommandType.TERMINATE}))

    manager.run()

    status_1 = responses_queue.get()
    status_2 = responses_queue.get()

    # then
    assert (
        status_1[0] == "1"
    ), "First request should be reported in responses_queue at first"
    assert (
        status_1[1]["status"] == OperationStatus.FAILURE
    ), "Init operation should fail"
    assert (
        status_1[1]["error_type"] == ErrorType.AUTHORISATION_ERROR
    ), "Authorisation error is expected"
    assert status_2 == (
        "2",
        {"status": OperationStatus.SUCCESS},
    ), "Termination of pipeline must happen"


@pytest.mark.timeout(30)
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init")
def test_inference_pipeline_manager_when_init_pipeline_operation_is_requested_but_model_not_found(
    pipeline_init_mock: MagicMock,
) -> None:
    # given
    pipeline_init_mock.side_effect = RoboflowAPINotNotFoundError()
    command_queue, responses_queue = Queue(), Queue()
    manager = InferencePipelineManager(
        command_queue=command_queue, responses_queue=responses_queue
    )
    init_payload = assembly_valid_init_payload()

    # when
    command_queue.put(("1", init_payload))
    command_queue.put(("2", {"type": CommandType.TERMINATE}))

    manager.run()

    status_1 = responses_queue.get()
    status_2 = responses_queue.get()

    # then
    assert (
        status_1[0] == "1"
    ), "First request should be reported in responses_queue at first"
    assert (
        status_1[1]["status"] == OperationStatus.FAILURE
    ), "Init operation should fail"
    assert (
        status_1[1]["error_type"] == ErrorType.NOT_FOUND
    ), "Not found error is expected"
    assert status_2 == (
        "2",
        {"status": OperationStatus.SUCCESS},
    ), "Termination of pipeline must happen"


@pytest.mark.timeout(30)
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init")
def test_inference_pipeline_manager_when_init_pipeline_operation_is_requested_but_unknown_error_appears(
    pipeline_init_mock: MagicMock,
) -> None:
    # given
    pipeline_init_mock.side_effect = Exception()
    command_queue, responses_queue = Queue(), Queue()
    manager = InferencePipelineManager(
        command_queue=command_queue, responses_queue=responses_queue
    )
    init_payload = assembly_valid_init_payload()

    # when
    command_queue.put(("1", init_payload))
    command_queue.put(("2", {"type": CommandType.TERMINATE}))

    manager.run()

    status_1 = responses_queue.get()
    status_2 = responses_queue.get()

    # then
    assert (
        status_1[0] == "1"
    ), "First request should be reported in responses_queue at first"
    assert (
        status_1[1]["status"] == OperationStatus.FAILURE
    ), "Init operation should fail"
    assert (
        status_1[1]["error_type"] == ErrorType.INTERNAL_ERROR
    ), "Internal error is expected"
    assert status_2 == (
        "2",
        {"status": OperationStatus.SUCCESS},
    ), "Termination of pipeline must happen"


@pytest.mark.timeout(30)
def test_inference_pipeline_manager_when_attempted_to_get_status_of_not_initialised_pipeline() -> (
    None
):
    # given
    command_queue, responses_queue = Queue(), Queue()
    manager = InferencePipelineManager(
        command_queue=command_queue, responses_queue=responses_queue
    )

    # when
    command_queue.put(("1", {"type": CommandType.STATUS}))
    command_queue.put(("2", {"type": CommandType.TERMINATE}))

    manager.run()

    status_1 = responses_queue.get()
    status_2 = responses_queue.get()

    # then
    assert (
        status_1[0] == "1"
    ), "First request should be reported in responses_queue at first"
    assert (
        status_1[1]["status"] == OperationStatus.FAILURE
    ), "Init operation should fail"
    assert (
        status_1[1]["error_type"] == ErrorType.OPERATION_ERROR
    ), "Operation error is expected"
    assert status_2 == (
        "2",
        {"status": OperationStatus.SUCCESS},
    ), "Termination of pipeline must happen"


@pytest.mark.timeout(30)
def test_inference_pipeline_manager_when_attempted_to_pause_of_not_initialised_pipeline() -> (
    None
):
    # given
    command_queue, responses_queue = Queue(), Queue()
    manager = InferencePipelineManager(
        command_queue=command_queue, responses_queue=responses_queue
    )

    # when
    command_queue.put(("1", {"type": CommandType.MUTE}))
    command_queue.put(("2", {"type": CommandType.TERMINATE}))

    manager.run()

    status_1 = responses_queue.get()
    status_2 = responses_queue.get()

    # then
    assert (
        status_1[0] == "1"
    ), "First request should be reported in responses_queue at first"
    assert (
        status_1[1]["status"] == OperationStatus.FAILURE
    ), "Init operation should fail"
    assert (
        status_1[1]["error_type"] == ErrorType.OPERATION_ERROR
    ), "Operation error is expected"
    assert status_2 == (
        "2",
        {"status": OperationStatus.SUCCESS},
    ), "Termination of pipeline must happen"


@pytest.mark.timeout(30)
def test_inference_pipeline_manager_when_attempted_to_resume_of_not_initialised_pipeline() -> (
    None
):
    # given
    command_queue, responses_queue = Queue(), Queue()
    manager = InferencePipelineManager(
        command_queue=command_queue, responses_queue=responses_queue
    )

    # when
    command_queue.put(("1", {"type": CommandType.RESUME}))
    command_queue.put(("2", {"type": CommandType.TERMINATE}))

    manager.run()

    status_1 = responses_queue.get()
    status_2 = responses_queue.get()

    # then
    assert (
        status_1[0] == "1"
    ), "First request should be reported in responses_queue at first"
    assert (
        status_1[1]["status"] == OperationStatus.FAILURE
    ), "Init operation should fail"
    assert (
        status_1[1]["error_type"] == ErrorType.OPERATION_ERROR
    ), "Operation error is expected"
    assert status_2 == (
        "2",
        {"status": OperationStatus.SUCCESS},
    ), "Termination of pipeline must happen"


@pytest.mark.timeout(30)
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init")
def test_inference_pipeline_manager_when_attempted_to_init_pause_resume_actions_successfully(
    pipeline_init_mock: MagicMock,
) -> None:
    # given
    pipeline_init_mock.return_value = MagicMock()
    command_queue, responses_queue = Queue(), Queue()
    manager = InferencePipelineManager(
        command_queue=command_queue, responses_queue=responses_queue
    )
    init_payload = assembly_valid_init_payload()

    # when
    command_queue.put(("1", init_payload))
    command_queue.put(("2", {"type": CommandType.MUTE}))
    command_queue.put(("3", {"type": CommandType.RESUME}))
    command_queue.put(("4", {"type": CommandType.TERMINATE}))

    manager.run()

    status_1 = responses_queue.get()
    status_2 = responses_queue.get()
    status_3 = responses_queue.get()
    status_4 = responses_queue.get()

    # then
    assert status_1 == (
        "1",
        {"status": OperationStatus.SUCCESS},
    ), "Initialisation operation must succeed"
    assert status_2 == (
        "2",
        {"status": OperationStatus.SUCCESS},
    ), "Pause of pipeline must happen"
    assert status_3 == (
        "3",
        {"status": OperationStatus.SUCCESS},
    ), "Resume of pipeline must happen"
    assert status_4 == (
        "4",
        {"status": OperationStatus.SUCCESS},
    ), "Termination of pipeline must happen"


@pytest.mark.timeout(30)
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init")
def test_inference_pipeline_manager_when_attempted_to_resume_running_sprint_causing_not_allowed_transition(
    pipeline_init_mock: MagicMock,
) -> None:
    # given
    pipeline_init_mock.return_value = MagicMock()
    pipeline_init_mock.return_value.resume_stream.side_effect = (
        StreamOperationNotAllowedError()
    )
    command_queue, responses_queue = Queue(), Queue()
    manager = InferencePipelineManager(
        command_queue=command_queue, responses_queue=responses_queue
    )
    init_payload = assembly_valid_init_payload()

    # when
    command_queue.put(("1", init_payload))
    command_queue.put(("2", {"type": CommandType.RESUME}))
    command_queue.put(("3", {"type": CommandType.TERMINATE}))

    manager.run()

    status_1 = responses_queue.get()
    status_2 = responses_queue.get()
    status_3 = responses_queue.get()

    # then
    assert status_1 == (
        "1",
        {"status": OperationStatus.SUCCESS},
    ), "Initialisation operation must succeed"
    assert status_2[0] == "2", "Second result must refer to request `2`"
    assert (
        status_2[1]["status"] is OperationStatus.FAILURE
    ), "Second request should fail, as we requested forbidden action"
    assert status_2[1]["error_type"] == ErrorType.OPERATION_ERROR
    assert status_3 == (
        "3",
        {"status": OperationStatus.SUCCESS},
    ), "Termination of pipeline must happen"


def assembly_valid_init_payload() -> dict:
    return {
        "type": CommandType.INIT,
        "sink_configuration": {
            "type": "udp_sink",
            "host": "127.0.0.1",
            "port": 6060,
        },
        "video_reference": "rtsp://128.0.0.1",
        "model_id": "some/1",
        "api_key": "my_key",
        "model_configuration": {"type": "object-detection"},
        "active_learning_enabled": True,
        "video_source_properties": {
            "fps": 30,
            "frame_width": 1920,
            "frame_height": 1080,
        },
    }
