"""
Unit tests in this module are realised using `InferencePipeline` mock - and within single process, submitting
command queues upfront, and then handling one-by-one in the same process.
"""

from datetime import datetime
from multiprocessing import Queue
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.core.exceptions import (
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
)
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.camera.exceptions import StreamOperationNotAllowedError
from inference.core.interfaces.camera.video_source import (
    BufferConsumptionStrategy,
    BufferFillingStrategy,
)
from inference.core.interfaces.stream.sinks import InMemoryBufferSink
from inference.core.interfaces.stream_manager.manager_app import (
    inference_pipeline_manager,
)
from inference.core.interfaces.stream_manager.manager_app.entities import (
    CommandType,
    ErrorType,
    InitialisePipelinePayload,
    OperationStatus,
    VideoConfiguration,
    WorkflowConfiguration,
)
from inference.core.interfaces.stream_manager.manager_app.inference_pipeline_manager import (
    InferencePipelineManager,
)


@pytest.mark.timeout(30)
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init_with_workflow")
def test_inference_pipeline_manager_when_init_pipeline_operation_is_requested(
    pipeline_init_mock: MagicMock,
) -> None:
    # given
    pipeline_init_mock.return_value = MagicMock()
    command_queue, responses_queue = Queue(), Queue()
    manager = InferencePipelineManager(
        pipeline_id="my_pipeline",
        command_queue=command_queue,
        responses_queue=responses_queue,
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
    actual_video_reference = pipeline_init_mock.call_args[1]["video_reference"]
    assert actual_video_reference == "rtsp://128.0.0.1"


@pytest.mark.timeout(30)
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init_with_workflow")
def test_inference_pipeline_manager_when_init_pipeline_operation_is_requested_but_invalid_payload_sent(
    pipeline_init_mock: MagicMock,
) -> None:
    # given
    pipeline_init_mock.return_value = MagicMock()
    command_queue, responses_queue = Queue(), Queue()
    manager = InferencePipelineManager(
        pipeline_id="my_pipeline",
        command_queue=command_queue,
        responses_queue=responses_queue,
    )
    init_payload = assembly_valid_init_payload()
    del init_payload["video_configuration"]["video_reference"]

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
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init_with_workflow")
def test_inference_pipeline_manager_when_init_pipeline_operation_is_requested_but_roboflow_operation_not_authorised(
    pipeline_init_mock: MagicMock,
) -> None:
    # given
    pipeline_init_mock.side_effect = RoboflowAPINotAuthorizedError()
    command_queue, responses_queue = Queue(), Queue()
    manager = InferencePipelineManager(
        pipeline_id="my_pipeline",
        command_queue=command_queue,
        responses_queue=responses_queue,
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
@mock.patch.object(inference_pipeline_manager.InMemoryBufferSink, "init")
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init_with_workflow")
def test_inference_pipeline_manager_consumption(
    pipeline_init_mock: MagicMock,
    buffer_init: MagicMock,
) -> None:
    # given
    pipeline_init_mock.return_value = MagicMock()
    filled_buffer = InMemoryBufferSink(queue_size=10)
    filled_buffer.on_prediction(
        predictions=[None, {"some": "value"}],
        video_frame=[
            None,
            VideoFrame(
                image=np.zeros((3, 3)),
                frame_id=0,
                frame_timestamp=datetime.now(),
                source_id=1,
            ),
        ],
    )
    filled_buffer.on_prediction(
        predictions=[None, {"other": "value"}],
        video_frame=[
            None,
            VideoFrame(
                image=np.zeros((3, 3)),
                frame_id=1,
                frame_timestamp=datetime.now(),
                source_id=1,
            ),
        ],
    )
    buffer_init.return_value = filled_buffer
    command_queue, responses_queue = Queue(), Queue()
    manager = InferencePipelineManager(
        pipeline_id="my_pipeline",
        command_queue=command_queue,
        responses_queue=responses_queue,
    )
    init_payload = assembly_valid_init_payload()

    # when
    command_queue.put(("1", init_payload))
    command_queue.put(("2", {"type": CommandType.CONSUME_RESULT}))
    command_queue.put(("3", {"type": CommandType.CONSUME_RESULT}))
    command_queue.put(("4", {"type": CommandType.CONSUME_RESULT}))
    command_queue.put(("5", {"type": CommandType.TERMINATE}))

    manager.run()

    status_1 = responses_queue.get()
    status_2 = responses_queue.get()
    status_3 = responses_queue.get()
    status_4 = responses_queue.get()
    status_5 = responses_queue.get()

    # then
    assert (
        status_1[0] == "1"
    ), "First request should be reported in responses_queue at first"
    assert (
        status_1[1]["status"] == OperationStatus.SUCCESS
    ), "Init operation should succeed"
    assert status_2[0] == "2", "2nd request should be reported in responses_queue 2nd"
    assert status_2[1]["status"] == OperationStatus.SUCCESS, "Operation should succeed"
    assert status_2[1]["outputs"] == [
        None,
        {"some": "value"},
    ], "Operation should yield first buffer result"
    assert status_3[0] == "3", "3rd request should be reported in responses_queue 3rd"
    assert status_3[1]["status"] == OperationStatus.SUCCESS, "Operation should succeed"
    assert status_3[1]["outputs"] == [
        None,
        {"other": "value"},
    ], "Operation should yield second buffer result"
    assert status_4[0] == "4", "4th request should be reported in responses_queue 4th"
    assert status_4[1]["status"] == OperationStatus.SUCCESS, "Operation should succeed"
    assert status_4[1]["outputs"] == [], "Operation should yield empty result"
    assert status_5[0] == "5", "5th request should be reported in responses_queue 5th"
    assert status_5[1]["status"] == OperationStatus.SUCCESS, "Operation should succeed"


@pytest.mark.timeout(30)
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init_with_workflow")
def test_inference_pipeline_manager_when_init_pipeline_operation_is_requested_but_model_not_found(
    pipeline_init_mock: MagicMock,
) -> None:
    # given
    pipeline_init_mock.side_effect = RoboflowAPINotNotFoundError()
    command_queue, responses_queue = Queue(), Queue()
    manager = InferencePipelineManager(
        pipeline_id="my_pipeline",
        command_queue=command_queue,
        responses_queue=responses_queue,
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
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init_with_workflow")
def test_inference_pipeline_manager_when_init_pipeline_operation_is_requested_but_unknown_error_appears(
    pipeline_init_mock: MagicMock,
) -> None:
    # given
    pipeline_init_mock.side_effect = Exception()
    command_queue, responses_queue = Queue(), Queue()
    manager = InferencePipelineManager(
        pipeline_id="my_pipeline",
        command_queue=command_queue,
        responses_queue=responses_queue,
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
        pipeline_id="my_pipeline",
        command_queue=command_queue,
        responses_queue=responses_queue,
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
        pipeline_id="my_pipeline",
        command_queue=command_queue,
        responses_queue=responses_queue,
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
        pipeline_id="my_pipeline",
        command_queue=command_queue,
        responses_queue=responses_queue,
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
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init_with_workflow")
def test_inference_pipeline_manager_when_attempted_to_init_pause_resume_actions_successfully(
    pipeline_init_mock: MagicMock,
) -> None:
    # given
    pipeline_init_mock.return_value = MagicMock()
    command_queue, responses_queue = Queue(), Queue()
    manager = InferencePipelineManager(
        pipeline_id="my_pipeline",
        command_queue=command_queue,
        responses_queue=responses_queue,
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
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init_with_workflow")
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
        pipeline_id="my_pipeline",
        command_queue=command_queue,
        responses_queue=responses_queue,
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
    specification = {
        "version": "1.0",
        "inputs": [
            {
                "type": "InferenceImage",
                "name": "image",
                "video_metadata_input_name": "test",
            },
        ],
        "steps": [
            {
                "type": "ObjectDetectionModel",
                "name": "people_detector",
                "image": "$inputs.image",
                "model_id": "yolov8n-640",
                "confidence": 0.5,
                "iou_threshold": 0.7,
                "class_filter": ["person"],
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.people_detector.predictions",
            }
        ],
    }
    valid_init_payload = InitialisePipelinePayload(
        video_configuration=VideoConfiguration(
            type="VideoConfiguration",
            video_reference="rtsp://128.0.0.1",
            source_buffer_filling_strategy=BufferFillingStrategy.DROP_OLDEST,
            source_buffer_consumption_strategy=BufferConsumptionStrategy.EAGER,
        ),
        processing_configuration=WorkflowConfiguration(
            type="WorkflowConfiguration",
            workflow_specification=specification,
        ),
        api_key="<MY-API-KEY>",
    ).dict()
    valid_init_payload["type"] = CommandType.INIT
    return valid_init_payload
