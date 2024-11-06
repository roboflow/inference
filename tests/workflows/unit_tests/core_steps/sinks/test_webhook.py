import time
from unittest import mock
from unittest.mock import MagicMock

from requests import Response

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    StringToUpperCase,
)
from inference.core.workflows.core_steps.sinks.webhook import v1
from inference.core.workflows.core_steps.sinks.webhook.v1 import (
    BlockManifest,
    WebhookSinkBlockV1,
    execute_operations_on_parameters,
    execute_request,
)


def test_manifest_parsing_when_the_input_is_valid() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/webhook_sink@v1",
        "name": "multipart_image_sink_put",
        "url": "http://127.0.0.1:9999/data-sink/multi-part-data",
        "method": "PUT",
        "multi_part_encoded_files": {
            "image": "$inputs.image",
        },
        "multi_part_encoded_files_operations": {
            "image": [{"type": "ConvertImageToJPEG"}],
        },
        "form_data": {
            "form_field": "$inputs.query_parameter",
        },
        "fire_and_forget": True,
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result == BlockManifest(
        type="roboflow_core/webhook_sink@v1",
        name="multipart_image_sink_put",
        url="http://127.0.0.1:9999/data-sink/multi-part-data",
        method="PUT",
        multi_part_encoded_files={
            "image": "$inputs.image",
        },
        multi_part_encoded_files_operations={
            "image": [{"type": "ConvertImageToJPEG"}],
        },
        form_data={
            "form_field": "$inputs.query_parameter",
        },
        fire_and_forget=True,
    )


@mock.patch.dict(v1.METHOD_TO_HANDLER, {"POST": MagicMock()}, clear=True)
def test_execute_request_when_request_is_successful() -> None:
    response = Response()
    response.status_code = 200
    v1.METHOD_TO_HANDLER["POST"].return_value = response

    # when
    result = execute_request(
        url="https://some.com",
        method="POST",
        query_parameters={"a": "b"},
        headers={"c": "d"},
        json_payload={"e": "f"},
        form_data={"field": "value"},
        multi_part_encoded_files={"file": b"data"},
        timeout=3,
    )

    # then
    assert result == (False, "Notification sent successfully")
    v1.METHOD_TO_HANDLER["POST"].assert_called_once_with(
        "https://some.com",
        params={"a": "b"},
        headers={"c": "d"},
        json={"e": "f"},
        files={"file": b"data"},
        data={"field": "value"},
        timeout=3,
    )


@mock.patch.dict(v1.METHOD_TO_HANDLER, {"POST": MagicMock()}, clear=True)
def test_execute_request_when_request_fails() -> None:
    response = Response()
    response.status_code = 500
    v1.METHOD_TO_HANDLER["POST"].return_value = response

    # when
    result = execute_request(
        url="https://some.com",
        method="POST",
        query_parameters={"a": "b"},
        headers={"c": "d"},
        json_payload={"e": "f"},
        form_data={"field": "value"},
        multi_part_encoded_files={"file": b"data"},
        timeout=3,
    )

    # then
    assert result[0] is True
    v1.METHOD_TO_HANDLER["POST"].assert_called_once_with(
        "https://some.com",
        params={"a": "b"},
        headers={"c": "d"},
        json={"e": "f"},
        files={"file": b"data"},
        data={"field": "value"},
        timeout=3,
    )


def test_execute_operations_on_parameters() -> None:
    # given
    parameters = {"a": "some", "b": "other"}
    operations = {"a": [StringToUpperCase(type="StringToUpperCase")]}

    # when
    result = execute_operations_on_parameters(
        parameters=parameters,
        operations=operations,
    )

    # then
    assert result == {
        "a": "SOME",
        "b": "other",
    }


def test_cooldown_in_webhook_notification_block() -> None:
    # given
    thread_pool_executor = MagicMock()
    block = WebhookSinkBlockV1(
        background_tasks=None,
        thread_pool_executor=thread_pool_executor,
    )

    # when
    results = []
    for _ in range(2):
        result = block.run(
            url="http://some.com",
            method="POST",
            query_parameters={"a": "b"},
            headers={"c": "d"},
            json_payload={"e": "f"},
            json_payload_operations={},
            form_data={"field": "value"},
            form_data_operations={},
            multi_part_encoded_files={"file": b"data"},
            multi_part_encoded_files_operations={},
            request_timeout=3,
            fire_and_forget=True,
            disable_sink=False,
            cooldown_seconds=100,
        )
        results.append(result)

    # then
    assert results[0] == {
        "error_status": False,
        "throttling_status": False,
        "message": "Notification sent in the background task",
    }
    assert results[1] == {
        "error_status": False,
        "throttling_status": True,
        "message": "Sink cooldown applies",
    }


def test_disabling_cooldown_in_webhook_notification_block() -> None:
    # given
    thread_pool_executor = MagicMock()
    block = WebhookSinkBlockV1(
        background_tasks=None,
        thread_pool_executor=thread_pool_executor,
    )

    # when
    results = []
    for _ in range(2):
        result = block.run(
            url="http://some.com",
            method="POST",
            query_parameters={"a": "b"},
            headers={"c": "d"},
            json_payload={"e": "f"},
            json_payload_operations={},
            form_data={"field": "value"},
            form_data_operations={},
            multi_part_encoded_files={"file": b"data"},
            multi_part_encoded_files_operations={},
            request_timeout=3,
            fire_and_forget=True,
            disable_sink=False,
            cooldown_seconds=0,
        )
        results.append(result)

    # then
    assert results[0] == {
        "error_status": False,
        "throttling_status": False,
        "message": "Notification sent in the background task",
    }
    assert results[1] == {
        "error_status": False,
        "throttling_status": False,
        "message": "Notification sent in the background task",
    }


def test_cooldown_recovery_in_webhook_notification_block() -> None:
    # given
    thread_pool_executor = MagicMock()
    block = WebhookSinkBlockV1(
        background_tasks=None,
        thread_pool_executor=thread_pool_executor,
    )

    # when
    results = []
    for _ in range(2):
        result = block.run(
            url="http://some.com",
            method="POST",
            query_parameters={"a": "b"},
            headers={"c": "d"},
            json_payload={"e": "f"},
            json_payload_operations={},
            form_data={"field": "value"},
            form_data_operations={},
            multi_part_encoded_files={"file": b"data"},
            multi_part_encoded_files_operations={},
            request_timeout=3,
            fire_and_forget=True,
            disable_sink=False,
            cooldown_seconds=1,
        )
        results.append(result)
        time.sleep(1.5)

    # then
    assert results[0] == {
        "error_status": False,
        "throttling_status": False,
        "message": "Notification sent in the background task",
    }
    assert results[1] == {
        "error_status": False,
        "throttling_status": False,
        "message": "Notification sent in the background task",
    }


@mock.patch.object(v1, "execute_request")
def test_sending_webhook_notification_synchronously(
    execute_request_mock: MagicMock,
) -> None:
    # given
    execute_request_mock.return_value = (False, "ok")
    block = WebhookSinkBlockV1(
        background_tasks=None,
        thread_pool_executor=None,
    )

    # when
    result = block.run(
        url="https://some.com",
        method="POST",
        query_parameters={"a": "b"},
        headers={"c": "d"},
        json_payload={"e": "f"},
        json_payload_operations={"e": [StringToUpperCase(type="StringToUpperCase")]},
        form_data={"field": "value"},
        form_data_operations={},
        multi_part_encoded_files={"file": b"data"},
        multi_part_encoded_files_operations={},
        request_timeout=3,
        fire_and_forget=False,
        disable_sink=False,
        cooldown_seconds=1,
    )

    # then
    assert result == {
        "error_status": False,
        "throttling_status": False,
        "message": "ok",
    }
    execute_request_mock.assert_called_once_with(
        url="https://some.com",
        method="POST",
        query_parameters={"a": "b"},
        headers={"c": "d"},
        json_payload={"e": "F"},
        multi_part_encoded_files={"file": b"data"},
        form_data={"field": "value"},
        timeout=3,
    )


def test_disabling_webhook_notification() -> None:
    # given
    block = WebhookSinkBlockV1(
        background_tasks=None,
        thread_pool_executor=None,
    )

    # when
    result = block.run(
        url="https://some.com",
        method="POST",
        query_parameters={"a": "b"},
        headers={"c": "d"},
        json_payload={"e": "f"},
        json_payload_operations={"e": [StringToUpperCase(type="StringToUpperCase")]},
        form_data={"field": "value"},
        form_data_operations={},
        multi_part_encoded_files={"file": b"data"},
        multi_part_encoded_files_operations={},
        request_timeout=3,
        fire_and_forget=False,
        disable_sink=True,
        cooldown_seconds=1,
    )

    # then
    assert result == {
        "error_status": False,
        "throttling_status": False,
        "message": "Sink was disabled by parameter `disable_sink`",
    }


def test_sending_webhook_notification_asynchronously_in_background_tasks() -> None:
    # given
    background_tasks = MagicMock()
    block = WebhookSinkBlockV1(
        background_tasks=background_tasks,
        thread_pool_executor=None,
    )

    # when
    result = block.run(
        url="https://some.com",
        method="POST",
        query_parameters={"a": "b"},
        headers={"c": "d"},
        json_payload={"e": "f"},
        json_payload_operations={"e": [StringToUpperCase(type="StringToUpperCase")]},
        form_data={"field": "value"},
        form_data_operations={},
        multi_part_encoded_files={"file": b"data"},
        multi_part_encoded_files_operations={},
        request_timeout=3,
        fire_and_forget=True,
        disable_sink=False,
        cooldown_seconds=1,
    )

    # then
    assert result == {
        "error_status": False,
        "throttling_status": False,
        "message": "Notification sent in the background task",
    }
    background_tasks.add_task.assert_called_once()


def test_sending_webhook_notification_asynchronously_in_thread_pool_executor() -> None:
    # given
    thread_pool_executor = MagicMock()
    block = WebhookSinkBlockV1(
        background_tasks=None,
        thread_pool_executor=thread_pool_executor,
    )

    # when
    result = block.run(
        url="https://some.com",
        method="POST",
        query_parameters={"a": "b"},
        headers={"c": "d"},
        json_payload={"e": "f"},
        json_payload_operations={"e": [StringToUpperCase(type="StringToUpperCase")]},
        form_data={"field": "value"},
        form_data_operations={},
        multi_part_encoded_files={"file": b"data"},
        multi_part_encoded_files_operations={},
        request_timeout=3,
        fire_and_forget=True,
        disable_sink=False,
        cooldown_seconds=1,
    )

    # then
    assert result == {
        "error_status": False,
        "throttling_status": False,
        "message": "Notification sent in the background task",
    }
    thread_pool_executor.submit.assert_called_once()
