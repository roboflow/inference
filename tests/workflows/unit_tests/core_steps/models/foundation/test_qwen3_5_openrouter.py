import time
from typing import Any
from unittest.mock import MagicMock

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.foundation.qwen3_5_openrouter.v1 import (
    BlockManifest,
    execute_qwen_request,
)


@pytest.mark.parametrize("value", [None, 1, "a", True])
def test_qwen3_5_openrouter_step_validation_when_image_is_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "roboflow_core/qwen3_5_openrouter@v1",
        "name": "step_1",
        "images": value,
        "prompt": "$inputs.prompt",
        "api_key": "$inputs.open_router_api_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_qwen3_5_openrouter_step_validation_when_prompt_is_given_directly() -> None:
    # given
    specification = {
        "type": "roboflow_core/qwen3_5_openrouter@v1",
        "name": "step_1",
        "images": "$inputs.image",
        "prompt": "This is my prompt",
        "api_key": "$inputs.open_router_api_key",
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result == BlockManifest(
        type="roboflow_core/qwen3_5_openrouter@v1",
        name="step_1",
        images="$inputs.image",
        prompt="This is my prompt",
        api_key="$inputs.open_router_api_key",
    )


@pytest.mark.parametrize("value", [None, []])
def test_qwen3_5_openrouter_step_validation_when_prompt_is_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "roboflow_core/qwen3_5_openrouter@v1",
        "name": "step_1",
        "images": "$inputs.image",
        "prompt": value,
        "api_key": "$inputs.open_router_api_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize(
    "value",
    [
        "$inputs.model",
        "Qwen 3.5 9B - OpenRouter",
        "Qwen 3.5 27B - OpenRouter",
        "Qwen 3.5 122B A10B - OpenRouter",
        "Qwen 3.5 397B A17B - OpenRouter",
        "Qwen 3.5 Flash 02-23 - OpenRouter",
        "Qwen 3.5 Plus 20260420 - OpenRouter",
    ],
)
def test_qwen3_5_openrouter_step_validation_when_model_type_valid(
    value: str,
) -> None:
    # given
    specification = {
        "type": "roboflow_core/qwen3_5_openrouter@v1",
        "name": "step_1",
        "images": "$inputs.image",
        "prompt": "This is my prompt",
        "api_key": "$inputs.open_router_api_key",
        "model_version": value,
    }

    # when
    result = BlockManifest.model_validate(specification)

    assert result == BlockManifest(
        type="roboflow_core/qwen3_5_openrouter@v1",
        name="step_1",
        images="$inputs.image",
        prompt="This is my prompt",
        api_key="$inputs.open_router_api_key",
        model_version=value,
    )


@pytest.mark.parametrize("value", ["some", None])
def test_qwen3_5_openrouter_step_validation_when_model_type_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "roboflow_core/qwen3_5_openrouter@v1",
        "name": "step_1",
        "images": "$inputs.image",
        "prompt": "This is my prompt",
        "api_key": "$inputs.open_router_api_key",
        "model_version": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_qwen3_5_openrouter_step_validation_when_api_key_not_given() -> None:
    # given
    specification = {
        "type": "roboflow_core/qwen3_5_openrouter@v1",
        "name": "step_1",
        "images": "$inputs.image",
        "prompt": "This is my prompt",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_qwen3_5_openrouter_step_validation_when_output_structure_invalid() -> None:
    # given
    specification = {
        "type": "roboflow_core/qwen3_5_openrouter@v1",
        "name": "step_1",
        "images": "$inputs.image",
        "prompt": "This is my prompt",
        "api_key": "$inputs.open_router_api_key",
        "output_structure": "INVALID",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize("value", [-0.5, 3.0])
def test_qwen3_5_openrouter_step_validation_when_temperature_is_invalid(
    value: float,
) -> None:
    # given
    specification = {
        "type": "roboflow_core/qwen3_5_openrouter@v1",
        "name": "step_1",
        "images": "$inputs.image",
        "prompt": "This is my prompt",
        "api_key": "$inputs.open_router_api_key",
        "temperature": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize("value", ["unconstrained", "visual-question-answering"])
def test_qwen3_5_openrouter_when_prompt_not_delivered_when_required(value: str) -> None:
    # given
    specification = {
        "type": "roboflow_core/qwen3_5_openrouter@v1",
        "task_type": value,
        "name": "step_1",
        "images": "$inputs.image",
        "api_key": "$inputs.open_router_api_key",
    }

    # when
    with pytest.raises(ValidationError) as e:
        _ = BlockManifest.model_validate(specification)

    # then
    assert "`prompt`" in str(e.value)


@pytest.mark.parametrize(
    "value",
    [
        "classification",
        "multi-label-classification",
        "object-detection",
    ],
)
def test_qwen3_5_openrouter_when_classes_not_delivered_when_required(
    value: str,
) -> None:
    # given
    specification = {
        "type": "roboflow_core/qwen3_5_openrouter@v1",
        "task_type": value,
        "name": "step_1",
        "images": "$inputs.image",
        "api_key": "$inputs.open_router_api_key",
    }

    # when
    with pytest.raises(ValidationError) as e:
        _ = BlockManifest.model_validate(specification)

    # then
    assert "`classes`" in str(e.value)


def test_qwen3_5_openrouter_when_output_structure_not_delivered_when_required() -> None:
    # given
    specification = {
        "type": "roboflow_core/qwen3_5_openrouter@v1",
        "task_type": "structured-answering",
        "name": "step_1",
        "images": "$inputs.image",
        "api_key": "$inputs.open_router_api_key",
    }

    # when
    with pytest.raises(ValidationError) as e:
        _ = BlockManifest.model_validate(specification)

    # then
    assert "`output_structure`" in str(e.value)


def test_execute_qwen_request_when_request_succeeds() -> None:
    # given
    client = MagicMock()
    client.chat.completions.create.return_value = ChatCompletion(
        id="38",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="This is content from Qwen",
                ),
            )
        ],
        created=int(time.time()),
        model="qwen/qwen3.5-27b",
        object="chat.completion",
    )

    # when
    result = execute_qwen_request(
        client=client,
        prompt=[{"content": [{"text": "prompt"}]}],
        qwen_model_version="qwen/qwen3.5-27b",
        max_tokens=300,
        temperature=0.5,
    )

    # then
    assert result == "This is content from Qwen"
    call_kwargs = client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "qwen/qwen3.5-27b"
    assert call_kwargs["max_tokens"] == 300
    assert (
        len(call_kwargs["messages"]) == 1
    ), "Only single message is expected to be prompted"
    assert (
        call_kwargs["messages"][0]["content"][0]["text"] == "prompt"
    ), "Text prompt is expected to be injected without modification"


def test_execute_qwen_request_when_request_fails() -> None:
    # given
    client = MagicMock()
    return_value = MagicMock()
    return_value.choices = None
    return_value.error = {"message": "Error MSG"}
    client.chat.completions.create.return_value = return_value

    # when
    with pytest.raises(RuntimeError) as e:
        _ = execute_qwen_request(
            client=client,
            prompt=[{"content": [{"text": "prompt"}]}],
            qwen_model_version="qwen/qwen3.5-27b",
            max_tokens=300,
            temperature=0.5,
        )

    # then
    call_kwargs = client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "qwen/qwen3.5-27b"
    assert call_kwargs["max_tokens"] == 300
    assert (
        len(call_kwargs["messages"]) == 1
    ), "Only single message is expected to be prompted"
    assert (
        call_kwargs["messages"][0]["content"][0]["text"] == "prompt"
    ), "Text prompt is expected to be injected without modification"
    assert "Details: Error MSG" in str(e.value)
