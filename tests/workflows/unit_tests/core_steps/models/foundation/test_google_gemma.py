import time
from typing import Any
from unittest.mock import MagicMock

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.foundation.google_gemma.v1 import (
    BlockManifest,
    execute_gemma_request,
)


@pytest.mark.parametrize("value", [None, 1, "a", True])
def test_google_gemma_step_validation_when_image_is_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "roboflow_core/google_gemma@v1",
        "name": "step_1",
        "images": value,
        "prompt": "$inputs.prompt",
        "api_key": "$inputs.open_router_api_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_google_gemma_step_validation_when_prompt_is_given_directly() -> None:
    # given
    specification = {
        "type": "roboflow_core/google_gemma@v1",
        "name": "step_1",
        "images": "$inputs.image",
        "prompt": "This is my prompt",
        "api_key": "$inputs.open_router_api_key",
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result == BlockManifest(
        type="roboflow_core/google_gemma@v1",
        name="step_1",
        images="$inputs.image",
        prompt="This is my prompt",
        api_key="$inputs.open_router_api_key",
    )


@pytest.mark.parametrize("value", [None, []])
def test_google_gemma_step_validation_when_prompt_is_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "roboflow_core/google_gemma@v1",
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
        "Gemma 4 31B - OpenRouter",
        "Gemma 4 26B A4B - OpenRouter",
    ],
)
def test_google_gemma_step_validation_when_model_type_valid(
    value: str,
) -> None:
    # given
    specification = {
        "type": "roboflow_core/google_gemma@v1",
        "name": "step_1",
        "images": "$inputs.image",
        "prompt": "This is my prompt",
        "api_key": "$inputs.open_router_api_key",
        "model_version": value,
    }

    # when
    result = BlockManifest.model_validate(specification)

    assert result == BlockManifest(
        type="roboflow_core/google_gemma@v1",
        name="step_1",
        images="$inputs.image",
        prompt="This is my prompt",
        api_key="$inputs.open_router_api_key",
        model_version=value,
    )


@pytest.mark.parametrize("value", ["some", None])
def test_google_gemma_step_validation_when_model_type_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "roboflow_core/google_gemma@v1",
        "name": "step_1",
        "images": "$inputs.image",
        "prompt": "This is my prompt",
        "api_key": "$inputs.open_router_api_key",
        "model_version": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_google_gemma_step_validation_when_api_key_not_given() -> None:
    # given
    specification = {
        "type": "roboflow_core/google_gemma@v1",
        "name": "step_1",
        "images": "$inputs.image",
        "prompt": "This is my prompt",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_google_gemma_step_validation_when_output_structure_invalid() -> None:
    # given
    specification = {
        "type": "roboflow_core/google_gemma@v1",
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
def test_google_gemma_step_validation_when_temperature_is_invalid(
    value: float,
) -> None:
    # given
    specification = {
        "type": "roboflow_core/google_gemma@v1",
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
def test_google_gemma_when_prompt_not_delivered_when_required(value: str) -> None:
    # given
    specification = {
        "type": "roboflow_core/google_gemma@v1",
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
def test_google_gemma_when_classes_not_delivered_when_required(
    value: str,
) -> None:
    # given
    specification = {
        "type": "roboflow_core/google_gemma@v1",
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


def test_google_gemma_when_output_structure_not_delivered_when_required() -> None:
    # given
    specification = {
        "type": "roboflow_core/google_gemma@v1",
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


def test_execute_gemma_request_when_request_succeeds() -> None:
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
                    content="This is content from Gemma",
                ),
            )
        ],
        created=int(time.time()),
        model="google/gemma-4-31b-it",
        object="chat.completion",
    )

    # when
    result = execute_gemma_request(
        client=client,
        prompt=[{"content": [{"text": "prompt"}]}],
        gemma_model_version="google/gemma-4-31b-it",
        max_tokens=300,
        temperature=0.5,
    )

    # then
    assert result == "This is content from Gemma"
    call_kwargs = client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "google/gemma-4-31b-it"
    assert call_kwargs["max_tokens"] == 300
    assert (
        len(call_kwargs["messages"]) == 1
    ), "Only single message is expected to be prompted"
    assert (
        call_kwargs["messages"][0]["content"][0]["text"] == "prompt"
    ), "Text prompt is expected to be injected without modification"


def test_execute_gemma_request_when_request_fails() -> None:
    # given
    client = MagicMock()
    return_value = MagicMock()
    return_value.choices = None
    return_value.error = {"message": "Error MSG"}
    client.chat.completions.create.return_value = return_value

    # when
    with pytest.raises(RuntimeError) as e:
        _ = execute_gemma_request(
            client=client,
            prompt=[{"content": [{"text": "prompt"}]}],
            gemma_model_version="google/gemma-4-31b-it",
            max_tokens=300,
            temperature=0.5,
        )

    # then
    call_kwargs = client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "google/gemma-4-31b-it"
    assert call_kwargs["max_tokens"] == 300
    assert (
        len(call_kwargs["messages"]) == 1
    ), "Only single message is expected to be prompted"
    assert (
        call_kwargs["messages"][0]["content"][0]["text"] == "prompt"
    ), "Text prompt is expected to be injected without modification"
    assert "Details: Error MSG" in str(e.value)
