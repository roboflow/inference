import time
from typing import Any
from unittest.mock import MagicMock

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.foundation.openpaths.v1 import (
    DEFAULT_OPENPATHS_MODEL,
    OPENPATHS_BASE_URL,
    BlockManifest,
    execute_openpaths_request,
)


@pytest.mark.parametrize("value", [None, 1, "a", True])
def test_openpaths_step_validation_when_image_is_invalid(value: Any) -> None:
    # given
    specification = {
        "type": "roboflow_core/open_paths@v1",
        "name": "step_1",
        "images": value,
        "prompt": "$inputs.prompt",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_openpaths_step_validation_when_prompt_is_given_directly() -> None:
    # given
    specification = {
        "type": "roboflow_core/open_paths@v1",
        "name": "step_1",
        "images": "$inputs.image",
        "prompt": "This is my prompt",
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.type == "roboflow_core/open_paths@v1"
    assert result.prompt == "This is my prompt"
    assert result.model_id == DEFAULT_OPENPATHS_MODEL
    assert result.api_key is None


def test_openpaths_step_validation_when_model_id_is_overridden() -> None:
    # given
    specification = {
        "type": "roboflow_core/open_paths@v1",
        "name": "step_1",
        "images": "$inputs.image",
        "prompt": "This is my prompt",
        "model_id": "openpaths/auto-vision",
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.model_id == "openpaths/auto-vision"


def test_openpaths_when_prompt_not_delivered_when_required() -> None:
    # given
    specification = {
        "type": "roboflow_core/open_paths@v1",
        "task_type": "visual-question-answering",
        "name": "step_1",
        "images": "$inputs.image",
    }

    # when
    with pytest.raises(ValidationError) as e:
        _ = BlockManifest.model_validate(specification)

    # then
    assert "`prompt`" in str(e.value)


@pytest.mark.parametrize(
    "value",
    ["classification", "multi-label-classification", "object-detection"],
)
def test_openpaths_when_classes_not_delivered_when_required(value: str) -> None:
    # given
    specification = {
        "type": "roboflow_core/open_paths@v1",
        "task_type": value,
        "name": "step_1",
        "images": "$inputs.image",
    }

    # when
    with pytest.raises(ValidationError) as e:
        _ = BlockManifest.model_validate(specification)

    # then
    assert "`classes`" in str(e.value)


def test_openpaths_when_output_structure_not_delivered_when_required() -> None:
    # given
    specification = {
        "type": "roboflow_core/open_paths@v1",
        "task_type": "structured-answering",
        "name": "step_1",
        "images": "$inputs.image",
    }

    # when
    with pytest.raises(ValidationError) as e:
        _ = BlockManifest.model_validate(specification)

    # then
    assert "`output_structure`" in str(e.value)


@pytest.mark.parametrize("value", [-0.5, 3.0])
def test_openpaths_step_validation_when_temperature_is_invalid(value: float) -> None:
    # given
    specification = {
        "type": "roboflow_core/open_paths@v1",
        "name": "step_1",
        "images": "$inputs.image",
        "prompt": "This is my prompt",
        "temperature": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_openpaths_base_url_is_openpaths() -> None:
    assert OPENPATHS_BASE_URL == "https://openpaths.io/v1"


def test_execute_openpaths_request_when_request_succeeds() -> None:
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
                    content="This is content from OpenPaths",
                ),
            )
        ],
        created=int(time.time()),
        model="openpaths/auto",
        object="chat.completion",
    )

    # when
    result = execute_openpaths_request(
        client=client,
        model="openpaths/auto",
        messages=[{"role": "user", "content": [{"type": "text", "text": "prompt"}]}],
        max_tokens=300,
        temperature=0.5,
    )

    # then
    assert result == "This is content from OpenPaths"
    call_kwargs = client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "openpaths/auto"
    assert call_kwargs["max_tokens"] == 300


def test_execute_openpaths_request_when_request_fails() -> None:
    # given
    client = MagicMock()
    return_value = MagicMock()
    return_value.choices = None
    return_value.error = {"message": "Error MSG"}
    client.chat.completions.create.return_value = return_value

    # when
    with pytest.raises(RuntimeError) as e:
        _ = execute_openpaths_request(
            client=client,
            model="openpaths/auto",
            messages=[{"role": "user", "content": [{"type": "text", "text": "p"}]}],
            max_tokens=300,
            temperature=0.5,
        )

    # then
    assert "Details: Error MSG" in str(e.value)
