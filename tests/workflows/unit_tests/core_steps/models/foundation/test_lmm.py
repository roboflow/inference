import json
import time
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from pydantic import ValidationError

from inference.core.entities.responses.cogvlm import CogVLMResponse
from inference.enterprise.workflows.core_steps.models.foundation import lmm
from inference.enterprise.workflows.core_steps.models.foundation.lmm import (
    BlockManifest,
    LMMConfig,
    execute_gpt_4v_request,
    get_cogvlm_generations_from_remote_api,
    get_cogvlm_generations_locally,
    try_parse_json,
    try_parse_lmm_output_to_json,
)
from inference.enterprise.workflows.entities.base import StepExecutionMode


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_lmm_step_validation_when_input_is_valid(images_field_alias: str) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        images_field_alias: "$inputs.image",
        "lmm_type": "$inputs.lmm_type",
        "prompt": "$inputs.prompt",
        "json_output": {"some_field": "some_description"},
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result == BlockManifest(
        type="LMM",
        name="step_1",
        images="$inputs.image",
        prompt="$inputs.prompt",
        lmm_type="$inputs.lmm_type",
        lmm_config=LMMConfig(),
        remote_api_key="$inputs.open_ai_key",
        json_output={"some_field": "some_description"},
    )


@pytest.mark.parametrize("value", [None, 1, "a", True])
def test_lmm_step_validation_when_image_is_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "images": value,
        "lmm_type": "$inputs.lmm_type",
        "prompt": "$inputs.prompt",
        "json_output": "$inputs.expected_output",
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_lmm_step_validation_when_prompt_is_given_directly() -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "images": "$inputs.image",
        "lmm_type": "$inputs.lmm_type",
        "prompt": "This is my prompt",
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result == BlockManifest(
        type="LMM",
        name="step_1",
        images="$inputs.image",
        prompt="This is my prompt",
        lmm_type="$inputs.lmm_type",
        lmm_config=LMMConfig(),
        remote_api_key="$inputs.open_ai_key",
    )


@pytest.mark.parametrize("value", [None, []])
def test_lmm_step_validation_when_prompt_is_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "images": "$inputs.image",
        "lmm_type": "$inputs.lmm_type",
        "prompt": value,
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize("value", ["$inputs.model", "gpt_4v", "cog_vlm"])
def test_lmm_step_validation_when_lmm_type_valid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "images": "$inputs.image",
        "lmm_type": value,
        "prompt": "$inputs.prompt",
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    result = BlockManifest.model_validate(specification)

    assert result == BlockManifest(
        type="LMM",
        name="step_1",
        images="$inputs.image",
        prompt="$inputs.prompt",
        lmm_type=value,
        lmm_config=LMMConfig(),
        remote_api_key="$inputs.open_ai_key",
    )


@pytest.mark.parametrize("value", ["some", None])
def test_lmm_step_validation_when_lmm_type_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "images": "$inputs.image",
        "lmm_type": value,
        "prompt": "$inputs.prompt",
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize("value", ["$inputs.api_key", "my-api-key", None])
def test_lmm_step_validation_when_remote_api_key_valid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "images": "$inputs.image",
        "lmm_type": "gpt_4v",
        "prompt": "$inputs.prompt",
        "remote_api_key": value,
    }

    # when
    result = BlockManifest.model_validate(specification)

    assert result == BlockManifest(
        type="LMM",
        name="step_1",
        images="$inputs.image",
        prompt="$inputs.prompt",
        lmm_type="gpt_4v",
        lmm_config=LMMConfig(),
        remote_api_key=value,
    )


@pytest.mark.parametrize(
    "value",
    [{"my_field": 3}, "some"],
)
def test_lmm_step_validation_when_json_output_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "images": "$inputs.image",
        "lmm_type": "gpt_4v",
        "prompt": "$inputs.prompt",
        "json_output": value,
        "remote_api_key": "some",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_try_parse_json_when_input_is_not_json_parsable() -> None:
    # when
    result = try_parse_json(
        content="for sure not a valid JSON",
        expected_output={"field_a": "my field", "field_b": "other_field"},
    )

    # then
    assert result == {
        "field_a": "not_detected",
        "field_b": "not_detected",
    }, "No field detected is expected output"


def test_try_parse_json_when_input_is_json_parsable_and_some_fields_are_missing() -> (
    None
):
    # when
    result = try_parse_json(
        content=json.dumps({"field_a": "XXX", "field_c": "additional_field"}),
        expected_output={"field_a": "my field", "field_b": "other_field"},
    )

    # then
    assert result == {
        "field_a": "XXX",
        "field_b": "not_detected",
    }, "field_a must be extracted, `field_b` is missing and field_c should be ignored"


def test_try_parse_json_when_input_is_json_parsable_and_all_values_are_delivered() -> (
    None
):
    # when
    result = try_parse_json(
        content=json.dumps({"field_a": "XXX", "field_b": "YYY"}),
        expected_output={"field_a": "my field", "field_b": "other_field"},
    )

    # then
    assert result == {
        "field_a": "XXX",
        "field_b": "YYY",
    }, "Both fields must be detected with values specified in content"


def test_try_parse_lmm_output_to_json_when_no_json_to_be_found_in_input() -> None:
    # when
    result = try_parse_lmm_output_to_json(
        output="for sure not a valid JSON",
        expected_output={"field_a": "my field", "field_b": "other_field"},
    )

    # then
    assert result == {
        "field_a": "not_detected",
        "field_b": "not_detected",
    }, "No field detected is expected output"


def test_try_parse_lmm_output_to_json_when_single_json_markdown_block_with_linearised_document_found() -> (
    None
):
    # given
    output = """
This is some comment produced by LLM

```json
{"field_a": 1, "field_b": 37}
```
""".strip()
    # when
    result = try_parse_lmm_output_to_json(
        output=output, expected_output={"field_a": "my field", "field_b": "other_field"}
    )

    # then
    assert result == {"field_a": 1, "field_b": 37}


def test_try_parse_lmm_output_to_json_when_single_json_markdown_block_with_multi_line_document_found() -> (
    None
):
    # given
    output = """
This is some comment produced by LLM

```json
{
    "field_a": 1,
    "field_b": 37
}
```
""".strip()
    # when
    result = try_parse_lmm_output_to_json(
        output=output, expected_output={"field_a": "my field", "field_b": "other_field"}
    )

    # then
    assert result == {"field_a": 1, "field_b": 37}


def test_try_parse_lmm_output_to_json_when_single_json_without_markdown_spotted() -> (
    None
):
    # given
    output = """
{
    "field_a": 1,
    "field_b": 37
}
""".strip()
    # when
    result = try_parse_lmm_output_to_json(
        output=output, expected_output={"field_a": "my field", "field_b": "other_field"}
    )

    # then
    assert result == {"field_a": 1, "field_b": 37}


def test_try_parse_lmm_output_to_json_when_multiple_json_markdown_blocks_with_linearised_document_found() -> (
    None
):
    # given
    output = """
This is some comment produced by LLM

```json
{"field_a": 1, "field_b": 37}
```
some other comment

```json
{"field_a": 2, "field_b": 47}
```
""".strip()
    # when
    result = try_parse_lmm_output_to_json(
        output=output, expected_output={"field_a": "my field", "field_b": "other_field"}
    )

    # then
    assert result == [{"field_a": 1, "field_b": 37}, {"field_a": 2, "field_b": 47}]


def test_try_parse_lmm_output_to_json_when_multiple_json_markdown_blocks_with_multi_line_document_found() -> (
    None
):
    # given
    output = """
This is some comment produced by LLM

```json
{
    "field_a": 1,
    "field_b": 37
}
```

Some other comment
```json
{
    "field_a": 2,
    "field_b": 47
}
```
""".strip()
    # when
    result = try_parse_lmm_output_to_json(
        output=output, expected_output={"field_a": "my field", "field_b": "other_field"}
    )

    # then
    assert result == [{"field_a": 1, "field_b": 37}, {"field_a": 2, "field_b": 47}]


@pytest.mark.asyncio
@mock.patch.object(lmm, "WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS", 2)
@mock.patch.object(lmm, "WORKFLOWS_REMOTE_API_TARGET", "self-hosted")
@mock.patch.object(lmm.InferenceHTTPClient, "init")
async def test_get_cogvlm_generations_from_remote_api(
    inference_client_init_mock: MagicMock,
) -> None:
    # given
    client_mock = AsyncMock()
    client_mock.prompt_cogvlm_async.side_effect = [
        {"response": "Response 1: 42"},
        {"response": "Response 2: 42"},
        {"response": "Response 3: 42"},
    ]
    inference_client_init_mock.return_value = client_mock

    # when
    result = await get_cogvlm_generations_from_remote_api(
        image=[
            {"type": "numpy_object", "value": np.zeros((192, 168, 3), dtype=np.uint8)},
            {"type": "numpy_object", "value": np.zeros((193, 168, 3), dtype=np.uint8)},
            {"type": "numpy_object", "value": np.zeros((194, 168, 3), dtype=np.uint8)},
        ],
        prompt="What is the meaning of life?",
        api_key="some",
    )

    # then
    assert result == [
        {"content": "Response 1: 42", "image": {"width": 168, "height": 192}},
        {"content": "Response 2: 42", "image": {"width": 168, "height": 193}},
        {"content": "Response 3: 42", "image": {"width": 168, "height": 194}},
    ]


@pytest.mark.asyncio
@mock.patch.object(lmm, "load_core_model", MagicMock())
async def test_get_cogvlm_generations_locally() -> None:
    # given
    model_manager = AsyncMock()
    model_manager.infer_from_request.side_effect = [
        CogVLMResponse.model_validate({"response": "Response 1: 42"}),
        CogVLMResponse.model_validate({"response": "Response 2: 42"}),
        CogVLMResponse.model_validate({"response": "Response 3: 42"}),
    ]

    # when
    result = await get_cogvlm_generations_locally(
        image=[
            {"type": "numpy_object", "value": np.zeros((192, 168, 3), dtype=np.uint8)},
            {"type": "numpy_object", "value": np.zeros((193, 168, 3), dtype=np.uint8)},
            {"type": "numpy_object", "value": np.zeros((194, 168, 3), dtype=np.uint8)},
        ],
        prompt="What is the meaning of life?",
        model_manager=model_manager,
        api_key="some",
    )

    # then
    assert result == [
        {"content": "Response 1: 42", "image": {"width": 168, "height": 192}},
        {"content": "Response 2: 42", "image": {"width": 168, "height": 193}},
        {"content": "Response 3: 42", "image": {"width": 168, "height": 194}},
    ]


@pytest.mark.asyncio
async def test_execute_gpt_4v_request() -> None:
    # given
    client = AsyncMock()
    client.chat.completions.create.return_value = ChatCompletion(
        id="38",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="This is content from GPT",
                ),
            )
        ],
        created=int(time.time()),
        model="gpt-4-vision-preview",
        object="chat.completion",
    )

    # when
    result = await execute_gpt_4v_request(
        client=client,
        image={
            "type": "numpy_object",
            "value": np.zeros((192, 168, 3), dtype=np.uint8),
        },
        prompt="My prompt",
        lmm_config=LMMConfig(gpt_image_detail="low", max_tokens=120),
    )

    # then
    assert result == {
        "content": "This is content from GPT",
        "image": {"width": 168, "height": 192},
    }
    call_kwargs = client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-4-vision-preview"
    assert call_kwargs["max_tokens"] == 120
    assert (
        len(call_kwargs["messages"]) == 1
    ), "Only single message is expected to be prompted"
    assert (
        call_kwargs["messages"][0]["content"][0]["text"] == "My prompt"
    ), "Text prompt is expected to be injected without modification"
    assert (
        call_kwargs["messages"][0]["content"][1]["image_url"]["detail"] == "low"
    ), "Image details level expected to be set to `low` as in LMMConfig"
