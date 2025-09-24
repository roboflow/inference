import json
import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.foundation.openai.v1 import (
    BlockManifest,
    LMMConfig,
    execute_gpt_4v_request,
    try_parse_json,
    try_parse_lmm_output_to_json,
)


@pytest.mark.parametrize("type_alias", ["roboflow_core/open_ai@v1", "OpenAI"])
@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_openai_step_validation_when_input_is_valid(
    type_alias: str, images_field_alias: str
) -> None:
    # given
    specification = {
        "type": type_alias,
        "name": "step_1",
        images_field_alias: "$inputs.image",
        "prompt": "$inputs.prompt",
        "json_output_format": {"some_field": "some_description"},
        "openai_api_key": "$inputs.openai_api_key",
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result == BlockManifest(
        type=type_alias,
        name="step_1",
        images="$inputs.image",
        prompt="$inputs.prompt",
        openai_api_key="$inputs.openai_api_key",
        json_output_format={"some_field": "some_description"},
    )


@pytest.mark.parametrize("value", [None, 1, "a", True])
def test_lmm_step_validation_when_image_is_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "OpenAI",
        "name": "step_1",
        "images": value,
        "prompt": "$inputs.prompt",
        "json_output_format": "$inputs.expected_output",
        "openai_api_key": "$inputs.openai_api_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_lmm_step_validation_when_prompt_is_given_directly() -> None:
    # given
    specification = {
        "type": "OpenAI",
        "name": "step_1",
        "images": "$inputs.image",
        "prompt": "This is my prompt",
        "openai_api_key": "$inputs.openai_api_key",
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result == BlockManifest(
        type="OpenAI",
        name="step_1",
        images="$inputs.image",
        prompt="This is my prompt",
        openai_api_key="$inputs.openai_api_key",
    )


@pytest.mark.parametrize("value", [None, []])
def test_lmm_step_validation_when_prompt_is_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "OpenAI",
        "name": "step_1",
        "images": "$inputs.image",
        "prompt": value,
        "openai_api_key": "$inputs.openai_api_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize("value", ["$inputs.model", "gpt-4o", "gpt-4o-mini"])
def test_lmm_step_validation_when_lmm_type_valid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "OpenAI",
        "name": "step_1",
        "images": "$inputs.image",
        "openai_model": value,
        "prompt": "$inputs.prompt",
        "openai_api_key": "$inputs.openai_api_key",
    }

    # when
    result = BlockManifest.model_validate(specification)

    assert result == BlockManifest(
        type="OpenAI",
        name="step_1",
        images="$inputs.image",
        prompt="$inputs.prompt",
        openai_model=value,
        openai_api_key="$inputs.openai_api_key",
    )


@pytest.mark.parametrize("value", ["some", None])
def test_lmm_step_validation_when_lmm_type_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "OpenAI",
        "name": "step_1",
        "images": "$inputs.image",
        "openai_model": value,
        "prompt": "$inputs.prompt",
        "openai_api_key": "$inputs.openai_api_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize("value", ["$inputs.api_key", "my-api-key", None])
def test_lmm_step_validation_when_api_key_valid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "OpenAI",
        "name": "step_1",
        "images": "$inputs.image",
        "openai_model": "gpt-4o-mini",
        "prompt": "$inputs.prompt",
        "openai_api_key": value,
    }

    # when
    result = BlockManifest.model_validate(specification)

    assert result == BlockManifest(
        type="OpenAI",
        name="step_1",
        images="$inputs.image",
        prompt="$inputs.prompt",
        openai_model="gpt-4o-mini",
        openai_api_key=value,
    )


@pytest.mark.parametrize(
    "value",
    [{"my_field": 3}, "some"],
)
def test_lmm_step_validation_when_json_output_format_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "OpenAI",
        "name": "step_1",
        "images": "$inputs.image",
        "openai_model": "gpt-4o-mini",
        "prompt": "$inputs.prompt",
        "json_output_format": value,
        "openai_api_key": "some",
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


def test_try_parse_openai_output_to_json_when_no_json_to_be_found_in_input() -> None:
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


def test_try_parse_openai_output_to_json_when_single_json_markdown_block_with_linearised_document_found() -> (
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


def test_try_parse_openai_output_to_json_when_single_json_markdown_block_with_multi_line_document_found() -> (
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


def test_try_parse_openai_output_to_json_when_single_json_without_markdown_spotted() -> (
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


def test_try_parse_openai_output_to_json_when_multiple_json_markdown_blocks_with_linearised_document_found() -> (
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


def test_try_parse_openai_output_to_json_when_multiple_json_markdown_blocks_with_multi_line_document_found() -> (
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


def test_execute_gpt_4v_request() -> None:
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
                    content="This is content from GPT",
                ),
            )
        ],
        created=int(time.time()),
        model="gpt-4o",
        object="chat.completion",
    )

    # when
    result = execute_gpt_4v_request(
        client=client,
        image={
            "type": "numpy_object",
            "value": np.zeros((192, 168, 3), dtype=np.uint8),
        },
        prompt="My prompt",
        lmm_config=LMMConfig(
            gpt_model_version="gpt-4o",
            gpt_image_detail="low",
            max_tokens=600,
        ),
    )

    # then
    assert result == {
        "content": "This is content from GPT",
        "image": {"width": 168, "height": 192},
    }
    call_kwargs = client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-4o"
    assert call_kwargs["max_tokens"] == 600
    assert (
        len(call_kwargs["messages"]) == 1
    ), "Only single message is expected to be prompted"
    assert (
        call_kwargs["messages"][0]["content"][0]["text"] == "My prompt"
    ), "Text prompt is expected to be injected without modification"
    assert (
        call_kwargs["messages"][0]["content"][1]["image_url"]["detail"] == "low"
    ), "Image details level expected to be set to `low` as in LMMConfig"


def test_execute_gpt_4v_request_gpt_4o_mini() -> None:
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
                    content="This is content from GPT",
                ),
            )
        ],
        created=int(time.time()),
        model="gpt-4o-mini",
        object="chat.completion",
    )

    # when
    result = execute_gpt_4v_request(
        client=client,
        image={
            "type": "numpy_object",
            "value": np.zeros((192, 168, 3), dtype=np.uint8),
        },
        prompt="My prompt",
        lmm_config=LMMConfig(
            gpt_model_version="gpt-4o-mini",
            gpt_image_detail="high",
            max_tokens=120,
        ),
    )

    # then
    assert result == {
        "content": "This is content from GPT",
        "image": {"width": 168, "height": 192},
    }
    call_kwargs = client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs["max_tokens"] == 120
    assert (
        len(call_kwargs["messages"]) == 1
    ), "Only single message is expected to be prompted"
    assert (
        call_kwargs["messages"][0]["content"][0]["text"] == "My prompt"
    ), "Text prompt is expected to be injected without modification"
    assert (
        call_kwargs["messages"][0]["content"][1]["image_url"]["detail"] == "high"
    ), "Image details level expected to be set to `low` as in LMMConfig"
