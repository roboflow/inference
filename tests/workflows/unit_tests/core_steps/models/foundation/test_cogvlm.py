import json
from typing import Any
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.entities.responses.cogvlm import CogVLMResponse
from inference.core.workflows.core_steps.models.foundation.cog_vlm import v1
from inference.core.workflows.core_steps.models.foundation.cog_vlm.v1 import (
    BlockManifest,
    get_cogvlm_generations_from_remote_api,
    get_cogvlm_generations_locally,
    try_parse_json,
    try_parse_lmm_output_to_json,
)


@pytest.mark.parametrize("type_alias", ["roboflow_core/cog_vlm@v1", "CogVLM"])
@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_cogvlm_step_validation_when_input_is_valid(
    type_alias: str, images_field_alias: str
) -> None:
    # given
    specification = {
        "type": type_alias,
        "name": "step_1",
        images_field_alias: "$inputs.image",
        "prompt": "$inputs.prompt",
        "json_output_format": {"some_field": "some_description"},
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result == BlockManifest(
        type=type_alias,
        name="step_1",
        images="$inputs.image",
        prompt="$inputs.prompt",
        json_output_format={"some_field": "some_description"},
    )


@pytest.mark.parametrize("value", [None, 1, "a", True])
def test_cogvlm_step_validation_when_image_is_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "CogVLM",
        "name": "step_1",
        "images": value,
        "prompt": "$inputs.prompt",
        "json_output_format": "$inputs.expected_output",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_cogvlm_step_validation_when_prompt_is_given_directly() -> None:
    # given
    specification = {
        "type": "CogVLM",
        "name": "step_1",
        "images": "$inputs.image",
        "prompt": "This is my prompt",
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result == BlockManifest(
        type="CogVLM", name="step_1", images="$inputs.image", prompt="This is my prompt"
    )


@pytest.mark.parametrize("value", [None, []])
def test_cogvlm_step_validation_when_prompt_is_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "CogVLM",
        "name": "step_1",
        "images": "$inputs.image",
        "prompt": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize(
    "value",
    [{"my_field": 3}, "some"],
)
def test_cogvlm_step_validation_when_json_output_format_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "CogVLM",
        "name": "step_1",
        "images": "$inputs.image",
        "prompt": "$inputs.prompt",
        "json_output_format": value,
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


def test_try_parse_cogvlm_output_to_json_when_no_json_to_be_found_in_input() -> None:
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


def test_try_parse_cogvlm_output_to_json_when_single_json_markdown_block_with_linearised_document_found() -> (
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


def test_try_parse_cogvlm_output_to_json_when_single_json_markdown_block_with_multi_line_document_found() -> (
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


def test_try_parse_cogvlm_output_to_json_when_single_json_without_markdown_spotted() -> (
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


def test_try_parse_cogvlm_output_to_json_when_multiple_json_markdown_blocks_with_linearised_document_found() -> (
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


def test_try_parse_cogvlm_output_to_json_when_multiple_json_markdown_blocks_with_multi_line_document_found() -> (
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


@mock.patch.object(v1, "WORKFLOWS_REMOTE_API_TARGET", "self-hosted")
@mock.patch.object(v1.InferenceHTTPClient, "init")
def test_get_cogvlm_generations_from_remote_api(
    inference_client_init_mock: MagicMock,
) -> None:
    # given
    client_mock = MagicMock()
    client_mock.prompt_cogvlm.side_effect = [
        {"response": "Response 1: 42"},
        {"response": "Response 2: 42"},
        {"response": "Response 3: 42"},
    ]
    inference_client_init_mock.return_value = client_mock

    # when
    result = get_cogvlm_generations_from_remote_api(
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


@mock.patch.object(v1, "load_core_model", MagicMock())
def test_get_cogvlm_generations_locally() -> None:
    # given
    model_manager = MagicMock()
    model_manager.infer_from_request_sync.side_effect = [
        CogVLMResponse.model_validate({"response": "Response 1: 42"}),
        CogVLMResponse.model_validate({"response": "Response 2: 42"}),
        CogVLMResponse.model_validate({"response": "Response 3: 42"}),
    ]

    # when
    result = get_cogvlm_generations_locally(
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
