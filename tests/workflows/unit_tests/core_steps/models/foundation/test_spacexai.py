from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.foundation.spacexai.v1 import (
    OBJECT_DETECTION_PROMPT_TEMPLATE,
    BlockManifest,
    _execute_proxied_spacexai_request,
    _extract_output_text,
    _is_unsupported_reasoning_error,
    encode_image_for_task,
    execute_spacexai_request,
    prepare_object_detection_prompt,
)
from inference.core.workflows.prototypes.block import third_party_model

PNG_MAGIC_BYTES = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC_BYTES = b"\xff\xd8\xff"


def test_spacexai_step_validation_when_input_is_valid() -> None:
    specification = {
        "type": "roboflow_core/spacexai@v1",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "unconstrained",
        "prompt": "$inputs.prompt",
        "api_key": "$inputs.xai_api_key",
    }

    result = BlockManifest.model_validate(specification)

    assert result.type == "roboflow_core/spacexai@v1"
    assert result.name == "step_1"
    assert result.images == "$inputs.image"
    assert result.task_type == "unconstrained"
    assert result.prompt == "$inputs.prompt"
    assert result.api_key == "$inputs.xai_api_key"


def test_spacexai_step_validation_with_default_api_key() -> None:
    specification = {
        "type": "roboflow_core/spacexai@v1",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
    }

    result = BlockManifest.model_validate(specification)

    assert result.api_key == "rf_key:account"
    assert result.model_version == "grok-4.6"


def test_spacexai_step_discovers_dependent_model() -> None:
    specification = {
        "type": "roboflow_core/spacexai@v1",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
        "model_version": "grok-4.5",
    }

    manifest = BlockManifest.model_validate(specification)

    result = manifest.discover_dependent_resources()

    assert result == [third_party_model(provider="xai", model_id="grok-4.5")]


@pytest.mark.parametrize("model_version", ["grok-4.6", "grok-4.5", "$inputs.model"])
def test_spacexai_step_validation_when_model_version_valid(
    model_version: str,
) -> None:
    specification = {
        "type": "roboflow_core/spacexai@v1",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
        "model_version": model_version,
    }

    result = BlockManifest.model_validate(specification)

    assert result.model_version == model_version


@pytest.mark.parametrize("value", ["invalid-model", 123])
def test_spacexai_step_validation_when_model_version_invalid(value: Any) -> None:
    specification = {
        "type": "roboflow_core/spacexai@v1",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
        "model_version": value,
    }

    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_spacexai_step_validation_requires_prompt_for_unconstrained() -> None:
    specification = {
        "type": "roboflow_core/spacexai@v1",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "unconstrained",
    }

    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_spacexai_step_validation_requires_classes_for_object_detection() -> None:
    specification = {
        "type": "roboflow_core/spacexai@v1",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "object-detection",
    }

    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_prepare_object_detection_prompt_uses_percent_contract() -> None:
    prompt = prepare_object_detection_prompt(
        base64_image="abc123",
        classes=["cat", "dog"],
    )

    content = prompt["input"][0]["content"]
    assert content[0]["type"] == "input_image"
    assert content[0]["image_url"].startswith("data:image/png;base64,")
    assert content[0]["detail"] == "high"
    assert content[1]["type"] == "input_text"
    assert "floats between 0 and 100" in content[1]["text"]
    assert "cat, dog" in content[1]["text"]
    assert OBJECT_DETECTION_PROMPT_TEMPLATE.format(class_list="cat, dog") == content[1][
        "text"
    ]


def test_encode_image_for_task_uses_png_for_detection() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    base64_image, width, height = encode_image_for_task(
        image, task_type="object-detection"
    )

    import base64

    raw = base64.b64decode(base64_image)
    assert raw.startswith(PNG_MAGIC_BYTES)
    assert (width, height) == (200, 100)


def test_encode_image_for_task_uses_jpeg_for_caption() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    base64_image, width, height = encode_image_for_task(image, task_type="caption")

    import base64

    raw = base64.b64decode(base64_image)
    assert raw.startswith(JPEG_MAGIC_BYTES)
    assert (width, height) == (200, 100)


def test_extract_output_text_from_completed_response() -> None:
    response_data = {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "hello world"}],
            }
        ],
    }

    assert _extract_output_text(response_data) == "hello world"


def test_extract_output_text_raises_on_max_tokens() -> None:
    response_data = {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
    }

    with pytest.raises(ValueError, match="max_tokens"):
        _extract_output_text(response_data)


def test_is_unsupported_reasoning_error() -> None:
    assert _is_unsupported_reasoning_error(
        ValueError("reasoning is not supported for this model")
    )
    assert not _is_unsupported_reasoning_error(ValueError("rate limit exceeded"))


@patch(
    "inference.core.workflows.core_steps.models.foundation.spacexai.v1."
    "_execute_proxied_spacexai_request"
)
def test_execute_spacexai_request_routes_rf_key_to_proxy(
    proxied_mock: MagicMock,
) -> None:
    proxied_mock.return_value = "proxied"
    result = execute_spacexai_request(
        roboflow_api_key="rf_abc",
        xai_api_key="rf_key:account",
        instructions=None,
        input_content=[{"role": "user", "content": []}],
        model_version="grok-4.6",
        reasoning_effort=None,
        max_tokens=None,
        temperature=None,
    )
    assert result == "proxied"
    proxied_mock.assert_called_once()


@patch(
    "inference.core.workflows.core_steps.models.foundation.spacexai.v1."
    "_execute_direct_spacexai_request"
)
def test_execute_spacexai_request_routes_direct_key(
    direct_mock: MagicMock,
) -> None:
    direct_mock.return_value = "direct"
    result = execute_spacexai_request(
        roboflow_api_key="rf_abc",
        xai_api_key="xai-secret",
        instructions=None,
        input_content=[{"role": "user", "content": []}],
        model_version="grok-4.6",
        reasoning_effort=None,
        max_tokens=None,
        temperature=None,
    )
    assert result == "direct"
    direct_mock.assert_called_once()


@patch(
    "inference.core.workflows.core_steps.models.foundation.spacexai.v1."
    "post_to_roboflow_api"
)
def test_execute_proxied_spacexai_request_payload(post_mock: MagicMock) -> None:
    post_mock.return_value = {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "ok"}],
            }
        ],
    }

    result = _execute_proxied_spacexai_request(
        roboflow_api_key="rf_abc",
        xai_api_key="rf_key:account",
        instructions="sys",
        input_content=[{"role": "user", "content": []}],
        model_version="grok-4.5",
        reasoning_effort="high",
        max_tokens=1024,
        temperature=0.2,
    )

    assert result == "ok"
    post_mock.assert_called_once()
    _, kwargs = post_mock.call_args
    assert kwargs["endpoint"] == "apiproxy/xai"
    payload = kwargs["payload"]
    assert payload["model"] == "grok-4.5"
    assert payload["xai_api_key"] == "rf_key:account"
    assert payload["store"] is False
    assert payload["max_output_tokens"] == 1024
    assert payload["reasoning"] == {"effort": "high"}
    assert payload["instructions"] == "sys"
