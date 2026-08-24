import base64
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.foundation.spacexai.v1 import (
    OBJECT_DETECTION_PROMPT_TEMPLATE,
    BlockManifest,
    _execute_direct_spacexai_request,
    _execute_proxied_spacexai_request,
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


def test_spacexai_step_validation_requires_api_key_when_managed_key_disabled() -> None:
    # WORKFLOWS_SPACEXAI_MANAGED_KEY_ENABLED is off by default, so the block
    # must demand a user-provided xAI key instead of defaulting to rf_key.
    specification = {
        "type": "roboflow_core/spacexai@v1",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
    }

    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_spacexai_step_discovers_dependent_model() -> None:
    specification = {
        "type": "roboflow_core/spacexai@v1",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
        "api_key": "xxx-xxx",
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
        "api_key": "xxx-xxx",
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
        "api_key": "xxx-xxx",
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
        "api_key": "xxx-xxx",
    }

    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_spacexai_step_validation_requires_classes_for_object_detection() -> None:
    specification = {
        "type": "roboflow_core/spacexai@v1",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "object-detection",
        "api_key": "xxx-xxx",
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
    assert (
        OBJECT_DETECTION_PROMPT_TEMPLATE.format(class_list="cat, dog")
        == content[1]["text"]
    )


def test_encode_image_for_task_uses_png_for_detection() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    base64_image, width, height = encode_image_for_task(
        image, task_type="object-detection"
    )

    raw = base64.b64decode(base64_image)
    assert raw.startswith(PNG_MAGIC_BYTES)
    assert (width, height) == (200, 100)


def test_encode_image_for_task_keeps_original_resolution_for_detection() -> None:
    # vlm-exam sent original-resolution images to xAI; the block must not
    # downscale large detection inputs.
    image = np.zeros((2100, 3000, 3), dtype=np.uint8)

    _, width, height = encode_image_for_task(image, task_type="object-detection")

    assert (width, height) == (3000, 2100)


def test_encode_image_for_task_uses_jpeg_for_caption() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    base64_image, width, height = encode_image_for_task(image, task_type="caption")

    raw = base64.b64decode(base64_image)
    assert raw.startswith(JPEG_MAGIC_BYTES)
    assert (width, height) == (200, 100)


def test_execute_spacexai_request_rejects_rf_key_when_managed_key_disabled() -> None:
    # Default flag state: managed keys are off and the proxy must never be
    # contacted; the block demands a user-provided xAI key.
    with pytest.raises(ValueError, match="Provide your own xAI API key"):
        execute_spacexai_request(
            roboflow_api_key="rf_abc",
            xai_api_key="rf_key:account",
            instructions=None,
            input_content=[{"role": "user", "content": []}],
            model_version="grok-4.6",
            reasoning_effort=None,
            max_tokens=None,
            temperature=None,
        )


@patch(
    "inference.core.workflows.core_steps.models.foundation.spacexai.v1."
    "WORKFLOWS_SPACEXAI_MANAGED_KEY_ENABLED",
    True,
)
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


_XAI_OK = {
    "status": "completed",
    "output": [
        {
            "type": "message",
            "content": [{"type": "output_text", "text": "ok"}],
        }
    ],
}


@patch(
    "inference.core.workflows.core_steps.models.foundation.spacexai.v1.post_to_roboflow_api"
)
def test_proxied_request_returns_usage(mock_post: MagicMock) -> None:
    mock_post.return_value = {
        **_XAI_OK,
        "usage": {"input_tokens": 16, "output_tokens": 5},
    }

    result = _execute_proxied_spacexai_request(
        roboflow_api_key="rf_abc",
        xai_api_key="rf_key:account",
        instructions=None,
        input_content=[],
        model_version="grok-4.6",
        reasoning_effort=None,
        max_tokens=None,
        temperature=None,
    )

    assert result == ("ok", 16, 5)


@patch(
    "inference.core.workflows.core_steps.models.foundation.spacexai.v1.post_to_roboflow_api"
)
def test_proxied_request_usage_none_when_omitted(mock_post: MagicMock) -> None:
    mock_post.return_value = _XAI_OK

    result = _execute_proxied_spacexai_request(
        roboflow_api_key="rf_abc",
        xai_api_key="rf_key:account",
        instructions=None,
        input_content=[],
        model_version="grok-4.6",
        reasoning_effort=None,
        max_tokens=None,
        temperature=None,
    )

    assert result == ("ok", None, None)


@patch("inference.core.workflows.core_steps.models.foundation.spacexai.v1.OpenAI")
def test_direct_request_returns_usage(mock_openai_cls: MagicMock) -> None:
    client = MagicMock()
    response = MagicMock()
    response.status = "completed"
    response.output_text = "ok"
    response.usage = MagicMock(input_tokens=12, output_tokens=4)
    client.responses.create.return_value = response
    mock_openai_cls.return_value = client

    result = _execute_direct_spacexai_request(
        xai_api_key="xai-secret",
        instructions=None,
        input_content=[],
        model_version="grok-4.6",
        reasoning_effort=None,
        max_tokens=None,
        temperature=None,
    )

    assert result == ("ok", 12, 4)
