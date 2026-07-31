import base64
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.foundation.openai.v5 import (
    DETECTION_MAX_EDGE_PIXELS,
    MODEL_REASONING_EFFORT_VALUES,
    MODELS_NOT_SUPPORTING_REASONING_EFFORT,
    MODELS_SUPPORTING_REASONING_EFFORT,
    BlockManifest,
    _execute_direct_openai_request,
    _execute_proxied_openai_request,
    _extract_output_text,
    encode_image_for_task,
    execute_openai_request,
    prepare_classification_prompt,
    prepare_multi_label_classification_prompt,
    prepare_object_detection_prompt,
    prepare_ocr_prompt,
    prepare_structured_answering_prompt,
    prepare_unconstrained_prompt,
    prepare_vqa_prompt,
)

PNG_MAGIC_BYTES = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC_BYTES = b"\xff\xd8\xff"


def test_openai_step_validation_when_input_is_valid() -> None:
    # given
    specification = {
        "type": "roboflow_core/open_ai@v5",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "unconstrained",
        "prompt": "$inputs.prompt",
        "api_key": "$inputs.openai_api_key",
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.type == "roboflow_core/open_ai@v5"
    assert result.name == "step_1"
    assert result.images == "$inputs.image"
    assert result.task_type == "unconstrained"
    assert result.prompt == "$inputs.prompt"
    assert result.api_key == "$inputs.openai_api_key"


def test_openai_step_validation_with_default_api_key() -> None:
    # given
    specification = {
        "type": "roboflow_core/open_ai@v5",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.api_key == "rf_key:account"


@pytest.mark.parametrize("value", [None, 1, "a", True])
def test_openai_step_validation_when_image_is_invalid(value: Any) -> None:
    # given
    specification = {
        "type": "roboflow_core/open_ai@v5",
        "name": "step_1",
        "images": value,
        "task_type": "unconstrained",
        "prompt": "$inputs.prompt",
        "api_key": "$inputs.openai_api_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_openai_step_validation_when_prompt_is_given_directly() -> None:
    # given
    specification = {
        "type": "roboflow_core/open_ai@v5",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "unconstrained",
        "prompt": "This is my prompt",
        "api_key": "$inputs.openai_api_key",
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.prompt == "This is my prompt"


@pytest.mark.parametrize(
    "model_version",
    [
        "gpt-5.1",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-4.1",
        "gpt-4o",
        "$inputs.model",
    ],
)
def test_openai_step_validation_when_model_version_valid(model_version: str) -> None:
    # given
    specification = {
        "type": "roboflow_core/open_ai@v5",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
        "api_key": "$inputs.openai_api_key",
        "model_version": model_version,
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.model_version == model_version


@pytest.mark.parametrize("value", ["invalid-model", 123])
def test_openai_step_validation_when_model_version_invalid(value: Any) -> None:
    # given
    specification = {
        "type": "roboflow_core/open_ai@v5",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
        "api_key": "$inputs.openai_api_key",
        "model_version": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize(
    "reasoning_effort", ["none", "minimal", "low", "medium", "high", "$inputs.effort"]
)
def test_openai_step_validation_with_reasoning_effort(reasoning_effort: str) -> None:
    # given
    specification = {
        "type": "roboflow_core/open_ai@v5",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
        "api_key": "$inputs.openai_api_key",
        "model_version": "gpt-5.1",
        "reasoning_effort": reasoning_effort,
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.reasoning_effort == reasoning_effort


@pytest.mark.parametrize("value", ["invalid", 123, "very_high"])
def test_openai_step_validation_when_reasoning_effort_invalid(value: Any) -> None:
    # given
    specification = {
        "type": "roboflow_core/open_ai@v5",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
        "api_key": "$inputs.openai_api_key",
        "reasoning_effort": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_openai_step_validation_with_temperature() -> None:
    # given
    specification = {
        "type": "roboflow_core/open_ai@v5",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
        "api_key": "$inputs.openai_api_key",
        "temperature": 0.7,
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.temperature == 0.7


@pytest.mark.parametrize("value", [-0.1, 2.1, "invalid"])
def test_openai_step_validation_when_temperature_invalid(value: Any) -> None:
    # given
    specification = {
        "type": "roboflow_core/open_ai@v5",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
        "api_key": "$inputs.openai_api_key",
        "temperature": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_openai_step_validation_with_max_tokens() -> None:
    # given
    specification = {
        "type": "roboflow_core/open_ai@v5",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
        "api_key": "$inputs.openai_api_key",
        "max_tokens": 100,
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.max_tokens == 100


@pytest.mark.parametrize("value", [15, 10, 0, -1])
def test_openai_step_validation_when_max_tokens_below_minimum(value: int) -> None:
    # given
    specification = {
        "type": "roboflow_core/open_ai@v5",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
        "api_key": "$inputs.openai_api_key",
        "max_tokens": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_openai_step_validation_without_required_prompt() -> None:
    # given - unconstrained requires prompt
    specification = {
        "type": "roboflow_core/open_ai@v5",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "unconstrained",
        "api_key": "$inputs.openai_api_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_openai_step_validation_without_required_classes() -> None:
    # given - classification requires classes
    specification = {
        "type": "roboflow_core/open_ai@v5",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "classification",
        "api_key": "$inputs.openai_api_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_openai_step_validation_without_required_output_structure() -> None:
    # given - structured-answering requires output_structure
    specification = {
        "type": "roboflow_core/open_ai@v5",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "structured-answering",
        "api_key": "$inputs.openai_api_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_openai_step_validation_with_object_detection_and_classes() -> None:
    # given
    specification = {
        "type": "roboflow_core/open_ai@v5",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "object-detection",
        "classes": ["person", "car"],
        "api_key": "$inputs.openai_api_key",
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.task_type == "object-detection"
    assert result.classes == ["person", "car"]


def test_extract_output_text_success() -> None:
    # given
    response_data = {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "This is the response"}],
            }
        ],
    }

    # when
    result = _extract_output_text(response_data)

    # then
    assert result == "This is the response"


def test_extract_output_text_failed_status() -> None:
    # given
    response_data = {
        "status": "failed",
        "error": {"code": "invalid_request", "message": "Bad request"},
    }

    # when/then
    with pytest.raises(ValueError) as exc_info:
        _extract_output_text(response_data)

    assert "OpenAI API request failed" in str(exc_info.value)
    assert "invalid_request" in str(exc_info.value)


def test_execute_openai_request_routes_to_proxy_for_rf_key_account() -> None:
    # given
    with patch(
        "inference.core.workflows.core_steps.models.foundation.openai.v5._execute_proxied_openai_request"
    ) as mock_proxy:
        mock_proxy.return_value = "proxied response"

        # when
        result = execute_openai_request(
            roboflow_api_key="rf_api_key",
            openai_api_key="rf_key:account",
            instructions="test",
            input_content=[],
            model_version="gpt-5.1",
            reasoning_effort=None,
            max_tokens=None,
            temperature=None,
        )

        # then
        assert result == "proxied response"
        mock_proxy.assert_called_once()


def test_execute_openai_request_routes_to_direct_for_regular_api_key() -> None:
    # given
    with patch(
        "inference.core.workflows.core_steps.models.foundation.openai.v5._execute_direct_openai_request"
    ) as mock_direct:
        mock_direct.return_value = "direct response"

        # when
        result = execute_openai_request(
            roboflow_api_key="rf_api_key",
            openai_api_key="sk-test-key",
            instructions="test",
            input_content=[],
            model_version="gpt-5.1",
            reasoning_effort=None,
            max_tokens=None,
            temperature=None,
        )

        # then
        assert result == "direct response"
        mock_direct.assert_called_once()


@patch(
    "inference.core.workflows.core_steps.models.foundation.openai.v5._get_openai_client"
)
def test_direct_request_with_valid_reasoning_effort_for_gpt_5_1(
    mock_get_client: Mock,
) -> None:
    # given
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status = "completed"
    mock_response.output_text = "response"
    mock_client.responses.create.return_value = mock_response
    mock_get_client.return_value = mock_client

    # when
    result = _execute_direct_openai_request(
        openai_api_key="sk-test",
        instructions="test",
        input_content=[{"role": "user", "content": []}],
        model_version="gpt-5.1",
        reasoning_effort="high",
        max_tokens=None,
        temperature=None,
    )

    # then
    assert result == "response"
    call_kwargs = mock_client.responses.create.call_args[1]
    assert call_kwargs["reasoning"] == {"effort": "high"}


@patch(
    "inference.core.workflows.core_steps.models.foundation.openai.v5._get_openai_client"
)
def test_direct_request_with_invalid_reasoning_effort_for_gpt_5_1_raises_error(
    mock_get_client: Mock,
) -> None:
    # given
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    # when/then
    with pytest.raises(ValueError) as exc_info:
        _execute_direct_openai_request(
            openai_api_key="sk-test",
            instructions="test",
            input_content=[{"role": "user", "content": []}],
            model_version="gpt-5.1",
            reasoning_effort="minimal",  # not supported by gpt-5.1
            max_tokens=None,
            temperature=None,
        )

    assert 'does not support reasoning effort "minimal"' in str(exc_info.value)


@patch(
    "inference.core.workflows.core_steps.models.foundation.openai.v5.post_to_roboflow_api"
)
def test_proxied_request_with_invalid_reasoning_effort_for_gpt_5_raises_error(
    mock_post: Mock,
) -> None:
    # when/then
    with pytest.raises(ValueError) as exc_info:
        _execute_proxied_openai_request(
            roboflow_api_key="rf_api_key",
            openai_api_key="rf_key:account",
            instructions="test",
            input_content=[{"role": "user", "content": []}],
            model_version="gpt-5",
            reasoning_effort="none",  # not supported by gpt-5
            max_tokens=None,
            temperature=None,
        )

    assert 'does not support reasoning effort "none"' in str(exc_info.value)


def test_prepare_unconstrained_prompt() -> None:
    # when
    result = prepare_unconstrained_prompt(
        base64_image="test_image_data",
        prompt="Describe this image",
        image_detail="high",
    )

    # then
    assert "input" in result
    assert len(result["input"]) == 1
    user_message = result["input"][0]
    assert user_message["role"] == "user"
    assert len(user_message["content"]) == 2
    assert user_message["content"][0]["type"] == "input_text"
    assert user_message["content"][0]["text"] == "Describe this image"
    assert user_message["content"][1]["type"] == "input_image"
    assert user_message["content"][1]["detail"] == "high"


def test_prepare_classification_prompt() -> None:
    # when
    result = prepare_classification_prompt(
        base64_image="test_image_data",
        classes=["cat", "dog", "bird"],
        image_detail="auto",
    )

    # then
    assert "instructions" in result
    assert "classification model" in result["instructions"]
    assert "JSON document" in result["instructions"]
    user_content = result["input"][0]["content"]
    assert "cat, dog, bird" in user_content[0]["text"]


def test_prepare_multi_label_classification_prompt() -> None:
    # when
    result = prepare_multi_label_classification_prompt(
        base64_image="test_image_data",
        classes=["sunny", "cloudy"],
        image_detail="low",
    )

    # then
    assert "instructions" in result
    assert "multi-label classification" in result["instructions"]
    assert "predicted_classes" in result["instructions"]


def test_prepare_vqa_prompt() -> None:
    # when
    result = prepare_vqa_prompt(
        base64_image="test_image_data",
        prompt="What color is the car?",
        image_detail="auto",
    )

    # then
    assert "instructions" in result
    assert "Visual Question Answering" in result["instructions"]
    user_content = result["input"][0]["content"]
    assert "Question: What color is the car?" in user_content[0]["text"]


def test_prepare_ocr_prompt() -> None:
    # when
    result = prepare_ocr_prompt(
        base64_image="test_image_data",
        image_detail="high",
    )

    # then
    assert "instructions" in result
    assert "OCR model" in result["instructions"]
    user_content = result["input"][0]["content"]
    assert len(user_content) == 1
    assert user_content[0]["type"] == "input_image"


def test_prepare_structured_answering_prompt() -> None:
    # when
    result = prepare_structured_answering_prompt(
        base64_image="test_image_data",
        output_structure={"name": "person name", "age": "estimated age"},
        image_detail="auto",
    )

    # then
    assert "instructions" in result
    assert "JSON" in result["instructions"]
    user_content = result["input"][0]["content"]
    assert "name" in user_content[0]["text"]
    assert "age" in user_content[0]["text"]


def test_prepare_object_detection_prompt_uses_box_2d_format() -> None:
    # when
    result = prepare_object_detection_prompt(
        base64_image="test_image_data",
        classes=["person", "car"],
        image_detail="high",
    )

    # then - no system instructions, single user message
    assert "instructions" not in result
    assert len(result["input"]) == 1
    user_message = result["input"][0]
    assert user_message["role"] == "user"

    # then - image first, text second
    assert len(user_message["content"]) == 2
    image_part, text_part = user_message["content"]
    assert image_part["type"] == "input_image"
    assert image_part["image_url"].startswith("data:image/png;base64,")
    assert "detail" not in image_part
    assert text_part["type"] == "input_text"

    # then - box_2d yxyx 0-1000 prompt with class list
    prompt_text = text_part["text"]
    assert '"box_2d"' in prompt_text
    assert '"label"' in prompt_text
    assert "[y_min, x_min, y_max, x_max]" in prompt_text
    assert "between 0 and 1000" in prompt_text
    assert "Return only the JSON list, with no extra text." in prompt_text
    assert "Only use these labels: person, car" in prompt_text


def test_encode_image_for_task_produces_png_for_object_detection() -> None:
    # given
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    # when
    result = encode_image_for_task(image, task_type="object-detection")

    # then
    decoded_bytes = base64.b64decode(result)
    assert decoded_bytes.startswith(PNG_MAGIC_BYTES)


def test_encode_image_for_task_produces_jpeg_for_other_tasks() -> None:
    # given
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    # when
    result = encode_image_for_task(image, task_type="caption")

    # then
    decoded_bytes = base64.b64decode(result)
    assert decoded_bytes.startswith(JPEG_MAGIC_BYTES)


def test_encode_image_for_task_downscales_large_image_for_object_detection() -> None:
    # given
    image = np.zeros((3000, 4000, 3), dtype=np.uint8)

    # when
    result = encode_image_for_task(image, task_type="object-detection")

    # then
    decoded_bytes = base64.b64decode(result)
    decoded_image = cv2.imdecode(
        np.frombuffer(decoded_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    height, width = decoded_image.shape[:2]
    assert width == DETECTION_MAX_EDGE_PIXELS
    assert height == round(3000 * DETECTION_MAX_EDGE_PIXELS / 4000)


def test_encode_image_for_task_downscales_portrait_image_for_object_detection() -> None:
    # given
    image = np.zeros((4000, 3000, 3), dtype=np.uint8)

    # when
    result = encode_image_for_task(image, task_type="object-detection")

    # then
    decoded_bytes = base64.b64decode(result)
    decoded_image = cv2.imdecode(
        np.frombuffer(decoded_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    height, width = decoded_image.shape[:2]
    assert height == DETECTION_MAX_EDGE_PIXELS
    assert width == round(3000 * DETECTION_MAX_EDGE_PIXELS / 4000)


def test_encode_image_for_task_does_not_upscale_small_image_for_object_detection() -> (
    None
):
    # given
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    # when
    result = encode_image_for_task(image, task_type="object-detection")

    # then
    decoded_bytes = base64.b64decode(result)
    decoded_image = cv2.imdecode(
        np.frombuffer(decoded_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    height, width = decoded_image.shape[:2]
    assert (height, width) == (480, 640)


def test_encode_image_for_task_does_not_resize_large_image_for_other_tasks() -> None:
    # given
    image = np.zeros((3000, 4000, 3), dtype=np.uint8)

    # when
    result = encode_image_for_task(image, task_type="caption")

    # then
    decoded_bytes = base64.b64decode(result)
    decoded_image = cv2.imdecode(
        np.frombuffer(decoded_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    height, width = decoded_image.shape[:2]
    assert (height, width) == (3000, 4000)
