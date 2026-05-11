from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.common.openrouter import (
    execute_openrouter_requests,
)
from inference.core.workflows.core_steps.models.foundation.openrouter.v1 import (
    CUSTOM_MODEL_CHOICE,
    BlockManifest,
    OpenRouterBlockV1,
    resolve_model_id,
)


def test_openrouter_step_validation_valid_input() -> None:
    specification = {
        "type": "roboflow_core/openrouter@v1",
        "name": "step_1",
        "model_version": "Qwen 3.6 35B A3B - OpenRouter",
        "prompt": "Describe what you see.",
    }
    result = BlockManifest.model_validate(specification)
    assert result.type == "roboflow_core/openrouter@v1"
    assert result.api_key == "rf_key:account"
    assert result.model_version == "Qwen 3.6 35B A3B - OpenRouter"
    assert result.privacy_level == "deny"


def test_openrouter_step_validation_accepts_custom_model_slug() -> None:
    specification = {
        "type": "roboflow_core/openrouter@v1",
        "name": "step_1",
        "model_version": CUSTOM_MODEL_CHOICE,
        "custom_model_slug": "qwen/qwen3.6-35b-a3b",
        "prompt": "Describe what you see.",
    }
    result = BlockManifest.model_validate(specification)
    assert result.model_version == CUSTOM_MODEL_CHOICE
    assert result.custom_model_slug == "qwen/qwen3.6-35b-a3b"


def test_openrouter_step_validation_rejects_custom_model_without_slug() -> None:
    specification = {
        "type": "roboflow_core/openrouter@v1",
        "name": "step_1",
        "model_version": CUSTOM_MODEL_CHOICE,
        "prompt": "Describe what you see.",
    }
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(specification)


def test_resolve_model_id_for_known_model() -> None:
    result = resolve_model_id(
        model_version="Kimi K2.6 - OpenRouter",
        custom_model_slug=None,
    )
    assert result == "moonshotai/kimi-k2.6"


def test_resolve_model_id_for_custom_model_slug() -> None:
    result = resolve_model_id(
        model_version=CUSTOM_MODEL_CHOICE,
        custom_model_slug="anthropic/claude-sonnet-4.5",
    )
    assert result == "anthropic/claude-sonnet-4.5"


@patch("inference.core.workflows.core_steps.common.openrouter.post_to_roboflow_api")
def test_openrouter_block_routes_managed_key_through_proxy(
    post_to_roboflow_api_mock: MagicMock,
) -> None:
    post_to_roboflow_api_mock.return_value = {
        "choices": [{"message": {"content": "The model response"}}],
        "usage": {"cost": 0.0001},
    }
    block = OpenRouterBlockV1(api_key="rf-api-key")

    result = block.run(
        api_key="rf_key:account",
        model_version="Qwen 3.6 35B A3B - OpenRouter",
        custom_model_slug=None,
        system_prompt="You are helpful.",
        prompt="Count {{ $parameters.object_type }}.",
        prompt_parameters={"object_type": "cars"},
        prompt_parameters_operations={},
        privacy_level="deny",
        max_tokens=100,
        temperature=0.2,
        max_concurrent_requests=None,
    )

    assert result["output"] == "The model response"
    assert result["error_status"] == ""
    post_to_roboflow_api_mock.assert_called_once()
    call_kwargs = post_to_roboflow_api_mock.call_args.kwargs
    assert call_kwargs["endpoint"] == "apiproxy/openrouter"
    assert call_kwargs["api_key"] == "rf-api-key"
    assert call_kwargs["payload"] == {
        "openrouter_api_key": "rf_key:account",
        "model": "qwen/qwen3.6-35b-a3b",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "user",
                "content": [{"type": "text", "text": "Count cars."}],
            },
        ],
        "max_tokens": 100,
        "temperature": 0.2,
        "privacy_level": "deny",
    }


def test_openrouter_requests_reject_user_key_reference() -> None:
    with pytest.raises(ValueError, match="User-stored OpenRouter API keys"):
        execute_openrouter_requests(
            roboflow_api_key="rf-api-key",
            openrouter_api_key="rf_key:user:abc",
            prompts=[[{"role": "user", "content": "hi"}]],
            model_version_id="qwen/qwen3.6-35b-a3b",
            max_tokens=100,
            temperature=None,
            max_concurrent_requests=None,
        )
