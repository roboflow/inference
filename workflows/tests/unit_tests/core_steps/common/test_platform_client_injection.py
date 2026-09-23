"""Every block that reaches the Roboflow proxy asks for - and uses - the port.

`get_init_parameters()` is the block's dependency declaration: if a block drops
`platform_client`, `steps_initialiser` never passes one and the block silently
falls back to the offline default that refuses every call. The Qwen blocks are
here because they override both the constructor and the declaration of the
OpenRouter base.

The package-wide "no workflows module imports the Roboflow proxy helper" checks
that used to live here are subsumed by the centralized zero-violations
decontamination lint (`tests/workflows/unit_tests/test_decontamination_lint.py`),
which scans every `inference.*` import under `inference/core/workflows`, not
just this one module.
"""

import importlib
import inspect

import pytest

from tests.unit_tests.prototypes.platform_client_double import RecordingPlatformClient

PREFIX = "roboflow_workflows.core_steps"
PROXY_BLOCKS = [
    (".models.foundation.openai.v3", "OpenAIBlockV3"),
    (".models.foundation.openai.v4", "OpenAIBlockV4"),
    (".models.foundation.openai.v5", "OpenAIBlockV5"),
    (".models.foundation.openai.v6", "OpenAIBlockV6"),
    (".models.foundation.openai.v7", "OpenAIBlockV7"),
    (".models.foundation.google_gemini.v3", "GoogleGeminiBlockV3"),
    (".models.foundation.google_gemini.v4", "GoogleGeminiBlockV4"),
    (".models.foundation.google_gemini.v5", "GoogleGeminiBlockV5"),
    (".models.foundation.google_gemini.v6", "GoogleGeminiBlockV6"),
    (".models.foundation.anthropic_claude.v3", "AnthropicClaudeBlockV3"),
    (".models.foundation.anthropic_claude.v4", "AnthropicClaudeBlockV4"),
    (".models.foundation.anthropic_claude.v5", "AnthropicClaudeBlockV5"),
    (".models.foundation.spacexai.v1", "SpaceXAIBlockV1"),
    (".models.foundation.spacexai.v2", "SpaceXAIBlockV2"),
    (".models.foundation.spacexai.v3", "SpaceXAIBlockV3"),
    (".models.foundation.google_vision_ocr.v1", "GoogleVisionOCRBlockV1"),
    (".models.foundation.google_vision_ocr.v1_tensor", "GoogleVisionOCRBlockV1"),
    (".sinks.email_notification.v2", "EmailNotificationBlockV2"),
    (".sinks.twilio.sms.v2", "TwilioSMSNotificationBlockV2"),
    (".common.openrouter", "OpenRouterWorkflowBlockBase"),
    (".models.foundation.qwen_vlm.v1", "QwenVlmBlockV1"),
    (".models.foundation.qwen_vlm.v2", "QwenVlmBlockV2"),
    (".models.foundation.qwen_vlm.v3", "QwenVlmBlockV3"),
    (".models.foundation.qwen_vlm.v4", "QwenVlmBlockV4"),
]


def _load(module_suffix: str, class_name: str):
    return getattr(importlib.import_module(PREFIX + module_suffix), class_name)


@pytest.mark.parametrize("module_suffix,class_name", PROXY_BLOCKS)
def test_block_declares_the_platform_client_init_parameter(module_suffix, class_name):
    assert "platform_client" in _load(module_suffix, class_name).get_init_parameters()


@pytest.mark.parametrize("module_suffix,class_name", PROXY_BLOCKS)
def test_block_stores_the_injected_client(module_suffix, class_name):
    block_class = _load(module_suffix, class_name)
    if inspect.isabstract(block_class):
        pytest.skip(
            f"{class_name} is abstract; storage is proven through QwenVlmBlockV1"
        )
    sentinel = RecordingPlatformClient()
    kwargs = {name: None for name in block_class.get_init_parameters()}
    kwargs["platform_client"] = sentinel
    assert block_class(**kwargs)._platform_client is sentinel


def test_openai_managed_key_path_uses_the_injected_client() -> None:
    """The whole chain: block -> partial -> thread pool ->
    `_execute_proxied_openai_request` -> `platform_client.post`."""
    from roboflow_workflows.core_steps.models.foundation.openai import v3

    client = RecordingPlatformClient(
        post_response={"choices": [{"message": {"content": "ok"}}]}
    )
    result = v3.execute_gpt_4v_requests(
        roboflow_api_key="rf-key",
        platform_client=client,
        openai_api_key="rf_key:account:abc",
        gpt4_prompts=[[{"role": "user", "content": "hi"}]],
        gpt_model_version="gpt-4o",
        max_tokens=10,
        temperature=None,
        max_concurrent_requests=1,
    )
    assert client.posts, "the managed-key path never reached the injected client"
    assert result


@pytest.mark.parametrize(
    "module_suffix,class_name,model_version",
    [
        (".models.foundation.openai.v7", "OpenAIBlockV7", "gpt-5.1"),
        (
            ".models.foundation.anthropic_claude.v5",
            "AnthropicClaudeBlockV5",
            "claude-sonnet-4-5",
        ),
        (
            ".models.foundation.google_gemini.v6",
            "GoogleGeminiBlockV6",
            "gemini-2.5-flash",
        ),
        (".models.foundation.spacexai.v3", "SpaceXAIBlockV3", "grok-4"),
    ],
)
def test_new_vlm_blocks_forward_managed_keys_through_the_injected_client(
    module_suffix, class_name, model_version
):
    import numpy as np
    from roboflow_workflows.execution_engine.entities.base import (
        Batch,
        ImageParentMetadata,
        WorkflowImageData,
    )

    client = RecordingPlatformClient(
        post_response={
            "content": [{"type": "text", "text": "ok"}],
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "ok"}]}
            ],
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
        }
    )
    block = _load(module_suffix, class_name)(
        api_key="workspace-key", platform_client=client
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="image"),
        numpy_image=np.zeros((8, 8, 3), dtype=np.uint8),
    )
    kwargs = {
        name: (
            parameter.default
            if parameter.default is not inspect.Parameter.empty
            else None
        )
        for name, parameter in inspect.signature(block.run).parameters.items()
    }
    kwargs.update(
        images=Batch.init(content=[image], indices=[(0,)]),
        task_type="caption",
        model_version=model_version,
        api_key="rf_key:account:managed-key",
        max_concurrent_requests=1,
    )
    if "max_image_size" in kwargs:
        kwargs["max_image_size"] = 512
    if "image_detail" in kwargs:
        kwargs["image_detail"] = "auto"
    result = block.run(**kwargs)
    assert result[0]["output"] == "ok"
    assert len(client.posts) == 1
    assert client.posts[0]["api_key"] == "workspace-key"
