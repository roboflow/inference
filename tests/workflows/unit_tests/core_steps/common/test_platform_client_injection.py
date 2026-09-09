"""Every block that reaches the Roboflow proxy asks for - and uses - the port.

`get_init_parameters()` is the block's dependency declaration: if a block drops
`platform_client`, `steps_initialiser` never passes one and the block silently
falls back to the offline default that refuses every call. The Qwen blocks are
here because they override both the constructor and the declaration of the
OpenRouter base.
"""

import ast
import importlib
import inspect
import pathlib
from functools import partial

import pytest

from tests.workflows.unit_tests.prototypes.platform_client_double import (
    RecordingPlatformClient,
)

PREFIX = "inference.core.workflows.core_steps"
PROXY_BLOCKS = [
    (".models.foundation.openai.v3", "OpenAIBlockV3"),
    (".models.foundation.openai.v4", "OpenAIBlockV4"),
    (".models.foundation.openai.v5", "OpenAIBlockV5"),
    (".models.foundation.openai.v6", "OpenAIBlockV6"),
    (".models.foundation.google_gemini.v3", "GoogleGeminiBlockV3"),
    (".models.foundation.google_gemini.v4", "GoogleGeminiBlockV4"),
    (".models.foundation.google_gemini.v5", "GoogleGeminiBlockV5"),
    (".models.foundation.anthropic_claude.v3", "AnthropicClaudeBlockV3"),
    (".models.foundation.anthropic_claude.v4", "AnthropicClaudeBlockV4"),
    (".models.foundation.spacexai.v1", "SpaceXAIBlockV1"),
    (".models.foundation.spacexai.v2", "SpaceXAIBlockV2"),
    (".models.foundation.google_vision_ocr.v1", "GoogleVisionOCRBlockV1"),
    (".models.foundation.google_vision_ocr.v1_tensor", "GoogleVisionOCRBlockV1"),
    (".sinks.email_notification.v2", "EmailNotificationBlockV2"),
    (".sinks.twilio.sms.v2", "TwilioSMSNotificationBlockV2"),
    (".common.openrouter", "OpenRouterWorkflowBlockBase"),
    (".models.foundation.qwen_vlm.v1", "QwenVlmBlockV1"),
    (".models.foundation.qwen_vlm.v2", "QwenVlmBlockV2"),
    (".models.foundation.qwen_vlm.v3", "QwenVlmBlockV3"),
]
WORKFLOWS_ROOT = (
    pathlib.Path(__file__).resolve().parents[5] / "inference" / "core" / "workflows"
)


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


def test_no_workflows_module_imports_the_roboflow_proxy_helper() -> None:
    """Symbol-specific: 16 files still legitimately import other names from
    `roboflow_api` until Tasks 9.5-9.6 (the 14 header users and the two engine
    files). Only `post_to_roboflow_api` is gone.
    """
    offenders = []
    for path in WORKFLOWS_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "inference.core.roboflow_api"
            ):
                if any(a.name == "post_to_roboflow_api" for a in node.names):
                    offenders.append(f"{path}:{node.lineno}")
    assert not offenders, offenders


def test_concurrent_blocks_keep_their_own_clients() -> None:
    """The client travels inside the `partial`, not in shared state:
    `common/utils.run_in_parallel` hands the partials to a thread pool."""
    from inference.core.workflows.core_steps.common.utils import run_in_parallel

    def helper(roboflow_api_key, platform_client):
        return platform_client

    a, b = RecordingPlatformClient(), RecordingPlatformClient()
    results = run_in_parallel(
        tasks=[
            partial(helper, roboflow_api_key="a", platform_client=a),
            partial(helper, roboflow_api_key="b", platform_client=b),
        ],
        max_workers=2,
    )
    assert results == [a, b]


def test_openai_managed_key_path_uses_the_injected_client() -> None:
    """The whole chain: block -> partial -> thread pool ->
    `_execute_proxied_openai_request` -> `platform_client.post`."""
    from inference.core.workflows.core_steps.models.foundation.openai import v3

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


def test_qwen_forwards_the_client_to_the_openrouter_base() -> None:
    from inference.core.workflows.core_steps.models.foundation.qwen_vlm.v1 import (
        QwenVlmBlockV1,
    )

    sentinel = RecordingPlatformClient()
    block = QwenVlmBlockV1(
        model_manager=None,
        api_key=None,
        step_execution_mode=None,
        platform_client=sentinel,
    )
    assert block._platform_client is sentinel


def test_no_workflows_module_imports_the_roboflow_api_client_at_all() -> None:
    """True from Task 9.6 onward: the last two importers were
    `block_scaffolding.py` and `reference_resolution.py`."""
    offenders = []
    for path in WORKFLOWS_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "inference.core.roboflow_api"
            ):
                offenders.append(f"{path}:{node.lineno}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("inference.core.roboflow_api"):
                        offenders.append(f"{path}:{node.lineno}")
    assert not offenders, offenders
