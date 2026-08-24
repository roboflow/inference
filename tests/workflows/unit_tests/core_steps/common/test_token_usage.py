"""Contracts for shared token-usage parsing and VLM block outputs."""

from types import SimpleNamespace

from inference.core.workflows.core_steps.common.token_usage import (
    TOKEN_OUTPUT_DEFINITIONS,
    as_optional_int,
    parse_chat_completion_usage,
    parse_gemini_usage_metadata,
    parse_responses_api_usage,
)
from inference.core.workflows.core_steps.models.foundation.anthropic_claude.v4 import (
    BlockManifest as ClaudeManifest,
)
from inference.core.workflows.core_steps.models.foundation.google_gemini.v5 import (
    BlockManifest as GeminiManifest,
)
from inference.core.workflows.core_steps.models.foundation.google_gemma.v3 import (
    BlockManifest as GemmaManifest,
)
from inference.core.workflows.core_steps.models.foundation.meta_vlm.v2 import (
    BlockManifest as MetaManifest,
)
from inference.core.workflows.core_steps.models.foundation.openai.v6 import (
    BlockManifest as OpenAIManifest,
)
from inference.core.workflows.core_steps.models.foundation.openrouter.v2 import (
    BlockManifest as OpenRouterManifest,
)
from inference.core.workflows.core_steps.models.foundation.qwen_vlm.v3 import (
    BlockManifest as QwenManifest,
)
from inference.core.workflows.core_steps.models.foundation.spacexai.v2 import (
    BlockManifest as SpaceXAIManifest,
)


def test_as_optional_int_rejects_bool_and_non_ints():
    assert as_optional_int(0) == 0
    assert as_optional_int(12) == 12
    assert as_optional_int(True) is None
    assert as_optional_int(1.5) is None
    assert as_optional_int("9") is None
    assert as_optional_int(None) is None


def test_parse_chat_completion_usage_from_dict_and_object():
    assert parse_chat_completion_usage(None) == (None, None)
    assert parse_chat_completion_usage({}) == (None, None)
    assert parse_chat_completion_usage(
        {"prompt_tokens": 11, "completion_tokens": 7}
    ) == (11, 7)
    assert parse_chat_completion_usage(
        SimpleNamespace(prompt_tokens=3, completion_tokens=1)
    ) == (3, 1)


def test_parse_responses_api_usage_from_dict_and_object():
    assert parse_responses_api_usage(None) == (None, None)
    assert parse_responses_api_usage({"input_tokens": 20, "output_tokens": 8}) == (
        20,
        8,
    )
    assert parse_responses_api_usage(
        SimpleNamespace(input_tokens=4, output_tokens=2)
    ) == (4, 2)


def test_parse_gemini_usage_adds_thoughts_to_output():
    assert parse_gemini_usage_metadata(None) == (None, None)
    assert parse_gemini_usage_metadata({}) == (None, None)
    assert parse_gemini_usage_metadata(
        {
            "promptTokenCount": 15,
            "candidatesTokenCount": 6,
            "thoughtsTokenCount": 4,
        }
    ) == (15, 10)
    assert parse_gemini_usage_metadata({"thoughtsTokenCount": 3}) == (None, 3)


def test_token_outputs_declared_on_all_remote_vlm_blocks():
    expected = {definition.name for definition in TOKEN_OUTPUT_DEFINITIONS}
    for manifest in (
        OpenRouterManifest,
        GemmaManifest,
        MetaManifest,
        QwenManifest,
        ClaudeManifest,
        GeminiManifest,
        OpenAIManifest,
        SpaceXAIManifest,
    ):
        names = {definition.name for definition in manifest.describe_outputs()}
        assert expected <= names, manifest.__module__
