"""Contracts for shared token-usage parsing and VLM block outputs."""

import importlib
from types import SimpleNamespace

from inference.core.workflows.core_steps.common.token_usage import (
    TOKEN_OUTPUT_DEFINITIONS,
    as_optional_int,
    parse_chat_completion_usage,
    parse_gemini_usage_metadata,
    parse_responses_api_usage,
)

_FOUNDATION = "inference.core.workflows.core_steps.models.foundation"

# New versions introduced for token usage: must declare the token outputs.
NEW_BLOCK_MODULES = [
    f"{_FOUNDATION}.openrouter.v2",
    f"{_FOUNDATION}.google_gemma.v3",
    f"{_FOUNDATION}.meta_vlm.v2",
    f"{_FOUNDATION}.qwen_vlm.v3",
    f"{_FOUNDATION}.anthropic_claude.v4",
    f"{_FOUNDATION}.anthropic_claude.v5",
    f"{_FOUNDATION}.google_gemini.v5",
    f"{_FOUNDATION}.openai.v6",
    f"{_FOUNDATION}.spacexai.v2",
]

# Shipped predecessors: their output contracts must stay frozen.
OLD_BLOCK_MODULES = [
    f"{_FOUNDATION}.openrouter.v1",
    f"{_FOUNDATION}.google_gemma.v2",
    f"{_FOUNDATION}.meta_vlm.v1",
    f"{_FOUNDATION}.qwen_vlm.v2",
    f"{_FOUNDATION}.anthropic_claude.v3",
    f"{_FOUNDATION}.google_gemini.v4",
    f"{_FOUNDATION}.openai.v4",
    f"{_FOUNDATION}.openai.v5",
    f"{_FOUNDATION}.spacexai.v1",
]


def _manifest(module_path: str):
    return importlib.import_module(module_path).BlockManifest


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


def test_token_outputs_declared_on_new_versions_and_absent_on_old():
    token_outputs = {definition.name for definition in TOKEN_OUTPUT_DEFINITIONS}
    for module_path in NEW_BLOCK_MODULES:
        names = {
            definition.name for definition in _manifest(module_path).describe_outputs()
        }
        assert token_outputs <= names, f"{module_path} must declare token outputs"
    for module_path in OLD_BLOCK_MODULES:
        names = {
            definition.name for definition in _manifest(module_path).describe_outputs()
        }
        assert not (
            token_outputs & names
        ), f"{module_path} is shipped — its output contract must stay frozen"


def test_new_vlm_block_versions_are_registered():
    from inference.core.workflows.core_steps.loader import load_blocks

    registered_manifests = {block.get_manifest() for block in load_blocks()}
    for module_path in NEW_BLOCK_MODULES:
        assert (
            _manifest(module_path) in registered_manifests
        ), f"{module_path} block is not registered in the loader"
