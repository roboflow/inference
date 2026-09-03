import logging

import pytest

from inference.core.workflows.core_steps.models.foundation.anthropic_claude import (
    model_capabilities,
)
from inference.core.workflows.core_steps.models.foundation.anthropic_claude.model_capabilities import (
    anthropic_model_supports_manual_thinking,
    anthropic_model_supports_temperature,
    build_thinking_config,
    normalize_anthropic_model_id,
    resolve_temperature,
)
from inference.core.workflows.core_steps.models.foundation.anthropic_claude.v1 import (
    EXACT_MODELS_VERSIONS_MAPPING as EXACT_MODEL_VERSIONS_V1,
)
from inference.core.workflows.core_steps.models.foundation.anthropic_claude.v4 import (
    EXACT_MODEL_VERSIONS as EXACT_MODEL_VERSIONS_V4,
)


@pytest.fixture(autouse=True)
def reset_warning_dedup() -> None:
    model_capabilities._TEMPERATURE_WARNINGS_EMITTED.clear()
    model_capabilities._THINKING_BUDGET_WARNINGS_EMITTED.clear()
    yield
    model_capabilities._TEMPERATURE_WARNINGS_EMITTED.clear()
    model_capabilities._THINKING_BUDGET_WARNINGS_EMITTED.clear()


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("claude-sonnet-4-5", "claude-sonnet-4-5"),
        ("claude-sonnet-4-5-20250929", "claude-sonnet-4-5"),
        ("claude-opus-4-5-latest", "claude-opus-4-5"),
        ("  Claude-Fable-5-1 ", "claude-fable-5-1"),
        ("claude_sonnet_4_5", "claude-sonnet-4-5"),
        ("claude-3-5-sonnet-v2", "claude-3-5-sonnet-v2"),
    ],
)
def test_normalize_anthropic_model_id(raw: str, expected: str) -> None:
    assert normalize_anthropic_model_id(raw) == expected


def test_unknown_model_defaults_to_new_generation_behaviour() -> None:
    assert anthropic_model_supports_temperature("claude-something-9") is False
    assert anthropic_model_supports_manual_thinking("claude-something-9") is False


def test_dated_wire_ids_used_by_block_versions_are_classified_like_labels() -> None:
    # The block may hand the helper either the friendly label or the exact wire
    # id; both spellings must agree for every model shipped in the blocks.
    for label, wire_id in {
        **EXACT_MODEL_VERSIONS_V1,
        **EXACT_MODEL_VERSIONS_V4,
    }.items():
        assert anthropic_model_supports_temperature(
            label
        ) == anthropic_model_supports_temperature(wire_id), (label, wire_id)


def test_resolve_temperature_drops_value_for_new_generation_and_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        assert resolve_temperature(0.3, model_version="claude-sonnet-4-5") == 0.3
        assert resolve_temperature(0.3, model_version="claude-fable-5-1") is None
        assert resolve_temperature(0.9, model_version="claude-fable-5-1") is None
        assert (
            resolve_temperature(0.9, model_version="claude-fable-5-1-20260901") is None
        )
        assert resolve_temperature(0.3, model_version="claude-opus-5") is None

    warned_models = [record.args[0] for record in caplog.records]
    assert warned_models == ["claude-fable-5-1", "claude-opus-5"]
    assert "temperature" in caplog.records[0].getMessage()


def test_build_thinking_config_uses_adaptive_and_ignores_budget_with_single_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        first = build_thinking_config(
            extended_thinking=True,
            thinking_budget_tokens=5000,
            model_version="claude-fable-5-1",
            max_tokens=128000,
        )
        second = build_thinking_config(
            extended_thinking=True,
            thinking_budget_tokens=7000,
            model_version="claude-fable-5-1-20260901",
            max_tokens=128000,
        )

    assert first == {"type": "adaptive"}
    assert second == {"type": "adaptive"}
    assert len(caplog.records) == 1
    assert "thinking_budget_tokens=5000" in caplog.records[0].getMessage()


@pytest.mark.parametrize(
    "max_tokens, expected_budget",
    [
        (64000, 32000),
        (6000, 3000),
        (1500, 1024),
    ],
)
def test_build_thinking_config_derives_default_budget_below_request_max_tokens(
    max_tokens: int, expected_budget: int
) -> None:
    result = build_thinking_config(
        extended_thinking=True,
        thinking_budget_tokens=None,
        model_version="claude-opus-4-6",
        max_tokens=max_tokens,
    )

    assert result == {"type": "enabled", "budget_tokens": expected_budget}
