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

LEGACY_CONTROL_MODELS = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "claude-opus-4-1",
    "claude-opus-4",
    "claude-sonnet-4",
]

ADAPTIVE_ONLY_MODELS = [
    "claude-fable-5-1",
    "claude-fable-5",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
]


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
        ("anthropic.claude-fable-5-1", "anthropic-claude-fable-5-1"),
        ("claude-3-5-sonnet-v2", "claude-3-5-sonnet-v2"),
    ],
)
def test_normalize_anthropic_model_id(raw: str, expected: str) -> None:
    assert normalize_anthropic_model_id(raw) == expected


@pytest.mark.parametrize("model_version", LEGACY_CONTROL_MODELS)
def test_legacy_models_keep_temperature_and_manual_thinking(
    model_version: str,
) -> None:
    assert anthropic_model_supports_temperature(model_version) is True
    assert anthropic_model_supports_manual_thinking(model_version) is True


@pytest.mark.parametrize("model_version", ADAPTIVE_ONLY_MODELS)
def test_new_generation_models_reject_temperature_and_manual_thinking(
    model_version: str,
) -> None:
    assert anthropic_model_supports_temperature(model_version) is False
    assert anthropic_model_supports_manual_thinking(model_version) is False


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


def test_v1_legacy_labels_that_alias_sonnet_4_5_keep_temperature() -> None:
    for label in [
        "claude-3-5-sonnet",
        "claude-3-5-sonnet-v2",
        "claude-4-5-sonnet",
        "claude-4-5-sonnet-v2",
        "claude-3-7-sonnet",
        "claude-3-opus",
    ]:
        assert anthropic_model_supports_temperature(label) is True, label


def test_resolve_temperature_passes_value_through_for_legacy_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        result = resolve_temperature(0.3, model_version="claude-sonnet-4-5")

    assert result == 0.3
    assert caplog.records == []


def test_resolve_temperature_returns_none_when_not_configured() -> None:
    assert resolve_temperature(None, model_version="claude-sonnet-4-5") is None
    assert resolve_temperature(None, model_version="claude-fable-5-1") is None


def test_resolve_temperature_drops_value_when_thinking_enabled_on_legacy_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        result = resolve_temperature(
            0.3, model_version="claude-sonnet-4-5", extended_thinking=True
        )

    # Anthropic forbids temperature with thinking; this is not a model
    # capability gap so no warning is expected.
    assert result is None
    assert caplog.records == []


def test_resolve_temperature_drops_value_and_warns_for_new_generation_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        result = resolve_temperature(0.3, model_version="claude-fable-5-1")

    assert result is None
    assert len(caplog.records) == 1
    assert "claude-fable-5-1" in caplog.records[0].getMessage()
    assert "temperature" in caplog.records[0].getMessage()


def test_resolve_temperature_warns_once_per_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        resolve_temperature(0.3, model_version="claude-fable-5-1")
        resolve_temperature(0.9, model_version="claude-fable-5-1")
        resolve_temperature(0.3, model_version="claude-opus-5")

    warned_models = [r.args[0] for r in caplog.records]
    assert warned_models == ["claude-fable-5-1", "claude-opus-5"]


def test_build_thinking_config_returns_none_when_thinking_disabled() -> None:
    for extended_thinking in (None, False):
        assert (
            build_thinking_config(
                extended_thinking=extended_thinking,
                thinking_budget_tokens=4096,
                model_version="claude-sonnet-4-5",
                model_max_output=64000,
            )
            is None
        )


def test_build_thinking_config_uses_manual_budget_for_legacy_model() -> None:
    result = build_thinking_config(
        extended_thinking=True,
        thinking_budget_tokens=5000,
        model_version="claude-sonnet-4-5",
        model_max_output=64000,
    )

    assert result == {"type": "enabled", "budget_tokens": 5000}


def test_build_thinking_config_defaults_budget_to_half_of_output_ceiling() -> None:
    result = build_thinking_config(
        extended_thinking=True,
        thinking_budget_tokens=None,
        model_version="claude-sonnet-4-5",
        model_max_output=64000,
    )

    assert result == {"type": "enabled", "budget_tokens": 32000}


@pytest.mark.parametrize("model_version", ADAPTIVE_ONLY_MODELS)
def test_build_thinking_config_requests_adaptive_thinking_for_new_generation(
    model_version: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        result = build_thinking_config(
            extended_thinking=True,
            thinking_budget_tokens=None,
            model_version=model_version,
            model_max_output=128000,
        )

    assert result == {"type": "adaptive"}
    # no budget was configured, so nothing was silently dropped
    assert caplog.records == []


def test_build_thinking_config_ignores_budget_with_warning_for_adaptive_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        result = build_thinking_config(
            extended_thinking=True,
            thinking_budget_tokens=5000,
            model_version="claude-fable-5-1",
            model_max_output=128000,
        )
        build_thinking_config(
            extended_thinking=True,
            thinking_budget_tokens=7000,
            model_version="claude-fable-5-1",
            model_max_output=128000,
        )

    assert result == {"type": "adaptive"}
    assert "budget_tokens" not in result
    assert len(caplog.records) == 1
    assert "thinking_budget_tokens=5000" in caplog.records[0].getMessage()
