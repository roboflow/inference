"""Per-model request capabilities shared by the Anthropic Claude block versions.

Starting with Claude Opus 4.7, Anthropic rejects ``temperature`` and manual
extended thinking (``thinking.type = "enabled"``) with HTTP 400; thinking on
those models is ``adaptive``. The allow-list below names the models that still
accept the legacy controls, so unknown or future ids default to the new
behaviour. Keep it in sync with ``TEMPERATURE_SUPPORTED_MODELS`` in the
Roboflow API proxy (``app/functions/services/anthropicProxy``).
"""

import re
from typing import Dict, FrozenSet, Optional, Set

from inference.core import logger

_MODELS_WITH_LEGACY_CONTROLS: FrozenSet[str] = frozenset(
    {
        "claude-opus-4-6",
        "claude-opus-4-5",
        "claude-opus-4-1",
        "claude-opus-4",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-sonnet-4",
        "claude-haiku-4-5",
        "claude-3-7-sonnet",
        "claude-3-5-haiku",
        "claude-3-opus",
        "claude-3-haiku",
        # v1 alias labels that resolve to Sonnet 4.5 on the wire
        "claude-3-5-sonnet",
        "claude-3-5-sonnet-v2",
        "claude-4-5-sonnet",
        "claude-4-5-sonnet-v2",
    }
)

TEMPERATURE_SUPPORTED_MODELS: FrozenSet[str] = _MODELS_WITH_LEGACY_CONTROLS
"""Models that still accept a non-default ``temperature``."""

MIN_THINKING_BUDGET_TOKENS = 1024

MANUAL_THINKING_SUPPORTED_MODELS: FrozenSet[str] = _MODELS_WITH_LEGACY_CONTROLS
"""Models that still accept ``thinking.type = "enabled"`` with a token budget."""

_DATED_SNAPSHOT_SUFFIX = re.compile(r"-\d{8}$")
_LATEST_SUFFIX = re.compile(r"-latest$")
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")

_TEMPERATURE_WARNINGS_EMITTED: Set[str] = set()
_THINKING_BUDGET_WARNINGS_EMITTED: Set[str] = set()


def normalize_anthropic_model_id(model_version: str) -> str:
    """Strip dated snapshot / ``-latest`` suffixes and slugify, e.g.
    ``claude-sonnet-4-5-20250929`` -> ``claude-sonnet-4-5``."""
    slug = _NON_ALPHANUMERIC.sub("-", model_version.strip().lower()).strip("-")
    slug = _LATEST_SUFFIX.sub("", slug)
    return _DATED_SNAPSHOT_SUFFIX.sub("", slug)


def anthropic_model_supports_temperature(model_version: str) -> bool:
    """False for Claude Opus 4.7 and newer and for any unknown id."""
    return normalize_anthropic_model_id(model_version) in TEMPERATURE_SUPPORTED_MODELS


def anthropic_model_supports_manual_thinking(model_version: str) -> bool:
    """False for Claude Opus 4.7 and newer and for any unknown id."""
    return (
        normalize_anthropic_model_id(model_version) in MANUAL_THINKING_SUPPORTED_MODELS
    )


def resolve_temperature(
    temperature: Optional[float],
    *,
    model_version: str,
    extended_thinking: Optional[bool] = None,
) -> Optional[float]:
    """Decide which ``temperature`` value, if any, to put on the request.

    Mirrors the Roboflow proxy: the value is dropped when thinking is enabled
    (Anthropic forbids the combination) and when the model no longer accepts
    sampling parameters, so the same workflow behaves identically with a
    customer key and with ``rf_key:account``. Dropping a user-provided value
    logs a warning once per model per process.

    Args:
        temperature: Value configured on the block, or ``None``.
        model_version: Model label or wire id as configured on the block.
        extended_thinking: Whether the block requested thinking.

    Returns:
        The temperature to send, or ``None`` when it must be omitted.
    """
    if temperature is None or extended_thinking:
        return None

    if anthropic_model_supports_temperature(model_version):
        return temperature

    normalized_model = normalize_anthropic_model_id(model_version)
    if normalized_model not in _TEMPERATURE_WARNINGS_EMITTED:
        _TEMPERATURE_WARNINGS_EMITTED.add(normalized_model)
        logger.warning(
            "Anthropic model `%s` does not accept the `temperature` parameter "
            "(Claude Opus 4.7 and newer, Sonnet 5, Opus 5 and Fable models reject "
            "non-default sampling parameters); ignoring temperature=%s.",
            model_version,
            temperature,
        )

    return None


def build_thinking_config(
    *,
    extended_thinking: Optional[bool],
    thinking_budget_tokens: Optional[int],
    model_version: str,
    max_tokens: int,
) -> Optional[Dict[str, object]]:
    """Build the ``thinking`` request block appropriate for a Claude model.

    Models that still support manual extended thinking receive
    ``{"type": "enabled", "budget_tokens": N}`` where ``N`` defaults to half
    of the request's ``max_tokens`` (Anthropic requires
    ``1024 <= budget_tokens < max_tokens``), mirroring the Roboflow proxy.
    Models whose thinking is adaptive only receive ``{"type": "adaptive"}``;
    a configured budget is ignored there and a warning is logged once per
    model per process.

    Args:
        extended_thinking: Whether the block requested thinking.
        thinking_budget_tokens: Budget configured on the block, or ``None``.
        model_version: Model label or wire id as configured on the block.
        max_tokens: Output-token limit that will be sent on the request.

    Returns:
        The ``thinking`` payload, or ``None`` when thinking was not requested.
    """
    if not extended_thinking:
        return None

    if anthropic_model_supports_manual_thinking(model_version):
        effective_budget = (
            thinking_budget_tokens
            if thinking_budget_tokens is not None
            else max(MIN_THINKING_BUDGET_TOKENS, max_tokens // 2)
        )
        return {"type": "enabled", "budget_tokens": effective_budget}

    normalized_model = normalize_anthropic_model_id(model_version)
    if (
        thinking_budget_tokens is not None
        and normalized_model not in _THINKING_BUDGET_WARNINGS_EMITTED
    ):
        _THINKING_BUDGET_WARNINGS_EMITTED.add(normalized_model)
        logger.warning(
            "Anthropic model `%s` only supports adaptive thinking; ignoring "
            "thinking_budget_tokens=%s and requesting `thinking.type=adaptive`.",
            model_version,
            thinking_budget_tokens,
        )

    return {"type": "adaptive"}
