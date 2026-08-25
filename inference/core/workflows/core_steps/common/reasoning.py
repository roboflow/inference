"""Shared reasoning/thinking-level contract for VLM blocks.

Extend it by adding a model-table row (empty ``reasoning_levels`` = no knob),
a value to ``REASONING_EFFORT_OPTIONS`` + metadata, or a new block that
validates with ``validate_reasoning_level``, attaches metadata with
``attach_reasoning_levels``, and registers in ``test_reasoning_contract.py``.
"""

from typing import Dict, List, Optional

# Full OpenRouter platform enum (https://openrouter.ai/docs/api/reference/parameters).
# OpenRouter normalizes the value to the nearest level each model supports.
REASONING_EFFORT_OPTIONS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]

REASONING_EFFORT_METADATA = {
    "none": {
        "name": "Disabled",
        "description": (
            "Explicitly turns extended reasoning off. Models that reject "
            "`reasoning: {enabled: false}` are retried without the config "
            "and keep their provider-default behavior."
        ),
    },
    "minimal": {
        "name": "Minimal",
        "description": (
            "Smallest reasoning budget above off. Models without a distinct "
            "minimal level are normalized to the nearest supported level."
        ),
    },
    "low": {
        "name": "Low",
        "description": "Small reasoning budget before answering.",
    },
    "medium": {
        "name": "Medium",
        "description": "Moderate reasoning budget before answering.",
    },
    "high": {
        "name": "High",
        "description": (
            "Large reasoning budget. Slowest and most expensive; consider "
            "raising `max_tokens` so reasoning does not crowd out the answer."
        ),
    },
    "xhigh": {
        "name": "Extra high",
        "description": (
            "Maximum reasoning budget on most models. Only some models "
            "support it (e.g. the GPT-5.2+ generation); models that reject "
            "it are retried without a reasoning config. Raise `max_tokens` "
            "accordingly."
        ),
    },
    "max": {
        "name": "Max",
        "description": (
            "Largest reasoning budget where available (e.g. the GPT-5.6 "
            "generation); elsewhere normalized down to the nearest supported "
            "level. Raise `max_tokens` accordingly."
        ),
    },
}


def validate_reasoning_level(
    model: Optional[str],
    level: Optional[str],
    levels_by_model: Dict[str, List[str]],
    parameter: str = "reasoning_effort",
) -> None:
    """Reject a level the selected model does not natively support.

    Unset values, selectors, and models absent from the table all pass so
    existing workflows and future selector-supplied models keep working.
    """
    if model is None or level is None:
        return
    if _is_selector(model) or _is_selector(level):
        return
    if model not in levels_by_model:
        return
    supported = levels_by_model[model]
    if level in supported:
        return
    if supported:
        raise ValueError(
            f"Model {model} supports {parameter} values {supported}, " f"got {level!r}."
        )
    raise ValueError(
        f"Model {model} does not support configurable {parameter} " f"(got {level!r})."
    )


def models_supporting_reasoning(
    levels_by_model: Dict[str, List[str]],
) -> List[str]:
    """List models with a non-empty reasoning-level set, preserving order."""
    return [model for model, levels in levels_by_model.items() if levels]


def attach_reasoning_levels(
    values_metadata: Dict[str, dict],
    levels_by_model: Dict[str, List[str]],
) -> Dict[str, dict]:
    """Copy ``values_metadata`` and attach each model's ``reasoning_levels``."""
    return {
        model: {**metadata, "reasoning_levels": levels_by_model.get(model, [])}
        for model, metadata in values_metadata.items()
    }


def build_openrouter_reasoning_config(
    reasoning_effort: Optional[str],
) -> Optional[dict]:
    """Translate the shared ``reasoning_effort`` value into OpenRouter's config.

    ``None`` (field unset) omits the config entirely so the model keeps its
    provider-default behavior — this is what shipped block versions without
    the knob always did. ``"none"`` explicitly disables reasoning; the
    remaining efforts map to ``{"effort": ...}``. Models that reject the
    config are retried without it by the shared executor.

    Args:
        reasoning_effort: ``None`` or one of ``REASONING_EFFORT_OPTIONS``.

    Returns:
        OpenRouter ``reasoning`` payload object, or ``None`` to omit it.
    """
    if reasoning_effort is None:
        return None
    if reasoning_effort == "none":
        return {"enabled": False}
    return {"effort": reasoning_effort}


def _is_selector(value: str) -> bool:
    return isinstance(value, str) and value.startswith("$")
