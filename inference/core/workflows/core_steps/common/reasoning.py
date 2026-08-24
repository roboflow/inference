"""Shared reasoning-effort control for OpenRouter-routed VLM blocks.

Serves the generic blocks (``openrouter@v2``, ``google_gemma@v3``) that accept
arbitrary or non-reasoning models. Model-family blocks (Qwen, Muse) keep their
own builders because they encode family-specific behavior: Qwen disables
reasoning by default and knows which variants reject ``enabled: false``; Muse
models require reasoning and have no "off" setting.
"""

from typing import Optional

REASONING_EFFORT_OPTIONS = ["none", "low", "medium", "high", "xhigh"]

REASONING_EFFORT_METADATA = {
    "none": {
        "name": "Disabled",
        "description": (
            "Explicitly turns extended reasoning off. Models that reject "
            "`reasoning: {enabled: false}` are retried without the config "
            "and keep their provider-default behavior."
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
            "Maximum reasoning budget. Only some models support it (e.g. the "
            "GPT-5.2+ generation); models that reject it are retried without "
            "a reasoning config. Raise `max_tokens` accordingly."
        ),
    },
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
