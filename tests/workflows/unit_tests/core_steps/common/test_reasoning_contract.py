"""Per-model reasoning levels: validator edges, table/Literal drift, spec rejects."""

import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.common.reasoning import (
    REASONING_EFFORT_OPTIONS,
    validate_reasoning_level,
)
from inference.core.workflows.core_steps.models.foundation.anthropic_claude import (
    v5 as anthropic_v5,
)
from inference.core.workflows.core_steps.models.foundation.google_gemini import (
    v5 as gemini_v5,
)
from inference.core.workflows.core_steps.models.foundation.meta_vlm import (
    v2 as meta_vlm_v2,
)
from inference.core.workflows.core_steps.models.foundation.openai import v6 as openai_v6
from inference.core.workflows.core_steps.models.foundation.qwen_vlm import (
    v3 as qwen_vlm_v3,
)
from inference.core.workflows.core_steps.models.foundation.spacexai import (
    v2 as spacexai_v2,
)
from inference.core.workflows.core_steps.models.foundation.zai_vlm import (
    v1 as zai_vlm_v1,
)

_LEVELS = {"model-a": ["low", "high"], "model-b": []}


@pytest.mark.parametrize(
    "model, level",
    [
        ("model-a", "low"),
        ("model-a", None),
        (None, "high"),
        ("model-a", "$inputs.effort"),
        ("$inputs.model", "xhigh"),
        ("future-model", "xhigh"),
    ],
)
def test_validate_reasoning_level_accepts(model, level):
    validate_reasoning_level(model=model, level=level, levels_by_model=_LEVELS)


def test_validate_reasoning_level_rejects_unsupported_level():
    with pytest.raises(ValueError, match=r"model-a supports .* \['low', 'high'\]"):
        validate_reasoning_level(
            model="model-a", level="xhigh", levels_by_model=_LEVELS
        )


def test_validate_reasoning_level_rejects_model_without_levels():
    with pytest.raises(
        ValueError, match="does not support configurable thinking_level"
    ):
        validate_reasoning_level(
            model="model-b",
            level="low",
            levels_by_model=_LEVELS,
            parameter="thinking_level",
        )


BLOCK_CONTRACTS = {
    "open_ai@v6": (
        openai_v6.MODEL_REASONING_EFFORT_VALUES,
        openai_v6.REASONING_EFFORT_VALUES,
        openai_v6.MODEL_VERSION_METADATA,
        None,
    ),
    "google_gemini@v5": (
        gemini_v5.MODEL_THINKING_LEVELS,
        gemini_v5.THINKING_LEVEL_VALUES,
        gemini_v5.MODEL_VERSION_METADATA,
        None,
    ),
    "spacexai@v2": (
        spacexai_v2.MODEL_REASONING_LEVELS,
        spacexai_v2.REASONING_EFFORT_VALUES,
        spacexai_v2.MODEL_VERSION_METADATA,
        None,
    ),
    "meta_vlm@v2": (
        meta_vlm_v2.MODEL_REASONING_LEVELS,
        meta_vlm_v2.REASONING_EFFORT_OPTIONS,
        meta_vlm_v2.MODEL_VERSION_METADATA,
        meta_vlm_v2.DEFAULT_REASONING_EFFORT,
    ),
    "qwen_vlm@v3": (
        qwen_vlm_v3.MODEL_REASONING_LEVELS,
        qwen_vlm_v3.REASONING_EFFORT_OPTIONS,
        qwen_vlm_v3.OPENROUTER_MODEL_VERSION_METADATA,
        "none",
    ),
    "zai_vlm@v1": (
        zai_vlm_v1.MODEL_REASONING_LEVELS,
        zai_vlm_v1.REASONING_EFFORT_OPTIONS,
        zai_vlm_v1.MODEL_VERSION_METADATA,
        zai_vlm_v1.DEFAULT_REASONING_EFFORT,
    ),
    "anthropic_claude@v5": (
        anthropic_v5.MODEL_REASONING_EFFORT_VALUES,
        anthropic_v5.REASONING_EFFORT_VALUES,
        anthropic_v5.MODEL_VERSION_METADATA,
        None,
    ),
}


@pytest.mark.parametrize("block", BLOCK_CONTRACTS, ids=BLOCK_CONTRACTS.keys())
def test_block_levels_stay_within_shared_vocabulary(block):
    levels_by_model, manifest_values, _, _ = BLOCK_CONTRACTS[block]
    shared = set(REASONING_EFFORT_OPTIONS)
    for model, levels in levels_by_model.items():
        unknown = set(levels) - shared
        assert not unknown, f"{block}: {model} declares unknown levels {unknown}"
    assert set(manifest_values) <= shared


@pytest.mark.parametrize("block", BLOCK_CONTRACTS, ids=BLOCK_CONTRACTS.keys())
def test_block_manifest_literal_covers_union_of_table_levels(block):
    levels_by_model, manifest_values, _, _ = BLOCK_CONTRACTS[block]
    union = set().union(*levels_by_model.values())
    missing = union - set(manifest_values)
    assert not missing, f"{block}: manifest Literal is missing levels {missing}"


@pytest.mark.parametrize("block", BLOCK_CONTRACTS, ids=BLOCK_CONTRACTS.keys())
def test_block_dropdown_metadata_matches_table(block):
    levels_by_model, _, values_metadata, _ = BLOCK_CONTRACTS[block]
    for model, metadata in values_metadata.items():
        assert (
            metadata["reasoning_levels"] == levels_by_model[model]
        ), f"{block}: dropdown metadata for {model} diverged from the table"
    for model, levels in levels_by_model.items():
        if levels:
            assert model in values_metadata, (
                f"{block}: {model} supports reasoning but is missing from "
                "dropdown metadata"
            )


@pytest.mark.parametrize("block", BLOCK_CONTRACTS, ids=BLOCK_CONTRACTS.keys())
def test_block_default_level_is_valid_for_every_model(block):
    levels_by_model, _, _, default = BLOCK_CONTRACTS[block]
    if default is None:
        return
    for model, levels in levels_by_model.items():
        if levels:
            assert (
                default in levels
            ), f"{block}: default level {default!r} is unsupported by {model}"


_MANIFESTS = {
    "open_ai@v6": openai_v6.BlockManifest,
    "google_gemini@v5": gemini_v5.BlockManifest,
    "spacexai@v2": spacexai_v2.BlockManifest,
    "meta_vlm@v2": meta_vlm_v2.BlockManifest,
    "qwen_vlm@v3": qwen_vlm_v3.BlockManifest,
    "anthropic_claude@v5": anthropic_v5.BlockManifest,
}


def _manifest(block_type: str, **overrides):
    spec = {
        "type": f"roboflow_core/{block_type}",
        "name": "step",
        "images": "$inputs.image",
        "task_type": "unconstrained",
        "prompt": "describe",
    }
    if block_type == "spacexai@v2":
        spec["api_key"] = "$inputs.xai_api_key"
    spec.update(overrides)
    return _MANIFESTS[block_type].model_validate(spec)


@pytest.mark.parametrize(
    "block_type, overrides",
    [
        ("open_ai@v6", {"model_version": "$inputs.model", "reasoning_effort": "xhigh"}),
        ("open_ai@v6", {"model_version": "gpt-4o"}),
        ("qwen_vlm@v3", {"backend": "native"}),
        (
            "anthropic_claude@v5",
            {"model_version": "claude-fable-5-1", "reasoning_effort": "medium"},
        ),
        (
            "anthropic_claude@v5",
            {"model_version": "$inputs.model", "reasoning_effort": "xhigh"},
        ),
    ],
)
def test_manifest_accepts_edges_rejects_cannot_cover(block_type, overrides):
    _manifest(block_type, **overrides)


@pytest.mark.parametrize(
    "block_type, overrides",
    [
        ("open_ai@v6", {"model_version": "gpt-5.4", "reasoning_effort": "max"}),
        ("open_ai@v6", {"model_version": "gpt-4o", "reasoning_effort": "high"}),
        (
            "google_gemini@v5",
            {"model_version": "gemini-3.1-pro-preview", "thinking_level": "minimal"},
        ),
        (
            "google_gemini@v5",
            {"model_version": "gemini-3.7-flash", "thinking_level": "minimal"},
        ),
        (
            "google_gemini@v5",
            {"model_version": "gemini-3.8-flash", "thinking_level": "minimal"},
        ),
        (
            "google_gemini@v5",
            {"model_version": "gemini-2.5-pro", "thinking_level": "low"},
        ),
        ("spacexai@v2", {"model_version": "grok-4.5", "reasoning_effort": "xhigh"}),
        (
            "meta_vlm@v2",
            {"model_version": "Muse Glimmer", "reasoning_effort": "minimal"},
        ),
        (
            "anthropic_claude@v5",
            {"model_version": "claude-sonnet-4-5", "reasoning_effort": "high"},
        ),
        (
            "anthropic_claude@v5",
            {"model_version": "claude-opus-4-6", "reasoning_effort": "xhigh"},
        ),
    ],
)
def test_manifest_rejects_unsupported_combination(block_type, overrides):
    with pytest.raises(ValidationError, match="support"):
        _manifest(block_type, **overrides)
