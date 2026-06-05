"""Unit tests for GLM-OCR v1 block: dynamic (selector) task_type support."""

import pytest

from inference.core.workflows.core_steps.models.foundation.glm_ocr.v1 import (
    BlockManifest,
    _resolve_prompt,
)

BASE = {
    "type": "roboflow_core/glm_ocr@v1",
    "name": "my_glm_step",
    "images": "$inputs.image",
}


# --- Manifest tests ---


def test_manifest_accepts_literal_task_type():
    result = BlockManifest.model_validate(
        {**BASE, "task_type": "custom", "prompt": "Read the label."}
    )
    assert result.task_type == "custom"


def test_manifest_accepts_selector_task_type():
    """task_type must accept $inputs.X references for dynamic mode selection."""
    result = BlockManifest.model_validate(
        {**BASE, "task_type": "$inputs.task_type"}
    )
    assert result.task_type == "$inputs.task_type"


def test_manifest_rejects_invalid_literal_task_type():
    with pytest.raises(Exception):
        BlockManifest.model_validate({**BASE, "task_type": "not-a-real-mode"})


def test_manifest_defers_required_checks_for_selector_task_type():
    """With a dynamic task_type we can't know it's 'custom', so a missing
    prompt must NOT raise at parse time (it is enforced at runtime instead)."""
    result = BlockManifest.model_validate(
        {**BASE, "task_type": "$inputs.task_type"}
    )
    assert result.prompt is None


def test_manifest_still_enforces_custom_requires_prompt_for_literal():
    with pytest.raises(ValueError):
        BlockManifest.model_validate({**BASE, "task_type": "custom"})


def test_task_type_schema_exposes_selector_and_enum():
    schema = BlockManifest.model_json_schema()
    branches = schema["properties"]["task_type"]["anyOf"]
    assert any(
        b.get("reference") is True and "pattern" in b for b in branches
    ), "task_type should expose a selector branch"
    enum_values = {v for b in branches for v in b.get("enum", [])}
    assert "custom" in enum_values and "text-recognition" in enum_values


# --- Runtime resolution tests ---


def test_resolve_prompt_preset_modes():
    assert _resolve_prompt("text-recognition", None, None) == "Text Recognition:"


def test_resolve_prompt_custom_returns_prompt():
    assert _resolve_prompt("custom", "my prompt", None) == "my prompt"


def test_resolve_prompt_custom_without_prompt_raises():
    with pytest.raises(ValueError):
        _resolve_prompt("custom", None, None)


def test_resolve_prompt_structured_without_structure_raises():
    with pytest.raises(ValueError):
        _resolve_prompt("structured-answering", None, None)


def test_resolve_prompt_unknown_task_type_raises():
    """A selector that resolves to an unsupported value gives a clear error,
    not a KeyError."""
    with pytest.raises(ValueError, match="Unsupported GLM-OCR task_type"):
        _resolve_prompt("garbage", None, None)
