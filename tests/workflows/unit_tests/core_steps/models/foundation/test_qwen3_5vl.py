"""Unit tests for the Qwen3.5-VL v1 block manifest selector support."""

from inference.core.workflows.core_steps.models.foundation.qwen3_5vl.v1 import (
    BlockManifest,
)

BASE = {
    "type": "roboflow_core/qwen3_5vl@v1",
    "name": "my_qwen_step",
    "images": "$inputs.image",
}


def test_manifest_accepts_literal_prompts():
    """Literal strings must still validate (backwards compatibility)."""
    result = BlockManifest.model_validate(
        {**BASE, "prompt": "What is in this image?", "system_prompt": "Be terse."}
    )
    assert result.prompt == "What is in this image?"
    assert result.system_prompt == "Be terse."


def test_manifest_accepts_input_selector_prompts():
    """prompt / system_prompt must accept $inputs.X references."""
    result = BlockManifest.model_validate(
        {
            **BASE,
            "prompt": "$inputs.prompt",
            "system_prompt": "$inputs.system_prompt",
        }
    )
    assert result.prompt == "$inputs.prompt"
    assert result.system_prompt == "$inputs.system_prompt"


def test_manifest_accepts_step_output_selector_prompt():
    """prompt must accept $steps.X.Y references."""
    result = BlockManifest.model_validate({**BASE, "prompt": "$steps.some_step.output"})
    assert result.prompt == "$steps.some_step.output"


def test_manifest_prompts_default_to_none():
    """Both prompt fields remain optional."""
    result = BlockManifest.model_validate(BASE)
    assert result.prompt is None
    assert result.system_prompt is None


def test_prompt_fields_expose_selector_in_schema():
    """Schema must advertise selector support (reference: true + pattern)."""
    schema = BlockManifest.model_json_schema()
    for field in ("prompt", "system_prompt"):
        branches = schema["properties"][field]["anyOf"]
        assert any(
            branch.get("reference") is True and "pattern" in branch
            for branch in branches
        ), f"{field} should expose a selector branch in its schema"
