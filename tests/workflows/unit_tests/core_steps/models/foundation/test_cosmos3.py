"""Unit tests for the Cosmos 3 Edge v1 block manifest."""

import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.foundation.cosmos3.v1 import (
    BlockManifest,
    _combine_prompt,
)

BASE = {
    "type": "roboflow_core/cosmos3_edge@v1",
    "name": "my_cosmos_step",
    "images": "$inputs.image",
}


def test_manifest_parses_with_defaults():
    result = BlockManifest.model_validate(BASE)
    assert result.prompt is None
    assert result.system_prompt is None
    assert result.model_version == "cosmos-3-edge"


def test_manifest_accepts_literal_prompts():
    result = BlockManifest.model_validate(
        {**BASE, "prompt": "Is the path clear?", "system_prompt": "Be terse."}
    )
    assert result.prompt == "Is the path clear?"
    assert result.system_prompt == "Be terse."


def test_manifest_accepts_input_selector_prompts():
    result = BlockManifest.model_validate(
        {
            **BASE,
            "prompt": "$inputs.prompt",
            "system_prompt": "$inputs.system_prompt",
        }
    )
    assert result.prompt == "$inputs.prompt"
    assert result.system_prompt == "$inputs.system_prompt"


def test_manifest_accepts_model_version_selector():
    result = BlockManifest.model_validate(
        {**BASE, "model_version": "$inputs.model_version"}
    )
    assert result.model_version == "$inputs.model_version"


def test_manifest_rejects_invalid_image_selector():
    with pytest.raises(ValidationError):
        BlockManifest.model_validate({**BASE, "images": 42})


def test_manifest_declares_single_output():
    outputs = BlockManifest.describe_outputs()
    assert [o.name for o in outputs] == ["output"]


def test_combine_prompt_uses_sentinel():
    assert (
        _combine_prompt(prompt="Question?", system_prompt="Context.")
        == "Question?<system_prompt>Context."
    )


def test_combine_prompt_applies_defaults():
    combined = _combine_prompt(prompt=None, system_prompt=None)
    assert combined.startswith("Describe what's in this image.<system_prompt>")
