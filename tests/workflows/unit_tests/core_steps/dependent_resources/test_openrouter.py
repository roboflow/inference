"""
Dependent-resources discovery tests for the generic OpenRouter block
(``roboflow_core/openrouter@v1``).

The block takes a free-form OpenRouter model slug in ``model_id`` and
``discover_dependent_resources()`` returns it verbatim as a third-party
model with provider ``openrouter`` — for both literal slugs and selectors.
"""

from inference.core.workflows.core_steps.models.foundation.openrouter.v1 import (
    BlockManifest as OpenRouterV1Manifest,
)
from inference.core.workflows.prototypes.block import third_party_model


def test_openrouter_v1_declares_model_slug_verbatim() -> None:
    manifest = OpenRouterV1Manifest.model_validate(
        {
            "type": "roboflow_core/openrouter@v1",
            "name": "vlm",
            "images": "$inputs.image",
            "model_id": "anthropic/claude-3.5-sonnet",
            "prompt": "What is in the image?",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(
            provider="openrouter", model_id="anthropic/claude-3.5-sonnet"
        ),
    ]


def test_openrouter_v1_returns_selector_fed_model_id_verbatim() -> None:
    manifest = OpenRouterV1Manifest.model_validate(
        {
            "type": "roboflow_core/openrouter@v1",
            "name": "vlm",
            "images": "$inputs.image",
            "model_id": "$inputs.openrouter_model_id",
            "prompt": "What is in the image?",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(
            provider="openrouter", model_id="$inputs.openrouter_model_id"
        ),
    ]
