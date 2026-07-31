"""
Dependent-resources discovery tests for the Qwen 3.6 OpenRouter block
(``roboflow_core/qwen3_6_openrouter@v1``).

Friendly labels are mapped through ``MODEL_VERSION_MAPPING`` to OpenRouter
slugs — verified against ``run()``, which resolves the same mapping before
calling OpenRouter.
"""

from inference.core.workflows.core_steps.models.foundation.qwen3_6_openrouter.v1 import (
    BlockManifest as Qwen36OpenRouterV1Manifest,
)
from inference.core.workflows.prototypes.block import third_party_model


def test_qwen3_6_openrouter_v1_maps_default_model_version_to_openrouter_slug() -> None:
    manifest = Qwen36OpenRouterV1Manifest.model_validate(
        {
            "type": "roboflow_core/qwen3_6_openrouter@v1",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.open_router_api_key",
        }
    )

    # default label "Qwen 3.6 35B A3B - OpenRouter" resolves to the slug
    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openrouter", model_id="qwen/qwen3.6-35b-a3b"),
    ]


def test_qwen3_6_openrouter_v1_maps_explicit_model_version_to_openrouter_slug() -> None:
    manifest = Qwen36OpenRouterV1Manifest.model_validate(
        {
            "type": "roboflow_core/qwen3_6_openrouter@v1",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.open_router_api_key",
            "model_version": "Qwen 3.6 Flash - OpenRouter",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openrouter", model_id="qwen/qwen3.6-flash"),
    ]


def test_qwen3_6_openrouter_v1_returns_selector_fed_model_version_verbatim() -> None:
    manifest = Qwen36OpenRouterV1Manifest.model_validate(
        {
            "type": "roboflow_core/qwen3_6_openrouter@v1",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.open_router_api_key",
            "model_version": "$inputs.qwen_model",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openrouter", model_id="$inputs.qwen_model"),
    ]
