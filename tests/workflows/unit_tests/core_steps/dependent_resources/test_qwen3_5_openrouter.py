"""
Dependent-resources discovery tests for the Qwen 3.5 OpenRouter block
(``roboflow_core/qwen3_5_openrouter@v1``).

Friendly labels are mapped through ``MODEL_VERSION_MAPPING`` to OpenRouter
slugs — verified against ``run()``, which resolves the same mapping before
calling OpenRouter.
"""

from inference.core.workflows.core_steps.models.foundation.qwen3_5_openrouter.v1 import (
    BlockManifest as Qwen35OpenRouterV1Manifest,
)
from inference.core.workflows.prototypes.block import third_party_model


def test_qwen3_5_openrouter_v1_maps_default_model_version_to_openrouter_slug() -> None:
    manifest = Qwen35OpenRouterV1Manifest.model_validate(
        {
            "type": "roboflow_core/qwen3_5_openrouter@v1",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.open_router_api_key",
        }
    )

    # default label "Qwen 3.5 27B - OpenRouter" resolves to the slug
    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openrouter", model_id="qwen/qwen3.5-27b"),
    ]


def test_qwen3_5_openrouter_v1_maps_explicit_model_version_to_openrouter_slug() -> None:
    manifest = Qwen35OpenRouterV1Manifest.model_validate(
        {
            "type": "roboflow_core/qwen3_5_openrouter@v1",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.open_router_api_key",
            "model_version": "Qwen 3.5 Flash 02-23 - OpenRouter",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openrouter", model_id="qwen/qwen3.5-flash-02-23"),
    ]


def test_qwen3_5_openrouter_v1_returns_selector_fed_model_version_verbatim() -> None:
    manifest = Qwen35OpenRouterV1Manifest.model_validate(
        {
            "type": "roboflow_core/qwen3_5_openrouter@v1",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.open_router_api_key",
            "model_version": "$inputs.qwen_model",
        }
    )

    resources = manifest.discover_dependent_resources()

    assert resources == [
        third_party_model(provider="openrouter", model_id="$inputs.qwen_model"),
    ]
    resolver = resources[0].metadata.model_id_resolver
    assert resolver is not None
    assert resolver("Qwen 3.5 Flash 02-23 - OpenRouter") == "qwen/qwen3.5-flash-02-23"
