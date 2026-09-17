"""
Dependent-resources discovery tests for the Kimi block family
(``roboflow_core/kimi_openrouter@v1..v2``), served via OpenRouter.

Both versions map friendly labels through ``MODEL_VERSION_MAPPING`` to
OpenRouter slugs — verified against ``run()``, which resolves the same
mapping before calling OpenRouter.
"""

from inference.core.workflows.core_steps.models.foundation.kimi_openrouter.v1 import (
    BlockManifest as KimiOpenRouterV1Manifest,
)
from inference.core.workflows.core_steps.models.foundation.kimi_openrouter.v2 import (
    BlockManifest as KimiOpenRouterV2Manifest,
)
from inference.core.workflows.prototypes.block import third_party_model

# ---------------------------------------------------------------------------
# v1 — api_key required
# ---------------------------------------------------------------------------


def test_kimi_openrouter_v1_maps_default_model_version_to_openrouter_slug() -> None:
    manifest = KimiOpenRouterV1Manifest.model_validate(
        {
            "type": "roboflow_core/kimi_openrouter@v1",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.open_router_api_key",
        }
    )

    # default label "Kimi K2.6 - OpenRouter" resolves to the slug
    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openrouter", model_id="moonshotai/kimi-k2.6"),
    ]


def test_kimi_openrouter_v1_maps_explicit_model_version_to_openrouter_slug() -> None:
    manifest = KimiOpenRouterV1Manifest.model_validate(
        {
            "type": "roboflow_core/kimi_openrouter@v1",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.open_router_api_key",
            "model_version": "Kimi K2.5 - OpenRouter",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openrouter", model_id="moonshotai/kimi-k2.5"),
    ]


def test_kimi_openrouter_v1_returns_selector_fed_model_version_verbatim() -> None:
    manifest = KimiOpenRouterV1Manifest.model_validate(
        {
            "type": "roboflow_core/kimi_openrouter@v1",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.open_router_api_key",
            "model_version": "$inputs.kimi_model",
        }
    )

    resources = manifest.discover_dependent_resources()

    assert resources == [
        third_party_model(provider="openrouter", model_id="$inputs.kimi_model"),
    ]
    resolver = resources[0].metadata.model_id_resolver
    assert resolver is not None
    assert resolver("Kimi K2.5 - OpenRouter") == "moonshotai/kimi-k2.5"


# ---------------------------------------------------------------------------
# v2 — api_key defaults to "rf_key:account" (OpenRouter manifest mixin)
# ---------------------------------------------------------------------------


def test_kimi_openrouter_v2_maps_default_model_version_to_openrouter_slug() -> None:
    manifest = KimiOpenRouterV2Manifest.model_validate(
        {
            "type": "roboflow_core/kimi_openrouter@v2",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openrouter", model_id="moonshotai/kimi-k2.6"),
    ]


def test_kimi_openrouter_v2_maps_explicit_model_version_to_openrouter_slug() -> None:
    manifest = KimiOpenRouterV2Manifest.model_validate(
        {
            "type": "roboflow_core/kimi_openrouter@v2",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "model_version": "Kimi K2.5 - OpenRouter",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openrouter", model_id="moonshotai/kimi-k2.5"),
    ]


def test_kimi_openrouter_v2_returns_selector_fed_model_version_verbatim() -> None:
    manifest = KimiOpenRouterV2Manifest.model_validate(
        {
            "type": "roboflow_core/kimi_openrouter@v2",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "model_version": "$inputs.kimi_model",
        }
    )

    resources = manifest.discover_dependent_resources()

    assert resources == [
        third_party_model(provider="openrouter", model_id="$inputs.kimi_model"),
    ]
    resolver = resources[0].metadata.model_id_resolver
    assert resolver is not None
    assert resolver("Kimi K2.5 - OpenRouter") == "moonshotai/kimi-k2.5"
