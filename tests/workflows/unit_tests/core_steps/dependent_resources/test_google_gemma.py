"""
Dependent-resources discovery tests for the Google Gemma block family
(``roboflow_core/google_gemma@v1..v2``), served via OpenRouter.

Both versions map friendly labels through ``MODEL_VERSION_MAPPING`` to
OpenRouter slugs — verified against ``run()``, which resolves the same
mapping before calling OpenRouter (v1 via ``MODEL_VERSION_MAPPING.get`` in
``run_gemma_llm_prompting``, v2 via direct indexing).
"""

from inference.core.workflows.core_steps.models.foundation.google_gemma.v1 import (
    BlockManifest as GoogleGemmaV1Manifest,
)
from inference.core.workflows.core_steps.models.foundation.google_gemma.v2 import (
    BlockManifest as GoogleGemmaV2Manifest,
)
from inference.core.workflows.prototypes.block import third_party_model

# ---------------------------------------------------------------------------
# v1 — api_key required
# ---------------------------------------------------------------------------


def test_google_gemma_v1_maps_default_model_version_to_openrouter_slug() -> None:
    manifest = GoogleGemmaV1Manifest.model_validate(
        {
            "type": "roboflow_core/google_gemma@v1",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.open_router_api_key",
        }
    )

    # default label "Gemma 4 31B - OpenRouter" resolves to the slug
    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openrouter", model_id="google/gemma-4-31b-it"),
    ]


def test_google_gemma_v1_maps_explicit_model_version_to_openrouter_slug() -> None:
    manifest = GoogleGemmaV1Manifest.model_validate(
        {
            "type": "roboflow_core/google_gemma@v1",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.open_router_api_key",
            "model_version": "Gemma 4 26B A4B - OpenRouter",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openrouter", model_id="google/gemma-4-26b-a4b-it"),
    ]


def test_google_gemma_v1_returns_selector_fed_model_version_verbatim() -> None:
    manifest = GoogleGemmaV1Manifest.model_validate(
        {
            "type": "roboflow_core/google_gemma@v1",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.open_router_api_key",
            "model_version": "$inputs.gemma_model",
        }
    )

    resources = manifest.discover_dependent_resources()

    assert resources == [
        third_party_model(provider="openrouter", model_id="$inputs.gemma_model"),
    ]
    resolver = resources[0].metadata.model_id_resolver
    assert resolver is not None
    assert resolver("Gemma 4 26B A4B - OpenRouter") == "google/gemma-4-26b-a4b-it"


# ---------------------------------------------------------------------------
# v2 — api_key defaults to "rf_key:account" (OpenRouter manifest mixin)
# ---------------------------------------------------------------------------


def test_google_gemma_v2_maps_default_model_version_to_openrouter_slug() -> None:
    manifest = GoogleGemmaV2Manifest.model_validate(
        {
            "type": "roboflow_core/google_gemma@v2",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openrouter", model_id="google/gemma-4-31b-it"),
    ]


def test_google_gemma_v2_maps_explicit_model_version_to_openrouter_slug() -> None:
    manifest = GoogleGemmaV2Manifest.model_validate(
        {
            "type": "roboflow_core/google_gemma@v2",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "model_version": "Gemma 4 26B A4B - OpenRouter",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openrouter", model_id="google/gemma-4-26b-a4b-it"),
    ]


def test_google_gemma_v2_returns_selector_fed_model_version_verbatim() -> None:
    manifest = GoogleGemmaV2Manifest.model_validate(
        {
            "type": "roboflow_core/google_gemma@v2",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "model_version": "$inputs.gemma_model",
        }
    )

    resources = manifest.discover_dependent_resources()

    assert resources == [
        third_party_model(provider="openrouter", model_id="$inputs.gemma_model"),
    ]
    resolver = resources[0].metadata.model_id_resolver
    assert resolver is not None
    assert resolver("Gemma 4 26B A4B - OpenRouter") == "google/gemma-4-26b-a4b-it"
