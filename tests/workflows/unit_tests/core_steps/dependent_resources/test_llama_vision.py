"""
Dependent-resources discovery tests for the Llama Vision block family
(``roboflow_core/llama_3_2_vision@v1`` and ``roboflow_core/llama_vision@v2``),
served via OpenRouter.

Both versions map friendly labels through ``MODEL_VERSION_MAPPING`` to
OpenRouter slugs — verified against ``run()``, which resolves the same
mapping before calling OpenRouter. Note: v2's ``ModelVersion`` literal has a
single member ("11B - OpenRouter"), so no non-default literal exists there.
"""

from inference.core.workflows.core_steps.models.foundation.llama_vision.v1 import (
    BlockManifest as LlamaVisionV1Manifest,
)
from inference.core.workflows.core_steps.models.foundation.llama_vision.v2 import (
    BlockManifest as LlamaVisionV2Manifest,
)
from inference.core.workflows.prototypes.block import third_party_model

# ---------------------------------------------------------------------------
# v1 — type is roboflow_core/llama_3_2_vision@v1, api_key required
# ---------------------------------------------------------------------------


def test_llama_vision_v1_maps_default_model_version_to_openrouter_slug() -> None:
    manifest = LlamaVisionV1Manifest.model_validate(
        {
            "type": "roboflow_core/llama_3_2_vision@v1",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.llama_api_key",
        }
    )

    # default label "11B (Free) - OpenRouter" resolves to the :free slug
    assert manifest.discover_dependent_resources() == [
        third_party_model(
            provider="openrouter",
            model_id="meta-llama/llama-3.2-11b-vision-instruct:free",
        ),
    ]


def test_llama_vision_v1_maps_explicit_model_version_to_openrouter_slug() -> None:
    manifest = LlamaVisionV1Manifest.model_validate(
        {
            "type": "roboflow_core/llama_3_2_vision@v1",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.llama_api_key",
            "model_version": "90B (Regular) - OpenRouter",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(
            provider="openrouter",
            model_id="meta-llama/llama-3.2-90b-vision-instruct",
        ),
    ]


def test_llama_vision_v1_returns_selector_fed_model_version_verbatim() -> None:
    manifest = LlamaVisionV1Manifest.model_validate(
        {
            "type": "roboflow_core/llama_3_2_vision@v1",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.llama_api_key",
            "model_version": "$inputs.llama_model",
        }
    )

    resources = manifest.discover_dependent_resources()

    assert resources == [
        third_party_model(provider="openrouter", model_id="$inputs.llama_model"),
    ]
    resolver = resources[0].metadata.model_id_resolver
    assert resolver is not None
    assert (
        resolver("90B (Regular) - OpenRouter")
        == "meta-llama/llama-3.2-90b-vision-instruct"
    )


# ---------------------------------------------------------------------------
# v2 — api_key defaults to "rf_key:account" (OpenRouter manifest mixin);
# the model literal offers a single member, so the explicit-literal test
# necessarily reuses the default label.
# ---------------------------------------------------------------------------


def test_llama_vision_v2_maps_default_model_version_to_openrouter_slug() -> None:
    manifest = LlamaVisionV2Manifest.model_validate(
        {
            "type": "roboflow_core/llama_vision@v2",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(
            provider="openrouter",
            model_id="meta-llama/llama-3.2-11b-vision-instruct",
        ),
    ]


def test_llama_vision_v2_maps_explicit_model_version_to_openrouter_slug() -> None:
    manifest = LlamaVisionV2Manifest.model_validate(
        {
            "type": "roboflow_core/llama_vision@v2",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "model_version": "11B - OpenRouter",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(
            provider="openrouter",
            model_id="meta-llama/llama-3.2-11b-vision-instruct",
        ),
    ]


def test_llama_vision_v2_returns_selector_fed_model_version_verbatim() -> None:
    manifest = LlamaVisionV2Manifest.model_validate(
        {
            "type": "roboflow_core/llama_vision@v2",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "model_version": "$inputs.llama_model",
        }
    )

    resources = manifest.discover_dependent_resources()

    assert resources == [
        third_party_model(provider="openrouter", model_id="$inputs.llama_model"),
    ]
    resolver = resources[0].metadata.model_id_resolver
    assert resolver is not None
    assert resolver("11B - OpenRouter") == "meta-llama/llama-3.2-11b-vision-instruct"
