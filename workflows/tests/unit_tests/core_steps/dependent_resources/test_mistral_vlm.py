"""
Dependent-resources discovery tests for ``roboflow_core/mistral_vlm@v1``,
served via OpenRouter. Friendly labels map through ``MODEL_IDS`` to
OpenRouter slugs, the same mapping ``run()`` resolves before calling
OpenRouter.
"""

from roboflow_workflows.core_steps.models.foundation.mistral_vlm.v1 import (
    MODEL_IDS,
    BlockManifest,
)
from roboflow_workflows.prototypes.block import third_party_model


def _build_payload(**overrides) -> dict:
    payload = {
        "type": "roboflow_core/mistral_vlm@v1",
        "name": "vlm",
        "images": "$inputs.image",
        "task_type": "caption",
    }
    payload.update(overrides)
    return payload


def test_mistral_vlm_v1_default_label_resolves_to_catalog_slug() -> None:
    manifest = BlockManifest.model_validate(_build_payload())

    assert MODEL_IDS["Mistral Large 4"] == "mistralai/mistral-large-4-0"
    assert manifest.discover_dependent_resources() == [
        third_party_model(
            provider="openrouter", model_id="mistralai/mistral-large-4-0"
        ),
    ]


def test_mistral_vlm_v1_selector_fed_label_is_returned_verbatim() -> None:
    manifest = BlockManifest.model_validate(
        _build_payload(model_version="$inputs.mistral_model")
    )

    resources = manifest.discover_dependent_resources()

    assert resources == [
        third_party_model(provider="openrouter", model_id="$inputs.mistral_model"),
    ]
    resolver = resources[0].metadata.model_id_resolver
    assert resolver is not None
    assert resolver("Mistral Large 4") == "mistralai/mistral-large-4-0"
    assert resolver("not-a-model") is None
