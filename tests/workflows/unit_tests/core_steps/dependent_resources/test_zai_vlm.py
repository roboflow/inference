"""
Dependent-resources discovery tests for ``roboflow_core/zai_vlm@v1``,
served via OpenRouter. Friendly labels map through ``MODEL_IDS`` to
OpenRouter slugs — the same mapping ``run()`` resolves before calling
OpenRouter.
"""

from inference.core.workflows.core_steps.models.foundation.zai_vlm.v1 import (
    MODEL_IDS,
    BlockManifest,
)
from inference.core.workflows.prototypes.block import third_party_model


def _build_payload(**overrides) -> dict:
    payload = {
        "type": "roboflow_core/zai_vlm@v1",
        "name": "vlm",
        "images": "$inputs.image",
        "task_type": "caption",
    }
    payload.update(overrides)
    return payload


def test_zai_vlm_v1_default_label_resolves_to_catalog_slug() -> None:
    manifest = BlockManifest.model_validate(_build_payload())

    # Default model_version is "GLM 5V Turbo"; cross-check the expected
    # slug against the catalog itself.
    assert MODEL_IDS["GLM 5V Turbo"] == "z-ai/glm-5v-turbo"
    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openrouter", model_id="z-ai/glm-5v-turbo"),
    ]


def test_zai_vlm_v1_selector_fed_label_is_returned_verbatim() -> None:
    manifest = BlockManifest.model_validate(
        _build_payload(model_version="$inputs.zai_model")
    )

    resources = manifest.discover_dependent_resources()

    assert resources == [
        third_party_model(provider="openrouter", model_id="$inputs.zai_model"),
    ]
    resolver = resources[0].metadata.model_id_resolver
    assert resolver is not None
    assert resolver("GLM 5V Turbo") == "z-ai/glm-5v-turbo"
    assert resolver("not-a-model") is None
