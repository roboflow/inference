"""
Dependent-resources discovery tests for the Stability AI Image Generation
block (``roboflow_core/stability_ai_image_gen@v1``).

``run()`` coerces any value of ``model`` that is not an ``ENDPOINT`` key to
``"core"`` before the API call; ``discover_dependent_resources()`` mirrors
that coercion for literal values and returns selectors verbatim (the final
id requires the ENDPOINT lookup after substitution).
"""

from inference.core.workflows.core_steps.models.foundation.stability_ai.image_gen.v1 import (
    BlockManifest as StabilityAIImageGenV1Manifest,
)
from inference.core.workflows.prototypes.block import third_party_model


def _build_manifest(**overrides) -> StabilityAIImageGenV1Manifest:
    payload = {
        "type": "roboflow_core/stability_ai_image_gen@v1",
        "name": "generator",
        "prompt": "a cat on a synthesizer",
        "api_key": "xxx-xxx",
    }
    payload.update(overrides)
    return StabilityAIImageGenV1Manifest.model_validate(payload)


def test_stability_ai_image_gen_v1_defaults_to_core_endpoint() -> None:
    manifest = _build_manifest()

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="stability_ai", model_id="core"),
    ]


def test_stability_ai_image_gen_v1_keeps_known_endpoint_literal() -> None:
    manifest = _build_manifest(model="ultra")

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="stability_ai", model_id="ultra"),
    ]


def test_stability_ai_image_gen_v1_coerces_unknown_literal_to_core() -> None:
    # Mirrors run(): `if model not in ENDPOINT.keys(): model = "core"`.
    manifest = _build_manifest(model="sdxl-legacy")

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="stability_ai", model_id="core"),
    ]


def test_stability_ai_image_gen_v1_coerces_explicit_none_to_core() -> None:
    manifest = _build_manifest(model=None)

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="stability_ai", model_id="core"),
    ]


def test_stability_ai_image_gen_v1_returns_selector_fed_model_verbatim() -> None:
    manifest = _build_manifest(model="$inputs.model")

    resources = manifest.discover_dependent_resources()

    assert resources == [
        third_party_model(provider="stability_ai", model_id="$inputs.model"),
    ]
    resolver = resources[0].metadata.model_id_resolver
    assert resolver is not None
    assert resolver("ultra") == "ultra"
    assert resolver("bogus-endpoint") == "core"
