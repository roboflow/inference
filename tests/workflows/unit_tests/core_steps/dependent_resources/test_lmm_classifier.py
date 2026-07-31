"""
Dependent-resources discovery tests for the deprecated LMM For
Classification block (``roboflow_core/lmm_for_classification@v1``).

Same contract as ``lmm@v1``: ``lmm_type="gpt_4v"`` resolves to an OpenAI
third-party model with id ``lmm_config.gpt_model_version`` (default
``gpt-4o``); a selector-fed ``lmm_type`` yields ``None`` (unknown), and the
trailing ``return []`` arm is unreachable through ``model_validate``
because ``lmm_type`` only accepts ``"gpt_4v"`` or a selector.
"""

import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.foundation.lmm_classifier.v1 import (
    BlockManifest as LMMClassifierV1Manifest,
)
from inference.core.workflows.prototypes.block import third_party_model


def _build_payload(**overrides) -> dict:
    payload = {
        "type": "roboflow_core/lmm_for_classification@v1",
        "name": "classifier",
        "images": "$inputs.image",
        "lmm_type": "gpt_4v",
        "classes": ["a", "b"],
    }
    payload.update(overrides)
    return payload


def test_lmm_classifier_v1_gpt_4v_declares_default_openai_model_version() -> None:
    manifest = LMMClassifierV1Manifest.model_validate(_build_payload())

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openai", model_id="gpt-4o"),
    ]


def test_lmm_classifier_v1_gpt_4v_declares_configured_openai_model_version() -> None:
    manifest = LMMClassifierV1Manifest.model_validate(
        _build_payload(
            lmm_config={
                "max_tokens": 200,
                "gpt_image_detail": "low",
                "gpt_model_version": "gpt-4o-mini",
            }
        )
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openai", model_id="gpt-4o-mini"),
    ]


def test_lmm_classifier_v1_selector_fed_lmm_type_reports_unknown_dependencies() -> None:
    manifest = LMMClassifierV1Manifest.model_validate(
        _build_payload(lmm_type="$inputs.lmm_type")
    )

    result = manifest.discover_dependent_resources()

    # None (unknown) — explicitly not [] (known-empty).
    assert result is None


def test_lmm_classifier_v1_rejects_non_selector_non_gpt_4v_lmm_type() -> None:
    # Documents that the method's trailing `return []` arm cannot be reached
    # via manifest validation.
    with pytest.raises(ValidationError):
        LMMClassifierV1Manifest.model_validate(_build_payload(lmm_type="cog_vlm"))
