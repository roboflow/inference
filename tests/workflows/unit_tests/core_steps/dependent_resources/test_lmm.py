"""
Dependent-resources discovery tests for the deprecated LMM block
(``roboflow_core/lmm@v1``).

``lmm_type="gpt_4v"`` resolves to an OpenAI third-party model whose id is
``lmm_config.gpt_model_version`` (default ``gpt-4o``). A selector-fed
``lmm_type`` makes provider/model statically unknown, so the declared
answer is ``None`` (unknown), not ``[]`` (known-empty).

The method's final ``return []`` arm (non-selector, non-``gpt_4v`` value)
is not reachable through ``model_validate``: ``lmm_type`` is typed
``Union[Selector, Literal["gpt_4v"]]`` and the selector pattern requires a
``$inputs.x`` / ``$steps.x.y`` shape, so any other literal is rejected —
which the last test documents.
"""

import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.foundation.lmm.v1 import (
    BlockManifest as LMMV1Manifest,
)
from inference.core.workflows.prototypes.block import third_party_model


def _build_payload(**overrides) -> dict:
    payload = {
        "type": "roboflow_core/lmm@v1",
        "name": "lmm",
        "images": "$inputs.image",
        "prompt": "my prompt",
        "lmm_type": "gpt_4v",
    }
    payload.update(overrides)
    return payload


def test_lmm_v1_gpt_4v_declares_default_openai_model_version() -> None:
    manifest = LMMV1Manifest.model_validate(_build_payload())

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openai", model_id="gpt-4o"),
    ]


def test_lmm_v1_gpt_4v_declares_configured_openai_model_version() -> None:
    manifest = LMMV1Manifest.model_validate(
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


def test_lmm_v1_selector_fed_lmm_type_reports_unknown_dependencies() -> None:
    manifest = LMMV1Manifest.model_validate(_build_payload(lmm_type="$inputs.lmm_type"))

    result = manifest.discover_dependent_resources()

    # None (unknown) — explicitly not [] (known-empty).
    assert result is None


def test_lmm_v1_rejects_non_selector_non_gpt_4v_lmm_type() -> None:
    # Documents that the method's trailing `return []` arm cannot be reached
    # via manifest validation.
    with pytest.raises(ValidationError):
        LMMV1Manifest.model_validate(_build_payload(lmm_type="cog_vlm"))
