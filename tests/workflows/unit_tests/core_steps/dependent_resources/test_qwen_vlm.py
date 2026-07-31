"""
Dependent-resources discovery tests for the unified Qwen-VL block
(``roboflow_core/qwen_vlm@v1``).

``discover_dependent_resources()`` mirrors the model-id resolution done in
``run()``:

- backend=native + catalog label -> Roboflow model id via ``MODEL_VARIANTS``,
- backend=native + ``Fine-tuned model`` sentinel -> ``fine_tuned_model_id``,
- backend=native + selector-fed ``model_version`` -> selector verbatim,
- backend=openrouter + catalog label -> OpenRouter slug via ``MODEL_VARIANTS``,
- backend=openrouter + selector-fed label -> selector verbatim.

The ``return []`` arm for the sentinel without ``fine_tuned_model_id`` is
unreachable through ``model_validate``: the manifest's model_validator
raises for that combination under backend=native, and under
backend=openrouter it resets the sentinel back to a pre-trained variant —
which the last test documents.
"""

import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.foundation.qwen_vlm.v1 import (
    DEFAULT_NATIVE_MODEL_VERSION,
    FINE_TUNED_NATIVE_LABEL,
    MODEL_VARIANTS,
)
from inference.core.workflows.core_steps.models.foundation.qwen_vlm.v1 import (
    BlockManifest as QwenVlmV1Manifest,
)
from inference.core.workflows.prototypes.block import (
    roboflow_platform_model,
    third_party_model,
)


def _build_payload(**overrides) -> dict:
    payload = {
        "type": "roboflow_core/qwen_vlm@v1",
        "name": "vlm",
        "images": "$inputs.image",
        "prompt": "Describe the image.",
    }
    payload.update(overrides)
    return payload


def test_qwen_vlm_v1_native_default_label_resolves_to_catalog_model_id() -> None:
    manifest = QwenVlmV1Manifest.model_validate(_build_payload())

    # Cross-check the expectation against the catalog itself.
    assert MODEL_VARIANTS[DEFAULT_NATIVE_MODEL_VERSION]["model_id"] == "qwen3_5-2b"
    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="qwen3_5-2b"),
    ]


def test_qwen_vlm_v1_native_fine_tuned_sentinel_uses_fine_tuned_model_id() -> None:
    manifest = QwenVlmV1Manifest.model_validate(
        _build_payload(
            model_version=FINE_TUNED_NATIVE_LABEL,
            fine_tuned_model_id="my_workspace/3",
        )
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="my_workspace/3"),
    ]


def test_qwen_vlm_v1_native_fine_tuned_selector_id_is_returned_verbatim() -> None:
    manifest = QwenVlmV1Manifest.model_validate(
        _build_payload(
            model_version=FINE_TUNED_NATIVE_LABEL,
            fine_tuned_model_id="$inputs.qwen_finetune",
        )
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="$inputs.qwen_finetune"),
    ]


def test_qwen_vlm_v1_native_selector_fed_model_version_is_returned_verbatim() -> None:
    manifest = QwenVlmV1Manifest.model_validate(
        _build_payload(model_version="$inputs.qwen_variant")
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="$inputs.qwen_variant"),
    ]


def test_qwen_vlm_v1_openrouter_default_label_resolves_to_catalog_slug() -> None:
    manifest = QwenVlmV1Manifest.model_validate(_build_payload(backend="openrouter"))

    # Default openrouter_model_version is "Qwen 3.6 27B"; cross-check the
    # expected slug against the catalog itself.
    assert MODEL_VARIANTS["Qwen 3.6 27B"]["model_id"] == "qwen/qwen3.6-27b"
    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openrouter", model_id="qwen/qwen3.6-27b"),
    ]


def test_qwen_vlm_v1_openrouter_selector_fed_label_is_returned_verbatim() -> None:
    manifest = QwenVlmV1Manifest.model_validate(
        _build_payload(
            backend="openrouter",
            openrouter_model_version="$inputs.qwen_label",
        )
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openrouter", model_id="$inputs.qwen_label"),
    ]


def test_qwen_vlm_v1_rejects_fine_tuned_sentinel_without_model_id() -> None:
    # Documents that the method's `return []` arm (sentinel without
    # fine_tuned_model_id) cannot be reached via manifest validation.
    with pytest.raises(ValidationError):
        QwenVlmV1Manifest.model_validate(
            _build_payload(model_version=FINE_TUNED_NATIVE_LABEL)
        )


def test_qwen_vlm_v1_openrouter_backend_resets_stale_fine_tuned_sentinel() -> None:
    # Under backend=openrouter the validator resets a stale sentinel to a
    # pre-trained variant, so discovery still resolves via the OpenRouter arm.
    manifest = QwenVlmV1Manifest.model_validate(
        _build_payload(backend="openrouter", model_version=FINE_TUNED_NATIVE_LABEL)
    )

    assert manifest.model_version == DEFAULT_NATIVE_MODEL_VERSION
    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openrouter", model_id="qwen/qwen3.6-27b"),
    ]
