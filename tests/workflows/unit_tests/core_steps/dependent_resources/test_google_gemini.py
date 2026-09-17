"""
Dependent-resources discovery tests for the Google Gemini block family
(``roboflow_core/google_gemini@v1..v3``).

All three versions declare ``model_version`` verbatim — verified against
``run()``, which embeds the id unchanged into the
``generativelanguage.googleapis.com/v1beta/models/{model_version}`` URL
(v3's proxied rf_key path also forwards it verbatim in the payload).
"""

from inference.core.workflows.core_steps.models.foundation.google_gemini.v1 import (
    BlockManifest as GoogleGeminiV1Manifest,
)
from inference.core.workflows.core_steps.models.foundation.google_gemini.v2 import (
    BlockManifest as GoogleGeminiV2Manifest,
)
from inference.core.workflows.core_steps.models.foundation.google_gemini.v3 import (
    BlockManifest as GoogleGeminiV3Manifest,
)
from inference.core.workflows.prototypes.block import third_party_model

# ---------------------------------------------------------------------------
# v1 — default model is gemini-2.5-flash, api_key required
# ---------------------------------------------------------------------------


def test_google_gemini_v1_declares_default_model_version() -> None:
    manifest = GoogleGeminiV1Manifest.model_validate(
        {
            "type": "roboflow_core/google_gemini@v1",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.google_api_key",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="google", model_id="gemini-2.5-flash"),
    ]


def test_google_gemini_v1_declares_explicit_model_version() -> None:
    manifest = GoogleGeminiV1Manifest.model_validate(
        {
            "type": "roboflow_core/google_gemini@v1",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.google_api_key",
            "model_version": "gemini-2.5-pro",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="google", model_id="gemini-2.5-pro"),
    ]


def test_google_gemini_v1_returns_selector_fed_model_version_verbatim() -> None:
    manifest = GoogleGeminiV1Manifest.model_validate(
        {
            "type": "roboflow_core/google_gemini@v1",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.google_api_key",
            "model_version": "$inputs.gemini_model",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="google", model_id="$inputs.gemini_model"),
    ]


# ---------------------------------------------------------------------------
# v2 — default model is gemini-3.1-pro-preview, api_key required
# ---------------------------------------------------------------------------


def test_google_gemini_v2_declares_default_model_version() -> None:
    manifest = GoogleGeminiV2Manifest.model_validate(
        {
            "type": "roboflow_core/google_gemini@v2",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.google_api_key",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="google", model_id="gemini-3.1-pro-preview"),
    ]


def test_google_gemini_v2_declares_explicit_model_version() -> None:
    manifest = GoogleGeminiV2Manifest.model_validate(
        {
            "type": "roboflow_core/google_gemini@v2",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.google_api_key",
            "model_version": "gemini-2.5-flash",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="google", model_id="gemini-2.5-flash"),
    ]


def test_google_gemini_v2_returns_selector_fed_model_version_verbatim() -> None:
    manifest = GoogleGeminiV2Manifest.model_validate(
        {
            "type": "roboflow_core/google_gemini@v2",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.google_api_key",
            "model_version": "$inputs.gemini_model",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="google", model_id="$inputs.gemini_model"),
    ]


# ---------------------------------------------------------------------------
# v3 — default model is gemini-3.1-pro-preview, api_key defaults to rf_key
# ---------------------------------------------------------------------------


def test_google_gemini_v3_declares_default_model_version() -> None:
    manifest = GoogleGeminiV3Manifest.model_validate(
        {
            "type": "roboflow_core/google_gemini@v3",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="google", model_id="gemini-3.1-pro-preview"),
    ]


def test_google_gemini_v3_declares_explicit_model_version() -> None:
    manifest = GoogleGeminiV3Manifest.model_validate(
        {
            "type": "roboflow_core/google_gemini@v3",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "model_version": "gemini-3.6-flash",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="google", model_id="gemini-3.6-flash"),
    ]


def test_google_gemini_v3_returns_selector_fed_model_version_verbatim() -> None:
    manifest = GoogleGeminiV3Manifest.model_validate(
        {
            "type": "roboflow_core/google_gemini@v3",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "model_version": "$inputs.gemini_model",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="google", model_id="$inputs.gemini_model"),
    ]
