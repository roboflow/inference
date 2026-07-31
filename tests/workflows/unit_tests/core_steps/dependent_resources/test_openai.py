"""
Dependent-resources discovery tests for the OpenAI VLM block family
(``roboflow_core/open_ai@v1..v4``).

All four versions declare the model id verbatim (no friendly-label mapping
exists in the package) — verified against ``run()``: v1 forwards
``openai_model`` into ``LMMConfig(gpt_model_version=...)``, v2/v3/v4 forward
``model_version`` straight into the request payload / OpenAI client call.
"""

from inference.core.workflows.core_steps.models.foundation.openai.v1 import (
    BlockManifest as OpenAIV1Manifest,
)
from inference.core.workflows.core_steps.models.foundation.openai.v2 import (
    BlockManifest as OpenAIV2Manifest,
)
from inference.core.workflows.core_steps.models.foundation.openai.v3 import (
    BlockManifest as OpenAIV3Manifest,
)
from inference.core.workflows.core_steps.models.foundation.openai.v4 import (
    BlockManifest as OpenAIV4Manifest,
)
from inference.core.workflows.prototypes.block import third_party_model

# ---------------------------------------------------------------------------
# v1 — model field is `openai_model`
# ---------------------------------------------------------------------------


def test_openai_v1_declares_default_model_version() -> None:
    manifest = OpenAIV1Manifest.model_validate(
        {
            "type": "roboflow_core/open_ai@v1",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "openai_api_key": "$inputs.openai_api_key",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openai", model_id="gpt-4o"),
    ]


def test_openai_v1_declares_explicit_model_version() -> None:
    manifest = OpenAIV1Manifest.model_validate(
        {
            "type": "roboflow_core/open_ai@v1",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "openai_api_key": "$inputs.openai_api_key",
            "openai_model": "gpt-4o-mini",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openai", model_id="gpt-4o-mini"),
    ]


def test_openai_v1_returns_selector_fed_model_version_verbatim() -> None:
    manifest = OpenAIV1Manifest.model_validate(
        {
            "type": "roboflow_core/open_ai@v1",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "openai_api_key": "$inputs.openai_api_key",
            "openai_model": "$inputs.openai_model",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openai", model_id="$inputs.openai_model"),
    ]


# ---------------------------------------------------------------------------
# v2 — model field is `model_version`, api_key required
# ---------------------------------------------------------------------------


def test_openai_v2_declares_default_model_version() -> None:
    manifest = OpenAIV2Manifest.model_validate(
        {
            "type": "roboflow_core/open_ai@v2",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.openai_api_key",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openai", model_id="gpt-4o"),
    ]


def test_openai_v2_declares_explicit_model_version() -> None:
    manifest = OpenAIV2Manifest.model_validate(
        {
            "type": "roboflow_core/open_ai@v2",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.openai_api_key",
            "model_version": "gpt-4o-mini",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openai", model_id="gpt-4o-mini"),
    ]


def test_openai_v2_returns_selector_fed_model_version_verbatim() -> None:
    manifest = OpenAIV2Manifest.model_validate(
        {
            "type": "roboflow_core/open_ai@v2",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.openai_api_key",
            "model_version": "$inputs.openai_model",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openai", model_id="$inputs.openai_model"),
    ]


# ---------------------------------------------------------------------------
# v3 — api_key defaults to "rf_key:account", default model is gpt-5
# ---------------------------------------------------------------------------


def test_openai_v3_declares_default_model_version() -> None:
    manifest = OpenAIV3Manifest.model_validate(
        {
            "type": "roboflow_core/open_ai@v3",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openai", model_id="gpt-5"),
    ]


def test_openai_v3_declares_explicit_model_version() -> None:
    manifest = OpenAIV3Manifest.model_validate(
        {
            "type": "roboflow_core/open_ai@v3",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "model_version": "gpt-4o",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openai", model_id="gpt-4o"),
    ]


def test_openai_v3_returns_selector_fed_model_version_verbatim() -> None:
    manifest = OpenAIV3Manifest.model_validate(
        {
            "type": "roboflow_core/open_ai@v3",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "model_version": "$inputs.openai_model",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openai", model_id="$inputs.openai_model"),
    ]


# ---------------------------------------------------------------------------
# v4 — default model is gpt-5.1 (catalog-driven literal list)
# ---------------------------------------------------------------------------


def test_openai_v4_declares_default_model_version() -> None:
    manifest = OpenAIV4Manifest.model_validate(
        {
            "type": "roboflow_core/open_ai@v4",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openai", model_id="gpt-5.1"),
    ]


def test_openai_v4_declares_explicit_model_version() -> None:
    manifest = OpenAIV4Manifest.model_validate(
        {
            "type": "roboflow_core/open_ai@v4",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "model_version": "gpt-4o",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openai", model_id="gpt-4o"),
    ]


def test_openai_v4_returns_selector_fed_model_version_verbatim() -> None:
    manifest = OpenAIV4Manifest.model_validate(
        {
            "type": "roboflow_core/open_ai@v4",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "model_version": "$inputs.openai_model",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openai", model_id="$inputs.openai_model"),
    ]
