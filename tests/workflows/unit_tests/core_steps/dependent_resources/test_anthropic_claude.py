"""
Dependent-resources discovery tests for the Anthropic Claude block family
(``roboflow_core/anthropic_claude@v1..v3``).

- v1 maps friendly labels through ``EXACT_MODELS_VERSIONS_MAPPING`` (direct
  indexing) — ``run()`` performs the identical lookup, so the declared id
  equals the wire id.
- v2 maps through ``EXACT_MODEL_VERSIONS.get(label, label)`` (pass-through
  fallback) — ``run()`` performs the identical lookup.
- v3 maps through ``EXACT_MODEL_VERSIONS.get(label, label)`` (pass-through
  fallback, like v2) — mirroring the direct-Anthropic path of ``run()``;
  unknown values and selectors pass through verbatim.
"""

from inference.core.workflows.core_steps.models.foundation.anthropic_claude.v1 import (
    BlockManifest as AnthropicClaudeV1Manifest,
)
from inference.core.workflows.core_steps.models.foundation.anthropic_claude.v2 import (
    BlockManifest as AnthropicClaudeV2Manifest,
)
from inference.core.workflows.core_steps.models.foundation.anthropic_claude.v3 import (
    BlockManifest as AnthropicClaudeV3Manifest,
)
from inference.core.workflows.prototypes.block import third_party_model

# ---------------------------------------------------------------------------
# v1 — EXACT_MODELS_VERSIONS_MAPPING lookup
# ---------------------------------------------------------------------------


def test_anthropic_claude_v1_maps_default_model_version_to_exact_id() -> None:
    manifest = AnthropicClaudeV1Manifest.model_validate(
        {
            "type": "roboflow_core/anthropic_claude@v1",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.anthropic_api_key",
        }
    )

    # default label "claude-sonnet-4-5" resolves to the dated API id
    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="anthropic", model_id="claude-sonnet-4-5-20250929"),
    ]


def test_anthropic_claude_v1_maps_explicit_model_version_to_exact_id() -> None:
    manifest = AnthropicClaudeV1Manifest.model_validate(
        {
            "type": "roboflow_core/anthropic_claude@v1",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.anthropic_api_key",
            "model_version": "claude-3-opus",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="anthropic", model_id="claude-3-opus-20240229"),
    ]


def test_anthropic_claude_v1_returns_selector_fed_model_version_verbatim() -> None:
    manifest = AnthropicClaudeV1Manifest.model_validate(
        {
            "type": "roboflow_core/anthropic_claude@v1",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.anthropic_api_key",
            "model_version": "$inputs.claude_model",
        }
    )

    resources = manifest.discover_dependent_resources()

    assert resources == [
        third_party_model(provider="anthropic", model_id="$inputs.claude_model"),
    ]
    resolver = resources[0].metadata.model_id_resolver
    assert resolver is not None
    assert resolver("claude-3-opus") == "claude-3-opus-20240229"


# ---------------------------------------------------------------------------
# v2 — EXACT_MODEL_VERSIONS .get() with pass-through fallback
# ---------------------------------------------------------------------------


def test_anthropic_claude_v2_maps_default_model_version_to_exact_id() -> None:
    manifest = AnthropicClaudeV2Manifest.model_validate(
        {
            "type": "roboflow_core/anthropic_claude@v2",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.anthropic_api_key",
        }
    )

    # default label "claude-sonnet-4-5" resolves to the dated API id
    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="anthropic", model_id="claude-sonnet-4-5-20250929"),
    ]


def test_anthropic_claude_v2_maps_explicit_model_version_to_exact_id() -> None:
    manifest = AnthropicClaudeV2Manifest.model_validate(
        {
            "type": "roboflow_core/anthropic_claude@v2",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.anthropic_api_key",
            "model_version": "claude-opus-4-1",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="anthropic", model_id="claude-opus-4-1-20250805"),
    ]


def test_anthropic_claude_v2_returns_selector_fed_model_version_verbatim() -> None:
    manifest = AnthropicClaudeV2Manifest.model_validate(
        {
            "type": "roboflow_core/anthropic_claude@v2",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.anthropic_api_key",
            "model_version": "$inputs.claude_model",
        }
    )

    resources = manifest.discover_dependent_resources()

    assert resources == [
        third_party_model(provider="anthropic", model_id="$inputs.claude_model"),
    ]
    resolver = resources[0].metadata.model_id_resolver
    assert resolver is not None
    assert resolver("claude-opus-4-1") == "claude-opus-4-1-20250805"
    assert resolver("unknown-label") == "unknown-label"


# ---------------------------------------------------------------------------
# v3 — discover() declares the friendly label verbatim, but run() maps it
# through EXACT_MODEL_VERSIONS on the direct-Anthropic path (the proxied
# rf_key path sends it verbatim). Until that divergence is resolved, only
# structural assertions are made — the declared model_id is NOT pinned.
# ---------------------------------------------------------------------------


def test_anthropic_claude_v3_discovery_maps_labels_to_exact_model_versions() -> None:
    cases = {
        # default model_version ("claude-sonnet-4-5") resolves to the dated id
        None: "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5": "claude-haiku-4-5-20251001",
        # selectors are not present in the mapping — pass-through verbatim
        "$inputs.claude_model": "$inputs.claude_model",
    }
    for model_version, expected_model_id in cases.items():
        payload = {
            "type": "roboflow_core/anthropic_claude@v3",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
        }
        if model_version is not None:
            payload["model_version"] = model_version
        manifest = AnthropicClaudeV3Manifest.model_validate(payload)

        assert manifest.discover_dependent_resources() == [
            third_party_model(provider="anthropic", model_id=expected_model_id)
        ]


def test_anthropic_claude_fable_5_1_declares_undated_wire_id_in_every_version() -> None:
    # Fable 5.1 has no dated snapshot alias, so every block version must
    # declare the exact id the request will carry on both auth paths.
    manifests = [
        AnthropicClaudeV1Manifest.model_validate(
            {
                "type": "roboflow_core/anthropic_claude@v1",
                "name": "vlm",
                "images": "$inputs.image",
                "prompt": "What is in the image?",
                "api_key": "$inputs.anthropic_api_key",
                "model_version": "claude-fable-5-1",
            }
        ),
        AnthropicClaudeV2Manifest.model_validate(
            {
                "type": "roboflow_core/anthropic_claude@v2",
                "name": "vlm",
                "images": "$inputs.image",
                "prompt": "What is in the image?",
                "api_key": "$inputs.anthropic_api_key",
                "model_version": "claude-fable-5-1",
            }
        ),
        AnthropicClaudeV3Manifest.model_validate(
            {
                "type": "roboflow_core/anthropic_claude@v3",
                "name": "vlm",
                "images": "$inputs.image",
                "prompt": "What is in the image?",
                "model_version": "claude-fable-5-1",
            }
        ),
    ]

    for manifest in manifests:
        assert manifest.discover_dependent_resources() == [
            third_party_model(provider="anthropic", model_id="claude-fable-5-1"),
        ]


def test_anthropic_claude_v3_selector_declaration_attaches_resolver() -> None:
    manifest = AnthropicClaudeV3Manifest.model_validate(
        {
            "type": "roboflow_core/anthropic_claude@v3",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "model_version": "$inputs.claude_model",
        }
    )

    (resource,) = manifest.discover_dependent_resources()

    resolver = resource.metadata.model_id_resolver
    assert resolver is not None
    assert resolver("claude-haiku-4-5") == "claude-haiku-4-5-20251001"
    assert resolver("unknown-label") == "unknown-label"
