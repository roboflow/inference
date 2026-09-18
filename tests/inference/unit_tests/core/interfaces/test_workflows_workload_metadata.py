"""`ServerModelMetadataProvider` - the host half of workload introspection.

The contract under test:

* `USE_INFERENCE_MODELS=False` -> `disabled`, and NOT ONE outbound call.
* a non-`roboflow` provider or `OFFLINE_MODE` -> `unavailable`, no call.
* otherwise exactly one metadata-only registry call, mapped
  `modelType -> model_type`, `modelVariant -> model_variant`,
  `taskType -> task_type`, with `modelLatencyMs` dropped.
* every failure mode -> `unavailable`, never an exception.
* the registry cache prefix is scoped per credential, so two api keys can never
  read each other's entry - and the scope digest never leaves the process.
"""

import logging
from unittest import mock

import pytest
from roboflow_workflows.execution_engine.entities.workload import (
    ModelMetadataLookup,
    ModelMetadataProvider,
)

import inference.core.env as inference_env
from inference.core.interfaces import workflows_workload_metadata
from inference.core.interfaces.workflows_workload_metadata import (
    WORKLOAD_CACHE_PREFIX_ROOT,
    ServerModelMetadataProvider,
    credential_scope_digest,
    workload_metadata_cache_prefix,
)

REGISTRY_PAYLOAD = {
    "modelType": "yolov8n",
    "taskType": "object-detection",
    "modelVariant": "coco",
    "modelLatencyMs": 12.5,
}


@pytest.fixture
def registry_call(monkeypatch):
    call = mock.MagicMock(return_value=dict(REGISTRY_PAYLOAD))
    monkeypatch.setattr(
        workflows_workload_metadata.roboflow_api,
        "get_model_metadata_from_inference_models_registry",
        call,
    )
    return call


@pytest.fixture
def enrichment_enabled(monkeypatch):
    monkeypatch.setattr(inference_env, "USE_INFERENCE_MODELS", True)
    monkeypatch.setattr(inference_env, "OFFLINE_MODE", False)


def test_provider_satisfies_the_standalone_package_protocol() -> None:
    # given / when
    provider = ServerModelMetadataProvider(api_key="some-key")

    # then - structural conformance only; the host type never inherits from the
    # standalone package's protocol
    assert isinstance(provider, ModelMetadataProvider)
    assert ModelMetadataProvider not in ServerModelMetadataProvider.__mro__


def test_flag_off_reports_disabled_without_any_lookup(
    monkeypatch, registry_call
) -> None:
    # given
    monkeypatch.setattr(inference_env, "USE_INFERENCE_MODELS", False)
    monkeypatch.setattr(inference_env, "OFFLINE_MODE", False)
    provider = ServerModelMetadataProvider(api_key="some-key")

    # when
    result = provider.resolve_model_metadata(
        provider="roboflow", model_id="my-project/3"
    )

    # then
    assert result == ModelMetadataLookup(status="disabled")
    assert result.metadata is None
    registry_call.assert_not_called()


def test_flag_is_read_at_call_time_not_at_construction(
    monkeypatch, registry_call
) -> None:
    # given - built while enrichment is ON
    monkeypatch.setattr(inference_env, "USE_INFERENCE_MODELS", True)
    monkeypatch.setattr(inference_env, "OFFLINE_MODE", False)
    provider = ServerModelMetadataProvider(api_key="some-key")

    # when - the process configuration changes before the call is served
    monkeypatch.setattr(inference_env, "USE_INFERENCE_MODELS", False)
    result = provider.resolve_model_metadata(
        provider="roboflow", model_id="my-project/3"
    )

    # then
    assert result.status == "disabled"
    registry_call.assert_not_called()


@pytest.mark.parametrize("third_party_provider", ["openai", "google", "anthropic", ""])
def test_third_party_provider_is_unavailable_without_a_lookup(
    enrichment_enabled, registry_call, third_party_provider
) -> None:
    # given
    provider = ServerModelMetadataProvider(api_key="some-key")

    # when
    result = provider.resolve_model_metadata(
        provider=third_party_provider, model_id="gpt-4o"
    )

    # then
    assert result == ModelMetadataLookup(status="unavailable")
    registry_call.assert_not_called()


def test_offline_mode_is_unavailable_without_a_lookup(
    monkeypatch, registry_call
) -> None:
    # given
    monkeypatch.setattr(inference_env, "USE_INFERENCE_MODELS", True)
    monkeypatch.setattr(inference_env, "OFFLINE_MODE", True)
    provider = ServerModelMetadataProvider(api_key="some-key")

    # when
    result = provider.resolve_model_metadata(
        provider="roboflow", model_id="my-project/3"
    )

    # then
    assert result == ModelMetadataLookup(status="unavailable")
    registry_call.assert_not_called()


def test_successful_lookup_maps_registry_fields_and_drops_latency(
    enrichment_enabled, registry_call
) -> None:
    # given
    provider = ServerModelMetadataProvider(api_key="some-key")

    # when
    result = provider.resolve_model_metadata(
        provider="roboflow", model_id="my-project/3"
    )

    # then
    assert result.status == "available"
    assert result.metadata.model_type == "yolov8n"
    assert result.metadata.model_variant == "coco"
    assert result.metadata.task_type == "object-detection"
    assert result.metadata.model_dump() == {
        "type": "model_metadata",
        "model_type": "yolov8n",
        "model_variant": "coco",
        "task_type": "object-detection",
    }
    registry_call.assert_called_once()
    assert registry_call.call_args.kwargs["api_key"] == "some-key"
    assert registry_call.call_args.kwargs["model_id"] == "my-project/3"


def test_lookup_never_asks_for_loading_or_weights(
    enrichment_enabled, registry_call
) -> None:
    # given - the only roboflow_api symbol the adapter is allowed to call
    forbidden = mock.MagicMock(
        side_effect=AssertionError("a model-resolution path was called")
    )
    with mock.patch.object(
        workflows_workload_metadata.roboflow_api, "get_roboflow_model_data", forbidden
    ), mock.patch.object(
        workflows_workload_metadata.roboflow_api,
        "get_roboflow_workspace",
        forbidden,
    ):
        provider = ServerModelMetadataProvider(api_key="some-key")

        # when
        result = provider.resolve_model_metadata(
            provider="roboflow", model_id="my-project/3"
        )

    # then
    assert result.status == "available"
    forbidden.assert_not_called()


def test_partial_metadata_is_still_available(
    monkeypatch, enrichment_enabled, registry_call
) -> None:
    # given
    registry_call.return_value = {
        "modelType": "yolov8n",
        "taskType": None,
        "modelVariant": None,
        "modelLatencyMs": 3.0,
    }
    provider = ServerModelMetadataProvider(api_key="some-key")

    # when
    result = provider.resolve_model_metadata(
        provider="roboflow", model_id="my-project/3"
    )

    # then
    assert result.status == "available"
    assert result.metadata.model_type == "yolov8n"
    assert result.metadata.task_type is None
    assert result.metadata.model_variant is None


@pytest.mark.parametrize(
    "payload",
    [
        {"modelType": None, "taskType": None, "modelVariant": None},
        {"modelLatencyMs": 3.0},
        {},
        {"modelType": "", "taskType": "", "modelVariant": ""},
    ],
)
def test_payload_without_any_substantive_field_is_unavailable(
    enrichment_enabled, registry_call, payload
) -> None:
    # given
    registry_call.return_value = payload
    provider = ServerModelMetadataProvider(api_key="some-key")

    # when
    result = provider.resolve_model_metadata(
        provider="roboflow", model_id="my-project/3"
    )

    # then
    assert result == ModelMetadataLookup(status="unavailable")


def test_non_dict_payload_is_unavailable(enrichment_enabled, registry_call) -> None:
    # given
    registry_call.return_value = ["unexpected"]
    provider = ServerModelMetadataProvider(api_key="some-key")

    # when
    result = provider.resolve_model_metadata(
        provider="roboflow", model_id="my-project/3"
    )

    # then
    assert result == ModelMetadataLookup(status="unavailable")


def test_lookup_exception_is_swallowed_into_unavailable(
    enrichment_enabled, registry_call
) -> None:
    # given
    registry_call.side_effect = RuntimeError("platform down")
    provider = ServerModelMetadataProvider(api_key="some-key")

    # when
    result = provider.resolve_model_metadata(
        provider="roboflow", model_id="my-project/3"
    )

    # then
    assert result == ModelMetadataLookup(status="unavailable")


def test_failure_log_contains_neither_the_key_nor_its_digest(
    enrichment_enabled, registry_call, caplog
) -> None:
    # given
    api_key = "super-secret-key"
    registry_call.side_effect = RuntimeError(f"failed for {api_key}")
    provider = ServerModelMetadataProvider(api_key=api_key)

    # when
    with caplog.at_level(logging.DEBUG, logger=workflows_workload_metadata.logger.name):
        provider.resolve_model_metadata(provider="roboflow", model_id="my-project/3")

    # then - the message text must not carry the credential or its scope token
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "my-project/3" in logged
    assert api_key not in logged
    assert credential_scope_digest(api_key) not in logged


def test_cache_prefix_is_scoped_per_credential_and_never_reversible() -> None:
    # given
    first = ServerModelMetadataProvider(api_key="key-one")
    second = ServerModelMetadataProvider(api_key="key-two")

    # then
    assert first.cache_prefix != second.cache_prefix
    assert first.cache_prefix.startswith(f"{WORKLOAD_CACHE_PREFIX_ROOT}:")
    assert "key-one" not in first.cache_prefix
    assert len(credential_scope_digest("key-one")) == 16
    # the same key always lands in the same partition
    assert first.cache_prefix == workload_metadata_cache_prefix("key-one")
    # a missing key is its own partition, not a crash
    assert ServerModelMetadataProvider(api_key=None).cache_prefix == (
        workload_metadata_cache_prefix(None)
    )


def test_workload_cache_keys_never_collide_with_the_execution_paths_keys() -> None:
    """Introspection must not populate or consume model-loading cache entries.

    The helper keys its cache as `f"{cache_prefix}:{model_id}"`; the execution
    paths leave `cache_prefix` at its default. The workload prefix adds two more
    segments (`workload` + the credential scope), so the two key spaces are
    disjoint for every model id.
    """
    # given
    execution_prefix = "roboflow_api_data:inference_models_registry"
    model_id = "my-project/3"

    # when
    workload_key = f"{workload_metadata_cache_prefix('some-key')}:{model_id}"
    execution_key = f"{execution_prefix}:{model_id}"

    # then
    assert WORKLOAD_CACHE_PREFIX_ROOT == f"{execution_prefix}:workload"
    assert workload_key != execution_key
    assert workload_metadata_cache_prefix("some-key") != execution_prefix


def test_two_credentials_reach_the_registry_twice_with_a_shared_cache(
    monkeypatch, enrichment_enabled
) -> None:
    """The regression the scoped prefix exists for.

    `get_model_metadata_from_inference_models_registry` caches by
    `f"{cache_prefix}:{model_id}"` and, with `MODELS_CACHE_AUTH_ENABLED=False`,
    READS that cache for every caller. This test runs the REAL helper against a
    fake in-memory cache and a stubbed HTTP fetch, so a process-wide prefix
    would serve workspace A's metadata to workspace B's key.
    """
    from inference.core import roboflow_api

    store = {}

    class _Cache:
        def get(self, key):
            return store.get(key)

        def set(self, key, value, expire=None):
            store[key] = value

    fetched = []

    def _fake_get_from_url(url, headers=None, json_response=True):
        fetched.append(headers.get("Authorization"))
        return {
            "modelMetadata": {
                "modelArchitecture": f"arch-{len(fetched)}",
                "taskType": "object-detection",
                "modelVariant": None,
            }
        }

    monkeypatch.setattr(roboflow_api, "cache", _Cache())
    monkeypatch.setattr(roboflow_api, "MODELS_CACHE_AUTH_ENABLED", False)
    monkeypatch.setattr(roboflow_api, "_get_from_url", _fake_get_from_url)
    monkeypatch.setattr(roboflow_api, "GCP_SERVERLESS", False)
    monkeypatch.setattr(roboflow_api, "ENFORCE_CREDITS_VERIFICATION", False)
    monkeypatch.setattr(roboflow_api, "ROBOFLOW_INTERNAL_SERVICE_SECRET", None)

    first = ServerModelMetadataProvider(api_key="key-one")
    second = ServerModelMetadataProvider(api_key="key-two")

    # when - the SAME model id is resolved under two different credentials
    first_result = first.resolve_model_metadata(
        provider="roboflow", model_id="my-project/3"
    )
    second_result = second.resolve_model_metadata(
        provider="roboflow", model_id="my-project/3"
    )
    # ... and the first credential asks again within the 10 s expiry
    repeated_result = first.resolve_model_metadata(
        provider="roboflow", model_id="my-project/3"
    )

    # then - two distinct HTTP calls, one per credential; the repeat is a hit
    assert fetched == ["Bearer key-one", "Bearer key-two"]
    assert first_result.metadata.model_type == "arch-1"
    assert second_result.metadata.model_type == "arch-2"
    assert repeated_result.metadata.model_type == "arch-1"
    assert sorted(store) == [
        f"{workload_metadata_cache_prefix('key-one')}:my-project/3",
        f"{workload_metadata_cache_prefix('key-two')}:my-project/3",
    ]


# ---------------------------------------------------------------------------
# Codex round-001 R001-F001: the trusted scope is (api key, assume-identity
# authorised workspace), not the api key alone. The end-to-end reproduction
# through the real middleware lives in
# tests/inference/unit_tests/core/interfaces/http/test_workflows_describe_workload.py;
# these pin the partition rule itself.
# ---------------------------------------------------------------------------

ASSUME_IDENTITY_TOKEN = "dummy-assume-token"


@pytest.fixture
def assume_identity_enabled(monkeypatch):
    from inference.core import roboflow_api

    monkeypatch.setattr(
        roboflow_api,
        "ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN",
        ASSUME_IDENTITY_TOKEN,
    )
    return roboflow_api


def _with_authorised_workspace(roboflow_api, workspace):
    """Set the ContextVar the auth middleware fills, returning the reset token."""
    return roboflow_api.assume_identity_authorised_workspace_db_id.set(workspace)


def test_scope_ignores_the_workspace_when_assume_identity_is_off(
    monkeypatch,
) -> None:
    from inference.core import roboflow_api

    monkeypatch.setattr(
        roboflow_api, "ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN", None
    )
    token = _with_authorised_workspace(roboflow_api, "workspace-db-a")
    try:
        # then - no token means no header on the wire, so the workspace is not
        # part of the trusted scope and the partition is exactly what it was
        assert workflows_workload_metadata.current_authorised_workspace() is None
        assert ServerModelMetadataProvider(api_key="k").cache_prefix == (
            workload_metadata_cache_prefix("k")
        )
    finally:
        roboflow_api.assume_identity_authorised_workspace_db_id.reset(token)


def test_scope_ignores_a_workspace_that_would_not_be_sent(
    assume_identity_enabled,
) -> None:
    roboflow_api = assume_identity_enabled
    # a value the header-safety check rejects is never put on the wire, so it
    # must not change the partition either
    token = _with_authorised_workspace(roboflow_api, "not a valid header value!!")
    try:
        assert roboflow_api.workspace_db_id_is_valid("not a valid header value!!") is (
            False
        )
        assert workflows_workload_metadata.current_authorised_workspace() is None
        assert ServerModelMetadataProvider(api_key="k").cache_prefix == (
            workload_metadata_cache_prefix("k")
        )
    finally:
        roboflow_api.assume_identity_authorised_workspace_db_id.reset(token)


def test_one_key_two_authorized_workspaces_are_two_partitions(
    assume_identity_enabled,
) -> None:
    roboflow_api = assume_identity_enabled
    prefixes = []
    for workspace in ("workspace-db-a", "workspace-db-b"):
        token = _with_authorised_workspace(roboflow_api, workspace)
        try:
            prefixes.append(
                ServerModelMetadataProvider(api_key="same-key").cache_prefix
            )
        finally:
            roboflow_api.assume_identity_authorised_workspace_db_id.reset(token)

    # then - the exact hole the reviewer found: same credential, different
    # authorised workspace, must never share a cache entry
    assert prefixes[0] != prefixes[1]
    # ... and neither equals the unscoped partition
    assert workload_metadata_cache_prefix("same-key") not in prefixes
    for prefix in prefixes:
        assert prefix.startswith(f"{WORKLOAD_CACHE_PREFIX_ROOT}:")


def test_scope_digest_cannot_be_re_split_or_collide(assume_identity_enabled) -> None:
    # the NUL separator keeps ("a", "b") distinct from a bare "ab" and from
    # ("ab", None); nothing about the inputs is recoverable from the token
    paired = credential_scope_digest("a", "b")
    assert paired != credential_scope_digest("ab")
    assert paired != credential_scope_digest("a")
    assert paired != credential_scope_digest("b")
    assert paired == credential_scope_digest("a", "b")
    assert len(paired) == 16
    assert all(character in "0123456789abcdef" for character in paired)
    # omitting the workspace reproduces the api-key-only token exactly, so a
    # deployment without assume-identity keeps the partition it already had
    assert credential_scope_digest("a", None) == credential_scope_digest("a")
    # a realistic pair leaks neither input
    scoped = credential_scope_digest("my-secret-api-key", "workspace-db-a")
    assert "my-secret-api-key" not in scoped
    assert "workspace-db-a" not in scoped


def test_cache_prefix_follows_the_context_it_is_read_in(
    assume_identity_enabled,
) -> None:
    """The provider is built once per request but the workspace lives in a
    ContextVar, so the prefix must be derived when the lookup runs."""
    roboflow_api = assume_identity_enabled
    provider = ServerModelMetadataProvider(api_key="k")
    unscoped = provider.cache_prefix

    token = _with_authorised_workspace(roboflow_api, "workspace-db-a")
    try:
        scoped = provider.cache_prefix
    finally:
        roboflow_api.assume_identity_authorised_workspace_db_id.reset(token)

    assert unscoped == workload_metadata_cache_prefix("k")
    assert scoped == workload_metadata_cache_prefix("k", "workspace-db-a")
    assert scoped != unscoped
    # and it goes back once the request context is gone
    assert provider.cache_prefix == unscoped


def test_lookup_uses_the_workspace_scoped_prefix(
    assume_identity_enabled, enrichment_enabled, registry_call
) -> None:
    roboflow_api = assume_identity_enabled
    token = _with_authorised_workspace(roboflow_api, "workspace-db-a")
    try:
        result = ServerModelMetadataProvider(api_key="k").resolve_model_metadata(
            provider="roboflow", model_id="my-project/3"
        )
    finally:
        roboflow_api.assume_identity_authorised_workspace_db_id.reset(token)

    assert result.status == "available"
    assert registry_call.call_args.kwargs["cache_prefix"] == (
        workload_metadata_cache_prefix("k", "workspace-db-a")
    )
