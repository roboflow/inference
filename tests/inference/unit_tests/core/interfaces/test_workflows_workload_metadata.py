"""`ServerModelMetadataProvider` - the host half of workload introspection.

The contract under test:

* `USE_INFERENCE_MODELS=False` -> `disabled`, and NOT ONE outbound call.
* a non-`roboflow` provider or `OFFLINE_MODE` -> `unavailable`, no call.
* otherwise at most one metadata-only registry call per cache key, mapped
  `modelType -> model_type`, `modelVariant -> model_variant`,
  `taskType -> task_type`, with `modelLatencyMs` dropped.
* every failure mode -> `unavailable`, never an exception.
* successful lookups are reused from a process-wide in-memory cache keyed by
  `(api_key, model_id, authorised_workspace, MODELS_CACHE_AUTH_ENABLED)`; the
  registry helper itself is called with its DEFAULT cache prefix, so no
  credential-derived key ever reaches the shared cache.
"""

import logging
from unittest import mock

import pytest
from cachetools import TTLCache
from roboflow_workflows.execution_engine.entities.workload import (
    ModelMetadataLookup,
    ModelMetadataProvider,
)

import inference.core.env as inference_env
from inference.core.interfaces import workflows_workload_metadata
from inference.core.interfaces.workflows_workload_metadata import (
    ServerModelMetadataProvider,
    clear_model_metadata_cache,
)

REGISTRY_PAYLOAD = {
    "modelType": "yolov8n",
    "taskType": "object-detection",
    "modelVariant": "coco",
    "modelLatencyMs": 12.5,
}

# The prefix `get_model_metadata_from_inference_models_registry` uses when no
# `cache_prefix` is passed - the adapter must not pass one.
DEFAULT_REGISTRY_CACHE_PREFIX = "roboflow_api_data:inference_models_registry"

ASSUME_IDENTITY_TOKEN = "dummy-assume-token"


@pytest.fixture(autouse=True)
def isolated_metadata_cache():
    """No test may inherit another test's mocked payloads or credentials."""
    clear_model_metadata_cache()
    yield
    clear_model_metadata_cache()


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


@pytest.fixture
def auth_enforcement_enabled(monkeypatch):
    """`MODELS_CACHE_AUTH_ENABLED=True`: the helper never reads the shared cache."""
    from inference.core import roboflow_api

    monkeypatch.setattr(roboflow_api, "MODELS_CACHE_AUTH_ENABLED", True)
    return roboflow_api


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


def _resolve(provider, model_id="my-project/3"):
    return provider.resolve_model_metadata(provider="roboflow", model_id=model_id)


class _FakeClock:
    """A monotonic clock the test moves by hand - no `sleep` anywhere."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _install_test_cache(monkeypatch, maxsize=100, ttl=900.0):
    clock = _FakeClock()
    monkeypatch.setattr(
        workflows_workload_metadata,
        "_METADATA_CACHE",
        TTLCache(maxsize=maxsize, ttl=ttl, timer=clock),
    )
    return clock


# --------------------------------------------------------------------------
# the protocol and the gates
# --------------------------------------------------------------------------


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


def test_gates_stay_in_front_of_a_warm_cache(
    monkeypatch, enrichment_enabled, registry_call
) -> None:
    """A cached entry must never resurrect a lookup a gate has to refuse."""
    # given - the exact key is warm
    provider = ServerModelMetadataProvider(api_key="some-key")
    assert _resolve(provider).status == "available"
    registry_call.assert_called_once()

    # when / then - enrichment off
    monkeypatch.setattr(inference_env, "USE_INFERENCE_MODELS", False)
    assert _resolve(provider).status == "disabled"

    # when / then - offline
    monkeypatch.setattr(inference_env, "USE_INFERENCE_MODELS", True)
    monkeypatch.setattr(inference_env, "OFFLINE_MODE", True)
    assert _resolve(provider).status == "unavailable"

    # when / then - a provider whose ids do not address the registry
    monkeypatch.setattr(inference_env, "OFFLINE_MODE", False)
    assert (
        provider.resolve_model_metadata(
            provider="openai", model_id="my-project/3"
        ).status
        == "unavailable"
    )
    registry_call.assert_called_once()


# --------------------------------------------------------------------------
# mapping the registry payload
# --------------------------------------------------------------------------


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
        "type": "model_metadata_v1",
        "model_type": "yolov8n",
        "model_variant": "coco",
        "task_type": "object-detection",
    }
    registry_call.assert_called_once()
    assert registry_call.call_args.kwargs == {
        "api_key": "some-key",
        "model_id": "my-project/3",
    }, "the helper is called with its default cache prefix and nothing else"


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


def test_failure_log_contains_neither_the_key_nor_the_workspace(
    assume_identity_enabled, enrichment_enabled, registry_call, caplog
) -> None:
    # given
    api_key = "super-secret-key"
    registry_call.side_effect = RuntimeError(f"failed for {api_key}")
    provider = ServerModelMetadataProvider(api_key=api_key)
    token = _with_authorised_workspace(assume_identity_enabled, "workspace-db-a")

    # when
    try:
        with caplog.at_level(
            logging.DEBUG, logger=workflows_workload_metadata.logger.name
        ):
            provider.resolve_model_metadata(
                provider="roboflow", model_id="my-project/3"
            )
    finally:
        assume_identity_enabled.assume_identity_authorised_workspace_db_id.reset(token)

    # then - the message text must not carry the credential or the identity it
    # was keyed under
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "my-project/3" in logged
    assert api_key not in logged
    assert "workspace-db-a" not in logged


# --------------------------------------------------------------------------
# the in-memory cache: what is shared, and what is never shared
# --------------------------------------------------------------------------


def test_same_identity_is_resolved_once_across_new_provider_instances(
    auth_enforcement_enabled, enrichment_enabled, registry_call
) -> None:
    """The cache is process level, so a per-request provider still benefits."""
    # when - three separate provider objects, same api key and model id
    results = [
        _resolve(ServerModelMetadataProvider(api_key="same-key")) for _ in range(3)
    ]

    # then
    assert [result.status for result in results] == ["available"] * 3
    assert {result.metadata.model_type for result in results} == {"yolov8n"}
    registry_call.assert_called_once()


def test_a_different_api_key_or_model_id_is_a_different_entry(
    auth_enforcement_enabled, enrichment_enabled, registry_call
) -> None:
    # when
    _resolve(ServerModelMetadataProvider(api_key="key-one"), model_id="project/1")
    _resolve(ServerModelMetadataProvider(api_key="key-two"), model_id="project/1")
    _resolve(ServerModelMetadataProvider(api_key="key-one"), model_id="project/2")
    # ... and each of them repeated
    _resolve(ServerModelMetadataProvider(api_key="key-one"), model_id="project/1")
    _resolve(ServerModelMetadataProvider(api_key="key-two"), model_id="project/1")
    _resolve(ServerModelMetadataProvider(api_key="key-one"), model_id="project/2")

    # then - three distinct identities, three lookups, no cross-serving
    assert registry_call.call_count == 3
    assert {
        (call.kwargs["api_key"], call.kwargs["model_id"])
        for call in registry_call.call_args_list
    } == {("key-one", "project/1"), ("key-two", "project/1"), ("key-one", "project/2")}


def test_missing_and_empty_api_keys_are_distinct_entries(
    auth_enforcement_enabled, enrichment_enabled, registry_call
) -> None:
    # when
    _resolve(ServerModelMetadataProvider(api_key=None))
    _resolve(ServerModelMetadataProvider(api_key=""))
    _resolve(ServerModelMetadataProvider(api_key=None))

    # then - `None` (no key at all) never shares an entry with the empty string
    assert registry_call.call_count == 2
    assert [call.kwargs["api_key"] for call in registry_call.call_args_list] == [
        None,
        "",
    ]


def test_one_key_two_authorised_workspaces_are_two_entries(
    assume_identity_enabled, auth_enforcement_enabled, enrichment_enabled, registry_call
) -> None:
    """Same credential, different assume-identity workspace: never one entry.

    The registry call carries the workspace the auth middleware put into the
    ContextVar in its assume-identity header, so the workspace is one of the
    request identity inputs the key partitions by - whatever precedence the
    platform gives that header over the Bearer key.
    """
    # when
    for workspace in ("workspace-db-a", "workspace-db-b", "workspace-db-a"):
        token = _with_authorised_workspace(assume_identity_enabled, workspace)
        try:
            assert _resolve(ServerModelMetadataProvider(api_key="same-key")).status == (
                "available"
            )
        finally:
            assume_identity_enabled.assume_identity_authorised_workspace_db_id.reset(
                token
            )

    # then - two workspaces, two lookups; the repeat of the first is a hit
    assert registry_call.call_count == 2


def test_the_workspace_is_ignored_when_it_would_not_be_sent(
    monkeypatch, auth_enforcement_enabled, enrichment_enabled, registry_call
) -> None:
    """No service token -> no header on the wire -> the component stays `None`."""
    # given
    from inference.core import roboflow_api

    monkeypatch.setattr(
        roboflow_api, "ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN", None
    )

    # when
    for workspace in ("workspace-db-a", "workspace-db-b"):
        token = _with_authorised_workspace(roboflow_api, workspace)
        try:
            assert workflows_workload_metadata.current_authorised_workspace() is None
            _resolve(ServerModelMetadataProvider(api_key="same-key"))
        finally:
            roboflow_api.assume_identity_authorised_workspace_db_id.reset(token)

    # then - both calls share one entry, exactly as a deployment without
    # assume-identity had before the workspace component existed
    registry_call.assert_called_once()


def test_a_workspace_the_header_check_rejects_is_ignored(
    assume_identity_enabled, auth_enforcement_enabled, enrichment_enabled, registry_call
) -> None:
    # given - a value `_add_assume_identity_headers` would never put on the wire
    unsafe_workspace = "not a valid header value!!"
    assert assume_identity_enabled.workspace_db_id_is_valid(unsafe_workspace) is False

    # when
    token = _with_authorised_workspace(assume_identity_enabled, unsafe_workspace)
    try:
        assert workflows_workload_metadata.current_authorised_workspace() is None
        _resolve(ServerModelMetadataProvider(api_key="same-key"))
    finally:
        assume_identity_enabled.assume_identity_authorised_workspace_db_id.reset(token)
    _resolve(ServerModelMetadataProvider(api_key="same-key"))

    # then - the rejected workspace changed nothing, so the second call hits
    registry_call.assert_called_once()


def test_an_enforcement_off_result_is_never_reused_as_enforcement_on(
    monkeypatch, enrichment_enabled, registry_call
) -> None:
    """A policy-mode transition must not promote an unauthorised answer.

    With `MODELS_CACHE_AUTH_ENABLED=False` the helper may answer from the shared
    model-id-keyed cache without authorising the caller. That answer must not
    become an authorization success once enforcement is turned on.
    """
    # given
    from inference.core import roboflow_api

    monkeypatch.setattr(roboflow_api, "MODELS_CACHE_AUTH_ENABLED", False)
    provider = ServerModelMetadataProvider(api_key="same-key")
    assert _resolve(provider).status == "available"
    registry_call.assert_called_once()

    # when - enforcement is switched on
    monkeypatch.setattr(roboflow_api, "MODELS_CACHE_AUTH_ENABLED", True)
    registry_call.return_value = {
        "modelType": "authorised-arch",
        "taskType": "object-detection",
        "modelVariant": None,
    }
    result = _resolve(provider)

    # then - a fresh, authorised lookup, not the enforcement-off entry
    assert registry_call.call_count == 2
    assert result.metadata.model_type == "authorised-arch"

    # ... and switching back does not lose the enforcement-off entry either
    monkeypatch.setattr(roboflow_api, "MODELS_CACHE_AUTH_ENABLED", False)
    assert _resolve(provider).metadata.model_type == "yolov8n"
    assert registry_call.call_count == 2


# --------------------------------------------------------------------------
# what is NOT cached, plus TTL and capacity
# --------------------------------------------------------------------------


def test_an_exception_is_not_cached_and_the_next_call_retries(
    auth_enforcement_enabled, enrichment_enabled, registry_call
) -> None:
    # given
    registry_call.side_effect = [RuntimeError("platform down"), dict(REGISTRY_PAYLOAD)]
    provider = ServerModelMetadataProvider(api_key="same-key")

    # when
    first = _resolve(provider)
    second = _resolve(provider)

    # then
    assert first.status == "unavailable"
    assert second.status == "available"
    assert registry_call.call_count == 2


@pytest.mark.parametrize(
    "unusable_payload",
    [
        {"modelType": None, "taskType": None, "modelVariant": None},
        {"modelLatencyMs": 3.0},
        ["unexpected"],
    ],
)
def test_an_unusable_payload_is_not_cached(
    auth_enforcement_enabled, enrichment_enabled, registry_call, unusable_payload
) -> None:
    # given
    registry_call.side_effect = [unusable_payload, dict(REGISTRY_PAYLOAD)]
    provider = ServerModelMetadataProvider(api_key="same-key")

    # when
    first = _resolve(provider)
    second = _resolve(provider)

    # then
    assert first.status == "unavailable"
    assert second.status == "available"
    assert registry_call.call_count == 2


def test_an_entry_expires_after_the_configured_ttl(
    monkeypatch, auth_enforcement_enabled, enrichment_enabled, registry_call
) -> None:
    # given - a hand-driven clock, so no test ever sleeps
    clock = _install_test_cache(monkeypatch, ttl=900.0)
    provider = ServerModelMetadataProvider(api_key="same-key")

    # when
    _resolve(provider)
    clock.now = 899.0
    _resolve(provider)
    # ... time passes beyond the TTL
    clock.now = 901.0
    _resolve(provider)

    # then - one lookup inside the window, a fresh one after it
    assert registry_call.call_count == 2


def test_the_cache_is_bounded_and_evicts_the_least_recently_used_entry(
    monkeypatch, auth_enforcement_enabled, enrichment_enabled, registry_call
) -> None:
    # given - a two-entry cache stands in for the 1000-entry production one
    _install_test_cache(monkeypatch, maxsize=2)
    provider = ServerModelMetadataProvider(api_key="same-key")

    # when - three models compete for two slots
    for model_id in ("project/1", "project/2", "project/3"):
        _resolve(provider, model_id=model_id)
    assert registry_call.call_count == 3
    assert len(workflows_workload_metadata._METADATA_CACHE) == 2

    # then - the newest entry is still a hit ...
    _resolve(provider, model_id="project/3")
    assert registry_call.call_count == 3
    # ... and the evicted one is resolved again instead of being lost silently
    _resolve(provider, model_id="project/1")
    assert registry_call.call_count == 4


def test_production_cache_is_bounded_and_uses_the_auth_cache_ttl() -> None:
    # then - the defaults the module ships with, not a test double
    cache = workflows_workload_metadata._METADATA_CACHE
    assert cache.maxsize == 1000
    assert cache.ttl == inference_env.MODELS_CACHE_AUTH_CACHE_TTL


def test_a_caller_cannot_poison_a_later_response(
    auth_enforcement_enabled, enrichment_enabled, registry_call
) -> None:
    """Only an immutable triple is cached; every response is rebuilt from it."""
    # given
    provider = ServerModelMetadataProvider(api_key="same-key")
    first = _resolve(provider)

    # when - the caller tampers with what it received (bypassing `frozen`, which
    # the DTO happens to set today)
    object.__setattr__(first.metadata, "model_type", "tampered")
    assert first.metadata.model_type == "tampered", "the tamper must really land"
    second = _resolve(provider)

    # then
    assert second.metadata.model_type == "yolov8n"
    assert second.metadata is not first.metadata
    registry_call.assert_called_once()


# --------------------------------------------------------------------------
# against the REAL registry helper: both authorization policies
# --------------------------------------------------------------------------


@pytest.fixture
def real_registry(monkeypatch):
    """The real helper over a fake shared cache and a fake HTTP fetch.

    Returns `(store, fetched)`: the shared-cache dict the helper reads/writes and
    the list of request headers that actually reached the platform.
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
        fetched.append(dict(headers or {}))
        return {
            "modelMetadata": {
                "modelArchitecture": f"arch-{len(fetched)}",
                "taskType": "object-detection",
                "modelVariant": None,
            }
        }

    monkeypatch.setattr(roboflow_api, "cache", _Cache())
    monkeypatch.setattr(roboflow_api, "_get_from_url", _fake_get_from_url)
    monkeypatch.setattr(roboflow_api, "GCP_SERVERLESS", False)
    monkeypatch.setattr(roboflow_api, "ENFORCE_CREDITS_VERIFICATION", False)
    monkeypatch.setattr(roboflow_api, "ROBOFLOW_INTERNAL_SERVICE_SECRET", None)
    return store, fetched


def test_enforcement_on_reaches_the_platform_despite_a_preseeded_shared_cache(
    real_registry, assume_identity_enabled, auth_enforcement_enabled, enrichment_enabled
) -> None:
    """A memory miss must not be answered by somebody else's shared entry."""
    # given - the shared cache already holds an entry for this model id
    store, fetched = real_registry
    store[f"{DEFAULT_REGISTRY_CACHE_PREFIX}:my-project/3"] = {
        "modelType": "someone-elses-arch",
        "taskType": "object-detection",
        "modelVariant": None,
        "modelLatencyMs": None,
    }

    # when
    token = _with_authorised_workspace(assume_identity_enabled, "workspace-db-a")
    try:
        result = _resolve(ServerModelMetadataProvider(api_key="key-one"))
    finally:
        assume_identity_enabled.assume_identity_authorised_workspace_db_id.reset(token)

    # then - the platform was asked, with this caller's own key and headers
    assert result.metadata.model_type == "arch-1"
    assert len(fetched) == 1
    assert fetched[0]["Authorization"] == "Bearer key-one"
    assert fetched[0][
        assume_identity_enabled.ASSUME_IDENTITY_AUTHORISED_WORKSPACE_HEADER
    ] == ("workspace-db-a")
    assert (
        fetched[0][assume_identity_enabled.ASSUME_IDENTITY_ACCESS_TOKEN_HEADER]
        == ASSUME_IDENTITY_TOKEN
    )
    # ... and the shared cache key it wrote carries no credential material
    assert sorted(store) == [f"{DEFAULT_REGISTRY_CACHE_PREFIX}:my-project/3"]


def test_enforcement_on_reuses_the_memory_entry_for_that_context_only(
    real_registry, assume_identity_enabled, auth_enforcement_enabled, enrichment_enabled
) -> None:
    # given
    _, fetched = real_registry

    def _resolve_as(api_key, workspace):
        token = _with_authorised_workspace(assume_identity_enabled, workspace)
        try:
            return _resolve(ServerModelMetadataProvider(api_key=api_key))
        finally:
            assume_identity_enabled.assume_identity_authorised_workspace_db_id.reset(
                token
            )

    # when
    first = _resolve_as("key-one", "workspace-db-a")
    repeated = _resolve_as("key-one", "workspace-db-a")
    other_workspace = _resolve_as("key-one", "workspace-db-b")
    other_key = _resolve_as("key-two", "workspace-db-a")

    # then - one platform call per identity; only the exact repeat is a hit
    assert [
        headers[assume_identity_enabled.ASSUME_IDENTITY_AUTHORISED_WORKSPACE_HEADER]
        for headers in fetched
    ] == ["workspace-db-a", "workspace-db-b", "workspace-db-a"]
    assert [headers["Authorization"] for headers in fetched] == [
        "Bearer key-one",
        "Bearer key-one",
        "Bearer key-two",
    ]
    assert first.metadata.model_type == "arch-1"
    assert repeated.metadata.model_type == "arch-1"
    assert other_workspace.metadata.model_type == "arch-2"
    assert other_key.metadata.model_type == "arch-3"


def test_enforcement_off_keeps_the_helpers_shared_model_id_cache_policy(
    monkeypatch, real_registry, enrichment_enabled
) -> None:
    """The accepted trade-off of `MODELS_CACHE_AUTH_ENABLED=False`.

    The helper reads its shared cache for every caller and keys it by model id
    under the DEFAULT prefix. A second credential therefore gets the first
    credential's entry without a platform call - the pre-existing policy of the
    helper, unchanged here. Hosted per-workspace isolation relies on enabling
    `MODELS_CACHE_AUTH_ENABLED`.
    """
    # given
    from inference.core import roboflow_api

    store, fetched = real_registry
    monkeypatch.setattr(roboflow_api, "MODELS_CACHE_AUTH_ENABLED", False)

    # when - two different credentials, the same model id
    first = _resolve(ServerModelMetadataProvider(api_key="key-one"))
    second = _resolve(ServerModelMetadataProvider(api_key="key-two"))

    # then - one platform call, and the shared entry answered the second caller
    assert [headers["Authorization"] for headers in fetched] == ["Bearer key-one"]
    assert first.metadata.model_type == "arch-1"
    assert second.metadata.model_type == "arch-1"
    # ... under the helper's default, credential-free key
    assert sorted(store) == [f"{DEFAULT_REGISTRY_CACHE_PREFIX}:my-project/3"]
