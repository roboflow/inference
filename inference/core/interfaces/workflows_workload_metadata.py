"""Host-side model metadata enrichment for Workflows workload introspection.

`describe_workflow_workload()` in the standalone `roboflow_workflows` package
builds the model INVENTORY on its own (provider + declared model id + referring
steps). It never talks to the Roboflow platform. Everything in this module is
the optional host half of that contract: given a `(provider, model_id)` pair the
compiler already resolved to a literal, answer with the platform's registry
metadata - or truthfully say the answer is not available.

Three hard rules shape the implementation:

* **No load, no registration, no weights.** The only outbound call is
  `get_model_metadata_from_inference_models_registry()`
  (`GET /models/v1/external/stat`), which is metadata-only. `get_model_type()`
  and `_resolve_model_type()` are deliberately NOT used: they enforce the
  inspecting deployment's model support and write telemetry/cache descriptors,
  which would turn an introspection call into a deployment decision.
* **`USE_INFERENCE_MODELS=False` means zero calls.** The flag is read through
  the `inference.core.env` module at call time, so the gate reflects the process
  configuration in effect when the request is served.
* **Every answer is authorised for the caller that receives it, and no
  credential material is ever turned into a SHARED cache key.** With
  `MODELS_CACHE_AUTH_ENABLED=False` the registry helper answers from the shared
  cache (`inference.core.cache.cache`: Redis, with an in-process `MemoryCache`
  fallback when Redis is not configured or cannot be reached), keyed by
  `{cache_prefix}:{model_id}` alone, BEFORE it authorises anybody. Under the
  default prefix a cold caller would therefore receive whatever entry another
  caller left for the same model id. Each uncached lookup instead passes a
  single-use prefix: a fixed namespace plus a fresh random nonce (see
  `_single_use_cache_prefix()`). Nothing can have been stored under it, so the
  helper always reaches the platform with this call's own key and
  assume-identity headers. The nonce is random, so the prefix holds no api
  key, digest, token or workspace id. The helper still writes its answer under
  that prefix with its usual 10 s expiry; nobody ever reads that entry. Shared
  entries under the default prefix are neither read nor written nor purged
  here, so the model-resolution path keeps its policy. Repeat lookups are
  absorbed by the in-memory cache below, whose key does contain the api key but
  never leaves the process.

In-memory cache
---------------
`_METADATA_CACHE` is a single process-wide `cachetools.TTLCache` shared by all
provider instances and requests, guarded by `_METADATA_CACHE_LOCK` because
cachetools containers are not thread-safe. Capacity is
`_METADATA_CACHE_CAPACITY` entries; the TTL is `MODELS_CACHE_AUTH_CACHE_TTL`
(default 15 minutes), the same TTL the authorization cache already uses.

The key is an exact Python tuple - no hashing, no string concatenation:

    (api_key, model_id, authorised_workspace, models_cache_auth_enabled)

* `api_key` is stored as given, so `None` (no key) and `""` are distinct keys.
* `authorised_workspace` is the workspace the call would send in the
  assume-identity header (see `current_authorised_workspace()`), resolved per
  call - before the lookup - because it lives in a per-request ContextVar. It is
  `None` when no such header would be sent; the api key, model id and
  authorization mode still key the entry.
* `models_cache_auth_enabled` is the authorization policy in force when the
  entry was written, read off `roboflow_api` - the very module attribute the
  helper itself consults. Every entry comes from an authorised platform call in
  either mode; the component keeps each policy's entries apart all the same.

Only successful, usable metadata is cached, as an immutable tuple of the three
mapped fields; every response object is rebuilt from it, so a caller can never
mutate what a later caller receives. Exceptions, `unavailable` payloads and
all-unknown metadata are not cached, so the next call retries.

`MODELS_CACHE_AUTH_ENABLED=True` vs `False`
-------------------------------------------
Both policies behave the same way for introspection: an in-memory miss reaches
the platform carrying this call's own key and headers, and a hit reuses that
answer for the TTL, for that exact key tuple only. With enforcement ON the
helper skips its shared-cache read by itself. With enforcement OFF the helper
still performs its cache `get`, but on the fresh single-use key it always
misses: the prefix is what prevents shared-entry reuse. The prefix is used in
both modes,
so no read of the mode between this module and the helper can pick the
default prefix by mistake. A hit therefore only ever serves an answer the
platform authorised for that exact tuple within the TTL. Switching to a
workspace, api key or policy whose tuple already holds a live entry reuses that
entry, including switching back to a policy that is still warm. A cold tuple,
an expired or evicted entry and a retry after a failure (never cached) are
misses, and each one is authorised again.

The cache is per process only. Concurrent cold misses for the same key may
issue more than one registry request; there is no request coalescing.
"""

import logging
import secrets
import threading
from typing import Any, Optional, Tuple

from cachetools import TTLCache
from roboflow_workflows.execution_engine.entities.workload import (
    ModelMetadata,
    ModelMetadataLookup,
)

import inference.core.env as inference_env
from inference.core import roboflow_api

logger = logging.getLogger(__name__)

# The only provider whose model ids address the Roboflow platform registry.
# Everything else (an explicit third-party model reference) is reported as
# `unavailable` WITHOUT a call - the inventory entry itself is kept.
ROBOFLOW_PROVIDER = "roboflow"

# Fixed part of the single-use registry cache prefix. It is deliberately
# outside the helper's default `roboflow_api_data:` namespace, so a single-use
# key can never coincide with an entry another caller reads.
_SINGLE_USE_CACHE_NAMESPACE = "workload_introspection:inference_models_registry"

# Hard bound on the number of cached entries; the least recently used entry is
# evicted once it is reached.
_METADATA_CACHE_CAPACITY = 1000

# The mapped `(model_type, model_variant, task_type)` triple of ONE successful
# lookup - immutable, so nothing shared is ever handed to a caller.
_MetadataFields = Tuple[Optional[str], Optional[str], Optional[str]]

_METADATA_CACHE: TTLCache = TTLCache(
    maxsize=_METADATA_CACHE_CAPACITY,
    ttl=inference_env.MODELS_CACHE_AUTH_CACHE_TTL,
)
_METADATA_CACHE_LOCK = threading.Lock()


def clear_model_metadata_cache() -> None:
    """Drop every in-memory entry. Intended for tests and fixtures."""
    with _METADATA_CACHE_LOCK:
        _METADATA_CACHE.clear()


def current_authorised_workspace() -> Optional[str]:
    """The workspace this request's registry call would carry in its headers.

    Mirrors `roboflow_api._add_assume_identity_headers` condition for condition:
    the workspace only reaches the platform when the service access token is
    configured AND the per-request ContextVar holds a header-safe workspace id.
    When it would not be sent it is `None`, which is simply one more distinct
    value of that cache-key component.
    """
    if not roboflow_api.ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN:
        return None
    authorised_workspace = roboflow_api.assume_identity_authorised_workspace_db_id.get()
    if not roboflow_api.workspace_db_id_is_valid(authorised_workspace):
        return None
    return authorised_workspace


class ServerModelMetadataProvider:
    """`ModelMetadataProvider` implementation backed by the Roboflow registry.

    Structural conformance is enough - the protocol lives in the standalone
    package and is `@runtime_checkable`, so this class deliberately does not
    inherit from it (the host must not force `roboflow_workflows` to know about
    server types).

    The instance holds nothing but the api key: the metadata cache is process
    level, so a provider built per request still benefits from what earlier
    requests in the same identity already resolved. The registry cache prefix
    is drawn per lookup and never kept on the instance, so one provider can
    safely serve several workspace contexts.
    """

    def __init__(self, api_key: Optional[str]) -> None:
        self._api_key = api_key

    def resolve_model_metadata(
        self,
        provider: str,
        model_id: str,
    ) -> ModelMetadataLookup:
        """Resolve registry metadata of one model for this provider's caller.

        An in-memory hit is served only for the exact
        `(api_key, model_id, authorised_workspace, models_cache_auth_enabled)`
        tuple that obtained it. A miss always reaches the platform with this
        call's own api key and assume-identity headers, whatever the
        `MODELS_CACHE_AUTH_ENABLED` policy. It never takes an answer from the
        shared registry cache that another caller populated.

        Args:
            provider: Provider the compiler resolved for the model reference.
                Only `roboflow` model ids address the platform registry.
            model_id: Literal model id declared by the workflow.

        Returns:
            `disabled` when `USE_INFERENCE_MODELS` is off; `unavailable` for a
            non-`roboflow` provider, in `OFFLINE_MODE`, on any lookup failure
            (including a denied authorization) or when no usable field came
            back; otherwise `available` with the mapped metadata.
        """
        if not inference_env.USE_INFERENCE_MODELS:
            # Contract: the flag being off is reported as `disabled`, and NOT a
            # single lookup is issued.
            return ModelMetadataLookup(status="disabled")
        if provider != ROBOFLOW_PROVIDER:
            return ModelMetadataLookup(status="unavailable")
        if inference_env.OFFLINE_MODE:
            return ModelMetadataLookup(status="unavailable")
        # Every gate above stays in front of the cache, so a cached entry can
        # never resurrect a lookup a disabled/offline deployment must not serve.
        cache_key = (
            self._api_key,
            model_id,
            current_authorised_workspace(),
            bool(roboflow_api.MODELS_CACHE_AUTH_ENABLED),
        )
        with _METADATA_CACHE_LOCK:
            cached_fields = _METADATA_CACHE.get(cache_key)
        if cached_fields is not None:
            return _available_lookup(fields=cached_fields)
        try:
            api_data = roboflow_api.get_model_metadata_from_inference_models_registry(
                api_key=self._api_key,
                model_id=model_id,
                cache_prefix=_single_use_cache_prefix(),
            )
        except Exception as error:
            # Deliberately not `logger.exception`: a failed lookup is an
            # expected outcome of introspection (unknown model, no permission,
            # platform down), and the api key must not reach the log.
            logger.debug(
                "Workload introspection could not resolve metadata for model "
                "'%s'. Error type: %s",
                model_id,
                type(error).__name__,
            )
            return ModelMetadataLookup(status="unavailable")
        fields = _usable_fields(api_data=api_data)
        if fields is None:
            # Nothing usable came back - not cached, so the next call retries.
            return ModelMetadataLookup(status="unavailable")
        with _METADATA_CACHE_LOCK:
            _METADATA_CACHE[cache_key] = fields
        return _available_lookup(fields=fields)


def _single_use_cache_prefix() -> str:
    """A registry cache prefix no other lookup has used or will use.

    The helper reads `{prefix}:{model_id}` from the shared cache before it
    authorises anybody when `MODELS_CACHE_AUTH_ENABLED=False`. A fresh 128-bit
    random nonce guarantees that read misses, so the lookup is authorised by
    the platform. The nonce comes from `secrets`: it is not derived from, and
    reveals nothing about, any api key, token or workspace.
    """
    prefix = f"{_SINGLE_USE_CACHE_NAMESPACE}:{secrets.token_hex(16)}"
    return prefix


def _available_lookup(fields: _MetadataFields) -> ModelMetadataLookup:
    model_type, model_variant, task_type = fields
    return ModelMetadataLookup(
        status="available",
        metadata=ModelMetadata(
            model_type=model_type,
            model_variant=model_variant,
            task_type=task_type,
        ),
    )


def _usable_fields(api_data: Any) -> Optional[_MetadataFields]:
    """The mapped triple of a usable payload, or `None` when there is none."""
    if not isinstance(api_data, dict):
        return None
    # `modelLatencyMs` is intentionally dropped: it is a runtime heuristic of
    # the platform, not a compile-time workload fact.
    fields = (
        _optional_str(api_data.get("modelType")),
        _optional_str(api_data.get("modelVariant")),
        _optional_str(api_data.get("taskType")),
    )
    if all(value is None for value in fields):
        # Every substantive field came back null - there is nothing to report.
        return None
    return fields


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    return str(value)
