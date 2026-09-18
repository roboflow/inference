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
* **Credentials never leak - not into responses, not into a shared cache.**
  The registry helper keeps its own 10-second cache keyed by
  `f"{cache_prefix}:{model_id}"`, and with `MODELS_CACHE_AUTH_ENABLED=False` that
  cache is READ for every caller. A process-wide prefix would therefore serve one
  workspace's metadata to another workspace's api key.

  The prefix is scoped by a non-reversible digest of the EFFECTIVE TRUSTED SCOPE
  of the lookup, which is the pair

      (api key, assume-identity authorised workspace)

  and not the api key alone. The second component matters because
  `_add_assume_identity_headers` adds `x-assume-identity-authorised-workspace`
  from a per-request ContextVar the auth middleware fills, and the platform
  authorises the registry call against THAT workspace. The middleware resolves
  the caller from query > header > body while this route only ever materialises
  header/body, so two requests can legitimately carry the same body key and still
  be authorised as different workspaces; keying the cache on the key alone would
  serve the first workspace's metadata to the second one without a lookup
  (Codex round-001 R001-F001).

  The workspace is folded in only under exactly the conditions
  `_add_assume_identity_headers` uses to put it on the wire, so a deployment
  without assume-identity keeps the api-key-only partition it had before.
  The digest is an internal cache-partition token: neither it nor its inputs are
  ever returned, logged or put into a discovery reason.
"""

import hashlib
import logging
from typing import Any, Dict, Optional

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

# Distinct from the default `roboflow_api_data:inference_models_registry` prefix
# used by the execution paths: introspection must not populate or consume the
# cache entries that model loading relies on.
WORKLOAD_CACHE_PREFIX_ROOT = "roboflow_api_data:inference_models_registry:workload"

_CREDENTIAL_SCOPE_DIGEST_LENGTH = 16


def current_authorised_workspace() -> Optional[str]:
    """The assume-identity workspace this request's registry call is authorised as.

    Mirrors `roboflow_api._add_assume_identity_headers` condition for condition:
    the workspace only reaches the platform when the service access token is
    configured AND the per-request ContextVar holds a header-safe workspace id.
    When it would not be sent, it is not part of the trusted scope either, so the
    partition stays exactly what it was before this scoping existed.
    """
    if not roboflow_api.ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN:
        return None
    authorised_workspace = roboflow_api.assume_identity_authorised_workspace_db_id.get()
    if not roboflow_api.workspace_db_id_is_valid(authorised_workspace):
        return None
    return authorised_workspace


def credential_scope_digest(
    api_key: Optional[str],
    authorised_workspace: Optional[str] = None,
) -> str:
    """Non-reversible cache-partition token for an effective trusted scope.

    A missing key is its own scope (the empty string), so anonymous lookups
    share one partition and never collide with an authenticated one. The
    assume-identity workspace, when one applies, is folded in behind a NUL
    separator - a byte no api key or workspace id may contain - so the pair
    cannot be re-split and `(key, workspace)` can never collide with a bare key.
    Omitting the workspace reproduces the api-key-only token exactly, which is
    what every non-assume-identity deployment keeps using.
    """
    material = api_key or ""
    if authorised_workspace is not None:
        material = f"{material}\x00{authorised_workspace}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return digest[:_CREDENTIAL_SCOPE_DIGEST_LENGTH]


def workload_metadata_cache_prefix(
    api_key: Optional[str],
    authorised_workspace: Optional[str] = None,
) -> str:
    return (
        f"{WORKLOAD_CACHE_PREFIX_ROOT}:"
        f"{credential_scope_digest(api_key, authorised_workspace)}"
    )


class ServerModelMetadataProvider:
    """`ModelMetadataProvider` implementation backed by the Roboflow registry.

    Structural conformance is enough - the protocol lives in the standalone
    package and is `@runtime_checkable`, so this class deliberately does not
    inherit from it (the host must not force `roboflow_workflows` to know about
    server types).
    """

    def __init__(self, api_key: Optional[str]) -> None:
        self._api_key = api_key

    @property
    def cache_prefix(self) -> str:
        """Computed per access, never cached on the instance.

        The assume-identity workspace lives in a per-request ContextVar, so a
        prefix frozen at construction time could outlive the context it was
        derived from. Reading it here keeps the partition tied to the scope the
        registry call is actually authorised under.
        """
        return workload_metadata_cache_prefix(
            api_key=self._api_key,
            authorised_workspace=current_authorised_workspace(),
        )

    def resolve_model_metadata(
        self,
        provider: str,
        model_id: str,
    ) -> ModelMetadataLookup:
        if not inference_env.USE_INFERENCE_MODELS:
            # Contract: the flag being off is reported as `disabled`, and NOT a
            # single lookup is issued.
            return ModelMetadataLookup(status="disabled")
        if provider != ROBOFLOW_PROVIDER:
            return ModelMetadataLookup(status="unavailable")
        if inference_env.OFFLINE_MODE:
            return ModelMetadataLookup(status="unavailable")
        try:
            api_data = roboflow_api.get_model_metadata_from_inference_models_registry(
                api_key=self._api_key,
                model_id=model_id,
                cache_prefix=self.cache_prefix,
            )
            return _build_lookup(api_data=api_data)
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


def _build_lookup(api_data: Any) -> ModelMetadataLookup:
    if not isinstance(api_data, dict):
        return ModelMetadataLookup(status="unavailable")
    metadata = _map_registry_payload(api_data=api_data)
    if not metadata.has_known_fields():
        # Every substantive field came back null - there is nothing to report.
        return ModelMetadataLookup(status="unavailable")
    return ModelMetadataLookup(status="available", metadata=metadata)


def _map_registry_payload(api_data: Dict[str, Any]) -> ModelMetadata:
    # `modelLatencyMs` is intentionally dropped: it is a runtime heuristic of
    # the platform, not a compile-time workload fact.
    return ModelMetadata(
        model_type=_optional_str(api_data.get("modelType")),
        model_variant=_optional_str(api_data.get("modelVariant")),
        task_type=_optional_str(api_data.get("taskType")),
    )


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    return str(value)
