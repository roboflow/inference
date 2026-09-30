"""Host glue for Workflows workload introspection (`describe_workload` routes).

The introspection itself lives in `roboflow_workflows`
(`describe_workflow_workload()`): it compiles the definition structurally
without initialising blocks, loading models or evaluating custom Python. This
module only supplies what the host owns - the request bodies, the
`workflows_core.*` platform bindings the compiler needs to inline saved inner
workflows, and the optional model metadata enrichment hook.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field
from roboflow_workflows.execution_engine.entities.workload import (
    ModelMetadata,
    ModelMetadataLookup,
)
from roboflow_workflows.execution_engine.introspection.workload import (
    describe_workflow_workload,
)
from roboflow_workflows.execution_engine.introspection.workload_entities import (
    WorkflowIntrospection,
)

from inference_models.weights_providers.roboflow import (
    get_one_page_of_model_metadata,
    roboflow_secure_gateway_proxy_url_builder,
)
from inference_server import configuration
from inference_server.framework.model_stat import _TtlLruCache
from inference_server.workflows import host

logger = logging.getLogger(__name__)

# The only provider whose model ids address the Roboflow platform registry.
# Everything else (an explicit third-party model reference) is reported as
# `unavailable` WITHOUT a call - the inventory entry itself is kept.
ROBOFLOW_PROVIDER = "roboflow"


class DescribeWorkloadRequest(BaseModel):
    api_key: Optional[str] = Field(
        default=None,
        description="Roboflow API Key used to resolve the workflow definition and, when "
        "model metadata enrichment is enabled, to look up model metadata. "
        "May alternatively be sent in the `Authorization: Bearer <api_key>` header - the route "
        "still requires a key through one of the channels.",
    )


class PredefinedWorkflowDescribeWorkloadRequest(DescribeWorkloadRequest):
    use_cache: bool = Field(
        default=True,
        description="Controls usage of cache for workflow definitions. Set this to False when you frequently modify "
        "definition saved in Roboflow app and want to fetch the newest version for the request. "
        "Only applies for Workflows definitions saved on Roboflow platform.",
    )
    workflow_version_id: Optional[str] = Field(
        default=None,
        description="Specific version of the workflow to fetch. If not provided, the latest version is used.",
    )


class WorkflowSpecificationDescribeWorkloadRequest(DescribeWorkloadRequest):
    specification: dict


# `(model_type, model_variant, task_type)` of ONE successful lookup - immutable,
# so nothing shared is ever handed to a caller.
_MetadataFields = Tuple[Optional[str], Optional[str], Optional[str]]

# Process-wide, keyed by `(api_key, model_id)`: an entry is only ever served to
# the credential the registry authorised it for. Only usable answers are
# stored; failures are retried on the next call.
_METADATA_CACHE = _TtlLruCache(
    configuration.MODEL_STAT_CACHE_SIZE, configuration.MODEL_STAT_CACHE_TTL_S
)
_METADATA_CACHE_LOCK = threading.Lock()


def clear_model_metadata_cache() -> None:
    with _METADATA_CACHE_LOCK:
        _METADATA_CACHE.clear()


class RegistryModelMetadataProvider:
    """`ModelMetadataProvider` backed by the Roboflow model registry.

    Metadata only: the one outbound call is the registry metadata lookup the
    server already uses to authorise model access. No model is loaded, no
    weights are downloaded and nothing is registered in the model manager.
    """

    def __init__(self, api_key: Optional[str]) -> None:
        self._api_key = api_key

    def resolve_model_metadata(
        self, provider: str, model_id: str
    ) -> ModelMetadataLookup:
        if provider != ROBOFLOW_PROVIDER:
            return ModelMetadataLookup(status="unavailable")
        if configuration.OFFLINE_MODE:
            return ModelMetadataLookup(status="unavailable")
        cache_key = (self._api_key, model_id)
        with _METADATA_CACHE_LOCK:
            cached_fields = _METADATA_CACHE.get(cache_key)
        if cached_fields is not None:
            return _available_lookup(fields=cached_fields)
        try:
            metadata = get_one_page_of_model_metadata(
                model_id=model_id,
                api_key=self._api_key or None,
                # No-op unless SECURE_GATEWAY is set; then the lookup goes
                # through the gateway like the weights download does.
                proxy_url_builder=roboflow_secure_gateway_proxy_url_builder,
            )
        except Exception as error:
            # A failed lookup is an expected outcome of introspection (unknown
            # model, no permission, platform down); the api key must not reach
            # the log, so only the error type is recorded.
            logger.debug(
                "Workload introspection could not resolve metadata for model "
                "'%s'. Error type: %s",
                model_id,
                type(error).__name__,
            )
            return ModelMetadataLookup(status="unavailable")
        fields = (
            _optional_str(metadata.model_architecture),
            _optional_str(metadata.model_variant),
            _optional_str(metadata.task_type),
        )
        if all(value is None for value in fields):
            return ModelMetadataLookup(status="unavailable")
        with _METADATA_CACHE_LOCK:
            _METADATA_CACHE.set(cache_key, fields)
        return _available_lookup(fields=fields)


def describe_workload(
    definition: dict, api_key: Optional[str]
) -> WorkflowIntrospection:
    """Compile-time workload facts for `definition`.

    The api key is used for two things only: the `workflows_core.*` platform
    bindings the compiler needs to inline saved inner workflows, and the
    optional model metadata lookup.
    """
    init_parameters: Dict[str, Any] = {
        "workflows_core.api_key": api_key,
        **host.workflows_platform_bindings(),
    }
    return describe_workflow_workload(
        definition,
        init_parameters=init_parameters,
        model_metadata_provider=RegistryModelMetadataProvider(api_key=api_key),
    )


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


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    return str(value)
