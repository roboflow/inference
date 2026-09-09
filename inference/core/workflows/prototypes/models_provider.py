from typing import Any, List, Optional, Protocol


class ModelsProvider(Protocol):
    """The port through which Workflows reach models.

    Implemented in the Roboflow inference server by
    ``inference.core.managers.base.ModelManager``. Declared here so that
    ``inference.core.workflows`` does not import the server package for a type
    annotation - that import alone pulls in FastAPI, the model registry, the
    cache, telemetry and usage tracking.

    Deliberately NOT ``runtime_checkable``: nothing does an ``isinstance``
    check against it, and a protocol carrying a data member cannot support one.

    ``add_model`` keeps ``**kwargs`` rather than naming ``endpoint_type``,
    ``countinference`` and ``service_secret`` explicitly - ``endpoint_type`` is
    typed ``ModelEndpointType`` from ``inference.core.roboflow_api``, and naming
    it here would reintroduce exactly the import this port removes.

    PROVISIONAL MEMBERS. ``infer_from_request_sync`` takes a pydantic request
    object built by the caller from ``inference.core.entities`` - it is the
    method Phase 11 option 2 removes entirely. ``__getitem__`` returns a raw
    model object that seven call sites introspect (``key_points_classes``,
    ``flush()``, ``_pipeline_depth``, ``shutdown_pipeline()``); it exists so
    Phase 2 stays a mechanical swap, and Phase 11 must replace it with
    first-class methods. Do not build new code against either.
    """

    content_addressed_artifact_cache: Any

    def add_model(
        self,
        model_id: str,
        api_key: str,
        model_id_alias: Optional[str] = None,
        **kwargs: Any,
    ) -> None: ...

    def infer_from_request_sync(
        self, model_id: str, request: Any, **kwargs: Any
    ) -> Any: ...

    def run_tensor_native_inference(self, model_id: str, **kwargs: Any) -> Any: ...

    def get_class_names(self, model_id: str) -> List[str]: ...

    def __contains__(self, model_id: str) -> bool: ...

    def __getitem__(self, key: str) -> Any: ...


# The `endpoint_type` value every core-model block registers with. It is the
# string form of `inference.core.roboflow_api.ModelEndpointType.CORE_MODEL`;
# importing that enum here would pull the Roboflow API client into every model
# block. `ModelManager.add_model` forwards it untouched and the two server
# functions that read it - `roboflow_api.get_roboflow_model_data` and
# `registries.roboflow._check_if_api_key_has_access_to_model` - coerce it back
# into the enum. All 37 uses inside Workflows were CORE_MODEL.
CORE_MODEL_ENDPOINT_TYPE: str = "core_model"
