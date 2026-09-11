from typing import Any, Dict, List, Optional, Protocol, Union


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
    ``countinference`` and ``service_secret`` explicitly. Workflows now pass
    the plain string constant ``CORE_MODEL_ENDPOINT_TYPE`` for ``endpoint_type``
    to avoid importing the server's ``ModelEndpointType`` enum; the server
    coerces it back as needed.

    The stream-pipeline members are prefixed because ``flush_stream_pipeline``,
    ``stream_pipeline_depth``, ``close_stream_pipeline`` and
    ``is_stream_pipelined`` are already a *block*-level duck-typed protocol that
    the executor and the server's stream handler call on step instances.

    PROVISIONAL MEMBER. ``infer_from_request_sync`` takes a pydantic request
    object built by the caller from ``inference.core.entities`` - it is the
    method Phase 11 removes entirely. Do not build new code against it.
    """

    content_addressed_artifact_cache: Any

    def add_model(
        self,
        model_id: str,
        api_key: str,
        model_id_alias: Optional[str] = None,
        **kwargs: Any,
    ) -> None: ...

    def load_action_recognition_model(
        self, model_id: str, api_key: Optional[str] = None, **kwargs: Any
    ) -> Any: ...

    def infer_from_request_sync(
        self, model_id: str, request: Any, **kwargs: Any
    ) -> Any: ...

    def run_tensor_native_inference(self, model_id: str, **kwargs: Any) -> Any: ...

    def get_class_names(self, model_id: str) -> List[str]: ...

    def get_keypoints_classes(self, model_id: str) -> List[List[str]]: ...

    def model_supports_stream_pipeline(self, model_id: str) -> bool: ...

    def get_model_pipeline_depth(self, model_id: str) -> int: ...

    def flush_model_stream_pipeline(self, model_id: str) -> Optional[List[Any]]: ...

    def shutdown_model_stream_pipeline(self, model_id: str) -> None: ...

    def __contains__(self, model_id: str) -> bool: ...


# The `endpoint_type` value every core-model block registers with. It is the
# string form of `inference.core.roboflow_api.ModelEndpointType.CORE_MODEL`;
# importing that enum here would pull the Roboflow API client into every model
# block. `ModelManager.add_model` forwards it untouched and the two server
# functions that read it - `roboflow_api.get_roboflow_model_data` and
# `registries.roboflow._check_if_api_key_has_access_to_model` - coerce it back
# into the enum. All 37 uses inside Workflows were CORE_MODEL.
CORE_MODEL_ENDPOINT_TYPE: str = "core_model"
