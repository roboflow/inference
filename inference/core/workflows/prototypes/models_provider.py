from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Union


class _Unset:
    """Sentinel type: 'the caller did not pass this argument at all'.

    Distinct from `None`: the adapter forwards `None` into the request exactly
    as the block used to, so a request field that rejects `None`
    (`multimask_output`, `enable_thinking`, `confidence`) or treats it
    differently from its default (`sam2_version_id`, `output_prob_thresh`,
    `enforce_dense_masks_in_inference_models`, `clip_version_id`) behaves as
    before. Every optional request argument of a `run_*` method that a block
    passes only sometimes defaults to UNSET.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "UNSET"


UNSET = _Unset()


@dataclass(slots=True)
class InferenceResultsDC:
    """What an inference call returns when the caller also needs the raw
    response objects - today only the rfdetr async stream handoff does
    (`run_instance_segmentation(..., return_raw_responses=True)`)."""

    predictions: List[dict] = field(default_factory=list)
    raw_responses: List[Any] = field(default_factory=list)


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

    def run_object_detection(
        self,
        model_id: str,
        images: List[Any],
        api_key: Optional[str] = None,
        class_agnostic_nms: Optional[bool] = None,
        class_filter: Optional[List[str]] = None,
        confidence: Optional[Union[float, str]] = None,
        iou_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
        max_candidates: Optional[int] = None,
        disable_active_learning: Optional[bool] = None,
        active_learning_target_dataset: Optional[str] = None,
    ) -> List[dict]: ...

    def run_classification(
        self,
        model_id: str,
        images: List[Any],
        api_key: Optional[str] = None,
        confidence: Optional[Union[float, str]] = None,
        disable_active_learning: Optional[bool] = None,
        active_learning_target_dataset: Optional[str] = None,
        inference_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[dict]: ...

    def run_keypoints_detection(
        self,
        model_id: str,
        images: List[Any],
        api_key: Optional[str] = None,
        class_agnostic_nms: Optional[bool] = None,
        class_filter: Optional[List[str]] = None,
        confidence: Optional[Union[float, str]] = None,
        iou_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
        max_candidates: Optional[int] = None,
        keypoint_confidence: Optional[float] = None,
        disable_active_learning: Optional[bool] = None,
        active_learning_target_dataset: Optional[str] = None,
    ) -> List[dict]: ...

    def run_semantic_segmentation(
        self,
        model_id: str,
        images: List[Any],
        api_key: Optional[str] = None,
        confidence: Union[float, str, None, _Unset] = UNSET,
        response_mask_format: str = "base64_png",
    ) -> List[dict]: ...

    def run_instance_segmentation(
        self,
        model_id: str,
        images: List[Any],
        api_key: Optional[str] = None,
        class_agnostic_nms: Optional[bool] = None,
        class_filter: Optional[List[str]] = None,
        confidence: Optional[Union[float, str]] = None,
        iou_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
        max_candidates: Optional[int] = None,
        mask_decode_mode: Optional[str] = None,
        tradeoff_factor: Optional[float] = None,
        response_mask_format: Union[str, None, _Unset] = UNSET,
        enforce_dense_masks_in_inference_models: Union[bool, None, _Unset] = UNSET,
        stream_pipeline_context_id: Union[str, None, _Unset] = UNSET,
        disable_active_learning: Optional[bool] = None,
        active_learning_target_dataset: Optional[str] = None,
        return_raw_responses: bool = False,
    ) -> Union[List[dict], InferenceResultsDC]: ...

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
