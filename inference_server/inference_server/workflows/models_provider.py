from __future__ import annotations

import logging
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from inference_sdk.http.utils.aliases import resolve_roboflow_model_alias
from roboflow_workflows import environment
from roboflow_workflows.prototypes.models_provider import (
    UNSET,
    InferenceResultsDC,
    _Unset,
)

from inference_model_manager.registry_defaults import IMAGE_EMBEDDINGS_ACTION
from inference_model_manager.stream_pipeline import STREAM_PIPELINE_PRODUCER_ID_KWARG
from inference_models.errors import BaseInferenceModelsError
from inference_models.models.base.action_recognition import VideoSampling
from inference_models.models.base.async_handoff import (
    STREAM_PIPELINE_CONTEXT_ID_KWARG,
    attach_async_response_future,
    get_async_response_context_id,
    get_async_response_future,
)
from inference_models.utils import model_blob_cache
from inference_server import pingback, telemetry
from inference_server.framework.input_parsers.image_limits import too_many_images
from inference_server.gateway import _load_failure
from inference_server.legacy.action_recognition import (
    ACTION_RECOGNITION_ACTION,
    ensure_action_recognition_route,
)
from inference_server.legacy.bridge import (
    Route,
    SyncLegacyBridge,
    requested_model_id_for,
    resolved_model_for,
)
from inference_server.legacy.common import (
    ImagePayload,
    _error_from_response,
    as_image_list,
    decode_inline_image,
    image_dims,
    keep_image_orientation,
    split_image,
)
from inference_server.legacy.entities import (
    ClassificationInferenceRequest,
    ClipCompareRequest,
    ClipImageEmbeddingRequest,
    ClipTextEmbeddingRequest,
    DepthEstimationRequest,
    DoctrOCRInferenceRequest,
    EasyOCRInferenceRequest,
    ImageEmbeddingRequest,
    InstanceSegmentationInferenceRequest,
    KeypointsDetectionInferenceRequest,
    LMMInferenceRequest,
    Moondream2InferenceRequest,
    ObjectDetectionInferenceRequest,
    PerceptionEncoderImageEmbeddingRequest,
    PerceptionEncoderTextEmbeddingRequest,
    PPOCRInferenceRequest,
    Sam2SegmentationRequest,
    Sam3SegmentationRequest,
    SemanticSegmentationInferenceRequest,
    YOLOWorldInferenceRequest,
)
from inference_server.legacy.errors import LegacyHTTPError
from inference_server.legacy.load_failures import load_failure_error
from inference_server.legacy.prompts import (
    Box,
    Point,
    Sam2Prompt,
    Sam2PromptSet,
    Sam3Prompt,
)
from inference_server.legacy.telemetry_recording import record_telemetry
from inference_server.legacy.translation import (
    IMAGE_EMBEDDING_OVERRIDE_FIELDS,
    build_embedding_calls,
    build_image_embedding_params,
    build_interactive_segmentation_params,
    build_open_vocabulary_params,
    build_task_params,
    build_vlm_params,
    ensure_ocr_request_supported,
    ensure_request_supported,
    make_embedding_info,
    repack_depth_estimation,
    repack_embedding_response,
    repack_image_embeddings,
    repack_interactive_segmentation_response,
    repack_moondream_detection,
    repack_object_detection_response,
    repack_prediction,
    repack_structured_ocr_response,
    repack_vlm_response,
    requested_open_vocabulary_classes,
    resolve_request_action,
)
from inference_server.prometheus import measure_inference
from inference_server.routing import (
    DEFAULT_EMBEDDING_OUTPUT_TYPE,
    IMAGE_EMBEDDINGS,
    capability_instance,
    registration_key,
    routing_key,
)
from inference_server.workflows.tensor_native import (
    SUPPORTED_TASK_TYPES,
    assemble_native_result,
    native_action,
    native_image_payloads,
    native_params,
    numpy_to_tensors,
)

logger = logging.getLogger(__name__)

_WORKFLOW_SOURCE = "workflow-execution"
_SAM3_3D_UNAVAILABLE = (
    "SAM3 3D object reconstruction is not available on inference_server"
)


def _passed(**arguments: Any) -> Dict[str, Any]:
    return {
        name: value
        for name, value in arguments.items()
        if not isinstance(value, _Unset)
    }


@dataclass(frozen=True)
class _StreamFrame:
    route: Route
    request: Any
    dims: Tuple[int, int]


@dataclass
class _StreamFrames:
    route: Route
    loaded_monotonic: Optional[float]
    frames: Dict[str, _StreamFrame] = field(default_factory=dict)

    def recorded_under(self, route: Route) -> bool:
        return self.route is route and self.loaded_monotonic == route.loaded_monotonic


class _ActionRecognitionModelProxy:
    """The model-like object the action recognition block drives: one bridge call per window."""

    def __init__(
        self, bridge: SyncLegacyBridge, route: Route, api_key: Optional[str]
    ) -> None:
        self._bridge = bridge
        self._route = route
        self._api_key = api_key

    @property
    def class_names(self) -> Optional[List[str]]:
        return self._route.class_names

    @property
    def video_sampling(self) -> VideoSampling:
        if self._route.video_sampling is None:
            return VideoSampling()
        return VideoSampling(**self._route.video_sampling)

    def infer(
        self,
        frames: List[Any],
        class_names: Optional[List[str]] = None,
        fps: Optional[float] = None,
    ) -> List[Any]:
        segments = self._bridge.infer_params_only(
            self._route,
            self._api_key,
            ACTION_RECOGNITION_ACTION,
            {"frames": frames, "class_names": class_names, "fps": fps},
        )

        return segments


class GatewayModelsProvider:
    """Implements roboflow_workflows.prototypes.models_provider.ModelsProvider on top of SyncLegacyBridge. One instance per workflow request."""

    def __init__(
        self,
        bridge: SyncLegacyBridge,
        api_key: Optional[str],
        request_path: Optional[str] = None,
    ) -> None:
        self._bridge = bridge
        self._api_key = api_key
        self._request_path = request_path
        self._model_keys: Dict[str, Optional[str]] = {}
        self._routes: Dict[str, Route] = {}
        self._registration_keys: Dict[str, str] = {}
        self._artifact_cache: Any = None
        self._artifact_cache_resolved = False
        self._producer_id = str(id(self))
        self._stream_frames: Dict[str, _StreamFrames] = {}

    @property
    def content_addressed_artifact_cache(self) -> Any:
        if not self._artifact_cache_resolved:
            self._artifact_cache = model_blob_cache.get_shared_model_blob_cache()
            self._artifact_cache_resolved = True
        return self._artifact_cache

    def add_model(
        self,
        model_id: str,
        api_key: str,
        model_id_alias: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        key = self._key_for(model_id, api_key)
        self._model_keys[model_id] = key
        if model_id_alias is not None:
            self._model_keys[model_id_alias] = key
        row_key = model_id if model_id_alias is None else model_id_alias
        alias = model_id if row_key != model_id else None
        path = self._request_path or ""
        required_capabilities = kwargs.get("required_capabilities")
        output_type = kwargs.get("output_type", DEFAULT_EMBEDDING_OUTPUT_TYPE)
        instance = capability_instance(required_capabilities, output_type)
        route = self._resolve_route(
            model_id,
            key,
            row_key=registration_key(row_key, instance),
            path=path,
            alias=alias,
            instance=instance,
        )
        if instance:
            row_key = registration_key(row_key, instance)
            self._registration_keys[registration_key(model_id, instance)] = routing_key(
                model_id, instance
            )
        else:
            self._routes[model_id] = route
        self._bridge.record_request(route, row_key, path, alias=alias)

    def _resolve_route(
        self,
        model_id: str,
        api_key: Optional[str],
        *,
        instance: str = "",
        **kwargs: Any,
    ) -> Route:
        if instance:
            kwargs["instance"] = instance
        try:
            route = self._bridge.resolve(model_id, api_key, **kwargs)
        except (PermissionError, LookupError, RuntimeError) as error:
            cause = error.__cause__
            if not isinstance(cause, BaseInferenceModelsError):
                raise
        else:
            return route

        rebuilt = load_failure_error(_load_failure(cause))
        if type(rebuilt) is not type(cause):
            cause.__cause__ = None
            cause.__context__ = None
            cause.__suppress_context__ = True
            raise cause

        raise rebuilt from None

    def _key_for(self, model_id: str, api_key: Optional[str] = None) -> Optional[str]:
        if api_key is not None:
            return api_key
        if model_id in self._model_keys:
            return self._model_keys[model_id]
        return self._api_key

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
    ) -> List[dict]:
        request = ObjectDetectionInferenceRequest(
            api_key=api_key,
            model_id=model_id,
            image=images,
            disable_active_learning=disable_active_learning,
            active_learning_target_dataset=active_learning_target_dataset,
            class_agnostic_nms=class_agnostic_nms,
            class_filter=class_filter,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            max_candidates=max_candidates,
            source=_WORKFLOW_SOURCE,
            confidence=confidence,
        )
        return self._dump(self._run_cv(model_id, request, api_key))

    def run_classification(
        self,
        model_id: str,
        images: List[Any],
        api_key: Optional[str] = None,
        confidence: Optional[Union[float, str]] = None,
        disable_active_learning: Optional[bool] = None,
        active_learning_target_dataset: Optional[str] = None,
        inference_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[dict]:
        request = ClassificationInferenceRequest(
            api_key=api_key,
            model_id=model_id,
            image=images,
            disable_active_learning=disable_active_learning,
            source=_WORKFLOW_SOURCE,
            active_learning_target_dataset=active_learning_target_dataset,
            confidence=confidence,
        )
        return self._dump(
            self._run_cv(model_id, request, api_key, extra_params=inference_kwargs)
        )

    def run_image_embeddings(
        self,
        model_id: str,
        images: List[Any],
        api_key: Optional[str] = None,
        output_type: str = DEFAULT_EMBEDDING_OUTPUT_TYPE,
    ) -> dict:
        request = ImageEmbeddingRequest(
            model_id=model_id,
            image=images,
            api_key=api_key,
            output_type=output_type,
            source=_WORKFLOW_SOURCE,
        )
        key = self._key_for(model_id, api_key)
        route = self._embedding_route(model_id, key, output_type)
        started = time.perf_counter()
        payloads = self._request_payloads(request)
        if request.disable_preproc_auto_orient:
            payloads = [
                keep_image_orientation(payload, ndarray_ok=self._bridge.accepts_ndarray)
                for payload in payloads
            ]
        params = build_image_embedding_params(request)
        results = self._bridge.infer(
            route, key, IMAGE_EMBEDDINGS_ACTION, payloads, params
        )
        response = repack_image_embeddings(
            results, resolve_roboflow_model_alias(model_id), request
        )
        response.time = time.perf_counter() - started
        pingback.record_inference(route.registry_id, request, response)
        result = response.model_dump(exclude_none=True)

        return result

    def run_tensor_image_embeddings(
        self,
        model_id: str,
        images: List[Any],
        *,
        input_color_format: str,
        api_key: Optional[str] = None,
        output_type: str = DEFAULT_EMBEDDING_OUTPUT_TYPE,
    ) -> dict:
        """Generate embeddings as a batched tensor on the workflow tensor device.

        Images cross the model manager as NumPy arrays and the embeddings come
        back marshalled the same way, so the result is rebuilt on
        ``WORKFLOWS_IMAGE_TENSOR_DEVICE`` rather than kept on the model device.

        Args:
            model_id: Classification model version or alias.
            images: Materialized CHW RGB tensors or HWC BGR NumPy images.
            input_color_format: Color ordering of the supplied images.
            api_key: Credential used to register the model.
            output_type: Feature vector or pre-activation logits.

        Returns:
            Mapping with a batched tensor under ``embeddings`` and the
            compatibility metadata under ``embedding_info``.
        """
        key = self._key_for(model_id, api_key)
        route = self._embedding_route(model_id, key, output_type)
        payloads = native_image_payloads(
            images, ndarray_ok=self._bridge.accepts_ndarray
        )
        params = {"output_type": output_type, "input_color_format": input_color_format}
        results = self._bridge.infer(
            route, key, IMAGE_EMBEDDINGS_ACTION, payloads, params
        )
        embeddings = torch.cat(
            [
                numpy_to_tensors(
                    result["embeddings"], environment.WORKFLOWS_IMAGE_TENSOR_DEVICE
                ).reshape(1, -1)
                for result in results
            ],
            dim=0,
        )
        info = make_embedding_info(
            resolve_roboflow_model_alias(model_id),
            results[0]["embedding_info"],
            {field: False for field in IMAGE_EMBEDDING_OVERRIDE_FIELDS},
        )
        result = {
            "embeddings": embeddings,
            "embedding_info": info.model_dump(exclude_none=True),
        }

        return result

    def _embedding_route(
        self, model_id: str, api_key: Optional[str], output_type: str
    ) -> Route:
        route = self._resolve_route(
            model_id,
            api_key,
            instance=capability_instance([IMAGE_EMBEDDINGS], output_type),
        )

        return route

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
    ) -> List[dict]:
        request = KeypointsDetectionInferenceRequest(
            api_key=api_key,
            model_id=model_id,
            image=images,
            disable_active_learning=disable_active_learning,
            active_learning_target_dataset=active_learning_target_dataset,
            class_agnostic_nms=class_agnostic_nms,
            class_filter=class_filter,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            max_candidates=max_candidates,
            keypoint_confidence=keypoint_confidence,
            source=_WORKFLOW_SOURCE,
            confidence=confidence,
        )
        return self._dump(self._run_cv(model_id, request, api_key))

    def run_semantic_segmentation(
        self,
        model_id: str,
        images: List[Any],
        api_key: Optional[str] = None,
        confidence: Union[float, str, None, _Unset] = UNSET,
        response_mask_format: str = "base64_png",
    ) -> List[dict]:
        request = SemanticSegmentationInferenceRequest(
            api_key=api_key,
            model_id=model_id,
            image=images,
            response_mask_format=response_mask_format,
            source=_WORKFLOW_SOURCE,
            **_passed(confidence=confidence),
        )
        return self._dump(self._run_cv(model_id, request, api_key))

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
    ) -> Union[List[dict], InferenceResultsDC]:
        request = InstanceSegmentationInferenceRequest(
            api_key=api_key,
            model_id=model_id,
            image=images,
            disable_active_learning=disable_active_learning,
            active_learning_target_dataset=active_learning_target_dataset,
            class_agnostic_nms=class_agnostic_nms,
            class_filter=class_filter,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            max_candidates=max_candidates,
            mask_decode_mode=mask_decode_mode,
            tradeoff_factor=tradeoff_factor,
            source=_WORKFLOW_SOURCE,
            confidence=confidence,
            **_passed(
                response_mask_format=response_mask_format,
                enforce_dense_masks_in_inference_models=enforce_dense_masks_in_inference_models,
                stream_pipeline_context_id=stream_pipeline_context_id,
            ),
        )
        responses = self._run_instance_segmentation(
            model_id, request, api_key, stream_pipeline_context_id
        )
        if return_raw_responses:
            return InferenceResultsDC(predictions=[], raw_responses=responses)
        return self._dump(responses)

    def _run_instance_segmentation(
        self,
        model_id: str,
        request: Any,
        api_key: Optional[str],
        stream_pipeline_context_id: Union[str, None, _Unset],
    ) -> List[Any]:
        if not isinstance(stream_pipeline_context_id, str):
            return self._run_cv(model_id, request, api_key)
        key = self._key_for(model_id, api_key)
        route = self._resolve(model_id, key)
        images, _ = as_image_list(request.image)
        if route.stream_pipeline_depth <= 1 or len(images) != 1:
            return self._run_cv(model_id, request, api_key, route=route)

        responses = self._run_stream_pipelined(
            request, key, route, stream_pipeline_context_id
        )

        return responses

    def _run_stream_pipelined(
        self, request: Any, key: Optional[str], route: Route, context_id: str
    ) -> List[Any]:
        ensure_request_supported(route.model_id, request, route)
        payloads = self._request_payloads(request)
        params = build_task_params(route.task_type, route.action, request, route)
        params[STREAM_PIPELINE_CONTEXT_ID_KWARG] = context_id
        params[STREAM_PIPELINE_PRODUCER_ID_KWARG] = self._producer_id
        current = _StreamFrame(
            route=route, request=request, dims=(payloads[0].width, payloads[0].height)
        )
        frames = self._stream_frames_for(route)

        started = time.perf_counter()
        predictions = self._bridge.infer(route, key, route.action, payloads, params)
        elapsed = time.perf_counter() - started
        prediction = predictions[0]
        response = self._repack(prediction, payloads[0], route, request, elapsed)

        response_future = get_async_response_future(prediction)
        response_context_id = get_async_response_context_id(prediction)
        if isinstance(response_future, Future):
            frame = _pop_stream_frame(frames, response_context_id) or current
            attach_async_response_future(
                response,
                self._chain_stream_response(response_future, frame, elapsed),
                response_context_id,
            )
            frames[context_id] = current
        elif response_context_id == context_id:
            if frames:
                logger.debug(
                    "Dropping %d stream frames of '%s' left by a previous pipeline: %s",
                    len(frames),
                    route.registry_id,
                    sorted(frames),
                )
                frames.clear()
            frames[context_id] = current
        else:
            attach_async_response_future(
                response, _completed_future([response]), context_id
            )
        pingback.record_inference(route.registry_id, request, [response])

        return [response]

    def _stream_frames_for(self, route: Route) -> Dict[str, _StreamFrame]:
        recorded = self._stream_frames.get(route.registry_id)
        if recorded is None or not recorded.recorded_under(route):
            recorded = _StreamFrames(
                route=route, loaded_monotonic=route.loaded_monotonic
            )
            self._stream_frames[route.registry_id] = recorded

        return recorded.frames

    def _chain_stream_response(
        self, response_future: Future, frame: _StreamFrame, elapsed: float
    ) -> Future:
        chained: Future = Future()

        def _repack_frame(finished: Future) -> None:
            try:
                responses = [
                    self._repack_stream_frame(prediction, frame, elapsed)
                    for prediction in finished.result()
                ]
            except Exception as error:
                chained.set_exception(error)
                return None
            chained.set_result(responses)
            return None

        response_future.add_done_callback(_repack_frame)

        return chained

    def _repack_stream_frame(
        self, prediction: Any, frame: _StreamFrame, elapsed: float
    ) -> Any:
        width, height = frame.dims
        payload = SimpleNamespace(width=width, height=height)

        response = self._repack(
            prediction, payload, frame.route, frame.request, elapsed
        )

        return response

    def run_lmm(
        self,
        model_id: str,
        image: Any,
        prompt: str,
        api_key: Optional[str] = None,
        enable_thinking: Union[bool, None, _Unset] = UNSET,
        max_new_tokens: Optional[int] = None,
    ) -> dict:
        kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "model_id": model_id,
            "image": image,
            "source": _WORKFLOW_SOURCE,
            "prompt": prompt,
            **_passed(enable_thinking=enable_thinking),
        }
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens
        request = LMMInferenceRequest(**kwargs)
        return self._dump(self._run_vlm(model_id, request, api_key))[0]

    def run_depth_estimation(self, model_id: str, image: Any) -> Any:
        request = DepthEstimationRequest(image=image, source=_WORKFLOW_SOURCE)
        key = self._key_for(model_id)
        route = self._resolve(model_id, key)
        payloads = self._request_payloads(request)
        predictions = self._bridge.infer(route, key, route.action, payloads, {})
        depth = repack_depth_estimation(predictions[0])
        pingback.record_inference(route.registry_id, request, depth)
        return {
            "normalized_depth": depth["normalized_depth"],
            "image": SimpleNamespace(base64_image=depth["image"]["base64_image"]),
        }

    def run_moondream2(
        self,
        model_id: str,
        image: Any,
        prompt: str,
        text: List[str],
        api_key: Optional[str] = None,
    ) -> dict:
        request = Moondream2InferenceRequest(
            api_key=api_key,
            model_id=model_id,
            image=image,
            text=text,
            prompt=prompt,
            source=_WORKFLOW_SOURCE,
        )
        return self._dump(self._run_vlm(model_id, request, api_key))[0]

    def run_clip_text_embedding(
        self,
        model_id: str,
        version_id: str,
        text: List[str],
        api_key: Optional[str] = None,
    ) -> List[List[float]]:
        request = ClipTextEmbeddingRequest(
            clip_version_id=version_id,
            text=text,
            api_key=api_key,
            source=_WORKFLOW_SOURCE,
        )
        return self._run_embedding(model_id, request, api_key).embeddings

    def run_clip_image_embedding(
        self,
        model_id: str,
        version_id: str,
        images: List[Any],
        api_key: Optional[str] = None,
    ) -> List[List[float]]:
        request = ClipImageEmbeddingRequest(
            clip_version_id=version_id,
            image=images,
            api_key=api_key,
            source=_WORKFLOW_SOURCE,
        )
        return self._run_embedding(model_id, request, api_key).embeddings

    def run_clip_comparison(
        self,
        subject: Any,
        subject_type: str,
        prompt: Any,
        prompt_type: str,
        api_key: Optional[str] = None,
        version_id: Union[str, None, _Unset] = UNSET,
    ) -> dict:
        request = ClipCompareRequest(
            api_key=api_key,
            subject=subject,
            subject_type=subject_type,
            prompt=prompt,
            prompt_type=prompt_type,
            source=_WORKFLOW_SOURCE,
            **_passed(clip_version_id=version_id),
        )
        core_model_id = f"clip/{request.clip_version_id}"
        self.add_model(core_model_id, api_key)
        return self._run_embedding(core_model_id, request, api_key).model_dump()

    def run_perception_encoder_text_embedding(
        self,
        model_id: str,
        version_id: str,
        text: List[str],
        api_key: Optional[str] = None,
    ) -> List[List[float]]:
        request = PerceptionEncoderTextEmbeddingRequest(
            perception_encoder_version_id=version_id,
            text=text,
            api_key=api_key,
            source=_WORKFLOW_SOURCE,
        )
        return self._run_embedding(model_id, request, api_key).embeddings

    def run_perception_encoder_image_embedding(
        self,
        model_id: str,
        version_id: str,
        images: List[Any],
        api_key: Optional[str] = None,
    ) -> List[List[float]]:
        request = PerceptionEncoderImageEmbeddingRequest(
            perception_encoder_version_id=version_id,
            image=images,
            api_key=api_key,
            source=_WORKFLOW_SOURCE,
        )
        return self._run_embedding(model_id, request, api_key).embeddings

    def run_doctr_ocr(
        self,
        model_id: str,
        image: Any,
        api_key: Optional[str] = None,
        generate_bounding_boxes: Union[bool, None, _Unset] = UNSET,
    ) -> dict:
        request = DoctrOCRInferenceRequest(
            image=image,
            api_key=api_key,
            source=_WORKFLOW_SOURCE,
            **_passed(generate_bounding_boxes=generate_bounding_boxes),
        )
        return self._dump(self._run_ocr(model_id, request, api_key))[0]

    def run_easy_ocr(
        self,
        model_id: str,
        version_id: str,
        image: Any,
        api_key: Optional[str] = None,
        language_codes: Optional[List[str]] = None,
        quantize: Optional[bool] = None,
    ) -> dict:
        request = EasyOCRInferenceRequest(
            easy_ocr_version_id=version_id,
            image=image,
            api_key=api_key,
            language_codes=language_codes,
            quantize=quantize,
            source=_WORKFLOW_SOURCE,
        )
        return self._dump(self._run_ocr(model_id, request, api_key))[0]

    def run_pp_ocr(
        self,
        image: Any,
        api_key: Optional[str] = None,
        text_detection: Union[str, None, _Unset] = UNSET,
        text_recognition: Union[str, None, _Unset] = UNSET,
    ) -> dict:
        request = PPOCRInferenceRequest(
            image=image,
            api_key=api_key,
            source=_WORKFLOW_SOURCE,
            **_passed(text_detection=text_detection, text_recognition=text_recognition),
        )
        core_model_id = request.model_id
        self.add_model(core_model_id, api_key)
        return self._dump(self._run_ocr(core_model_id, request, api_key))[0]

    def run_yolo_world(
        self,
        model_id: str,
        version_id: str,
        image: Any,
        text: List[str],
        api_key: Optional[str] = None,
        confidence: Optional[Union[float, str]] = None,
    ) -> dict:
        request = YOLOWorldInferenceRequest(
            image=image,
            yolo_world_version_id=version_id,
            confidence=confidence,
            text=text,
            api_key=api_key,
            source=_WORKFLOW_SOURCE,
        )
        return self._dump(self._run_open_vocabulary(model_id, request, api_key))[0]

    @staticmethod
    def _sam2_prompt_set(prompts: List[dict]) -> Sam2PromptSet:
        revived = []
        for prompt in prompts:
            if "box" not in prompt and "points" not in prompt:
                raise ValueError(
                    f"SAM2 prompt must carry 'box' or 'points'; got {sorted(prompt)}"
                )
            kwargs: Dict[str, Any] = {}
            if "box" in prompt:
                kwargs["box"] = Box(**prompt["box"])
            if "points" in prompt:
                kwargs["points"] = [Point(**point) for point in prompt["points"]]
            revived.append(Sam2Prompt(**kwargs))
        return Sam2PromptSet(prompts=revived)

    def run_sam2_segmentation(
        self,
        model_id: str,
        image: Any,
        prompts: List[dict],
        api_key: Optional[str] = None,
        version_id: Union[str, None, _Unset] = UNSET,
        request_model_id: Union[str, None, _Unset] = UNSET,
        multimask_output: Union[bool, None, _Unset] = UNSET,
        threshold: Union[float, None, _Unset] = UNSET,
    ) -> List[Any]:
        request = Sam2SegmentationRequest(
            image=image,
            api_key=api_key,
            source=_WORKFLOW_SOURCE,
            prompts=self._sam2_prompt_set(prompts),
            **_passed(
                sam2_version_id=version_id,
                model_id=request_model_id,
                multimask_output=multimask_output,
                threshold=threshold,
            ),
        )
        return [self._run_interactive_segmentation(model_id, request, api_key)]

    def run_sam3_segmentation(
        self,
        model_id: str,
        image: Any,
        prompts: List[dict],
        api_key: Optional[str] = None,
        output_prob_thresh: Union[float, None, _Unset] = UNSET,
        nms_iou_threshold: Union[float, None, _Unset] = UNSET,
        format: Union[str, None, _Unset] = UNSET,
    ) -> List[Any]:
        request = Sam3SegmentationRequest(
            api_key=api_key,
            model_id=model_id,
            image=image,
            prompts=[Sam3Prompt(**prompt) for prompt in prompts],
            source=_WORKFLOW_SOURCE,
            **_passed(
                output_prob_thresh=output_prob_thresh,
                nms_iou_threshold=nms_iou_threshold,
                format=format,
            ),
        )
        return [self._run_interactive_segmentation(model_id, request, api_key)]

    def run_sam3_3d_objects(
        self,
        model_id: str,
        image: Any,
        mask_input: Any,
        api_key: Optional[str] = None,
    ) -> Any:
        raise LegacyHTTPError(501, f"{_SAM3_3D_UNAVAILABLE}.")

    def run_tensor_native_inference(self, model_id: str, **kwargs: Any) -> Any:
        key = self._key_for(model_id)
        route = self._resolve(model_id, key)
        if route.task_type not in SUPPORTED_TASK_TYPES:
            raise LegacyHTTPError(
                501,
                f"tensor-native execution is not available for {route.task_type} "
                "on inference_server",
            )
        action = native_action(route, kwargs.pop("action", None))
        images = kwargs.pop("images", None)
        params = native_params(
            route.task_type,
            kwargs,
            action=action,
            model_class_name=route.model_class_name,
        )
        if images is None:
            raw = [self._bridge.infer_params_only(route, key, action, params)]
        else:
            payloads = native_image_payloads(
                images, ndarray_ok=self._bridge.accepts_ndarray
            )
            raw = self._bridge.infer(route, key, action, payloads, params)

        result = assemble_native_result(
            route.task_type,
            raw,
            environment.WORKFLOWS_IMAGE_TENSOR_DEVICE,
            action=action,
            model_class_name=route.model_class_name,
        )

        return result

    def load_action_recognition_model(
        self, model_id: str, api_key: Optional[str] = None, **kwargs: Any
    ) -> Any:
        key = self._key_for(model_id, api_key)
        route = self._resolve(model_id, key)
        ensure_action_recognition_route(model_id, route)
        model = _ActionRecognitionModelProxy(self._bridge, route, key)

        return model

    def get_class_names(self, model_id: str) -> List[str]:
        return list(self._route_for(model_id).class_names or [])

    def get_keypoints_classes(self, model_id: str) -> List[List[str]]:
        return list(self._route_for(model_id).key_points_classes or [])

    def model_supports_stream_pipeline(self, model_id: str) -> bool:
        return self.get_model_pipeline_depth(model_id) > 1

    def get_model_pipeline_depth(self, model_id: str) -> int:
        route = self._routes.get(model_id)
        if route is None:
            return 1
        return route.stream_pipeline_depth

    def flush_model_stream_pipeline(self, model_id: str) -> Optional[List[Any]]:
        route = self._routes.get(model_id)
        if route is None or route.stream_pipeline_depth <= 1:
            return None
        recorded = self._stream_frames.pop(route.registry_id, None)
        predictions = self._bridge.flush_model_stream_pipeline(route.registry_id)
        if predictions is None:
            return None
        frames: Dict[str, _StreamFrame] = {}
        if recorded is not None and recorded.recorded_under(route):
            frames = recorded.frames
        responses = []
        for prediction in predictions:
            frame = _pop_stream_frame(frames, get_async_response_context_id(prediction))
            if frame is None:
                frame = _StreamFrame(
                    route=route,
                    request=InstanceSegmentationInferenceRequest(
                        model_id=route.model_id, image=[], source=_WORKFLOW_SOURCE
                    ),
                    dims=_prediction_dims(prediction),
                )
            responses.append(self._repack_stream_frame(prediction, frame, 0.0))
        if frames:
            logger.warning(
                "Dropping %d stream frames of '%s' left unmatched by the flush: %s",
                len(frames),
                route.registry_id,
                sorted(frames),
            )

        return responses

    def shutdown_model_stream_pipeline(self, model_id: str) -> None:
        route = self._routes.get(model_id)
        if route is None or route.stream_pipeline_depth <= 1:
            return None
        self._stream_frames.pop(route.registry_id, None)
        self._bridge.shutdown_model_stream_pipeline(route.registry_id)

        return None

    def __contains__(self, model_id: str) -> bool:
        return self._registration_keys.get(model_id, model_id) in self._bridge

    def _resolve(self, model_id: str, api_key: Optional[str]) -> Route:
        route = self._resolve_route(model_id, api_key)
        self._routes[model_id] = route
        return route

    def _route_for(self, model_id: str) -> Route:
        route = self._routes.get(model_id)
        if route is not None:
            return route
        return self._resolve(model_id, self._key_for(model_id))

    def _payloads(self, images: List[Any]) -> List[ImagePayload]:
        limit_error = too_many_images(len(images))
        if limit_error is not None:
            raise _error_from_response(limit_error)
        payloads: List[ImagePayload] = []
        for image in images:
            image_type, value = split_image(image)
            if image_type == "url":
                data = self._bridge.fetch_image(value)
                width, height = image_dims(data)
                payloads.append(ImagePayload(data, width, height))
                continue
            payloads.append(
                decode_inline_image(image, ndarray_ok=self._bridge.accepts_ndarray)
            )
        return payloads

    def _request_payloads(self, request: Any) -> List[ImagePayload]:
        images, _ = as_image_list(request.image)
        return self._payloads(images)

    def _run_cv(
        self,
        model_id: str,
        request: Any,
        api_key: Optional[str],
        extra_params: Optional[Dict[str, Any]] = None,
        route: Optional[Route] = None,
    ) -> List[Any]:
        key = self._key_for(model_id, api_key)
        if route is None:
            route = self._resolve(model_id, key)
        ensure_request_supported(model_id, request, route)
        payloads = self._request_payloads(request)
        params = build_task_params(route.task_type, route.action, request, route)
        params.update(extra_params or {})
        started = time.perf_counter()
        predictions = self._bridge.infer(route, key, route.action, payloads, params)
        elapsed = time.perf_counter() - started
        responses = [
            self._repack(prediction, payload, route, request, elapsed)
            for prediction, payload in zip(predictions, payloads)
        ]
        pingback.record_inference(route.registry_id, request, responses)
        return responses

    def _repack(
        self, prediction: Any, payload: Any, route: Route, request: Any, elapsed: float
    ) -> Any:
        response = self._stamp(
            repack_prediction(
                route.task_type,
                route.action,
                prediction,
                (payload.width, payload.height),
                route,
                request,
            ),
            route,
            elapsed,
            request,
        )

        return response

    def _run_vlm(
        self, model_id: str, request: Any, api_key: Optional[str]
    ) -> List[Any]:
        key = self._key_for(model_id, api_key)
        route = self._resolve(model_id, key)
        ensure_request_supported(model_id, request, route)
        action = resolve_request_action(route, request)
        if action == "detect":
            params = {"classes": [getattr(request, "prompt", None)]}
        else:
            params = build_vlm_params(request, model_class_name=route.model_class_name)
        payloads = self._request_payloads(request)
        started = time.perf_counter()
        predictions = self._bridge.infer(route, key, action, payloads, params)
        elapsed = time.perf_counter() - started
        responses = []
        for prediction, payload in zip(predictions, payloads):
            dims = (payload.width, payload.height)
            if action == "detect":
                response = repack_moondream_detection(prediction, request, dims)
            else:
                response = repack_vlm_response(prediction, dims)
            responses.append(self._stamp(response, route, elapsed, request))
        pingback.record_inference(route.registry_id, request, responses)
        return responses

    def _run_ocr(
        self, model_id: str, request: Any, api_key: Optional[str]
    ) -> List[Any]:
        ensure_ocr_request_supported(request)
        key = self._key_for(model_id, api_key)
        route = self._resolve(model_id, key)
        payloads = self._request_payloads(request)
        started = time.perf_counter()
        predictions = self._bridge.infer(route, key, route.action, payloads, {})
        elapsed = time.perf_counter() - started
        responses = []
        for prediction, payload in zip(predictions, payloads):
            response = repack_structured_ocr_response(
                prediction, (payload.width, payload.height), route.class_names, request
            )
            responses.append(self._stamp(response, route, elapsed))
        pingback.record_inference(route.registry_id, request, responses)
        return responses

    def _run_open_vocabulary(
        self, model_id: str, request: Any, api_key: Optional[str]
    ) -> List[Any]:
        params = build_open_vocabulary_params(request)
        key = self._key_for(model_id, api_key)
        route = self._resolve(model_id, key)
        ensure_request_supported(model_id, request, route)
        class_names = requested_open_vocabulary_classes(request)
        payloads = self._request_payloads(request)
        started = time.perf_counter()
        predictions = self._bridge.infer(route, key, route.action, payloads, params)
        elapsed = time.perf_counter() - started
        responses = [
            self._stamp(
                repack_object_detection_response(
                    prediction, (payload.width, payload.height), class_names, request
                ),
                route,
                elapsed,
                request,
            )
            for prediction, payload in zip(predictions, payloads)
        ]
        pingback.record_inference(route.registry_id, request, responses)
        return responses

    def _run_embedding(
        self, model_id: str, request: Any, api_key: Optional[str]
    ) -> Any:
        key = self._key_for(model_id, api_key)
        route = self._resolve(model_id, key)
        action = resolve_request_action(route, request)
        calls, prompt_keys = build_embedding_calls(action, request)
        image_positions = [
            position for position, call in enumerate(calls) if call["image"] is not None
        ]
        payloads = self._payloads(
            [calls[position]["image"] for position in image_positions]
        )
        payload_by_position = dict(zip(image_positions, payloads))
        self._bridge.ensure_loaded(route, key)
        started = time.perf_counter()
        results = []
        with measure_inference(route.registry_id, responses=1):
            for position, call in enumerate(calls):
                payload = payload_by_position.get(position)
                if payload is None:
                    results.append(
                        self._bridge.infer_params_only(
                            route, key, call["action"], call["params"], record=False
                        )
                    )
                    continue
                results.extend(
                    self._bridge.infer(
                        route,
                        key,
                        call["action"],
                        [payload],
                        call["params"],
                        record=False,
                    )
                )
        elapsed = time.perf_counter() - started
        record_telemetry(
            telemetry.record_inference,
            requested_model_id_for(route.registry_id),
            elapsed,
        )
        response = repack_embedding_response(action, request, results, prompt_keys)
        self._stamp(response, route, elapsed)
        pingback.record_inference(route.registry_id, request, response)
        return response

    def _run_interactive_segmentation(
        self, model_id: str, request: Any, api_key: Optional[str]
    ) -> Any:
        key = self._key_for(model_id, api_key)
        route = self._resolve(model_id, key)
        action = resolve_request_action(route, request)
        params = build_interactive_segmentation_params(action, request, key)
        image = getattr(request, "image", None)
        started = time.perf_counter()
        if image is None:
            prediction = self._bridge.infer_params_only(route, key, action, params)
        else:
            images, _ = as_image_list(image)
            payloads = self._payloads(images)
            prediction = self._bridge.infer(route, key, action, payloads, params)[0]
        elapsed = time.perf_counter() - started
        response = repack_interactive_segmentation_response(
            action, prediction, request, key
        )
        self._stamp(response, route, elapsed, request)
        pingback.record_inference(route.registry_id, request, response)
        return response

    @staticmethod
    def _stamp(response: Any, route: Route, elapsed: float, request: Any = None) -> Any:
        response.time = elapsed
        response.resolved_model = resolved_model_for(route)
        if request is not None:
            response.inference_id = request.id
        return response

    @staticmethod
    def _dump(responses: List[Any]) -> List[dict]:
        return [
            response.model_dump(by_alias=True, exclude_none=True)
            for response in responses
        ]


def _completed_future(responses: List[Any]) -> Future:
    future: Future = Future()
    future.set_result(responses)

    return future


def _pop_stream_frame(
    frames: Dict[str, _StreamFrame], context_id: Optional[str]
) -> Optional[_StreamFrame]:
    if context_id is not None and context_id in frames:
        return frames.pop(context_id)
    if frames:
        return frames.pop(next(iter(frames)))
    return None


def _prediction_dims(prediction: Any) -> Tuple[int, int]:
    image_size = getattr(getattr(prediction, "mask", None), "image_size", None)
    if not image_size:
        return 0, 0
    height, width = image_size
    return int(width), int(height)
