from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

from roboflow_workflows import environment
from roboflow_workflows.prototypes.models_provider import (
    UNSET,
    InferenceResultsDC,
    _Unset,
)

from inference_models.utils import model_blob_cache
from inference_server.framework.input_parsers.image_limits import too_many_images
from inference_server.legacy.bridge import Route, SyncLegacyBridge, resolved_model_for
from inference_server.legacy.common import (
    ImagePayload,
    _error_from_response,
    _image_attribute,
    as_image_list,
    decode_inline_image,
    image_dims,
)
from inference_server.legacy.entities import (
    ClassificationInferenceRequest,
    ClipCompareRequest,
    ClipImageEmbeddingRequest,
    ClipTextEmbeddingRequest,
    DepthEstimationRequest,
    DoctrOCRInferenceRequest,
    EasyOCRInferenceRequest,
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
from inference_server.legacy.prompts import (
    Box,
    Point,
    Sam2Prompt,
    Sam2PromptSet,
    Sam3Prompt,
)
from inference_server.legacy.translation import (
    build_embedding_calls,
    build_interactive_segmentation_params,
    build_open_vocabulary_params,
    build_task_params,
    build_vlm_params,
    ensure_ocr_request_supported,
    ensure_request_supported,
    repack_depth_estimation,
    repack_embedding_response,
    repack_interactive_segmentation_response,
    repack_moondream_detection,
    repack_object_detection_response,
    repack_prediction,
    repack_structured_ocr_response,
    repack_vlm_response,
    requested_open_vocabulary_classes,
    resolve_request_action,
)
from inference_server.workflows.tensor_native import (
    SUPPORTED_TASK_TYPES,
    assemble_native_result,
    native_image_payloads,
    native_params,
)

_WORKFLOW_SOURCE = "workflow-execution"
_SAM3_3D_UNAVAILABLE = (
    "SAM3 3D object reconstruction is not available on inference_server"
)
_ACTION_RECOGNITION_UNAVAILABLE = (
    "Action recognition models are not available on inference_server"
)


def _passed(**arguments: Any) -> Dict[str, Any]:
    return {
        name: value
        for name, value in arguments.items()
        if not isinstance(value, _Unset)
    }


class GatewayModelsProvider:
    """Implements roboflow_workflows.prototypes.models_provider.ModelsProvider on top of SyncLegacyBridge. One instance per workflow request."""

    def __init__(self, bridge: SyncLegacyBridge, api_key: Optional[str]) -> None:
        self._bridge = bridge
        self._api_key = api_key
        self._model_keys: Dict[str, Optional[str]] = {}
        self._routes: Dict[str, Route] = {}
        self._artifact_cache: Any = None
        self._artifact_cache_resolved = False

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
        self._resolve(model_id, key)

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
        responses = self._run_cv(model_id, request, api_key)
        if return_raw_responses:
            return InferenceResultsDC(predictions=[], raw_responses=responses)
        return self._dump(responses)

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
        request = DepthEstimationRequest(image=image)
        key = self._key_for(model_id)
        route = self._resolve(model_id, key)
        payloads = self._request_payloads(request)
        predictions = self._bridge.infer(route, key, route.action, payloads, {})
        depth = repack_depth_estimation(predictions[0])
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
            clip_version_id=version_id, text=text, api_key=api_key
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
            clip_version_id=version_id, image=images, api_key=api_key
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
            perception_encoder_version_id=version_id, text=text, api_key=api_key
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
            perception_encoder_version_id=version_id, image=images, api_key=api_key
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
            **_passed(text_detection=text_detection, text_recognition=text_recognition),
        )
        core_model_id = f"pp_ocr/{request.pp_ocr_version_id}"
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
        images = kwargs.pop("images")
        payloads = native_image_payloads(
            images, ndarray_ok=self._bridge.accepts_ndarray
        )
        params = native_params(route.task_type, kwargs)
        raw = self._bridge.infer(route, key, route.action, payloads, params)
        return assemble_native_result(
            route.task_type, raw, environment.WORKFLOWS_IMAGE_TENSOR_DEVICE
        )

    def load_action_recognition_model(
        self, model_id: str, api_key: Optional[str] = None, **kwargs: Any
    ) -> Any:
        raise LegacyHTTPError(
            501, f"{_ACTION_RECOGNITION_UNAVAILABLE} for model '{model_id}'."
        )

    def get_class_names(self, model_id: str) -> List[str]:
        return list(self._route_for(model_id).class_names or [])

    def get_keypoints_classes(self, model_id: str) -> List[List[str]]:
        return list(self._route_for(model_id).key_points_classes or [])

    def model_supports_stream_pipeline(self, model_id: str) -> bool:
        return False

    def get_model_pipeline_depth(self, model_id: str) -> int:
        return 0

    def flush_model_stream_pipeline(self, model_id: str) -> Optional[List[Any]]:
        return None

    def shutdown_model_stream_pipeline(self, model_id: str) -> None:
        return None

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._bridge

    def _resolve(self, model_id: str, api_key: Optional[str]) -> Route:
        route = self._bridge.resolve(model_id, api_key)
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
            if _image_attribute(image, "type") == "url":
                data = self._bridge.fetch_image(_image_attribute(image, "value"))
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
    ) -> List[Any]:
        key = self._key_for(model_id, api_key)
        route = self._resolve(model_id, key)
        ensure_request_supported(model_id, request, route)
        payloads = self._request_payloads(request)
        params = build_task_params(route.task_type, route.action, request, route)
        params.update(extra_params or {})
        started = time.perf_counter()
        predictions = self._bridge.infer(route, key, route.action, payloads, params)
        elapsed = time.perf_counter() - started
        return [
            self._stamp(
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
            for prediction, payload in zip(predictions, payloads)
        ]

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
            params = build_vlm_params(request)
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
        return [
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
        started = time.perf_counter()
        results = []
        for position, call in enumerate(calls):
            payload = payload_by_position.get(position)
            if payload is None:
                results.append(
                    self._bridge.infer_params_only(
                        route, key, call["task"], call["params"]
                    )
                )
                continue
            results.extend(
                self._bridge.infer(route, key, call["task"], [payload], call["params"])
            )
        elapsed = time.perf_counter() - started
        response = repack_embedding_response(action, request, results, prompt_keys)
        return self._stamp(response, route, elapsed)

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
        return self._stamp(response, route, elapsed, request)

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
