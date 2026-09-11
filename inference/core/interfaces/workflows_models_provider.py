"""Server-side implementation of the Workflows `ModelsProvider` port.

Workflow blocks describe an inference call with plain arguments; this adapter
turns those into the server's pydantic request objects and runs them through
`ModelManager`, so `inference.core.workflows` never imports
`inference.core.entities`. Every `run_*` method reproduces, argument for
argument, the request a block used to build inline - see
`tests/inference/unit_tests/core/interfaces/test_workflows_models_provider.py`,
which pins each one against the pydantic class.

MODEL REGISTRATION IS NOT DONE HERE, with two exceptions. `add_model` /
`load_core_model` stay in the blocks: `core_steps/models/roboflow/
instance_segmentation/v3.py` registers the model, then reads its pipeline depth
to decide whether to queue a stream frame context, and only then infers.
Registering inside the inference call would make that check report "no
pipeline" on a cold model and the first delayed response would find no pending
context. The exceptions are `run_clip_comparison` and `run_pp_ocr`, whose core
model id exists only on the validated request; they register in the position
the blocks used (build -> register -> infer).

Bound at the four composition roots as
`init_parameters["workflows_core.model_manager"]`, and by the test fixtures
that used to inject a raw manager (Task 11.7 Step 6b).

Phase 12 note: a second implementation backed by `inference_sdk`
(`InferenceHTTPClientModelsProvider`) would satisfy the same port for REMOTE
step execution. Keep the argument names aligned with
`inference_sdk.http.entities.InferenceConfiguration`.
"""

from typing import Any, Dict, List, Optional, Union

from inference.core.entities.requests.clip import (
    ClipCompareRequest,
    ClipImageEmbeddingRequest,
    ClipTextEmbeddingRequest,
)
from inference.core.entities.requests.doctr import DoctrOCRInferenceRequest
from inference.core.entities.requests.easy_ocr import EasyOCRInferenceRequest
from inference.core.entities.requests.inference import (
    ClassificationInferenceRequest,
    DepthEstimationRequest,
    InstanceSegmentationInferenceRequest,
    KeypointsDetectionInferenceRequest,
    LMMInferenceRequest,
    ObjectDetectionInferenceRequest,
    SemanticSegmentationInferenceRequest,
)
from inference.core.entities.requests.moondream2 import Moondream2InferenceRequest
from inference.core.entities.requests.perception_encoder import (
    PerceptionEncoderImageEmbeddingRequest,
    PerceptionEncoderTextEmbeddingRequest,
)
from inference.core.entities.requests.pp_ocr import PPOCRInferenceRequest
from inference.core.entities.requests.sam2 import (
    Box,
    Point,
    Sam2Prompt,
    Sam2PromptSet,
    Sam2SegmentationRequest,
)
from inference.core.entities.requests.sam3 import Sam3Prompt, Sam3SegmentationRequest
from inference.core.entities.requests.sam3_3d import Sam3_3D_Objects_InferenceRequest
from inference.core.entities.requests.yolo_world import YOLOWorldInferenceRequest
from inference.core.managers.base import ModelManager
from inference.core.roboflow_api import ModelEndpointType
from inference.core.workflows.prototypes.models_provider import (
    UNSET,
    InferenceResultsDC,
    _Unset,
)

_WORKFLOW_SOURCE = "workflow-execution"


def _passed(**arguments: Any) -> Dict[str, Any]:
    """The request keywords the caller actually passed: drops UNSET, keeps None."""
    return {
        name: value
        for name, value in arguments.items()
        if not isinstance(value, _Unset)
    }


class ModelManagerModelsProvider:
    """Implements `inference.core.workflows.prototypes.models_provider.ModelsProvider`."""

    def __init__(self, model_manager: ModelManager):
        self._model_manager = model_manager

    # -- forwarded members -------------------------------------------------

    @property
    def content_addressed_artifact_cache(self) -> Any:
        return self._model_manager.content_addressed_artifact_cache

    def add_model(
        self,
        model_id: str,
        api_key: str,
        model_id_alias: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # Forwarded as the block made the call (keywords, no synthesised
        # `model_id_alias=None`), so a class-level test patch on
        # `ModelManager.add_model` observes the same call it observes today.
        if model_id_alias is not None:
            kwargs["model_id_alias"] = model_id_alias
        return self._model_manager.add_model(
            model_id=model_id, api_key=api_key, **kwargs
        )

    def infer_from_request_sync(
        self, model_id: str, request: Any, **kwargs: Any
    ) -> Any:
        return self._model_manager.infer_from_request_sync(
            model_id=model_id, request=request, **kwargs
        )

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
            confidence=confidence,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            max_candidates=max_candidates,
            source=_WORKFLOW_SOURCE,
        )
        return self._dump(self._infer(model_id=model_id, request=request))

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
            confidence=confidence,
            disable_active_learning=disable_active_learning,
            source=_WORKFLOW_SOURCE,
            active_learning_target_dataset=active_learning_target_dataset,
        )
        return self._dump(
            self._infer(model_id=model_id, request=request, **(inference_kwargs or {}))
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
            confidence=confidence,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            max_candidates=max_candidates,
            keypoint_confidence=keypoint_confidence,
            source=_WORKFLOW_SOURCE,
        )
        return self._dump(self._infer(model_id=model_id, request=request))

    def run_semantic_segmentation(
        self,
        model_id: str,
        images: List[Any],
        api_key: Optional[str] = None,
        confidence: Union[float, str, None, _Unset] = UNSET,
        response_mask_format: str = "base64_png",
    ) -> List[dict]:
        # UNSET means "v1, which never sets confidence" -> omitted, so the
        # pydantic default applies. An explicit None means "v2 resolved its
        # manifest to None" -> forwarded, so the ValidationError it raises today
        # still happens. Collapsing the two would be a silent behaviour change.
        request = SemanticSegmentationInferenceRequest(
            api_key=api_key,
            model_id=model_id,
            image=images,
            response_mask_format=response_mask_format,
            source=_WORKFLOW_SOURCE,
            **_passed(confidence=confidence),
        )
        return self._dump(self._infer(model_id=model_id, request=request))

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
        # The three UNSET-defaulted fields are each set by only some of the
        # four block versions. Whatever a version passes - None included - is
        # forwarded (v1/v2/v3 pass enforce_dense_masks_in_inference_models even
        # when a selector resolved it to None, and the request stored that
        # None); what it does not pass is left to the pydantic default.
        request = InstanceSegmentationInferenceRequest(
            api_key=api_key,
            model_id=model_id,
            image=images,
            disable_active_learning=disable_active_learning,
            active_learning_target_dataset=active_learning_target_dataset,
            class_agnostic_nms=class_agnostic_nms,
            class_filter=class_filter,
            confidence=confidence,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            max_candidates=max_candidates,
            mask_decode_mode=mask_decode_mode,
            tradeoff_factor=tradeoff_factor,
            source=_WORKFLOW_SOURCE,
            **_passed(
                response_mask_format=response_mask_format,
                enforce_dense_masks_in_inference_models=enforce_dense_masks_in_inference_models,
                stream_pipeline_context_id=stream_pipeline_context_id,
            ),
        )
        responses = self._infer(model_id=model_id, request=request)
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
            # Forwarded exactly as the block passed it (five blocks do; the
            # request field is a non-optional bool, so a None keeps raising).
            **_passed(enable_thinking=enable_thinking),
        }
        # The blocks only ever added max_new_tokens to the request when it was
        # not None (`if max_new_tokens is not None: request_kwargs[...] = ...`);
        # mirror that, so the pydantic default applies otherwise.
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens
        request = LMMInferenceRequest(**kwargs)
        return self._dump(self._infer(model_id=model_id, request=request))[0]

    def run_depth_estimation(self, model_id: str, image: Any) -> Any:
        request = DepthEstimationRequest(image=image)
        return self._infer(model_id=model_id, request=request)[0].response

    def run_moondream2(
        self,
        model_id: str,
        image: Any,
        prompt: str,
        text: List[str],
        api_key: Optional[str] = None,
    ) -> dict:
        # `text` is a required list on the request; the block passes `[]`.
        request = Moondream2InferenceRequest(
            api_key=api_key,
            model_id=model_id,
            image=image,
            text=text,
            prompt=prompt,
        )
        return self._dump(self._infer(model_id=model_id, request=request))[0]

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
        return self._infer(model_id=model_id, request=request)[0].embeddings

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
        return self._infer(model_id=model_id, request=request)[0].embeddings

    def run_clip_comparison(
        self,
        subject: Any,
        subject_type: str,
        prompt: Any,
        prompt_type: str,
        api_key: Optional[str] = None,
        version_id: Union[str, None, _Unset] = UNSET,
    ) -> dict:
        # v1 never sets the version (UNSET -> the pydantic default,
        # env.CLIP_VERSION_ID); v2 passes its `version` local, forwarded as is -
        # a None stays None, and the id below is then "clip/None", exactly what
        # `load_core_model` derived from the validated request before.
        request = ClipCompareRequest(
            api_key=api_key,
            subject=subject,
            subject_type=subject_type,
            prompt=prompt,
            prompt_type=prompt_type,
            **_passed(clip_version_id=version_id),
        )
        # Registration in the same position the block used: build -> register ->
        # infer. The id is only knowable from the validated request when the
        # version came from the pydantic default (clip_comparison/v1.py).
        core_model_id = f"clip/{request.clip_version_id}"
        self._model_manager.add_model(
            core_model_id, api_key, endpoint_type=ModelEndpointType.CORE_MODEL
        )
        # The two clip_comparison blocks used the BARE `model_dump()`, not the
        # by_alias/exclude_none form - keep that, the output keys depend on it.
        return self._infer(model_id=core_model_id, request=request)[0].model_dump()

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
        return self._infer(model_id=model_id, request=request)[0].embeddings

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
        return self._infer(model_id=model_id, request=request)[0].embeddings

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
        return self._dump(self._infer(model_id=model_id, request=request))[0]

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
        return self._dump(self._infer(model_id=model_id, request=request))[0]

    def run_pp_ocr(
        self,
        image: Any,
        api_key: Optional[str] = None,
        text_detection: Union[str, None, _Unset] = UNSET,
        text_recognition: Union[str, None, _Unset] = UNSET,
    ) -> dict:
        # The request has its own "omitted" sentinel with different semantics
        # from None (requests/pp_ocr.py:13): omitted -> "small", None -> the
        # stage is disabled. Forward only what the caller actually passed.
        request = PPOCRInferenceRequest(
            image=image,
            api_key=api_key,
            **_passed(text_detection=text_detection, text_recognition=text_recognition),
        )
        # The validator derived `pp_ocr_version_id` from the normalised stages;
        # `load_core_model` read exactly that attribute before, so the id is
        # unchanged. Registration keeps its position: build -> register ->
        # infer.
        core_model_id = f"pp_ocr/{request.pp_ocr_version_id}"
        self._model_manager.add_model(
            core_model_id, api_key, endpoint_type=ModelEndpointType.CORE_MODEL
        )
        return self._dump(self._infer(model_id=core_model_id, request=request))[0]

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
        return self._dump(self._infer(model_id=model_id, request=request))[0]

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
        # segment_anything2/v1.py passes sam2_version_id, threshold and
        # multimask_output; segment_anything3_interactive/v1.py passes model_id
        # and multimask_output. Each is forwarded exactly as passed - None
        # included: `multimask_output: bool` rejects it, as it did inline.
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
        return self._infer(model_id=model_id, request=request)

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
        return self._infer(model_id=model_id, request=request)

    def run_sam3_3d_objects(
        self,
        model_id: str,
        image: Any,
        mask_input: Any,
        api_key: Optional[str] = None,
    ) -> Any:
        request = Sam3_3D_Objects_InferenceRequest(
            image=image, mask_input=mask_input, api_key=api_key, model_id=model_id
        )
        return self._infer(model_id=model_id, request=request)[0]

    def run_tensor_native_inference(self, model_id: str, **kwargs: Any) -> Any:
        return self._model_manager.run_tensor_native_inference(
            model_id=model_id, **kwargs
        )

    def get_class_names(self, model_id: str) -> List[str]:
        return self._model_manager.get_class_names(model_id)

    def load_action_recognition_model(
        self, model_id: str, api_key: Optional[str] = None, **kwargs: Any
    ) -> Any:
        # Phase 9's loader (Task 9.9); the action block calls it with keywords.
        return self._model_manager.load_action_recognition_model(
            model_id=model_id, api_key=api_key, **kwargs
        )

    def get_keypoints_classes(self, model_id: str) -> List[List[str]]:
        return self._model_manager.get_keypoints_classes(model_id)

    def model_supports_stream_pipeline(self, model_id: str) -> bool:
        return self._model_manager.model_supports_stream_pipeline(model_id)

    def get_model_pipeline_depth(self, model_id: str) -> int:
        return self._model_manager.get_model_pipeline_depth(model_id)

    def flush_model_stream_pipeline(self, model_id: str) -> Optional[List[Any]]:
        return self._model_manager.flush_model_stream_pipeline(model_id)

    def shutdown_model_stream_pipeline(self, model_id: str) -> None:
        return self._model_manager.shutdown_model_stream_pipeline(model_id)

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._model_manager

    # -- shared helpers used by the run_* methods --------------------------

    def _infer(self, model_id: str, request: Any, **kwargs: Any) -> List[Any]:
        """Run one request and normalise the result to a list.

        `kwargs` are forwarded to `ModelManager.infer_from_request_sync`, which
        passes them on to `model_infer_sync` - two multi-label blocks send an
        extra `confidence` that way today.

        The caller has already registered `model_id` unless this adapter owns
        that registration (`run_clip_comparison`, `run_pp_ocr`).
        """
        responses = self._model_manager.infer_from_request_sync(
            model_id=model_id, request=request, **kwargs
        )
        if not isinstance(responses, list):
            responses = [responses]
        return responses

    @staticmethod
    def _dump(responses: List[Any]) -> List[dict]:
        """The dict form every block's `_post_process_result` consumes.

        Identical to the `e.model_dump(by_alias=True, exclude_none=True)` the
        blocks ran inline, with the `to_dict()` fast path the rfdetr adapter's
        dataclass responses expose (see
        `inference/core/entities/responses/inference.py`).
        """
        return [
            (
                response.to_dict()
                if callable(getattr(response, "to_dict", None))
                else response.model_dump(by_alias=True, exclude_none=True)
            )
            for response in responses
        ]
