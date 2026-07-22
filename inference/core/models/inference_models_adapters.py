import base64
import io
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from inspect import Parameter, signature
from io import BytesIO
from threading import local
from time import perf_counter
from typing import Any, Deque, List, Optional, Tuple, Union
from uuid import uuid4
from weakref import finalize

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils

from inference.core.entities.requests import (
    ClassificationInferenceRequest,
    InferenceRequest,
)
from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    InferenceResponse,
    InferenceResponseImage,
    InferenceResponseImageDC,
    InstanceSegmentationInferenceResponse,
    InstanceSegmentationInferenceResponseDC,
    InstanceSegmentationPrediction,
    InstanceSegmentationPredictionDC,
    InstanceSegmentationRLEPrediction,
    Keypoint,
    KeypointsDetectionInferenceResponse,
    KeypointsPrediction,
    LMMInferenceResponse,
    MultiLabelClassificationInferenceResponse,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
    Point,
    PointDC,
    SemanticSegmentationInferenceResponse,
    SemanticSegmentationPrediction,
)
from inference.core.env import (
    ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
    ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
    API_KEY,
    DISABLED_INFERENCE_MODELS_BACKENDS,
    GCP_SERVERLESS,
    RFDETR_ONNX_MAX_RESOLUTION,
    VALID_INFERENCE_MODELS_BACKENDS,
    WORKFLOWS_ASYNC_FUTURE_RESULT_TIMEOUT,
)
from inference.core.exceptions import PostProcessingError
from inference.core.models.base import Model
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.roboflow_api import get_extra_weights_provider_headers
from inference.core.utils.image_utils import load_image_bgr, load_image_rgb
from inference.core.utils.postprocess import bitpacked_masks2poly, mask2poly, masks2poly
from inference.core.utils.rle_to_polygon import rle_masks_to_polygons
from inference.core.utils.visualisation import draw_detection_predictions
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from inference.models.aliases import resolve_roboflow_model_alias
from inference_models import (
    AutoModel,
    ClassificationModel,
    ClassificationPrediction,
    DepthEstimationModel,
    Detections,
    InstanceDetections,
    InstanceSegmentationModel,
    KeyPoints,
    KeyPointsDetectionModel,
    MultiLabelClassificationModel,
    MultiLabelClassificationPrediction,
    ObjectDetectionModel,
    PreProcessingOverrides,
    SemanticSegmentationModel,
)
from inference_models.configuration import (
    INFERENCE_MODELS_RFDETR_TRITON_POSTPROC_ENABLED,
    MAX_RFDETR_PIPELINE_DEPTH,
    get_rfdetr_pipeline_depth,
)
from inference_models.models.base.async_handoff import (
    STREAM_PIPELINE_CONTEXT_ID_KWARG,
    adapter_gpu_work_submitted,
    attach_adapter_mapped_kwargs,
    attach_async_response_future,
    get_adapter_gpu_submit_generation,
    get_adapter_mapped_kwargs,
    get_adapter_stream_pipeline_context_id,
    get_deferred_postprocess_done_event,
    get_deferred_postprocess_finalizer,
    mark_adapter_gpu_work_submitted,
)
from inference_models.models.base.instance_segmentation import InferenceFuture
from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationResult,
)
from inference_models.models.base.types import InstancesRLEMasks, PreprocessingMetadata
from inference_models.models.common.rle_utils import torch_mask_to_coco_rle

DEFAULT_COLOR_PALETTE = [
    "#A351FB",
    "#FF4040",
    "#FFA1A0",
    "#FF7633",
    "#FFB633",
    "#D1D435",
    "#4CFB12",
    "#94CF1A",
    "#40DE8A",
    "#1B9640",
    "#00D6C1",
    "#2E9CAA",
    "#00C4FF",
    "#364797",
    "#6675FF",
    "#0019EF",
    "#863AFF",
    "#530087",
    "#CD3AFF",
    "#FF97CA",
    "#FF39C9",
]

_PINNED_HOST_BUFFER_CACHE_SIZE = 16
_PINNED_HOST_BUFFER_CONTEXT = local()


def get_pinned_buffer(name: str, shape, dtype: torch.dtype) -> torch.Tensor:
    """Return a thread-local pinned CPU scratch tensor for async DtoH copies.

    Response finalization can run on a worker thread while the inference thread
    submits later GPU work. Keeping this cache thread-local avoids two workers
    writing into the same scratch tensor. The small LRU cap prevents retaining a
    new pinned allocation for every transient shape.

    The cache is keyed by ``(name, dtype)`` only. When a cached buffer is large
    enough, this returns a **view** into that entry, not a fresh allocation.
    ``.numpy()`` and any slices alias the scratch memory until the next
    ``copy_`` into the same cache slot. Values that outlive the current
    finalization must be copied or reduced to scalars / fresh polygon arrays
    before returning.
    """
    cache = getattr(_PINNED_HOST_BUFFER_CONTEXT, "cache", None)
    if cache is None:
        cache = OrderedDict()
        _PINNED_HOST_BUFFER_CONTEXT.cache = cache
    key = (name, dtype)
    buf = cache.get(key)
    if buf is not None and all(buf.shape[i] >= shape[i] for i in range(len(shape))):
        cache.move_to_end(key)
        return buf[tuple(slice(0, s) for s in shape)]
    buf = torch.empty(shape, dtype=dtype, pin_memory=True)
    cache[key] = buf
    cache.move_to_end(key)
    while len(cache) > _PINNED_HOST_BUFFER_CACHE_SIZE:
        cache.popitem(last=False)
    return buf


def _resolve_response_future(
    future: Future,
    context: str,
):
    try:
        return future.result(timeout=WORKFLOWS_ASYNC_FUTURE_RESULT_TIMEOUT)
    except TimeoutError as error:
        raise RuntimeError(f"Timed out while waiting for {context}.") from error


class _PipelinePrimingSentinel:
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return "<_PIPELINE_PRIMING>"


_PIPELINE_PRIMING = _PipelinePrimingSentinel()


def _supports_independent_stage_execution(pre_process) -> bool:
    """Return whether preprocessing declares the composed-execution control."""
    try:
        parameters = signature(pre_process).parameters
    except (TypeError, ValueError):
        return False
    parameter = parameters.get("independent_stage_execution")
    return parameter is not None and parameter.kind in {
        Parameter.POSITIONAL_OR_KEYWORD,
        Parameter.KEYWORD_ONLY,
    }


class InferenceModelsObjectDetectionAdapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY
        model_id = resolve_roboflow_model_alias(model_id=model_id)

        self.task_type = "object-detection"

        extra_weights_provider_headers = get_extra_weights_provider_headers(
            countinference=kwargs.get("countinference"),
            service_secret=kwargs.get("service_secret"),
        )
        backend = list(
            VALID_INFERENCE_MODELS_BACKENDS.difference(
                DISABLED_INFERENCE_MODELS_BACKENDS
            )
        )
        self._model: ObjectDetectionModel = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            weights_provider_extra_headers=extra_weights_provider_headers,
            backend=backend,
            rf_detr_max_input_resolution=RFDETR_ONNX_MAX_RESOLUTION,
            **kwargs,
        )
        self._preprocess_supports_independent_stage_execution = (
            _supports_independent_stage_execution(self._model.pre_process)
        )
        self.class_names = list(self._model.class_names)

    def map_inference_kwargs(self, kwargs: dict) -> dict:
        kwargs["input_color_format"] = "bgr"
        pre_processing_overrides = PreProcessingOverrides(
            disable_contrast_enhancement=kwargs.get("disable_preproc_contrast", False),
            disable_grayscale=kwargs.get("disable_preproc_grayscale", False),
            disable_static_crop=kwargs.get("disable_preproc_static_crop", False),
        )
        kwargs["pre_processing_overrides"] = pre_processing_overrides
        return kwargs

    def preprocess(self, image: Any, **kwargs):
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        np_images: List[np.ndarray] = [
            load_image_bgr(
                v,
                disable_preproc_auto_orient=kwargs.get(
                    "disable_preproc_auto_orient", False
                ),
            )
            for v in images
        ]
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        if self._preprocess_supports_independent_stage_execution:
            mapped_kwargs["independent_stage_execution"] = False
        return self._model.pre_process(np_images, **mapped_kwargs)

    def predict(self, img_in, **kwargs):
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.forward(img_in, **mapped_kwargs)

    def postprocess(
        self,
        predictions: List[Detections],
        preprocess_return_metadata: PreprocessingMetadata,
        **kwargs,
    ) -> List[ObjectDetectionInferenceResponse]:
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        detections_list = self._model.post_process(
            predictions, preprocess_return_metadata, **mapped_kwargs
        )

        responses: List[ObjectDetectionInferenceResponse] = []
        for preproc_metadata, det in zip(preprocess_return_metadata, detections_list):
            H = preproc_metadata.original_size.height
            W = preproc_metadata.original_size.width

            xyxy = det.xyxy.detach().cpu().numpy()
            confs = det.confidence.detach().cpu().numpy()
            class_ids = det.class_id.detach().cpu().numpy()

            predictions: List[ObjectDetectionPrediction] = []

            for (x1, y1, x2, y2), conf, class_id in zip(xyxy, confs, class_ids):
                cx = (float(x1) + float(x2)) / 2.0
                cy = (float(y1) + float(y2)) / 2.0
                w = float(x2) - float(x1)
                h = float(y2) - float(y1)
                class_id_int = int(class_id)
                class_name = (
                    self.class_names[class_id_int]
                    if 0 <= class_id_int < len(self.class_names)
                    else str(class_id_int)
                )
                if (
                    kwargs.get("class_filter")
                    and class_name not in kwargs["class_filter"]
                ):
                    continue
                predictions.append(
                    ObjectDetectionPrediction(
                        x=cx,
                        y=cy,
                        width=w,
                        height=h,
                        confidence=float(conf),
                        **{"class": class_name},
                        class_id=class_id_int,
                    )
                )

            responses.append(
                ObjectDetectionInferenceResponse(
                    predictions=predictions,
                    image=InferenceResponseImage(width=W, height=H),
                )
            )
        return responses

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        """Clears any cache if necessary. TODO: Implement this to delete the cache from the experimental model.

        Args:
            delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to True.
        """
        pass

    def draw_predictions(
        self,
        inference_request: InferenceRequest,
        inference_response: InferenceResponse,
    ) -> bytes:
        """Draw predictions from an inference response onto the original image provided by an inference request

        Args:
            inference_request (ObjectDetectionInferenceRequest): The inference request containing the image on which to draw predictions
            inference_response (ObjectDetectionInferenceResponse): The inference response containing predictions to be drawn

        Returns:
            str: A base64 encoded image string
        """
        class_id_2_color = {
            i: DEFAULT_COLOR_PALETTE[i % len(DEFAULT_COLOR_PALETTE)]
            for i, class_name in enumerate(self._model.class_names)
        }
        return draw_detection_predictions(
            inference_request=inference_request,
            inference_response=inference_response,
            colors=class_id_2_color,
        )


class InferenceModelsInstanceSegmentationAdapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY
        model_id = resolve_roboflow_model_alias(model_id=model_id)

        self.task_type = "instance-segmentation"

        extra_weights_provider_headers = get_extra_weights_provider_headers(
            countinference=kwargs.get("countinference"),
            service_secret=kwargs.get("service_secret"),
        )
        backend = list(
            VALID_INFERENCE_MODELS_BACKENDS.difference(
                DISABLED_INFERENCE_MODELS_BACKENDS
            )
        )
        self._model: InstanceSegmentationModel = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            weights_provider_extra_headers=extra_weights_provider_headers,
            backend=backend,
            rf_detr_max_input_resolution=RFDETR_ONNX_MAX_RESOLUTION,
            **kwargs,
        )
        self.class_names = list(self._model.class_names)
        # Stream pipelining: depth=1 means original synchronous behavior
        # (preprocess→forward→postprocess on each frame, in order). depth=2
        # means two stages in parallel: while the GPU works on the current
        # frame, the CPU prepares/submits the next frame, then harvests the
        # previous response. Only models that explicitly support the deferred
        # GPU handoff contract can use this; other instance-segmentation
        # backends keep depth=1 even if RFDETR_PIPELINE_DEPTH is set.
        self._pipeline_depth = self._resolve_pipeline_depth()
        self._response_delay = max(1, self._pipeline_depth - 1)
        # Per-adapter in-flight futures + metadata. Not thread-safe; the
        # InferencePipeline is single-producer and the adapter is owned by a
        # single worker.
        self._pending_gpu_submissions: Deque[
            Tuple[InferenceFuture, PreprocessingMetadata, dict]
        ] = deque()
        self._pending_futures: Deque[
            Tuple[InferenceFuture, PreprocessingMetadata, dict]
        ] = deque()
        self._gpu_submit_generation = 0
        self._response_executor: Optional[ThreadPoolExecutor] = None
        self._response_executor_finalizer = None
        self._response_futures: Deque[
            Tuple[
                Future[List[InstanceSegmentationInferenceResponse]],
                Optional[str],
            ]
        ] = deque()

    def _resolve_pipeline_depth(self) -> int:
        requested_depth = min(get_rfdetr_pipeline_depth(), MAX_RFDETR_PIPELINE_DEPTH)
        if requested_depth <= 1 or self._model_supports_stream_pipeline():
            return requested_depth
        return 1

    def _model_supports_stream_pipeline(self) -> bool:
        supports_stream_pipeline = getattr(
            self._model, "supports_stream_pipeline", False
        )
        if callable(supports_stream_pipeline):
            return bool(supports_stream_pipeline())
        return bool(supports_stream_pipeline)

    def map_inference_kwargs(self, kwargs: dict) -> dict:
        kwargs["input_color_format"] = "bgr"
        pre_processing_overrides = PreProcessingOverrides(
            disable_contrast_enhancement=kwargs.get("disable_preproc_contrast", False),
            disable_grayscale=kwargs.get("disable_preproc_grayscale", False),
            disable_static_crop=kwargs.get("disable_preproc_static_crop", False),
        )
        if GCP_SERVERLESS:
            enforce_dense_masks_in_inference_models = False
        else:
            enforce_dense_masks_in_inference_models = kwargs.get(
                "enforce_dense_masks_in_inference_models",
                False,
            )
        kwargs["pre_processing_overrides"] = pre_processing_overrides
        if (
            "rle" in self._model.supported_mask_formats
            and not enforce_dense_masks_in_inference_models
        ):
            kwargs["mask_format"] = "rle"
        kwargs.pop(STREAM_PIPELINE_CONTEXT_ID_KWARG, None)
        return kwargs

    def preprocess(self, image: Any, **kwargs):
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        np_images: List[np.ndarray] = [
            load_image_bgr(
                v,
                disable_preproc_auto_orient=kwargs.get(
                    "disable_preproc_auto_orient", False
                ),
            )
            for v in images
        ]
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.pre_process(np_images, **mapped_kwargs)

    def _request_batch_size(self, img_in: Any) -> int:
        pre_processing_meta = getattr(img_in, "_pre_processing_meta", None)
        if isinstance(pre_processing_meta, (list, tuple)):
            return len(pre_processing_meta)
        shape = getattr(img_in, "shape", None)
        if shape is not None and len(shape) > 0:
            return int(shape[0])
        if isinstance(img_in, (list, tuple)):
            return len(img_in)
        return 1

    def predict(self, img_in, **kwargs):
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        if self._pipeline_depth <= 1:
            # Original path: forward on current frame, postprocess on
            # current frame, all synchronous.
            return self._model.forward(img_in, **mapped_kwargs)
        if self._request_batch_size(img_in) > 1:
            return self._model.forward(img_in, **mapped_kwargs)

        mapped_kwargs["defer_count_to_adapter"] = (
            kwargs.get("response_mask_format") != "rle"
        )
        mapped_kwargs["defer_postprocess_sync"] = True
        mapped_kwargs["reuse_trt_graph_outputs"] = True
        # Pipelined path: before launching frame N's forward, enqueue the
        # oldest frame whose postprocess metadata is already known. That keeps
        # postprocess off the current frame's postprocess() host path while
        # still preserving the correctness dependency for reused TRT outputs.
        self._submit_next_pending_gpu_work()
        pre_processing_meta = getattr(img_in, "_pre_processing_meta", None)
        fut = self._model.forward_async(img_in, pre_processing_meta, **mapped_kwargs)
        stream_pipeline_context_id = kwargs.get(STREAM_PIPELINE_CONTEXT_ID_KWARG)
        if not isinstance(stream_pipeline_context_id, str):
            stream_pipeline_context_id = None
        attach_adapter_mapped_kwargs(
            fut,
            mapped_kwargs,
            stream_pipeline_context_id=stream_pipeline_context_id,
        )
        if pre_processing_meta is not None:
            self._submit_future_gpu_work(fut, pre_processing_meta, mapped_kwargs)
        self._submit_ready_responses()
        return fut

    def flush(self) -> List[InstanceSegmentationInferenceResponse]:
        """Drain the tail of the pipelined queue.

        Returns responses for any in-flight frames whose forward/postprocess
        GPU work was submitted but whose CPU-visible response has not yet been
        materialized. Callers that use `RFDETR_PIPELINE_DEPTH>=2` MUST invoke
        this at stream end or the final frames will be dropped.
        """
        if self._pipeline_depth <= 1:
            return []
        self._submit_all_pending_gpu_work()
        self._submit_all_pending_responses()
        responses: List[InstanceSegmentationInferenceResponse] = []
        while self._response_futures:
            response_future, _ = self._response_futures.popleft()
            responses.extend(
                _resolve_response_future(
                    future=response_future,
                    context="RF-DETR stream pipeline flush",
                )
            )
        return responses

    def shutdown_pipeline(self) -> None:
        if self._response_executor is None:
            return None
        finalizer = self._response_executor_finalizer
        if finalizer is not None and finalizer.alive:
            finalizer.detach()
        self._response_executor.shutdown(wait=False)
        self._response_executor = None
        self._response_executor_finalizer = None

    def _get_response_executor(self) -> ThreadPoolExecutor:
        if self._response_executor is None:
            self._response_executor = ThreadPoolExecutor(max_workers=1)
            self._response_executor_finalizer = finalize(
                self,
                self._response_executor.shutdown,
                wait=False,
            )
        return self._response_executor

    def _submit_future_gpu_work(
        self,
        fut: InferenceFuture,
        meta: PreprocessingMetadata,
        mapped_kwargs: dict,
    ) -> None:
        if adapter_gpu_work_submitted(fut):
            return None
        fut._meta = meta  # type: ignore[attr-defined]
        fut._kwargs = mapped_kwargs  # type: ignore[attr-defined]
        submit_gpu_work = getattr(fut, "submit_gpu_work", None)
        if callable(submit_gpu_work):
            submit_gpu_work(meta)
            self._gpu_submit_generation = getattr(self, "_gpu_submit_generation", 0) + 1
            mark_adapter_gpu_work_submitted(fut, self._gpu_submit_generation)

    def _submit_next_pending_gpu_work(self) -> None:
        if not self._pending_gpu_submissions:
            return None
        self._submit_future_gpu_work(*self._pending_gpu_submissions.popleft())

    def _submit_all_pending_gpu_work(self) -> None:
        while self._pending_gpu_submissions:
            self._submit_future_gpu_work(*self._pending_gpu_submissions.popleft())

    def _submit_response_build(
        self,
        fut: InferenceFuture,
        meta: PreprocessingMetadata,
        mapped_kwargs: dict,
    ) -> None:
        fut._meta = meta  # type: ignore[attr-defined]
        fut._kwargs = mapped_kwargs  # type: ignore[attr-defined]
        response_future = self._get_response_executor().submit(
            self._finalize_future,
            fut,
            meta,
            mapped_kwargs,
        )
        context_id = get_adapter_stream_pipeline_context_id(fut)
        self._response_futures.append(
            (
                response_future,
                context_id,
            )
        )

    def _submit_ready_responses(self) -> None:
        while self._pending_futures:
            fut, meta, mapped_kwargs = self._pending_futures[0]
            submit_generation = get_adapter_gpu_submit_generation(fut)
            if submit_generation is None:
                self._submit_future_gpu_work(fut, meta, mapped_kwargs)
                submit_generation = get_adapter_gpu_submit_generation(fut)
            if submit_generation is None:
                break
            gpu_submit_generation = getattr(self, "_gpu_submit_generation", 0)
            if gpu_submit_generation < submit_generation + self._response_delay:
                break
            self._submit_response_build(*self._pending_futures.popleft())

    def _submit_all_pending_responses(self) -> None:
        while self._pending_futures:
            self._submit_response_build(*self._pending_futures.popleft())

    def postprocess(
        self,
        predictions,
        preprocess_return_metadata: PreprocessingMetadata,
        **kwargs,
    ) -> List[InstanceSegmentationInferenceResponse]:
        if self._pipeline_depth <= 1 or not isinstance(predictions, InferenceFuture):
            return self._postprocess_sync(
                predictions, preprocess_return_metadata, **kwargs
            )
        fut: InferenceFuture = predictions
        mapped_kwargs = get_adapter_mapped_kwargs(fut)
        self._pending_gpu_submissions.append(
            (
                fut,
                preprocess_return_metadata,
                mapped_kwargs,
            )
        )
        self._pending_futures.append((fut, preprocess_return_metadata, mapped_kwargs))
        if len(self._pending_futures) > self._response_delay:
            self._submit_next_pending_gpu_work()
            self._submit_ready_responses()

        if not self._response_futures:
            return self._empty_responses_for_metadata(
                preprocess_return_metadata=preprocess_return_metadata,
                workflow_execution=kwargs.get("source") == "workflow-execution",
            )

        response_future, context_id = self._response_futures.popleft()
        if kwargs.get("source") == "workflow-execution":
            responses = self._empty_responses_for_metadata(
                preprocess_return_metadata=preprocess_return_metadata,
                workflow_execution=True,
            )
            if responses:
                attach_async_response_future(
                    response=responses[0],
                    response_future=response_future,
                    context_id=context_id,
                )
            return responses
        return _resolve_response_future(
            future=response_future,
            context="RF-DETR stream pipeline response finalization",
        )

    def _empty_responses_for_metadata(
        self,
        preprocess_return_metadata: PreprocessingMetadata,
        workflow_execution: bool,
    ) -> List[InstanceSegmentationInferenceResponse]:
        if workflow_execution:
            return [
                InstanceSegmentationInferenceResponseDC(
                    predictions=[],
                    image=InferenceResponseImageDC(
                        width=m.original_size.width,
                        height=m.original_size.height,
                    ),
                )
                for m in preprocess_return_metadata
            ]
        return [
            InstanceSegmentationInferenceResponse(
                predictions=[],
                image=InferenceResponseImage(
                    width=m.original_size.width,
                    height=m.original_size.height,
                ),
            )
            for m in preprocess_return_metadata
        ]

    def _finalize_future(
        self,
        fut: InferenceFuture,
        preprocess_return_metadata: PreprocessingMetadata,
        mapped_kwargs: dict,
    ) -> List[InstanceSegmentationInferenceResponse]:
        # Override the future's stashed meta (which was `None` at submit
        # time) with the correct metadata for the frame whose forward pass
        # the future represents. This is an allowed private-surface tweak
        # because _DirectInferenceFuture's post_process is memoised.
        fut._meta = preprocess_return_metadata  # type: ignore[attr-defined]
        fut._kwargs = mapped_kwargs  # type: ignore[attr-defined]
        detections_list = fut.result()
        return self._build_responses_from_detections(
            detections_list, preprocess_return_metadata, **mapped_kwargs
        )

    def _postprocess_sync(
        self,
        predictions: List[InstanceDetections],
        preprocess_return_metadata: PreprocessingMetadata,
        **kwargs,
    ) -> List[InstanceSegmentationInferenceResponse]:
        return_in_rle = kwargs.get("response_mask_format") == "rle"
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        mapped_kwargs["defer_count_to_adapter"] = not return_in_rle
        detections_list = self._model.post_process(
            predictions, preprocess_return_metadata, **mapped_kwargs
        )
        return self._build_responses_from_detections(
            detections_list, preprocess_return_metadata, **kwargs
        )

    def _build_responses_from_detections(
        self,
        detections_list: List[InstanceDetections],
        preprocess_return_metadata: PreprocessingMetadata,
        **kwargs,
    ) -> List[InstanceSegmentationInferenceResponse]:
        return_in_rle = kwargs.get("response_mask_format") == "rle"
        # Workflow callers consume a plain dict via `_is_response_dc_to_dict`;
        # dataclasses avoid pydantic validation + `model_dump` overhead per
        # frame. Keep the pydantic path for RLE responses and for non-workflow
        # callers that rely on the response model type.
        use_dc = (
            kwargs.get("source") == "workflow-execution"
            and not return_in_rle
            and getattr(self, "_pipeline_depth", 1) > 1
        )

        responses: List[InstanceSegmentationInferenceResponse] = []
        for preproc_metadata, det in zip(preprocess_return_metadata, detections_list):
            finalize_pending = get_deferred_postprocess_finalizer(det)
            if callable(finalize_pending):
                det = finalize_pending()
            H = preproc_metadata.original_size.height
            W = preproc_metadata.original_size.width

            combined_gpu = getattr(det, "_combined_gpu", None)
            mask_gpu = getattr(det, "_mask_gpu", None)
            mask_packed_gpu = getattr(det, "_mask_packed_gpu", None)
            mask_cpu = getattr(det, "_mask_cpu", None)
            defer_count_to_adapter = getattr(det, "_defer_count_to_adapter", False)
            done_event = get_deferred_postprocess_done_event(det)
            dense_mask_cuda = isinstance(mask_gpu, torch.Tensor) and mask_gpu.is_cuda
            packed_mask_cuda = (
                isinstance(mask_packed_gpu, torch.Tensor) and mask_packed_gpu.is_cuda
            )
            if (
                not return_in_rle
                and done_event is not None
                and (dense_mask_cuda or packed_mask_cuda)
            ):
                device = mask_gpu.device if dense_mask_cuda else mask_packed_gpu.device
                stream = torch.cuda.current_stream(device)
                done_event.wait(stream)

                if (
                    defer_count_to_adapter
                    and isinstance(combined_gpu, torch.Tensor)
                    and combined_gpu.is_cuda
                ):
                    # combined_np / class_column / combined_slice are scratch views;
                    # use only for survivor counting and in-loop scalar extraction.
                    combined_host = get_pinned_buffer(
                        "combined_full",
                        tuple(combined_gpu.shape),
                        combined_gpu.dtype,
                    )
                    combined_host.copy_(combined_gpu, non_blocking=True)
                    stream.synchronize()
                    combined_np = combined_host.numpy()
                    class_column = combined_np[:, 5]
                    inactive_indices = np.flatnonzero(class_column < 0)
                    n_survivors = (
                        int(inactive_indices[0])
                        if inactive_indices.size > 0
                        else int(class_column.shape[0])
                    )
                    if n_survivors == 0:
                        xyxy = np.empty((0, 4), dtype=np.int32)
                        confs = np.empty((0,), dtype=np.float32)
                        class_ids = np.empty((0,), dtype=np.int32)
                        polys_or_rles = []
                    else:
                        combined_slice = combined_np[:n_survivors]
                        xyxy = combined_slice[:, :4]
                        confs = combined_slice[:, 4].view(np.float32)
                        class_ids = combined_slice[:, 5]
                        if packed_mask_cuda:
                            packed_slice = mask_packed_gpu[:n_survivors]
                            packed_host = get_pinned_buffer(
                                "mask_packed",
                                tuple(packed_slice.shape),
                                packed_slice.dtype,
                            )
                            packed_host.copy_(packed_slice, non_blocking=True)
                            stream.synchronize()
                            polys_or_rles = bitpacked_masks2poly(
                                packed_host.numpy(), width=W
                            )
                        else:
                            mask_slice = mask_gpu[:n_survivors]
                            mask_host = get_pinned_buffer(
                                "mask", tuple(mask_slice.shape), mask_slice.dtype
                            )
                            mask_host.copy_(mask_slice, non_blocking=True)
                            stream.synchronize()
                            polys_or_rles = masks2poly(mask_host.numpy())
                else:
                    n_survivors = int(det.xyxy.shape[0])
                    if n_survivors == 0:
                        xyxy = np.empty((0, 4), dtype=np.int32)
                        confs = np.empty((0,), dtype=np.float32)
                        class_ids = np.empty((0,), dtype=np.int32)
                        polys_or_rles = []
                    else:
                        mask_slice = mask_gpu[:n_survivors]
                        mask_host = get_pinned_buffer(
                            "mask", tuple(mask_slice.shape), mask_slice.dtype
                        )
                        if (
                            isinstance(combined_gpu, torch.Tensor)
                            and combined_gpu.is_cuda
                            and tuple(combined_gpu.shape)
                            == (n_survivors, det.xyxy.shape[1] + 2)
                        ):
                            combined_slice = combined_gpu[:n_survivors]
                            combined_host = get_pinned_buffer(
                                "combined",
                                tuple(combined_slice.shape),
                                combined_slice.dtype,
                            )
                            combined_host.copy_(combined_slice, non_blocking=True)
                            mask_host.copy_(mask_slice, non_blocking=True)
                            stream.synchronize()
                            combined_np = combined_host.numpy()
                            xyxy = combined_np[:, :4]
                            confs = combined_np[:, 4].view(np.float32)
                            class_ids = combined_np[:, 5]
                            polys_or_rles = masks2poly(mask_host.numpy())
                        else:
                            xyxy_host = get_pinned_buffer(
                                "xyxy", tuple(det.xyxy.shape), det.xyxy.dtype
                            )
                            conf_host = get_pinned_buffer(
                                "conf",
                                tuple(det.confidence.shape),
                                det.confidence.dtype,
                            )
                            class_host = get_pinned_buffer(
                                "class_id",
                                tuple(det.class_id.shape),
                                det.class_id.dtype,
                            )
                            xyxy_host.copy_(det.xyxy, non_blocking=True)
                            conf_host.copy_(det.confidence, non_blocking=True)
                            class_host.copy_(det.class_id, non_blocking=True)
                            mask_host.copy_(mask_slice, non_blocking=True)
                            stream.synchronize()
                            xyxy = xyxy_host.numpy()
                            confs = conf_host.numpy()
                            class_ids = class_host.numpy()
                            polys_or_rles = masks2poly(mask_host.numpy())
            elif not return_in_rle and isinstance(mask_cpu, np.ndarray):
                xyxy = det.xyxy.detach().cpu().numpy()
                confs = det.confidence.detach().cpu().numpy()
                class_ids = det.class_id.detach().cpu().numpy()
                polys_or_rles = masks2poly(mask_cpu)
            else:
                xyxy = det.xyxy.detach().cpu().numpy()
                confs = det.confidence.detach().cpu().numpy()
                if isinstance(det.mask, torch.Tensor):
                    masks = det.mask.detach().cpu().numpy()
                    if return_in_rle:
                        polys_or_rles = [
                            torch_mask_to_coco_rle(mask=mask) for mask in masks
                        ]
                    else:
                        polys_or_rles = masks2poly(masks)
                else:
                    if return_in_rle:
                        polys_or_rles = det.mask.to_coco_rle_masks()
                    else:
                        polys_or_rles = rle_masks2poly(det.mask)
                class_ids = det.class_id.detach().cpu().numpy()

            # Some branches above intentionally keep numpy views into
            # thread-local pinned scratch buffers. Only scalar values and
            # polygon/RLE lists may be stored on responses below; do not return
            # those arrays or any view derived from them.
            predictions: List[
                Union[InstanceSegmentationPrediction, InstanceSegmentationRLEPrediction]
            ] = []

            for (x1, y1, x2, y2), mask_as_poly_or_rle, conf, class_id in zip(
                xyxy, polys_or_rles, confs, class_ids
            ):
                cx = (float(x1) + float(x2)) / 2.0
                cy = (float(y1) + float(y2)) / 2.0
                w = float(x2) - float(x1)
                h = float(y2) - float(y1)
                class_id_int = int(class_id)
                class_name = (
                    self.class_names[class_id_int]
                    if 0 <= class_id_int < len(self.class_names)
                    else str(class_id_int)
                )
                if (
                    kwargs.get("class_filter")
                    and class_name not in kwargs["class_filter"]
                ):
                    continue
                if use_dc:
                    predictions.append(
                        InstanceSegmentationPredictionDC(
                            x=cx,
                            y=cy,
                            width=w,
                            height=h,
                            confidence=float(conf),
                            class_name=class_name,
                            class_id=class_id_int,
                            points=[
                                PointDC(x=float(point[0]), y=float(point[1]))
                                for point in mask_as_poly_or_rle
                            ],
                        )
                    )
                else:
                    if not return_in_rle:
                        predictions.append(
                            InstanceSegmentationPrediction(
                                x=cx,
                                y=cy,
                                width=w,
                                height=h,
                                confidence=float(conf),
                                points=[
                                    Point(x=point[0], y=point[1])
                                    for point in mask_as_poly_or_rle
                                ],
                                **{"class": class_name},
                                class_id=class_id_int,
                            )
                        )
                    else:
                        if isinstance(mask_as_poly_or_rle["counts"], bytes):
                            mask_as_poly_or_rle["counts"] = mask_as_poly_or_rle[
                                "counts"
                            ].decode("ascii")
                        predictions.append(
                            InstanceSegmentationRLEPrediction(
                                x=cx,
                                y=cy,
                                width=w,
                                height=h,
                                confidence=float(conf),
                                rle=mask_as_poly_or_rle,
                                **{"class": class_name},
                                class_id=class_id_int,
                            )
                        )

            if use_dc:
                responses.append(
                    InstanceSegmentationInferenceResponseDC(
                        predictions=predictions,
                        image=InferenceResponseImageDC(width=W, height=H),
                    )
                )
            else:
                responses.append(
                    InstanceSegmentationInferenceResponse(
                        predictions=predictions,
                        image=InferenceResponseImage(width=W, height=H),
                    )
                )
        return responses

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        """Clears any cache if necessary. TODO: Implement this to delete the cache from the experimental model.

        Args:
            delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to True.
        """
        pass

    def draw_predictions(
        self,
        inference_request: InferenceRequest,
        inference_response: InferenceResponse,
    ) -> bytes:
        """Draw predictions from an inference response onto the original image provided by an inference request

        Args:
            inference_request (ObjectDetectionInferenceRequest): The inference request containing the image on which to draw predictions
            inference_response (ObjectDetectionInferenceResponse): The inference response containing predictions to be drawn

        Returns:
            str: A base64 encoded image string
        """
        class_id_2_color = {
            i: DEFAULT_COLOR_PALETTE[i % len(DEFAULT_COLOR_PALETTE)]
            for i, class_name in enumerate(self._model.class_names)
        }
        return draw_detection_predictions(
            inference_request=inference_request,
            inference_response=inference_response,
            colors=class_id_2_color,
        )


def rle_masks2poly(masks: InstancesRLEMasks) -> List[np.ndarray]:
    if INFERENCE_MODELS_RFDETR_TRITON_POSTPROC_ENABLED:
        return rle_masks_to_polygons(masks=masks)

    segments = []
    h, w = masks.image_size
    for counts in masks.masks:
        rle_dict = {"size": [h, w], "counts": counts}
        decoded_rle = np.ascontiguousarray(mask_utils.decode(rle_dict))
        if not np.any(decoded_rle):
            segments.append(np.zeros((0, 2), dtype=np.float32))
            continue
        segments.append(mask2poly(decoded_rle))
    return segments


class InferenceModelsKeyPointsDetectionAdapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY
        model_id = resolve_roboflow_model_alias(model_id=model_id)

        self.task_type = "keypoint-detection"

        extra_weights_provider_headers = get_extra_weights_provider_headers(
            countinference=kwargs.get("countinference"),
            service_secret=kwargs.get("service_secret"),
        )
        backend = list(
            VALID_INFERENCE_MODELS_BACKENDS.difference(
                DISABLED_INFERENCE_MODELS_BACKENDS
            )
        )
        self._model: KeyPointsDetectionModel = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            weights_provider_extra_headers=extra_weights_provider_headers,
            backend=backend,
            **kwargs,
        )
        self.class_names = list(self._model.class_names)

    def map_inference_kwargs(self, kwargs: dict) -> dict:
        kwargs["input_color_format"] = "bgr"
        if "request" in kwargs:
            keypoint_confidence_threshold = kwargs["request"].keypoint_confidence
            kwargs["key_points_threshold"] = keypoint_confidence_threshold
        pre_processing_overrides = PreProcessingOverrides(
            disable_contrast_enhancement=kwargs.get("disable_preproc_contrast", False),
            disable_grayscale=kwargs.get("disable_preproc_grayscale", False),
            disable_static_crop=kwargs.get("disable_preproc_static_crop", False),
        )
        kwargs["pre_processing_overrides"] = pre_processing_overrides
        return kwargs

    def preprocess(self, image: Any, **kwargs):
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        np_images: List[np.ndarray] = [
            load_image_bgr(
                v,
                disable_preproc_auto_orient=kwargs.get(
                    "disable_preproc_auto_orient", False
                ),
            )
            for v in images
        ]
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.pre_process(np_images, **mapped_kwargs)

    def predict(self, img_in, **kwargs):
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.forward(img_in, **mapped_kwargs)

    def postprocess(
        self,
        predictions: Tuple[List[KeyPoints], Optional[List[Detections]]],
        preprocess_return_metadata: PreprocessingMetadata,
        **kwargs,
    ) -> List[KeypointsDetectionInferenceResponse]:
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        keypoints_list, detections_list = self._model.post_process(
            predictions, preprocess_return_metadata, **mapped_kwargs
        )
        if detections_list is None:
            raise RuntimeError(
                "Keypoints detection model does not provide instances detection - this is not supported for "
                "models from `inference-models` package which are adapted to work with `inference`."
            )
        key_points_classes = self._model.key_points_classes
        responses: List[KeypointsDetectionInferenceResponse] = []
        for preproc_metadata, keypoints, det in zip(
            preprocess_return_metadata, keypoints_list, detections_list
        ):

            H = preproc_metadata.original_size.height
            W = preproc_metadata.original_size.width

            xyxy = det.xyxy.detach().cpu().numpy()
            confs = det.confidence.detach().cpu().numpy()
            class_ids = det.class_id.detach().cpu().numpy()
            keypoints_xy = keypoints.xy.detach().cpu().tolist()
            keypoints_class_id = keypoints.class_id.detach().cpu().tolist()
            keypoints_confidence = keypoints.confidence.detach().cpu().tolist()
            predictions: List[KeypointsPrediction] = []

            for (
                (x1, y1, x2, y2),
                conf,
                class_id,
                instance_keypoints_xy,
                instance_keypoints_class_id,
                instance_keypoints_confidence,
            ) in zip(
                xyxy,
                confs,
                class_ids,
                keypoints_xy,
                keypoints_class_id,
                keypoints_confidence,
            ):
                cx = (float(x1) + float(x2)) / 2.0
                cy = (float(y1) + float(y2)) / 2.0
                w = float(x2) - float(x1)
                h = float(y2) - float(y1)
                class_id_int = int(class_id)
                class_name = (
                    self.class_names[class_id_int]
                    if 0 <= class_id_int < len(self.class_names)
                    else str(class_id_int)
                )
                if (
                    kwargs.get("class_filter")
                    and class_name not in kwargs["class_filter"]
                ):
                    continue
                predictions.append(
                    KeypointsPrediction(
                        x=cx,
                        y=cy,
                        width=w,
                        height=h,
                        confidence=float(conf),
                        **{"class": class_name},
                        class_id=class_id_int,
                        keypoints=model_keypoints_to_response(
                            instance_keypoints_xy=instance_keypoints_xy,
                            instance_keypoints_confidence=instance_keypoints_confidence,
                            instance_keypoints_class_id=instance_keypoints_class_id,
                            key_points_classes=key_points_classes,
                        ),
                    )
                )

            responses.append(
                KeypointsDetectionInferenceResponse(
                    predictions=predictions,
                    image=InferenceResponseImage(width=W, height=H),
                )
            )

        return responses

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        """Clears any cache if necessary. TODO: Implement this to delete the cache from the experimental model.

        Args:
            delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to True.
        """
        pass

    def draw_predictions(
        self,
        inference_request: InferenceRequest,
        inference_response: InferenceResponse,
    ) -> bytes:
        """Draw predictions from an inference response onto the original image provided by an inference request

        Args:
            inference_request (ObjectDetectionInferenceRequest): The inference request containing the image on which to draw predictions
            inference_response (ObjectDetectionInferenceResponse): The inference response containing predictions to be drawn

        Returns:
            str: A base64 encoded image string
        """
        class_id_2_color = {
            i: DEFAULT_COLOR_PALETTE[i % len(DEFAULT_COLOR_PALETTE)]
            for i, class_name in enumerate(self._model.class_names)
        }
        return draw_detection_predictions(
            inference_request=inference_request,
            inference_response=inference_response,
            colors=class_id_2_color,
        )


def model_keypoints_to_response(
    instance_keypoints_xy: List[
        List[Union[float, int]]
    ],  # (num_key_points_foc_class_of_object, 2)
    instance_keypoints_confidence: List[float],  # (instance_key_points, )
    instance_keypoints_class_id: int,
    key_points_classes: List[List[str]],
) -> List[Keypoint]:
    keypoint_classes = key_points_classes[instance_keypoints_class_id]
    results = []
    for keypoint_class_id, ((x, y), confidence, keypoint_class_name) in enumerate(
        zip(instance_keypoints_xy, instance_keypoints_confidence, keypoint_classes)
    ):
        if confidence <= 0.0:
            continue
        keypoint = Keypoint(
            x=x,
            y=y,
            confidence=confidence,
            class_id=keypoint_class_id,
            **{"class": keypoint_class_name},
        )
        results.append(keypoint)
    return results


class InferenceModelsClassificationAdapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY
        model_id = resolve_roboflow_model_alias(model_id=model_id)

        self.task_type = "classification"
        extra_weights_provider_headers = get_extra_weights_provider_headers(
            countinference=kwargs.get("countinference"),
            service_secret=kwargs.get("service_secret"),
        )
        backend = list(
            VALID_INFERENCE_MODELS_BACKENDS.difference(
                DISABLED_INFERENCE_MODELS_BACKENDS
            )
        )
        self._model: Union[ClassificationModel, MultiLabelClassificationModel] = (
            AutoModel.from_pretrained(
                model_id_or_path=model_id,
                api_key=self.api_key,
                allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
                allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
                weights_provider_extra_headers=extra_weights_provider_headers,
                backend=backend,
                **kwargs,
            )
        )
        self.class_names = list(self._model.class_names)

    def map_inference_kwargs(self, kwargs: dict) -> dict:
        kwargs["input_color_format"] = "bgr"
        pre_processing_overrides = PreProcessingOverrides(
            disable_contrast_enhancement=kwargs.get("disable_preproc_contrast", False),
            disable_grayscale=kwargs.get("disable_preproc_grayscale", False),
            disable_static_crop=kwargs.get("disable_preproc_static_crop", False),
        )
        kwargs["pre_processing_overrides"] = pre_processing_overrides
        return kwargs

    def preprocess(self, image: Any, **kwargs):
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        np_images: List[np.ndarray] = [
            load_image_bgr(
                v,
                disable_preproc_auto_orient=kwargs.get(
                    "disable_preproc_auto_orient", False
                ),
            )
            for v in images
        ]
        images_shapes = [i.shape[:2] for i in np_images]
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.pre_process(np_images, **mapped_kwargs), images_shapes

    def predict(self, img_in, **kwargs):
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.forward(img_in, **mapped_kwargs)

    def postprocess(
        self,
        predictions: Tuple[List[KeyPoints], Optional[List[Detections]]],
        returned_metadata: List[Tuple[int, int]],
        **kwargs,
    ) -> Union[
        List[MultiLabelClassificationInferenceResponse],
        List[ClassificationInferenceResponse],
    ]:
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        if isinstance(self._model, MultiLabelClassificationModel):
            post_processed_predictions = self._model.post_process(
                predictions, **mapped_kwargs
            )
            return prepare_multi_label_classification_response(
                post_processed_predictions,
                image_sizes=returned_metadata,
                class_names=self.class_names,
            )
        # Single-label classification: top-1 always wins regardless of
        # confidence, so per-class refinement isn't meaningful here. The base
        # class deliberately opts out of recommendedParameters entirely. The
        # response builder still uses the confidence as a cutoff that decides
        # which alternative classes show up — string-valued "best"/"default"
        # have no meaningful mapping here, so fall back to 0.5.
        post_processed_predictions = self._model.post_process(
            predictions, **mapped_kwargs
        )
        raw_confidence = kwargs.get("confidence")
        confidence_threshold = (
            raw_confidence if isinstance(raw_confidence, (int, float)) else 0.5
        )
        return prepare_classification_response(
            post_processed_predictions,
            image_sizes=returned_metadata,
            class_names=self.class_names,
            confidence_threshold=confidence_threshold,
        )

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        """Clears any cache if necessary. TODO: Implement this to delete the cache from the experimental model.

        Args:
            delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to True.
        """
        pass

    def infer_from_request(
        self,
        request: ClassificationInferenceRequest,
    ) -> Union[List[InferenceResponse], InferenceResponse]:
        """
        Handle an inference request to produce an appropriate response.

        Args:
            request (ClassificationInferenceRequest): The request object encapsulating the image(s) and relevant parameters.

        Returns:
            Union[List[InferenceResponse], InferenceResponse]: The response object(s) containing the predictions, visualization, and other pertinent details. If a list of images was provided, a list of responses is returned. Otherwise, a single response is returned.

        Notes:
            - Starts a timer at the beginning to calculate inference time.
            - Processes the image(s) through the `infer` method.
            - Generates the appropriate response object(s) using `make_response`.
            - Calculates and sets the time taken for inference.
            - If visualization is requested, the predictions are drawn on the image.
        """
        t1 = perf_counter()
        responses = self.infer(**request.dict(), return_image_dims=True)
        for response in responses:
            response.time = perf_counter() - t1
            response.inference_id = getattr(request, "id", None)

        if request.visualize_predictions:
            for response in responses:
                response.visualization = draw_predictions(
                    request, response, self.class_names
                )

        if not isinstance(request.image, list):
            responses = responses[0]

        return responses


def prepare_multi_label_classification_response(
    post_processed_predictions: List[MultiLabelClassificationPrediction],
    image_sizes: List[Tuple[int, int]],
    class_names: List[str],
) -> List[MultiLabelClassificationInferenceResponse]:
    """Build the API response from a model's post-processed predictions.

    `prediction.class_ids` is the authoritative list of "passed" classes —
    the model's `post_process` already applied the
    full priority chain (user → per-class → global → default), so the
    response builder doesn't re-threshold here. The full per-class score
    vector is still emitted in `image_predictions_dict` for UI display.
    """
    results = []
    for prediction, image_size in zip(post_processed_predictions, image_sizes):
        class_confidences = _reshape_classification_confidences(
            confidence=prediction.confidence.cpu(),
            expected_num_images=1,
            class_names=class_names,
        )[0].tolist()
        image_predictions_dict = {
            class_names[class_id]: {
                "confidence": confidence,
                "class_id": class_id,
            }
            for class_id, confidence in enumerate(class_confidences)
        }
        predicted_classes = [
            class_names[class_id] for class_id in prediction.class_ids.tolist()
        ]
        results.append(
            MultiLabelClassificationInferenceResponse(
                predictions=image_predictions_dict,
                predicted_classes=predicted_classes,
                image=InferenceResponseImage(width=image_size[1], height=image_size[0]),
                # essentially pushing a dummy values as I have no intention breaking the new API for the sake of delivering value that has no practical use
            )
        )
    return results


def prepare_classification_response(
    post_processed_predictions: ClassificationPrediction,
    image_sizes: List[Tuple[int, int]],
    class_names: List[str],
    confidence_threshold: float,
) -> List[ClassificationInferenceResponse]:
    responses = []
    batch_confidences = _reshape_classification_confidences(
        confidence=post_processed_predictions.confidence.cpu(),
        expected_num_images=len(image_sizes),
        class_names=class_names,
    )
    for classes_confidence, image_size in zip(batch_confidences.tolist(), image_sizes):
        individual_classes_predictions = []
        for i, cls_name in enumerate(class_names):
            class_score = float(classes_confidence[i])
            if class_score < confidence_threshold:
                continue
            class_prediction = {
                "class_id": i,
                "class": cls_name,
                "confidence": round(class_score, 4),
            }
            individual_classes_predictions.append(class_prediction)
        individual_classes_predictions = sorted(
            individual_classes_predictions, key=lambda x: x["confidence"], reverse=True
        )
        response = ClassificationInferenceResponse(
            image=InferenceResponseImage(width=image_size[1], height=image_size[0]),
            # essentially pushing a dummy values as I have no intention breaking the new API for the sake of delivering value that has no practical use
            predictions=individual_classes_predictions,
            top=(
                individual_classes_predictions[0]["class"]
                if individual_classes_predictions
                else ""
            ),
            confidence=(
                individual_classes_predictions[0]["confidence"]
                if individual_classes_predictions
                else 0.0
            ),
        )
        responses.append(response)
    return responses


def _reshape_classification_confidences(
    confidence: torch.Tensor,
    expected_num_images: int,
    class_names: List[str],
) -> torch.Tensor:
    expected_num_classes = len(class_names)
    expected_num_scores = expected_num_images * expected_num_classes
    actual_num_scores = confidence.numel()
    if actual_num_scores != expected_num_scores:
        raise PostProcessingError(
            "Classification model output has shape "
            f"{tuple(confidence.shape)} containing {actual_num_scores} confidence "
            f"score(s), but response metadata expects {expected_num_images} image(s) "
            f"x {expected_num_classes} class name(s) = {expected_num_scores} score(s). "
            "This usually means the model package class names metadata does not match "
            "the classifier head."
        )
    return confidence.reshape(expected_num_images, expected_num_classes)


def draw_predictions(inference_request, inference_response, class_names: List[str]):
    """Draw prediction visuals on an image.

    This method overlays the predictions on the input image, including drawing rectangles and text to visualize the predicted classes.

    Args:
        inference_request: The request object containing the image and parameters.
        inference_response: The response object containing the predictions and other details.
        class_names: List of class names corresponding to the model's classes.

    Returns:
        bytes: The bytes of the visualized image in JPEG format.
    """
    image = load_image_rgb(inference_request.image)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    class_id_2_color = {
        i: DEFAULT_COLOR_PALETTE[i % len(DEFAULT_COLOR_PALETTE)]
        for i, class_name in enumerate(class_names)
    }
    if isinstance(inference_response.predictions, list):
        prediction = inference_response.predictions[0]
        color = class_id_2_color.get(prediction.class_id, "#4892EA")
        draw.rectangle(
            [0, 0, image.size[1], image.size[0]],
            outline=color,
            width=inference_request.visualization_stroke_width,
        )
        text = f"{prediction.class_id} - {prediction.class_name} {prediction.confidence:.2f}"
        text_size = font.getbbox(text)

        # set button size + 10px margins
        button_size = (text_size[2] + 20, text_size[3] + 20)
        button_img = Image.new("RGBA", button_size, color)
        # put text on button with 10px margins
        button_draw = ImageDraw.Draw(button_img)
        button_draw.text((10, 10), text, font=font, fill=(255, 255, 255, 255))

        # put button on source image in position (0, 0)
        image.paste(button_img, (0, 0))
    else:
        if len(inference_response.predictions) > 0:
            box_color = "#4892EA"
            draw.rectangle(
                [0, 0, image.size[1], image.size[0]],
                outline=box_color,
                width=inference_request.visualization_stroke_width,
            )
        row = 0
        predictions = [
            (cls_name, pred)
            for cls_name, pred in inference_response.predictions.items()
        ]
        predictions = sorted(predictions, key=lambda x: x[1].confidence, reverse=True)
        for i, (cls_name, pred) in enumerate(predictions):
            color = class_id_2_color.get(cls_name, "#4892EA")
            text = f"{cls_name} {pred.confidence:.2f}"
            text_size = font.getbbox(text)

            # set button size + 10px margins
            button_size = (text_size[2] + 20, text_size[3] + 20)
            button_img = Image.new("RGBA", button_size, color)
            # put text on button with 10px margins
            button_draw = ImageDraw.Draw(button_img)
            button_draw.text((10, 10), text, font=font, fill=(255, 255, 255, 255))

            # put button on source image in position (0, 0)
            image.paste(button_img, (0, row))
            row += button_size[1]

    buffered = BytesIO()
    image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    return buffered.getvalue()


class InferenceModelsSemanticSegmentationAdapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY
        model_id = resolve_roboflow_model_alias(model_id=model_id)

        self.task_type = "semantic-segmentation"

        extra_weights_provider_headers = get_extra_weights_provider_headers(
            countinference=kwargs.get("countinference"),
            service_secret=kwargs.get("service_secret"),
        )
        backend = list(
            VALID_INFERENCE_MODELS_BACKENDS.difference(
                DISABLED_INFERENCE_MODELS_BACKENDS
            )
        )
        self._model: SemanticSegmentationModel = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            weights_provider_extra_headers=extra_weights_provider_headers,
            backend=backend,
            **kwargs,
        )
        self.class_names = list(self._model.class_names)

    @property
    def class_map(self):
        # match segment.roboflow.com
        return {str(k): v for k, v in enumerate(self.class_names)}

    def map_inference_kwargs(self, kwargs: dict) -> dict:
        kwargs["input_color_format"] = "bgr"
        pre_processing_overrides = PreProcessingOverrides(
            disable_contrast_enhancement=kwargs.get("disable_preproc_contrast", False),
            disable_grayscale=kwargs.get("disable_preproc_grayscale", False),
            disable_static_crop=kwargs.get("disable_preproc_static_crop", False),
        )
        kwargs["pre_processing_overrides"] = pre_processing_overrides
        return kwargs

    def preprocess(self, image: Any, **kwargs):
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        np_images: List[np.ndarray] = [
            load_image_bgr(
                v,
                disable_preproc_auto_orient=kwargs.get(
                    "disable_preproc_auto_orient", False
                ),
            )
            for v in images
        ]
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.pre_process(np_images, **mapped_kwargs)

    def predict(self, img_in, **kwargs):
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.forward(img_in, **mapped_kwargs)

    def postprocess(
        self,
        predictions: torch.Tensor,
        preprocess_return_metadata: PreprocessingMetadata,
        **kwargs,
    ) -> List[SemanticSegmentationInferenceResponse]:
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        segmentation_results = self._model.post_process(
            predictions, preprocess_return_metadata, **mapped_kwargs
        )

        responses: List[SemanticSegmentationInferenceResponse] = []
        for preproc_metadata, segmentation in zip(
            preprocess_return_metadata, segmentation_results
        ):
            height = preproc_metadata.original_size.height
            width = preproc_metadata.original_size.width
            response_image = InferenceResponseImage(width=width, height=height)
            # WARNING! This way of conversion is hazardous - first of all, if background class is not in class names,
            # for certain pre-processing, we end up with -1 values which will be wrapped to 255 - second of all,
            # we can support only 256 classes - those constraints should be fine until inference 2.0
            response_predictions = SemanticSegmentationPrediction(
                segmentation_mask=self.img_to_b64_str(
                    segmentation.segmentation_map.to(torch.uint8)
                ),
                confidence_mask=self.img_to_b64_str(
                    (segmentation.confidence * 255).to(torch.uint8)
                ),
                class_map=self.class_map,
                image=dict(response_image),
            )
            response = SemanticSegmentationInferenceResponse(
                predictions=response_predictions,
                image=response_image,
            )
            responses.append(response)
        return responses

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        """Clears any cache if necessary. TODO: Implement this to delete the cache from the experimental model.

        Args:
            delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to True.
        """
        pass

    def img_to_b64_str(self, img: torch.Tensor) -> str:
        if img.dtype != torch.uint8:
            raise ValueError(
                f"img_to_b64_str requires uint8 tensor but got dtype {img.dtype}"
            )

        img = Image.fromarray(img.cpu().numpy())
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")

        img_str = base64.b64encode(buffered.getvalue())
        img_str = img_str.decode("ascii")

        return img_str

    def draw_predictions(
        self,
        inference_request: InferenceRequest,
        inference_response: InferenceResponse,
    ) -> bytes:
        raise NotImplementedError(
            "draw_predictions(...) is not implemented for semantic segmentation models - responses contain "
            "visualization already."
        )


class InferenceModelsDepthEstimationAdapter(Model):
    """Serves any `inference_models` DepthEstimationModel (e.g. YOLO26-depth)
    behind the depth-estimation contract shared with the DepthAnything
    adapters: per-image min-max-normalized depth plus a viridis
    visualization."""

    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY
        model_id = resolve_roboflow_model_alias(model_id=model_id)

        self.task_type = "depth-estimation"

        extra_weights_provider_headers = get_extra_weights_provider_headers(
            countinference=kwargs.get("countinference"),
            service_secret=kwargs.get("service_secret"),
        )
        backend = list(
            VALID_INFERENCE_MODELS_BACKENDS.difference(
                DISABLED_INFERENCE_MODELS_BACKENDS
            )
        )
        self._model: DepthEstimationModel = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            weights_provider_extra_headers=extra_weights_provider_headers,
            backend=backend,
            **kwargs,
        )

    def preprocess(self, image: Any, **kwargs):
        if isinstance(image, list):
            raise ValueError(
                "Depth estimation does not support batched inference."
            )
        np_image = load_image_bgr(
            image,
            disable_preproc_auto_orient=kwargs.get(
                "disable_preproc_auto_orient", False
            ),
        )
        return np_image, PreprocessReturnMetadata(
            {"image_dims": (np_image.shape[1], np_image.shape[0])}
        )

    def predict(self, inputs: np.ndarray, **kwargs) -> Tuple[dict]:
        import matplotlib.pyplot as plt

        predictions = self._model(inputs)[0]
        depth_map = predictions.to(torch.float32).cpu().numpy()
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max == depth_min:
            raise ValueError("Depth map has no variation (min equals max)")
        normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)

        depth_for_viz = (normalized_depth * 255.0).astype(np.uint8)
        cmap = plt.get_cmap("viridis")
        colored_depth = (cmap(depth_for_viz)[:, :, :3] * 255).astype(np.uint8)

        parent_metadata = ImageParentMetadata(parent_id=f"{uuid4()}")
        colored_depth_image = WorkflowImageData(
            numpy_image=colored_depth, parent_metadata=parent_metadata
        )
        result = {
            "image": colored_depth_image,
            "normalized_depth": normalized_depth,
        }
        return (result,)

    def postprocess(
        self,
        predictions: Tuple[dict],
        preprocess_return_metadata: PreprocessReturnMetadata,
        **kwargs,
    ) -> List[LMMInferenceResponse]:
        image_dims = preprocess_return_metadata["image_dims"]
        response = LMMInferenceResponse(
            response=predictions[0],
            image=InferenceResponseImage(width=image_dims[0], height=image_dims[1]),
        )
        return [response]

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        pass
