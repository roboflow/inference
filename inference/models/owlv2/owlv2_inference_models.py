import gc
import random
import time
import weakref
from contextlib import nullcontext
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from inference.core import logger
from inference.core.entities.responses import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference.core.env import (
    ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
    ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
    API_KEY,
    DISABLED_INFERENCE_MODELS_BACKENDS,
    MAX_DETECTIONS,
    OWLV2_CACHE_SEND_TO_CPU,
    OWLV2_COMPILE_MODEL,
    OWLV2_DIAGNOSTIC_CUDA_SYNC,
    OWLV2_DIAGNOSTIC_INFLIGHT_THRESHOLD,
    OWLV2_DIAGNOSTIC_LOGGING,
    OWLV2_DIAGNOSTIC_PHASE_LOGGING,
    OWLV2_DIAGNOSTIC_SAMPLE_RATE,
    OWLV2_DIAGNOSTIC_SLOW_MS,
    OWLV2_IMAGE_CACHE_SIZE,
    OWLV2_MODEL_CACHE_SIZE,
    PRELOAD_HF_IDS,
    USE_INFERENCE_MODELS,
    VALID_INFERENCE_MODELS_BACKENDS,
)
from inference.core.models.base import Model
from inference.core.models.roboflow import DEFAULT_COLOR_PALETTE
from inference.core.roboflow_api import (
    ENFORCE_CREDITS_VERIFICATION_HEADER,
    get_extra_weights_provider_headers,
)
from inference.core.utils.image_utils import load_image_bgr
from inference.core.utils.visualisation import draw_detection_predictions
from inference_models import AutoModel, Detections
from inference_models.models.owlv2.cache import (
    InMemoryOwlV2ClassEmbeddingsCache,
    InMemoryOwlV2ImageEmbeddingsCache,
)
from inference_models.models.owlv2.entities import (
    ReferenceBoundingBox,
    ReferenceExample,
)
from inference_models.models.owlv2.owlv2_hf import (
    OWLv2HF,
    monkey_patch_vision_encoder_before_compilation,
)

PRELOADED_HF_MODELS = {}
_OWLV2_DIAGNOSTIC_LOCK = Lock()
_OWLV2_DIAGNOSTIC_INFLIGHT = 0


def _change_owlv2_diagnostic_inflight(delta: int) -> int:
    global _OWLV2_DIAGNOSTIC_INFLIGHT
    with _OWLV2_DIAGNOSTIC_LOCK:
        _OWLV2_DIAGNOSTIC_INFLIGHT += delta
        return _OWLV2_DIAGNOSTIC_INFLIGHT


def _sync_cuda_for_diagnostics() -> None:
    try:
        if (
            OWLV2_DIAGNOSTIC_CUDA_SYNC
            and torch.cuda.is_available()
            and torch.cuda.is_initialized()
        ):
            torch.cuda.synchronize()
    except Exception:
        return None


def _cuda_memory_snapshot() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {}
    try:
        _sync_cuda_for_diagnostics()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "cuda_allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 2),
            "cuda_reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 2),
            "cuda_max_allocated_mb": round(
                torch.cuda.max_memory_allocated() / (1024**2), 2
            ),
            "cuda_free_mb": round(free_bytes / (1024**2), 2),
            "cuda_total_mb": round(total_bytes / (1024**2), 2),
        }
    except Exception as exc:
        return {"cuda_memory_error": str(exc)}


def _cache_state(cache: Any) -> Optional[Any]:
    state = getattr(cache, "_state", None)
    if state is None:
        return None
    return state


def _cache_len(cache: Any) -> Optional[int]:
    state = _cache_state(cache)
    if state is None:
        return None
    try:
        return len(state)
    except Exception:
        return None


def _tensor_stats(value: Any) -> Tuple[int, int, int]:
    if isinstance(value, torch.Tensor):
        bytes_count = value.nelement() * value.element_size()
        return bytes_count, int(value.is_cuda), bytes_count if value.is_cuda else 0
    if isinstance(value, dict):
        return _sum_tensor_stats(value.values())
    if isinstance(value, (list, tuple, set)):
        return _sum_tensor_stats(value)
    if hasattr(value, "__dict__"):
        return _sum_tensor_stats(vars(value).values())
    return 0, 0, 0


def _sum_tensor_stats(values: Any) -> Tuple[int, int, int]:
    total_bytes, cuda_tensors, cuda_bytes = 0, 0, 0
    for value in values:
        value_total, value_cuda_tensors, value_cuda_bytes = _tensor_stats(value)
        total_bytes += value_total
        cuda_tensors += value_cuda_tensors
        cuda_bytes += value_cuda_bytes
    return total_bytes, cuda_tensors, cuda_bytes


def _cache_snapshot(model: OWLv2HF, include_tensor_stats: bool) -> Dict[str, Any]:
    try:
        image_cache = getattr(model, "_owlv2_images_embeddings_cache", None)
        class_cache = getattr(model, "_owlv2_class_embeddings_cache", None)
        result = {
            "image_cache_size": _cache_len(image_cache),
            "class_cache_size": _cache_len(class_cache),
        }
        if include_tensor_stats:
            image_cache_state = _cache_state(image_cache)
            if image_cache_state is not None:
                total_bytes, cuda_tensors, cuda_bytes = _tensor_stats(image_cache_state)
                result.update(
                    {
                        "image_cache_tensor_mb": round(total_bytes / (1024**2), 2),
                        "image_cache_cuda_tensors": cuda_tensors,
                        "image_cache_cuda_mb": round(cuda_bytes / (1024**2), 2),
                    }
                )
            class_cache_state = _cache_state(class_cache)
            if class_cache_state is not None:
                total_bytes, cuda_tensors, cuda_bytes = _tensor_stats(class_cache_state)
                result.update(
                    {
                        "class_cache_tensor_mb": round(total_bytes / (1024**2), 2),
                        "class_cache_cuda_tensors": cuda_tensors,
                        "class_cache_cuda_mb": round(cuda_bytes / (1024**2), 2),
                    }
                )
        return result
    except Exception as exc:
        return {"cache_snapshot_error": str(exc)}


def _log_owlv2_diagnostic(message: str, **payload: Any) -> None:
    try:
        logger.info(message, diagnostic_event="owlv2_diagnostic", **payload)
    except TypeError:
        try:
            logger.info("%s %s", message, payload)
        except Exception:
            return None
    except Exception:
        return None


class _Owlv2Diagnostics:
    def __init__(
        self,
        *,
        model_id: str,
        model: OWLv2HF,
        training_data: List[dict],
        image_count: int,
        confidence: float,
        iou_threshold: float,
        max_detections: int,
    ):
        self.model_id = model_id
        self.model = model
        self.enabled = OWLV2_DIAGNOSTIC_LOGGING
        self.sampled = False
        self.pressure_sampled = False
        self.phase_logging = False
        self.inflight_at_start = 0
        self.start_time = 0.0
        self.phases: Dict[str, float] = {}
        self.start_cache: Dict[str, Any] = {}
        self.start_cuda: Dict[str, Any] = {}
        self.capture_start_snapshot = False
        self.request = {
            "model_id": model_id,
            "image_count": image_count,
            "confidence": confidence,
            "iou_threshold": iou_threshold,
            "max_detections": max_detections,
            "compile_model": OWLV2_COMPILE_MODEL,
            "cache_send_to_cpu": OWLV2_CACHE_SEND_TO_CPU,
        }
        try:
            self.request.update(
                {
                    "training_examples": len(training_data),
                    "training_boxes": sum(
                        len(example.get("boxes", [])) for example in training_data
                    ),
                    "training_classes": len(
                        {
                            box.get("cls")
                            for example in training_data
                            for box in example.get("boxes", [])
                            if box.get("cls") is not None
                        }
                    ),
                }
            )
        except Exception as exc:
            self.request["request_shape_error"] = str(exc)

    def __enter__(self) -> "_Owlv2Diagnostics":
        if not self.enabled:
            return self
        self.inflight_at_start = _change_owlv2_diagnostic_inflight(1)
        self.sampled = random.random() < max(0.0, OWLV2_DIAGNOSTIC_SAMPLE_RATE)
        self.pressure_sampled = (
            self.inflight_at_start >= OWLV2_DIAGNOSTIC_INFLIGHT_THRESHOLD
        )
        self.phase_logging = OWLV2_DIAGNOSTIC_PHASE_LOGGING and (
            self.sampled or self.pressure_sampled
        )
        self.capture_start_snapshot = self.sampled or self.pressure_sampled
        self.start_time = time.perf_counter()
        if self.capture_start_snapshot:
            self.start_cache = _cache_snapshot(self.model, include_tensor_stats=False)
            self.start_cuda = _cuda_memory_snapshot()
        if self.phase_logging:
            self.log(
                "OWLv2 diagnostic request start",
                event_type="request_start",
                inflight=self.inflight_at_start,
                **self.request,
                **self.start_cache,
                **self.start_cuda,
            )
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if not self.enabled:
            return None
        total_ms = round((time.perf_counter() - self.start_time) * 1000, 2)
        inflight_after = _change_owlv2_diagnostic_inflight(-1)
        should_log = (
            exc is not None
            or self.sampled
            or self.pressure_sampled
            or total_ms >= OWLV2_DIAGNOSTIC_SLOW_MS
        )
        if should_log:
            end_cache = _cache_snapshot(self.model, include_tensor_stats=True)
            end_cuda = _cuda_memory_snapshot()
            self.log(
                "OWLv2 diagnostic request summary",
                event_type="request_summary",
                total_ms=total_ms,
                slow=total_ms >= OWLV2_DIAGNOSTIC_SLOW_MS,
                sampled=self.sampled,
                pressure_sampled=self.pressure_sampled,
                start_snapshot_captured=self.capture_start_snapshot,
                inflight_at_start=self.inflight_at_start,
                inflight_after=inflight_after,
                phases_ms=self.phases,
                error_type=exc_type.__name__ if exc_type is not None else None,
                error=str(exc) if exc is not None else None,
                cache_start=self.start_cache,
                cache_end=end_cache,
                cuda_start=self.start_cuda,
                cuda_end=end_cuda,
                **self.request,
            )
        return None

    def phase(self, name: str, **metadata: Any):
        if not self.enabled:
            return nullcontext()
        return _Owlv2DiagnosticPhase(self, name, metadata)

    def log(self, message: str, **payload: Any) -> None:
        _log_owlv2_diagnostic(message, **payload)


class _Owlv2DiagnosticPhase:
    def __init__(
        self, diagnostics: _Owlv2Diagnostics, name: str, metadata: Dict[str, Any]
    ):
        self.diagnostics = diagnostics
        self.name = name
        self.metadata = metadata
        self.start_time = 0.0

    def __enter__(self):
        _sync_cuda_for_diagnostics()
        self.start_time = time.perf_counter()
        if self.diagnostics.phase_logging:
            self.diagnostics.log(
                "OWLv2 diagnostic phase start",
                event_type="phase_start",
                phase=self.name,
                **self.metadata,
            )
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        _sync_cuda_for_diagnostics()
        duration_ms = round((time.perf_counter() - self.start_time) * 1000, 2)
        self.diagnostics.phases[self.name] = (
            self.diagnostics.phases.get(self.name, 0.0) + duration_ms
        )
        if self.diagnostics.phase_logging:
            cache_fields = _cache_snapshot(
                self.diagnostics.model, include_tensor_stats=False
            )
            self.diagnostics.log(
                "OWLv2 diagnostic phase end",
                event_type="phase_end",
                phase=self.name,
                duration_ms=duration_ms,
                error_type=exc_type.__name__ if exc_type is not None else None,
                error=str(exc) if exc is not None else None,
                **self.metadata,
                **cache_fields,
            )
        return None


class Owlv2AdapterSingleton:
    _instances = weakref.WeakValueDictionary()

    def __new__(
        cls,
        huggingface_id: str,
        api_key: Optional[str] = None,
        disable_credits_verification_enforcement: bool = False,
    ):
        if huggingface_id in PRELOADED_HF_MODELS:
            logger.info("Using preloaded OWLv2 instance for %s", huggingface_id)
            return PRELOADED_HF_MODELS[huggingface_id]
        if huggingface_id not in cls._instances:
            owlv2_class_embeddings_cache = InMemoryOwlV2ClassEmbeddingsCache.init(
                size_limit=OWLV2_MODEL_CACHE_SIZE,
                send_to_cpu=OWLV2_CACHE_SEND_TO_CPU,
            )
            owlv2_images_embeddings_cache = InMemoryOwlV2ImageEmbeddingsCache.init(
                size_limit=OWLV2_IMAGE_CACHE_SIZE,
                send_to_cpu=OWLV2_CACHE_SEND_TO_CPU,
            )
            weights_provider_extra_headers = get_extra_weights_provider_headers()
            if (
                disable_credits_verification_enforcement
                and weights_provider_extra_headers is not None
            ):
                weights_provider_extra_headers.pop(
                    ENFORCE_CREDITS_VERIFICATION_HEADER, None
                )
            backend = list(
                VALID_INFERENCE_MODELS_BACKENDS.difference(
                    DISABLED_INFERENCE_MODELS_BACKENDS
                )
            )
            model: OWLv2HF = AutoModel.from_pretrained(
                model_id_or_path=huggingface_id,
                api_key=api_key or API_KEY,
                allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
                allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
                owlv2_class_embeddings_cache=owlv2_class_embeddings_cache,
                owlv2_images_embeddings_cache=owlv2_images_embeddings_cache,
                weights_provider_extra_headers=weights_provider_extra_headers,
                backend=backend,
            )
            logger.info("Creating new OWLv2 instance for %s", huggingface_id)
            if OWLV2_COMPILE_MODEL:
                logger.info("Compiling OWLv2 model %s", huggingface_id)
                torch._dynamo.config.suppress_errors = True
                model._model = monkey_patch_vision_encoder_before_compilation(
                    model._model
                )
                model._model.owlv2.vision_model = torch.compile(
                    model._model.owlv2.vision_model
                )
                model._compiled = True
            logger.info("Caching OWLv2 model %s", huggingface_id)
            cls._instances[huggingface_id] = model
            instance = cls.assembly_instance(huggingface_id, model)
            return instance
        model = cls._instances[huggingface_id]
        return cls.assembly_instance(huggingface_id, model)

    @classmethod
    def assembly_instance(cls, huggingface_id: str, model):
        instance = super().__new__(cls)
        instance.huggingface_id = huggingface_id
        instance.model = model
        return instance


@torch.inference_mode()
def dummy_infer(
    hf_id: str,
    api_key: Optional[str] = None,
    disable_credits_verification_enforcement: bool = False,
):
    # Below code is copied from Owlv2.__init__
    singleton = Owlv2AdapterSingleton(
        hf_id,
        api_key=api_key,
        disable_credits_verification_enforcement=disable_credits_verification_enforcement,
    )
    model = singleton.model
    # Below code is copied from Owlv2.embed_image
    np_image = np.zeros((256, 256, 3), dtype=np.uint8)
    pixel_values, _ = model.pre_process(np_image)
    device_type = model._device.type
    with torch.autocast(
        device_type=device_type, dtype=torch.float16, enabled=device_type == "cuda"
    ):
        embeddings, *_ = model._model.image_embedder(pixel_values=pixel_values)
    del pixel_values, np_image, embeddings
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return singleton


def preload_owlv2_model(
    hf_id: str,
    api_key: Optional[str] = None,
    disable_credits_verification_enforcement: bool = False,
):
    logger.info("Preloading OWLv2 model for %s (this may take a while)", hf_id)
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated_gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            logger.info(
                f"Allocated GPU memory before loading model: {allocated_gpu_memory:.2f} GB"
            )
        t1 = time.time()
        singleton = dummy_infer(
            hf_id,
            api_key=api_key,
            disable_credits_verification_enforcement=disable_credits_verification_enforcement,
        )
        t2 = time.time()
        logger.info("Preloaded OWLv2 model for %s in %0.2f seconds", hf_id, t2 - t1)
        if torch.cuda.is_available():
            allocated_gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            logger.info(
                f"Allocated GPU memory after loading model: {allocated_gpu_memory:.2f} GB"
            )
        # Store the singleton instance directly in PRELOADED_HF_MODELS
        PRELOADED_HF_MODELS[hf_id] = singleton
    except Exception as exc:
        logger.error("Failed to preload OWLv2 model for %s: %s", hf_id, exc)


if PRELOAD_HF_IDS and USE_INFERENCE_MODELS:
    hf_ids = PRELOAD_HF_IDS
    if not isinstance(hf_ids, list):
        hf_ids = [hf_ids]
    for hf_id in hf_ids:
        preload_owlv2_model(
            hf_id, api_key=API_KEY, disable_credits_verification_enforcement=True
        )


class InferenceModelsOwlV2Adapter(Model):
    def __init__(
        self,
        model_id: str = "owlv2/owlv2-large-patch14-ensemble",
        api_key: str = None,
        **kwargs,
    ):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY

        self.task_type = "object-detection"
        self.model_id = model_id
        singleton_instance = Owlv2AdapterSingleton(model_id, api_key=self.api_key)
        self._model: OWLv2HF = singleton_instance.model

    def draw_predictions(
        self,
        inference_request,
        inference_response,
    ) -> bytes:
        """Draw predictions from an inference response onto the original image provided by an inference request

        Args:
            inference_request (ObjectDetectionInferenceRequest): The inference request containing the image on which to draw predictions
            inference_response (ObjectDetectionInferenceResponse): The inference response containing predictions to be drawn

        Returns:
            str: A base64 encoded image string
        """
        all_class_names = [x.class_name for x in inference_response.predictions]
        all_class_names = sorted(list(set(all_class_names)))
        return draw_detection_predictions(
            inference_request=inference_request,
            inference_response=inference_response,
            colors={
                class_name: DEFAULT_COLOR_PALETTE[i % len(DEFAULT_COLOR_PALETTE)]
                for (i, class_name) in enumerate(all_class_names)
            },
        )

    def embed_image(self, image: np.ndarray) -> Tuple[str, tuple]:
        image_embeddings = self._model.embed_image(image=image)
        image_embeds = (
            image_embeddings.objectness,
            image_embeddings.boxes,
            image_embeddings.image_class_embeddings,
            image_embeddings.logit_shift,
            image_embeddings.logit_scale,
        )
        return image_embeddings.image_hash, image_embeds

    def infer(
        self,
        image: Any,
        training_data: List[dict],
        confidence: float = 0.95,
        iou_threshold: float = 0.3,
        max_detections: int = MAX_DETECTIONS,
        **kwargs,
    ):
        diagnostics = _Owlv2Diagnostics(
            model_id=getattr(self, "model_id", "owlv2/owlv2-large-patch14-ensemble"),
            model=self._model,
            training_data=training_data,
            image_count=len(image) if isinstance(image, list) else 1,
            confidence=confidence,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )
        with diagnostics:
            with diagnostics.phase("build_reference_examples"):
                reference_examples = []
                for example in training_data:
                    boxes = []
                    for box in example["boxes"]:
                        boxes.append(ReferenceBoundingBox.model_validate(box))
                    reference_examples.append(
                        ReferenceExample(
                            image=example["image"]["value"],
                            boxes=boxes,
                        )
                    )
            with diagnostics.phase("load_target_images"):
                if isinstance(image, list):
                    image_decoded = [load_image_bgr(i) for i in image]
                else:
                    image_decoded = [load_image_bgr(image)]
                image_sizes: List[Tuple[int, int]] = [
                    i.shape[:2][::-1] for i in image_decoded
                ]  # type: ignore

            if not OWLV2_DIAGNOSTIC_LOGGING:
                results = self._model.infer_with_reference_examples(
                    images=image_decoded,
                    reference_examples=reference_examples,
                    confidence=confidence,
                    iou_threshold=iou_threshold,
                    max_detections=max_detections,
                )
                return self.make_response(predictions=results, image_sizes=image_sizes)

            with diagnostics.phase("prepare_reference_embeddings"):
                reference_embeddings = (
                    self._model.prepare_reference_examples_embeddings(
                        reference_examples=reference_examples,
                        iou_threshold=iou_threshold,
                    )
                )
            with diagnostics.phase("embed_target_images"):
                images_embeddings, image_dimensions = self._model.embed_images(
                    images=image_decoded, max_detections=max_detections
                )
            with diagnostics.phase(
                "forward_precomputed_embeddings",
                class_count=len(reference_embeddings.class_embeddings),
            ):
                images_predictions = (
                    self._model.forward_pass_with_precomputed_embeddings(
                        images_embeddings=images_embeddings,
                        class_embeddings=reference_embeddings.class_embeddings,
                        confidence=confidence,
                        iou_threshold=iou_threshold,
                    )
                )
            with diagnostics.phase("post_process_predictions"):
                results = (
                    self._model.post_process_predictions_for_precomputed_embeddings(
                        predictions=images_predictions,
                        images_dimensions=image_dimensions,
                        max_detections=max_detections,
                        iou_threshold=iou_threshold,
                    )
                )
            with diagnostics.phase(
                "make_response",
                prediction_count=sum(
                    prediction.xyxy.shape[0] for prediction in results
                ),
            ):
                return self.make_response(predictions=results, image_sizes=image_sizes)

    def make_response(
        self,
        predictions: List[Detections],
        image_sizes: List[Tuple[int, int]],
        **kwargs,
    ) -> List[ObjectDetectionInferenceResponse]:
        responses = []
        for image_prediction, image_size in zip(predictions, image_sizes):
            image_instances = []
            for instance_id in range(image_prediction.xyxy.shape[0]):
                x_min, y_min, x_max, y_max = image_prediction.xyxy[instance_id].tolist()
                width = x_max - x_min
                height = y_max - y_min
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                class_id = image_prediction.class_id[instance_id].item()
                confidence = image_prediction.confidence[instance_id].item()
                class_name = image_prediction.image_metadata["class_names"][class_id]
                image_instances.append(
                    ObjectDetectionPrediction(
                        # Passing args as a dictionary here since one of the args is 'class' (a protected term in Python)
                        **{
                            "x": center_x,
                            "y": center_y,
                            "width": width,
                            "height": height,
                            "confidence": confidence,
                            "class": class_name,
                            "class_id": class_id,
                        }
                    )
                )
            responses.append(
                ObjectDetectionInferenceResponse(
                    predictions=image_instances,
                    image=InferenceResponseImage(
                        width=image_size[0], height=image_size[1]
                    ),
                )
            )
        return responses

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        """Clears any cache if necessary. TODO: Implement this to delete the cache from the experimental model.

        Args:
            delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to True.
        """
        pass
