from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


def detect_max_batch_size(model) -> Optional[int]:
    """Duck-type max batch size from a model instance.

    Models use various attribute names. Returns None if not discoverable.
    """
    bs = (
        getattr(model, "max_batch_size", None)
        or getattr(model, "_max_batch_size", None)
        or getattr(model, "_input_batch_size", None)
    )
    if callable(bs):
        bs = None
    if bs is not None:
        return bs
    # TorchScript/TRT models store it in inference_config
    cfg = getattr(model, "_inference_config", None)
    if cfg is not None:
        fwd = getattr(cfg, "forward_pass", None)
        if fwd is not None:
            sbs = getattr(fwd, "static_batch_size", None)
            if sbs is not None:
                return sbs
            mdb = getattr(fwd, "max_dynamic_batch_size", None)
            if mdb is not None:
                return mdb
    # TRT config (separate from inference_config)
    trt_cfg = getattr(model, "_trt_config", None)
    if trt_cfg is not None:
        sbs = getattr(trt_cfg, "static_batch_size", None)
        if sbs is not None:
            return sbs
        return getattr(trt_cfg, "dynamic_batch_size_max", None)
    return None


def attach_model_caches(model) -> None:
    """Replace null-object embedding caches with real in-memory ones.

    Cache objects hold locks so they cannot cross the worker spawn boundary
    in model_kwargs; they must be attached in-process after the model loads.
    Caches that are already in-memory instances are kept — a shared-base
    worker re-runs this per head against the same resident base and must not
    wipe its warm cache.
    """
    _attach_sam_caches(model)
    _attach_sam2_caches(model)
    _attach_sam3_caches(model)
    _attach_owlv2_caches(model)
    feature_extractor = getattr(model, "_feature_extractor", None)
    if feature_extractor is not None:
        _attach_owlv2_caches(feature_extractor)


def _attach_sam_caches(model) -> None:
    if type(model).__name__ != "SAMTorch":
        return
    from inference_model_manager import configuration as cfg
    from inference_models.models.sam.cache import (
        SamImageEmbeddingsInMemoryCache,
        SamLowResolutionMasksInMemoryCache,
    )

    model._sam_allow_client_generated_hash_ids = True

    if not isinstance(
        model._sam_image_embeddings_cache, SamImageEmbeddingsInMemoryCache
    ):
        model._sam_image_embeddings_cache = SamImageEmbeddingsInMemoryCache.init(
            size_limit=cfg.SAM_MAX_EMBEDDING_CACHE_SIZE,
            send_to_cpu=True,
        )
    if not isinstance(
        model._sam_low_resolution_masks_cache, SamLowResolutionMasksInMemoryCache
    ):
        model._sam_low_resolution_masks_cache = SamLowResolutionMasksInMemoryCache.init(
            size_limit=cfg.SAM_MAX_EMBEDDING_CACHE_SIZE,
            send_to_cpu=True,
        )


def _attach_sam2_caches(model) -> None:
    if type(model).__name__ != "SAM2Torch":
        return
    from inference_model_manager import configuration as cfg
    from inference_models.models.sam2.cache import (
        Sam2ImageEmbeddingsInMemoryCache,
        Sam2LowResolutionMasksInMemoryCache,
    )

    model._sam2_allow_client_generated_hash_ids = True

    if not isinstance(
        model._sam2_image_embeddings_cache, Sam2ImageEmbeddingsInMemoryCache
    ):
        model._sam2_image_embeddings_cache = Sam2ImageEmbeddingsInMemoryCache.init(
            size_limit=cfg.SAM2_MAX_EMBEDDING_CACHE_SIZE,
            send_to_cpu=True,
        )
    if not isinstance(
        model._sam2_low_resolution_masks_cache, Sam2LowResolutionMasksInMemoryCache
    ):
        model._sam2_low_resolution_masks_cache = (
            Sam2LowResolutionMasksInMemoryCache.init(
                size_limit=cfg.SAM2_MAX_LOGITS_CACHE_SIZE,
                send_to_cpu=True,
            )
        )


def _attach_sam3_caches(model) -> None:
    if type(model).__name__ != "SAM3Torch":
        return
    from inference_model_manager import configuration as cfg
    from inference_models.models.sam3.cache import (
        Sam3ImageEmbeddingsInMemoryCache,
        Sam3LowResolutionMasksInMemoryCache,
    )

    model._sam3_allow_client_generated_hash_ids = True

    if not isinstance(
        model._sam3_image_embeddings_cache, Sam3ImageEmbeddingsInMemoryCache
    ):
        model._sam3_image_embeddings_cache = Sam3ImageEmbeddingsInMemoryCache.init(
            size_limit=cfg.SAM3_MAX_EMBEDDING_CACHE_SIZE,
            send_to_cpu=cfg.SAM3_INTERACTIVE_CACHE_SEND_TO_CPU,
        )
    if not isinstance(
        model._sam3_low_resolution_masks_cache, Sam3LowResolutionMasksInMemoryCache
    ):
        model._sam3_low_resolution_masks_cache = (
            Sam3LowResolutionMasksInMemoryCache.init(
                size_limit=cfg.SAM3_MAX_LOGITS_CACHE_SIZE,
                send_to_cpu=cfg.SAM3_INTERACTIVE_CACHE_SEND_TO_CPU,
            )
        )


def _attach_owlv2_caches(model) -> None:
    if not hasattr(model, "_owlv2_class_embeddings_cache"):
        return
    from inference_model_manager import configuration as cfg
    from inference_models.models.owlv2.cache import (
        InMemoryOwlV2ClassEmbeddingsCache,
        InMemoryOwlV2ImageEmbeddingsCache,
    )

    if not isinstance(
        model._owlv2_class_embeddings_cache, InMemoryOwlV2ClassEmbeddingsCache
    ):
        model._owlv2_class_embeddings_cache = InMemoryOwlV2ClassEmbeddingsCache.init(
            size_limit=cfg.OWLV2_MODEL_CACHE_SIZE,
            send_to_cpu=cfg.OWLV2_CACHE_SEND_TO_CPU,
        )
    if not isinstance(
        model._owlv2_images_embeddings_cache, InMemoryOwlV2ImageEmbeddingsCache
    ):
        model._owlv2_images_embeddings_cache = InMemoryOwlV2ImageEmbeddingsCache.init(
            size_limit=cfg.OWLV2_IMAGE_CACHE_SIZE,
            send_to_cpu=cfg.OWLV2_CACHE_SEND_TO_CPU,
        )


class Backend(ABC):
    """Public contract for a model backend registered with ``ModelManager``.

    One instance per loaded model. A community backend subclasses this ABC
    and implements its abstract surface below; ``ModelManager`` holds a
    ``Dict[str, Backend]`` keyed by ``model_id`` and consumes that surface
    directly:

    - ``state``: must be the literal string ``"loaded"`` for the model to
      appear in ``ModelManager.loaded_models``. Also returned verbatim by
      ``health()`` and included in ``list_models()``.
    - ``device``: surfaced in ``list_models()``.
    - ``is_healthy``: read by ``ModelManager.is_healthy()``.
    - ``is_accepting``: drives ``ModelManager.is_ready()``; also surfaced in
      ``list_models()`` and gates the direct-dispatch path in ``submit()``.
    - ``max_batch_size``: not read directly by the manager — expected to
      appear as a key in the dict this backend's own ``stats()`` returns
      (see below), which is how it reaches ``ModelManager.stats()`` /
      ``model_stats()``.
    - ``queue_depth``: surfaced in ``list_models()``.
    - ``stats()``: aggregated into ``ModelManager.stats()``; also read
      directly by ``model_stats()`` for a single model.
    - ``class_names``: surfaced in ``ModelManager.stats()``.
    - ``unload()``: called by ``ModelManager.unload()`` when not draining;
      releases the backend's resources.
    - ``drain_and_unload()``: called by ``ModelManager.unload(..., drain=True)``;
      inherited concrete default just calls ``unload()``.
    - ``record_inference()``: called after each direct-dispatch inference;
      inherited concrete default is a no-op.
    - ``worker_pid``: surfaced in ``list_models()``; inherited concrete
      default is ``None``.

    Override these inherited concrete defaults only if the backend needs
    different behavior.

    Inference is exposed one of two ways:

    - Set ``.model`` to an in-process ``inference_models`` model instance —
      ``ModelManager`` dispatches tasks against it through the task registry.
    - Implement ``submit_request(*, task, raw_input, validate, **kwargs) ->
      concurrent.futures.Future`` — ``ModelManager.process()`` /
      ``submit()`` route through it instead.

    Optionally set ``_model_mro_names: list[str]`` (the loaded model's MRO
    class names) so the registry can validate inputs and serialize results
    for backends whose model does not live in-process. Without it, requests
    skip validation and results are returned raw.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def unload(self) -> None:
        """Release all resources (GPU memory, worker processes, SHM).

        Any in-flight requests are cancelled with an error.
        """
        ...

    def drain_and_unload(self, timeout_s: float = 30.0) -> None:
        """Stop accepting new work, wait for in-flight to finish, then unload.

        1. Set state to 'draining' — new submit/signal calls are rejected.
        2. Wait up to ``timeout_s`` for pending work to complete.
        3. If timeout expires, force-cancel remaining work.
        4. Call ``unload()`` for final cleanup.

        Default implementation just calls ``unload()`` immediately.
        Backends override to implement graceful drain.
        """
        self.unload()

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def device(self) -> str:
        """Device this backend runs inference on: 'cpu', 'cuda:0', etc."""
        ...

    @property
    @abstractmethod
    def state(self) -> str:
        """Current state: 'loading', 'loaded', 'draining', 'unhealthy'."""
        ...

    @property
    @abstractmethod
    def is_healthy(self) -> bool:
        """Whether this backend is in a usable state."""
        ...

    @property
    @abstractmethod
    def is_accepting(self) -> bool:
        """Whether this backend can accept new requests right now."""
        ...

    @property
    @abstractmethod
    def max_batch_size(self) -> Optional[int]:
        """Maximum batch size this backend supports, or None if unlimited."""
        ...

    @property
    @abstractmethod
    def queue_depth(self) -> int:
        """Number of pending requests waiting in the batch queue."""
        ...

    def record_inference(self, t0: float, error: bool = False) -> None:
        """Record an inference for stats tracking. Called by ModelManager."""
        pass

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Runtime statistics snapshot. Must be non-blocking.

        Returns dict with at minimum:
            model_id, backend_type, state, is_accepting,
            queue_depth, max_batch_size,
            throughput_fps, latency_p50_ms, latency_p99_ms,
            inference_count, error_count, last_inference_ts
        """
        ...

    @property
    @abstractmethod
    def class_names(self) -> Optional[List[str]]:
        """Class names for the loaded model, if available."""
        ...

    @property
    def model(self) -> Any:
        """Underlying model instance. Used by ModelManager.invoke() for task dispatch.

        Returns None for subprocess backends (model lives in worker process).
        """
        return None

    @property
    def worker_pid(self) -> Optional[int]:
        """OS PID of the worker subprocess, if applicable. None for in-process backends."""
        return None

    @property
    def last_used_ts(self) -> Optional[float]:
        """Monotonic timestamp of the last inference (load time if never
        inferred), used for LRU eviction ordering. None = unknown."""
        return None
