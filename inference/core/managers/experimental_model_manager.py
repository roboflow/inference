import threading
from collections import deque
from typing import Deque, Dict, List, Optional, Union

import torch

from inference_exp.models.auto_loaders.core import AutoModel, AnyModel
from inference.core import logger
from inference.core.env import MEMORY_FREE_THRESHOLD


class ExperimentalModelManager:
    """Thread-safe in-memory manager for inference-exp models.

    Responsibilities:
    - Keep a cache of instantiated models keyed by `model_id`.
    - Ensure only one instantiation per `model_id` via per-model locks.
    - Return existing instances without re-loading when available.
    - Provide removal and clearing utilities (best-effort GPU memory release).
    - expose a basic GPU memory availability heuristic.

    Notes:
    - File download and on-disk locking are already handled by `AutoModel`.
    - This manager focuses solely on in-memory instance management.
    - Cross-process deduplication is out of scope for now; rely on `AutoModel` file locks.
    """

    _instance: Optional["ExperimentalModelManager"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._models: Dict[str, AnyModel] = {}
        self._models_lock = threading.Lock()
        # Per-model instantiation locks to avoid concurrent in-memory loads
        self._model_locks: Dict[str, threading.Lock] = {}
        # LRU queue of model_ids; right side is most recently used
        self._lru_queue: Deque[str] = deque()

    @classmethod
    def get_instance(cls) -> "ExperimentalModelManager":
        # Double-checked locking for singleton instance
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = ExperimentalModelManager()
        return cls._instance

    def is_loaded(self, model_id: str) -> bool:
        with self._models_lock:
            return model_id in self._models

    def list_models(self) -> List[str]:
        with self._models_lock:
            return list(self._models.keys())

    def _get_model_lock(self, model_id: str) -> threading.Lock:
        # Ensure a stable per-model lock exists
        with self._models_lock:
            lock = self._model_locks.get(model_id)
            if lock is None:
                lock = threading.Lock()
                self._model_locks[model_id] = lock
            return lock

    def _mark_as_used(self, model_id: str) -> None:
        with self._models_lock:
            try:
                self._lru_queue.remove(model_id)
            except ValueError:
                pass
            self._lru_queue.append(model_id)

    def _evict_if_needed(
        self,
        *,
        min_free_ratio: Optional[float] = None,
        estimated_bytes: Optional[int] = None,
    ) -> None:
        """Evict least-recently used models under memory pressure only.

        If `min_free_ratio` is None, use MEMORY_FREE_THRESHOLD when valid.
        If `estimated_bytes` is provided, ensure free >= estimated_bytes.
        """
        if min_free_ratio is None:
            if (
                isinstance(MEMORY_FREE_THRESHOLD, float)
                and 0.0 < MEMORY_FREE_THRESHOLD < 1.0
            ):
                min_free_ratio = MEMORY_FREE_THRESHOLD
        self.evict_until_memory(
            min_free_ratio=min_free_ratio, estimated_bytes=estimated_bytes
        )

    def get_or_load_model(
        self,
        model_id: str,
        *,
        api_key: Optional[str] = None,
        device: Optional[torch.device] = None,
        backends: Optional[Union[str, List[str]]] = None,
        batch_size: Optional[Union[int, tuple]] = None,
        quantization: Optional[Union[str, List[str]]] = None,
        onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
        verbose: bool = False,
        min_free_gpu_memory_ratio: Optional[float] = None,
        estimated_gpu_memory_bytes: Optional[int] = None,
        **kwargs,
    ) -> AnyModel:
        # Fast path: return cached without global lock if present (benign race acceptable)
        with self._models_lock:
            cached = self._models.get(model_id)
        if cached is not None:
            # mark as most recently used (must be outside _models_lock to avoid deadlock)
            self._mark_as_used(model_id)
            return cached

        # Serialize instantiation per model_id
        model_lock = self._get_model_lock(model_id)
        with model_lock:
            # Re-check after acquiring the lock
            with self._models_lock:
                cached = self._models.get(model_id)
            if cached is not None:
                # mark as most recently used (must be outside _models_lock to avoid deadlock)
                self._mark_as_used(model_id)
                return cached

            # Optionally check GPU memory before attempting load
            # (heuristic; proceed if unknown)
            if device is None and torch.cuda.is_available():
                device = torch.device("cuda")

            # Proactively evict before loading if needed (memory-only policy for now)
            self._evict_if_needed(
                min_free_ratio=min_free_gpu_memory_ratio,
                estimated_bytes=estimated_gpu_memory_bytes,
            )

            model = AutoModel.from_pretrained(
                model_name_or_path=model_id,
                api_key=api_key,
                backends=backends,
                batch_size=batch_size,
                quantization=quantization,
                onnx_execution_providers=onnx_execution_providers,
                device=device if device is not None else torch.device("cpu"),
                verbose=verbose,
                **kwargs,
            )
            with self._models_lock:
                self._models[model_id] = model
            # Mark as most recently used outside of _models_lock to avoid deadlock
            self._mark_as_used(model_id)
            return model

    def remove_model(self, model_id: str) -> None:
        with self._models_lock:
            if model_id in self._models:
                # Drop strong reference to allow GC; best-effort GPU cache release below
                del self._models[model_id]
            try:
                self._lru_queue.remove(model_id)
            except ValueError:
                pass
        self._best_effort_gpu_memory_release()

    def clear(self) -> None:
        with self._models_lock:
            self._models.clear()
            self._lru_queue.clear()
        self._best_effort_gpu_memory_release()

    def _best_effort_gpu_memory_release(self) -> None:
        # Attempt to free CUDA cache if available
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            # Ignore any CUDA context errors; memory will be reclaimed by GC
            pass

    def gpu_memory_info(self) -> Optional[Dict[str, int]]:
        """Return basic GPU memory info in bytes if available, otherwise None.

        Returns a dict with keys: total, free, used.
        """
        try:
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()  # type: ignore[attr-defined]
                used = total - free
                return {"total": int(total), "free": int(free), "used": int(used)}
        except Exception:
            return None
        return None

    def has_sufficient_memory(
        self,
        *,
        estimated_bytes: Optional[int] = None,
        min_free_ratio: Optional[float] = None,
    ) -> bool:
        """Check if current device memory appears sufficient.

        - If CUDA not available, return True.
        - If memory info unavailable, return True.
        - If `estimated_bytes` provided, require free >= estimated_bytes.
        - Else require free/total >= min_free_ratio (or MEMORY_FREE_THRESHOLD if not provided and valid).
        """
        try:
            if not torch.cuda.is_available():
                return True
            mem = self.gpu_memory_info()
            if not mem or mem["total"] <= 0:
                return True
            if estimated_bytes is not None:
                return mem["free"] >= estimated_bytes
            if min_free_ratio is None:
                if (
                    isinstance(MEMORY_FREE_THRESHOLD, float)
                    and 0.0 < MEMORY_FREE_THRESHOLD < 1.0
                ):
                    min_free_ratio = MEMORY_FREE_THRESHOLD
            if min_free_ratio is None:
                return True
            return (mem["free"] / mem["total"]) >= float(min_free_ratio)
        except Exception:
            return True

    # Backwards-compatible alias
    def can_load_model(
        self,
        *,
        estimated_bytes: Optional[int] = None,
        min_free_ratio: float = 0.1,
    ) -> bool:
        return self.has_sufficient_memory(
            estimated_bytes=estimated_bytes, min_free_ratio=min_free_ratio
        )

    def evict_until_memory(
        self,
        *,
        min_free_ratio: Optional[float] = None,
        estimated_bytes: Optional[int] = None,
    ) -> None:
        """Evict LRU models until memory constraint is satisfied (GPU only).

        No-op if CUDA unavailable or memory metrics cannot be obtained.
        """
        try:
            if not torch.cuda.is_available():
                return None
            if min_free_ratio is None:
                if (
                    isinstance(MEMORY_FREE_THRESHOLD, float)
                    and 0.0 < MEMORY_FREE_THRESHOLD < 1.0
                ):
                    min_free_ratio = MEMORY_FREE_THRESHOLD
            # quick check: if already sufficient, return
            if self.has_sufficient_memory(
                estimated_bytes=estimated_bytes, min_free_ratio=min_free_ratio
            ):
                return None
            with self._models_lock:
                free, total = torch.cuda.mem_get_info()  # type: ignore[attr-defined]
                while self._lru_queue:
                    # stop if satisfied
                    if estimated_bytes is not None and free >= estimated_bytes:
                        break
                    if (
                        min_free_ratio is not None
                        and total > 0
                        and float(free) / float(total) >= float(min_free_ratio)
                    ):
                        break
                    to_remove = self._lru_queue.popleft()
                    if to_remove in self._models:
                        logger.debug(
                            f"ExperimentalModelManager: Evicting LRU model: {to_remove}"
                        )
                        del self._models[to_remove]
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    # refresh memory
                    free, total = torch.cuda.mem_get_info()  # type: ignore[attr-defined]
        except Exception as e:
            logger.debug(
                f"ExperimentalModelManager: evict_until_memory skipped due to error: {e}"
            )
