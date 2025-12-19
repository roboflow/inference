from abc import ABC, abstractmethod
from collections import OrderedDict
from threading import Lock
from typing import Optional, List

import torch
from inference_exp.errors import EnvironmentConfigurationError
from inference_exp.models.sam2.entities import SAM2ImageEmbeddings, SAM2MaskCacheEntry


class Sam2ImageEmbeddingsCache(ABC):

    @abstractmethod
    def retrieve_embeddings(self, key: str) -> Optional[SAM2ImageEmbeddings]:
        pass

    @abstractmethod
    def save_embeddings(self, key: str, embeddings: SAM2ImageEmbeddings) -> None:
        pass


class Sam2ImageEmbeddingsCacheNullObject(Sam2ImageEmbeddingsCache):

    def retrieve_embeddings(self, key: str) -> Optional[SAM2ImageEmbeddings]:
        pass

    def save_embeddings(self, key: str, embeddings: SAM2ImageEmbeddings) -> None:
        pass


class Sam2ImageEmbeddingsInMemoryCache(Sam2ImageEmbeddingsCache):

    @classmethod
    def init(
        cls, size_limit: Optional[int], send_to_cpu: bool = True
    ) -> "Sam2ImageEmbeddingsInMemoryCache":
        return cls(
            state=OrderedDict(),
            size_limit=size_limit,
            send_to_cpu=send_to_cpu,
        )

    def __init__(
        self,
        state: OrderedDict,
        size_limit: Optional[int],
        send_to_cpu: bool = True,
    ):
        self._state = state
        self._size_limit = size_limit
        self._send_to_cpu = send_to_cpu
        self._state_lock = Lock()

    def retrieve_embeddings(self, key: str) -> Optional[SAM2ImageEmbeddings]:
        return self._state.get(key)

    def save_embeddings(self, key: str, embeddings: SAM2ImageEmbeddings) -> None:
        with self._state_lock:
            if key in self._state:
                return None
            self._ensure_cache_has_capacity()
            if self._send_to_cpu:
                embeddings = embeddings.to(device=torch.device("cpu"))
            self._state[key] = embeddings

    def _ensure_cache_has_capacity(self) -> None:
        if self._size_limit < 1:
            raise EnvironmentConfigurationError(
                message=f"In memory cache size for SAM2 embeddings was set to invalid value. "
                f"If you are running inference locally - adjust settings of your deployment. If you see this "
                f"error running on Roboflow platform - contact us to get help.",
                help_url="https://todo",
            )
        if self._size_limit is None or self._size_limit < 1:
            return None
        while len(self._state) > self._size_limit:
            _ = self._state.popitem(last=False)


class Sam2LowResolutionMasksCache(ABC):

    @abstractmethod
    def retrieve_all_masks_for_image(self, key: str) -> List[torch.Tensor]:
        pass

    @abstractmethod
    def save_mask(self, key: str, mask: SAM2MaskCacheEntry) -> None:
        pass


class Sam2LowResolutionMasksCacheNullObject(Sam2LowResolutionMasksCache):

    def retrieve_all_masks_for_image(self, key: str) -> List[torch.Tensor]:
        pass

    def save_mask(self, key: str, mask: SAM2MaskCacheEntry) -> None:
        pass


class Sam2LowResolutionMasksInMemoryCache(Sam2LowResolutionMasksCache):

    @classmethod
    def init(
        cls, size_limit: Optional[int], send_to_cpu: bool = True
    ) -> "Sam2LowResolutionMasksInMemoryCache":
        return cls(
            state=OrderedDict(),
            size_limit=size_limit,
            send_to_cpu=send_to_cpu,
        )

    def __init__(
        self,
        state: OrderedDict,
        
        size_limit: Optional[int],
        send_to_cpu: bool = True,
    ):
        self._state = state
        self._size_limit = size_limit
        self._send_to_cpu = send_to_cpu
        self._state_lock = Lock()

    def retrieve_all_masks_for_image(self, key: str) -> List[torch.Tensor]:
        return self._state.get(key, [])

    def save_mask(self, key: str, mask: torch.Tensor) -> None:
        with self._state_lock:
            if key in self._state:
                return None
            self._ensure_cache_has_capacity()
            if self._send_to_cpu:
                mask = mask.to(device=torch.device("cpu"))
            self._state[key] = mask

    def _ensure_cache_has_capacity(self) -> None:
        if self._size_limit < 1:
            raise EnvironmentConfigurationError(
                message=f"In memory cache size for SAM embeddings was set to invalid value. "
                f"If you are running inference locally - adjust settings of your deployment. If you see this "
                f"error running on Roboflow platform - contact us to get help.",
                help_url="https://todo",
            )
        if self._size_limit is None or self._size_limit < 1:
            return None
        while len(self._state) > self._size_limit:
            _ = self._state.popitem(last=False)
