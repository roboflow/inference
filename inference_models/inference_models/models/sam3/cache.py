from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from threading import Lock
from typing import DefaultDict, List, Optional

import torch

from inference_models.errors import EnvironmentConfigurationError
from inference_models.models.sam3.entities import (
    SAM3ImageEmbeddings,
    SAM3MaskCacheEntry,
)


class Sam3ImageEmbeddingsCache(ABC):

    @abstractmethod
    def retrieve_embeddings(self, key: str) -> Optional[SAM3ImageEmbeddings]:
        pass

    @abstractmethod
    def save_embeddings(self, key: str, embeddings: SAM3ImageEmbeddings) -> None:
        pass


class Sam3ImageEmbeddingsCacheNullObject(Sam3ImageEmbeddingsCache):

    def retrieve_embeddings(self, key: str) -> Optional[SAM3ImageEmbeddings]:
        pass

    def save_embeddings(self, key: str, embeddings: SAM3ImageEmbeddings) -> None:
        pass


class Sam3ImageEmbeddingsInMemoryCache(Sam3ImageEmbeddingsCache):

    @classmethod
    def init(
        cls, size_limit: Optional[int], send_to_cpu: bool = True
    ) -> "Sam3ImageEmbeddingsInMemoryCache":
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

    def retrieve_embeddings(self, key: str) -> Optional[SAM3ImageEmbeddings]:
        return self._state.get(key)

    def save_embeddings(self, key: str, embeddings: SAM3ImageEmbeddings) -> None:
        with self._state_lock:
            if key in self._state:
                return None
            self._ensure_cache_has_capacity()
            if self._send_to_cpu:
                embeddings = embeddings.to(device=torch.device("cpu"))
            self._state[key] = embeddings

    def _ensure_cache_has_capacity(self):
        if self._size_limit is None:
            return
        if self._size_limit < 1:
            raise EnvironmentConfigurationError(
                message=f"In memory cache size for SAM3 embeddings was set to invalid value. "
                f"If you are running inference locally - adjust settings of your deployment. If you see this "
                f"error running on Roboflow platform - contact us to get help.",
                help_url="https://todo",
            )
        while len(self._state) > self._size_limit:
            _ = self._state.popitem(last=False)


class Sam3LowResolutionMasksCache(ABC):

    @abstractmethod
    def retrieve_all_masks_for_image(self, key: str) -> List[SAM3MaskCacheEntry]:
        pass

    @abstractmethod
    def save_mask(self, key: str, mask: SAM3MaskCacheEntry) -> None:
        pass


class Sam3LowResolutionMasksCacheNullObject(Sam3LowResolutionMasksCache):

    def retrieve_all_masks_for_image(self, key: str) -> List[SAM3MaskCacheEntry]:
        return []

    def save_mask(self, key: str, mask: SAM3MaskCacheEntry) -> None:
        pass


class Sam3LowResolutionMasksInMemoryCache(Sam3LowResolutionMasksCache):

    @classmethod
    def init(
        cls, size_limit: Optional[int], send_to_cpu: bool = True
    ) -> "Sam3LowResolutionMasksInMemoryCache":
        return cls(
            ordering_state=OrderedDict(),
            cache_state=defaultdict(list),
            size_limit=size_limit,
            send_to_cpu=send_to_cpu,
        )

    def __init__(
        self,
        ordering_state: OrderedDict,
        cache_state: DefaultDict[str, List[SAM3MaskCacheEntry]],
        size_limit: Optional[int],
        send_to_cpu: bool = True,
    ):
        self._ordering_state = ordering_state
        self._cache_state = cache_state
        self._size_limit = size_limit
        self._send_to_cpu = send_to_cpu
        self._state_lock = Lock()

    def retrieve_all_masks_for_image(self, key: str) -> List[SAM3MaskCacheEntry]:
        return self._cache_state.get(key, [])

    def save_mask(self, key: str, mask: SAM3MaskCacheEntry) -> None:
        with self._state_lock:
            if (key, mask.prompt_hash) in self._ordering_state:
                return None
            self._ensure_cache_has_capacity()
            if self._send_to_cpu:
                mask = mask.to(device=torch.device("cpu"))
            self._ordering_state[(key, mask.prompt_hash)] = True
            self._cache_state[key].append(mask)

    def _ensure_cache_has_capacity(self):
        if self._size_limit is None:
            return
        if self._size_limit < 1:
            raise EnvironmentConfigurationError(
                message=f"In memory cache size for SAM3 low resolution masks was set to invalid value. "
                f"If you are running inference locally - adjust settings of your deployment. If you see this "
                f"error running on Roboflow platform - contact us to get help.",
                help_url="https://todo",
            )
        while len(self._ordering_state) > self._size_limit:
            image_key, prompt_hash = self._ordering_state.popitem(last=False)
            entries_for_image = self._cache_state[image_key]
            to_remove_idx = None
            for i, element in enumerate(entries_for_image):
                if element.prompt_hash == prompt_hash:
                    to_remove_idx = i
                    break
            if to_remove_idx is not None:
                del entries_for_image[to_remove_idx]
