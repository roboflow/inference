from abc import ABC, abstractmethod
from collections import OrderedDict
from threading import Lock
from typing import Optional

import torch

from inference_models.errors import EnvironmentConfigurationError
from inference_models.models.sam.entities import SAMImageEmbeddings


class SamImageEmbeddingsCache(ABC):

    @abstractmethod
    def retrieve_embeddings(self, key: str) -> Optional[SAMImageEmbeddings]:
        pass

    @abstractmethod
    def save_embeddings(self, key: str, embeddings: SAMImageEmbeddings) -> None:
        pass


class SamImageEmbeddingsCacheNullObject(SamImageEmbeddingsCache):

    def retrieve_embeddings(self, key: str) -> Optional[SAMImageEmbeddings]:
        pass

    def save_embeddings(self, key: str, embeddings: SAMImageEmbeddings) -> None:
        pass


class SamImageEmbeddingsInMemoryCache(SamImageEmbeddingsCache):

    @classmethod
    def init(
        cls, size_limit: Optional[int], send_to_cpu: bool = True
    ) -> "SamImageEmbeddingsInMemoryCache":
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

    def retrieve_embeddings(self, key: str) -> Optional[SAMImageEmbeddings]:
        return self._state.get(key)

    def save_embeddings(self, key: str, embeddings: SAMImageEmbeddings) -> None:
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
                message=f"In memory cache size for SAM embeddings was set to invalid value. "
                f"If you are running inference locally - adjust settings of your deployment. If you see this "
                f"error running on Roboflow platform - contact us to get help.",
                help_url="https://inference-models.roboflow.com/errors/runtime-environment/#environmentconfigurationerror",
            )
        if self._size_limit is None or self._size_limit < 1:
            return None
        while len(self._state) > self._size_limit:
            _ = self._state.popitem(last=False)


class SamLowResolutionMasksCache(ABC):

    @abstractmethod
    def retrieve_mask(self, key: str) -> Optional[torch.Tensor]:
        pass

    @abstractmethod
    def save_mask(self, key: str, mask: torch.Tensor) -> None:
        pass


class SamLowResolutionMasksCacheNullObject(SamLowResolutionMasksCache):

    def retrieve_mask(self, key: str) -> Optional[torch.Tensor]:
        pass

    def save_mask(self, key: str, mask: torch.Tensor) -> None:
        pass


class SamLowResolutionMasksInMemoryCache(SamLowResolutionMasksCache):

    @classmethod
    def init(
        cls, size_limit: Optional[int], send_to_cpu: bool = True
    ) -> "SamLowResolutionMasksInMemoryCache":
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

    def retrieve_mask(self, key: str) -> Optional[torch.Tensor]:
        return self._state.get(key)

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
                help_url="https://inference-models.roboflow.com/errors/runtime-environment/#environmentconfigurationerror",
            )
        if self._size_limit is None or self._size_limit < 1:
            return None
        while len(self._state) > self._size_limit:
            _ = self._state.popitem(last=False)
