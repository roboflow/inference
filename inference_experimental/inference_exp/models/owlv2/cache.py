import hashlib
import json
from abc import ABC, abstractmethod
from threading import Lock
from typing import Dict, List, Optional, OrderedDict

import torch
from inference_exp.errors import EnvironmentConfigurationError
from inference_exp.models.owlv2.entities import (
    ImageEmbeddings,
    LazyReferenceExample,
    ReferenceExamplesClassEmbeddings,
)


class OwlV2ClassEmbeddingsCache(ABC):

    @abstractmethod
    def retrieve_embeddings(
        self, key: str
    ) -> Optional[Dict[str, ReferenceExamplesClassEmbeddings]]:
        pass

    @abstractmethod
    def save_embeddings(
        self, key: str, embeddings: Dict[str, ReferenceExamplesClassEmbeddings]
    ) -> None:
        pass


class OwlV2ClassEmbeddingsCacheNullObject(OwlV2ClassEmbeddingsCache):

    def retrieve_embeddings(
        self, key: str
    ) -> Optional[Dict[str, ReferenceExamplesClassEmbeddings]]:
        return None

    def save_embeddings(
        self, key: str, embeddings: Dict[str, ReferenceExamplesClassEmbeddings]
    ) -> None:
        pass


class InMemoryOwlV2ClassEmbeddingsCache(OwlV2ClassEmbeddingsCache):

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

    def retrieve_embeddings(
        self, key: str
    ) -> Optional[Dict[str, ReferenceExamplesClassEmbeddings]]:
        return self._state.get(key)

    def save_embeddings(
        self, key: str, embeddings: Dict[str, ReferenceExamplesClassEmbeddings]
    ) -> None:
        with self._state_lock:
            if key in self._state:
                return None
            self._ensure_cache_has_capacity()
            if self._send_to_cpu:
                embeddings = {
                    k: v.to(device=torch.device("cpu")) for k, v in embeddings.items()
                }
            self._state[key] = embeddings

    def _ensure_cache_has_capacity(self) -> None:
        if self._size_limit < 1:
            raise EnvironmentConfigurationError(
                message=f"In memory cache size for OWLv2 embeddings was set to invalid value. "
                f"If you are running inference locally - adjust settings of your deployment. If you see this "
                f"error running on Roboflow platform - contact us to get help.",
                help_url="https://todo",
            )
        if self._size_limit is None or self._size_limit < 1:
            return None
        while len(self._state) > self._size_limit:
            _ = self._state.popitem(last=False)


class OwlV2ImageEmbeddingsCache(ABC):

    @abstractmethod
    def retrieve_embeddings(self, key: str) -> Optional[ImageEmbeddings]:
        pass

    @abstractmethod
    def save_embeddings(self, embeddings: ImageEmbeddings) -> None:
        pass


class OwlV2ImageEmbeddingsCacheNullObject(OwlV2ImageEmbeddingsCache):

    def retrieve_embeddings(self, key: str) -> Optional[ImageEmbeddings]:
        return None

    def save_embeddings(self, embeddings: ImageEmbeddings) -> None:
        pass


class InMemoryOwlV2ImageEmbeddingsCache(OwlV2ImageEmbeddingsCache):

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

    def retrieve_embeddings(self, key: str) -> Optional[ImageEmbeddings]:
        return self._state.get(key)

    def save_embeddings(self, embeddings: ImageEmbeddings) -> None:
        with self._state_lock:
            if embeddings.image_hash in self._state:
                return None
            self._ensure_cache_has_capacity()
            if self._send_to_cpu:
                embeddings = embeddings.to(device=torch.device("cpu"))
            self._state[embeddings.image_hash] = embeddings

    def _ensure_cache_has_capacity(self) -> None:
        if self._size_limit < 1:
            raise EnvironmentConfigurationError(
                message=f"In memory cache size for OWLv2 embeddings was set to invalid value. "
                f"If you are running inference locally - adjust settings of your deployment. If you see this "
                f"error running on Roboflow platform - contact us to get help.",
                help_url="https://todo",
            )
        if self._size_limit is None or self._size_limit < 1:
            return None
        while len(self._state) > self._size_limit:
            _ = self._state.popitem(last=False)


def hash_reference_examples(reference_examples: List[LazyReferenceExample]) -> str:
    result = hashlib.sha1()
    for example in reference_examples:
        image_hash = example.image.get_hash()
        result.update(image_hash.encode())
        bboxes_hash_base = "---".join(
            [
                json.dumps(box.model_dump(), sort_keys=True, separators=(",", ":"))
                for box in example.boxes
            ]
        )
        result.update(bboxes_hash_base.encode())
    return result.hexdigest()
