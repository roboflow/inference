import hashlib
import json
from abc import ABC, abstractmethod
from typing import Dict, Optional, List

from inference_exp.models.owlv2.entities import ReferenceExamplesClassEmbeddings, ImageEmbeddings, \
    LazyReferenceExample


class OwlV2ClassEmbeddingsCache(ABC):

    @abstractmethod
    def retrieve_embeddings(self, key: str) -> Optional[Dict[str, ReferenceExamplesClassEmbeddings]]:
        pass

    @abstractmethod
    def save_embeddings(self, key: str, embeddings: Dict[str, ReferenceExamplesClassEmbeddings]) -> None:
        pass


class OwlV2ClassEmbeddingsCacheNullObject(OwlV2ClassEmbeddingsCache):

    def retrieve_embeddings(self, key: str) -> Optional[Dict[str, ReferenceExamplesClassEmbeddings]]:
        return None

    def save_embeddings(self, key: str, embeddings: Dict[str, ReferenceExamplesClassEmbeddings]) -> None:
        pass


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


def hash_reference_examples(reference_examples: List[LazyReferenceExample]) -> str:
    result = hashlib.sha1()
    for example in reference_examples:
        image_hash = example.image.get_hash()
        result.update(image_hash.encode())
        bboxes_hash_base = "---".join([
            json.dumps(box.model_dump(), sort_keys=True, separators=(",", ":")) for box in example.boxes
        ])
        result.update(bboxes_hash_base.encode())
    return result.hexdigest()


