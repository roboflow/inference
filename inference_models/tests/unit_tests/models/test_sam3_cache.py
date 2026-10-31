import pytest
import torch

from inference_models.errors import EnvironmentConfigurationError
from inference_models.models.sam3.cache import (
    Sam3ImageEmbeddingsInMemoryCache,
    Sam3LowResolutionMasksInMemoryCache,
)
from inference_models.models.sam3.entities import (
    SAM3ImageEmbeddings,
    SAM3MaskCacheEntry,
)


def _make_embeddings(key: str) -> SAM3ImageEmbeddings:
    return SAM3ImageEmbeddings(
        image_hash=key,
        image_size_hw=(10, 10),
        embeddings={"feat": torch.zeros(1)},
    )


def _make_mask_entry(prompt_hash: str) -> SAM3MaskCacheEntry:
    return SAM3MaskCacheEntry(
        prompt_hash=prompt_hash,
        serialized_prompt=[{"text": prompt_hash}],
        mask=torch.zeros(1, 1, 1),
    )


def test_embeddings_cache_size_limit_not_exceeded_after_insert() -> None:
    cache = Sam3ImageEmbeddingsInMemoryCache.init(size_limit=3, send_to_cpu=False)
    for i in range(10):
        cache.save_embeddings(key=str(i), embeddings=_make_embeddings(str(i)))
        assert len(cache._state) <= 3, f"cache exceeded limit at step {i}"


def test_embeddings_cache_evicts_oldest_when_full() -> None:
    cache = Sam3ImageEmbeddingsInMemoryCache.init(size_limit=2, send_to_cpu=False)
    cache.save_embeddings(key="a", embeddings=_make_embeddings("a"))
    cache.save_embeddings(key="b", embeddings=_make_embeddings("b"))
    cache.save_embeddings(key="c", embeddings=_make_embeddings("c"))
    assert cache.retrieve_embeddings("a") is None
    assert cache.retrieve_embeddings("b") is not None
    assert cache.retrieve_embeddings("c") is not None


def test_embeddings_cache_unlimited_when_size_limit_none() -> None:
    cache = Sam3ImageEmbeddingsInMemoryCache.init(size_limit=None, send_to_cpu=False)
    for i in range(50):
        cache.save_embeddings(key=str(i), embeddings=_make_embeddings(str(i)))
    assert len(cache._state) == 50


def test_embeddings_cache_invalid_size_limit_raises() -> None:
    cache = Sam3ImageEmbeddingsInMemoryCache.init(size_limit=0, send_to_cpu=False)
    with pytest.raises(EnvironmentConfigurationError):
        cache.save_embeddings(key="a", embeddings=_make_embeddings("a"))


def test_masks_cache_size_limit_not_exceeded_after_insert() -> None:
    cache = Sam3LowResolutionMasksInMemoryCache.init(size_limit=3, send_to_cpu=False)
    for i in range(10):
        cache.save_mask(key=f"img_{i}", mask=_make_mask_entry(f"p_{i}"))
        assert len(cache._ordering_state) <= 3, f"cache exceeded limit at step {i}"


def test_masks_cache_evicts_oldest_when_full() -> None:
    cache = Sam3LowResolutionMasksInMemoryCache.init(size_limit=2, send_to_cpu=False)
    cache.save_mask(key="img_a", mask=_make_mask_entry("p_a"))
    cache.save_mask(key="img_b", mask=_make_mask_entry("p_b"))
    cache.save_mask(key="img_c", mask=_make_mask_entry("p_c"))
    assert cache.retrieve_all_masks_for_image("img_a") == []
    assert len(cache.retrieve_all_masks_for_image("img_b")) == 1
    assert len(cache.retrieve_all_masks_for_image("img_c")) == 1


def test_masks_cache_unlimited_when_size_limit_none() -> None:
    cache = Sam3LowResolutionMasksInMemoryCache.init(size_limit=None, send_to_cpu=False)
    for i in range(50):
        cache.save_mask(key=f"img_{i}", mask=_make_mask_entry(f"p_{i}"))
    assert len(cache._ordering_state) == 50


def test_masks_cache_invalid_size_limit_raises() -> None:
    cache = Sam3LowResolutionMasksInMemoryCache.init(size_limit=0, send_to_cpu=False)
    with pytest.raises(EnvironmentConfigurationError):
        cache.save_mask(key="img_a", mask=_make_mask_entry("p_a"))
