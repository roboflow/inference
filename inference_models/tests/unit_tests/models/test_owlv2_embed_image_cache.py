import numpy as np
import pytest
import torch

from inference_models.models.owlv2.cache import InMemoryOwlV2ImageEmbeddingsCache
from inference_models.models.owlv2.entities import ImageEmbeddings
from inference_models.models.owlv2.owlv2_hf import OWLv2HF
from inference_models.models.owlv2.reference_dataset import (
    LazyImageWrapper,
    compute_image_hash,
)

IMAGE_URL = "https://storage.googleapis.com/bucket/workspace/image-1/original.jpg"


def build_model_shell(cache: InMemoryOwlV2ImageEmbeddingsCache) -> OWLv2HF:
    """OWLv2HF with no weights loaded — cache-hit paths must not need them."""
    return OWLv2HF(
        model=None,
        processor=None,
        device=torch.device("cpu"),
        owlv2_class_embeddings_cache=None,
        owlv2_images_embeddings_cache=cache,
        allow_url_input=True,
        allow_non_https_url=False,
        allow_url_without_fqdn=False,
        whitelisted_domains=None,
        blacklisted_domains=None,
        allow_local_storage_access_for_reference_images=False,
    )


def build_embeddings(image_hash: str) -> ImageEmbeddings:
    return ImageEmbeddings(
        image_hash=image_hash,
        objectness=torch.zeros(3),
        boxes=torch.zeros(3, 4),
        image_class_embeddings=torch.zeros(3, 8),
        logit_shift=torch.zeros(3),
        logit_scale=torch.ones(3),
        image_size_wh=(640, 480),
    )


def wrap_url_reference(url: str) -> LazyImageWrapper:
    return LazyImageWrapper.init(
        image=url,
        allow_url_input=True,
        allow_non_https_url=False,
        allow_url_without_fqdn=False,
        whitelisted_domains=None,
        blacklisted_domains=None,
        allow_local_storage_access=False,
    )


def test_embed_image_cache_hit_does_not_materialize_url_reference(monkeypatch) -> None:
    # given - embeddings for the URL's hash are already cached; downloading
    # the image on this path is the regression this test guards against
    cache = InMemoryOwlV2ImageEmbeddingsCache.init(size_limit=10, send_to_cpu=True)
    wrapper = wrap_url_reference(IMAGE_URL)
    cached = build_embeddings(image_hash=wrapper.get_hash())
    cache.save_embeddings(embeddings=cached)
    model = build_model_shell(cache=cache)

    def fail_on_download() -> np.ndarray:
        raise AssertionError("image was downloaded despite embeddings-cache hit")

    monkeypatch.setattr(wrapper, "as_numpy", fail_on_download)

    # when
    result = model.embed_image(image=wrapper)

    # then
    assert result.image_hash == cached.image_hash


def test_embed_image_cache_hit_for_in_memory_image(monkeypatch) -> None:
    # given
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image_hash = compute_image_hash(image=image)
    cache = InMemoryOwlV2ImageEmbeddingsCache.init(size_limit=10, send_to_cpu=True)
    cache.save_embeddings(embeddings=build_embeddings(image_hash=image_hash))
    model = build_model_shell(cache=cache)

    # when
    result = model.embed_image(image=image)

    # then
    assert result.image_hash == image_hash


def test_embed_image_cache_miss_materializes_the_reference(monkeypatch) -> None:
    # given - empty cache: the miss path MUST load the image (and then fail
    # further down in this weightless shell, proving as_numpy was reached)
    cache = InMemoryOwlV2ImageEmbeddingsCache.init(size_limit=10, send_to_cpu=True)
    wrapper = wrap_url_reference(IMAGE_URL)
    model = build_model_shell(cache=cache)
    materialized = {"called": False}

    def record_materialization() -> np.ndarray:
        materialized["called"] = True
        raise RuntimeError("stop after materialization")

    monkeypatch.setattr(wrapper, "as_numpy", record_materialization)

    # when
    with pytest.raises(RuntimeError, match="stop after materialization"):
        _ = model.embed_image(image=wrapper)

    # then
    assert materialized["called"] is True
