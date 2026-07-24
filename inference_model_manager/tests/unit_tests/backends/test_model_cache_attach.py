from inference_model_manager.backends.base import attach_model_caches
from inference_models.models.owlv2.cache import (
    InMemoryOwlV2ClassEmbeddingsCache,
    InMemoryOwlV2ImageEmbeddingsCache,
    OwlV2ClassEmbeddingsCacheNullObject,
    OwlV2ImageEmbeddingsCacheNullObject,
)
from inference_models.models.sam3.cache import (
    Sam3ImageEmbeddingsCacheNullObject,
    Sam3ImageEmbeddingsInMemoryCache,
    Sam3LowResolutionMasksCacheNullObject,
    Sam3LowResolutionMasksInMemoryCache,
)


class SAM3Torch:
    def __init__(self):
        self._sam3_image_embeddings_cache = Sam3ImageEmbeddingsCacheNullObject()
        self._sam3_low_resolution_masks_cache = Sam3LowResolutionMasksCacheNullObject()


class FakeOwlV2:
    def __init__(self):
        self._owlv2_class_embeddings_cache = OwlV2ClassEmbeddingsCacheNullObject()
        self._owlv2_images_embeddings_cache = OwlV2ImageEmbeddingsCacheNullObject()


class FakeInstantHead:
    def __init__(self, feature_extractor):
        self._feature_extractor = feature_extractor


class OtherModel:
    pass


def test_attach_replaces_null_caches_on_sam3():
    model = SAM3Torch()

    attach_model_caches(model)

    assert isinstance(
        model._sam3_image_embeddings_cache, Sam3ImageEmbeddingsInMemoryCache
    )
    assert isinstance(
        model._sam3_low_resolution_masks_cache, Sam3LowResolutionMasksInMemoryCache
    )


def test_attach_replaces_null_caches_on_owlv2():
    model = FakeOwlV2()

    attach_model_caches(model)

    assert isinstance(
        model._owlv2_class_embeddings_cache, InMemoryOwlV2ClassEmbeddingsCache
    )
    assert isinstance(
        model._owlv2_images_embeddings_cache, InMemoryOwlV2ImageEmbeddingsCache
    )


def test_attach_reaches_owlv2_feature_extractor_of_head_model():
    base = FakeOwlV2()
    head = FakeInstantHead(feature_extractor=base)

    attach_model_caches(head)

    assert isinstance(
        base._owlv2_class_embeddings_cache, InMemoryOwlV2ClassEmbeddingsCache
    )
    assert isinstance(
        base._owlv2_images_embeddings_cache, InMemoryOwlV2ImageEmbeddingsCache
    )


def test_attach_keeps_existing_in_memory_caches():
    base = FakeOwlV2()
    attach_model_caches(base)
    warm_class_cache = base._owlv2_class_embeddings_cache
    warm_image_cache = base._owlv2_images_embeddings_cache

    attach_model_caches(FakeInstantHead(feature_extractor=base))

    assert base._owlv2_class_embeddings_cache is warm_class_cache
    assert base._owlv2_images_embeddings_cache is warm_image_cache


def test_attach_ignores_other_models():
    model = OtherModel()

    attach_model_caches(model)

    assert not hasattr(model, "_sam3_image_embeddings_cache")
    assert not hasattr(model, "_owlv2_class_embeddings_cache")
