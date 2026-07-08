from inference_model_manager.backends.base import attach_sam3_caches
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


class OtherModel:
    pass


def test_attach_replaces_null_caches_on_sam3():
    model = SAM3Torch()

    attach_sam3_caches(model)

    assert isinstance(
        model._sam3_image_embeddings_cache, Sam3ImageEmbeddingsInMemoryCache
    )
    assert isinstance(
        model._sam3_low_resolution_masks_cache, Sam3LowResolutionMasksInMemoryCache
    )


def test_attach_ignores_non_sam3_models():
    model = OtherModel()

    attach_sam3_caches(model)

    assert not hasattr(model, "_sam3_image_embeddings_cache")
    assert not hasattr(model, "_sam3_low_resolution_masks_cache")
