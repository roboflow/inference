from unittest.mock import MagicMock

from inference.models.sam.segment_anything_inference_models import (
    InferenceModelsSAMAdapter,
)


def test_tensor_native_embed_returns_embeddings_without_segmentation() -> None:
    adapter = object.__new__(InferenceModelsSAMAdapter)
    adapter._model = MagicMock()
    embeddings = [object()]
    adapter._model.embed_images.return_value = embeddings

    result = adapter.run_tensor_native_inference(action="embed", images=["image"])

    assert result is embeddings
    adapter._model.embed_images.assert_called_once_with(images=["image"])
    adapter._model.segment_images.assert_not_called()
