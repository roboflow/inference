from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from inference.models.owlv2 import owlv2_inference_models


@pytest.mark.parametrize("send_to_cpu", [True, False])
def test_adapter_singleton_honors_cache_device_setting(send_to_cpu: bool) -> None:
    owlv2_inference_models.Owlv2AdapterSingleton._instances.clear()
    owlv2_inference_models.PRELOADED_HF_MODELS.clear()

    with patch.object(
        owlv2_inference_models.InMemoryOwlV2ClassEmbeddingsCache, "init"
    ) as class_cache_init, patch.object(
        owlv2_inference_models.InMemoryOwlV2ImageEmbeddingsCache, "init"
    ) as image_cache_init, patch.object(
        owlv2_inference_models.AutoModel,
        "from_pretrained",
        return_value=MagicMock(),
    ), patch.object(
        owlv2_inference_models, "get_extra_weights_provider_headers", return_value=None
    ), patch.object(
        owlv2_inference_models, "OWLV2_CACHE_SEND_TO_CPU", send_to_cpu
    ), patch.object(
        owlv2_inference_models, "OWLV2_COMPILE_MODEL", False
    ):
        owlv2_inference_models.Owlv2AdapterSingleton(
            f"owlv2/test-cache-device-{send_to_cpu}", api_key="test-key"
        )

    assert class_cache_init.call_args.kwargs["send_to_cpu"] is send_to_cpu
    assert image_cache_init.call_args.kwargs["send_to_cpu"] is send_to_cpu


def test_infer_uses_existing_path_when_diagnostics_are_disabled() -> None:
    adapter = owlv2_inference_models.InferenceModelsOwlV2Adapter.__new__(
        owlv2_inference_models.InferenceModelsOwlV2Adapter
    )
    adapter.model_id = "owlv2/test"
    adapter._model = MagicMock()
    adapter._model.infer_with_reference_examples.return_value = []

    with patch.object(
        owlv2_inference_models, "OWLV2_DIAGNOSTIC_LOGGING", False
    ), patch.object(
        owlv2_inference_models,
        "load_image_bgr",
        return_value=np.zeros((10, 20, 3), dtype=np.uint8),
    ):
        result = adapter.infer(
            image="target-image",
            training_data=[
                {
                    "image": {"value": "reference-image"},
                    "boxes": [{"x": 1, "y": 2, "w": 3, "h": 4, "cls": "widget"}],
                }
            ],
        )

    assert result == []
    adapter._model.infer_with_reference_examples.assert_called_once()
    adapter._model.prepare_reference_examples_embeddings.assert_not_called()
    adapter._model.embed_images.assert_not_called()
    adapter._model.forward_pass_with_precomputed_embeddings.assert_not_called()
    adapter._model.post_process_predictions_for_precomputed_embeddings.assert_not_called()


def test_infer_uses_phase_path_when_diagnostics_are_enabled() -> None:
    adapter = owlv2_inference_models.InferenceModelsOwlV2Adapter.__new__(
        owlv2_inference_models.InferenceModelsOwlV2Adapter
    )
    adapter.model_id = "owlv2/test"
    adapter._model = MagicMock()
    adapter._model._owlv2_images_embeddings_cache = None
    adapter._model._owlv2_class_embeddings_cache = None
    adapter._model.prepare_reference_examples_embeddings.return_value = SimpleNamespace(
        class_embeddings={"widget": MagicMock()}
    )
    adapter._model.embed_images.return_value = (["image-embedding"], [])
    adapter._model.forward_pass_with_precomputed_embeddings.return_value = [
        "predictions"
    ]
    adapter._model.post_process_predictions_for_precomputed_embeddings.return_value = []

    with patch.object(
        owlv2_inference_models, "OWLV2_DIAGNOSTIC_LOGGING", True
    ), patch.object(
        owlv2_inference_models,
        "OWLV2_DIAGNOSTIC_SAMPLE_RATE",
        1.0,
    ), patch.object(
        owlv2_inference_models,
        "load_image_bgr",
        return_value=np.zeros((10, 20, 3), dtype=np.uint8),
    ), patch.object(
        owlv2_inference_models,
        "logger",
    ) as logger:
        result = adapter.infer(
            image="target-image",
            training_data=[
                {
                    "image": {"value": "reference-image"},
                    "boxes": [{"x": 1, "y": 2, "w": 3, "h": 4, "cls": "widget"}],
                }
            ],
        )

    assert result == []
    adapter._model.infer_with_reference_examples.assert_not_called()
    adapter._model.prepare_reference_examples_embeddings.assert_called_once()
    adapter._model.embed_images.assert_called_once()
    adapter._model.forward_pass_with_precomputed_embeddings.assert_called_once()
    adapter._model.post_process_predictions_for_precomputed_embeddings.assert_called_once()
    assert logger.info.called
