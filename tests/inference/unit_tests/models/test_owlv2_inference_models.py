from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from inference.core.managers.base import ModelManager
from inference.models.owlv2 import owlv2_inference_models, rf_instant_inference_models
from inference_models import Detections


@pytest.mark.parametrize("saved_labels", [["leaf", "flower"], None])
def test_model_manager_reads_saved_owlv2_labels(saved_labels):
    native = owlv2_inference_models.OWLv2HF.__new__(owlv2_inference_models.OWLv2HF)
    native._class_names = saved_labels
    with patch.object(
        owlv2_inference_models,
        "Owlv2AdapterSingleton",
        return_value=MagicMock(model=native),
    ):
        adapter = owlv2_inference_models.InferenceModelsOwlV2Adapter(
            model_id="uploaded/1", api_key="test-key"
        )
    manager = ModelManager(
        model_registry=MagicMock(),
        models={"uploaded/1": adapter},
        content_addressed_artifact_cache=MagicMock(),
    )
    assert manager.get_class_names("uploaded/1") == (saved_labels or [])
    assert not hasattr(native, "class_names")


@pytest.mark.parametrize("training_data", [None, []])
def test_saved_prompts_do_not_replace_reference_example_requests(training_data):
    adapter = owlv2_inference_models.InferenceModelsOwlV2Adapter.__new__(
        owlv2_inference_models.InferenceModelsOwlV2Adapter
    )
    adapter.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}
    adapter._model = MagicMock()
    labels = ["saved label"] if training_data is None else ["reference label"]
    detections = [
        Detections(
            xyxy=torch.tensor([[1, 2, 7, 8]]),
            class_id=torch.tensor([0]),
            confidence=torch.tensor([0.8]),
            image_metadata={"class_names": labels},
        )
    ]
    adapter._model.infer.return_value = detections
    adapter._model.infer_with_reference_examples.return_value = detections
    response = adapter.infer(
        np.zeros((12, 16, 3), dtype=np.uint8),
        training_data=training_data,
        confidence=0.2,
        max_detections=1,
        class_agnostic_nms=True,
    )[0]
    assert response.predictions[0].class_name == labels[0]
    assert (response.image.height, response.image.width) == (12, 16)
    if training_data is None:
        adapter._model.infer_with_reference_examples.assert_not_called()
        assert adapter._model.infer.call_args.kwargs["input_color_format"] == "bgr"
        assert adapter._model.infer.call_args.kwargs["class_agnostic_nms"] is True
    else:
        adapter._model.infer.assert_not_called()
        assert (
            adapter._model.infer_with_reference_examples.call_args.kwargs[
                "reference_examples"
            ]
            == []
        )


@pytest.mark.parametrize("send_to_cpu", [True, False])
def test_adapter_singleton_honors_cache_device_setting(send_to_cpu: bool) -> None:
    owlv2_inference_models.Owlv2AdapterSingleton._instances.clear()
    owlv2_inference_models.PRELOADED_HF_MODELS.clear()
    artifact_cache = MagicMock()

    with patch.object(
        owlv2_inference_models.InMemoryOwlV2ClassEmbeddingsCache, "init"
    ) as class_cache_init, patch.object(
        owlv2_inference_models.InMemoryOwlV2ImageEmbeddingsCache, "init"
    ) as image_cache_init, patch.object(
        owlv2_inference_models.AutoModel,
        "from_pretrained",
        return_value=MagicMock(),
    ) as auto_model, patch.object(
        owlv2_inference_models, "get_extra_weights_provider_headers", return_value=None
    ), patch.object(
        owlv2_inference_models, "OWLV2_CACHE_SEND_TO_CPU", send_to_cpu
    ), patch.object(
        owlv2_inference_models, "OWLV2_COMPILE_MODEL", False
    ):
        owlv2_inference_models.Owlv2AdapterSingleton(
            f"owlv2/test-cache-device-{send_to_cpu}",
            api_key="test-key",
            content_addressed_artifact_cache=artifact_cache,
        )

    assert class_cache_init.call_args.kwargs["send_to_cpu"] is send_to_cpu
    assert image_cache_init.call_args.kwargs["send_to_cpu"] is send_to_cpu
    assert (
        auto_model.call_args.kwargs["content_addressed_artifact_cache"]
        is artifact_cache
    )


def test_rf_instant_dependency_forwards_artifact_cache() -> None:
    artifact_cache = MagicMock()
    dependency_model = MagicMock()
    singleton = MagicMock(model=dependency_model)
    access_manager = (
        rf_instant_inference_models.RFInstantSpecificLiberalModelAccessManager(
            content_addressed_artifact_cache=artifact_cache
        )
    )

    with patch.object(
        rf_instant_inference_models,
        "Owlv2AdapterSingleton",
        return_value=singleton,
    ) as singleton_factory:
        result = access_manager.retrieve_model_instance(
            model_id=(f"owlv2/{rf_instant_inference_models.OWLV2_VERSION_ID}"),
            package_id=None,
            api_key="test-key",
            loading_parameter_digest=None,
        )

    assert result is dependency_model
    assert (
        singleton_factory.call_args.kwargs["content_addressed_artifact_cache"]
        is artifact_cache
    )


def test_adapter_singleton_defaults_to_shared_blob_cache() -> None:
    owlv2_inference_models.Owlv2AdapterSingleton._instances.clear()
    owlv2_inference_models.PRELOADED_HF_MODELS.clear()
    shared_cache = MagicMock()

    with patch.object(
        owlv2_inference_models.InMemoryOwlV2ClassEmbeddingsCache, "init"
    ), patch.object(
        owlv2_inference_models.InMemoryOwlV2ImageEmbeddingsCache, "init"
    ), patch.object(
        owlv2_inference_models.AutoModel,
        "from_pretrained",
        return_value=MagicMock(),
    ) as auto_model, patch.object(
        owlv2_inference_models, "get_extra_weights_provider_headers", return_value=None
    ), patch.object(
        owlv2_inference_models, "get_shared_model_blob_cache", return_value=shared_cache
    ) as shared_getter, patch.object(
        owlv2_inference_models, "OWLV2_COMPILE_MODEL", False
    ):
        owlv2_inference_models.Owlv2AdapterSingleton(
            "owlv2/test-default-shared-cache", api_key="test-key"
        )

    shared_getter.assert_called_once_with()
    assert (
        auto_model.call_args.kwargs["content_addressed_artifact_cache"] is shared_cache
    )
