from unittest.mock import MagicMock, patch

import pytest

from inference.models.owlv2 import owlv2_inference_models, rf_instant_inference_models


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
        auto_model.call_args.kwargs["content_addressed_artifact_cache"]
        is shared_cache
    )
