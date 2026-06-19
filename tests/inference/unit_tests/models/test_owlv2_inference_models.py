from unittest.mock import MagicMock, patch

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
