import sys
from unittest.mock import MagicMock

import torch

# Unit-test CI installs only ``[test]`` (no onnx-* extra). ``onnx.py`` hard-imports
# onnxruntime at module load, but the provider-default helpers under test do not
# call into it — stub the dependency so collection and assertions still run.
try:
    import onnxruntime  # noqa: F401
except ImportError:
    sys.modules["onnxruntime"] = MagicMock()

from inference_models.models.common import onnx


def test_offline_tensorrt_provider_never_writes_engine_cache(
    monkeypatch,
) -> None:
    monkeypatch.setattr(onnx, "OFFLINE_MODE", True)

    providers = onnx.set_onnx_execution_provider_defaults(
        providers=["TensorrtExecutionProvider", "CPUExecutionProvider"],
        model_package_path="/read-only/model-package",
        device=torch.device("cuda:0"),
        enable_fp16=True,
    )

    provider_name, provider_options = providers[0]
    assert provider_name == "TensorrtExecutionProvider"
    assert provider_options["trt_engine_cache_enable"] is False
    assert "trt_engine_cache_path" not in provider_options
    assert providers[1] == "CPUExecutionProvider"


def test_online_tensorrt_provider_retains_engine_cache(
    monkeypatch,
) -> None:
    monkeypatch.setattr(onnx, "OFFLINE_MODE", False)

    providers = onnx.set_onnx_execution_provider_defaults(
        providers=["TensorrtExecutionProvider"],
        model_package_path="/writable/model-package",
        device=torch.device("cuda:1"),
        enable_fp16=False,
    )

    provider_name, provider_options = providers[0]
    assert provider_name == "TensorrtExecutionProvider"
    assert provider_options["trt_engine_cache_enable"] is True
    assert provider_options["trt_engine_cache_path"] == ("/writable/model-package")
    assert provider_options["device_id"] == 1


def test_offline_custom_tensorrt_tuple_cannot_reenable_file_caches(
    monkeypatch,
) -> None:
    monkeypatch.setattr(onnx, "OFFLINE_MODE", True)
    original_options = {
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "/read-only/model-package",
        "trt_engine_cache_prefix": "engine",
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": "/read-only/model-package/timing",
        "trt_force_timing_cache": True,
        "trt_dump_subgraphs": True,
        "trt_dump_ep_context_model": True,
        "trt_ep_context_file_path": "/read-only/model-package/context.onnx",
        "trt_onnx_model_folder_path": "/read-only/model-package",
        "trt_fp16_enable": True,
    }

    providers = onnx.set_onnx_execution_provider_defaults(
        providers=[("TensorrtExecutionProvider", original_options)],
        model_package_path="/read-only/model-package",
        device=torch.device("cuda:0"),
        default_onnx_trt_options=False,
    )

    provider_name, provider_options = providers[0]
    assert provider_name == "TensorrtExecutionProvider"
    assert provider_options["trt_engine_cache_enable"] is False
    assert provider_options["trt_timing_cache_enable"] is False
    assert provider_options["trt_force_timing_cache"] is False
    assert provider_options["trt_dump_subgraphs"] is False
    assert provider_options["trt_dump_ep_context_model"] is False
    assert not {
        "trt_engine_cache_path",
        "trt_engine_cache_prefix",
        "trt_timing_cache_path",
        "trt_ep_context_file_path",
        "trt_onnx_model_folder_path",
    }.intersection(provider_options)
    assert provider_options["trt_fp16_enable"] is True
    assert original_options["trt_engine_cache_enable"] is True
    assert original_options["trt_engine_cache_path"] == ("/read-only/model-package")
