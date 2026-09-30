import pytest
import torch

pytest.importorskip(
    "onnxruntime",
    reason="onnxruntime is not installed (requires the onnx-* extra)",
)

from inference_models.models.common import onnx


def test_tensorrt_provider_retains_engine_cache() -> None:
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
