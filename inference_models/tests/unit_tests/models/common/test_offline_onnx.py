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


class _FakeOrtInput:
    def __init__(self, name: str, type: str) -> None:
        self.name = name
        self.type = type


class _FakeSession:
    def __init__(self, inputs) -> None:
        self._inputs = inputs

    def get_inputs(self):
        return self._inputs


def test_auto_cast_session_inputs_widens_bfloat16_to_float32() -> None:
    session = _FakeSession([_FakeOrtInput(name="images", type="tensor(float)")])
    inputs = {"images": torch.ones((1, 3, 4, 4), dtype=torch.bfloat16)}

    result = onnx.auto_cast_session_inputs(session=session, inputs=inputs)

    assert result["images"].dtype == torch.float32


def test_auto_cast_session_inputs_rejects_narrowing_float32_to_bfloat16() -> None:
    assert not onnx.can_model_input_be_casted(
        source=torch.float32, target=torch.bfloat16
    )
