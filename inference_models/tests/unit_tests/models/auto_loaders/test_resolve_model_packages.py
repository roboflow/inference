from unittest import mock

import pytest
from packaging.version import Version

from inference_models import AutoModel
from inference_models.entities import ResolvedModelMetadata
from inference_models.models.auto_loaders import auto_negotiation, core
from inference_models.models.auto_loaders.entities import BackendType
from inference_models.runtime_introspection.core import RuntimeXRayResult
from inference_models.weights_providers.entities import (
    ModelMetadata,
    ModelPackageMetadata,
    ONNXPackageDetails,
    Quantization,
)


@pytest.fixture(autouse=True)
def onnx_cpu_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = RuntimeXRayResult(
        gpu_available=False,
        gpu_devices=[],
        gpu_devices_cc=[],
        driver_version=None,
        cuda_version=None,
        trt_version=None,
        jetson_type=None,
        l4t_version=None,
        os_version="ubuntu-22.04",
        torch_available=True,
        torch_version=Version("2.7.0"),
        torchvision_version=Version("0.22.0"),
        onnxruntime_version=Version("1.21.0"),
        available_onnx_execution_providers={"CPUExecutionProvider"},
        hf_transformers_available=False,
        trt_python_package_available=False,
    )
    monkeypatch.setattr(auto_negotiation, "x_ray_runtime_environment", lambda: runtime)


def test_resolve_model_packages_without_loading_weights() -> None:
    metadata = ModelMetadata(
        model_id="workspace/model/1",
        model_architecture="yolov8",
        task_type="object-detection",
        model_packages=[
            ModelPackageMetadata(
                package_id="onnx-fp32",
                backend=BackendType.ONNX,
                quantization=Quantization.FP32,
                package_artefacts=[],
                trusted_source=True,
                onnx_package_details=ONNXPackageDetails(opset=17),
            )
        ],
    )
    with mock.patch.object(
        core, "get_model_from_provider", return_value=metadata
    ) as provider, mock.patch.object(AutoModel, "from_pretrained") as load:
        result = AutoModel.resolve_model_packages(
            "model-alias",
            api_key="current-key",
            backend="onnx",
            quantization="fp32",
            device="cpu",
            onnx_execution_providers=["CPUExecutionProvider"],
        )

    assert result == [
        ResolvedModelMetadata(
            model_id="workspace/model/1",
            model_package_id="onnx-fp32",
            backend="onnx",
            quantization="fp32",
        )
    ]
    assert provider.call_args.kwargs["api_key"] == "current-key"
    load.assert_not_called()


def test_resolve_model_packages_checks_each_callers_access() -> None:
    from inference_models.errors import UnauthorizedModelAccessError

    package = ModelPackageMetadata(
        package_id="onnx-fp32",
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
        package_artefacts=[],
        trusted_source=True,
        onnx_package_details=ONNXPackageDetails(opset=17),
    )
    metadata = ModelMetadata(
        model_id="workspace/model/1",
        model_architecture="yolov8",
        task_type="object-detection",
        model_packages=[package],
    )
    denied = UnauthorizedModelAccessError(message="Access denied")
    with mock.patch.object(
        core, "get_model_from_provider", side_effect=[metadata, denied]
    ) as provider:
        AutoModel.resolve_model_packages(
            "workspace/model/1",
            api_key="authorized-key",
            device="cpu",
            onnx_execution_providers=["CPUExecutionProvider"],
        )
        with pytest.raises(UnauthorizedModelAccessError):
            AutoModel.resolve_model_packages(
                "workspace/model/1", api_key="unauthorized-key"
            )

    assert [call.kwargs["api_key"] for call in provider.call_args_list] == [
        "authorized-key",
        "unauthorized-key",
    ]


@pytest.mark.parametrize("selectors", [{"backend": "onnx"}, {}])
def test_resolve_model_packages_rejects_pinned_trt_on_cpu(selectors: dict) -> None:
    from inference_models.errors import ModelPackagePolicyError

    metadata = ModelMetadata(
        model_id="workspace/model/1",
        model_architecture="yolov8",
        task_type="object-detection",
        model_packages=[
            ModelPackageMetadata(
                package_id="trt-fp16",
                backend=BackendType.TRT,
                quantization=Quantization.FP16,
                package_artefacts=[],
                trusted_source=True,
            )
        ],
    )
    with mock.patch.object(core, "get_model_from_provider", return_value=metadata):
        with pytest.raises(ModelPackagePolicyError):
            AutoModel.resolve_model_packages(
                "workspace/model/1",
                model_package_id="trt-fp16",
                device="cpu",
                **selectors,
            )


@pytest.mark.parametrize(
    "selectors, expected_packages",
    [
        ({"backend": "onnx", "quantization": "fp32"}, ["onnx-fp32"]),
        ({"quantization": "fp16"}, ["onnx-fp16"]),
        ({"model_package_id": "onnx-fp16"}, ["onnx-fp16"]),
        ({"quantization": ["fp32", "fp16"]}, ["onnx-fp16", "onnx-fp32"]),
    ],
)
def test_resolve_model_packages_selects_matching_package(
    selectors: dict, expected_packages: list
) -> None:
    metadata = ModelMetadata(
        model_id="workspace/model/1",
        model_architecture="yolov8",
        task_type="object-detection",
        model_packages=[
            ModelPackageMetadata(
                package_id=f"onnx-{quantization.value}",
                backend=BackendType.ONNX,
                quantization=quantization,
                package_artefacts=[],
                trusted_source=True,
                onnx_package_details=ONNXPackageDetails(opset=17),
            )
            for quantization in [Quantization.FP32, Quantization.FP16]
        ],
    )
    with mock.patch.object(core, "get_model_from_provider", return_value=metadata):
        result = AutoModel.resolve_model_packages(
            "workspace/model/1",
            device="cpu",
            onnx_execution_providers=["CPUExecutionProvider"],
            **selectors,
        )

    assert [package.model_package_id for package in result] == expected_packages


def test_resolve_model_packages_uses_offline_provider_and_request_headers() -> None:
    from inference_models.errors import UnauthorizedModelAccessError

    with mock.patch.object(core, "OFFLINE_MODE", True), mock.patch.object(
        core, "OFFLINE_MODE_WARM_UP", False
    ), mock.patch.object(
        core,
        "get_model_from_provider",
        side_effect=UnauthorizedModelAccessError(message="Access denied"),
    ) as provider:
        with pytest.raises(UnauthorizedModelAccessError):
            AutoModel.resolve_model_packages(
                "workspace/model/1",
                api_key="current-key",
                weights_provider_extra_headers={"X-Request-Id": "request-1"},
            )

    assert (
        provider.call_args.kwargs["provider"] == core.ROBOFLOW_OFFLINE_WEIGHTS_PROVIDER
    )
    assert provider.call_args.kwargs["api_key"] == "current-key"
    assert provider.call_args.kwargs["weights_provider_extra_headers"] == {
        "X-Request-Id": "request-1"
    }
