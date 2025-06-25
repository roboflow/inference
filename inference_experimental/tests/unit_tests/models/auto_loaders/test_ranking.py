from unittest import mock
from unittest.mock import MagicMock

import torch
from inference_exp.models.auto_loaders import ranking
from inference_exp.models.auto_loaders.ranking import (
    rank_cuda_versions,
    rank_model_packages,
    rank_trt_versions,
)
from inference_exp.runtime_introspection.core import RuntimeXRayResult
from inference_exp.weights_providers.entities import (
    BackendType,
    JetsonEnvironmentRequirements,
    ModelPackageMetadata,
    ONNXPackageDetails,
    Quantization,
    ServerEnvironmentRequirements,
    TRTPackageDetails,
)
from packaging.version import Version


def test_rank_model_packages_when_backends_should_be_prioritised_correctly() -> None:
    # given
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            package_artefacts=[],
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.ONNX,
            quantization=Quantization.FP32,
            package_artefacts=[],
        ),
        ModelPackageMetadata(
            package_id="my-package-id-3",
            backend=BackendType.ULTRALYTICS,
            quantization=Quantization.FP32,
            package_artefacts=[],
        ),
        ModelPackageMetadata(
            package_id="my-package-id-4",
            backend=BackendType.HF,
            quantization=Quantization.FP32,
            package_artefacts=[],
        ),
    ]

    # when
    result = rank_model_packages(model_packages=model_packages)

    # then
    assert [e.package_id for e in result] == [
        "my-package-id-1",
        "my-package-id-4",
        "my-package-id-2",
        "my-package-id-3",
    ]


def test_rank_model_packages_when_quantization_should_be_prioritised_correctly() -> (
    None
):
    # given
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            package_artefacts=[],
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.ONNX,
            quantization=Quantization.FP32,
            package_artefacts=[],
        ),
        ModelPackageMetadata(
            package_id="my-package-id-3",
            backend=BackendType.ONNX,
            quantization=Quantization.INT8,
            package_artefacts=[],
        ),
        ModelPackageMetadata(
            package_id="my-package-id-4",
            backend=BackendType.ONNX,
            quantization=Quantization.FP16,
            package_artefacts=[],
        ),
        ModelPackageMetadata(
            package_id="my-package-id-5",
            backend=BackendType.ONNX,
            quantization=Quantization.UNKNOWN,
            package_artefacts=[],
        ),
    ]

    # when
    result = rank_model_packages(model_packages=model_packages)

    # then
    assert [e.package_id for e in result] == [
        "my-package-id-1",
        "my-package-id-3",
        "my-package-id-4",
        "my-package-id-2",
        "my-package-id-5",
    ]


def test_rank_model_packages_when_dynamic_batch_size_should_be_prioritised_correctly() -> (
    None
):
    # given
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=False,
            static_batch_size=1,
            package_artefacts=[],
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
        ),
    ]

    # when
    result = rank_model_packages(model_packages=model_packages)

    # then
    assert [e.package_id for e in result] == ["my-package-id-2", "my-package-id-1"]


def test_rank_model_packages_when_static_batch_size_should_be_prioritised_correctly() -> (
    None
):
    # given
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=False,
            static_batch_size=5,
            package_artefacts=[],
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=False,
            static_batch_size=3,
            package_artefacts=[],
        ),
    ]

    # when
    result = rank_model_packages(model_packages=model_packages)

    # then
    assert [e.package_id for e in result] == ["my-package-id-2", "my-package-id-1"]


def test_rank_model_packages_when_opset_should_be_prioritised_correctly() -> None:
    # given
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.ONNX,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            onnx_package_details=ONNXPackageDetails(opset=17),
            package_artefacts=[],
        ),
        ModelPackageMetadata(
            package_id="my-package-id-3",
            backend=BackendType.ONNX,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            onnx_package_details=ONNXPackageDetails(opset=19),
            package_artefacts=[],
        ),
    ]

    # when
    result = rank_model_packages(model_packages=model_packages)

    # then
    assert [e.package_id for e in result] == [
        "my-package-id-1",
        "my-package-id-3",
        "my-package-id-2",
    ]


def test_rank_model_packages_when_trt_forward_compatibility_should_be_prioritised_correctly() -> (
    None
):
    # given
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            trt_package_details=TRTPackageDetails(same_cc_compatible=True),
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            trt_package_details=TRTPackageDetails(same_cc_compatible=False),
        ),
    ]

    # when
    result = rank_model_packages(model_packages=model_packages)

    # then
    assert [e.package_id for e in result] == ["my-package-id-2", "my-package-id-1"]


@mock.patch.object(ranking, "x_ray_runtime_environment")
def test_rank_model_packages_when_cuda_device_match_should_be_prioritised_correctly_when_cuda_device_selected(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["tesla-t4", "nvidia-l4"],
        gpu_devices_cc=[Version("7.5"), Version("8.7")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.3.0.1"),
        jetson_type=None,
        l4t_version=None,
        os_version="ubuntu-20.04",
        torch_available=True,
        onnxruntime_version=Version("1.21.0"),
        available_onnx_execution_providers={
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        },
        hf_transformers_available=True,
        ultralytics_available=True,
        trt_python_package_available=True,
    )
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            trt_package_details=TRTPackageDetails(same_cc_compatible=True),
            environment_requirements=ServerEnvironmentRequirements(
                cuda_device_cc=Version("7.5"),
                cuda_device_name="tesla-t4",
                cuda_version=Version("12.6"),
                driver_version=Version("510.0.4"),
                os_version="ubuntu-20.04",
                trt_version=Version("10.3.0.1"),
            ),
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            trt_package_details=TRTPackageDetails(same_cc_compatible=True),
            environment_requirements=ServerEnvironmentRequirements(
                cuda_device_cc=Version("8.7"),
                cuda_device_name="nvidia-l4",
                cuda_version=Version("12.6"),
                driver_version=Version("510.0.4"),
                os_version="ubuntu-20.04",
                trt_version=Version("10.3.0.1"),
            ),
        ),
    ]

    # when
    result_1 = rank_model_packages(
        model_packages=model_packages,
        selected_device=torch.device(type="cuda", index=0),
    )
    result_2 = rank_model_packages(
        model_packages=model_packages,
        selected_device=torch.device(type="cuda", index=1),
    )

    # then
    assert [e.package_id for e in result_1] == ["my-package-id-1", "my-package-id-2"]
    assert [e.package_id for e in result_2] == ["my-package-id-2", "my-package-id-1"]


@mock.patch.object(ranking, "x_ray_runtime_environment")
def test_rank_model_packages_when_cuda_device_match_should_be_prioritised_correctly_when_no_device_selected(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["nvidia-l4"],
        gpu_devices_cc=[Version("8.7")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.3.0.1"),
        jetson_type=None,
        l4t_version=None,
        os_version="ubuntu-20.04",
        torch_available=True,
        onnxruntime_version=Version("1.21.0"),
        available_onnx_execution_providers={
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        },
        hf_transformers_available=True,
        ultralytics_available=True,
        trt_python_package_available=True,
    )
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            trt_package_details=TRTPackageDetails(same_cc_compatible=True),
            environment_requirements=ServerEnvironmentRequirements(
                cuda_device_cc=Version("7.5"),
                cuda_device_name="tesla-t4",
                cuda_version=Version("12.6"),
                driver_version=Version("510.0.4"),
                os_version="ubuntu-20.04",
                trt_version=Version("10.3.0.1"),
            ),
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            trt_package_details=TRTPackageDetails(same_cc_compatible=True),
            environment_requirements=ServerEnvironmentRequirements(
                cuda_device_cc=Version("8.7"),
                cuda_device_name="nvidia-l4",
                cuda_version=Version("12.6"),
                driver_version=Version("510.0.4"),
                os_version="ubuntu-20.04",
                trt_version=Version("10.3.0.1"),
            ),
        ),
    ]

    # when
    result = rank_model_packages(
        model_packages=model_packages,
    )

    # then
    assert [e.package_id for e in result] == ["my-package-id-2", "my-package-id-1"]


def test_rank_model_packages_when_cuda_rank_should_be_prioritised_correctly() -> None:
    # given
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            environment_requirements=ServerEnvironmentRequirements(
                cuda_device_cc=Version("8.7"),
                cuda_device_name="nvidia-l4",
                cuda_version=Version("12.6.1"),
                driver_version=Version("510.0.4"),
                os_version="ubuntu-20.04",
                trt_version=Version("10.3.0.1"),
            ),
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            environment_requirements=ServerEnvironmentRequirements(
                cuda_device_cc=Version("8.7"),
                cuda_device_name="nvidia-l4",
                cuda_version=Version("12.6"),
                driver_version=Version("510.0.4"),
                os_version="ubuntu-20.04",
                trt_version=Version("10.3.0.1"),
            ),
        ),
    ]

    # when
    result = rank_model_packages(model_packages=model_packages)

    # then
    assert [e.package_id for e in result] == ["my-package-id-2", "my-package-id-1"]


def test_rank_model_packages_when_trt_rank_should_be_prioritised_correctly() -> None:
    # given
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            environment_requirements=ServerEnvironmentRequirements(
                cuda_device_cc=Version("8.7"),
                cuda_device_name="nvidia-l4",
                cuda_version=Version("12.6"),
                driver_version=Version("510.0.4"),
                os_version="ubuntu-20.04",
                trt_version=Version("10.3.0.1"),
            ),
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            environment_requirements=ServerEnvironmentRequirements(
                cuda_device_cc=Version("8.7"),
                cuda_device_name="nvidia-l4",
                cuda_version=Version("12.6"),
                driver_version=Version("510.0.4"),
                os_version="ubuntu-20.04",
                trt_version=Version("10.3.0"),
            ),
        ),
    ]

    # when
    result = rank_model_packages(model_packages=model_packages)

    # then
    assert [e.package_id for e in result] == ["my-package-id-2", "my-package-id-1"]


@mock.patch.object(ranking, "x_ray_runtime_environment")
def test_rank_model_packages_when_onnx_incompatible_providers_should_be_prioritised_correctly(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["nvidia-l4"],
        gpu_devices_cc=[Version("8.7")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.3.0.1"),
        jetson_type=None,
        l4t_version=None,
        os_version="ubuntu-20.04",
        torch_available=True,
        onnxruntime_version=Version("1.21.0"),
        available_onnx_execution_providers={
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        },
        hf_transformers_available=True,
        ultralytics_available=True,
        trt_python_package_available=True,
    )
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.ONNX,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            onnx_package_details=ONNXPackageDetails(
                incompatible_providers=["CUDAExecutionProvider"], opset=19
            ),
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.ONNX,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            onnx_package_details=ONNXPackageDetails(
                incompatible_providers=[], opset=19
            ),
        ),
    ]

    # when
    result = rank_model_packages(model_packages=model_packages)

    # then
    assert [e.package_id for e in result] == ["my-package-id-2", "my-package-id-1"]


def test_rank_model_packages_when_trt_dynamic_batch_size_should_be_prioritised_correctly() -> (
    None
):
    # given
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            trt_package_details=TRTPackageDetails(
                min_dynamic_batch_size=1,
                opt_dynamic_batch_size=8,
                max_dynamic_batch_size=16,
            ),
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            trt_package_details=TRTPackageDetails(
                min_dynamic_batch_size=1,
                opt_dynamic_batch_size=8,
                max_dynamic_batch_size=32,
            ),
        ),
    ]

    # when
    result = rank_model_packages(model_packages=model_packages)

    # then
    assert [e.package_id for e in result] == ["my-package-id-2", "my-package-id-1"]


def test_rank_model_packages_when_trt_lean_runtime_excluded_should_be_prioritised_correctly() -> (
    None
):
    # given
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            trt_package_details=TRTPackageDetails(
                trt_lean_runtime_excluded=True,
            ),
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            trt_package_details=TRTPackageDetails(
                trt_lean_runtime_excluded=False,
            ),
        ),
    ]

    # when
    result = rank_model_packages(model_packages=model_packages)

    # then
    assert [e.package_id for e in result] == ["my-package-id-2", "my-package-id-1"]


@mock.patch.object(ranking, "x_ray_runtime_environment")
def test_rank_model_packages_when_os_version_match_should_be_prioritised_correctly(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["nvidia-l4"],
        gpu_devices_cc=[Version("8.7")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.3.0.1"),
        jetson_type=None,
        l4t_version=None,
        os_version="ubuntu-20.04",
        torch_available=True,
        onnxruntime_version=Version("1.21.0"),
        available_onnx_execution_providers={
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        },
        hf_transformers_available=True,
        ultralytics_available=True,
        trt_python_package_available=True,
    )
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            environment_requirements=ServerEnvironmentRequirements(
                cuda_device_cc=Version("8.7"),
                cuda_device_name="nvidia-l4",
                cuda_version=Version("12.6"),
                driver_version=Version("510.0.4"),
                os_version="ubuntu-22.04",
                trt_version=Version("10.3.0"),
            ),
            trt_package_details=TRTPackageDetails(),
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            environment_requirements=ServerEnvironmentRequirements(
                cuda_device_cc=Version("8.7"),
                cuda_device_name="nvidia-l4",
                cuda_version=Version("12.6"),
                driver_version=Version("510.0.4"),
                os_version="ubuntu-20.04",
                trt_version=Version("10.3.0"),
            ),
            trt_package_details=TRTPackageDetails(),
        ),
    ]

    # when
    result = rank_model_packages(model_packages=model_packages)

    # then
    assert [e.package_id for e in result] == ["my-package-id-2", "my-package-id-1"]


@mock.patch.object(ranking, "x_ray_runtime_environment")
def test_rank_model_packages_when_l4t_version_match_should_be_prioritised_correctly(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["nvidia-l4"],
        gpu_devices_cc=[Version("8.7")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.3.0.1"),
        jetson_type=None,
        l4t_version=Version("36.4.0"),
        os_version="ubuntu-20.04",
        torch_available=True,
        onnxruntime_version=Version("1.21.0"),
        available_onnx_execution_providers={
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        },
        hf_transformers_available=True,
        ultralytics_available=True,
        trt_python_package_available=True,
    )
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            environment_requirements=JetsonEnvironmentRequirements(
                cuda_device_cc=Version("8.9"),
                cuda_device_name="orin",
                jetson_product_name="jetson-orin-nx",
                cuda_version=Version("12.6"),
                driver_version=Version("510.0.4"),
                l4t_version=Version("36.4.3"),
                trt_version=Version("10.3.0"),
            ),
            trt_package_details=TRTPackageDetails(),
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            environment_requirements=JetsonEnvironmentRequirements(
                cuda_device_cc=Version("8.9"),
                cuda_device_name="orin",
                jetson_product_name="jetson-orin-nx",
                cuda_version=Version("12.6"),
                driver_version=Version("510.0.4"),
                l4t_version=Version("36.4.0"),
                trt_version=Version("10.3.0"),
            ),
            trt_package_details=TRTPackageDetails(),
        ),
    ]

    # when
    result = rank_model_packages(model_packages=model_packages)

    # then
    assert [e.package_id for e in result] == ["my-package-id-2", "my-package-id-1"]


@mock.patch.object(ranking, "x_ray_runtime_environment")
def test_rank_model_packages_when_driver_version_match_should_be_prioritised_correctly(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["nvidia-l4"],
        gpu_devices_cc=[Version("8.7")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.3.0.1"),
        jetson_type=None,
        l4t_version=Version("36.4.0"),
        os_version="ubuntu-20.04",
        torch_available=True,
        onnxruntime_version=Version("1.21.0"),
        available_onnx_execution_providers={
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        },
        hf_transformers_available=True,
        ultralytics_available=True,
        trt_python_package_available=True,
    )
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            environment_requirements=JetsonEnvironmentRequirements(
                cuda_device_cc=Version("8.9"),
                cuda_device_name="orin",
                jetson_product_name="jetson-orin-nx",
                cuda_version=Version("12.6"),
                driver_version=Version("510.0.6"),
                l4t_version=Version("36.4.0"),
                trt_version=Version("10.3.0"),
            ),
            trt_package_details=TRTPackageDetails(),
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            environment_requirements=JetsonEnvironmentRequirements(
                cuda_device_cc=Version("8.9"),
                cuda_device_name="orin",
                jetson_product_name="jetson-orin-nx",
                cuda_version=Version("12.6"),
                driver_version=Version("510.0.4"),
                l4t_version=Version("36.4.0"),
                trt_version=Version("10.3.0"),
            ),
            trt_package_details=TRTPackageDetails(),
        ),
    ]

    # when
    result = rank_model_packages(model_packages=model_packages)

    # then
    assert [e.package_id for e in result] == ["my-package-id-2", "my-package-id-1"]


@mock.patch.object(ranking, "x_ray_runtime_environment")
def test_rank_model_packages_when_jetson_device_match_should_be_prioritised_correctly(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["orin"],
        gpu_devices_cc=[Version("8.9")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.3.0.1"),
        jetson_type="jetson-orin-nx",
        l4t_version=Version("36.4.0"),
        os_version="ubuntu-20.04",
        torch_available=True,
        onnxruntime_version=Version("1.21.0"),
        available_onnx_execution_providers={
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        },
        hf_transformers_available=True,
        ultralytics_available=True,
        trt_python_package_available=True,
    )
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            environment_requirements=JetsonEnvironmentRequirements(
                cuda_device_cc=Version("8.9"),
                cuda_device_name="orin",
                jetson_product_name="jetson-orin-agx",
                cuda_version=Version("12.6"),
                driver_version=Version("510.0.6"),
                l4t_version=Version("36.4.0"),
                trt_version=Version("10.3.0"),
            ),
            trt_package_details=TRTPackageDetails(),
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            environment_requirements=JetsonEnvironmentRequirements(
                cuda_device_cc=Version("8.9"),
                cuda_device_name="orin",
                jetson_product_name="jetson-orin-nx",
                cuda_version=Version("12.6"),
                driver_version=Version("510.0.6"),
                l4t_version=Version("36.4.0"),
                trt_version=Version("10.3.0"),
            ),
            trt_package_details=TRTPackageDetails(),
        ),
    ]

    # when
    result = rank_model_packages(model_packages=model_packages)

    # then
    assert [e.package_id for e in result] == ["my-package-id-2", "my-package-id-1"]


def test_rank_cuda_versions() -> None:
    # given
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            environment_requirements=ServerEnvironmentRequirements(
                cuda_device_cc=Version("8.7"),
                cuda_device_name="nvidia-l4",
                cuda_version=Version("12.6"),
                driver_version=Version("510.0.4"),
                os_version="ubuntu-20.04",
                trt_version=Version("10.3.0"),
            ),
            trt_package_details=TRTPackageDetails(),
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            environment_requirements=JetsonEnvironmentRequirements(
                cuda_device_cc=Version("8.9"),
                cuda_device_name="orin",
                jetson_product_name="jetson-orin-nx",
                cuda_version=Version("12.5"),
                driver_version=Version("510.0.6"),
                l4t_version=Version("36.4.0"),
                trt_version=Version("10.3.0"),
            ),
            trt_package_details=TRTPackageDetails(),
        ),
        ModelPackageMetadata(
            package_id="my-package-id-3",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            environment_requirements=JetsonEnvironmentRequirements(
                cuda_device_cc=Version("8.9"),
                cuda_device_name="orin",
                jetson_product_name="jetson-orin-nx",
                cuda_version=Version("12.7"),
                driver_version=Version("510.0.6"),
                l4t_version=Version("36.4.0"),
                trt_version=Version("10.3.0"),
            ),
            trt_package_details=TRTPackageDetails(),
        ),
        ModelPackageMetadata(
            package_id="my-package-id-4",
            backend=BackendType.ONNX,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            trt_package_details=TRTPackageDetails(),
        ),
    ]

    # when
    result = rank_cuda_versions(model_packages=model_packages)

    # then
    assert result == [-1, 0, -2, -3], (
        "Expected indices to be ordered such that lower cu version comes with highest rank and the higher cu version, "
        "the lower rank"
    )


def test_rank_trt_versions() -> None:
    # given
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            environment_requirements=ServerEnvironmentRequirements(
                cuda_device_cc=Version("8.7"),
                cuda_device_name="nvidia-l4",
                cuda_version=Version("12.6"),
                driver_version=Version("510.0.4"),
                os_version="ubuntu-20.04",
                trt_version=Version("10.3.0.4"),
            ),
            trt_package_details=TRTPackageDetails(),
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            environment_requirements=JetsonEnvironmentRequirements(
                cuda_device_cc=Version("8.9"),
                cuda_device_name="orin",
                jetson_product_name="jetson-orin-nx",
                cuda_version=Version("12.6"),
                driver_version=Version("510.0.6"),
                l4t_version=Version("36.4.0"),
                trt_version=Version("10.3.0.3"),
            ),
            trt_package_details=TRTPackageDetails(),
        ),
        ModelPackageMetadata(
            package_id="my-package-id-3",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            environment_requirements=JetsonEnvironmentRequirements(
                cuda_device_cc=Version("8.9"),
                cuda_device_name="orin",
                jetson_product_name="jetson-orin-nx",
                cuda_version=Version("12.6"),
                driver_version=Version("510.0.6"),
                l4t_version=Version("36.4.0"),
                trt_version=Version("10.3.0.2"),
            ),
            trt_package_details=TRTPackageDetails(),
        ),
        ModelPackageMetadata(
            package_id="my-package-id-4",
            backend=BackendType.ONNX,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            trt_package_details=TRTPackageDetails(),
        ),
    ]

    # when
    result = rank_trt_versions(model_packages=model_packages)

    # then
    assert result == [-2, -1, 0, -3], (
        "Expected indices to be ordered such that lower cu version comes with highest rank and the higher cu version, "
        "the lower rank"
    )
