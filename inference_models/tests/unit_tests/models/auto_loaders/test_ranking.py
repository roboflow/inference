from typing import Union
from unittest import mock
from unittest.mock import MagicMock

import pytest
import torch
from packaging.version import Version

from inference_models.models.auto_loaders import ranking
from inference_models.models.auto_loaders.entities import BackendType
from inference_models.models.auto_loaders.ranking import (
    rank_cuda_versions,
    rank_model_packages,
    rank_packages_ids,
    rank_trt_versions,
    retrieve_cuda_device_match_score,
    retrieve_driver_version_match_score,
    retrieve_fused_nms_rank,
    retrieve_jetson_device_name_match_score,
    retrieve_l4t_version_match_score,
    retrieve_onnx_incompatible_providers_score,
    retrieve_os_version_match_score,
    retrieve_same_trt_cc_compatibility_score,
    retrieve_trt_dynamic_batch_size_score,
    retrieve_trt_forward_compatible_match_score,
    retrieve_trt_lean_runtime_excluded_score,
)
from inference_models.runtime_introspection.core import RuntimeXRayResult
from inference_models.weights_providers.entities import (
    JetsonEnvironmentRequirements,
    ModelPackageMetadata,
    ONNXPackageDetails,
    Quantization,
    ServerEnvironmentRequirements,
    TRTPackageDetails,
)


def test_rank_model_packages_when_trusted_source_flag_is_taken_into_account() -> None:
    # given
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            package_artefacts=[],
            trusted_source=False,
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            package_artefacts=[],
            trusted_source=True,
        ),
    ]

    # when
    result = rank_model_packages(model_packages=model_packages)

    # then
    assert [e.package_id for e in result] == [
        "my-package-id-2",
        "my-package-id-1",
    ]


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
        torch_version=Version("2.7.0"),
        torchvision_version=Version("0.22.0"),
        onnxruntime_version=Version("1.21.0"),
        available_onnx_execution_providers={
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        },
        hf_transformers_available=True,
        ultralytics_available=True,
        trt_python_package_available=True,
        mediapipe_available=False,
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
        torch_version=Version("2.7.0"),
        torchvision_version=Version("0.22.0"),
        onnxruntime_version=Version("1.21.0"),
        available_onnx_execution_providers={
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        },
        hf_transformers_available=True,
        ultralytics_available=True,
        trt_python_package_available=True,
        mediapipe_available=False,
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
        torch_version=Version("2.7.0"),
        torchvision_version=Version("0.22.0"),
        onnxruntime_version=Version("1.21.0"),
        available_onnx_execution_providers={
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        },
        hf_transformers_available=True,
        ultralytics_available=True,
        trt_python_package_available=True,
        mediapipe_available=False,
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
        torch_version=Version("2.7.0"),
        torchvision_version=Version("0.22.0"),
        onnxruntime_version=Version("1.21.0"),
        available_onnx_execution_providers={
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        },
        hf_transformers_available=True,
        ultralytics_available=True,
        trt_python_package_available=True,
        mediapipe_available=False,
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
        torch_version=Version("2.7.0"),
        torchvision_version=Version("0.22.0"),
        onnxruntime_version=Version("1.21.0"),
        available_onnx_execution_providers={
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        },
        hf_transformers_available=True,
        ultralytics_available=True,
        trt_python_package_available=True,
        mediapipe_available=False,
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
        torch_version=Version("2.7.0"),
        torchvision_version=Version("0.22.0"),
        onnxruntime_version=Version("1.21.0"),
        available_onnx_execution_providers={
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        },
        hf_transformers_available=True,
        ultralytics_available=True,
        trt_python_package_available=True,
        mediapipe_available=False,
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
        torch_version=Version("2.7.0"),
        torchvision_version=Version("0.22.0"),
        onnxruntime_version=Version("1.21.0"),
        available_onnx_execution_providers={
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        },
        hf_transformers_available=True,
        ultralytics_available=True,
        trt_python_package_available=True,
        mediapipe_available=False,
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


def test_retrieve_jetson_device_name_match_score_when_not_a_trt_package() -> None:
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.ONNX,
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
    )

    # when
    result = retrieve_jetson_device_name_match_score(model_package=model_package)

    # then
    assert result == 0


def test_retrieve_jetson_device_name_match_score_when_package_does_not_provide_environment_requirements() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(),
    )

    # when
    result = retrieve_jetson_device_name_match_score(model_package=model_package)

    # then
    assert result == 0


def test_retrieve_jetson_device_name_match_score_when_unknown_environment_requirements() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(),
        environment_requirements=MagicMock(),
    )

    # when
    result = retrieve_jetson_device_name_match_score(model_package=model_package)

    # then
    assert result == 0


@mock.patch.object(ranking, "x_ray_runtime_environment")
def test_retrieve_jetson_device_name_match_score_when_jetson_device_match(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value.jetson_type = "jetson-orin-nx"
    model_package = ModelPackageMetadata(
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
    )

    # when
    result = retrieve_jetson_device_name_match_score(model_package=model_package)

    # then
    assert result == 1


@mock.patch.object(ranking, "x_ray_runtime_environment")
def test_retrieve_jetson_device_name_match_score_when_jetson_device_does_not_match(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value.jetson_type = "jetson-orin-agx"
    model_package = ModelPackageMetadata(
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
    )

    # when
    result = retrieve_jetson_device_name_match_score(model_package=model_package)

    # then
    assert result == 0


def test_retrieve_driver_version_match_score_when_not_a_trt_package() -> None:
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.ONNX,
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
    )

    # when
    result = retrieve_driver_version_match_score(model_package=model_package)

    # then
    assert result == 0


def test_retrieve_driver_version_match_score_when_no_environment_requirements() -> None:
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(),
    )

    # when
    result = retrieve_driver_version_match_score(model_package=model_package)

    # then
    assert result == 0


def test_retrieve_driver_version_match_score_when_unknown_environment_requirements() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(),
        environment_requirements=MagicMock(),
    )

    # when
    result = retrieve_driver_version_match_score(model_package=model_package)

    # then
    assert result == 0


def test_retrieve_driver_version_match_score_when_no_driver_version_manifested() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(),
        environment_requirements=JetsonEnvironmentRequirements(
            cuda_device_cc=Version("8.9"),
            cuda_device_name="orin",
            jetson_product_name="jetson-orin-nx",
            cuda_version=Version("12.6"),
            driver_version=None,
            l4t_version=Version("36.4.0"),
            trt_version=Version("10.3.0.3"),
        ),
    )

    # when
    result = retrieve_driver_version_match_score(model_package=model_package)

    # then
    assert result == 0


@mock.patch.object(ranking, "x_ray_runtime_environment")
def test_retrieve_driver_version_match_score_when_gpu_driver_matches(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value.driver_version = Version("510.0.13")
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(),
        environment_requirements=ServerEnvironmentRequirements(
            cuda_device_cc=Version("7.5"),
            cuda_device_name="tesla-t4",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.13"),
            os_version="ubuntu-20.04",
            trt_version=Version("10.3.0.1"),
        ),
    )

    # when
    result = retrieve_driver_version_match_score(model_package=model_package)

    # then
    assert result == 1


@mock.patch.object(ranking, "x_ray_runtime_environment")
def test_retrieve_driver_version_match_score_when_gpu_driver_does_not_match(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value.driver_version = Version("510.0.14")
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(),
        environment_requirements=ServerEnvironmentRequirements(
            cuda_device_cc=Version("7.5"),
            cuda_device_name="tesla-t4",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.13"),
            os_version="ubuntu-20.04",
            trt_version=Version("10.3.0.1"),
        ),
    )

    # when
    result = retrieve_driver_version_match_score(model_package=model_package)

    # then
    assert result == 0


@mock.patch.object(ranking, "x_ray_runtime_environment")
def test_retrieve_driver_version_match_score_when_jetson_driver_matches(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value.driver_version = Version("510.0.13")
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(),
        environment_requirements=JetsonEnvironmentRequirements(
            cuda_device_cc=Version("8.9"),
            cuda_device_name="orin",
            jetson_product_name="jetson-orin-nx",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.13"),
            l4t_version=Version("36.4.0"),
            trt_version=Version("10.3.0.3"),
        ),
    )

    # when
    result = retrieve_driver_version_match_score(model_package=model_package)

    # then
    assert result == 1


@mock.patch.object(ranking, "x_ray_runtime_environment")
def test_retrieve_driver_version_match_score_when_jetson_driver_does_not_match(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value.driver_version = Version("510.0.13")
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(),
        environment_requirements=JetsonEnvironmentRequirements(
            cuda_device_cc=Version("8.9"),
            cuda_device_name="orin",
            jetson_product_name="jetson-orin-nx",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.14"),
            l4t_version=Version("36.4.0"),
            trt_version=Version("10.3.0.3"),
        ),
    )

    # when
    result = retrieve_driver_version_match_score(model_package=model_package)

    # then
    assert result == 0


def test_retrieve_l4t_version_match_score_when_not_a_trt_package() -> None:
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.ONNX,
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
    )

    # when
    result = retrieve_l4t_version_match_score(model_package=model_package)

    # then
    assert result == 0


def test_retrieve_l4t_version_match_score_when_no_environment_requirements() -> None:
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(),
    )

    # when
    result = retrieve_l4t_version_match_score(model_package=model_package)

    # then
    assert result == 0


def test_retrieve_l4t_version_match_score_when_not_a_jetson_environment() -> None:
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(),
        environment_requirements=ServerEnvironmentRequirements(
            cuda_device_cc=Version("7.5"),
            cuda_device_name="tesla-t4",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.13"),
            os_version="ubuntu-20.04",
            trt_version=Version("10.3.0.1"),
        ),
    )

    # when
    result = retrieve_l4t_version_match_score(model_package=model_package)

    # then
    assert result == 0


@mock.patch.object(ranking, "x_ray_runtime_environment")
def test_retrieve_l4t_version_match_score_when_jetpack_matches(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value.l4t_version = Version("36.4.0")
    model_package = ModelPackageMetadata(
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
    )

    # when
    result = retrieve_l4t_version_match_score(model_package=model_package)

    # then
    assert result == 1


@mock.patch.object(ranking, "x_ray_runtime_environment")
def test_retrieve_l4t_version_match_score_when_jetpack_does_not_match(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value.l4t_version = Version("36.4.3")
    model_package = ModelPackageMetadata(
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
    )

    # when
    result = retrieve_l4t_version_match_score(model_package=model_package)

    # then
    assert result == 0


def test_retrieve_os_version_match_score_when_not_a_trt_package() -> None:
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        environment_requirements=ServerEnvironmentRequirements(
            cuda_device_cc=Version("7.5"),
            cuda_device_name="tesla-t4",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.13"),
            os_version="ubuntu-20.04",
            trt_version=Version("10.3.0.1"),
        ),
    )

    # when
    result = retrieve_os_version_match_score(model_package=model_package)

    # then
    assert result == 0


def test_retrieve_os_version_match_score_when_no_environment_requirements() -> None:
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
    )

    # when
    result = retrieve_os_version_match_score(model_package=model_package)

    # then
    assert result == 0


def test_retrieve_os_version_match_score_when_not_a_server_requirements() -> None:
    # given
    model_package = ModelPackageMetadata(
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
    )

    # when
    result = retrieve_os_version_match_score(model_package=model_package)

    # then
    assert result == 0


@mock.patch.object(ranking, "x_ray_runtime_environment")
def test_retrieve_os_version_match_score_when_os_matches(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value.os_version = "ubuntu-20.04"
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        environment_requirements=ServerEnvironmentRequirements(
            cuda_device_cc=Version("7.5"),
            cuda_device_name="tesla-t4",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.13"),
            os_version="ubuntu-20.04",
            trt_version=Version("10.3.0.1"),
        ),
    )

    # when
    result = retrieve_os_version_match_score(model_package=model_package)

    # then
    assert result == 1


@mock.patch.object(ranking, "x_ray_runtime_environment")
def test_retrieve_os_version_match_score_when_os_does_not_match(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value.os_version = "ubuntu-22.04"
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        environment_requirements=ServerEnvironmentRequirements(
            cuda_device_cc=Version("7.5"),
            cuda_device_name="tesla-t4",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.13"),
            os_version="ubuntu-20.04",
            trt_version=Version("10.3.0.1"),
        ),
    )

    # when
    result = retrieve_os_version_match_score(model_package=model_package)

    # then
    assert result == 0


def test_retrieve_trt_lean_runtime_excluded_score_when_no_trt_package_details() -> None:
    # given
    model_package = ModelPackageMetadata(
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
    )

    # when
    result = retrieve_trt_lean_runtime_excluded_score(model_package=model_package)

    # then
    assert result == 0


def test_retrieve_trt_lean_runtime_excluded_score_when_runtime_excluded() -> None:
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(trt_lean_runtime_excluded=True),
        environment_requirements=JetsonEnvironmentRequirements(
            cuda_device_cc=Version("8.9"),
            cuda_device_name="orin",
            jetson_product_name="jetson-orin-nx",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.6"),
            l4t_version=Version("36.4.0"),
            trt_version=Version("10.3.0.3"),
        ),
    )

    # when
    result = retrieve_trt_lean_runtime_excluded_score(model_package=model_package)

    # then
    assert result == 0


def test_retrieve_trt_lean_runtime_excluded_score_when_runtime_included() -> None:
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(trt_lean_runtime_excluded=False),
        environment_requirements=JetsonEnvironmentRequirements(
            cuda_device_cc=Version("8.9"),
            cuda_device_name="orin",
            jetson_product_name="jetson-orin-nx",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.6"),
            l4t_version=Version("36.4.0"),
            trt_version=Version("10.3.0.3"),
        ),
    )

    # when
    result = retrieve_trt_lean_runtime_excluded_score(model_package=model_package)

    # then
    assert result == 1


def test_retrieve_trt_dynamic_batch_size_score_when_no_trt_package_details() -> None:
    # given
    model_package = ModelPackageMetadata(
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
    )

    # when
    result = retrieve_trt_dynamic_batch_size_score(model_package=model_package)

    # then
    assert result == 0


def test_retrieve_trt_dynamic_batch_size_score_when_batch_size_details_not_filled() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(),
        environment_requirements=JetsonEnvironmentRequirements(
            cuda_device_cc=Version("8.9"),
            cuda_device_name="orin",
            jetson_product_name="jetson-orin-nx",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.6"),
            l4t_version=Version("36.4.0"),
            trt_version=Version("10.3.0.3"),
        ),
    )

    # when
    result = retrieve_trt_dynamic_batch_size_score(model_package=model_package)

    # then
    assert result == 0


@pytest.mark.parametrize(
    "min_dynamic_batch_size, max_dynamic_batch_size, expected_score",
    [
        (1, 8, 7),
        (1, 16, 15),
    ],
)
def test_retrieve_trt_dynamic_batch_size_score_when_batch_size_range_declared(
    min_dynamic_batch_size: int,
    max_dynamic_batch_size: int,
    expected_score: int,
) -> None:
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(
            min_dynamic_batch_size=min_dynamic_batch_size,
            max_dynamic_batch_size=max_dynamic_batch_size,
        ),
        environment_requirements=JetsonEnvironmentRequirements(
            cuda_device_cc=Version("8.9"),
            cuda_device_name="orin",
            jetson_product_name="jetson-orin-nx",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.6"),
            l4t_version=Version("36.4.0"),
            trt_version=Version("10.3.0.3"),
        ),
    )

    # when
    result = retrieve_trt_dynamic_batch_size_score(model_package=model_package)

    # then
    assert result == expected_score


def test_retrieve_onnx_incompatible_providers_score_when_no_onnx_package_details() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
    )

    # when
    result = retrieve_onnx_incompatible_providers_score(model_package=model_package)

    # then
    assert result == 0


def test_retrieve_onnx_incompatible_providers_score_when_no_incompatible_providers() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        onnx_package_details=ONNXPackageDetails(opset=19),
    )

    # when
    result = retrieve_onnx_incompatible_providers_score(model_package=model_package)

    # then
    assert result == 0


@mock.patch.object(ranking, "x_ray_runtime_environment")
def test_retrieve_onnx_incompatible_providers_score_when_incompatible_providers_overlap_with_runtime(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value.available_onnx_execution_providers = [
        "CoreMLExecutionProvider",
        "CUDAExecutionProvider",
        "TensorRTExecutionProvider",
    ]
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        onnx_package_details=ONNXPackageDetails(
            opset=19,
            incompatible_providers=[
                "CoreMLExecutionProvider",
                "AzureExecutionProvider",
                "CUDAExecutionProvider",
            ],
        ),
    )

    # when
    result = retrieve_onnx_incompatible_providers_score(model_package=model_package)

    # then
    assert result == -2


@mock.patch.object(ranking, "x_ray_runtime_environment")
def test_retrieve_onnx_incompatible_providers_score_when_no_incompatible_providers_overlap_with_runtime(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value.available_onnx_execution_providers = [
        "CUDAExecutionProvider",
        "TensorRTExecutionProvider",
    ]
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        onnx_package_details=ONNXPackageDetails(
            opset=19,
            incompatible_providers=[
                "CoreMLExecutionProvider",
                "AzureExecutionProvider",
            ],
        ),
    )

    # when
    result = retrieve_onnx_incompatible_providers_score(model_package=model_package)

    # then
    assert result == 0


def test_retrieve_trt_forward_compatible_match_score_when_no_trt_package_details() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
    )

    # when
    result = retrieve_trt_forward_compatible_match_score(model_package=model_package)

    # then
    assert result == 1


def test_retrieve_trt_forward_compatible_match_score_when_forward_compatible() -> None:
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(trt_forward_compatible=True),
    )

    # when
    result = retrieve_trt_forward_compatible_match_score(model_package=model_package)

    # then
    assert result == 0


def test_retrieve_trt_forward_compatible_match_score_when_not_forward_compatible() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(trt_forward_compatible=False),
    )

    # when
    result = retrieve_trt_forward_compatible_match_score(model_package=model_package)

    # then
    assert result == 1


def test_retrieve_same_trt_cc_compatibility_score_when_no_trt_package_details() -> None:
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
    )

    # when
    result = retrieve_same_trt_cc_compatibility_score(model_package=model_package)

    # then
    assert result == 1


def test_retrieve_same_trt_cc_compatibility_score_when_compatibility_enabled() -> None:
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(same_cc_compatible=True),
    )

    # when
    result = retrieve_same_trt_cc_compatibility_score(model_package=model_package)

    # then
    assert result == 0


def test_retrieve_same_trt_cc_compatibility_score_when_compatibility_disabled() -> None:
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(same_cc_compatible=False),
    )

    # when
    result = retrieve_same_trt_cc_compatibility_score(model_package=model_package)

    # then
    assert result == 1


def test_retrieve_cuda_device_match_score_when_not_a_trt_package() -> None:
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(same_cc_compatible=False),
        environment_requirements=JetsonEnvironmentRequirements(
            cuda_device_cc=Version("8.9"),
            cuda_device_name="orin",
            jetson_product_name="jetson-orin-nx",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.6"),
            l4t_version=Version("36.4.0"),
            trt_version=Version("10.3.0.3"),
        ),
    )

    # when
    result = retrieve_cuda_device_match_score(model_package=model_package)

    # then
    assert result == 0


def test_retrieve_cuda_device_match_score_when_no_environment_requirements() -> None:
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(same_cc_compatible=False),
    )

    # when
    result = retrieve_cuda_device_match_score(model_package=model_package)

    # then
    assert result == 0


def test_retrieve_cuda_device_match_score_when_unknown_environment_requirements() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(same_cc_compatible=False),
        environment_requirements=MagicMock(),
    )

    # when
    result = retrieve_cuda_device_match_score(model_package=model_package)

    # then
    assert result == 0


@mock.patch.object(ranking, "x_ray_runtime_environment")
def test_retrieve_cuda_device_match_score_when_no_selected_device_and_one_available_device_matches_for_jetson(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value.gpu_devices = ["orin"]
    x_ray_runtime_environment_mock.return_value.gpu_devices_cc = [Version("8.9")]
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(same_cc_compatible=False),
        environment_requirements=JetsonEnvironmentRequirements(
            cuda_device_cc=Version("8.9"),
            cuda_device_name="orin",
            jetson_product_name="jetson-orin-nx",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.6"),
            l4t_version=Version("36.4.0"),
            trt_version=Version("10.3.0.3"),
        ),
    )

    # when
    result = retrieve_cuda_device_match_score(model_package=model_package)

    # then
    assert result == 1


@mock.patch.object(ranking, "x_ray_runtime_environment")
def test_retrieve_cuda_device_match_score_when_no_selected_device_and_no_available_device_matches_for_jetson(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value.gpu_devices = ["orin-super"]
    x_ray_runtime_environment_mock.return_value.gpu_devices_cc = [Version("8.9")]
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(same_cc_compatible=False),
        environment_requirements=JetsonEnvironmentRequirements(
            cuda_device_cc=Version("8.9"),
            cuda_device_name="orin",
            jetson_product_name="jetson-orin-nx",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.6"),
            l4t_version=Version("36.4.0"),
            trt_version=Version("10.3.0.3"),
        ),
    )

    # when
    result = retrieve_cuda_device_match_score(model_package=model_package)

    # then
    assert result == 0


@mock.patch.object(ranking, "x_ray_runtime_environment")
def test_retrieve_cuda_device_match_score_when_no_selected_device_and_one_available_device_matches_for_server(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value.gpu_devices = ["tesla-t4", "nvidia-l4"]
    x_ray_runtime_environment_mock.return_value.gpu_devices_cc = [
        Version("7.5"),
        Version("8.7"),
    ]
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(same_cc_compatible=False),
        environment_requirements=ServerEnvironmentRequirements(
            cuda_device_cc=Version("7.5"),
            cuda_device_name="tesla-t4",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.13"),
            os_version="ubuntu-20.04",
            trt_version=Version("10.3.0.1"),
        ),
    )

    # when
    result = retrieve_cuda_device_match_score(model_package=model_package)

    # then
    assert result == 1


@mock.patch.object(ranking, "x_ray_runtime_environment")
def test_retrieve_cuda_device_match_score_when_no_selected_device_and_no_available_device_matches_for_server(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value.gpu_devices = ["tesla-t4", "nvidia-l4"]
    x_ray_runtime_environment_mock.return_value.gpu_devices_cc = [
        Version("7.5"),
        Version("8.7"),
    ]
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(same_cc_compatible=False),
        environment_requirements=ServerEnvironmentRequirements(
            cuda_device_cc=Version("7.5"),
            cuda_device_name="nvidia-l40",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.13"),
            os_version="ubuntu-20.04",
            trt_version=Version("10.3.0.1"),
        ),
    )

    # when
    result = retrieve_cuda_device_match_score(model_package=model_package)

    # then
    assert result == 0


@mock.patch.object(ranking, "x_ray_runtime_environment")
def test_retrieve_cuda_device_match_score_when_no_selected_device_and_multiple_available_device_matches_for_server(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value.gpu_devices = ["tesla-t4", "tesla-t4"]
    x_ray_runtime_environment_mock.return_value.gpu_devices_cc = [
        Version("7.5"),
        Version("7.5"),
    ]
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(same_cc_compatible=False),
        environment_requirements=ServerEnvironmentRequirements(
            cuda_device_cc=Version("7.5"),
            cuda_device_name="tesla-t4",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.13"),
            os_version="ubuntu-20.04",
            trt_version=Version("10.3.0.1"),
        ),
    )

    # when
    result = retrieve_cuda_device_match_score(model_package=model_package)

    # then
    assert result == 2


@mock.patch.object(ranking, "x_ray_runtime_environment")
def test_retrieve_cuda_device_match_score_when_selected_device_matches_for_jetson(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value.gpu_devices = ["orin"]
    x_ray_runtime_environment_mock.return_value.gpu_devices_cc = [Version("8.9")]
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(same_cc_compatible=False),
        environment_requirements=JetsonEnvironmentRequirements(
            cuda_device_cc=Version("8.9"),
            cuda_device_name="orin",
            jetson_product_name="jetson-orin-nx",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.6"),
            l4t_version=Version("36.4.0"),
            trt_version=Version("10.3.0.3"),
        ),
    )

    # when
    result = retrieve_cuda_device_match_score(
        model_package=model_package, selected_device=torch.device(type="cuda", index=0)
    )

    # then
    assert result == 1


@mock.patch.object(ranking, "x_ray_runtime_environment")
def test_retrieve_cuda_device_match_score_when_selected_device_matches_for_server(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value.gpu_devices = ["tesla-t4", "nvidia-l4"]
    x_ray_runtime_environment_mock.return_value.gpu_devices_cc = [
        Version("7.5"),
        Version("8.7"),
    ]
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(same_cc_compatible=False),
        environment_requirements=ServerEnvironmentRequirements(
            cuda_device_cc=Version("7.5"),
            cuda_device_name="tesla-t4",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.13"),
            os_version="ubuntu-20.04",
            trt_version=Version("10.3.0.1"),
        ),
    )

    # when
    result = retrieve_cuda_device_match_score(
        model_package=model_package, selected_device=torch.device(type="cuda", index=0)
    )

    # then
    assert result == 1


@mock.patch.object(ranking, "x_ray_runtime_environment")
def test_retrieve_cuda_device_match_score_when_selected_device_does_not_match_for_jetson(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value.gpu_devices = ["orin"]
    x_ray_runtime_environment_mock.return_value.gpu_devices_cc = [Version("8.9")]
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(same_cc_compatible=False),
        environment_requirements=JetsonEnvironmentRequirements(
            cuda_device_cc=Version("8.9"),
            cuda_device_name="orin",
            jetson_product_name="jetson-orin-nx",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.6"),
            l4t_version=Version("36.4.0"),
            trt_version=Version("10.3.0.3"),
        ),
    )

    # when
    result = retrieve_cuda_device_match_score(
        model_package=model_package, selected_device=torch.device(type="cpu")
    )

    # then
    assert result == 0


@mock.patch.object(ranking, "x_ray_runtime_environment")
def test_retrieve_cuda_device_match_score_when_selected_device_does_not_match_for_server(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value.gpu_devices = ["tesla-t4", "nvidia-l4"]
    x_ray_runtime_environment_mock.return_value.gpu_devices_cc = [
        Version("7.5"),
        Version("8.7"),
    ]
    model_package = ModelPackageMetadata(
        package_id="my-package-id-2",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(same_cc_compatible=False),
        environment_requirements=ServerEnvironmentRequirements(
            cuda_device_cc=Version("7.5"),
            cuda_device_name="tesla-t4",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.13"),
            os_version="ubuntu-20.04",
            trt_version=Version("10.3.0.1"),
        ),
    )

    # when
    result = retrieve_cuda_device_match_score(
        model_package=model_package, selected_device=torch.device(type="cuda", index=1)
    )

    # then
    assert result == 0


def test_rank_packages_ids() -> None:
    # given
    model_packages = [
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
    ]

    # when
    result = rank_packages_ids(model_packages=model_packages)

    # then
    assert result == [1, 0]


@pytest.mark.parametrize(
    "nms_fusion_preferences", [None, True, False, {"max_detections": (100, 200)}]
)
def test_retrieve_fused_nms_rank_when_no_model_features_declared(
    nms_fusion_preferences: Union[bool, dict, None]
) -> None:
    # given
    model_package = ModelPackageMetadata(
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
    )

    # when
    result = retrieve_fused_nms_rank(
        model_package=model_package,
        nms_fusion_preferences=nms_fusion_preferences,
    )

    # then
    assert result == 0


@pytest.mark.parametrize(
    "nms_fusion_preferences", [None, True, False, {"max_detections": (100, 200)}]
)
def test_retrieve_fused_nms_rank_when_model_features_declared_but_without_nsm_fused(
    nms_fusion_preferences: Union[bool, dict, None]
) -> None:
    # given
    model_package = ModelPackageMetadata(
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
        model_features={},
    )

    # when
    result = retrieve_fused_nms_rank(
        model_package=model_package, nms_fusion_preferences=nms_fusion_preferences
    )

    # then
    assert result == 0


@pytest.mark.parametrize("nms_fusion_preferences", [None, False])
def test_retrieve_fused_nms_rank_when_model_features_declared_but_with_nms_fused_turned_on_and_no_nms_preferences(
    nms_fusion_preferences: Union[bool, dict, None]
) -> None:
    # given
    model_package = ModelPackageMetadata(
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
        model_features={
            "nms_fused": {
                "max_detections": 300,
                "confidence_threshold": 0.4,
                "iou_threshold": 0.7,
                "class_agnostic": True,
            },
        },
    )

    # when
    result = retrieve_fused_nms_rank(
        model_package=model_package,
        nms_fusion_preferences=nms_fusion_preferences,
    )

    # then
    assert result == 0


@pytest.mark.parametrize(
    "nms_fusion_preferences",
    [
        {"max_detections": 500},
        {"max_detections": (400, 600)},
        {"confidence_threshold": 0.5},
        {"confidence_threshold": (0.45, 0.65)},
        {"iou_threshold": 0.5},
        {"iou_threshold": (0.3, 0.5)},
        {"class_agnostic": False},
        {
            "max_detections": 500,
            "confidence_threshold": (0.45, 0.65),
            "iou_threshold": (0.3, 0.5),
            "class_agnostic": False,
        },
    ],
)
def test_retrieve_fused_nms_rank_when_model_features_declared_but_with_nms_fused_turned_on_nms_preferences_not_matching(
    nms_fusion_preferences: Union[bool, dict, None]
) -> None:
    # given
    model_package = ModelPackageMetadata(
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
        model_features={
            "nms_fused": {
                "max_detections": 300,
                "confidence_threshold": 0.4,
                "iou_threshold": 0.7,
                "class_agnostic": True,
            },
        },
    )

    # when
    result = retrieve_fused_nms_rank(
        model_package=model_package,
        nms_fusion_preferences=nms_fusion_preferences,
    )

    # then
    assert result == 0


@pytest.mark.parametrize(
    "nms_fusion_preferences, expected_score",
    [
        ({"max_detections": 300}, 1.0),
        ({"max_detections": (200, 400)}, 1.0),
        ({"max_detections": (300, 500)}, 0.5),
        ({"max_detections": (300, 600)}, 0.5),
        ({"max_detections": (100, 600)}, 0.9),
        ({"max_detections": (0, 600)}, 1.0),
        ({"max_detections": (0, 1200)}, 0.75),
        ({"confidence_threshold": 0.4}, 1.0),
        ({"confidence_threshold": (0.3, 0.5)}, 1.0),
        ({"confidence_threshold": (0.0, 0.8)}, 1.0),
        ({"confidence_threshold": (0.0, 1.0)}, 0.9),
        ({"iou_threshold": 0.7}, 1.0),
        ({"iou_threshold": (0.6, 0.8)}, 1.0),
        ({"iou_threshold": (0.5, 0.9)}, 1.0),
        ({"iou_threshold": (0.7, 1.0)}, 0.5),
        ({"class_agnostic": True}, 1.0),
        (
            {
                "max_detections": (300, 500),
                "confidence_threshold": (0.0, 1.0),
                "iou_threshold": (0.7, 1.0),
            },
            1.9,
        ),
        (
            {
                "max_detections": (300, 500),
                "confidence_threshold": (0.0, 1.0),
                "iou_threshold": (0.7, 1.0),
                "class_agnostic": True,
            },
            2.9,
        ),
    ],
)
def test_retrieve_fused_nms_rank_when_model_features_declared_but_with_nms_fused_turned_on_nms_preferences_matching(
    nms_fusion_preferences: Union[bool, dict, None],
    expected_score: float,
) -> None:
    # given
    model_package = ModelPackageMetadata(
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
        model_features={
            "nms_fused": {
                "max_detections": 300,
                "confidence_threshold": 0.4,
                "iou_threshold": 0.7,
                "class_agnostic": True,
            },
        },
    )

    # when
    result = retrieve_fused_nms_rank(
        model_package=model_package,
        nms_fusion_preferences=nms_fusion_preferences,
    )

    # then
    assert abs(result - expected_score) < 1e-5


def test_rank_model_packages_when_package_id_should_be_ordered_correctly() -> None:
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
                max_dynamic_batch_size=32,
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
    assert [r.package_id for r in result] == ["my-package-id-2", "my-package-id-1"]


def test_rank_model_packages_when_nms_fused_should_be_ordered_correctly_when_nms_preferred() -> (
    None
):
    # given
    model_packages = [
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
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            trt_package_details=TRTPackageDetails(
                min_dynamic_batch_size=1,
                opt_dynamic_batch_size=8,
                max_dynamic_batch_size=32,
            ),
            model_features={
                "nms_fused": {
                    "max_detections": 300,
                    "confidence_threshold": 0.4,
                    "iou_threshold": 0.7,
                    "class_agnostic": True,
                },
            },
        ),
    ]

    # when
    result = rank_model_packages(
        model_packages=model_packages,
        nms_fusion_preferences=True,
    )

    # then
    assert [r.package_id for r in result] == ["my-package-id-1", "my-package-id-2"]


def test_rank_model_packages_when_nms_fused_should_be_ordered_correctly_when_nms_not_preferred() -> (
    None
):
    # given
    model_packages = [
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
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.TRT,
            quantization=Quantization.FP32,
            dynamic_batch_size_supported=True,
            package_artefacts=[],
            trt_package_details=TRTPackageDetails(
                min_dynamic_batch_size=1,
                opt_dynamic_batch_size=8,
                max_dynamic_batch_size=32,
            ),
            model_features={
                "nms_fused": {
                    "max_detections": 300,
                    "confidence_threshold": 0.4,
                    "iou_threshold": 0.7,
                    "class_agnostic": True,
                },
            },
        ),
    ]

    # when
    result = rank_model_packages(
        model_packages=model_packages,
        nms_fusion_preferences=False,
    )

    # then
    assert [r.package_id for r in result] == ["my-package-id-2", "my-package-id-1"]
