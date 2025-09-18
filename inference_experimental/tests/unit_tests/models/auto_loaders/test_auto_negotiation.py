from typing import Tuple
from unittest import mock
from unittest.mock import MagicMock

import pytest
import torch
from inference_exp.errors import (
    AmbiguousModelPackageResolutionError,
    InvalidRequestedBatchSizeError,
    ModelPackageNegotiationError,
    NoModelPackagesAvailableError,
    UnknownBackendTypeError,
    UnknownQuantizationError,
)
from inference_exp.models.auto_loaders import auto_negotiation
from inference_exp.models.auto_loaders.auto_negotiation import (
    determine_default_allowed_quantization,
    filter_model_packages_based_on_model_features,
    filter_model_packages_by_requested_batch_size,
    filter_model_packages_by_requested_quantization,
    hf_transformers_package_matches_runtime_environment,
    model_package_matches_batch_size_request,
    model_package_matches_runtime_environment,
    onnx_package_matches_runtime_environment,
    parse_backend_type,
    parse_batch_size,
    parse_quantization,
    parse_requested_quantization,
    range_within_other,
    remove_packages_not_matching_implementation,
    remove_untrusted_packages,
    select_model_package_by_id,
    torch_package_matches_runtime_environment,
    torch_script_package_matches_runtime_environment,
    trt_package_matches_runtime_environment,
    ultralytics_package_matches_runtime_environment,
    verify_trt_package_compatibility_with_cuda_device,
    verify_versions_up_to_major_and_minor,
)
from inference_exp.runtime_introspection.core import RuntimeXRayResult
from inference_exp.weights_providers.entities import (
    BackendType,
    FileDownloadSpecs,
    JetsonEnvironmentRequirements,
    ModelPackageMetadata,
    ONNXPackageDetails,
    Quantization,
    ServerEnvironmentRequirements,
    TorchScriptPackageDetails,
    TRTPackageDetails,
)
from packaging.version import Version


def test_parse_quantization_when_quantization_recognized() -> None:
    # when
    result = parse_quantization(value="fp16")

    # then
    assert result is Quantization.FP16


def test_parse_quantization_when_quantization_not_recognized() -> None:
    # when
    with pytest.raises(UnknownQuantizationError):
        _ = parse_quantization(value="invalid")


def test_parse_requested_quantization_when_single_enum_provided() -> None:
    # when
    result = parse_requested_quantization(value=Quantization.BF16)

    # then
    assert result == {Quantization.BF16}


def test_parse_requested_quantization_when_single_string_provided() -> None:
    # when
    result = parse_requested_quantization(value="fp32")

    # then
    assert result == {Quantization.FP32}


def test_parse_requested_quantization_when_multiple_options_provided() -> None:
    # when
    result = parse_requested_quantization(value=[Quantization.BF16, "bf16", "fp16"])

    # then
    assert result == {Quantization.BF16, Quantization.FP16}


def test_parse_backend_type_when_recognized_backend_provided() -> None:
    # when
    result = parse_backend_type(value="trt")

    # then
    assert result is BackendType.TRT


def test_parse_backend_type_when_not_recognized_backend_provided() -> None:
    # when
    with pytest.raises(UnknownBackendTypeError):
        _ = parse_backend_type(value="unknown")


def test_parse_batch_size_when_single_value_provided() -> None:
    # when
    result = parse_batch_size(requested_batch_size=8)

    # then
    assert result == (8, 8)


def test_parse_batch_size_when_single_value_provided_but_invalid() -> None:
    # when
    with pytest.raises(InvalidRequestedBatchSizeError):
        _ = parse_batch_size(requested_batch_size=0)


def test_parse_batch_size_when_single_value_provided_but_not_a_number() -> None:
    # when
    with pytest.raises(InvalidRequestedBatchSizeError):
        _ = parse_batch_size(requested_batch_size="some")


def test_parse_batch_size_when_valid_tuple_provided() -> None:
    # when
    result = parse_batch_size(requested_batch_size=(8, 16))

    # then
    assert result == (8, 16)


def test_parse_batch_size_when_not_a_tuple_provided() -> None:
    # when
    with pytest.raises(InvalidRequestedBatchSizeError):
        _ = parse_batch_size(requested_batch_size=[8, 16])


def test_parse_batch_size_when_tuple_of_invalid_values_provided() -> None:
    # when
    with pytest.raises(InvalidRequestedBatchSizeError):
        _ = parse_batch_size(requested_batch_size=(8, "16"))


def test_parse_batch_size_when_to_small_min_value_provided() -> None:
    # when
    with pytest.raises(InvalidRequestedBatchSizeError):
        _ = parse_batch_size(requested_batch_size=(0, 16))


def test_parse_batch_size_when_max_lower_than_min() -> None:
    # when
    with pytest.raises(InvalidRequestedBatchSizeError):
        _ = parse_batch_size(requested_batch_size=(16, 8))


def test_parse_batch_size_when_the_same_valid_number_provided_as_min_and_max() -> None:
    # when
    result = parse_batch_size(requested_batch_size=(8, 8))

    # then
    assert result == (8, 8)


def test_parse_batch_size_when_different_numbers_provided() -> None:
    # when
    result = parse_batch_size(requested_batch_size=(1, 8))

    # then
    assert result == (1, 8)


@pytest.mark.parametrize(
    "external_range, internal_range", [((10, 20), (21, 30)), ((21, 30), (10, 20))]
)
def test_range_within_other_when_ranges_do_not_overlap(
    external_range: Tuple[int, int],
    internal_range: Tuple[int, int],
) -> None:
    # when
    result = range_within_other(
        external_range=external_range, internal_range=internal_range
    )

    # then
    assert result is False


@pytest.mark.parametrize(
    "external_range, internal_range", [((10, 20), (20, 30)), ((20, 30), (10, 20))]
)
def test_range_within_other_when_ranges_do_overlap_only_with_one_edge(
    external_range: Tuple[int, int],
    internal_range: Tuple[int, int],
) -> None:
    # when
    result = range_within_other(
        external_range=external_range, internal_range=internal_range
    )

    # then
    assert result is False


@pytest.mark.parametrize(
    "external_range, internal_range", [((10, 25), (20, 30)), ((20, 30), (10, 25))]
)
def test_range_within_other_when_ranges_intersect_partially(
    external_range: Tuple[int, int],
    internal_range: Tuple[int, int],
) -> None:
    # when
    result = range_within_other(
        external_range=external_range, internal_range=internal_range
    )

    # then
    assert result is False


def test_range_within_other_when_range_inside_other() -> None:
    # when
    result = range_within_other(external_range=(1, 16), internal_range=(1, 8))

    # then
    assert result is True


def test_verify_versions_up_to_major_and_minor_when_versions_match() -> None:
    # when
    result = verify_versions_up_to_major_and_minor(
        x=Version("1.21.40"),
        y=Version("1.21.47"),
    )

    # then
    assert result is True


def test_verify_versions_up_to_major_and_minor_when_versions_do_not_match() -> None:
    # when
    result = verify_versions_up_to_major_and_minor(
        x=Version("1.21.41"),
        y=Version("1.22.40"),
    )

    # then
    assert result is False


def test_verify_trt_package_compatibility_with_cuda_device_when_no_device_selected_and_no_cc_compatibility_and_there_is_a_match() -> (
    None
):
    # when
    result = verify_trt_package_compatibility_with_cuda_device(
        selected_device=None,
        all_available_cuda_devices=["tesla-t4", "nvidia-l4"],
        all_available_devices_cc=[Version("7.5"), Version("8.7")],
        compilation_device="nvidia-l4",
        compilation_device_cc=Version("8.7"),
        trt_compiled_with_cc_compatibility=False,
    )

    # then
    assert result is True


def test_verify_trt_package_compatibility_with_cuda_device_when_no_device_selected_and_no_cc_compatibility_and_there_is_no_match() -> (
    None
):
    # when
    result = verify_trt_package_compatibility_with_cuda_device(
        selected_device=None,
        all_available_cuda_devices=["tesla-t4", "nvidia-l4"],
        all_available_devices_cc=[Version("7.5"), Version("8.7")],
        compilation_device="nvidia-l40",
        compilation_device_cc=Version("8.7"),
        trt_compiled_with_cc_compatibility=False,
    )

    # then
    assert result is False


def test_verify_trt_package_compatibility_with_cuda_device_when_no_device_selected_and_cc_compatibility_and_there_is_no_match() -> (
    None
):
    # when
    result = verify_trt_package_compatibility_with_cuda_device(
        selected_device=None,
        all_available_cuda_devices=["tesla-t4", "nvidia-l4"],
        all_available_devices_cc=[Version("7.5"), Version("8.7")],
        compilation_device="nvidia-l40",
        compilation_device_cc=Version("8.8"),
        trt_compiled_with_cc_compatibility=True,
    )

    # then
    assert result is False


def test_verify_trt_package_compatibility_with_cuda_device_when_no_device_selected_and_cc_compatibility_and_there_is_match() -> (
    None
):
    # when
    result = verify_trt_package_compatibility_with_cuda_device(
        selected_device=None,
        all_available_cuda_devices=["tesla-t4", "nvidia-l4"],
        all_available_devices_cc=[Version("7.5"), Version("8.7")],
        compilation_device="nvidia-l40",
        compilation_device_cc=Version("8.7"),
        trt_compiled_with_cc_compatibility=True,
    )

    # then
    assert result is True


def test_verify_trt_package_compatibility_with_cuda_device_when_cpu_device_selected() -> (
    None
):
    # when
    result = verify_trt_package_compatibility_with_cuda_device(
        selected_device=torch.device("cpu"),
        all_available_cuda_devices=["tesla-t4", "nvidia-l4"],
        all_available_devices_cc=[Version("7.5"), Version("8.7")],
        compilation_device="nvidia-l4",
        compilation_device_cc=Version("8.7"),
        trt_compiled_with_cc_compatibility=True,
    )

    # then
    assert result is False


def test_verify_trt_package_compatibility_with_cuda_device_when_strict_matching_device_selected() -> (
    None
):
    # when
    result = verify_trt_package_compatibility_with_cuda_device(
        selected_device=torch.device(type="cuda", index=1),
        all_available_cuda_devices=["tesla-t4", "nvidia-l4"],
        all_available_devices_cc=[Version("7.5"), Version("8.7")],
        compilation_device="nvidia-l4",
        compilation_device_cc=Version("8.7"),
        trt_compiled_with_cc_compatibility=False,
    )

    # then
    assert result is True


def test_verify_trt_package_compatibility_with_cuda_device_when_cc_matching_device_selected_without_cc_compatibility() -> (
    None
):
    # when
    result = verify_trt_package_compatibility_with_cuda_device(
        selected_device=torch.device(type="cuda", index=1),
        all_available_cuda_devices=["tesla-t4", "nvidia-l4"],
        all_available_devices_cc=[Version("7.5"), Version("8.7")],
        compilation_device="nvidia-l40",
        compilation_device_cc=Version("8.7"),
        trt_compiled_with_cc_compatibility=False,
    )

    # then
    assert result is False


def test_verify_trt_package_compatibility_with_cuda_device_when_cc_matching_device_selected_with_cc_compatibility() -> (
    None
):
    # when
    result = verify_trt_package_compatibility_with_cuda_device(
        selected_device=torch.device(type="cuda", index=1),
        all_available_cuda_devices=["tesla-t4", "nvidia-l4"],
        all_available_devices_cc=[Version("7.5"), Version("8.7")],
        compilation_device="nvidia-l40",
        compilation_device_cc=Version("8.7"),
        trt_compiled_with_cc_compatibility=True,
    )

    # then
    assert result is True


def test_verify_trt_package_compatibility_with_cuda_device_when_not_matching_device_selected() -> (
    None
):
    # when
    result = verify_trt_package_compatibility_with_cuda_device(
        selected_device=torch.device(type="cuda", index=0),
        all_available_cuda_devices=["tesla-t4", "nvidia-l4"],
        all_available_devices_cc=[Version("7.5"), Version("8.7")],
        compilation_device="nvidia-l4",
        compilation_device_cc=Version("8.7"),
        trt_compiled_with_cc_compatibility=True,
    )

    # then
    assert result is False


def test_trt_package_matches_runtime_environment_when_trt_not_detected_in_env() -> None:
    # given
    model_package = ModelPackageMetadata(
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
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["nvidia-l4"],
        gpu_devices_cc=[Version("8.7")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=None,
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
        trt_python_package_available=False,
    )

    # when
    result, reason = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
    )

    # then
    assert result is False
    assert reason is not None


def test_trt_package_matches_runtime_environment_when_trt_python_package_not_detected_in_env() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
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
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["nvidia-l4"],
        gpu_devices_cc=[Version("8.7")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.3.0.11"),
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
        trt_python_package_available=False,
    )

    # when
    result, reason = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
    )

    # then
    assert result is False
    assert reason is not None


def test_trt_package_matches_runtime_environment_when_environment_requirements_not_manifested() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["nvidia-l4"],
        gpu_devices_cc=[Version("8.7")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.3.0.11"),
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
    )

    # when
    result, reason = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
    )

    # then
    assert result is False
    assert reason is not None


def test_trt_package_matches_runtime_for_jetson_when_trt_version_not_declared_in_runtime_env() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
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
            trt_version=None,
        ),
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["nvidia-l4"],
        gpu_devices_cc=[Version("8.7")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.3.0.11"),
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
    )

    # when
    result, reason = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
    )

    # then
    assert result is False
    assert reason is not None


def test_trt_package_matches_runtime_for_jetson_when_cpu_device_declared() -> None:
    # given
    model_package = ModelPackageMetadata(
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
            trt_version=Version("10.3.0.11-1"),
        ),
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["nvidia-l4"],
        gpu_devices_cc=[Version("8.7")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.3.0.11"),
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
    )

    # when
    result, reason = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
        device=torch.device("cpu"),
    )

    # then
    assert result is False
    assert reason is not None


def test_trt_package_matches_runtime_for_jetson_when_device_not_declared_but_does_not_match_available_by_type() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        environment_requirements=JetsonEnvironmentRequirements(
            cuda_device_cc=Version("8.9"),
            cuda_device_name="orin-super",
            jetson_product_name="jetson-orin-nx",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.6"),
            l4t_version=Version("36.4.0"),
            trt_version=Version("10.3.0.11-1"),
        ),
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["orin"],
        gpu_devices_cc=[Version("8.9")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.3.0.11"),
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
    )

    # when
    result, reason = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
    )

    # then
    assert result is False
    assert reason is not None


def test_trt_package_matches_runtime_for_jetson_when_device_not_declared_but_does_not_match_available_by_type_but_matches_through_cc_and_cc_compatibility_enabled() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(same_cc_compatible=True),
        environment_requirements=JetsonEnvironmentRequirements(
            cuda_device_cc=Version("8.9"),
            cuda_device_name="orin-super",
            jetson_product_name="jetson-orin-nx",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.6"),
            l4t_version=Version("36.4.0"),
            trt_version=Version("10.3.0.11-1"),
        ),
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["orin"],
        gpu_devices_cc=[Version("8.9")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.3.0.11-1"),
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
    )

    # when
    result, reason = trt_package_matches_runtime_environment(
        model_package=model_package, runtime_x_ray=runtime_x_ray, verbose=True
    )

    # then
    assert result is True
    assert reason is None


def test_trt_package_matches_runtime_for_jetson_when_trt_versions_missmatch() -> None:
    # given
    model_package = ModelPackageMetadata(
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
            trt_version=Version("10.3.0.11-1"),
        ),
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["orin"],
        gpu_devices_cc=[Version("8.9")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.2.0.11"),
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
    )

    # when
    result, reason = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
    )

    # then
    assert result is False
    assert reason is not None


def test_trt_package_matches_runtime_for_jetson_when_l4t_versions_missmatch() -> None:
    # given
    model_package = ModelPackageMetadata(
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
            trt_version=Version("10.3.0.11-1"),
        ),
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["orin"],
        gpu_devices_cc=[Version("8.9")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.3.0.11"),
        jetson_type=None,
        l4t_version=Version("36.0.0"),
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
    )

    # when
    result, reason = trt_package_matches_runtime_environment(
        model_package=model_package, runtime_x_ray=runtime_x_ray, verbose=True
    )

    # then
    assert result is False
    assert reason is not None


def test_trt_package_matches_runtime_for_jetson_when_trt_versions_missmatch_despite_forward_capability() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(trt_forward_compatible=True),
        environment_requirements=JetsonEnvironmentRequirements(
            cuda_device_cc=Version("8.9"),
            cuda_device_name="orin",
            jetson_product_name="jetson-orin-nx",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.6"),
            l4t_version=Version("36.4.3"),
            trt_version=Version("10.3.0.11-1"),
        ),
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["orin"],
        gpu_devices_cc=[Version("8.9")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.2.0.11"),
        jetson_type=None,
        l4t_version=Version("36.4.3"),
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
    )

    # when
    result, reason = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
    )

    # then
    assert result is False
    assert reason is not None


def test_trt_package_matches_runtime_for_jetson_when_trt_versions_missmatch_but_forward_capability_saves_the_day() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(trt_forward_compatible=True),
        environment_requirements=JetsonEnvironmentRequirements(
            cuda_device_cc=Version("8.9"),
            cuda_device_name="orin",
            jetson_product_name="jetson-orin-nx",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.6"),
            l4t_version=Version("36.4.3"),
            trt_version=Version("10.3.0.11-1"),
        ),
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["orin"],
        gpu_devices_cc=[Version("8.9")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.0.11"),
        jetson_type=None,
        l4t_version=Version("36.4.3"),
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
    )

    # when
    result, reason = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
    )

    # then
    assert result is True


def test_trt_package_matches_runtime_for_jetson_when_trt_versions_missmatch_forward_capability_enabled_but_with_lean_runtime_excluded() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(
            trt_forward_compatible=True, trt_lean_runtime_excluded=True
        ),
        environment_requirements=JetsonEnvironmentRequirements(
            cuda_device_cc=Version("8.9"),
            cuda_device_name="orin",
            jetson_product_name="jetson-orin-nx",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.6"),
            l4t_version=Version("36.4.3"),
            trt_version=Version("10.3.0.11-1"),
        ),
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["orin"],
        gpu_devices_cc=[Version("8.9")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.0.11"),
        jetson_type=None,
        l4t_version=Version("36.4.3"),
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
    )

    # when
    result, reason = trt_package_matches_runtime_environment(
        model_package=model_package, runtime_x_ray=runtime_x_ray, verbose=True
    )

    # then
    assert result is False
    assert reason is not None


def test_trt_package_matches_runtime_for_jetson_when_trt_versions_missmatch_with_forward_capability_mode_but_without_permission_to_execute_host_code() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(trt_forward_compatible=True),
        environment_requirements=JetsonEnvironmentRequirements(
            cuda_device_cc=Version("8.9"),
            cuda_device_name="orin",
            jetson_product_name="jetson-orin-nx",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.6"),
            l4t_version=Version("36.4.3"),
            trt_version=Version("10.3.0.11"),
        ),
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["orin"],
        gpu_devices_cc=[Version("8.9")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.0.11-1"),
        jetson_type=None,
        l4t_version=Version("36.4.3"),
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
    )

    # when
    result, reason = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
        trt_engine_host_code_allowed=False,
    )

    # then
    assert result is False
    assert reason is not None


def test_trt_package_matches_runtime_when_unknown_environment_requirements_declared() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(trt_forward_compatible=False),
        environment_requirements=MagicMock(),
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["orin"],
        gpu_devices_cc=[Version("8.9")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.0.11"),
        jetson_type=None,
        l4t_version=Version("36.4.3"),
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
    )

    # when
    with pytest.raises(ModelPackageNegotiationError):
        _ = trt_package_matches_runtime_environment(
            model_package=model_package, runtime_x_ray=runtime_x_ray, verbose=True
        )


def test_trt_package_matches_runtime_for_server_when_trt_version_not_declared() -> None:
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(trt_forward_compatible=False),
        environment_requirements=ServerEnvironmentRequirements(
            cuda_device_cc=Version("7.5"),
            cuda_device_name="tesla-t4",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.4"),
            os_version="ubuntu-20.04",
            trt_version=None,
        ),
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["tesla-t4"],
        gpu_devices_cc=[Version("7.5")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.0.11"),
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
    )

    # when
    result, reason = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
    )

    # then
    assert result is False
    assert reason is not None


def test_trt_package_matches_runtime_environment_for_server_when_package_excluded_by_device_incompatibility() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(trt_forward_compatible=False),
        environment_requirements=ServerEnvironmentRequirements(
            cuda_device_cc=Version("7.5"),
            cuda_device_name="tesla-t4",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.4"),
            os_version="ubuntu-20.04",
            trt_version=Version("10.5.0.11"),
        ),
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["nvidia-l4"],
        gpu_devices_cc=[Version("8.7")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.0.11"),
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
    )

    # when
    result, reason = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
    )

    # then
    assert result is False
    assert reason is not None


def test_trt_package_matches_runtime_environment_for_server_when_package_excluded_by_choosing_cpu_device() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(trt_forward_compatible=False),
        environment_requirements=ServerEnvironmentRequirements(
            cuda_device_cc=Version("7.5"),
            cuda_device_name="tesla-t4",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.4"),
            os_version="ubuntu-20.04",
            trt_version=Version("10.5.0.11"),
        ),
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["tesla-t4"],
        gpu_devices_cc=[Version("7.5")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.0.11"),
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
    )

    # when
    result, reason = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
        device=torch.device("cpu"),
    )

    # then
    assert result is False
    assert reason is not None


def test_trt_package_matches_runtime_environment_for_server_when_selected_device_compatible() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(trt_forward_compatible=False),
        environment_requirements=ServerEnvironmentRequirements(
            cuda_device_cc=Version("7.5"),
            cuda_device_name="tesla-t4",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.4"),
            os_version="ubuntu-20.04",
            trt_version=Version("10.5.0.11"),
        ),
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["tesla-t4"],
        gpu_devices_cc=[Version("7.5")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.0.11"),
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
    )

    # when
    result, reason = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
        device=torch.device(type="cuda", index=0),
    )

    # then
    assert result is True
    assert reason is None


def test_trt_package_matches_runtime_environment_for_server_when_selected_device_compatible_when_cc_compatibility_saves_the_day() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(same_cc_compatible=True),
        environment_requirements=ServerEnvironmentRequirements(
            cuda_device_cc=Version("7.5"),
            cuda_device_name="tesla-t44",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.4"),
            os_version="ubuntu-20.04",
            trt_version=Version("10.5.0.11"),
        ),
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["tesla-t4"],
        gpu_devices_cc=[Version("7.5")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.0.11"),
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
    )

    # when
    result, reason = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
        device=torch.device(type="cuda", index=0),
    )

    # then
    assert result is True
    assert reason is None


def test_trt_package_matches_runtime_environment_for_server_when_selected_device_compatible_when_cc_compatibility_would_save_the_day_but_disabled() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(same_cc_compatible=False),
        environment_requirements=ServerEnvironmentRequirements(
            cuda_device_cc=Version("7.5"),
            cuda_device_name="tesla-t44",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.4"),
            os_version="ubuntu-20.04",
            trt_version=Version("10.5.0.11"),
        ),
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["tesla-t4"],
        gpu_devices_cc=[Version("7.5")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.0.11"),
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
    )

    # when
    result, reason = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
        device=torch.device(type="cuda", index=0),
    )

    # then
    assert result is False
    assert reason is not None


def test_trt_package_matches_runtime_environment_for_server_when_no_trt_forward_compatibility_but_trt_version_missmatch() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(trt_forward_compatible=False),
        environment_requirements=ServerEnvironmentRequirements(
            cuda_device_cc=Version("7.5"),
            cuda_device_name="tesla-t4",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.4"),
            os_version="ubuntu-20.04",
            trt_version=Version("10.5.1.11"),
        ),
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["tesla-t4"],
        gpu_devices_cc=[Version("7.5")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.0.11"),
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
    )

    # when
    result, reason = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
        device=torch.device(type="cuda", index=0),
    )

    # then
    assert result is False
    assert reason is not None


def test_trt_package_matches_runtime_environment_for_server_when_trt_forward_compatibility_missmatch() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(trt_forward_compatible=True),
        environment_requirements=ServerEnvironmentRequirements(
            cuda_device_cc=Version("7.5"),
            cuda_device_name="tesla-t4",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.4"),
            os_version="ubuntu-20.04",
            trt_version=Version("10.5.1.11"),
        ),
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["tesla-t4"],
        gpu_devices_cc=[Version("7.5")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.0.11"),
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
    )

    # when
    result, reason = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
        device=torch.device(type="cuda", index=0),
        verbose=True,
    )

    # then
    assert result is False
    assert reason is not None


def test_trt_package_matches_runtime_environment_for_server_when_trt_forward_compatibility_saves_the_day() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(trt_forward_compatible=True),
        environment_requirements=ServerEnvironmentRequirements(
            cuda_device_cc=Version("7.5"),
            cuda_device_name="tesla-t4",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.4"),
            os_version="ubuntu-20.04",
            trt_version=Version("10.5.0.0"),
        ),
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["tesla-t4"],
        gpu_devices_cc=[Version("7.5")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.1.11"),
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
    )

    # when
    result, reason = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
        device=torch.device(type="cuda", index=0),
    )

    # then
    assert result is True
    assert reason is None


def test_trt_package_matches_runtime_environment_for_server_when_trt_forward_compatibility_enabled_but_lean_runtime_excluded() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(
            trt_forward_compatible=True, trt_lean_runtime_excluded=True
        ),
        environment_requirements=ServerEnvironmentRequirements(
            cuda_device_cc=Version("7.5"),
            cuda_device_name="tesla-t4",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.4"),
            os_version="ubuntu-20.04",
            trt_version=Version("10.5.0.0"),
        ),
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["tesla-t4"],
        gpu_devices_cc=[Version("7.5")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.1.11"),
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
    )

    # when
    result, reason = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
        device=torch.device(type="cuda", index=0),
    )

    # then
    assert result is False
    assert reason is not None


def test_trt_package_matches_runtime_environment_for_server_when_trt_forward_compatibility_enabled_but_unsafe_deserialisation_prevented() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.TRT,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        package_artefacts=[],
        trt_package_details=TRTPackageDetails(trt_forward_compatible=True),
        environment_requirements=ServerEnvironmentRequirements(
            cuda_device_cc=Version("7.5"),
            cuda_device_name="tesla-t4",
            cuda_version=Version("12.6"),
            driver_version=Version("510.0.4"),
            os_version="ubuntu-20.04",
            trt_version=Version("10.5.0.0"),
        ),
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["tesla-t4"],
        gpu_devices_cc=[Version("7.5")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.1.11"),
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
    )

    # when
    result, reason = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
        device=torch.device(type="cuda", index=0),
        trt_engine_host_code_allowed=False,
    )

    # then
    assert result is False
    assert reason is not None


def test_ultralytics_package_matches_runtime_environment_when_ultralytics_not_available() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.ULTRALYTICS,
        quantization=Quantization.FP32,
        package_artefacts=[],
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["tesla-t4"],
        gpu_devices_cc=[Version("7.5")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.1.11"),
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
        ultralytics_available=False,
        trt_python_package_available=True,
    )

    # when
    result, reason = ultralytics_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
    )

    # then
    assert result is False
    assert reason is not None


def test_ultralytics_package_matches_runtime_environment_when_ultralytics_available() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.ULTRALYTICS,
        quantization=Quantization.FP32,
        package_artefacts=[],
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["tesla-t4"],
        gpu_devices_cc=[Version("7.5")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.1.11"),
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
    )

    # when
    result, reason = ultralytics_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
    )

    # then
    assert result is True
    assert reason is None


def test_hf_transformers_package_matches_runtime_environment_when_ultralytics_not_available() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.ULTRALYTICS,
        quantization=Quantization.FP32,
        package_artefacts=[],
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["tesla-t4"],
        gpu_devices_cc=[Version("7.5")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.1.11"),
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
        hf_transformers_available=False,
        ultralytics_available=False,
        trt_python_package_available=True,
    )

    # when
    result, reason = hf_transformers_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
    )

    # then
    assert result is False
    assert reason is not None


def test_hf_transformers_package_matches_runtime_environment_when_ultralytics_available() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.ULTRALYTICS,
        quantization=Quantization.FP32,
        package_artefacts=[],
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["tesla-t4"],
        gpu_devices_cc=[Version("7.5")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.1.11"),
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
    )

    # when
    result, reason = hf_transformers_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
    )

    # then
    assert result is True
    assert reason is None


def test_torch_package_matches_runtime_environment_when_ultralytics_not_available() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.ULTRALYTICS,
        quantization=Quantization.FP32,
        package_artefacts=[],
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["tesla-t4"],
        gpu_devices_cc=[Version("7.5")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.1.11"),
        jetson_type=None,
        l4t_version=None,
        os_version="ubuntu-20.04",
        torch_available=False,
        torch_version=Version("2.7.0"),
        torchvision_version=Version("0.22.0"),
        onnxruntime_version=Version("1.21.0"),
        available_onnx_execution_providers={
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        },
        hf_transformers_available=False,
        ultralytics_available=False,
        trt_python_package_available=True,
    )

    # when
    result, reason = torch_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
    )

    # then
    assert result is False
    assert reason is not None


def test_torch_package_matches_runtime_environment_when_ultralytics_available() -> None:
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.ULTRALYTICS,
        quantization=Quantization.FP32,
        package_artefacts=[],
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["tesla-t4"],
        gpu_devices_cc=[Version("7.5")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.1.11"),
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
    )

    # when
    result, reason = torch_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
    )

    # then
    assert result is True
    assert reason is None


def test_onnx_package_matches_runtime_environment_when_onnx_not_detected_in_environment() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
        onnx_package_details=ONNXPackageDetails(opset=19),
        package_artefacts=[],
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["tesla-t4"],
        gpu_devices_cc=[Version("7.5")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.1.11"),
        jetson_type=None,
        l4t_version=None,
        os_version="ubuntu-20.04",
        torch_available=False,
        torch_version=None,
        torchvision_version=None,
        onnxruntime_version=None,
        available_onnx_execution_providers={
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        },
        hf_transformers_available=False,
        ultralytics_available=False,
        trt_python_package_available=True,
    )

    # when
    result, reason = onnx_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
    )

    # then
    assert result is False
    assert reason is not None


def test_onnx_package_matches_runtime_environment_when_no_available_onnx_ep() -> None:
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
        onnx_package_details=ONNXPackageDetails(opset=19),
        package_artefacts=[],
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["tesla-t4"],
        gpu_devices_cc=[Version("7.5")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.1.11"),
        jetson_type=None,
        l4t_version=None,
        os_version="ubuntu-20.04",
        torch_available=False,
        torch_version=None,
        torchvision_version=None,
        onnxruntime_version=Version("1.15.0"),
        available_onnx_execution_providers=None,
        hf_transformers_available=False,
        ultralytics_available=False,
        trt_python_package_available=True,
    )

    # when
    result, reason = onnx_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
    )

    # then
    assert result is False
    assert reason is not None


def test_onnx_package_matches_runtime_environment_when_no_onnx_package_details() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
        onnx_package_details=None,
        package_artefacts=[],
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["tesla-t4"],
        gpu_devices_cc=[Version("7.5")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.1.11"),
        jetson_type=None,
        l4t_version=None,
        os_version="ubuntu-20.04",
        torch_available=False,
        torch_version=None,
        torchvision_version=None,
        onnxruntime_version=Version("1.15.0"),
        available_onnx_execution_providers={"CPUExecutionProvider"},
        hf_transformers_available=False,
        ultralytics_available=False,
        trt_python_package_available=True,
    )

    # when
    result, reason = onnx_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
    )

    # then
    assert result is False
    assert reason is not None


@mock.patch.object(auto_negotiation, "get_selected_onnx_execution_providers")
def test_onnx_package_matches_runtime_environment_when_no_matching_execution_providers_selected_in_env(
    get_selected_onnx_execution_providers_mock: MagicMock,
) -> None:
    # given
    get_selected_onnx_execution_providers_mock.return_value = []
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
        onnx_package_details=ONNXPackageDetails(opset=19),
        package_artefacts=[],
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["tesla-t4"],
        gpu_devices_cc=[Version("7.5")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.1.11"),
        jetson_type=None,
        l4t_version=None,
        os_version="ubuntu-20.04",
        torch_available=False,
        torch_version=None,
        torchvision_version=None,
        onnxruntime_version=Version("1.15.0"),
        available_onnx_execution_providers={"CPUExecutionProvider"},
        hf_transformers_available=False,
        ultralytics_available=False,
        trt_python_package_available=True,
    )

    # when
    result, reason = onnx_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
    )

    # then
    assert result is False
    assert reason is not None


def test_onnx_package_matches_runtime_environment_when_no_matching_execution_providers_selected_in_param() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
        onnx_package_details=ONNXPackageDetails(opset=19),
        package_artefacts=[],
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["tesla-t4"],
        gpu_devices_cc=[Version("7.5")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.1.11"),
        jetson_type=None,
        l4t_version=None,
        os_version="ubuntu-20.04",
        torch_available=False,
        torch_version=None,
        torchvision_version=None,
        onnxruntime_version=Version("1.15.0"),
        available_onnx_execution_providers={"CPUExecutionProvider"},
        hf_transformers_available=False,
        ultralytics_available=False,
        trt_python_package_available=True,
    )

    # when
    result, reason = onnx_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
        onnx_execution_providers=["CUDAExecutionProvider"],
    )

    # then
    assert result is False
    assert reason is not None


def test_onnx_package_matches_runtime_environment_when_no_matching_execution_providers_when_incompatible_providers_taken_into_account() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
        onnx_package_details=ONNXPackageDetails(
            opset=19, incompatible_providers=["CUDAExecutionProvider"]
        ),
        package_artefacts=[],
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["tesla-t4"],
        gpu_devices_cc=[Version("7.5")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.1.11"),
        jetson_type=None,
        l4t_version=None,
        os_version="ubuntu-20.04",
        torch_available=False,
        torch_version=None,
        torchvision_version=None,
        onnxruntime_version=Version("1.15.0"),
        available_onnx_execution_providers={
            "CPUExecutionProvider",
            "TensorRTExecutionProvider",
            "CUDAExecutionProvider",
        },
        hf_transformers_available=False,
        ultralytics_available=False,
        trt_python_package_available=True,
    )

    # when
    result, reason = onnx_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
        onnx_execution_providers=["CUDAExecutionProvider", "TensorRTExecutionProvider"],
    )

    # then
    assert result is False
    assert reason is not None


def test_onnx_package_matches_runtime_environment_when_unknown_onnx_version_spotted_and_opset_below_oldest_supported_version() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
        onnx_package_details=ONNXPackageDetails(opset=19),
        package_artefacts=[],
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["tesla-t4"],
        gpu_devices_cc=[Version("7.5")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.1.11"),
        jetson_type=None,
        l4t_version=None,
        os_version="ubuntu-20.04",
        torch_available=False,
        torch_version=None,
        torchvision_version=None,
        onnxruntime_version=Version("1.10.0"),
        available_onnx_execution_providers={
            "CPUExecutionProvider",
            "TensorRTExecutionProvider",
            "CUDAExecutionProvider",
        },
        hf_transformers_available=False,
        ultralytics_available=False,
        trt_python_package_available=True,
    )

    # when
    result, reason = onnx_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
        onnx_execution_providers=["CUDAExecutionProvider", "TensorRTExecutionProvider"],
    )

    # then
    assert result is True
    assert reason is None


def test_onnx_package_matches_runtime_environment_when_unknown_onnx_version_spotted_and_opset_above_oldest_supported_version() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
        onnx_package_details=ONNXPackageDetails(opset=20),
        package_artefacts=[],
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["tesla-t4"],
        gpu_devices_cc=[Version("7.5")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.1.11"),
        jetson_type=None,
        l4t_version=None,
        os_version="ubuntu-20.04",
        torch_available=False,
        torch_version=None,
        torchvision_version=None,
        onnxruntime_version=Version("1.10.0"),
        available_onnx_execution_providers={
            "CPUExecutionProvider",
            "TensorRTExecutionProvider",
            "CUDAExecutionProvider",
        },
        hf_transformers_available=False,
        ultralytics_available=False,
        trt_python_package_available=True,
    )

    # when
    result, reason = onnx_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
        onnx_execution_providers=["CUDAExecutionProvider", "TensorRTExecutionProvider"],
    )

    # then
    assert result is False
    assert reason is not None


def test_onnx_package_matches_runtime_environment_when_opset_matches() -> None:
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
        onnx_package_details=ONNXPackageDetails(opset=23),
        package_artefacts=[],
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["tesla-t4"],
        gpu_devices_cc=[Version("7.5")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.1.11"),
        jetson_type=None,
        l4t_version=None,
        os_version="ubuntu-20.04",
        torch_available=False,
        torch_version=None,
        torchvision_version=None,
        onnxruntime_version=Version("1.22.0"),
        available_onnx_execution_providers={
            "CPUExecutionProvider",
            "TensorRTExecutionProvider",
            "CUDAExecutionProvider",
        },
        hf_transformers_available=False,
        ultralytics_available=False,
        trt_python_package_available=True,
    )

    # when
    result, reason = onnx_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
        onnx_execution_providers=["CUDAExecutionProvider", "TensorRTExecutionProvider"],
    )

    # then
    assert result is True
    assert reason is None


def test_onnx_package_matches_runtime_environment_when_opset_to_high() -> None:
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
        onnx_package_details=ONNXPackageDetails(opset=24),
        package_artefacts=[],
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["tesla-t4"],
        gpu_devices_cc=[Version("7.5")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.1.11"),
        jetson_type=None,
        l4t_version=None,
        os_version="ubuntu-20.04",
        torch_available=False,
        torch_version=None,
        torchvision_version=None,
        onnxruntime_version=Version("1.22.0"),
        available_onnx_execution_providers={
            "CPUExecutionProvider",
            "TensorRTExecutionProvider",
            "CUDAExecutionProvider",
        },
        hf_transformers_available=False,
        ultralytics_available=False,
        trt_python_package_available=True,
    )

    # when
    result, reason = onnx_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
        onnx_execution_providers=["CUDAExecutionProvider", "TensorRTExecutionProvider"],
    )

    # then
    assert result is False
    assert reason is not None


def test_model_package_matches_runtime_environment_when_backend_is_not_registered() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=MagicMock(),
        quantization=Quantization.FP32,
        onnx_package_details=ONNXPackageDetails(opset=24),
        package_artefacts=[],
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["tesla-t4"],
        gpu_devices_cc=[Version("7.5")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.1.11"),
        jetson_type=None,
        l4t_version=None,
        os_version="ubuntu-20.04",
        torch_available=False,
        torch_version=None,
        torchvision_version=None,
        onnxruntime_version=Version("1.22.0"),
        available_onnx_execution_providers={
            "CPUExecutionProvider",
            "TensorRTExecutionProvider",
            "CUDAExecutionProvider",
        },
        hf_transformers_available=False,
        ultralytics_available=False,
        trt_python_package_available=True,
    )

    # when
    with pytest.raises(ModelPackageNegotiationError):
        _ = model_package_matches_runtime_environment(
            model_package=model_package, runtime_x_ray=runtime_x_ray
        )


def test_model_package_matches_runtime_environment_when_package_should_be_allowed() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
        onnx_package_details=ONNXPackageDetails(opset=23),
        package_artefacts=[],
    )
    runtime_x_ray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["tesla-t4"],
        gpu_devices_cc=[Version("7.5")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=Version("10.5.1.11"),
        jetson_type=None,
        l4t_version=None,
        os_version="ubuntu-20.04",
        torch_available=False,
        torch_version=None,
        torchvision_version=None,
        onnxruntime_version=Version("1.22.0"),
        available_onnx_execution_providers={
            "CPUExecutionProvider",
            "TensorRTExecutionProvider",
            "CUDAExecutionProvider",
        },
        hf_transformers_available=False,
        ultralytics_available=False,
        trt_python_package_available=True,
    )

    # when
    result, reason = model_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
        onnx_execution_providers=["CPUExecutionProvider"],
    )

    # then
    assert result is True
    assert reason is None


def test_model_package_matches_batch_size_request_when_static_batch_size_supported_and_size_does_not_match() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
        onnx_package_details=ONNXPackageDetails(opset=23),
        static_batch_size=1,
        package_artefacts=[],
    )

    # when
    result = model_package_matches_batch_size_request(
        model_package=model_package,
        min_batch_size=4,
        max_batch_size=16,
    )

    # then
    assert result is False


def test_model_package_matches_batch_size_request_when_dynamic_batch_size_supported_without_claiming_boundaries() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
        onnx_package_details=ONNXPackageDetails(opset=23),
        dynamic_batch_size_supported=True,
        package_artefacts=[],
    )

    # when
    result = model_package_matches_batch_size_request(
        model_package=model_package,
        min_batch_size=1,
        max_batch_size=10000000,
    )

    # then
    assert result is True


def test_model_package_matches_batch_size_request_when_dynamic_batch_size_supported_claiming_boundaries_which_are_met() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
        onnx_package_details=ONNXPackageDetails(opset=23),
        dynamic_batch_size_supported=True,
        trt_package_details=TRTPackageDetails(
            min_dynamic_batch_size=1,
            opt_dynamic_batch_size=8,
            max_dynamic_batch_size=16,
        ),
        package_artefacts=[],
    )

    # when
    result = model_package_matches_batch_size_request(
        model_package=model_package,
        min_batch_size=1,
        max_batch_size=8,
    )

    # then
    assert result is True


def test_model_package_matches_batch_size_request_when_dynamic_batch_size_supported_claiming_boundaries_which_are_not_met() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id-1",
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
        onnx_package_details=ONNXPackageDetails(opset=23),
        dynamic_batch_size_supported=True,
        trt_package_details=TRTPackageDetails(
            min_dynamic_batch_size=1,
            opt_dynamic_batch_size=8,
            max_dynamic_batch_size=16,
        ),
        package_artefacts=[],
    )

    # when
    result = model_package_matches_batch_size_request(
        model_package=model_package,
        min_batch_size=1,
        max_batch_size=24,
    )

    # then
    assert result is False


def test_filter_model_packages_by_requested_quantization_when_nothing_left() -> None:
    # given
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.ONNX,
            quantization=Quantization.FP32,
            onnx_package_details=ONNXPackageDetails(opset=23),
            package_artefacts=[],
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.ONNX,
            quantization=Quantization.FP16,
            onnx_package_details=ONNXPackageDetails(opset=23),
            package_artefacts=[],
        ),
    ]

    # when
    result, discarded = filter_model_packages_by_requested_quantization(
        model_packages=model_packages,
        requested_quantization="int8",
        default_quantization_used=True,
    )

    # then
    assert len(result) == 0
    assert len(discarded) == 2
    assert discarded[0].package_id == "my-package-id-1"
    assert discarded[1].package_id == "my-package-id-2"


def test_filter_model_packages_by_requested_quantization_when_something_left() -> None:
    # given
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.ONNX,
            quantization=Quantization.FP32,
            onnx_package_details=ONNXPackageDetails(opset=23),
            package_artefacts=[],
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.ONNX,
            quantization=Quantization.FP16,
            onnx_package_details=ONNXPackageDetails(opset=23),
            package_artefacts=[],
        ),
    ]

    # when
    result, discarded = filter_model_packages_by_requested_quantization(
        model_packages=model_packages,
        requested_quantization="fp16",
        default_quantization_used=False,
    )

    # then
    assert len(result) == 1
    assert result[0].package_id == "my-package-id-2"
    assert len(discarded) == 1
    assert discarded[0].package_id == "my-package-id-1"


def test_filter_model_packages_by_requested_batch_size_when_nothing_left() -> None:
    # given
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.ONNX,
            quantization=Quantization.FP32,
            onnx_package_details=ONNXPackageDetails(opset=23),
            static_batch_size=1,
            package_artefacts=[],
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.ONNX,
            quantization=Quantization.FP16,
            onnx_package_details=ONNXPackageDetails(opset=23),
            static_batch_size=1,
            package_artefacts=[],
        ),
    ]

    # when
    result, discarded = filter_model_packages_by_requested_batch_size(
        model_packages=model_packages,
        requested_batch_size=(2, 8),
    )

    # then
    assert len(result) == 0
    assert len(discarded) == 2
    assert discarded[0].package_id == "my-package-id-1"
    assert discarded[1].package_id == "my-package-id-2"


def test_filter_model_packages_by_requested_batch_size_when_something_left() -> None:
    # given
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.ONNX,
            quantization=Quantization.FP32,
            onnx_package_details=ONNXPackageDetails(opset=23),
            static_batch_size=1,
            package_artefacts=[],
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.ONNX,
            quantization=Quantization.FP16,
            onnx_package_details=ONNXPackageDetails(opset=23),
            static_batch_size=4,
            package_artefacts=[],
        ),
    ]

    # when
    result, discarded = filter_model_packages_by_requested_batch_size(
        model_packages=model_packages,
        requested_batch_size=(2, 8),
    )

    # then
    assert len(result) == 1
    assert result[0].package_id == "my-package-id-2"
    assert len(discarded) == 1
    assert discarded[0].package_id == "my-package-id-1"


def test_select_model_package_by_id_when_no_package_matches() -> None:
    # given
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.ONNX,
            quantization=Quantization.FP32,
            onnx_package_details=ONNXPackageDetails(opset=23),
            static_batch_size=1,
            package_artefacts=[],
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.ONNX,
            quantization=Quantization.FP16,
            onnx_package_details=ONNXPackageDetails(opset=23),
            static_batch_size=4,
            package_artefacts=[],
        ),
    ]

    # when
    with pytest.raises(NoModelPackagesAvailableError):
        _ = select_model_package_by_id(
            model_packages=model_packages, requested_model_package_id="invalid"
        )


def test_select_model_package_by_id_when_there_is_ambiguous_match() -> None:
    # given
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.ONNX,
            quantization=Quantization.FP32,
            onnx_package_details=ONNXPackageDetails(opset=23),
            static_batch_size=1,
            package_artefacts=[],
        ),
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.ONNX,
            quantization=Quantization.FP16,
            onnx_package_details=ONNXPackageDetails(opset=23),
            static_batch_size=4,
            package_artefacts=[],
        ),
    ]

    # when
    with pytest.raises(AmbiguousModelPackageResolutionError):
        _ = select_model_package_by_id(
            model_packages=model_packages, requested_model_package_id="my-package-id-1"
        )


def test_select_model_package_by_id_when_there_is_single_match() -> None:
    # given
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.ONNX,
            quantization=Quantization.FP32,
            onnx_package_details=ONNXPackageDetails(opset=23),
            static_batch_size=1,
            package_artefacts=[],
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.ONNX,
            quantization=Quantization.FP16,
            onnx_package_details=ONNXPackageDetails(opset=23),
            static_batch_size=4,
            package_artefacts=[],
        ),
    ]

    # when
    result = select_model_package_by_id(
        model_packages=model_packages, requested_model_package_id="my-package-id-2"
    )

    # then
    assert result.package_id == "my-package-id-2"


def test_remove_untrusted_packages() -> None:
    # given
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.ONNX,
            quantization=Quantization.FP32,
            onnx_package_details=ONNXPackageDetails(opset=23),
            static_batch_size=1,
            package_artefacts=[],
            trusted_source=True,
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.ONNX,
            quantization=Quantization.FP16,
            onnx_package_details=ONNXPackageDetails(opset=23),
            static_batch_size=4,
            package_artefacts=[],
        ),
    ]

    # when
    result, removed = remove_untrusted_packages(
        model_packages=model_packages,
    )

    # then
    assert len(result) == 1
    assert result[0].package_id == "my-package-id-1"
    assert len(removed) == 1
    assert removed[0].package_id == "my-package-id-2"


def test_remove_untrusted_packages_when_nothing_left() -> None:
    # given
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.ONNX,
            quantization=Quantization.FP32,
            onnx_package_details=ONNXPackageDetails(opset=23),
            static_batch_size=1,
            package_artefacts=[],
            trusted_source=False,
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.ONNX,
            quantization=Quantization.FP16,
            onnx_package_details=ONNXPackageDetails(opset=23),
            static_batch_size=4,
            package_artefacts=[],
            trusted_source=False,
        ),
    ]

    # when
    result, removed = remove_untrusted_packages(
        model_packages=model_packages,
    )

    # then
    assert len(result) == 0
    assert len(removed) == 2
    assert removed[0].package_id == "my-package-id-1"
    assert removed[1].package_id == "my-package-id-2"


def test_determine_default_allowed_quantization_for_cpu_device() -> None:
    # given
    determine_default_allowed_quantization.cache_clear()

    try:
        # when
        results = determine_default_allowed_quantization(device=torch.device("cpu"))
    finally:
        determine_default_allowed_quantization.cache_clear()

    # then
    assert set(results) == {
        Quantization.UNKNOWN,
        Quantization.FP32,
        Quantization.BF16,
    }


def test_determine_default_allowed_quantization_for_cuda_device() -> None:
    # given
    determine_default_allowed_quantization.cache_clear()

    try:
        # when
        results = determine_default_allowed_quantization(
            device=torch.device(type="cuda")
        )
    finally:
        determine_default_allowed_quantization.cache_clear()

    # then
    assert set(results) == {
        Quantization.UNKNOWN,
        Quantization.FP32,
        Quantization.FP16,
    }


def test_determine_default_allowed_quantization_for_mps_device() -> None:
    # given
    determine_default_allowed_quantization.cache_clear()

    try:
        # when
        results = determine_default_allowed_quantization(
            device=torch.device(type="mps")
        )
    finally:
        determine_default_allowed_quantization.cache_clear()

    # then
    assert set(results) == {
        Quantization.UNKNOWN,
        Quantization.FP32,
        Quantization.FP16,
    }


@mock.patch.object(auto_negotiation, "x_ray_runtime_environment")
def test_determine_default_allowed_quantization_for_cuda_device_detected_in_runtime(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["nvidia-l4"],
        gpu_devices_cc=[Version("8.7")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=None,
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
        trt_python_package_available=False,
    )
    determine_default_allowed_quantization.cache_clear()

    try:
        # when
        results = determine_default_allowed_quantization()
    finally:
        determine_default_allowed_quantization.cache_clear()

    # then
    assert set(results) == {
        Quantization.UNKNOWN,
        Quantization.FP32,
        Quantization.FP16,
    }


@mock.patch.object(auto_negotiation, "x_ray_runtime_environment")
def test_determine_default_allowed_quantization_for_no_cuda_device_detected_in_runtime(
    x_ray_runtime_environment_mock: MagicMock,
) -> None:
    # given
    x_ray_runtime_environment_mock.return_value = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=[],
        gpu_devices_cc=[],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=None,
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
        trt_python_package_available=False,
    )
    determine_default_allowed_quantization.cache_clear()

    try:
        # when
        results = determine_default_allowed_quantization()
    finally:
        determine_default_allowed_quantization.cache_clear()

    # then
    assert set(results) == {
        Quantization.UNKNOWN,
        Quantization.FP32,
        Quantization.FP16,
        Quantization.BF16,
    }


def test_remove_packages_not_matching_implementation_when_nothing_to_be_removed() -> (
    None
):
    # given
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.ONNX,
            quantization=Quantization.FP32,
            onnx_package_details=ONNXPackageDetails(opset=23),
            static_batch_size=1,
            package_artefacts=[],
            trusted_source=False,
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.ONNX,
            quantization=Quantization.FP16,
            onnx_package_details=ONNXPackageDetails(opset=23),
            static_batch_size=4,
            package_artefacts=[],
            trusted_source=False,
        ),
    ]

    # when
    result, discarded = remove_packages_not_matching_implementation(
        model_architecture="yolov8",
        task_type="object-detection",
        model_packages=model_packages,
    )

    # then
    assert result == model_packages
    assert len(discarded) == 0


def test_remove_packages_not_matching_implementation_when_some_entries_to_be_removed() -> (
    None
):
    # given
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-1",
            backend=BackendType.ONNX,
            quantization=Quantization.FP32,
            onnx_package_details=ONNXPackageDetails(opset=23),
            static_batch_size=1,
            package_artefacts=[],
            trusted_source=False,
        ),
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.HF,
            quantization=Quantization.FP16,
            static_batch_size=4,
            package_artefacts=[],
            trusted_source=False,
        ),
    ]

    # when
    result, discarded = remove_packages_not_matching_implementation(
        model_architecture="yolov8",
        task_type="object-detection",
        model_packages=model_packages,
    )

    # then
    assert len(result) == 1
    assert result[0] == model_packages[0]
    assert len(discarded) == 1
    assert discarded[0].package_id == "my-package-id-2"


def test_remove_packages_not_matching_implementation_when_all_entries_to_be_removed() -> (
    None
):
    # given
    model_packages = [
        ModelPackageMetadata(
            package_id="my-package-id-2",
            backend=BackendType.HF,
            quantization=Quantization.FP16,
            static_batch_size=4,
            package_artefacts=[],
            trusted_source=False,
        ),
    ]

    # when
    result, discarded = remove_packages_not_matching_implementation(
        model_architecture="yolov8",
        task_type="object-detection",
        model_packages=model_packages,
    )

    # then
    assert len(result) == 0
    assert len(discarded) == 1
    assert discarded[0].package_id == "my-package-id-2"


def test_torch_script_package_matches_runtime_environment_when_no_torch_available() -> (
    None
):
    # given
    runtime_xray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["nvidia-l4"],
        gpu_devices_cc=[Version("8.7")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=None,
        jetson_type=None,
        l4t_version=Version("36.4.0"),
        os_version="ubuntu-20.04",
        torch_available=False,
        torch_version=None,
        torchvision_version=None,
        onnxruntime_version=Version("1.21.0"),
        available_onnx_execution_providers={
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        },
        hf_transformers_available=True,
        ultralytics_available=True,
        trt_python_package_available=False,
    )
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )

    # when
    result = torch_script_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_xray,
        device=torch.device("cpu"),
    )

    # then
    assert result[0] is False
    assert result[1] is not None


def test_torch_script_package_matches_runtime_environment_when_no_torch_script_package_details_available() -> (
    None
):
    # given
    runtime_xray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["nvidia-l4"],
        gpu_devices_cc=[Version("8.7")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=None,
        jetson_type=None,
        l4t_version=Version("36.4.0"),
        os_version="ubuntu-20.04",
        torch_available=True,
        torch_version=Version("2.6.0"),
        torchvision_version=Version("0.22.0"),
        onnxruntime_version=Version("1.21.0"),
        available_onnx_execution_providers={
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        },
        hf_transformers_available=True,
        ultralytics_available=True,
        trt_python_package_available=False,
    )
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        torch_script_package_details=None,
    )

    # when
    result = torch_script_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_xray,
        device=torch.device("cpu"),
    )

    # then
    assert result[0] is False
    assert result[1] is not None


def test_torch_script_package_matches_runtime_environment_when_device_not_available() -> (
    None
):
    # given
    runtime_xray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["nvidia-l4"],
        gpu_devices_cc=[Version("8.7")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=None,
        jetson_type=None,
        l4t_version=Version("36.4.0"),
        os_version="ubuntu-20.04",
        torch_available=True,
        torch_version=Version("2.6.0"),
        torchvision_version=Version("0.22.0"),
        onnxruntime_version=Version("1.21.0"),
        available_onnx_execution_providers={
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        },
        hf_transformers_available=True,
        ultralytics_available=True,
        trt_python_package_available=False,
    )
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )

    # when
    result = torch_script_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_xray,
        device=None,
    )

    # then
    assert result[0] is False
    assert result[1] is not None


def test_torch_script_package_matches_runtime_environment_when_device_not_supported() -> (
    None
):
    # given
    runtime_xray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["nvidia-l4"],
        gpu_devices_cc=[Version("8.7")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=None,
        jetson_type=None,
        l4t_version=Version("36.4.0"),
        os_version="ubuntu-20.04",
        torch_available=True,
        torch_version=Version("2.6.0"),
        torchvision_version=Version("0.22.0"),
        onnxruntime_version=Version("1.21.0"),
        available_onnx_execution_providers={
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        },
        hf_transformers_available=True,
        ultralytics_available=True,
        trt_python_package_available=False,
    )
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )

    # when
    result = torch_script_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_xray,
        device=torch.device("cpu"),
    )

    # then
    assert result[0] is False
    assert result[1] is not None


def test_torch_script_package_matches_runtime_environment_when_torch_version_not_available_in_env() -> (
    None
):
    # given
    runtime_xray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["nvidia-l4"],
        gpu_devices_cc=[Version("8.7")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=None,
        jetson_type=None,
        l4t_version=Version("36.4.0"),
        os_version="ubuntu-20.04",
        torch_available=True,
        torch_version=None,
        torchvision_version=Version("0.22.0"),
        onnxruntime_version=Version("1.21.0"),
        available_onnx_execution_providers={
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        },
        hf_transformers_available=True,
        ultralytics_available=True,
        trt_python_package_available=False,
    )
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )

    # when
    result = torch_script_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_xray,
        device=torch.device("cpu"),
    )

    # then
    assert result[0] is False
    assert result[1] is not None


def test_torch_script_package_matches_runtime_environment_when_torch_version_does_not_match() -> (
    None
):
    # given
    runtime_xray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["nvidia-l4"],
        gpu_devices_cc=[Version("8.7")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=None,
        jetson_type=None,
        l4t_version=Version("36.4.0"),
        os_version="ubuntu-20.04",
        torch_available=True,
        torch_version=Version("2.5.0"),
        torchvision_version=Version("0.22.0"),
        onnxruntime_version=Version("1.21.0"),
        available_onnx_execution_providers={
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        },
        hf_transformers_available=True,
        ultralytics_available=True,
        trt_python_package_available=False,
    )
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )

    # when
    result = torch_script_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_xray,
        device=torch.device("cpu"),
    )

    # then
    assert result[0] is False
    assert result[1] is not None


def test_torch_script_package_matches_runtime_environment_when_torch_version_equal() -> (
    None
):
    # given
    runtime_xray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["nvidia-l4"],
        gpu_devices_cc=[Version("8.7")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=None,
        jetson_type=None,
        l4t_version=Version("36.4.0"),
        os_version="ubuntu-20.04",
        torch_available=True,
        torch_version=Version("2.6.0"),
        torchvision_version=Version("0.22.0"),
        onnxruntime_version=Version("1.21.0"),
        available_onnx_execution_providers={
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        },
        hf_transformers_available=True,
        ultralytics_available=True,
        trt_python_package_available=False,
    )
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=None,
        ),
    )

    # when
    result = torch_script_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_xray,
        device=torch.device("cpu"),
    )

    # then
    assert result[0] is True
    assert result[1] is None


def test_torch_script_package_matches_runtime_environment_when_torch_version_higher() -> (
    None
):
    # given
    runtime_xray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["nvidia-l4"],
        gpu_devices_cc=[Version("8.7")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=None,
        jetson_type=None,
        l4t_version=Version("36.4.0"),
        os_version="ubuntu-20.04",
        torch_available=True,
        torch_version=Version("2.6.1"),
        torchvision_version=Version("0.22.0"),
        onnxruntime_version=Version("1.21.0"),
        available_onnx_execution_providers={
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        },
        hf_transformers_available=True,
        ultralytics_available=True,
        trt_python_package_available=False,
    )
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=None,
        ),
    )

    # when
    result = torch_script_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_xray,
        device=torch.device("cpu"),
    )

    # then
    assert result[0] is True
    assert result[1] is None


def test_torch_script_package_matches_runtime_environment_when_torchvision_version_required_but_not_found() -> (
    None
):
    # given
    runtime_xray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["nvidia-l4"],
        gpu_devices_cc=[Version("8.7")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=None,
        jetson_type=None,
        l4t_version=Version("36.4.0"),
        os_version="ubuntu-20.04",
        torch_available=True,
        torch_version=Version("2.6.0"),
        torchvision_version=None,
        onnxruntime_version=Version("1.21.0"),
        available_onnx_execution_providers={
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        },
        hf_transformers_available=True,
        ultralytics_available=True,
        trt_python_package_available=False,
    )
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )

    # when
    result = torch_script_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_xray,
        device=torch.device("cpu"),
    )

    # then
    assert result[0] is False
    assert result[1] is not None


def test_torch_script_package_matches_runtime_environment_when_torchvision_version_required_and_found_too_low() -> (
    None
):
    # given
    runtime_xray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["nvidia-l4"],
        gpu_devices_cc=[Version("8.7")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=None,
        jetson_type=None,
        l4t_version=Version("36.4.0"),
        os_version="ubuntu-20.04",
        torch_available=True,
        torch_version=Version("2.6.0"),
        torchvision_version=Version("0.21.0"),
        onnxruntime_version=Version("1.21.0"),
        available_onnx_execution_providers={
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        },
        hf_transformers_available=True,
        ultralytics_available=True,
        trt_python_package_available=False,
    )
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )

    # when
    result = torch_script_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_xray,
        device=torch.device("cpu"),
    )

    # then
    assert result[0] is False
    assert result[1] is not None


def test_torch_script_package_matches_runtime_environment_when_torchvision_version_required_and_matches_exactly() -> (
    None
):
    # given
    runtime_xray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["nvidia-l4"],
        gpu_devices_cc=[Version("8.7")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=None,
        jetson_type=None,
        l4t_version=Version("36.4.0"),
        os_version="ubuntu-20.04",
        torch_available=True,
        torch_version=Version("2.6.0"),
        torchvision_version=Version("0.22.0"),
        onnxruntime_version=Version("1.21.0"),
        available_onnx_execution_providers={
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        },
        hf_transformers_available=True,
        ultralytics_available=True,
        trt_python_package_available=False,
    )
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )

    # when
    result = torch_script_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_xray,
        device=torch.device("cpu"),
    )

    # then
    assert result[0] is True
    assert result[1] is None


def test_torch_script_package_matches_runtime_environment_when_torchvision_version_required_and_matches_higher() -> (
    None
):
    # given
    runtime_xray = RuntimeXRayResult(
        gpu_available=True,
        gpu_devices=["nvidia-l4"],
        gpu_devices_cc=[Version("8.7")],
        driver_version=Version("510.0.4"),
        cuda_version=Version("12.6"),
        trt_version=None,
        jetson_type=None,
        l4t_version=Version("36.4.0"),
        os_version="ubuntu-20.04",
        torch_available=True,
        torch_version=Version("2.6.0"),
        torchvision_version=Version("0.23.0"),
        onnxruntime_version=Version("1.21.0"),
        available_onnx_execution_providers={
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        },
        hf_transformers_available=True,
        ultralytics_available=True,
        trt_python_package_available=False,
    )
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )

    # when
    result = torch_script_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_xray,
        device=torch.device("cpu"),
    )

    # then
    assert result[0] is True
    assert result[1] is None


def test_filter_model_packages_based_on_model_features_when_package_should_not_be_eliminated_as_no_nms_fused_features_registered() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )

    # when
    remaining_packages, discarded_packages = (
        filter_model_packages_based_on_model_features(
            model_packages=[model_package],
            nms_fusion_preferences=True,
            model_architecture="yolov8",
            task_type="object-detection",
        )
    )

    # then
    assert remaining_packages == [model_package]
    assert len(discarded_packages) == 0


def test_filter_model_packages_based_on_model_features_when_package_should_be_eliminated_as_nms_fused_features_registered_but_no_preferences() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        model_features={
            "nms_fused": {
                "max_detections": 300,
                "confidence_threshold": 0.3,
                "iou_threshold": 0.7,
                "class_agnostic": True,
            }
        },
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )

    # when
    remaining_packages, discarded_packages = (
        filter_model_packages_based_on_model_features(
            model_packages=[model_package],
            nms_fusion_preferences=None,
            model_architecture="yolov8",
            task_type="object-detection",
        )
    )

    # then
    assert len(remaining_packages) == 0
    assert len(discarded_packages) == 1
    assert discarded_packages[0].package_id == "my-package-id"


def test_filter_model_packages_based_on_model_features_when_package_should_be_eliminated_as_nms_fused_features_registered_but_nms_not_preferred() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        model_features={
            "nms_fused": {
                "max_detections": 300,
                "confidence_threshold": 0.3,
                "iou_threshold": 0.7,
                "class_agnostic": True,
            }
        },
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )

    # when
    remaining_packages, discarded_packages = (
        filter_model_packages_based_on_model_features(
            model_packages=[model_package],
            nms_fusion_preferences=False,
            model_architecture="yolov8",
            task_type="object-detection",
        )
    )

    # then
    assert len(remaining_packages) == 0
    assert len(discarded_packages) == 1
    assert discarded_packages[0].package_id == "my-package-id"


def test_filter_model_packages_based_on_model_features_when_package_should_be_eliminated_as_malformed_nms_fused_features_registered() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        model_features={
            "nms_fused": {
                "invalid_key_max_detections": 300,
                "confidence_threshold": 0.3,
                "iou_threshold": 0.7,
                "class_agnostic": True,
            }
        },
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )

    # when
    remaining_packages, discarded_packages = (
        filter_model_packages_based_on_model_features(
            model_packages=[model_package],
            nms_fusion_preferences=True,
            model_architecture="yolov8",
            task_type="object-detection",
        )
    )

    # then
    assert len(remaining_packages) == 0
    assert len(discarded_packages) == 1
    assert discarded_packages[0].package_id == "my-package-id"


def test_filter_model_packages_based_on_model_features_when_package_should_not_be_eliminated_as_matches_default() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        model_features={
            "nms_fused": {
                "max_detections": 300,
                "confidence_threshold": 0.25,
                "iou_threshold": 0.7,
                "class_agnostic": False,
            }
        },
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )

    # when
    remaining_packages, discarded_packages = (
        filter_model_packages_based_on_model_features(
            model_packages=[model_package],
            nms_fusion_preferences=True,
            model_architecture="yolov8",
            task_type="object-detection",
        )
    )

    # then
    assert remaining_packages == [model_package]
    assert len(discarded_packages) == 0


def test_filter_model_packages_based_on_model_features_when_package_should_be_eliminated_based_on_max_nms_detections_equal_comparison() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        model_features={
            "nms_fused": {
                "max_detections": 300,
                "confidence_threshold": 0.25,
                "iou_threshold": 0.7,
                "class_agnostic": False,
            }
        },
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )

    # when
    remaining_packages, discarded_packages = (
        filter_model_packages_based_on_model_features(
            model_packages=[model_package],
            nms_fusion_preferences={
                "max_detections": 350,
            },
            model_architecture="yolov8",
            task_type="object-detection",
        )
    )

    # then
    assert len(remaining_packages) == 0
    assert len(discarded_packages) == 1


def test_filter_model_packages_based_on_model_features_when_package_should_be_eliminated_based_on_max_nms_detections_range_comparison() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        model_features={
            "nms_fused": {
                "max_detections": 300,
                "confidence_threshold": 0.25,
                "iou_threshold": 0.7,
                "class_agnostic": False,
            }
        },
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )

    # when
    remaining_packages, discarded_packages = (
        filter_model_packages_based_on_model_features(
            model_packages=[model_package],
            nms_fusion_preferences={
                "max_detections": (350, 400),
            },
            model_architecture="yolov8",
            task_type="object-detection",
        )
    )

    # then
    assert len(remaining_packages) == 0
    assert len(discarded_packages) == 1


def test_filter_model_packages_based_on_model_features_when_package_should_be_accepted_based_on_max_nms_detections_range_comparison() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        model_features={
            "nms_fused": {
                "max_detections": 300,
                "confidence_threshold": 0.25,
                "iou_threshold": 0.7,
                "class_agnostic": False,
            }
        },
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )

    # when
    remaining_packages, discarded_packages = (
        filter_model_packages_based_on_model_features(
            model_packages=[model_package],
            nms_fusion_preferences={
                "max_detections": (250, 400),
            },
            model_architecture="yolov8",
            task_type="object-detection",
        )
    )

    # then
    assert len(remaining_packages) == 1
    assert len(discarded_packages) == 0


def test_filter_model_packages_based_on_model_features_when_package_should_be_accepted_based_on_max_nms_detections_equal_comparison() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        model_features={
            "nms_fused": {
                "max_detections": 300,
                "confidence_threshold": 0.25,
                "iou_threshold": 0.7,
                "class_agnostic": False,
            }
        },
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )

    # when
    remaining_packages, discarded_packages = (
        filter_model_packages_based_on_model_features(
            model_packages=[model_package],
            nms_fusion_preferences={
                "max_detections": 300,
            },
            model_architecture="yolov8",
            task_type="object-detection",
        )
    )

    # then
    assert len(remaining_packages) == 1
    assert len(discarded_packages) == 0


def test_filter_model_packages_based_on_model_features_when_package_should_be_eliminated_based_on_confidence_equal_comparison() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        model_features={
            "nms_fused": {
                "max_detections": 300,
                "confidence_threshold": 0.25,
                "iou_threshold": 0.7,
                "class_agnostic": False,
            }
        },
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )

    # when
    remaining_packages, discarded_packages = (
        filter_model_packages_based_on_model_features(
            model_packages=[model_package],
            nms_fusion_preferences={
                "confidence_threshold": 0.3,
            },
            model_architecture="yolov8",
            task_type="object-detection",
        )
    )

    # then
    assert len(remaining_packages) == 0
    assert len(discarded_packages) == 1


def test_filter_model_packages_based_on_model_features_when_package_should_be_eliminated_based_on_confidence_range_comparison() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        model_features={
            "nms_fused": {
                "max_detections": 300,
                "confidence_threshold": 0.25,
                "iou_threshold": 0.7,
                "class_agnostic": False,
            }
        },
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )

    # when
    remaining_packages, discarded_packages = (
        filter_model_packages_based_on_model_features(
            model_packages=[model_package],
            nms_fusion_preferences={
                "confidence_threshold": (0.3, 0.7),
            },
            model_architecture="yolov8",
            task_type="object-detection",
        )
    )

    # then
    assert len(remaining_packages) == 0
    assert len(discarded_packages) == 1


def test_filter_model_packages_based_on_model_features_when_package_should_be_accepted_based_on_confidence_range_comparison() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        model_features={
            "nms_fused": {
                "max_detections": 300,
                "confidence_threshold": 0.25,
                "iou_threshold": 0.7,
                "class_agnostic": False,
            }
        },
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )

    # when
    remaining_packages, discarded_packages = (
        filter_model_packages_based_on_model_features(
            model_packages=[model_package],
            nms_fusion_preferences={
                "confidence_threshold": (0.2, 0.3),
            },
            model_architecture="yolov8",
            task_type="object-detection",
        )
    )

    # then
    assert len(remaining_packages) == 1
    assert len(discarded_packages) == 0


def test_filter_model_packages_based_on_model_features_when_package_should_be_accepted_based_on_confidence_equal_comparison() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        model_features={
            "nms_fused": {
                "max_detections": 300,
                "confidence_threshold": 0.25,
                "iou_threshold": 0.7,
                "class_agnostic": False,
            }
        },
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )

    # when
    remaining_packages, discarded_packages = (
        filter_model_packages_based_on_model_features(
            model_packages=[model_package],
            nms_fusion_preferences={
                "confidence_threshold": 0.25,
            },
            model_architecture="yolov8",
            task_type="object-detection",
        )
    )

    # then
    assert len(remaining_packages) == 1
    assert len(discarded_packages) == 0


# xxx


def test_filter_model_packages_based_on_model_features_when_package_should_be_eliminated_based_on_iou_equal_comparison() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        model_features={
            "nms_fused": {
                "max_detections": 300,
                "confidence_threshold": 0.25,
                "iou_threshold": 0.7,
                "class_agnostic": False,
            }
        },
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )

    # when
    remaining_packages, discarded_packages = (
        filter_model_packages_based_on_model_features(
            model_packages=[model_package],
            nms_fusion_preferences={
                "iou_threshold": 0.3,
            },
            model_architecture="yolov8",
            task_type="object-detection",
        )
    )

    # then
    assert len(remaining_packages) == 0
    assert len(discarded_packages) == 1


def test_filter_model_packages_based_on_model_features_when_package_should_be_eliminated_based_on_iou_range_comparison() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        model_features={
            "nms_fused": {
                "max_detections": 300,
                "confidence_threshold": 0.25,
                "iou_threshold": 0.7,
                "class_agnostic": False,
            }
        },
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )

    # when
    remaining_packages, discarded_packages = (
        filter_model_packages_based_on_model_features(
            model_packages=[model_package],
            nms_fusion_preferences={
                "iou_threshold": (0.3, 0.65),
            },
            model_architecture="yolov8",
            task_type="object-detection",
        )
    )

    # then
    assert len(remaining_packages) == 0
    assert len(discarded_packages) == 1


def test_filter_model_packages_based_on_model_features_when_package_should_be_accepted_based_on_iou_range_comparison() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        model_features={
            "nms_fused": {
                "max_detections": 300,
                "confidence_threshold": 0.25,
                "iou_threshold": 0.7,
                "class_agnostic": False,
            }
        },
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )

    # when
    remaining_packages, discarded_packages = (
        filter_model_packages_based_on_model_features(
            model_packages=[model_package],
            nms_fusion_preferences={
                "iou_threshold": (0.2, 0.7),
            },
            model_architecture="yolov8",
            task_type="object-detection",
        )
    )

    # then
    assert len(remaining_packages) == 1
    assert len(discarded_packages) == 0


def test_filter_model_packages_based_on_model_features_when_package_should_be_accepted_based_on_iou_equal_comparison() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        model_features={
            "nms_fused": {
                "max_detections": 300,
                "confidence_threshold": 0.25,
                "iou_threshold": 0.7,
                "class_agnostic": False,
            }
        },
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )

    # when
    remaining_packages, discarded_packages = (
        filter_model_packages_based_on_model_features(
            model_packages=[model_package],
            nms_fusion_preferences={
                "iou_threshold": 0.7,
            },
            model_architecture="yolov8",
            task_type="object-detection",
        )
    )

    # then
    assert len(remaining_packages) == 1
    assert len(discarded_packages) == 0


def test_filter_model_packages_based_on_model_features_when_package_should_be_eliminated_based_on_class_agnostic() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        model_features={
            "nms_fused": {
                "max_detections": 300,
                "confidence_threshold": 0.25,
                "iou_threshold": 0.7,
                "class_agnostic": False,
            }
        },
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )

    # when
    remaining_packages, discarded_packages = (
        filter_model_packages_based_on_model_features(
            model_packages=[model_package],
            nms_fusion_preferences={
                "class_agnostic": True,
            },
            model_architecture="yolov8",
            task_type="object-detection",
        )
    )

    # then
    assert len(remaining_packages) == 0
    assert len(discarded_packages) == 1


def test_filter_model_packages_based_on_model_features_when_package_not_should_be_eliminated_based_on_class_agnostic() -> (
    None
):
    # given
    model_package = ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        model_features={
            "nms_fused": {
                "max_detections": 300,
                "confidence_threshold": 0.25,
                "iou_threshold": 0.7,
                "class_agnostic": False,
            }
        },
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )

    # when
    remaining_packages, discarded_packages = (
        filter_model_packages_based_on_model_features(
            model_packages=[model_package],
            nms_fusion_preferences={
                "class_agnostic": False,
            },
            model_architecture="yolov8",
            task_type="object-detection",
        )
    )

    # then
    assert len(remaining_packages) == 1
    assert len(discarded_packages) == 0
