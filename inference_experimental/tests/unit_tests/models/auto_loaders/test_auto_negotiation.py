from typing import Tuple
from unittest.mock import MagicMock

import pytest
import torch
from inference_exp.errors import (
    InvalidRequestedBatchSizeError,
    ModelPackageNegotiationError,
    UnknownBackendTypeError,
    UnknownQuantizationError,
)
from inference_exp.models.auto_loaders.auto_negotiation import (
    parse_backend_type,
    parse_batch_size,
    parse_quantization,
    parse_requested_quantization,
    range_within_other,
    trt_package_matches_runtime_environment,
    verify_trt_package_compatibility_with_cuda_device,
    verify_versions_up_to_major_and_minor,
)
from inference_exp.runtime_introspection.core import RuntimeXRayResult
from inference_exp.weights_providers.entities import (
    BackendType,
    JetsonEnvironmentRequirements,
    ModelPackageMetadata,
    Quantization,
    ServerEnvironmentRequirements,
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
    result = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
    )

    # then
    assert result is False


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
    result = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
    )

    # then
    assert result is False


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
    result = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
    )

    # then
    assert result is False


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
    result = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
    )

    # then
    assert result is False


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
    result = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
        device=torch.device("cpu"),
    )

    # then
    assert result is False


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
    result = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
    )

    # then
    assert result is False


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
    result = trt_package_matches_runtime_environment(
        model_package=model_package, runtime_x_ray=runtime_x_ray, verbose=True
    )

    # then
    assert result is True


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
    result = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
    )

    # then
    assert result is False


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
    result = trt_package_matches_runtime_environment(
        model_package=model_package, runtime_x_ray=runtime_x_ray, verbose=True
    )

    # then
    assert result is False


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
    result = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
    )

    # then
    assert result is False


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
    result = trt_package_matches_runtime_environment(
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
    result = trt_package_matches_runtime_environment(
        model_package=model_package, runtime_x_ray=runtime_x_ray, verbose=True
    )

    # then
    assert result is False


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
    result = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
        trt_engine_host_code_allowed=False,
    )

    # then
    assert result is False


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
    result = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
    )

    assert result is False


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
    result = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
    )

    assert result is False


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
    result = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
        device=torch.device("cpu"),
    )

    assert result is False


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
    result = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
        device=torch.device(type="cuda", index=0),
    )

    assert result is True


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
    result = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
        device=torch.device(type="cuda", index=0),
    )

    assert result is True


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
    result = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
        device=torch.device(type="cuda", index=0),
    )

    assert result is False


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
    result = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
        device=torch.device(type="cuda", index=0),
    )

    assert result is False


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
    result = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
        device=torch.device(type="cuda", index=0),
        verbose=True,
    )

    assert result is False


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
    result = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
        device=torch.device(type="cuda", index=0),
    )

    assert result is True


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
    result = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
        device=torch.device(type="cuda", index=0),
    )

    assert result is False


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
    result = trt_package_matches_runtime_environment(
        model_package=model_package,
        runtime_x_ray=runtime_x_ray,
        device=torch.device(type="cuda", index=0),
        trt_engine_host_code_allowed=False,
    )

    assert result is False
