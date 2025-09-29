import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Union

from inference_exp.runtime_introspection.core import RuntimeXRayResult
from inference_exp.utils.download import get_content_length
from inference_exp.weights_providers.entities import (
    FileDownloadSpecs,
    JetsonEnvironmentRequirements,
    ModelPackageMetadata,
    Quantization,
    ServerEnvironmentRequirements,
)
from rich.table import Table

BYTES_IN_MB = 1024**2


def render_table_with_model_overview(
    model_id: str,
    requested_model_id: str,
    model_architecture: str,
    task_type: Optional[str],
    weights_provider: str,
    registered_packages: int,
) -> Table:
    table = Table(title="Model overview", show_header=False, box=None)
    table.add_column(justify="left", no_wrap=True, style="bold green4")
    table.add_column(justify="left")
    model_id_str = model_id
    if requested_model_id != model_id:
        model_id_str = f"{model_id_str} (alias: {requested_model_id})"
    table.add_row("Model ID:", model_id_str)
    table.add_row("Architecture:", model_architecture)
    table.add_row("Task:", task_type or "N/A")
    table.add_row("Weights provider:", weights_provider)
    table.add_row("Number of packages:", str(registered_packages))
    return table


def calculate_size_of_all_model_packages_artefacts(
    model_packages: List[ModelPackageMetadata],
    max_workers: int = 16,
) -> List[Tuple[int, bool]]:
    all_artefacts = [package.package_artefacts for package in model_packages]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(calculate_artefacts_size, all_artefacts))


def calculate_artefacts_size(
    package_artefacts: List[FileDownloadSpecs],
) -> Tuple[int, bool]:
    result = 0
    success = True
    for artefact in package_artefacts:
        try:
            result += get_content_length(url=artefact.download_url)
        except Exception:
            success = False
    return result, success


def render_table_with_model_packages(
    model_packages: List[ModelPackageMetadata],
    model_packages_size: Optional[List[Tuple[int, bool]]],
) -> Table:
    table = Table(title="Model packages", show_lines=True)
    table.add_column("ID", justify="center", no_wrap=True, style="bold")
    table.add_column("backend", justify="center")
    table.add_column("batch size", justify="center")
    table.add_column("quantization", justify="center")
    table.add_column("size", justify="center")
    table.add_column("trusted", justify="center")
    if model_packages_size is None:
        model_packages_size = [None] * len(model_packages)
    for model_package, package_size in zip(model_packages, model_packages_size):
        if package_size is None:
            size_str = "N/A"
        else:
            size, status = package_size
            size_str = bytes_to_human_format(size=size)
            if not status:
                size_str = f"{size_str} ⚠️"
        batch_size = (
            str(model_package.static_batch_size)
            if model_package.static_batch_size
            else "N"
        )
        if model_package.quantization is Quantization.UNKNOWN:
            quantization_str = "N/A"
        else:
            quantization_str = model_package.quantization.value
        table.add_row(
            model_package.package_id,
            model_package.backend.value,
            batch_size,
            quantization_str,
            size_str,
            str(model_package.trusted_source),
        )
    return table


def render_model_package_details_table(
    model_id: str,
    requested_model_id: str,
    model_package: ModelPackageMetadata,
    artefacts_size: Optional[Tuple[int, bool]],
) -> Table:
    model_id_str = model_id
    if requested_model_id != model_id:
        model_id_str = f"{model_id_str} (alias: {requested_model_id})"
    if artefacts_size is None:
        size_str = "N/A"
    else:
        size, status = artefacts_size
        size_str = bytes_to_human_format(size=size)
        if not status:
            size_str = f"{size_str} ⚠️"
    batch_size = (
        str(model_package.static_batch_size) if model_package.static_batch_size else "N"
    )
    if model_package.quantization is Quantization.UNKNOWN:
        quantization_str = "N/A"
    else:
        quantization_str = model_package.quantization.value
    table = Table(title="Model package overview", show_header=False, box=None)
    table.add_column(justify="left", no_wrap=True, style="bold green4")
    table.add_column(justify="left")
    table.add_row("Model ID:", model_id_str)
    table.add_row("Package ID:", model_package.package_id)
    table.add_row("Backend:", model_package.backend.value)
    table.add_row("Batch size:", batch_size)
    table.add_row("Quantization:", quantization_str)
    table.add_row("Package files:", str(len(model_package.package_artefacts)))
    table.add_row("Package size:", size_str)
    table.add_row("Trusted source:", str(model_package.trusted_source))
    if model_package.trt_package_details is not None:
        if model_package.dynamic_batch_size_supported:
            dynamic_batch_size_str = f"min: {model_package.trt_package_details.min_dynamic_batch_size}, opt: {model_package.trt_package_details.opt_dynamic_batch_size}, max: {model_package.trt_package_details.max_dynamic_batch_size}"
            table.add_row("TRT dynamic batch size:", dynamic_batch_size_str)
        table.add_row(
            "TRT same CUDA CC compatibility:",
            str(model_package.trt_package_details.same_cc_compatible),
        )
        table.add_row(
            "TRT forward compatibility:",
            str(model_package.trt_package_details.trt_forward_compatible),
        )
        table.add_row(
            "TRT lean runtime excluded:",
            str(model_package.trt_package_details.trt_lean_runtime_excluded),
        )
        table = render_compilation_device_details(
            table=table,
            environment_requirements=model_package.environment_requirements,
        )
    if model_package.onnx_package_details:
        if model_package.onnx_package_details.incompatible_providers:
            incompatible_providers_str = ", ".join(
                model_package.onnx_package_details.incompatible_providers
            )
        else:
            incompatible_providers_str = "N/A"
        table.add_row("ONNX opset:", str(model_package.onnx_package_details.opset))
        table.add_row("Incompatible providers:", incompatible_providers_str)
    if model_package.torch_script_package_details:
        supported_device_types_str = ", ".join(
            model_package.torch_script_package_details.supported_device_types
        )
        torch_version_str = str(
            model_package.torch_script_package_details.torch_version
        )
        if model_package.torch_script_package_details.torch_vision_version:
            torch_vision_version_str = str(
                model_package.torch_script_package_details.torch_vision_version
            )
        else:
            torch_vision_version_str = "N/A"
        table.add_row("Supported devices:", supported_device_types_str)
        table.add_row("Required torch version:", torch_version_str)
        table.add_row("Required torchvision version:", torch_vision_version_str)
    if model_package.model_features:
        table.add_row("Model features:", json.dumps(model_package.model_features))
    return table


def render_compilation_device_details(
    table: Table,
    environment_requirements: Optional[
        Union[ServerEnvironmentRequirements, JetsonEnvironmentRequirements]
    ],
) -> Table:
    if environment_requirements is None:
        return table
    if isinstance(environment_requirements, ServerEnvironmentRequirements):
        table.add_row(
            "Compilation device name:", str(environment_requirements.cuda_device_name)
        )
        table.add_row(
            "Compilation device CUDA CC:", str(environment_requirements.cuda_device_cc)
        )
        cuda_version_str = (
            str(environment_requirements.cuda_version)
            if environment_requirements.cuda_version
            else "N/A"
        )
        driver_version_str = (
            str(environment_requirements.driver_version)
            if environment_requirements.driver_version
            else "N/A"
        )
        trt_version_str = (
            str(environment_requirements.trt_version)
            if environment_requirements.trt_version
            else "N/A"
        )
        table.add_row("Compilation device CUDA version:", cuda_version_str)
        table.add_row("Compilation device driver:", driver_version_str)
        table.add_row("Compilation device TRT Version:", trt_version_str)
        return table
    jetson_product_name_str = (
        str(environment_requirements.jetson_product_name)
        if environment_requirements.jetson_product_name
        else "N/A"
    )
    table.add_row("Compilation Jetson type:", jetson_product_name_str)
    table.add_row(
        "Compilation device name:", str(environment_requirements.cuda_device_name)
    )
    table.add_row(
        "Compilation device CUDA CC:", str(environment_requirements.cuda_device_cc)
    )
    cuda_version_str = (
        str(environment_requirements.cuda_version)
        if environment_requirements.cuda_version
        else "N/A"
    )
    l4t_version_str = (
        str(environment_requirements.l4t_version)
        if environment_requirements.l4t_version
        else "N/A"
    )
    trt_version_str = (
        str(environment_requirements.trt_version)
        if environment_requirements.trt_version
        else "N/A"
    )
    table.add_row("Compilation device CUDA version:", cuda_version_str)
    table.add_row("Compilation device L4T:", l4t_version_str)
    table.add_row("Compilation device TRT Version:", trt_version_str)
    return table


def render_runtime_x_ray(runtime_x_ray: RuntimeXRayResult) -> Table:
    table = Table(title="Compute environment details", show_header=False, box=None)
    table.add_column(justify="left", no_wrap=True, style="bold green4")
    table.add_column(justify="left")
    detected_gpus = (
        ", ".join(runtime_x_ray.gpu_devices) if runtime_x_ray.gpu_devices else "N/A"
    )
    table.add_row("Detected GPUs:", detected_gpus)
    detected_gpus_cc = (
        ", ".join([str(cc) for cc in runtime_x_ray.gpu_devices_cc])
        if runtime_x_ray.gpu_devices_cc
        else "N/A"
    )
    table.add_row("Detected GPUs CUDA CC:", detected_gpus_cc)
    nvidia_driver = (
        str(runtime_x_ray.driver_version) if runtime_x_ray.driver_version else "N/A"
    )
    table.add_row("NVIDIA driver:", nvidia_driver)
    cuda_version = (
        str(runtime_x_ray.cuda_version) if runtime_x_ray.cuda_version else "N/A"
    )
    table.add_row("CUDA version:", cuda_version)
    trt_version = str(runtime_x_ray.trt_version) if runtime_x_ray.trt_version else "N/A"
    table.add_row("TRT version:", trt_version)
    table.add_row(
        "TRT Python package available:", str(runtime_x_ray.trt_python_package_available)
    )
    if runtime_x_ray.jetson_type is not None:
        table.add_row("Jetson device type:", runtime_x_ray.jetson_type)
    if runtime_x_ray.l4t_version is not None:
        table.add_row("L4T version:", str(runtime_x_ray.l4t_version))
    os_version = runtime_x_ray.os_version if runtime_x_ray.os_version else "N/A"
    table.add_row("OS version:", os_version)
    torch_version = (
        str(runtime_x_ray.torch_version) if runtime_x_ray.torch_version else "N/A"
    )
    table.add_row("torch version:", torch_version)
    torchvision_version = (
        str(runtime_x_ray.torchvision_version)
        if runtime_x_ray.torchvision_version
        else "N/A"
    )
    table.add_row("torchvision version:", torchvision_version)
    onnxruntime_version = (
        str(runtime_x_ray.onnxruntime_version)
        if runtime_x_ray.onnxruntime_version
        else "N/A"
    )
    table.add_row("ONNX runtime version:", onnxruntime_version)
    available_onnx_execution_providers = (
        ", ".join(runtime_x_ray.available_onnx_execution_providers)
        if runtime_x_ray.available_onnx_execution_providers
        else "N/A"
    )
    table.add_row(
        "Detected ONNX execution providers:", available_onnx_execution_providers
    )
    return table


def bytes_to_human_format(size: int) -> str:
    mega_bytes = size / BYTES_IN_MB
    if mega_bytes <= 512:
        return f"{round(mega_bytes, 2)} MB"
    giga_bytes = mega_bytes / 1024
    return f"{round(giga_bytes, 2)} GB"
