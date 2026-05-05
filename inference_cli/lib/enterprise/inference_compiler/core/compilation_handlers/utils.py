import logging
import os
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

from rich.console import Console

from inference_cli.lib.enterprise.inference_compiler.adapters.models_service import (
    FileConfirmation,
    ModelPackageRegistrationResponse,
    ModelsServiceClient,
)
from inference_cli.lib.enterprise.inference_compiler.constants import MODEL_CONFIG_FILE
from inference_cli.lib.enterprise.inference_compiler.core.compilation_handlers.engine_builder import (
    EngineBuilder,
)
from inference_cli.lib.enterprise.inference_compiler.core.compilation_handlers.timing_cache_manager import (
    TimingCacheManager,
)
from inference_cli.lib.enterprise.inference_compiler.core.entities import (
    GPUServerSpecsV1,
    JetsonMachineSpecsV1,
    TRTConfig,
    TRTMachineType,
    TRTModelPackageV1,
)
from inference_cli.lib.enterprise.inference_compiler.errors import (
    AlreadyCompiledError,
    CompiledPackageRegistrationError,
    CorruptedPackageError,
    LackOfSourcePackageError,
    ModelVerificationError,
    PackageDownloadError,
    PackageNegotiationError,
    RequestError,
)
from inference_cli.lib.enterprise.inference_compiler.utils.file_system import (
    dump_json,
    read_json,
)
from inference_cli.lib.enterprise.inference_compiler.utils.http import (
    upload_file_to_cloud,
)
from inference_cli.lib.enterprise.inference_compiler.utils.logging import (
    print_to_console,
)
from inference_models.errors import NoModelPackagesAvailableError
from inference_models.models.auto_loaders.auto_negotiation import (
    negotiate_model_packages,
)
from inference_models.runtime_introspection.core import x_ray_runtime_environment
from inference_models.utils.download import download_files_to_directory
from inference_models.weights_providers.entities import (
    BackendType,
    ModelMetadata,
    ModelPackageMetadata,
    Quantization,
)

logger = logging.getLogger("inference_cli.inference_compiler")


def safe_negotiate_model_packages(
    model_metadata: ModelMetadata,
    requested_backends: Union[BackendType, List[BackendType]] = BackendType.ONNX,
    requested_quantization: Union[Quantization, List[Quantization]] = Quantization.FP32,
    allow_untrusted_packages: bool = False,
) -> Tuple[ModelPackageMetadata, Optional[ModelPackageMetadata]]:
    try:
        matching_model_packages = negotiate_model_packages(
            model_architecture=model_metadata.model_architecture,
            task_type=model_metadata.task_type,
            model_packages=model_metadata.model_packages,
            requested_backends=requested_backends,
            requested_quantization=requested_quantization,
            allow_untrusted_packages=allow_untrusted_packages,
            verbose=True,
        )
        package_with_static_batch_size = select_package_with_static_batch_size(
            model_packages=matching_model_packages
        )
        package_with_dynamic_batch_size = select_package_with_dynamic_batch_size(
            model_packages=matching_model_packages
        )
        return package_with_static_batch_size, package_with_dynamic_batch_size
    except LackOfSourcePackageError as error:
        raise error
    except NoModelPackagesAvailableError as error:
        raise LackOfSourcePackageError(
            "Could not find a model package to use as a compilation source."
        ) from error
    except Exception as error:
        logger.exception("Error selecting model packages for compilation")
        raise PackageNegotiationError(
            "Error selecting model packages for compilation."
        ) from error


def select_package_with_static_batch_size(
    model_packages: List[ModelPackageMetadata],
) -> ModelPackageMetadata:
    static_bs_models = [p for p in model_packages if not p.dynamic_batch_size_supported]
    if len(static_bs_models) == 0:
        raise LackOfSourcePackageError(
            "Could not find model package with static batch size"
        )
    return static_bs_models[0]


def select_package_with_dynamic_batch_size(
    model_packages: List[ModelPackageMetadata],
) -> Optional[ModelPackageMetadata]:
    dynamic_bs_models = [p for p in model_packages if p.dynamic_batch_size_supported]
    if len(dynamic_bs_models) == 0:
        return None
    return dynamic_bs_models[0]


def download_model_package(
    model_architecture: str,
    task_type: str,
    model_package: ModelPackageMetadata,
    target_directory: str,
    expected_files: List[str],
    verify_model: Optional[Callable[[str], None]] = None,
) -> Dict[str, str]:
    files_specs = [
        (a.file_handle, a.download_url, a.md5_hash)
        for a in model_package.package_artefacts
    ]
    package_dir = os.path.join(target_directory, model_package.package_id)
    file_mapping = {
        a.file_handle: os.path.join(package_dir, a.file_handle)
        for a in model_package.package_artefacts
    }
    if any(f not in file_mapping for f in expected_files):
        raise CorruptedPackageError(
            f"At least one of the required files {expected_files} is missing from model package {model_package.package_id}"
        )
    try:
        os.makedirs(package_dir, exist_ok=True)
        download_files_to_directory(
            target_dir=package_dir,
            files_specs=files_specs,
            verbose=True,
        )
    except Exception as error:
        logger.exception(
            "Error downloading model package: %s", model_package.package_id
        )
        raise PackageDownloadError(
            f"Could not download model package: {error}"
        ) from error
    if verify_model is not None:
        try:
            model_config_path = os.path.join(package_dir, MODEL_CONFIG_FILE)
            model_config = {
                "model_architecture": model_architecture,
                "task_type": task_type,
                "backend_type": model_package.backend.value,
            }
            dump_json(path=model_config_path, content=model_config)
            verify_model(package_dir)
        except ModelVerificationError as error:
            raise error
        except Exception as error:
            raise ModelVerificationError(
                "Could not verify compiled model correctness"
            ) from error
    return file_mapping


def get_training_input_size(inference_config_path: str) -> Tuple[int, int]:
    try:
        inference_config = read_json(path=inference_config_path)
        dimensions = inference_config["network_input"]["training_input_size"]
        return dimensions["height"], dimensions["width"]
    except Exception as error:
        raise CorruptedPackageError(
            f"Could not get training input size from inference config - {error}"
        )


def execute_compilation(
    models_service_client: ModelsServiceClient,
    model_id: str,
    model_architecture: str,
    task_type: Optional[str],
    model_variant: Optional[str],
    file_handles_to_register: List[str],
    compilation_directory: str,
    onnx_path: str,
    precision: Literal["fp32", "fp16"],
    model_input_size: Tuple[int, int],
    workspace_size_gb: int,
    min_batch_size: Optional[int] = None,
    opt_batch_size: Optional[int] = None,
    max_batch_size: Optional[int] = None,
    trt_version_compatible: bool = True,
    same_compute_compatibility: bool = False,
    registered_model_features: Optional[dict] = None,
    console: Optional[Console] = None,
) -> Tuple[str, TRTConfig, ModelPackageRegistrationResponse]:
    runtime_xray = x_ray_runtime_environment()
    logger.info(
        "Runtime environment: gpu=%s, cc=%s, cuda=%s, trt=%s, driver=%s, l4t=%s",
        runtime_xray.gpu_devices[0] if runtime_xray.gpu_devices else "unknown",
        runtime_xray.gpu_devices_cc[0] if runtime_xray.gpu_devices_cc else "unknown",
        runtime_xray.cuda_version,
        runtime_xray.trt_version,
        runtime_xray.driver_version,
        runtime_xray.l4t_version,
    )
    os.makedirs(compilation_directory, exist_ok=True)
    engine_builder = EngineBuilder(workspace=workspace_size_gb)
    engine_builder.create_network(onnx_path=onnx_path)
    dynamic_batch_sizes = None
    dynamic_dimensions_in_use = all(
        e is not None for e in [min_batch_size, opt_batch_size, max_batch_size]
    )
    static_batch_size = None
    if dynamic_dimensions_in_use:
        dynamic_batch_sizes = min_batch_size, opt_batch_size, max_batch_size
    else:
        static_batch_size = engine_builder.get_static_batch_size_of_input()
    if dynamic_batch_sizes:
        print_to_console(
            message=f"Compiling model with dynamic batch sizes: {dynamic_batch_sizes}, quantization: {precision}",
            console=console,
        )
        trt_config = TRTConfig(
            dynamic_batch_size_min=dynamic_batch_sizes[0],
            dynamic_batch_size_opt=dynamic_batch_sizes[1],
            dynamic_batch_size_max=dynamic_batch_sizes[2],
        )
    else:
        print_to_console(
            message=f"Compiling model with static batch size: {static_batch_size}, quantization: {precision}",
            console=console,
        )
        trt_config = TRTConfig(
            static_batch_size=static_batch_size,
        )
    if runtime_xray.l4t_version is not None:
        machine_type = TRTMachineType.JETSON
        machine_specs = JetsonMachineSpecsV1(
            type="jetson-machine-specs-v1",
            l4t_version=str(runtime_xray.l4t_version),
            device_name=runtime_xray.jetson_type or "unknown",
            driver_version=str(runtime_xray.driver_version),
        )
    else:
        machine_type = TRTMachineType.GPU_SERVER
        machine_specs = GPUServerSpecsV1(
            type="gpu-server-specs-v1",
            driver_version=str(runtime_xray.driver_version),
            os_version=runtime_xray.os_version,
        )
    package_manifest = TRTModelPackageV1(
        type="trt-model-package-v1",
        backend_type="trt",
        dynamic_batch_size=dynamic_batch_sizes is not None,
        static_batch_size=static_batch_size,
        min_batch_size=trt_config.dynamic_batch_size_min,
        opt_batch_size=trt_config.dynamic_batch_size_opt,
        max_batch_size=trt_config.dynamic_batch_size_max,
        quantization=Quantization(precision),
        cuda_device_type=runtime_xray.gpu_devices[0],
        cuda_device_cc=str(runtime_xray.gpu_devices_cc[0]),
        cuda_version=str(runtime_xray.cuda_version),
        trt_version=str(runtime_xray.trt_version),
        same_cc_compatible=same_compute_compatibility,
        trt_forward_compatible=trt_version_compatible,
        trt_lean_runtime_excluded=False,
        machine_type=machine_type,
        machine_specs=machine_specs,
    )
    try:
        # stating the registration, to see if package already sealed
        _ = models_service_client.register_model_package(
            model_id=model_id,
            package_manifest=package_manifest.model_dump(
                by_alias=True, mode="json", exclude_none=True
            ),
            file_handles=file_handles_to_register,
            model_features=registered_model_features,
        )
    except RequestError as error:
        if error.status_code == 409:
            raise AlreadyCompiledError("Model package already compiled.")
        logger.exception("Could not pre-register model package")
        raise CompiledPackageRegistrationError(
            f"Could not register model package: {error}"
        ) from error
    except Exception as error:
        logger.exception("Error while registering model package")
        raise CompiledPackageRegistrationError(
            f"Could not register model package: {error}"
        ) from error
    compilation_features = {
        "modelArchitecture": model_architecture,
        "taskType": task_type,
        "modelVariant": model_variant,
        "modelInputSize": model_input_size,
        "precision": precision,
        "workspaceSizeGb": workspace_size_gb,
        "trtForwardCompatible": trt_version_compatible,
        "sameCCCompatible": same_compute_compatibility,
        "dynamicBatchSizes": dynamic_batch_sizes,
        "cudaDeviceType": runtime_xray.gpu_devices[0],
        "trtVersion": (
            str(runtime_xray.trt_version) if runtime_xray.trt_version else None
        ),
    }
    cache_manager = TimingCacheManager.init(
        models_service_client=models_service_client,
        compilation_features=compilation_features,
    )
    engine_builder.set_timing_cache_manager(cache_manager=cache_manager)
    engine_path = os.path.join(compilation_directory, "engine.plan")
    engine_builder.create_engine(
        engine_path=engine_path,
        precision=precision,
        input_size=model_input_size,
        dynamic_batch_sizes=dynamic_batch_sizes,  # type: ignore
        trt_version_compatible=trt_version_compatible,
        same_compute_compatibility=same_compute_compatibility,
    )
    try:
        # performing registration again, so that we have fresh upload URL
        registration_result = models_service_client.register_model_package(
            model_id=model_id,
            package_manifest=package_manifest.model_dump(
                by_alias=True, mode="json", exclude_none=True
            ),
            file_handles=file_handles_to_register,
            model_features=registered_model_features,
        )
        return engine_path, trt_config, registration_result
    except RequestError as error:
        if error.status_code == 409:
            raise AlreadyCompiledError("Model package already compiled.")
        logger.exception("Could not register model package after compilation")
        raise CompiledPackageRegistrationError(
            f"Could not register model package: {error}"
        ) from error
    except Exception as error:
        logger.exception("Error while registering model package")
        raise CompiledPackageRegistrationError(
            f"Could not register model package: {error}"
        ) from error


def register_model_package_artefacts(
    registration_response: ModelPackageRegistrationResponse,
    local_files_mapping: Dict[str, Tuple[str, str]],
    models_service_client: ModelsServiceClient,
) -> None:
    try:
        confirmations = []
        for file_upload_spec in registration_response.file_upload_specs:
            file_path, file_md5 = local_files_mapping[file_upload_spec.file_handle]
            upload_file_to_cloud(
                file_path=file_path,
                url=file_upload_spec.signed_url_details.upload_url,
                headers=file_upload_spec.signed_url_details.extension_headers,
            )
            confirmations.append(
                FileConfirmation(
                    file_handle=file_upload_spec.file_handle,
                    md5_hash=file_md5,
                )
            )
        models_service_client.confirm_model_package_artefacts(
            model_id=registration_response.model_id,
            model_package_id=registration_response.model_package_id,
            confirmations=confirmations,
            seal_model_package=True,
        )
        logger.info(
            "Registered package %s for model %s",
            registration_response.model_package_id,
            registration_response.model_id,
        )
    except Exception as error:
        logger.exception(
            "Could not register artefacts for package %s",
            registration_response.model_package_id,
        )
        raise CompiledPackageRegistrationError(
            f"Could not register artefacts for package {registration_response.model_package_id}"
        ) from error
