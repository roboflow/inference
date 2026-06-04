import logging
import os.path
import tempfile
from typing import Callable, Dict, Literal, Optional, Tuple

from rich.console import Console

from inference_cli.lib.enterprise.inference_compiler.adapters.models_service import (
    ModelPackageRegistrationResponse,
    ModelsServiceClient,
)
from inference_cli.lib.enterprise.inference_compiler.constants import (
    CLASS_NAMES_FILE,
    ENGINE_PLAN_FILE,
    INFERENCE_CONFIG_FILE,
    KEYPOINT_DETECTION_TASK_TYPE,
    KEYPOINTS_METADATA_FILE,
    MODEL_CONFIG_FILE,
    TRT_CONFIG_FILE,
    WEIGHTS_ONNX_FILE,
)
from inference_cli.lib.enterprise.inference_compiler.core.compilation_handlers.utils import (
    download_model_package,
    execute_compilation,
    get_training_input_size,
    register_model_package_artefacts,
    safe_negotiate_model_packages,
)
from inference_cli.lib.enterprise.inference_compiler.core.entities import (
    CompilationConfig,
    TRTConfig,
)
from inference_cli.lib.enterprise.inference_compiler.errors import (
    AlreadyCompiledError,
    CompiledPackageRegistrationError,
    ModelVerificationError,
)
from inference_cli.lib.enterprise.inference_compiler.utils.file_system import (
    calculate_local_file_md5,
    dump_json,
    read_json,
)
from inference_cli.lib.enterprise.inference_compiler.utils.logging import (
    print_to_console,
)
from inference_models.weights_providers.entities import ModelMetadata

logger = logging.getLogger("inference_cli.inference_compiler")


def compile_and_register_default_model(
    model_metadata: ModelMetadata,
    models_service_client: ModelsServiceClient,
    compilation_directory: str,
    trt_forward_compatible: bool,
    trt_same_cc_compatible: bool,
    console: Optional[Console],
    compilation_config: CompilationConfig,
) -> None:
    (
        package_with_static_batch_size,
        package_with_dynamic_batch_size,
    ) = safe_negotiate_model_packages(
        model_metadata=model_metadata,
    )
    print_to_console(message="Downloading source model artefacts...", console=console)
    source_packages_directory = os.path.join(compilation_directory, "source_packages")
    expected_files = [INFERENCE_CONFIG_FILE, CLASS_NAMES_FILE, WEIGHTS_ONNX_FILE]
    if model_metadata.task_type == KEYPOINT_DETECTION_TASK_TYPE:
        expected_files.append(KEYPOINTS_METADATA_FILE)
    if package_with_dynamic_batch_size is not None:
        print_to_console(
            message="Found model package with dynamic input dimensions, downloading...",
            console=console,
        )
        package_files = download_model_package(
            model_architecture=model_metadata.model_architecture,
            task_type=model_metadata.task_type,
            model_package=package_with_dynamic_batch_size,
            target_directory=source_packages_directory,
            expected_files=expected_files,
            verify_model=compilation_config.verify_model,
        )
    else:
        print_to_console(
            message="Found model package with static input dimensions, downloading...",
            console=console,
        )
        package_files = download_model_package(
            model_architecture=model_metadata.model_architecture,
            task_type=model_metadata.task_type,
            model_package=package_with_static_batch_size,
            target_directory=source_packages_directory,
            expected_files=expected_files,
            verify_model=compilation_config.verify_model,
        )
    print_to_console(message="Artefacts downloaded.", console=console)
    training_size = get_training_input_size(
        inference_config_path=package_files[INFERENCE_CONFIG_FILE]
    )
    compilation_output_dir = os.path.join(compilation_directory, "compilation_output")
    os.makedirs(compilation_output_dir, exist_ok=True)
    if package_with_dynamic_batch_size is None:
        static_bs_fp32_engine_directory = os.path.join(
            compilation_output_dir, "static_bs_fp32"
        )
        compile_and_register_default_model_trt_variant(
            models_service_client=models_service_client,
            model_metadata=model_metadata,
            compilation_directory=static_bs_fp32_engine_directory,
            local_files=package_files,
            training_size=training_size,
            precision="fp32",
            workspace_size_gb=compilation_config.workspace_size_gb,
            trt_forward_compatible=trt_forward_compatible,
            same_compute_compatibility=trt_same_cc_compatible,
            verify_model=compilation_config.verify_model,
            console=console,
        )
        static_bs_fp16_engine_directory = os.path.join(
            compilation_output_dir, "static_bs_fp16"
        )
        compile_and_register_default_model_trt_variant(
            models_service_client=models_service_client,
            model_metadata=model_metadata,
            compilation_directory=static_bs_fp16_engine_directory,
            local_files=package_files,
            training_size=training_size,
            precision="fp16",
            workspace_size_gb=compilation_config.workspace_size_gb,
            trt_forward_compatible=trt_forward_compatible,
            same_compute_compatibility=trt_same_cc_compatible,
            verify_model=compilation_config.verify_model,
            console=console,
        )
        return None
    dynamic_bs_fp32_engine_directory = os.path.join(
        compilation_output_dir, "dynamic_bs_fp32"
    )
    compile_and_register_default_model_trt_variant(
        models_service_client=models_service_client,
        model_metadata=model_metadata,
        compilation_directory=dynamic_bs_fp32_engine_directory,
        local_files=package_files,
        training_size=training_size,
        precision="fp32",
        workspace_size_gb=compilation_config.workspace_size_gb,
        min_batch_size=compilation_config.min_batch_size,
        opt_batch_size=compilation_config.opt_batch_size,
        max_batch_size=compilation_config.max_batch_size,
        trt_forward_compatible=trt_forward_compatible,
        same_compute_compatibility=trt_same_cc_compatible,
        verify_model=compilation_config.verify_model,
        console=console,
    )
    dynamic_bs_fp16_engine_directory = os.path.join(
        compilation_output_dir, "dynamic_bs_fp16"
    )
    compile_and_register_default_model_trt_variant(
        models_service_client=models_service_client,
        model_metadata=model_metadata,
        compilation_directory=dynamic_bs_fp16_engine_directory,
        local_files=package_files,
        training_size=training_size,
        precision="fp16",
        workspace_size_gb=compilation_config.workspace_size_gb,
        min_batch_size=compilation_config.min_batch_size,
        opt_batch_size=compilation_config.opt_batch_size,
        max_batch_size=compilation_config.max_batch_size,
        trt_forward_compatible=trt_forward_compatible,
        same_compute_compatibility=trt_same_cc_compatible,
        verify_model=compilation_config.verify_model,
        console=console,
    )


def compile_and_register_default_model_trt_variant(
    models_service_client: ModelsServiceClient,
    model_metadata: ModelMetadata,
    compilation_directory: str,
    local_files: Dict[str, str],
    training_size: Tuple[int, int],
    precision: Literal["fp32", "fp16"],
    workspace_size_gb: int,
    min_batch_size: Optional[int] = None,
    opt_batch_size: Optional[int] = None,
    max_batch_size: Optional[int] = None,
    trt_forward_compatible: bool = False,
    same_compute_compatibility: bool = False,
    verify_model: Optional[Callable[[str], None]] = None,
    console: Optional[Console] = None,
) -> None:
    print_to_console(
        message=f"Building TRT engine - precision={precision}", console=console
    )
    try:
        file_handles_to_register = [
            CLASS_NAMES_FILE,
            INFERENCE_CONFIG_FILE,
            TRT_CONFIG_FILE,
            ENGINE_PLAN_FILE,
        ]
        if KEYPOINTS_METADATA_FILE in local_files:
            file_handles_to_register.append(KEYPOINTS_METADATA_FILE)

        engine_path, trt_config, registration_response = execute_compilation(
            models_service_client=models_service_client,
            model_id=model_metadata.model_id,
            model_architecture=model_metadata.model_architecture,
            task_type=model_metadata.task_type,
            model_variant=model_metadata.model_variant,
            file_handles_to_register=file_handles_to_register,
            compilation_directory=compilation_directory,
            onnx_path=local_files[WEIGHTS_ONNX_FILE],
            precision=precision,
            model_input_size=training_size,
            workspace_size_gb=workspace_size_gb,
            min_batch_size=min_batch_size,
            opt_batch_size=opt_batch_size,
            max_batch_size=max_batch_size,
            trt_version_compatible=trt_forward_compatible,
            same_compute_compatibility=same_compute_compatibility,
            console=console,
        )
    except AlreadyCompiledError:
        print_to_console(
            message="Model package already registered - skipping", console=console
        )
        return None
    if verify_model is not None:
        print_to_console(message="Verifying compiled artefacts...", console=console)
        verify_model_package(
            model_metadata=model_metadata,
            model_package_id=registration_response.model_package_id,
            trt_config=trt_config,
            inference_config_path=local_files[INFERENCE_CONFIG_FILE],
            class_names_path=local_files[CLASS_NAMES_FILE],
            engine_path=engine_path,
            verify_model=verify_model,
            keypoints_metadata_path=local_files.get(KEYPOINTS_METADATA_FILE),
        )
    register_default_model_package_artefacts(
        registration_response=registration_response,
        trt_config=trt_config,
        inference_config_path=local_files[INFERENCE_CONFIG_FILE],
        class_names_path=local_files[CLASS_NAMES_FILE],
        keypoints_metadata_path=local_files.get(KEYPOINTS_METADATA_FILE),
        engine_path=engine_path,
        compilation_directory=compilation_directory,
        models_service_client=models_service_client,
    )
    print_to_console(
        message="Successfully compiled and registered model package", console=console
    )


def verify_model_package(
    model_metadata: ModelMetadata,
    model_package_id: str,
    trt_config: TRTConfig,
    inference_config_path: str,
    class_names_path: str,
    engine_path: str,
    verify_model: Callable[[str], None],
    keypoints_metadata_path: Optional[str],
) -> None:
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger.info("Verifying model package %s", model_package_id)
            adjusted_inference_config_path = os.path.join(
                tmp_dir, INFERENCE_CONFIG_FILE
            )
            prepare_adjusted_inference_config(
                inference_config_path=inference_config_path,
                target_path=adjusted_inference_config_path,
            )
            trt_config_path = os.path.join(tmp_dir, TRT_CONFIG_FILE)
            dump_json(path=trt_config_path, content=trt_config.model_dump())
            local_class_names_path = os.path.join(tmp_dir, CLASS_NAMES_FILE)
            os.symlink(class_names_path, local_class_names_path)
            local_engine_path = os.path.join(tmp_dir, ENGINE_PLAN_FILE)
            os.symlink(engine_path, local_engine_path)
            model_config_path = os.path.join(tmp_dir, MODEL_CONFIG_FILE)
            model_config = {
                "model_architecture": model_metadata.model_architecture,
                "task_type": model_metadata.task_type,
                "backend_type": "trt",
            }
            dump_json(path=model_config_path, content=model_config)
            if keypoints_metadata_path is not None:
                local_keypoints_metadata_path = os.path.join(
                    tmp_dir, KEYPOINTS_METADATA_FILE
                )
                os.symlink(keypoints_metadata_path, local_keypoints_metadata_path)
            verify_model(tmp_dir)
            logger.info("Model package %s verified", model_package_id)
    except ModelVerificationError as error:
        raise error
    except Exception as error:
        raise ModelVerificationError(
            "Could not verify compiled model correctness"
        ) from error


def register_default_model_package_artefacts(
    registration_response: ModelPackageRegistrationResponse,
    trt_config: TRTConfig,
    inference_config_path: str,
    class_names_path: str,
    keypoints_metadata_path: Optional[str],
    engine_path: str,
    compilation_directory: str,
    models_service_client: ModelsServiceClient,
) -> None:
    try:
        adjusted_inference_config_path = os.path.join(
            compilation_directory, "adjusted_inference_config.json"
        )
        prepare_adjusted_inference_config(
            inference_config_path=inference_config_path,
            target_path=adjusted_inference_config_path,
        )
        trt_config_path = os.path.join(compilation_directory, TRT_CONFIG_FILE)
        dump_json(path=trt_config_path, content=trt_config.model_dump())
        local_files_mapping = {
            INFERENCE_CONFIG_FILE: (
                adjusted_inference_config_path,
                calculate_local_file_md5(file_path=adjusted_inference_config_path),
            ),
            CLASS_NAMES_FILE: (
                class_names_path,
                calculate_local_file_md5(file_path=class_names_path),
            ),
            TRT_CONFIG_FILE: (
                trt_config_path,
                calculate_local_file_md5(file_path=trt_config_path),
            ),
            ENGINE_PLAN_FILE: (
                engine_path,
                calculate_local_file_md5(file_path=engine_path),
            ),
        }
        if keypoints_metadata_path is not None:
            local_files_mapping[KEYPOINTS_METADATA_FILE] = (
                keypoints_metadata_path,
                calculate_local_file_md5(file_path=keypoints_metadata_path),
            )
    except Exception as error:
        logger.exception(
            "Could not register artefacts for package %s",
            registration_response.model_package_id,
        )
        raise CompiledPackageRegistrationError(
            f"Could not register artefacts for package {registration_response.model_package_id}"
        ) from error
    register_model_package_artefacts(
        registration_response=registration_response,
        local_files_mapping=local_files_mapping,
        models_service_client=models_service_client,
    )


def prepare_adjusted_inference_config(
    inference_config_path: str,
    target_path: str,
) -> None:
    inference_config = read_json(inference_config_path)
    inference_config["network_input"]["dynamic_spatial_size_supported"] = False
    inference_config["network_input"]["dynamic_spatial_size_mode"] = None
    dump_json(path=target_path, content=inference_config)
