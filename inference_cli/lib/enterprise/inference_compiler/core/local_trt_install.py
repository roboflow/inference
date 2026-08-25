"""Install compiled TRT artefacts into inference-models models-cache for local lookup."""

import hashlib
import json
import logging
import os
import shutil
from typing import Dict, Optional, Tuple

from inference_cli.lib.enterprise.inference_compiler.constants import (
    CLASS_NAMES_FILE,
    ENGINE_PLAN_FILE,
    INFERENCE_CONFIG_FILE,
    KEYPOINTS_METADATA_FILE,
    TRT_CONFIG_FILE,
)
from inference_cli.lib.enterprise.inference_compiler.core.entities import (
    TRTConfig,
    TRTModelPackageV1,
)
from inference_cli.lib.enterprise.inference_compiler.utils.file_system import (
    calculate_local_file_md5,
    dump_json,
    read_json,
)
from inference_models.weights_providers.local_trt_constants import (
    LOCAL_TRT_MANIFEST_FILE,
    LOCAL_TRT_PACKAGE_PREFIX,
)

logger = logging.getLogger("inference_cli.inference_compiler")


def _prepare_adjusted_inference_config(
    inference_config_path: str,
    target_path: str,
) -> None:
    """Mirror of compilation_handlers.default.prepare_adjusted_inference_config.

    Kept local so this module never imports the compilation handlers, which
    require tensorrt at import time — the installer itself only touches files.
    """
    inference_config = read_json(inference_config_path)
    inference_config["network_input"]["dynamic_spatial_size_supported"] = False
    inference_config["network_input"]["dynamic_spatial_size_mode"] = None
    dump_json(path=target_path, content=inference_config)


def local_package_id_for_manifest(package_manifest: TRTModelPackageV1) -> str:
    manifest_bytes = json.dumps(
        package_manifest.model_dump(by_alias=True, mode="json", exclude_none=True),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.blake2s(manifest_bytes, digest_size=6).hexdigest()
    return f"{LOCAL_TRT_PACKAGE_PREFIX}{digest}"


def install_compiled_trt_package(
    model_id: str,
    model_architecture: str,
    task_type: Optional[str],
    package_manifest: TRTModelPackageV1,
    trt_config: TRTConfig,
    engine_path: str,
    inference_config_path: str,
    class_names_path: str,
    compilation_directory: str,
    keypoints_metadata_path: Optional[str] = None,
) -> Tuple[str, str]:
    """Install compiled TRT into models-cache. Returns (package_id, install_dir)."""
    from inference_models.models.auto_loaders.core import (
        generate_model_package_cache_path,
        generate_shared_blobs_path,
    )

    package_id = local_package_id_for_manifest(package_manifest)
    install_dir = generate_model_package_cache_path(
        model_id=model_id, package_id=package_id
    )
    if os.path.isdir(install_dir):
        shutil.rmtree(install_dir, ignore_errors=True)
    os.makedirs(install_dir, exist_ok=True)

    adjusted_inference_config_path = os.path.join(
        compilation_directory, "adjusted_inference_config.json"
    )
    _prepare_adjusted_inference_config(
        inference_config_path=inference_config_path,
        target_path=adjusted_inference_config_path,
    )
    trt_config_path = os.path.join(compilation_directory, TRT_CONFIG_FILE)
    dump_json(path=trt_config_path, content=trt_config.model_dump())

    source_files = {
        INFERENCE_CONFIG_FILE: adjusted_inference_config_path,
        CLASS_NAMES_FILE: class_names_path,
        TRT_CONFIG_FILE: trt_config_path,
        ENGINE_PLAN_FILE: engine_path,
    }
    if keypoints_metadata_path is not None:
        source_files[KEYPOINTS_METADATA_FILE] = keypoints_metadata_path

    file_md5: Dict[str, str] = {}
    shared_blobs_dir = generate_shared_blobs_path()
    os.makedirs(shared_blobs_dir, exist_ok=True)
    for handle, source_path in source_files.items():
        md5_hash = calculate_local_file_md5(file_path=source_path)
        file_md5[handle] = md5_hash
        shared_blob_path = os.path.join(shared_blobs_dir, md5_hash)
        if not os.path.isfile(shared_blob_path):
            shutil.copy2(source_path, shared_blob_path)
        # Materialize the artefact as a REGULAR file. Local TRT discovery
        # (inference_models.weights_providers.local_trt_packages) rejects
        # symlinked artefacts, so a symlink here would make the installed
        # package silently invisible to the loader.
        shutil.copy2(source_path, os.path.join(install_dir, handle))

    manifest_payload = {
        "packageManifest": package_manifest.model_dump(
            by_alias=True, mode="json", exclude_none=True
        ),
        "files": file_md5,
        "modelArchitecture": model_architecture,
        "taskType": task_type,
    }
    manifest_path = os.path.join(install_dir, LOCAL_TRT_MANIFEST_FILE)
    dump_json(path=manifest_path, content=manifest_payload)

    try:
        _record_package_in_offline_registry(
            model_id=model_id,
            model_architecture=model_architecture,
            task_type=task_type,
            package_id=package_id,
            manifest_payload=manifest_payload,
            manifest_path=manifest_path,
            file_md5=file_md5,
        )
    except Exception as error:
        logger.warning(
            "Could not append offline-weights registry record for model_id=%s "
            "package_id=%s error=%s - the package is installed and discoverable "
            "online, but will not be loadable in OFFLINE_MODE until warmed up.",
            model_id,
            package_id,
            error,
        )

    logger.info(
        "Installed local TRT package model_id=%s package_id=%s path=%s backend=trt "
        "compiled=true installed_local=true files=%s",
        model_id,
        package_id,
        install_dir,
        list(source_files.keys()),
    )
    return package_id, install_dir


def _record_package_in_offline_registry(
    model_id: str,
    model_architecture: str,
    task_type: Optional[str],
    package_id: str,
    manifest_payload: dict,
    manifest_path: str,
    file_md5: Dict[str, str],
) -> None:
    """Append the installed package to the offline-weights registry.

    Builds the same ``ModelPackageMetadata`` that local TRT discovery
    (``_parse_local_trt_package``) would produce for this package, so
    OFFLINE_MODE loads see exactly what an online discovery would report.
    """
    from inference_models.models.auto_loaders.entities import BackendType
    from inference_models.weights_providers import offline_registry
    from inference_models.weights_providers.entities import (
        LocalFileArtefactSpecs,
        ModelMetadata,
        ModelPackageMetadata,
        PackageSourceType,
        Quantization,
        TRTPackageDetails,
    )
    from inference_models.weights_providers.local_trt_packages import (
        _environment_requirements_from_manifest,
    )
    from inference_models.weights_providers.trt_manifest import TrtModelPackageV1

    parsed_manifest = TrtModelPackageV1.model_validate(
        manifest_payload["packageManifest"]
    )
    environment_requirements = _environment_requirements_from_manifest(parsed_manifest)
    trt_package_details = TRTPackageDetails(
        min_dynamic_batch_size=parsed_manifest.min_batch_size,
        opt_dynamic_batch_size=parsed_manifest.opt_batch_size,
        max_dynamic_batch_size=parsed_manifest.max_batch_size,
        same_cc_compatible=parsed_manifest.same_cc_compatible,
        trt_forward_compatible=parsed_manifest.trt_forward_compatible,
        trt_lean_runtime_excluded=parsed_manifest.trt_lean_runtime_excluded,
    )
    package_artefacts = [
        LocalFileArtefactSpecs(file_handle=handle, md5_hash=md5_hash)
        for handle, md5_hash in file_md5.items()
    ]
    package_artefacts.append(
        LocalFileArtefactSpecs(
            file_handle=LOCAL_TRT_MANIFEST_FILE,
            # Same identity discovery computes: md5 of the manifest file bytes.
            md5_hash=calculate_local_file_md5(file_path=manifest_path),
        )
    )
    package_metadata = ModelPackageMetadata(
        package_id=package_id,
        backend=BackendType.TRT,
        quantization=Quantization(parsed_manifest.quantization),
        dynamic_batch_size_supported=parsed_manifest.dynamic_batch_size,
        static_batch_size=parsed_manifest.static_batch_size,
        package_artefacts=package_artefacts,
        package_source=PackageSourceType.LOCAL_CACHE,
        environment_requirements=environment_requirements,
        trt_package_details=trt_package_details,
        # Locally compiled engines are not platform-authoritative; matches the
        # trust level local TRT discovery assigns to this package.
        trusted_source=False,
        cache_model_id=model_id,
        model_features=None,
        recommended_parameters=None,
    )
    model_metadata = ModelMetadata(
        model_id=model_id,
        model_architecture=model_architecture,
        model_packages=[package_metadata],
        task_type=task_type,
    )
    offline_registry.record_successful_load(
        model_metadata=model_metadata,
        requested_model_id=model_id,
        proven_package_id=package_id,
        source=offline_registry.RECORD_SOURCE_CLI_INSTALL,
    )
