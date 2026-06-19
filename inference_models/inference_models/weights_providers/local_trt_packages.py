"""Discover locally installed TRT packages under INFERENCE_HOME/models-cache."""

import json
import logging
import os
from typing import List, Optional, Union

from inference_models.errors import FileHashSumMissmatch
from inference_models.models.auto_loaders.entities import BackendType
from inference_models.models.auto_loaders.model_cache_paths import (
    generate_model_cache_root_for_model_id,
    generate_model_package_cache_path,
    generate_shared_blobs_path,
)
from inference_models.utils.download import (
    is_valid_md5_hash,
    verify_hash_sum_of_local_file,
)
from inference_models.weights_providers.entities import (
    JetsonEnvironmentRequirements,
    LocalFileArtefactSpecs,
    ModelPackageMetadata,
    PackageArtefactSpec,
    PackageSourceType,
    Quantization,
    ServerEnvironmentRequirements,
    TRTPackageDetails,
)
from inference_models.weights_providers.local_trt_constants import (
    ALLOWED_LOCAL_TRT_FILE_HANDLES,
    LOCAL_TRT_MANIFEST_FILE,
    LOCAL_TRT_PACKAGE_PREFIX,
)
from inference_models.weights_providers.trt_manifest import (
    GPUServerSpecsV1,
    JetsonMachineSpecsV1,
    TrtModelPackageV1,
    as_version,
)

logger = logging.getLogger(__name__)

ENGINE_PLAN_FILE = "engine.plan"


def discover_local_trt_packages(model_id: str) -> List[ModelPackageMetadata]:
    cache_root = generate_model_cache_root_for_model_id(model_id=model_id)
    if not os.path.isdir(cache_root):
        return []

    shared_blobs_dir = generate_shared_blobs_path()
    discovered: List[ModelPackageMetadata] = []
    for package_id in sorted(os.listdir(cache_root)):
        if not package_id.startswith(LOCAL_TRT_PACKAGE_PREFIX):
            continue
        try:
            package_dir = generate_model_package_cache_path(
                model_id=model_id, package_id=package_id
            )
            metadata = _parse_local_trt_package(
                model_id=model_id,
                package_id=package_id,
                package_dir=package_dir,
                shared_blobs_dir=shared_blobs_dir,
            )
        except Exception as error:
            logger.warning(
                "Skipping unreadable local TRT package model_id=%s package_id=%s error=%s",
                model_id,
                package_id,
                error,
            )
            continue
        if metadata is not None:
            discovered.append(metadata)
    if discovered:
        logger.info(
            "Discovered %s local TRT package(s) for model_id=%s package_ids=%s",
            len(discovered),
            model_id,
            [package.package_id for package in discovered],
        )
    return discovered


def _is_safe_local_trt_file_handle(handle: str) -> bool:
    if handle not in ALLOWED_LOCAL_TRT_FILE_HANDLES:
        return False
    if handle != os.path.basename(handle):
        return False
    return True


def _parse_local_trt_package(
    model_id: str,
    package_id: str,
    package_dir: str,
    shared_blobs_dir: str,
) -> Optional[ModelPackageMetadata]:
    manifest_path = os.path.join(package_dir, LOCAL_TRT_MANIFEST_FILE)
    engine_path = os.path.join(package_dir, ENGINE_PLAN_FILE)
    if not os.path.isfile(manifest_path) or not os.path.isfile(engine_path):
        return None

    try:
        with open(manifest_path, encoding="utf-8") as manifest_file:
            manifest_data = json.load(manifest_file)
        parsed_manifest = TrtModelPackageV1.model_validate(
            manifest_data["packageManifest"]
        )
        file_md5 = manifest_data["files"]
    except Exception as error:
        logger.warning(
            "Skipping invalid local TRT manifest model_id=%s package_id=%s error=%s",
            model_id,
            package_id,
            error,
        )
        return None

    environment_requirements = _environment_requirements_from_manifest(parsed_manifest)
    if environment_requirements is None:
        return None

    package_artefacts = _build_local_package_artefacts(
        model_id=model_id,
        package_id=package_id,
        package_dir=package_dir,
        file_md5=file_md5,
    )
    if package_artefacts is None:
        return None

    trt_package_details = TRTPackageDetails(
        min_dynamic_batch_size=parsed_manifest.min_batch_size,
        opt_dynamic_batch_size=parsed_manifest.opt_batch_size,
        max_dynamic_batch_size=parsed_manifest.max_batch_size,
        same_cc_compatible=parsed_manifest.same_cc_compatible,
        trt_forward_compatible=parsed_manifest.trt_forward_compatible,
        trt_lean_runtime_excluded=parsed_manifest.trt_lean_runtime_excluded,
    )

    return ModelPackageMetadata(
        package_id=package_id,
        backend=BackendType.TRT,
        quantization=Quantization(parsed_manifest.quantization),
        dynamic_batch_size_supported=parsed_manifest.dynamic_batch_size,
        static_batch_size=parsed_manifest.static_batch_size,
        package_artefacts=package_artefacts,
        package_source=PackageSourceType.LOCAL_CACHE,
        environment_requirements=environment_requirements,
        trt_package_details=trt_package_details,
        # Locally compiled engines are not platform-authoritative; require an
        # explicit opt-in (allow_untrusted_packages) to be loaded.
        trusted_source=False,
        cache_model_id=model_id,
        model_features=None,
        recommended_parameters=None,
    )


def _build_local_package_artefacts(
    model_id: str,
    package_id: str,
    package_dir: str,
    file_md5: dict,
) -> Optional[List[PackageArtefactSpec]]:
    package_artefacts: List[PackageArtefactSpec] = []
    for handle, md5_hash in file_md5.items():
        if not _is_safe_local_trt_file_handle(handle=handle):
            logger.warning(
                "Local TRT package has disallowed file handle model_id=%s package_id=%s handle=%s",
                model_id,
                package_id,
                handle,
            )
            return None
        if not is_valid_md5_hash(md5_hash):
            logger.warning(
                "Local TRT package has invalid md5 model_id=%s package_id=%s handle=%s",
                model_id,
                package_id,
                handle,
            )
            return None
        package_file_path = os.path.join(package_dir, handle)
        if not os.path.isfile(package_file_path):
            logger.warning(
                "Local TRT package missing artefact model_id=%s package_id=%s handle=%s",
                model_id,
                package_id,
                handle,
            )
            return None
        # md5 here detects on-disk corruption only; the manifest is locally
        # written so this is not a tamper guarantee. Authenticity is gated by
        # trusted_source=False (requires allow_untrusted_packages to load).
        try:
            verify_hash_sum_of_local_file(
                url=f"local-cache://{handle}",
                file_path=package_file_path,
                expected_md5_hash=md5_hash,
            )
        except FileHashSumMissmatch as error:
            logger.warning(
                "Local TRT package failed md5 verification model_id=%s package_id=%s handle=%s error=%s",
                model_id,
                package_id,
                handle,
                error,
            )
            return None
        package_artefacts.append(
            LocalFileArtefactSpecs(file_handle=handle, md5_hash=md5_hash)
        )
    return package_artefacts


def _environment_requirements_from_manifest(
    parsed_manifest: TrtModelPackageV1,
) -> Optional[Union[ServerEnvironmentRequirements, JetsonEnvironmentRequirements]]:
    if parsed_manifest.machine_type == "gpu-server":
        if not isinstance(parsed_manifest.machine_specs, GPUServerSpecsV1):
            return None
        return ServerEnvironmentRequirements(
            cuda_device_cc=as_version(parsed_manifest.cuda_device_cc),
            cuda_device_name=parsed_manifest.cuda_device_type,
            driver_version=as_version(parsed_manifest.machine_specs.driver_version),
            cuda_version=as_version(parsed_manifest.cuda_version),
            trt_version=as_version(parsed_manifest.trt_version),
            os_version=parsed_manifest.machine_specs.os_version,
        )
    if parsed_manifest.machine_type == "jetson":
        if not isinstance(parsed_manifest.machine_specs, JetsonMachineSpecsV1):
            return None
        return JetsonEnvironmentRequirements(
            cuda_device_cc=as_version(parsed_manifest.cuda_device_cc),
            cuda_device_name=parsed_manifest.cuda_device_type,
            l4t_version=as_version(parsed_manifest.machine_specs.l4t_version),
            jetson_product_name=parsed_manifest.machine_specs.device_name,
            cuda_version=as_version(parsed_manifest.cuda_version),
            trt_version=as_version(parsed_manifest.trt_version),
            driver_version=as_version(parsed_manifest.machine_specs.driver_version),
        )
    return None
