"""Discover locally installed TRT packages under INFERENCE_HOME/models-cache."""

import hashlib
import json
import logging
import os
import re
from typing import List, Optional

from inference_models.models.auto_loaders.entities import BackendType
from inference_models.weights_providers.entities import (
    FileDownloadSpecs,
    JetsonEnvironmentRequirements,
    ModelPackageMetadata,
    Quantization,
    ServerEnvironmentRequirements,
    TRTPackageDetails,
)

logger = logging.getLogger(__name__)

LOCAL_TRT_PACKAGE_PREFIX = "localtrt"
LOCAL_TRT_MANIFEST_FILE = "local_trt_package_manifest.json"


def _slugify_model_id(model_id: str) -> str:
    model_id_slug = re.sub(r"[^A-Za-z0-9_-]+", "-", model_id)
    model_id_slug = re.sub(r"[_-]{2,}", "-", model_id_slug)
    if not model_id_slug:
        model_id_slug = "special-char-only-model-id"
    if len(model_id_slug) > 48:
        model_id_slug = model_id_slug[:48]
    digest = hashlib.blake2s(model_id.encode("utf-8"), digest_size=4).hexdigest()
    return f"{model_id_slug}-{digest}"


def discover_local_trt_packages(model_id: str) -> List[ModelPackageMetadata]:
    model_slug = _slugify_model_id(model_id=model_id)
    cache_root = os.path.join(
        os.environ.get("INFERENCE_HOME", "/tmp/cache"), "models-cache", model_slug
    )
    if not os.path.isdir(cache_root):
        return []

    discovered: List[ModelPackageMetadata] = []
    for package_id in sorted(os.listdir(cache_root)):
        if not package_id.startswith(LOCAL_TRT_PACKAGE_PREFIX):
            continue
        package_dir = os.path.join(cache_root, package_id)
        metadata = _parse_local_trt_package(
            model_id=model_id,
            package_id=package_id,
            package_dir=package_dir,
        )
        if metadata is not None:
            discovered.append(metadata)
    if discovered:
        logger.info(
            "Discovered %s local TRT package(s) for model_id=%s package_ids=%s",
            len(discovered),
            model_id,
            [p.package_id for p in discovered],
        )
    return discovered


def _parse_local_trt_package(
    model_id: str,
    package_id: str,
    package_dir: str,
) -> Optional[ModelPackageMetadata]:
    manifest_path = os.path.join(package_dir, LOCAL_TRT_MANIFEST_FILE)
    engine_path = os.path.join(package_dir, "engine.plan")
    if not os.path.isfile(manifest_path) or not os.path.isfile(engine_path):
        return None

    try:
        with open(manifest_path, encoding="utf-8") as manifest_file:
            manifest_data = json.load(manifest_file)
        from inference_models.weights_providers.roboflow import (
            GPUServerSpecsV1,
            JetsonMachineSpecsV1,
            TrtModelPackageV1,
            as_version,
        )

        parsed_manifest = TrtModelPackageV1.model_validate(manifest_data["packageManifest"])
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

    package_artefacts = []
    for handle, md5_hash in file_md5.items():
        file_path = os.path.join(package_dir, handle)
        if not os.path.isfile(file_path):
            logger.warning(
                "Local TRT package missing file model_id=%s package_id=%s handle=%s",
                model_id,
                package_id,
                handle,
            )
            return None
        package_artefacts.append(
            FileDownloadSpecs(
                download_url=f"file://{file_path}",
                file_handle=handle,
                md5_hash=md5_hash,
            )
        )

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
        environment_requirements=environment_requirements,
        trt_package_details=trt_package_details,
        trusted_source=True,
        model_features=None,
        recommended_parameters=None,
    )


def _environment_requirements_from_manifest(parsed_manifest):
    from inference_models.weights_providers.roboflow import (
        GPUServerSpecsV1,
        JetsonMachineSpecsV1,
        as_version,
    )

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
