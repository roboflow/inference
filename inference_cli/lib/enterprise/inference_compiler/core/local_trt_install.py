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
from inference_cli.lib.enterprise.inference_compiler.core.entities import TRTConfig, TRTModelPackageV1
from inference_cli.lib.enterprise.inference_compiler.utils.file_system import (
    calculate_local_file_md5,
    dump_json,
)

from inference_models.weights_providers.local_trt_constants import (
    LOCAL_TRT_MANIFEST_FILE,
    LOCAL_TRT_PACKAGE_PREFIX,
)

logger = logging.getLogger("inference_cli.inference_compiler")


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
    from inference_cli.lib.enterprise.inference_compiler.core.compilation_handlers.default import (
        prepare_adjusted_inference_config,
    )
    from inference_models.models.auto_loaders.core import (
        create_symlinks_to_shared_blobs,
        generate_model_package_cache_path,
        generate_shared_blobs_path,
    )

    package_id = local_package_id_for_manifest(package_manifest)
    install_dir = generate_model_package_cache_path(model_id=model_id, package_id=package_id)
    if os.path.isdir(install_dir):
        shutil.rmtree(install_dir, ignore_errors=True)
    os.makedirs(install_dir, exist_ok=True)

    adjusted_inference_config_path = os.path.join(
        compilation_directory, "adjusted_inference_config.json"
    )
    prepare_adjusted_inference_config(
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
    shared_files_mapping: Dict[str, str] = {}
    for handle, source_path in source_files.items():
        md5_hash = calculate_local_file_md5(file_path=source_path)
        file_md5[handle] = md5_hash
        shared_blob_path = os.path.join(shared_blobs_dir, md5_hash)
        if not os.path.isfile(shared_blob_path):
            shutil.copy2(source_path, shared_blob_path)
        shared_files_mapping[handle] = shared_blob_path

    create_symlinks_to_shared_blobs(
        model_dir=install_dir,
        shared_files_mapping=shared_files_mapping,
    )

    manifest_payload = {
        "packageManifest": package_manifest.model_dump(
            by_alias=True, mode="json", exclude_none=True
        ),
        "files": file_md5,
        "modelArchitecture": model_architecture,
        "taskType": task_type,
    }
    dump_json(
        path=os.path.join(install_dir, LOCAL_TRT_MANIFEST_FILE),
        content=manifest_payload,
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
