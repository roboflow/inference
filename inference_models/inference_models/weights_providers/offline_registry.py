"""Registry of models proven to load locally, powering `roboflow-offline-weights`.

The registry is written during ONLINE runs with ``OFFLINE_MODE_WARM_UP=True``
(and by the local TRT compiler installer) and read in ``OFFLINE_MODE`` by the
``roboflow-offline-weights`` provider. Each record captures the original
provider response for one canonical model — every package with its environment
requirements and backend details — so OFFLINE loads re-run the exact same
auto-negotiation as ONLINE loads, just against recorded metadata.

Storage layout: one JSON file per canonical model id under
``${INFERENCE_HOME}/offline-weights-registry/``, mutated with a file lock and
an atomic temp-write + rename. Reads are tolerant: unknown fields are ignored
and records with a newer ``format_version`` are skipped with a warning.
"""

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from filelock import FileLock
from packaging.version import InvalidVersion, Version

from inference_models.configuration import (
    FILE_LOCK_ACQUIRE_TIMEOUT,
    INFERENCE_HOME,
)
from inference_models.errors import BaseInferenceModelsError
from inference_models.logger import LOGGER
from inference_models.models.auto_loaders.entities import BackendType
from inference_models.models.auto_loaders.model_cache_paths import (
    resolve_existing_model_package_cache_path,
    slugify_model_id_to_os_safe_format,
)
from inference_models.utils.file_system import read_json
from inference_models.weights_providers.entities import (
    JetsonEnvironmentRequirements,
    LocalFileArtefactSpecs,
    ModelDependency,
    ModelMetadata,
    ModelPackageMetadata,
    ONNXPackageDetails,
    PackageSourceType,
    Quantization,
    RecommendedParameters,
    ServerEnvironmentRequirements,
    TorchScriptPackageDetails,
    TRTPackageDetails,
)

REGISTRY_FORMAT_VERSION = 1
REGISTRY_DIR_NAME = "offline-weights-registry"

RECORD_SOURCE_WARMUP = "warmup"
RECORD_SOURCE_CLI_INSTALL = "cli-install"


def generate_offline_registry_dir() -> str:
    return os.path.abspath(os.path.join(INFERENCE_HOME, REGISTRY_DIR_NAME))


def _record_path(canonical_model_id: str) -> str:
    record_file_name = (
        f"{slugify_model_id_to_os_safe_format(model_id=canonical_model_id)}.json"
    )
    return os.path.join(generate_offline_registry_dir(), record_file_name)


def _version_to_str(value: Optional[Version]) -> Optional[str]:
    return None if value is None else str(value)


def _str_to_version(value: Optional[str]) -> Optional[Version]:
    if value is None:
        return None
    try:
        return Version(value)
    except InvalidVersion:
        return None


def _serialize_environment_requirements(
    requirements: Optional[
        Union[ServerEnvironmentRequirements, JetsonEnvironmentRequirements]
    ],
) -> Optional[dict]:
    if requirements is None:
        return None
    if isinstance(requirements, JetsonEnvironmentRequirements):
        return {
            "kind": "jetson",
            "cuda_device_cc": _version_to_str(requirements.cuda_device_cc),
            "cuda_device_name": requirements.cuda_device_name,
            "l4t_version": _version_to_str(requirements.l4t_version),
            "jetson_product_name": requirements.jetson_product_name,
            "cuda_version": _version_to_str(requirements.cuda_version),
            "trt_version": _version_to_str(requirements.trt_version),
            "driver_version": _version_to_str(requirements.driver_version),
        }
    return {
        "kind": "gpu-server",
        "cuda_device_cc": _version_to_str(requirements.cuda_device_cc),
        "cuda_device_name": requirements.cuda_device_name,
        "driver_version": _version_to_str(requirements.driver_version),
        "cuda_version": _version_to_str(requirements.cuda_version),
        "trt_version": _version_to_str(requirements.trt_version),
        "os_version": requirements.os_version,
    }


def _deserialize_environment_requirements(
    payload: Optional[dict],
) -> Optional[Union[ServerEnvironmentRequirements, JetsonEnvironmentRequirements]]:
    if not isinstance(payload, dict):
        return None
    cuda_device_cc = _str_to_version(payload.get("cuda_device_cc"))
    if cuda_device_cc is None or not isinstance(payload.get("cuda_device_name"), str):
        return None
    if payload.get("kind") == "jetson":
        l4t_version = _str_to_version(payload.get("l4t_version"))
        if l4t_version is None:
            return None
        return JetsonEnvironmentRequirements(
            cuda_device_cc=cuda_device_cc,
            cuda_device_name=payload["cuda_device_name"],
            l4t_version=l4t_version,
            jetson_product_name=payload.get("jetson_product_name"),
            cuda_version=_str_to_version(payload.get("cuda_version")),
            trt_version=_str_to_version(payload.get("trt_version")),
            driver_version=_str_to_version(payload.get("driver_version")),
        )
    return ServerEnvironmentRequirements(
        cuda_device_cc=cuda_device_cc,
        cuda_device_name=payload["cuda_device_name"],
        driver_version=_str_to_version(payload.get("driver_version")),
        cuda_version=_str_to_version(payload.get("cuda_version")),
        trt_version=_str_to_version(payload.get("trt_version")),
        os_version=payload.get("os_version"),
    )


def _serialize_package(package: ModelPackageMetadata) -> Optional[dict]:
    artifacts = []
    for artefact in package.package_artefacts:
        artifacts.append(
            {
                "file_handle": artefact.file_handle,
                "md5_hash": artefact.md5_hash,
            }
        )
    trt_details = None
    if package.trt_package_details is not None:
        trt_details = {
            "min_dynamic_batch_size": package.trt_package_details.min_dynamic_batch_size,
            "opt_dynamic_batch_size": package.trt_package_details.opt_dynamic_batch_size,
            "max_dynamic_batch_size": package.trt_package_details.max_dynamic_batch_size,
            "same_cc_compatible": package.trt_package_details.same_cc_compatible,
            "trt_forward_compatible": package.trt_package_details.trt_forward_compatible,
            "trt_lean_runtime_excluded": package.trt_package_details.trt_lean_runtime_excluded,
        }
    onnx_details = None
    if package.onnx_package_details is not None:
        onnx_details = {
            "opset": package.onnx_package_details.opset,
            "incompatible_providers": package.onnx_package_details.incompatible_providers,
        }
    torch_script_details = None
    if package.torch_script_package_details is not None:
        torch_script_details = {
            "supported_device_types": sorted(
                package.torch_script_package_details.supported_device_types
            ),
            "torch_version": _version_to_str(
                package.torch_script_package_details.torch_version
            ),
            "torch_vision_version": _version_to_str(
                package.torch_script_package_details.torch_vision_version
            ),
        }
    return {
        "package_id": package.package_id,
        "backend": package.backend.value,
        "quantization": (
            package.quantization.value if package.quantization is not None else None
        ),
        "dynamic_batch_size_supported": package.dynamic_batch_size_supported,
        "static_batch_size": package.static_batch_size,
        "trusted_source": package.trusted_source,
        "model_features": package.model_features,
        "recommended_parameters": (
            package.recommended_parameters.model_dump(mode="json")
            if package.recommended_parameters is not None
            else None
        ),
        "cache_model_id": package.cache_model_id,
        "environment_requirements": _serialize_environment_requirements(
            package.environment_requirements
        ),
        "trt_package_details": trt_details,
        "onnx_package_details": onnx_details,
        "torch_script_package_details": torch_script_details,
        "artifacts": artifacts,
    }


def _deserialize_package(payload: dict) -> Optional[ModelPackageMetadata]:
    try:
        artifacts = []
        for artefact in payload.get("artifacts") or []:
            file_handle = artefact.get("file_handle")
            md5_hash = artefact.get("md5_hash")
            if not isinstance(file_handle, str) or not isinstance(md5_hash, str):
                return None
            artifacts.append(
                LocalFileArtefactSpecs(file_handle=file_handle, md5_hash=md5_hash)
            )
        trt_details = None
        trt_payload = payload.get("trt_package_details")
        if isinstance(trt_payload, dict):
            trt_details = TRTPackageDetails(
                min_dynamic_batch_size=trt_payload.get("min_dynamic_batch_size"),
                opt_dynamic_batch_size=trt_payload.get("opt_dynamic_batch_size"),
                max_dynamic_batch_size=trt_payload.get("max_dynamic_batch_size"),
                same_cc_compatible=bool(trt_payload.get("same_cc_compatible", False)),
                trt_forward_compatible=bool(
                    trt_payload.get("trt_forward_compatible", False)
                ),
                trt_lean_runtime_excluded=bool(
                    trt_payload.get("trt_lean_runtime_excluded", False)
                ),
            )
        onnx_details = None
        onnx_payload = payload.get("onnx_package_details")
        if isinstance(onnx_payload, dict) and isinstance(
            onnx_payload.get("opset"), int
        ):
            onnx_details = ONNXPackageDetails(
                opset=onnx_payload["opset"],
                incompatible_providers=onnx_payload.get("incompatible_providers"),
            )
        torch_script_details = None
        torch_script_payload = payload.get("torch_script_package_details")
        if isinstance(torch_script_payload, dict):
            torch_version = _str_to_version(torch_script_payload.get("torch_version"))
            if torch_version is not None:
                torch_script_details = TorchScriptPackageDetails(
                    supported_device_types=set(
                        torch_script_payload.get("supported_device_types") or []
                    ),
                    torch_version=torch_version,
                    torch_vision_version=_str_to_version(
                        torch_script_payload.get("torch_vision_version")
                    ),
                )
        quantization = None
        if payload.get("quantization") is not None:
            quantization = Quantization(payload["quantization"])
        recommended_parameters = None
        if isinstance(payload.get("recommended_parameters"), dict):
            recommended_parameters = RecommendedParameters.model_validate(
                payload["recommended_parameters"]
            )
        return ModelPackageMetadata(
            package_id=payload["package_id"],
            backend=BackendType(payload["backend"]),
            package_artefacts=artifacts,
            package_source=PackageSourceType.LOCAL_CACHE,
            quantization=quantization,
            dynamic_batch_size_supported=payload.get("dynamic_batch_size_supported"),
            static_batch_size=payload.get("static_batch_size"),
            trt_package_details=trt_details,
            onnx_package_details=onnx_details,
            torch_script_package_details=torch_script_details,
            trusted_source=payload.get("trusted_source") is True,
            environment_requirements=_deserialize_environment_requirements(
                payload.get("environment_requirements")
            ),
            model_features=payload.get("model_features"),
            recommended_parameters=recommended_parameters,
            cache_model_id=payload.get("cache_model_id"),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _resolve_package_dir(
    cache_model_id: str,
    package_id: str,
) -> Optional[str]:
    """Resolve an existing package dir, treating malformed identities as absent."""

    try:
        return resolve_existing_model_package_cache_path(
            model_id=cache_model_id,
            package_id=package_id,
            allow_unattributed_local_cache=True,
        )
    except (BaseInferenceModelsError, OSError):
        return None


def _md5_of_file(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_with_backfilled_md5(
    model_metadata: ModelMetadata,
    package: ModelPackageMetadata,
) -> Optional[ModelPackageMetadata]:
    """Return the package with every artefact carrying an MD5 identity.

    Provider responses may declare artefacts without hashes. The offline
    registry requires an identity for every file, so missing hashes are
    computed once from the files materialized during the warm load.
    """

    if all(artefact.md5_hash is not None for artefact in package.package_artefacts):
        return package
    package_dir = _resolve_package_dir(
        cache_model_id=package.cache_model_id or model_metadata.model_id,
        package_id=package.package_id,
    )
    if package_dir is None:
        return None
    backfilled_artefacts = []
    for artefact in package.package_artefacts:
        if artefact.md5_hash is not None:
            backfilled_artefacts.append(
                LocalFileArtefactSpecs(
                    file_handle=artefact.file_handle,
                    md5_hash=artefact.md5_hash,
                )
            )
            continue
        artefact_path = os.path.join(package_dir, artefact.file_handle)
        if not os.path.isfile(artefact_path):
            return None
        try:
            md5_hash = _md5_of_file(path=artefact_path)
        except OSError:
            return None
        backfilled_artefacts.append(
            LocalFileArtefactSpecs(
                file_handle=artefact.file_handle,
                md5_hash=md5_hash,
            )
        )
    return ModelPackageMetadata(
        package_id=package.package_id,
        backend=package.backend,
        package_artefacts=backfilled_artefacts,
        package_source=package.package_source,
        quantization=package.quantization,
        dynamic_batch_size_supported=package.dynamic_batch_size_supported,
        static_batch_size=package.static_batch_size,
        trt_package_details=package.trt_package_details,
        onnx_package_details=package.onnx_package_details,
        torch_script_package_details=package.torch_script_package_details,
        trusted_source=package.trusted_source,
        environment_requirements=package.environment_requirements,
        model_features=package.model_features,
        recommended_parameters=package.recommended_parameters,
        cache_model_id=package.cache_model_id,
    )


def _read_record_file(record_path: str) -> Optional[dict]:
    if not os.path.isfile(record_path) or os.path.islink(record_path):
        return None
    try:
        content = read_json(path=record_path)
    except (OSError, ValueError) as error:
        LOGGER.warning(
            "Could not decode offline-weights registry record %s: %s",
            record_path,
            error,
        )
        return None
    if not isinstance(content, dict):
        return None
    format_version = content.get("format_version")
    if not isinstance(format_version, int) or isinstance(format_version, bool):
        return None
    if format_version > REGISTRY_FORMAT_VERSION:
        LOGGER.warning(
            "Skipping offline-weights registry record %s written by a newer "
            "inference-models release (format_version=%s > %s).",
            record_path,
            format_version,
            REGISTRY_FORMAT_VERSION,
        )
        return None
    return content


def _write_record_file(record_path: str, content: dict) -> None:
    record_dir = os.path.dirname(record_path)
    os.makedirs(record_dir, exist_ok=True)
    if os.path.islink(record_dir) or os.path.islink(record_path):
        raise OSError(
            f"Refusing to write offline-weights registry record through a "
            f"symlink: {record_path}"
        )
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=record_dir,
            prefix=f".{os.path.basename(record_path)}.",
            suffix=".tmp",
            delete=False,
        ) as file_handle:
            temporary_path = file_handle.name
            json.dump(content, file_handle, indent=2, sort_keys=True)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary_path, record_path)
        temporary_path = None
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _record_lock_path(record_path: str) -> str:
    record_dir, record_file_name = os.path.split(record_path)
    return os.path.join(record_dir, f".{record_file_name}.lock")


def record_successful_load(
    model_metadata: ModelMetadata,
    requested_model_id: str,
    proven_package_id: str,
    source: str = RECORD_SOURCE_WARMUP,
    file_lock_acquire_timeout: int = FILE_LOCK_ACQUIRE_TIMEOUT,
) -> bool:
    """Merge a proven model load into the offline-weights registry.

    Returns True when the record was written, False when it was skipped
    (e.g. an artefact required an MD5 backfill but was not materialized).
    """

    canonical_model_id = model_metadata.model_id
    serialized_packages: Dict[str, dict] = {}
    for package in model_metadata.model_packages:
        complete_package = _package_with_backfilled_md5(
            model_metadata=model_metadata,
            package=package,
        )
        if complete_package is None:
            if package.package_id == proven_package_id:
                LOGGER.warning(
                    "Not registering offline record for %s: proven package %s "
                    "has artefacts without a computable MD5 identity.",
                    canonical_model_id,
                    proven_package_id,
                )
                return False
            continue
        serialized = _serialize_package(package=complete_package)
        if serialized is not None:
            serialized_packages[complete_package.package_id] = serialized
    if proven_package_id not in serialized_packages:
        LOGGER.warning(
            "Not registering offline record for %s: proven package %s missing "
            "from provider metadata.",
            canonical_model_id,
            proven_package_id,
        )
        return False

    record_path = _record_path(canonical_model_id=canonical_model_id)
    os.makedirs(os.path.dirname(record_path), exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    with FileLock(
        _record_lock_path(record_path=record_path),
        timeout=file_lock_acquire_timeout,
    ):
        existing = _read_record_file(record_path=record_path) or {}
        packages: Dict[str, dict] = {}
        for package_payload in existing.get("packages") or []:
            package_id = (
                package_payload.get("package_id")
                if isinstance(package_payload, dict)
                else None
            )
            if isinstance(package_id, str):
                packages[package_id] = package_payload
        packages.update(serialized_packages)
        aliases = set(existing.get("requested_aliases") or [])
        if requested_model_id != canonical_model_id:
            aliases.add(requested_model_id)
        proven = dict(existing.get("proven") or {})
        proven[proven_package_id] = {"last_proven_at": now}
        content = {
            "format_version": REGISTRY_FORMAT_VERSION,
            "canonical_model_id": canonical_model_id,
            "requested_aliases": sorted(aliases),
            "recorded_at": now,
            "source": source,
            "model": {
                "model_architecture": model_metadata.model_architecture,
                "task_type": model_metadata.task_type,
                "model_variant": model_metadata.model_variant,
                "model_dependencies": [
                    dependency.model_dump(mode="json")
                    for dependency in (model_metadata.model_dependencies or [])
                ],
                "recommended_parameters": (
                    model_metadata.recommended_parameters.model_dump(mode="json")
                    if model_metadata.recommended_parameters is not None
                    else None
                ),
            },
            "packages": [
                packages[package_id] for package_id in sorted(packages.keys())
            ],
            "proven": proven,
        }
        _write_record_file(record_path=record_path, content=content)
    return True


def _record_to_model_metadata(record: dict) -> Optional[ModelMetadata]:
    canonical_model_id = record.get("canonical_model_id")
    model_payload = record.get("model")
    if not isinstance(canonical_model_id, str) or not isinstance(model_payload, dict):
        return None
    model_architecture = model_payload.get("model_architecture")
    if not isinstance(model_architecture, str):
        return None
    packages = []
    for package_payload in record.get("packages") or []:
        if not isinstance(package_payload, dict):
            continue
        package = _deserialize_package(payload=package_payload)
        if package is not None:
            packages.append(package)
    if not packages:
        return None
    model_dependencies = None
    raw_dependencies = model_payload.get("model_dependencies")
    if raw_dependencies:
        model_dependencies = []
        for dependency_payload in raw_dependencies:
            try:
                model_dependencies.append(
                    ModelDependency.model_validate(dependency_payload)
                )
            except ValueError:
                return None
    recommended_parameters = None
    if isinstance(model_payload.get("recommended_parameters"), dict):
        recommended_parameters = RecommendedParameters.model_validate(
            model_payload["recommended_parameters"]
        )
    return ModelMetadata(
        model_id=canonical_model_id,
        model_architecture=model_architecture,
        model_packages=packages,
        task_type=model_payload.get("task_type"),
        model_variant=model_payload.get("model_variant"),
        model_dependencies=model_dependencies,
        recommended_parameters=recommended_parameters,
    )


def load_record_raw(model_id: str) -> Optional[dict]:
    """Return the raw registry record for a canonical model id or an alias."""

    direct_record = _read_record_file(
        record_path=_record_path(canonical_model_id=model_id)
    )
    if direct_record is not None:
        return direct_record
    for record in iterate_records():
        if model_id in (record.get("requested_aliases") or []):
            return record
    return None


def load_model_metadata(model_id: str) -> Optional[ModelMetadata]:
    record = load_record_raw(model_id=model_id)
    if record is None:
        return None
    return _record_to_model_metadata(record=record)


def iterate_records() -> List[dict]:
    registry_dir = generate_offline_registry_dir()
    if not os.path.isdir(registry_dir):
        return []
    records = []
    try:
        entries = sorted(os.listdir(registry_dir))
    except OSError:
        return []
    for entry in entries:
        if not entry.endswith(".json") or entry.startswith("."):
            continue
        record = _read_record_file(record_path=os.path.join(registry_dir, entry))
        if record is not None:
            records.append(record)
    return records


def _package_presence(
    canonical_model_id: str,
    package_payload: dict,
) -> Tuple[str, List[str]]:
    """Return (status, missing_files) for a recorded package."""

    package_id = package_payload.get("package_id")
    cache_model_id = package_payload.get("cache_model_id") or canonical_model_id
    if not isinstance(package_id, str):
        return "malformed", []
    package_dir = _resolve_package_dir(
        cache_model_id=cache_model_id,
        package_id=package_id,
    )
    if package_dir is None:
        return "missing", []
    missing = []
    for artefact in package_payload.get("artifacts") or []:
        file_handle = artefact.get("file_handle") if isinstance(artefact, dict) else None
        if not isinstance(file_handle, str):
            return "malformed", []
        if not os.path.isfile(os.path.join(package_dir, file_handle)):
            missing.append(file_handle)
    return ("ok", []) if not missing else ("incomplete", missing)


def list_records_status() -> List[Dict[str, Any]]:
    """One status entry per record, with per-package presence."""

    statuses = []
    for record in iterate_records():
        canonical_model_id = record.get("canonical_model_id")
        if not isinstance(canonical_model_id, str):
            continue
        package_statuses = []
        for package_payload in record.get("packages") or []:
            if not isinstance(package_payload, dict):
                continue
            presence, missing = _package_presence(
                canonical_model_id=canonical_model_id,
                package_payload=package_payload,
            )
            package_statuses.append(
                {
                    "package_id": package_payload.get("package_id"),
                    "backend": package_payload.get("backend"),
                    "quantization": package_payload.get("quantization"),
                    "static_batch_size": package_payload.get("static_batch_size"),
                    "dynamic_batch_size_supported": package_payload.get(
                        "dynamic_batch_size_supported"
                    ),
                    "trusted_source": package_payload.get("trusted_source"),
                    "presence": presence,
                    "missing_files": missing,
                }
            )
        statuses.append(
            {
                "canonical_model_id": canonical_model_id,
                "requested_aliases": record.get("requested_aliases") or [],
                "source": record.get("source"),
                "recorded_at": record.get("recorded_at"),
                "proven": record.get("proven") or {},
                "packages": package_statuses,
            }
        )
    return statuses


def verify_records(
    model_id: Optional[str] = None,
    check_hashes: bool = False,
) -> List[Dict[str, Any]]:
    """Presence (and optionally MD5) verification for recorded artefacts."""

    if model_id is not None:
        record = load_record_raw(model_id=model_id)
        records = [] if record is None else [record]
    else:
        records = iterate_records()
    results = []
    for record in records:
        canonical_model_id = record.get("canonical_model_id")
        if not isinstance(canonical_model_id, str):
            continue
        for package_payload in record.get("packages") or []:
            if not isinstance(package_payload, dict):
                continue
            package_id = package_payload.get("package_id")
            cache_model_id = (
                package_payload.get("cache_model_id") or canonical_model_id
            )
            package_dir = (
                _resolve_package_dir(
                    cache_model_id=cache_model_id,
                    package_id=package_id,
                )
                if isinstance(package_id, str)
                else None
            )
            for artefact in package_payload.get("artifacts") or []:
                file_handle = (
                    artefact.get("file_handle") if isinstance(artefact, dict) else None
                )
                if not isinstance(file_handle, str):
                    continue
                artefact_path = (
                    os.path.join(package_dir, file_handle) if package_dir else None
                )
                if artefact_path is None or not os.path.isfile(artefact_path):
                    status = "missing"
                elif check_hashes:
                    expected_md5 = artefact.get("md5_hash")
                    try:
                        actual_md5 = _md5_of_file(path=artefact_path)
                        status = (
                            "ok" if actual_md5 == expected_md5 else "hash-mismatch"
                        )
                    except OSError:
                        status = "unreadable"
                else:
                    status = "ok"
                results.append(
                    {
                        "canonical_model_id": canonical_model_id,
                        "package_id": package_id,
                        "file_handle": file_handle,
                        "status": status,
                    }
                )
    return results


def purge_record(
    model_id: str,
    file_lock_acquire_timeout: int = FILE_LOCK_ACQUIRE_TIMEOUT,
) -> bool:
    """Remove the registry record for a model. Returns True when removed."""

    record = load_record_raw(model_id=model_id)
    canonical_model_id = (
        record.get("canonical_model_id") if isinstance(record, dict) else None
    )
    if not isinstance(canonical_model_id, str):
        canonical_model_id = model_id
    record_path = _record_path(canonical_model_id=canonical_model_id)
    if not os.path.isfile(record_path):
        return False
    with FileLock(
        _record_lock_path(record_path=record_path),
        timeout=file_lock_acquire_timeout,
    ):
        if os.path.isfile(record_path):
            os.unlink(record_path)
            return True
    return False
