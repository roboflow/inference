import hashlib
import importlib
import importlib.util
import inspect
import json
import os.path
import re
import stat
import tempfile
import urllib.parse
from contextvars import ContextVar
from datetime import datetime
from functools import partial
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union

import torch
from filelock import FileLock
from rich.console import Console
from rich.text import Text

from inference_models.configuration import (
    DEFAULT_DEVICE,
    FILE_LOCK_ACQUIRE_TIMEOUT,
    INFERENCE_HOME,
    OFFLINE_MODE,
    ROBOFLOW_API_KEY,
)
from inference_models.errors import (
    CorruptedModelPackageError,
    DirectLocalStorageAccessError,
    ForbiddenLocalCodePackageAccessError,
    ForbiddenModelAccessError,
    InvalidModelInitParameterError,
    InvalidParameterError,
    MissingModelInitParameterError,
    ModelPackageAlternativesExhaustedError,
    ModelRetrievalError,
    NoModelPackagesAvailableError,
    RetryError,
    UnauthorizedModelAccessError,
    UntrustedFileError,
)
from inference_models.logger import LOGGER, verbose_info
from inference_models.models.auto_loaders.access_manager import (
    AccessIdentifiers,
    LiberalModelAccessManager,
    ModelAccessManager,
)
from inference_models.models.auto_loaders.auto_negotiation import (
    determine_default_allowed_quantization,
    filter_model_packages_based_on_model_features,
    filter_model_packages_by_requested_backend,
    filter_model_packages_by_requested_batch_size,
    filter_model_packages_by_requested_quantization,
    negotiate_model_packages,
    parse_backend_type,
)
from inference_models.models.auto_loaders.auto_resolution_cache import (
    AutoResolutionCache,
    AutoResolutionCacheEntry,
    BaseAutoLoadMetadataCache,
)
from inference_models.models.auto_loaders.constants import (
    MODEL_DEPENDENCIES_KEY,
    MODEL_DEPENDENCIES_SUB_DIR,
)
from inference_models.models.auto_loaders.dependency_models import (
    prepare_dependency_model_parameters,
)
from inference_models.models.auto_loaders.entities import (
    MODEL_CONFIG_FILE_NAME,
    AnyModel,
    BackendType,
    InferenceModelConfig,
    ModelArchitecture,
    TaskType,
)
from inference_models.models.auto_loaders.model_cache_paths import (
    generate_model_cache_root_candidates_for_model_id,
    generate_model_package_cache_path,
    generate_models_cache_dir,
    generate_shared_blobs_path,
    resolve_existing_model_package_cache_path,
)
from inference_models.models.auto_loaders.models_registry import (
    INSTANCE_SEGMENTATION_TASK,
    OBJECT_DETECTION_TASK,
    model_implementation_exists,
    resolve_model_class,
)
from inference_models.models.auto_loaders.presentation_utils import (
    calculate_artefacts_size,
    calculate_size_of_all_model_packages_artefacts,
    render_model_package_details_table,
    render_runtime_x_ray,
    render_table_with_model_overview,
    render_table_with_model_packages,
)
from inference_models.models.auto_loaders.ranking import rank_model_packages
from inference_models.runtime_introspection.core import x_ray_runtime_environment
from inference_models.utils.download import (
    FileHandle,
    download_files_to_directory,
    is_valid_md5_hash,
)
from inference_models.utils.file_system import dump_json, read_json
from inference_models.utils.hashing import hash_dict_content
from inference_models.weights_providers.core import (
    get_model_from_provider,
    model_provider_requires_network,
)
from inference_models.weights_providers.entities import (
    FileDownloadSpecs,
    LocalFileArtefactSpecs,
    ModelDependency,
    ModelPackageMetadata,
    PackageSourceType,
    Quantization,
    RecommendedParameters,
)
from inference_models.weights_providers.roboflow import LOCAL_API_KEY

MODEL_TYPES_TO_LOAD_FROM_CHECKPOINT = {
    "rfdetr-base",
    "rfdetr-small",
    "rfdetr-medium",
    "rfdetr-nano",
    "rfdetr-large",
    "rfdetr-xlarge",
    "rfdetr-2xlarge",
    "rfdetr-seg-preview",
    "rfdetr-seg-nano",
    "rfdetr-seg-small",
    "rfdetr-seg-medium",
    "rfdetr-seg-large",
    "rfdetr-seg-xlarge",
    "rfdetr-seg-2xlarge",
    "rfdetr-seg-xxlarge",
}
OFFLINE_CACHE_MANIFEST_VERSION = 4
CACHE_ATTRIBUTION_VERSION = 2

SIGNED_DOWNLOAD_URL_MARKER_PARAMS = {
    "x-goog-signature",
    "x-amz-signature",
    "awsaccesskeyid",
    "googleaccessid",
    "sig",
    "signature",
}
SIGNED_DOWNLOAD_URL_AUTH_PARAMS = SIGNED_DOWNLOAD_URL_MARKER_PARAMS | {
    "x-goog-algorithm",
    "x-goog-credential",
    "x-goog-date",
    "x-goog-expires",
    "x-goog-signedheaders",
    "x-amz-algorithm",
    "x-amz-credential",
    "x-amz-date",
    "x-amz-expires",
    "x-amz-signedheaders",
    "x-amz-security-token",
    "expires",
    "policy",
    "key-pair-id",
    "sv",
    "ss",
    "srt",
    "sp",
    "se",
    "st",
    "spr",
    "sr",
    "sip",
    "ses",
    "sdd",
    "skoid",
    "sktid",
    "skt",
    "ske",
    "sks",
    "skv",
    "saoid",
    "suoid",
    "scid",
}
_HELD_PACKAGE_MATERIALIZATION_LOCKS: ContextVar[frozenset[str]] = ContextVar(
    "held_package_materialization_locks",
    default=frozenset(),
)

DEFAULT_KWARGS_PARAMS_TO_BE_FORWARDED_TO_DEPENDENT_MODELS = [
    "owlv2_enforce_model_compilation",
    "owlv2_class_embeddings_cache",
    "owlv2_images_embeddings_cache",
]


def _resolve_effective_api_key(
    api_key: Optional[str],
    provider_requires_network: bool,
) -> Optional[str]:
    if provider_requires_network and (api_key is None or api_key == LOCAL_API_KEY):
        return ROBOFLOW_API_KEY
    return api_key


def _credential_hash(api_key: Optional[str]) -> str:
    return hash_dict_content(content={"api_key": api_key})


def _record_model_package_path(model: AnyModel, package_dir: str) -> None:
    try:
        setattr(
            model,
            "_inference_models_package_path",
            os.path.realpath(package_dir),
        )
    except Exception:
        LOGGER.debug(
            "Could not attach package path attribution to model instance %s.",
            type(model),
        )


def _retrieve_access_manager_model_package_path(
    model_access_manager: ModelAccessManager,
    model: AnyModel,
    model_id: str,
    package_id: Optional[str],
    api_key: Optional[str],
    loading_parameter_digest: str,
) -> Optional[str]:
    path_retriever = getattr(
        model_access_manager,
        "retrieve_model_storage_path",
        None,
    )
    if not callable(path_retriever):
        return None
    package_dir = path_retriever(
        model=model,
        model_id=model_id,
        package_id=package_id,
        api_key=api_key,
        loading_parameter_digest=loading_parameter_digest,
    )
    if (
        not isinstance(package_dir, str)
        or "\0" in package_dir
        or os.path.islink(package_dir)
        or not os.path.isdir(package_dir)
    ):
        return None
    return os.path.realpath(package_dir)


def _canonicalize_download_source(download_url: str) -> str:
    parsed_url = urllib.parse.urlparse(download_url)
    if not parsed_url.query:
        return download_url
    query_params = urllib.parse.parse_qsl(
        parsed_url.query,
        keep_blank_values=True,
    )
    query_param_names = {key.lower() for key, _ in query_params}
    if query_param_names.isdisjoint(SIGNED_DOWNLOAD_URL_MARKER_PARAMS):
        return download_url
    content_params = sorted(
        (key, value)
        for key, value in query_params
        if key.lower() not in SIGNED_DOWNLOAD_URL_AUTH_PARAMS
    )
    canonical_query = urllib.parse.urlencode(content_params)
    return urllib.parse.urlunparse(
        parsed_url._replace(query=canonical_query, fragment="")
    )


def _validate_remote_model_id(model_id: object) -> str:
    if not isinstance(model_id, str) or not model_id.strip() or "\0" in model_id:
        raise InvalidParameterError(
            message="Remote model identity must be a non-empty string.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#invalidparametererror",
        )
    return model_id


def _canonicalize_unordered_request_values(
    value: object,
    case_insensitive: bool = False,
) -> object:
    """Stabilize values whose package-selection semantics treat them as sets."""

    if value is None:
        return None
    values = value if isinstance(value, list) else [value]
    serialized_values = set()
    for item in values:
        serialized_item = getattr(item, "value", item)
        if case_insensitive and isinstance(serialized_item, str):
            serialized_item = serialized_item.lower()
        serialized_values.add(serialized_item)
    return sorted(serialized_values)


def _canonicalize_cache_hash_value(value: object) -> object:
    """Best-effort stable JSON representation for nested dependency options."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, dict):
        return {
            str(key): _canonicalize_cache_hash_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_cache_hash_value(item) for item in value]
    if isinstance(value, set):
        canonical_items = [_canonicalize_cache_hash_value(item) for item in value]
        return sorted(
            canonical_items,
            key=lambda item: json.dumps(item, sort_keys=True, default=str),
        )
    if hasattr(value, "model_dump"):
        return _canonicalize_cache_hash_value(value.model_dump(mode="json"))
    serialized_value = getattr(value, "value", None)
    if isinstance(serialized_value, (bool, int, float, str)):
        return serialized_value
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def _runtime_compatibility_content(runtime_x_ray: object) -> dict:
    """Return stable machine-compatibility data independent of display text."""

    def stringify(value: object) -> Optional[str]:
        return None if value is None else str(value)

    available_providers = getattr(
        runtime_x_ray, "available_onnx_execution_providers", None
    )
    return {
        "version": 1,
        "gpu_available": getattr(runtime_x_ray, "gpu_available", False),
        "gpu_devices": list(getattr(runtime_x_ray, "gpu_devices", [])),
        "gpu_devices_cc": [
            str(value) for value in getattr(runtime_x_ray, "gpu_devices_cc", [])
        ],
        "driver_version": stringify(getattr(runtime_x_ray, "driver_version", None)),
        "cuda_version": stringify(getattr(runtime_x_ray, "cuda_version", None)),
        "trt_version": stringify(getattr(runtime_x_ray, "trt_version", None)),
        "jetson_type": getattr(runtime_x_ray, "jetson_type", None),
        "l4t_version": stringify(getattr(runtime_x_ray, "l4t_version", None)),
        "os_version": getattr(runtime_x_ray, "os_version", None),
        "torch_available": getattr(runtime_x_ray, "torch_available", False),
        "torch_version": stringify(getattr(runtime_x_ray, "torch_version", None)),
        "torchvision_version": stringify(
            getattr(runtime_x_ray, "torchvision_version", None)
        ),
        "onnxruntime_version": stringify(
            getattr(runtime_x_ray, "onnxruntime_version", None)
        ),
        "available_onnx_execution_providers": (
            sorted(available_providers) if available_providers is not None else None
        ),
        "hf_transformers_available": getattr(
            runtime_x_ray, "hf_transformers_available", False
        ),
        "trt_python_package_available": getattr(
            runtime_x_ray, "trt_python_package_available", False
        ),
    }


def _runtime_compatibility_hash(runtime_x_ray: object) -> str:
    return hash_dict_content(
        content=_runtime_compatibility_content(runtime_x_ray=runtime_x_ray)
    )


def _validate_portable_cache_name(value: object, kind: str) -> str:
    windows_reserved_names = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\0" in value
        or "/" in value
        or "\\" in value
        or value in {".", ".."}
        or os.path.basename(value) != value
        or len(value) > 255
        or re.fullmatch(r"[A-Za-z0-9._-]+", value) is None
        or value.rstrip(". ") != value
        or value.split(".", 1)[0].upper() in windows_reserved_names
    ):
        raise CorruptedModelPackageError(
            message=f"Model package contains an unsafe or empty {kind}: {value!r}.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    return value


def _validate_package_artifact_handle(value: object, kind: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\0" in value
        or "\\" in value
        or value.startswith("/")
        or value.endswith("/")
    ):
        raise CorruptedModelPackageError(
            message=f"Model package contains an unsafe or empty {kind}: {value!r}.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    path_segments = value.split("/")
    if any(not segment for segment in path_segments):
        raise CorruptedModelPackageError(
            message=f"Model package contains an unsafe or empty {kind}: {value!r}.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    for segment in path_segments:
        _validate_portable_cache_name(value=segment, kind=kind)
        if segment.startswith(".") and segment.casefold().endswith(".lock"):
            raise CorruptedModelPackageError(
                message=(
                    f"Model package artefact `{value}` collides with an "
                    "internal lock path."
                ),
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
    if path_segments[0].casefold() in {
        MODEL_CONFIG_FILE_NAME.casefold(),
        MODEL_DEPENDENCIES_SUB_DIR.casefold(),
        "dependencies",
    }:
        raise CorruptedModelPackageError(
            message=(
                f"Model package artefact `{value}` collides with the "
                "offline package's internal metadata or lock paths."
            ),
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    return value


def _register_artifact_handle(
    file_handle: str,
    path_prefixes: Dict[Tuple[str, ...], Tuple[str, ...]],
    terminal_paths: Set[Tuple[str, ...]],
) -> None:
    path_segments = tuple(file_handle.split("/"))
    casefold_segments = tuple(segment.casefold() for segment in path_segments)
    for prefix_length in range(1, len(path_segments) + 1):
        prefix = path_segments[:prefix_length]
        casefold_prefix = casefold_segments[:prefix_length]
        existing_prefix = path_prefixes.get(casefold_prefix)
        if existing_prefix is not None and existing_prefix != prefix:
            raise CorruptedModelPackageError(
                message=(
                    "Model package contains case-ambiguous artefact paths: "
                    f"`{'/'.join(existing_prefix)}` and `{'/'.join(prefix)}`."
                ),
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        path_prefixes[casefold_prefix] = prefix
        if prefix_length < len(path_segments) and casefold_prefix in terminal_paths:
            raise CorruptedModelPackageError(
                message=(
                    "Model package contains artefact file/directory prefix "
                    f"collision at `{'/'.join(prefix)}`."
                ),
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
    if casefold_segments in terminal_paths:
        raise CorruptedModelPackageError(
            message=f"Model package contains duplicate artefact path `{file_handle}`.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    if any(
        terminal_path[: len(casefold_segments)] == casefold_segments
        for terminal_path in terminal_paths
    ):
        raise CorruptedModelPackageError(
            message=(
                "Model package contains artefact file/directory prefix "
                f"collision at `{file_handle}`."
            ),
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    terminal_paths.add(casefold_segments)


def _package_artifact_declarations(
    package_artefacts: List[Union[FileDownloadSpecs, LocalFileArtefactSpecs]],
) -> List[dict]:
    """Validate provider artefacts and return their canonical declarations.

    This runs before any package directory or download is created.  Keeping the
    declaration independent of URLs is deliberate: hashed files are identified
    by MD5, while unhashed files receive a SHA-256 identity after download.
    """

    declarations = []
    path_prefixes: Dict[Tuple[str, ...], Tuple[str, ...]] = {}
    terminal_paths: Set[Tuple[str, ...]] = set()
    for artefact in package_artefacts:
        if not isinstance(artefact, (FileDownloadSpecs, LocalFileArtefactSpecs)):
            raise CorruptedModelPackageError(
                message="Model package contains unsupported artefact metadata.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        file_handle = _validate_package_artifact_handle(
            value=artefact.file_handle,
            kind="artefact file handle",
        )
        _register_artifact_handle(
            file_handle=file_handle,
            path_prefixes=path_prefixes,
            terminal_paths=terminal_paths,
        )
        md5_hash = artefact.md5_hash
        if md5_hash is not None and not is_valid_md5_hash(md5_hash):
            raise CorruptedModelPackageError(
                message=(
                    f"Model package artefact `{file_handle}` has an invalid MD5 "
                    f"identity `{md5_hash}`."
                ),
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        if isinstance(artefact, LocalFileArtefactSpecs) and md5_hash is None:
            raise CorruptedModelPackageError(
                message=(f"Local cache artefact `{file_handle}` has no MD5 identity."),
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        source_hash = None
        if isinstance(artefact, FileDownloadSpecs) and md5_hash is None:
            source_identity = _canonicalize_download_source(
                download_url=artefact.download_url
            )
            source_hash = hash_dict_content(
                content={"download_source": source_identity}
            )
        declarations.append(
            {
                "file_handle": file_handle,
                "md5_hash": md5_hash,
                "unhashed": md5_hash is None,
                "source_hash": source_hash,
                "storage": (
                    "package_file"
                    if isinstance(artefact, LocalFileArtefactSpecs) or md5_hash is None
                    else "shared_blob"
                ),
            }
        )
    return sorted(
        declarations,
        key=lambda item: (item["file_handle"].casefold(), item["file_handle"]),
    )


def _parse_package_artifact_identities(value: object) -> List[dict]:
    if not isinstance(value, list):
        raise CorruptedModelPackageError(
            message="Cached model config contains invalid package_artifacts metadata.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    identities = []
    path_prefixes: Dict[Tuple[str, ...], Tuple[str, ...]] = {}
    terminal_paths: Set[Tuple[str, ...]] = set()
    expected_keys = {
        "file_handle",
        "md5_hash",
        "unhashed",
        "sha256_hash",
        "source_hash",
        "storage",
    }
    for item in value:
        if not isinstance(item, dict) or set(item) != expected_keys:
            raise CorruptedModelPackageError(
                message="Cached model config contains malformed package artifact identity.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        file_handle = _validate_package_artifact_handle(
            value=item.get("file_handle"),
            kind="cached artefact file handle",
        )
        _register_artifact_handle(
            file_handle=file_handle,
            path_prefixes=path_prefixes,
            terminal_paths=terminal_paths,
        )
        md5_hash = item.get("md5_hash")
        unhashed = item.get("unhashed")
        sha256_hash = item.get("sha256_hash")
        source_hash = item.get("source_hash")
        storage = item.get("storage")
        if not isinstance(unhashed, bool):
            raise CorruptedModelPackageError(
                message="Cached package artifact has an invalid unhashed marker.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        if storage not in {"shared_blob", "package_file"}:
            raise CorruptedModelPackageError(
                message="Cached package artifact has invalid storage metadata.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        if unhashed:
            if (
                md5_hash is not None
                or not isinstance(sha256_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", sha256_hash) is None
                or not isinstance(source_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None
            ):
                raise CorruptedModelPackageError(
                    message=(
                        "Cached unhashed package artifact has no valid content "
                        "and download-source identity."
                    ),
                    help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
                )
            if storage != "package_file":
                raise CorruptedModelPackageError(
                    message="Cached unhashed artifact must be stored in-package.",
                    help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
                )
        elif (
            not is_valid_md5_hash(md5_hash)
            or sha256_hash is not None
            or source_hash is not None
        ):
            raise CorruptedModelPackageError(
                message="Cached hashed package artifact has invalid identity metadata.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        identities.append(
            {
                "file_handle": file_handle,
                "md5_hash": md5_hash,
                "unhashed": unhashed,
                "sha256_hash": sha256_hash,
                "source_hash": source_hash,
                "storage": storage,
            }
        )
    canonical_identities = sorted(
        identities,
        key=lambda item: (item["file_handle"].casefold(), item["file_handle"]),
    )
    if identities != canonical_identities:
        raise CorruptedModelPackageError(
            message="Cached package artifact identities are not canonically ordered.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    return canonical_identities


def _artifact_declarations_from_identities(identities: List[dict]) -> List[dict]:
    return [
        {
            "file_handle": item["file_handle"],
            "md5_hash": item["md5_hash"],
            "unhashed": item["unhashed"],
            "source_hash": item["source_hash"],
            "storage": item["storage"],
        }
        for item in identities
    ]


def _hash_regular_file(path: str, algorithm: str) -> str:
    try:
        path_stat = os.lstat(path)
    except OSError as error:
        raise CorruptedModelPackageError(
            message=f"Cannot inspect cached model artefact `{path}`.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        ) from error
    if not stat.S_ISREG(path_stat.st_mode):
        raise CorruptedModelPackageError(
            message=f"Cached model artefact `{path}` is not a regular file.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    file_descriptor = -1
    try:
        file_descriptor = os.open(path, flags)
        opened_stat = os.fstat(file_descriptor)
        if not stat.S_ISREG(opened_stat.st_mode) or (
            opened_stat.st_dev,
            opened_stat.st_ino,
        ) != (path_stat.st_dev, path_stat.st_ino):
            raise CorruptedModelPackageError(
                message=f"Cached model artefact `{path}` changed while being opened.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        digest = hashlib.new(algorithm)
        with os.fdopen(file_descriptor, "rb") as file_handle:
            file_descriptor = -1
            for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                digest.update(chunk)
            completed_stat = os.fstat(file_handle.fileno())
        if (
            opened_stat.st_size,
            opened_stat.st_mtime_ns,
            opened_stat.st_ctime_ns,
        ) != (
            completed_stat.st_size,
            completed_stat.st_mtime_ns,
            completed_stat.st_ctime_ns,
        ):
            raise CorruptedModelPackageError(
                message=f"Cached model artefact `{path}` changed while being hashed.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        return digest.hexdigest()
    except OSError as error:
        raise CorruptedModelPackageError(
            message=f"Cannot safely read cached model artefact `{path}`.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        ) from error
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)


def _sha256_file(path: str) -> str:
    return _hash_regular_file(path=path, algorithm="sha256")


def _md5_file(path: str) -> str:
    return _hash_regular_file(path=path, algorithm="md5")


def _materialize_package_artifact_identities(
    package_dir: str,
    declarations: List[dict],
) -> List[dict]:
    identities = []
    shared_blobs_dir = os.path.abspath(generate_shared_blobs_path())
    if any(
        declaration["storage"] == "shared_blob" for declaration in declarations
    ) and os.path.islink(shared_blobs_dir):
        raise CorruptedModelPackageError(
            message="Shared model blob directory cannot be a symbolic link.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    for declaration in declarations:
        package_path = os.path.join(package_dir, declaration["file_handle"])
        md5_hash = declaration["md5_hash"]
        if declaration["storage"] == "shared_blob":
            expected_blob = os.path.join(shared_blobs_dir, md5_hash)
            if (
                not os.path.islink(package_path)
                or os.path.islink(expected_blob)
                or not os.path.isfile(expected_blob)
                or os.path.realpath(package_path) != os.path.realpath(expected_blob)
                or _md5_file(path=expected_blob) != md5_hash
            ):
                raise CorruptedModelPackageError(
                    message=(
                        f"Cached package artefact `{declaration['file_handle']}` "
                        "does not point at its expected shared blob."
                    ),
                    help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
                )
            sha256_hash = None
        else:
            if os.path.islink(package_path) or not os.path.isfile(package_path):
                raise CorruptedModelPackageError(
                    message=(
                        f"Cached in-package artefact "
                        f"`{declaration['file_handle']}` is not a regular in-package file."
                    ),
                    help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
                )
            if md5_hash is not None:
                if _md5_file(path=package_path) != md5_hash:
                    raise CorruptedModelPackageError(
                        message=(
                            f"Cached package artefact "
                            f"`{declaration['file_handle']}` does not match its MD5 identity."
                        ),
                        help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
                    )
                sha256_hash = None
            else:
                sha256_hash = _sha256_file(path=package_path)
        identities.append(
            {
                **declaration,
                "sha256_hash": sha256_hash,
            }
        )
    return identities


def _validate_package_directory_layout(
    package_dir: str,
    artifact_declarations: List[dict],
    dependency_package_paths: List[dict],
) -> None:
    """Reject package entries that are not derived from this exact package.

    A failed initialization deliberately leaves no manifest. Without this
    inventory check, a later provider revision reusing the same package ID
    could silently consume a stale optional file that it no longer declares.
    """

    if not os.path.lexists(package_dir):
        return
    if os.path.islink(package_dir) or not os.path.isdir(package_dir):
        raise CorruptedModelPackageError(
            message="Cached model package path is not a regular directory.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )

    _remove_generated_bytecode_caches(package_dir=package_dir)

    expected_directories: Set[str] = set()
    expected_regular_files = {
        MODEL_CONFIG_FILE_NAME,
        f".{MODEL_CONFIG_FILE_NAME}.lock",
    }
    expected_symlinks: Set[str] = set()
    for declaration in artifact_declarations:
        file_handle = declaration["file_handle"]
        segments = file_handle.split("/")
        for prefix_length in range(1, len(segments)):
            expected_directories.add("/".join(segments[:prefix_length]))
        parent = "/".join(segments[:-1])
        lock_name = f".{segments[-1]}.lock"
        expected_regular_files.add(f"{parent}/{lock_name}" if parent else lock_name)
        if declaration["storage"] == "shared_blob":
            expected_symlinks.add(file_handle)
        else:
            expected_regular_files.add(file_handle)

    if dependency_package_paths:
        expected_directories.add(MODEL_DEPENDENCIES_SUB_DIR)
    for dependency_identity in dependency_package_paths:
        dependency_name = dependency_identity["name"]
        expected_symlinks.add(f"{MODEL_DEPENDENCIES_SUB_DIR}/{dependency_name}")
        expected_regular_files.add(
            f"{MODEL_DEPENDENCIES_SUB_DIR}/.{dependency_name}.lock"
        )

    def inspect_directory(directory: str, relative_parent: str = "") -> None:
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise CorruptedModelPackageError(
                message=f"Cannot inspect cached model package directory `{directory}`.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            ) from error
        for entry in entries:
            relative_path = (
                f"{relative_parent}/{entry.name}" if relative_parent else entry.name
            )
            try:
                if entry.is_symlink():
                    valid_entry = relative_path in expected_symlinks
                elif entry.is_dir(follow_symlinks=False):
                    valid_entry = relative_path in expected_directories
                    if valid_entry:
                        inspect_directory(
                            directory=entry.path,
                            relative_parent=relative_path,
                        )
                elif entry.is_file(follow_symlinks=False):
                    valid_entry = relative_path in expected_regular_files
                else:
                    valid_entry = False
            except OSError as error:
                raise CorruptedModelPackageError(
                    message=(
                        f"Cannot inspect cached model package entry "
                        f"`{relative_path}`."
                    ),
                    help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
                ) from error
            if not valid_entry:
                raise CorruptedModelPackageError(
                    message=(
                        "Cached model package contains undeclared or unsafe "
                        f"entry `{relative_path}`."
                    ),
                    help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
                )

    inspect_directory(directory=package_dir)


def _remove_generated_bytecode_caches(package_dir: str) -> None:
    """Remove bytecode caches created by package-local imports in older releases.

    Only real ``__pycache__`` directories containing regular ``.pyc`` files are
    removed. Any symlink, nested directory, or unexpected file is preserved so
    the subsequent package-layout validation rejects it.
    """

    for directory, directory_names, _ in os.walk(
        package_dir, topdown=False, followlinks=False
    ):
        for directory_name in directory_names:
            if directory_name != "__pycache__":
                continue
            bytecode_cache_dir = os.path.join(directory, directory_name)
            if os.path.islink(bytecode_cache_dir):
                continue
            try:
                entries = list(os.scandir(bytecode_cache_dir))
                contains_only_bytecode = all(
                    entry.is_file(follow_symlinks=False) and entry.name.endswith(".pyc")
                    for entry in entries
                )
                if not contains_only_bytecode:
                    continue
                for entry in entries:
                    os.unlink(entry.path)
                os.rmdir(bytecode_cache_dir)
            except OSError:
                # Leave entries that cannot be inspected or removed for the
                # strict package-layout validation below to reject.
                continue


def _package_has_current_offline_manifest(package_dir: str) -> bool:
    config_path = os.path.join(package_dir, MODEL_CONFIG_FILE_NAME)
    if not os.path.isfile(config_path) or os.path.islink(config_path):
        return False
    try:
        content = read_json(path=config_path)
    except (OSError, ValueError):
        return False
    return (
        isinstance(content, dict)
        and content.get("offline_manifest_version") == OFFLINE_CACHE_MANIFEST_VERSION
    )


def _remove_unattributed_unhashed_artifacts(
    package_dir: str,
    declarations: List[dict],
) -> None:
    """Force a fresh fetch when no manifest binds an unhashed cached file."""

    if _package_has_current_offline_manifest(package_dir=package_dir):
        return
    for declaration in declarations:
        if not declaration["unhashed"]:
            continue
        package_path = os.path.join(package_dir, declaration["file_handle"])
        if not os.path.lexists(package_path):
            continue
        if os.path.islink(package_path) or not os.path.isfile(package_path):
            raise CorruptedModelPackageError(
                message=(
                    f"Unattributed unhashed artefact "
                    f"`{declaration['file_handle']}` is not a regular file."
                ),
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        try:
            os.unlink(package_path)
        except OSError as error:
            raise CorruptedModelPackageError(
                message=(
                    f"Cannot replace unattributed unhashed artefact "
                    f"`{declaration['file_handle']}` with a fresh download."
                ),
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            ) from error


def _validate_cached_package_artifacts(
    package_dir: str,
    identities: Optional[List[dict]],
) -> bool:
    try:
        parsed_identities = _parse_package_artifact_identities(identities)
        current_identities = _materialize_package_artifact_identities(
            package_dir=package_dir,
            declarations=_artifact_declarations_from_identities(
                identities=parsed_identities
            ),
        )
    except (OSError, CorruptedModelPackageError) as error:
        LOGGER.warning(
            "Ignoring cached package %s because artefact identity validation "
            "failed: %s",
            package_dir,
            error,
        )
        return False
    return current_identities == parsed_identities


def _validate_cached_package_layout(
    package_dir: str,
    artifact_identities: Optional[List[dict]],
    dependency_identities: Optional[List[dict]],
) -> bool:
    try:
        parsed_artifacts = _parse_package_artifact_identities(artifact_identities)
        parsed_dependencies = _parse_dependency_package_path_identities(
            dependency_identities
        )
        _validate_package_directory_layout(
            package_dir=package_dir,
            artifact_declarations=_artifact_declarations_from_identities(
                identities=parsed_artifacts
            ),
            dependency_package_paths=parsed_dependencies,
        )
    except (OSError, CorruptedModelPackageError) as error:
        LOGGER.warning(
            "Ignoring cached package %s because its directory layout is not "
            "fully declared by the package manifest: %s",
            package_dir,
            error,
        )
        return False
    return True


def _dependency_package_path_identities(
    model_dependencies: Optional[List[ModelDependency]],
    model_dependencies_directories: Dict[str, str],
) -> List[dict]:
    dependencies = model_dependencies or []
    names_by_casefold: Dict[str, str] = {}
    dependency_names = set()
    for dependency in dependencies:
        dependency_name = _validate_portable_cache_name(
            value=dependency.name,
            kind="dependency name",
        )
        if dependency_name.startswith(".") and dependency_name.casefold().endswith(
            ".lock"
        ):
            raise CorruptedModelPackageError(
                message=(
                    f"Dependency name `{dependency_name}` collides with an "
                    "internal dependency lock path."
                ),
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        existing_name = names_by_casefold.get(dependency_name.casefold())
        if existing_name is not None:
            raise CorruptedModelPackageError(
                message=(
                    "Model package contains duplicate or case-ambiguous "
                    f"dependency names: `{existing_name}` and `{dependency_name}`."
                ),
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        names_by_casefold[dependency_name.casefold()] = dependency_name
        dependency_names.add(dependency_name)
    if set(model_dependencies_directories) != dependency_names:
        raise CorruptedModelPackageError(
            message="Model dependency directories do not match declared dependencies.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    identities = []
    for dependency_name in sorted(
        dependency_names, key=lambda value: (value.casefold(), value)
    ):
        dependency_directory = model_dependencies_directories[dependency_name]
        if not isinstance(dependency_directory, str) or "\0" in dependency_directory:
            raise CorruptedModelPackageError(
                message=f"Dependency `{dependency_name}` has an invalid package path.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        identities.append(
            _dependency_package_identity_for_path(
                dependency_name=dependency_name,
                dependency_directory=dependency_directory,
            )
        )
    return identities


def _dependency_package_identity_for_path(
    dependency_name: str,
    dependency_directory: str,
) -> dict:
    canonical_path = os.path.realpath(dependency_directory)
    if os.path.islink(dependency_directory) or not os.path.isdir(canonical_path):
        raise CorruptedModelPackageError(
            message=f"Dependency `{dependency_name}` package path is not a regular directory.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    dependency_config = parse_model_config(
        config_path=os.path.join(canonical_path, MODEL_CONFIG_FILE_NAME)
    )
    if (
        dependency_config.offline_manifest_version != OFFLINE_CACHE_MANIFEST_VERSION
        or not dependency_config.model_id
        or not dependency_config.canonical_model_id
        or not dependency_config.manifest_content_hash
        or re.fullmatch(r"[0-9a-f]{64}", dependency_config.manifest_content_hash)
        is None
        or not _validate_cached_package_artifacts(
            package_dir=canonical_path,
            identities=dependency_config.package_artifacts,
        )
        or dependency_config.model_dependencies not in (None, [])
        or dependency_config.dependency_package_paths != []
        or not _validate_cached_package_layout(
            package_dir=canonical_path,
            artifact_identities=dependency_config.package_artifacts,
            dependency_identities=dependency_config.dependency_package_paths,
        )
    ):
        raise CorruptedModelPackageError(
            message=(
                f"Dependency `{dependency_name}` has no current, verifiable "
                "package manifest."
            ),
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    return {
        "name": dependency_name,
        "target_path": canonical_path,
        "cache_model_id": dependency_config.model_id,
        "canonical_model_id": dependency_config.canonical_model_id,
        "model_package_id": os.path.basename(canonical_path),
        "package_manifest_hash": dependency_config.manifest_content_hash,
    }


def _parse_dependency_package_path_identities(value: object) -> List[dict]:
    if not isinstance(value, list):
        raise CorruptedModelPackageError(
            message="Cached model config contains invalid dependency_package_paths metadata.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    identities = []
    names_by_casefold: Dict[str, str] = {}
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "name",
            "target_path",
            "cache_model_id",
            "canonical_model_id",
            "model_package_id",
            "package_manifest_hash",
        }:
            raise CorruptedModelPackageError(
                message="Cached model config contains malformed dependency path identity.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        name = _validate_portable_cache_name(
            value=item.get("name"),
            kind="cached dependency name",
        )
        if name.startswith(".") and name.casefold().endswith(".lock"):
            raise CorruptedModelPackageError(
                message="Cached dependency name collides with an internal lock path.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        if name.casefold() in names_by_casefold:
            raise CorruptedModelPackageError(
                message="Cached dependency path identities are case-ambiguous.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        names_by_casefold[name.casefold()] = name
        target_path = item.get("target_path")
        if (
            not isinstance(target_path, str)
            or not os.path.isabs(target_path)
            or "\0" in target_path
            or os.path.normpath(target_path) != target_path
        ):
            raise CorruptedModelPackageError(
                message="Cached dependency path identity has an invalid target.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        for identity_field in (
            "cache_model_id",
            "canonical_model_id",
            "model_package_id",
        ):
            identity_value = item.get(identity_field)
            if not isinstance(identity_value, str) or not identity_value.strip():
                raise CorruptedModelPackageError(
                    message=(
                        f"Cached dependency path identity has invalid "
                        f"{identity_field} metadata."
                    ),
                    help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
                )
        package_manifest_hash = item.get("package_manifest_hash")
        if (
            not isinstance(package_manifest_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", package_manifest_hash) is None
        ):
            raise CorruptedModelPackageError(
                message="Cached dependency path identity has invalid manifest identity.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        identities.append(
            {
                "name": name,
                "target_path": target_path,
                "cache_model_id": item["cache_model_id"],
                "canonical_model_id": item["canonical_model_id"],
                "model_package_id": item["model_package_id"],
                "package_manifest_hash": package_manifest_hash,
            }
        )
    canonical_identities = sorted(
        identities, key=lambda item: (item["name"].casefold(), item["name"])
    )
    if identities != canonical_identities:
        raise CorruptedModelPackageError(
            message="Cached dependency path identities are not canonically ordered.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    return canonical_identities


def _validate_cached_dependency_package_paths(
    package_dir: str,
    identities: Optional[List[dict]],
) -> bool:
    try:
        parsed_identities = _parse_dependency_package_path_identities(identities)
        for identity in parsed_identities:
            dependency_path = os.path.join(
                package_dir,
                MODEL_DEPENDENCIES_SUB_DIR,
                identity["name"],
            )
            target_path = identity["target_path"]
            if (
                not os.path.islink(dependency_path)
                or os.path.islink(target_path)
                or not os.path.isdir(target_path)
                or os.path.realpath(dependency_path) != target_path
            ):
                raise CorruptedModelPackageError(
                    message=(
                        f"Dependency link `{identity['name']}` does not point "
                        "at its warmed package directory."
                    ),
                    help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
                )
            if (
                _dependency_package_identity_for_path(
                    dependency_name=identity["name"],
                    dependency_directory=target_path,
                )
                != identity
            ):
                raise CorruptedModelPackageError(
                    message=(
                        f"Dependency `{identity['name']}` package manifest "
                        "changed since the parent package was warmed."
                    ),
                    help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
                )
    except (OSError, CorruptedModelPackageError) as error:
        LOGGER.warning(
            "Ignoring cached package %s because dependency path validation "
            "failed: %s",
            package_dir,
            error,
        )
        return False
    return True


def _expected_dependency_package_paths(
    model_dependencies: List[ModelDependency],
    identities: Optional[List[dict]],
) -> Dict[str, dict]:
    parsed_identities = _parse_dependency_package_path_identities(identities)
    expected_paths = {identity["name"]: identity for identity in parsed_identities}
    dependency_names = {dependency.name for dependency in model_dependencies}
    if set(expected_paths) != dependency_names:
        raise CorruptedModelPackageError(
            message=(
                "Cached dependency path identities do not match the model's "
                "declared dependencies."
            ),
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    return expected_paths


class AutoModel:

    @classmethod
    def describe_model(
        cls,
        model_id: str,
        weights_provider: str = "roboflow",
        api_key: Optional[str] = None,
        pull_artefacts_size: bool = False,
        weights_provider_extra_query_params: Optional[List[Tuple[str, str]]] = None,
        weights_provider_extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Display comprehensive metadata and available packages for a model.

        Shows detailed information about a model without loading it, including:

        - Model architecture and variant
        - Task type (object detection, classification, etc.)
        - Available model packages (different backends, quantizations, batch sizes)
        - Package requirements and compatibility
        - Model dependencies (if any)
        - Package sizes (optional, requires network requests)

        This is useful for:

        - Exploring available models before loading
        - Understanding which backends are available for a model
        - Checking model requirements and compatibility
        - Debugging model loading issues
        - Selecting the right package for your environment

        Args:
            model_id: Model identifier. Can be:
                - Pre-trained model ID (e.g., "yolov8n-640", "rfdetr-base")
                - Custom Roboflow model (e.g., "my-project/2")

            weights_provider: Source for model metadata. Options:
                - "roboflow" (default): Query Roboflow platform
                - Custom provider name (if registered)

            api_key: Roboflow API key for accessing private models. If not provided,
                uses the `ROBOFLOW_API_KEY` environment variable. Not required for
                public pre-trained models.

            pull_artefacts_size: Whether to calculate and display the total size of
                each model package. This requires making network requests to check
                file sizes, so it's slower. Default: False.

            weights_provider_extra_query_params: Extra query parameters to pass to the weights' provider. Advanced
                usage only.

            weights_provider_extra_headers: Extra headers to pass to the weights' provider. Advanced
                usage only.

        Returns:
            None. Prints formatted tables to the console showing:
                1. Model overview table with architecture, task type, and dependencies
                2. Available packages table with backend, quantization, and batch size info

        Raises:
            UnauthorizedModelAccessError: If API key is invalid or model access is denied.
            ModelNotFoundError: If the model ID doesn't exist in the weights provider.

        Examples:
            View model information:

            >>> from inference_models import AutoModel
            >>> AutoModel.describe_model("yolov8n-640")

            View with package sizes:

            >>> AutoModel.describe_model("rfdetr-base", pull_artefacts_size=True)
            # Same as above, but includes a "Size" column showing package sizes

            View private model:

            >>> AutoModel.describe_model(
            ...     "my-workspace/my-model/2",
            ...     api_key="your_api_key"
            ... )

        See Also:
            - `AutoModel.describe_model_package()`: View detailed info for a specific package
            - `AutoModel.describe_compute_environment()`: Check your runtime environment
            - `AutoModel.from_pretrained()`: Load a model after inspecting it
        """
        model_metadata = get_model_from_provider(
            provider=weights_provider,
            model_id=model_id,
            api_key=api_key,
            weights_provider_extra_query_params=weights_provider_extra_query_params,
            weights_provider_extra_headers=weights_provider_extra_headers,
        )
        model_packages_size = None
        if pull_artefacts_size:
            model_packages_size = calculate_size_of_all_model_packages_artefacts(
                model_packages=model_metadata.model_packages
            )
        console = Console()
        model_overview_table = render_table_with_model_overview(
            model_id=model_metadata.model_id,
            requested_model_id=model_id,
            model_architecture=model_metadata.model_architecture,
            model_variant=model_metadata.model_variant,
            task_type=model_metadata.task_type,
            weights_provider=weights_provider,
            registered_packages=len(model_metadata.model_packages),
            model_dependencies=model_metadata.model_dependencies,
        )
        console.print(model_overview_table)
        console.print("\n")
        packages_overview_table = render_table_with_model_packages(
            model_packages=model_metadata.model_packages,
            model_packages_size=model_packages_size,
        )
        console.print(packages_overview_table)
        text = Text.assemble(
            ("\nWant to check more details about specific package?", "bold"),
            "\nUse AutoModel.describe_model_package('model_id', 'package_id').",
        )
        console.print(text)
        if not pull_artefacts_size:
            text = Text.assemble(
                ("\nWant to verify the size of model package?", "bold"),
                "\nUse AutoModel.describe_model('model_id', pull_artefacts_size=True) - the execution will be "
                "slightly longer, as we must collect the size of all elements of model package.",
            )
            console.print(text)

    @classmethod
    def describe_model_package(
        cls,
        model_id: str,
        package_id: str,
        weights_provider: str = "roboflow",
        api_key: Optional[str] = None,
        pull_artefacts_size: bool = True,
        weights_provider_extra_query_params: Optional[List[Tuple[str, str]]] = None,
        weights_provider_extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Display detailed information about a specific model package.

        Shows comprehensive details for a single model package, including:

        - Backend type (PyTorch, ONNX, TensorRT, etc.)
        - Quantization level (FP32, FP16, INT8, etc.)
        - Batch size configuration (fixed or dynamic)
        - Required dependencies and environment
        - Package artifacts (model files, configs, etc.)
        - Total package size (optional)
        - Hardware requirements (CUDA version, TensorRT version, etc.)

        This is useful for:

        - Understanding package requirements before loading
        - Debugging compatibility issues
        - Checking package size before download
        - Verifying package contents

        Args:
            model_id: Model identifier. Can be:
                - Pre-trained model ID (e.g., "yolov8n-640", "rfdetr-base")
                - Custom Roboflow model (e.g., "my-project/2")

            package_id: Specific package identifier to inspect. Get this from
                `AutoModel.describe_model()` output.

            weights_provider: Source for model metadata. Options:
                - "roboflow" (default): Query Roboflow platform
                - Custom provider name (if registered)

            api_key: Roboflow API key for accessing private models. If not provided,
                uses the `ROBOFLOW_API_KEY` environment variable. Not required for
                public pre-trained models.

            pull_artefacts_size: Whether to calculate and display the size of each
                artifact in the package. This requires making network requests to check
                file sizes, so it's slower. Default: True.

            weights_provider_extra_query_params: Extra query parameters to pass to the weights' provider. Advanced
                usage only.

            weights_provider_extra_headers: Extra headers to pass to the weights' provider. Advanced
                usage only.

        Returns:
            None. Prints a formatted table to the console showing package details.

        Raises:
            UnauthorizedModelAccessError: If API key is invalid or model access is denied.
            ModelNotFoundError: If the model ID doesn't exist in the weights provider.
            NoModelPackagesAvailableError: If the specified package_id doesn't exist
                for this model.

        Examples:
            View package details:

            >>> from inference_models import AutoModel
            >>> # First, see available packages
            >>> AutoModel.describe_model("yolov8n-640")
            >>> # Then inspect a specific package
            >>> AutoModel.describe_model_package("yolov8n-640", "pkg-trt-fp16-1-32")

            View without artifact sizes (faster):

            >>> AutoModel.describe_model_package(
            ...     "rfdetr-base",
            ...     "pkg-torch-fp32",
            ...     pull_artefacts_size=False
            ... )

        See Also:
            - `AutoModel.describe_model()`: View all available packages for a model
            - `AutoModel.describe_compute_environment()`: Check your runtime environment
            - `AutoModel.from_pretrained()`: Load a model with a specific package
        """
        model_metadata = get_model_from_provider(
            provider=weights_provider,
            model_id=model_id,
            api_key=api_key,
            weights_provider_extra_query_params=weights_provider_extra_query_params,
            weights_provider_extra_headers=weights_provider_extra_headers,
        )
        selected_package = None
        for package in model_metadata.model_packages:
            if package.package_id == package_id:
                selected_package = package
        if selected_package is None:
            raise NoModelPackagesAvailableError(
                message=f"Selected model package {package_id} does not exist for model {model_id}. Make sure provided "
                f"value is valid.",
                help_url="https://inference-models.roboflow.com/errors/package-negotiation/#nomodelpackagesavailableerror",
            )
        artefacts_size = None
        if pull_artefacts_size:
            artefacts_size = calculate_artefacts_size(
                package_artefacts=selected_package.package_artefacts
            )
        table = render_model_package_details_table(
            model_id=model_metadata.model_id,
            requested_model_id=model_id,
            artefacts_size=artefacts_size,
            model_package=selected_package,
        )
        console = Console()
        console.print(table)
        if not pull_artefacts_size:
            text = Text.assemble(
                ("\nWant to verify the size of model package?", "bold"),
                "\nUse AutoModel.describe_model_package('model_id', 'package_id', pull_artefacts_size=True)"
                "- the execution will be slightly longer, as we must collect the size of all elements of model package.",
            )
            console.print(text)

    @classmethod
    def describe_compute_environment(cls) -> None:
        """Inspect and display the current runtime environment and available backends.

        Performs a comprehensive scan of your system to detect:

        - **Hardware**: GPU availability, GPU models, compute capability
        - **CUDA**: Driver version, CUDA toolkit version
        - **TensorRT**: TensorRT version and availability
        - **PyTorch**: PyTorch and torchvision versions
        - **ONNX Runtime**: Version and available execution providers
        - **Other backends**: Hugging Face Transformers, Ultralytics
        - **Platform**: OS version, Jetson type (if applicable), L4T version

        This is useful for:

        - Debugging model loading issues
        - Verifying backend installations
        - Checking hardware compatibility
        - Understanding which model packages will work in your environment
        - Troubleshooting performance issues

        Returns:
            None. Prints a formatted table to the console showing all detected
            environment information.

        Examples:
            Check your environment:

            >>> from inference_models import AutoModel
            >>> AutoModel.describe_compute_environment()
            # Displays output like:
                                        Compute environment details
            Detected GPUs:                      N/A
            Detected GPUs CUDA CC:              N/A
            NVIDIA driver:                      N/A
            CUDA version:                       N/A
            TRT version:                        N/A
            TRT Python package available:       False
            OS version:                         macos-26.2-arm64-arm-64bit
            torch version:                      2.6.0
            torchvision version:                0.21.0
            ONNX runtime version:               1.21.0
            Detected ONNX execution providers:  CoreMLExecutionProvider, AzureExecutionProvider, CPUExecutionProvider

        See Also:
            - `AutoModel.describe_model()`: View model metadata and requirements
            - `AutoModel.from_pretrained()`: Load a model (uses this environment info)
        """
        runtime_x_ray = x_ray_runtime_environment()
        table = render_runtime_x_ray(runtime_x_ray=runtime_x_ray)
        console = Console()
        console.print(table)

    @classmethod
    def resolve_class(
        cls,
        model_id: str,
        api_key: Optional[str] = None,
        weights_provider: str = "roboflow",
        backend: Optional[Union[str, List[str]]] = None,
        device: Optional[str] = None,
    ) -> type:
        """Resolve model_id to model class WITHOUT loading or downloading.

        Calls the Roboflow API (cached 24h) to get model metadata, negotiates
        the best package, and returns the model class. No artifacts are downloaded
        and no model is instantiated.

        Useful for discovering model capabilities (e.g. ``supported_tasks``)
        before committing to a load.

        Args:
            model_id: Model identifier (e.g. ``"yolov8n-640"``, ``"workspace/model/1"``).
            api_key: Roboflow API key. Needed for custom models.
            weights_provider: Weights provider name. Default ``"roboflow"``.
            backend: Preferred backend(s). Default: auto-negotiate.
            device: Target device (influences TRT/ONNX selection).

        Returns:
            Model class (not instance). Has ``get_supported_tasks()`` classmethod.

        Raises:
            ModelNotFoundError: If model_id doesn't exist.
            ModelPackageAlternativesExhaustedError: If no compatible package found.
        """
        model_metadata = get_model_from_provider(
            provider=weights_provider,
            model_id=model_id,
            api_key=api_key,
        )
        matching = negotiate_model_packages(
            model_architecture=model_metadata.model_architecture,
            task_type=model_metadata.task_type,
            model_packages=model_metadata.model_packages,
            requested_backends=backend,
            device=device,
        )
        if not matching:
            raise RuntimeError(
                f"No compatible model package found for '{model_id}'"
            )
        best = matching[0]
        model_features = (
            set(best.model_features.keys()) if best.model_features else None
        )
        return resolve_model_class(
            model_architecture=model_metadata.model_architecture,
            task_type=model_metadata.task_type,
            backend=best.backend,
            model_features=model_features,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        weights_provider: str = "roboflow",
        api_key: Optional[str] = None,
        model_package_id: Optional[str] = None,
        backend: Optional[
            Union[str, BackendType, List[Union[str, BackendType]]]
        ] = None,
        batch_size: Optional[Union[int, Tuple[int, int]]] = None,
        quantization: Optional[
            Union[str, Quantization, List[Union[str, Quantization]]]
        ] = None,
        onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
        device: Union[torch.device, str] = DEFAULT_DEVICE,
        default_onnx_trt_options: bool = True,
        max_package_loading_attempts: Optional[int] = None,
        verbose: bool = False,
        model_download_file_lock_acquire_timeout: int = FILE_LOCK_ACQUIRE_TIMEOUT,
        allow_untrusted_packages: bool = False,
        trt_engine_host_code_allowed: bool = True,
        allow_local_code_packages: bool = True,
        verify_hash_while_download: bool = True,
        download_files_without_hash: bool = False,
        use_auto_resolution_cache: bool = True,
        auto_resolution_cache: Optional[AutoResolutionCache] = None,
        allow_direct_local_storage_loading: bool = True,
        model_access_manager: Optional[ModelAccessManager] = None,
        nms_fusion_preferences: Optional[Union[bool, dict]] = None,
        model_type: Optional[str] = None,
        task_type: Optional[str] = None,
        allow_loading_dependency_models: bool = True,
        dependency_models_params: Optional[dict] = None,
        point_model_directory: Optional[Callable[[str], None]] = None,
        forwarded_kwargs: Optional[List[str]] = None,
        weights_provider_extra_query_params: Optional[List[Tuple[str, str]]] = None,
        weights_provider_extra_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> AnyModel:
        """Load and initialize a computer vision model with automatic backend selection.

        This is the primary entry point for loading models in `inference-models`. It automatically:

        - Downloads model weights from the specified provider (default: Roboflow)
        - Selects the optimal backend (TensorRT > PyTorch Hugging Face> > ONNX > others)
        - Configures the model for your hardware (CPU/GPU)
        - Handles caching of atrefacts

        Args:
            model_id_or_path: Model identifier or local path. Can be:
                - Pre-trained model ID (e.g., "yolov8n-640", "rfdetr-base", "resnet50")
                - Custom Roboflow model (e.g., "my-project/2")
                - Local directory path containing model files
                - Local checkpoint file path (e.g., "/path/to/checkpoint.pth")

            weights_provider: Source for model weights. Options:
                - "roboflow" (default): Download from Roboflow platform
                - "local": Load from local filesystem
                - Custom provider name (if registered via `register_model_provider()`)

            api_key: Roboflow API key for accessing private models. If not provided,
                uses the `ROBOFLOW_API_KEY` environment variable. Not required for
                public pre-trained models.

            model_package_id: Specific model package to load (advanced). If not provided,
                automatically selects the best package based on your environment and
                requested backend/quantization. Use `AutoModel.describe_model()` to see
                available packages.

            backend: Preferred inference backend(s). Can be:
                - Single backend: "torch", "onnx", "trt" (TensorRT), "hugging-face"
                - List of allowed backends: ["trt", "torch"] (the normal
                  compatibility and package ranking rules choose among them)
                - BackendType enum value(s)
                - None (default): Automatic selection (TensorRT > PyTorch > ONNX > HF)

            batch_size: Preferred batch size for inference. Can be:
                - Single integer: Fixed batch size (e.g., 1, 8, 16)
                - Tuple: Range of batch sizes (e.g., (1, 8) for dynamic batching)
                - None (default): Use model's default batch size
                Note: Only affects models with multiple batch size variants.

            quantization: Model quantization level(s). Can be:
                - Single value: "fp32", "fp16", "bf16", "int8"
                - List of allowed values: ["fp16", "fp32"] (the normal package
                  ranking rules choose among them)
                - Quantization enum value(s)
                - None (default): Automatic selection based on device capabilities

            onnx_execution_providers: ONNX Runtime execution providers (ONNX backend only).
                Examples:
                - ["CUDAExecutionProvider", "CPUExecutionProvider"]
                - [("TensorrtExecutionProvider", {"trt_fp16_enable": True})]
                If not provided, automatically selects based on available hardware.

            device: PyTorch device for model execution. Can be:
                - String: "cpu", "cuda", "cuda:0", "cuda:1", "mps"
                - torch.device object
                Default: "cuda" if available, otherwise "cpu"

            default_onnx_trt_options: Whether to use default TensorRT optimization options
                for ONNX Runtime's TensorRT execution provider. Default: True.

            max_package_loading_attempts: Maximum number of model packages to try before
                failing. Useful when multiple packages are available. Default: Try all
                matching packages.

            verbose: Enable detailed logging during model loading. Useful for debugging
                package selection and download issues. Default: False.

            model_download_file_lock_acquire_timeout: Timeout in seconds for acquiring
                file locks during concurrent downloads. Default: FILE_LOCK_ACQUIRE_TIMEOUT (20).

            allow_untrusted_packages: Allow loading model packages with custom code that
                haven't been verified. **Security risk** - only enable for trusted sources.
                Default: False.

            trt_engine_host_code_allowed: Allow TensorRT engines to execute host code.
                Required for some TensorRT optimizations. Default: True.

            allow_local_code_packages: Allow loading models with custom Python code from
                local directories. Default: True.

            verify_hash_while_download: Verify file integrity using checksums during
                download. Recommended for production. Default: True.

            download_files_without_hash: Allow downloading files that don't have checksums.
                **Security risk** - only enable for trusted sources. Default: False.

            use_auto_resolution_cache: Enable caching of model resolution results to speed
                up subsequent loads. Default: True.

            auto_resolution_cache: Custom cache implementation. If None, uses default
                file-based cache. Advanced usage only.

            allow_direct_local_storage_loading: Allow loading models directly from local
                paths without going through the weights provider. Default: True.

            model_access_manager: Custom model access control manager. If None, uses
                permissive default. Advanced usage only.

            nms_fusion_preferences: Non-Maximum Suppression fusion preferences for ONNX
                models. Can be:
                - True: Enable NMS fusion with default settings
                - False: Disable NMS fusion
                - dict: Custom NMS fusion configuration
                - None (default): Use model's default settings

            model_type: Override model architecture type (advanced). Only needed when
                loading local models without metadata. Examples: "yolov8", "rfdetr".

            task_type: Override task type (advanced). Only needed when loading local
                models without metadata. Examples: "object-detection", "classification".

            allow_loading_dependency_models: Allow loading models that depend on other
                models (e.g., some VLMs depend on separate vision encoders). Default: True.

            dependency_models_params: Parameters to pass to dependency models. Dict mapping
                dependency names to parameter dicts. Advanced usage only.

            point_model_directory: Callback function called with the model directory path
                after loading. Advanced usage only.

            forwarded_kwargs: List of kwargs to forward to dependency models. Advanced
                usage only.

            weights_provider_extra_query_params: Extra query parameters to pass to the weights' provider. Advanced
                usage only.

            weights_provider_extra_headers: Extra headers to pass to the weights' provider. Advanced
                usage only.

            **kwargs: Additional model-specific parameters passed to the model's
                `from_pretrained()` method. Varies by model type.

        Returns:
            Loaded model instance. The specific type depends on the model's task:
                - ObjectDetectionModel: For object detection (YOLO, RF-DETR, etc.)
                - ClassificationModel: For single-label classification
                - MultiLabelClassificationModel: For multi-label classification
                - InstanceSegmentationModel: For instance segmentation
                - KeyPointsDetectionModel: For keypoint detection
                - DepthEstimationModel: For depth estimation
                - StructuredOCRModel: For OCR with structured output
                - TextImageEmbeddingModel: For vision-language embeddings (CLIP, etc.)
                - OpenVocabularyObjectDetectionModel: For open-vocabulary detection

        Raises:
            UnauthorizedModelAccessError: If API key is invalid or model access is denied.
            ModelPackageNotFoundError: If no compatible model package is found for your
                environment and requested parameters.
            CorruptedModelPackageError: If model files are corrupted or incomplete.
            InvalidParameterError: If provided parameters are invalid.
            DirectLocalStorageAccessError: If local path loading is disabled but a local
                path was provided.

        Examples:
            Basic usage with pre-trained model:

            >>> from inference_models import AutoModel
            >>> model = AutoModel.from_pretrained("yolov8n-640")
            >>> predictions = model(image)

            Load custom Roboflow model:

            >>> model = AutoModel.from_pretrained(
            ...     "my-project/2",
            ...     api_key="your_api_key"
            ... )

            Force specific backend and device:

            >>> model = AutoModel.from_pretrained(
            ...     "rfdetr-base",
            ...     backend="torch",
            ...     device="cuda:1"
            ... )

            Load with quantization:

            >>> model = AutoModel.from_pretrained(
            ...     "yolov8n-640",
            ...     quantization="fp16"
            ... )

            Load from local checkpoint:

            >>> model = AutoModel.from_pretrained(
            ...     "/path/to/checkpoint.pth",
            ...     model_type="rfdetr-base",
            ...     labels=["cat", "dog"]
            ... )

            Enable verbose logging:

            >>> model = AutoModel.from_pretrained(
            ...     "yolov8n-640",
            ...     verbose=True
            ... )

        See Also:
            - `AutoModel.describe_model()`: View model metadata before loading
            - `AutoModel.describe_model_package()`: View specific package details
            - `AutoModel.describe_compute_environment()`: Check available backends
            - `AutoModel.list_available_models()`: List all registered models
        """
        if isinstance(model_id_or_path, os.PathLike):
            model_id_or_path = os.fspath(model_id_or_path)
        if not isinstance(model_id_or_path, str):
            _validate_remote_model_id(model_id=model_id_or_path)
        model_path_exists = os.path.exists(model_id_or_path)
        if not model_path_exists:
            _validate_remote_model_id(model_id=model_id_or_path)
        provider_requires_network = False
        if not model_path_exists:
            provider_requires_network = model_provider_requires_network(
                provider=weights_provider
            )
        api_key = _resolve_effective_api_key(
            api_key=api_key,
            provider_requires_network=provider_requires_network,
        )
        if model_access_manager is None:
            model_access_manager = LiberalModelAccessManager()
        if model_access_manager.is_model_access_forbidden(
            model_id=model_id_or_path, api_key=api_key
        ):
            raise UnauthorizedModelAccessError(
                message=f"Unauthorized not access model with ID: {model_package_id}. Are you sure you use valid "
                f"API key? The default weights provider is Roboflow - see Roboflow authentication details: "
                f"https://docs.roboflow.com/api-reference/authentication "
                f"and export key to `ROBOFLOW_API_KEY` environment variable. If you use custom weights "
                f"provider - verify access constraints relevant for the provider.",
                help_url="https://inference-models.roboflow.com/errors/model-retrieval/#unauthorizedmodelaccesserror",
            )
        if auto_resolution_cache is None:

            def register_file_created_for_model_package(
                file_path: str, model_id: str, package_id: str
            ) -> None:
                access_identifiers = AccessIdentifiers(
                    model_id=model_id,
                    package_id=package_id,
                    api_key=api_key,
                )
                model_access_manager.on_file_created(
                    file_path=file_path,
                    access_identifiers=access_identifiers,
                )

            auto_resolution_cache = BaseAutoLoadMetadataCache(
                file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
                verbose=verbose,
                on_file_created=register_file_created_for_model_package,
                on_file_deleted=model_access_manager.on_file_deleted,
            )
        if isinstance(device, str):
            try:
                device = torch.device(device)
            except RuntimeError as error:
                raise InvalidParameterError(
                    message="Could not parse `device` parameter value - make sure that it is a valid string "
                    f"representation of torch device. Valid values: 'cpu', 'cuda' or 'cuda:0'. If you see this error "
                    "while using Roboflow infrastructure - contact us to get help. Otherwise - verify your setup.",
                    help_url="https://inference-models.roboflow.com/errors/input-validation/#invalidparametererror",
                ) from error
        model_init_kwargs = {
            "onnx_execution_providers": onnx_execution_providers,
            "device": device,
            "default_onnx_trt_options": default_onnx_trt_options,
            "engine_host_code_allowed": trt_engine_host_code_allowed,
        }
        model_init_kwargs.update(kwargs)
        if not model_path_exists:
            # QUESTION: is it enough to assume presence of local dir as the intent to load
            # model from disc drive? What if we have clash of model id / model alias with
            # contents of someone's local drive - shall we then try to load from both sources?
            # that still may end up with ambiguous behavior - probably the solution would be
            # to require prefix like file://... to denote the intent of loading model from local
            # drive?
            credential_bound_cache_request = bool(api_key)
            dependency_models_params = dependency_models_params or {}
            if forwarded_kwargs is None:
                forwarded_kwargs = (
                    DEFAULT_KWARGS_PARAMS_TO_BE_FORWARDED_TO_DEPENDENT_MODELS
                )
            forwarded_kwargs_values = {
                name: kwargs[name] for name in forwarded_kwargs if name in kwargs
            }
            runtime_x_ray = x_ray_runtime_environment()
            runtime_compatibility = _runtime_compatibility_content(
                runtime_x_ray=runtime_x_ray
            )
            # This digest describes whether an already-warmed package can be
            # loaded, so it deliberately excludes provider transport, retry,
            # and download-integrity options. Those still belong to the exact
            # online negotiation key below because they can affect acquisition.
            offline_compatibility_content = {
                "provider": weights_provider,
                "model_id": model_id_or_path,
                "requested_model_package_id": model_package_id,
                "requested_backends": _canonicalize_unordered_request_values(
                    backend,
                    case_insensitive=True,
                ),
                "requested_batch_size": batch_size,
                "requested_quantization": _canonicalize_unordered_request_values(
                    quantization
                ),
                "device": str(device),
                "onnx_execution_providers": onnx_execution_providers,
                "default_onnx_trt_options": default_onnx_trt_options,
                "allow_untrusted_packages": allow_untrusted_packages,
                "trt_engine_host_code_allowed": trt_engine_host_code_allowed,
                "allow_local_code_packages": allow_local_code_packages,
                "allow_loading_dependency_models": allow_loading_dependency_models,
                "nms_fusion_preferences": nms_fusion_preferences,
                "dependency_models_params": _canonicalize_cache_hash_value(
                    dependency_models_params
                ),
                "forwarded_dependency_kwargs": _canonicalize_cache_hash_value(
                    forwarded_kwargs_values
                ),
                "runtime_compatibility": runtime_compatibility,
            }
            offline_compatibility_hash = hash_dict_content(
                content=offline_compatibility_content
            )
            auto_negotiation_hash = hash_dict_content(
                content={
                    **offline_compatibility_content,
                    "verify_hash_while_download": verify_hash_while_download,
                    "download_files_without_hash": download_files_without_hash,
                    "max_package_loading_attempts": max_package_loading_attempts,
                    "weights_provider_extra_query_params": weights_provider_extra_query_params,
                    "weights_provider_extra_headers": weights_provider_extra_headers,
                    "api_key": api_key,
                }
            )
            model_from_access_manager = model_access_manager.retrieve_model_instance(
                model_id=model_id_or_path,
                package_id=model_package_id,
                api_key=api_key,
                loading_parameter_digest=auto_negotiation_hash,
            )
            if model_from_access_manager:
                if point_model_directory is None:
                    return model_from_access_manager
                model_package_dir = _retrieve_access_manager_model_package_path(
                    model_access_manager=model_access_manager,
                    model=model_from_access_manager,
                    model_id=model_id_or_path,
                    package_id=model_package_id,
                    api_key=api_key,
                    loading_parameter_digest=auto_negotiation_hash,
                )
                if model_package_dir is not None:
                    point_model_directory(model_package_dir)
                    return model_from_access_manager

            def attempt_cached_load(cache_hash: str) -> Optional[AnyModel]:
                return attempt_loading_model_with_auto_load_cache(
                    use_auto_resolution_cache=use_auto_resolution_cache,
                    auto_resolution_cache=auto_resolution_cache,
                    auto_negotiation_hash=cache_hash,
                    model_access_manager=model_access_manager,
                    model_name_or_path=model_id_or_path,
                    model_init_kwargs=dict(model_init_kwargs),
                    api_key=api_key,
                    allow_loading_dependency_models=allow_loading_dependency_models,
                    forwarded_kwargs_values=forwarded_kwargs_values,
                    verbose=verbose,
                    weights_provider=weights_provider,
                    max_package_loading_attempts=max_package_loading_attempts,
                    model_download_file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
                    allow_untrusted_packages=allow_untrusted_packages,
                    trt_engine_host_code_allowed=trt_engine_host_code_allowed,
                    allow_local_code_packages=allow_local_code_packages,
                    verify_hash_while_download=verify_hash_while_download,
                    download_files_without_hash=download_files_without_hash,
                    allow_direct_local_storage_loading=allow_direct_local_storage_loading,
                    dependency_models_params=dependency_models_params,
                    weights_provider_extra_query_params=weights_provider_extra_query_params,
                    weights_provider_extra_headers=weights_provider_extra_headers,
                    expected_offline_compatibility_hash=offline_compatibility_hash,
                )

            def verified_cached_model_directory(
                cache_hash: str,
            ) -> Optional[str]:
                cache_entry = auto_resolution_cache.retrieve(
                    auto_negotiation_hash=cache_hash
                )
                if cache_entry is None or cache_entry.model_id != model_id_or_path:
                    return None
                return _verified_auto_cache_package_dir(cache_entry=cache_entry)

            raw_cache_fallback_blocked = False

            def attempt_compatible_cached_load() -> Optional[AnyModel]:
                nonlocal raw_cache_fallback_blocked
                if not use_auto_resolution_cache or credential_bound_cache_request:
                    return None
                compatible_candidates = (
                    auto_resolution_cache.find_compatible_candidates(
                        offline_compatibility_hash=offline_compatibility_hash
                    )
                )
                if not compatible_candidates:
                    return None
                # Once current alias metadata exists, a raw package attributed
                # directly to the requested string is safe only if it proves
                # the same model identity. Otherwise an alias/canonical name
                # collision could turn an ambiguity into a direct-cache hit.
                raw_cache_fallback_blocked = True
                candidate_identities = set()
                for _, cache_entry in compatible_candidates:
                    if (
                        cache_entry.model_id != model_id_or_path
                        or cache_entry.cache_attribution_version
                        != CACHE_ATTRIBUTION_VERSION
                        or not cache_entry.canonical_model_id
                        or not cache_entry.canonical_model_id.strip()
                        or not cache_entry.cache_model_id
                        or not cache_entry.cache_model_id.strip()
                        or not isinstance(cache_entry.credential_hash, str)
                        or re.fullmatch(r"[0-9a-f]{64}", cache_entry.credential_hash)
                        is None
                    ):
                        LOGGER.warning(
                            "Ignoring credential-free cache fallback for %s because "
                            "at least one compatible entry has no current canonical "
                            "attribution.",
                            model_id_or_path,
                        )
                        return None
                    candidate_identities.add(
                        (
                            cache_entry.canonical_model_id,
                            cache_entry.task_type,
                            cache_entry.model_architecture,
                        )
                    )
                if len(candidate_identities) != 1:
                    LOGGER.warning(
                        "Ignoring credential-free cache fallback for %s because "
                        "compatible entries resolve to conflicting canonical "
                        "model metadata.",
                        model_id_or_path,
                    )
                    return None
                candidate_identity = next(iter(candidate_identities))
                direct_package = _find_direct_cached_model_package_dir(
                    model_id=model_id_or_path
                )
                if direct_package is not None:
                    direct_identity = _cached_package_model_identity(
                        package_dir=direct_package
                    )
                    if direct_identity != candidate_identity:
                        LOGGER.warning(
                            "Ignoring credential-free cache fallback for %s because "
                            "its direct package conflicts with current alias metadata.",
                            model_id_or_path,
                        )
                        return None
                    raw_cache_fallback_blocked = False
                for compatible_hash, cache_entry in compatible_candidates:
                    if compatible_hash == auto_negotiation_hash:
                        continue
                    model = attempt_cached_load(compatible_hash)
                    if model is None:
                        continue
                    if point_model_directory:
                        cache_dir = verified_cached_model_directory(
                            cache_hash=compatible_hash
                        )
                        if cache_dir is None:
                            continue
                        point_model_directory(cache_dir)
                    return model
                return None

            def attempt_raw_cached_load() -> Optional[Tuple[AnyModel, str]]:
                return attempt_loading_model_from_offline_cache(
                    model_id=model_id_or_path,
                    model_init_kwargs=dict(model_init_kwargs),
                    requested_model_package_id=model_package_id,
                    requested_backends=backend,
                    requested_batch_size=batch_size,
                    requested_quantization=quantization,
                    model_access_manager=model_access_manager,
                    api_key=api_key,
                    allow_local_code_packages=allow_local_code_packages,
                    allow_untrusted_packages=allow_untrusted_packages,
                    allow_loading_dependency_models=allow_loading_dependency_models,
                    dependency_models_params=dependency_models_params,
                    forwarded_kwargs_values=forwarded_kwargs_values,
                    weights_provider=weights_provider,
                    auto_resolution_cache=auto_resolution_cache,
                    use_auto_resolution_cache=use_auto_resolution_cache,
                    max_package_loading_attempts=max_package_loading_attempts,
                    model_download_file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
                    trt_engine_host_code_allowed=trt_engine_host_code_allowed,
                    verify_hash_while_download=verify_hash_while_download,
                    download_files_without_hash=download_files_without_hash,
                    allow_direct_local_storage_loading=allow_direct_local_storage_loading,
                    nms_fusion_preferences=nms_fusion_preferences,
                    weights_provider_extra_query_params=weights_provider_extra_query_params,
                    weights_provider_extra_headers=weights_provider_extra_headers,
                    verbose=verbose,
                    offline_compatibility_hash=offline_compatibility_hash,
                )

            model_from_cache = attempt_cached_load(auto_negotiation_hash)
            if model_from_cache:
                if point_model_directory:
                    cache_dir = verified_cached_model_directory(
                        cache_hash=auto_negotiation_hash
                    )
                    if cache_dir is None:
                        model_from_cache = None
                    else:
                        point_model_directory(cache_dir)
            if model_from_cache:
                return model_from_cache
            if OFFLINE_MODE and provider_requires_network:
                if not credential_bound_cache_request:
                    model_from_cache = attempt_compatible_cached_load()
                    if model_from_cache is not None:
                        return model_from_cache
                    if not raw_cache_fallback_blocked:
                        offline_result = attempt_raw_cached_load()
                        if offline_result is not None:
                            model, offline_cache_dir = offline_result
                            if point_model_directory:
                                point_model_directory(offline_cache_dir)
                            return model
                raise ModelRetrievalError(
                    message=f"Cannot load model {model_id_or_path} in OFFLINE_MODE - "
                    f"no compatible cached model package found in "
                    f"{INFERENCE_HOME}/models-cache/. "
                    f"Pre-populate the cache by running once with network access, "
                    f"or disable OFFLINE_MODE.",
                    help_url="https://inference-models.roboflow.com/errors/model-retrieval/#modelretrievalerror",
                )
            try:
                model_metadata = get_model_from_provider(
                    provider=weights_provider,
                    model_id=model_id_or_path,
                    api_key=api_key,
                    weights_provider_extra_query_params=weights_provider_extra_query_params,
                    weights_provider_extra_headers=weights_provider_extra_headers,
                )
                if (
                    not isinstance(model_metadata.model_id, str)
                    or not model_metadata.model_id.strip()
                ):
                    raise CorruptedModelPackageError(
                        message=(
                            f"Weights provider {weights_provider} returned an "
                            "empty or invalid canonical model ID."
                        ),
                        help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
                    )
                if (
                    model_metadata.model_dependencies
                    and not allow_loading_dependency_models
                ):
                    raise CorruptedModelPackageError(
                        message=f"Could not load model {model_id_or_path} as it defines another models which are "
                        f"it's dependency, but the auto-loader prevents loading dependencies at certain "
                        f"nesting depth to avoid excessive resolution procedure. This is a limitation of "
                        f"current implementation. Provide us the context of your use-case to get help.",
                        help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
                    )
                if model_metadata.model_id != model_id_or_path:
                    model_access_manager.on_model_alias_discovered(
                        alias=model_id_or_path,
                        model_id=model_metadata.model_id,
                    )
                model_dependencies = model_metadata.model_dependencies or []
                for model_dependency in model_dependencies:
                    model_access_manager.on_model_dependency_discovered(
                        base_model_id=model_dependency.model_id,
                        base_model_package_id=model_dependency.model_package_id,
                        dependent_model_id=model_metadata.model_id,
                    )
                for model_package in model_metadata.model_packages:
                    package_access_identifiers = AccessIdentifiers(
                        model_id=model_metadata.model_id,
                        package_id=model_package.package_id,
                        api_key=api_key,
                    )
                    model_access_manager.on_model_package_access_granted(
                        package_access_identifiers
                    )
            except (
                UnauthorizedModelAccessError,
                ForbiddenModelAccessError,
            ) as error:
                model_access_manager.on_model_access_forbidden(
                    model_id=model_id_or_path, api_key=api_key
                )
                raise error
            except RetryError:
                if not OFFLINE_MODE:
                    verbose_info(
                        message=(
                            f"API unreachable for model {model_id_or_path}; "
                            "cache fallback is disabled while running online."
                        ),
                        verbose_requested=verbose,
                    )
                    raise
                if credential_bound_cache_request:
                    verbose_info(
                        message=(
                            f"API unreachable for model {model_id_or_path}; "
                            "credential-independent cache fallback is disabled "
                            "for keyed requests."
                        ),
                        verbose_requested=verbose,
                    )
                    raise
                verbose_info(
                    message=f"API unreachable for model {model_id_or_path}, "
                    f"attempting offline cache fallback.",
                    verbose_requested=verbose,
                )
                model_from_cache = attempt_compatible_cached_load()
                if model_from_cache is not None:
                    return model_from_cache
                offline_result = (
                    None if raw_cache_fallback_blocked else attempt_raw_cached_load()
                )
                if offline_result is None:
                    raise
                model, offline_cache_dir = offline_result
                if point_model_directory:
                    point_model_directory(offline_cache_dir)
                return model
            # here we verify if de-aliasing or access confirmation from auth master changed something
            model_from_access_manager = model_access_manager.retrieve_model_instance(
                model_id=model_id_or_path,
                package_id=model_package_id,
                api_key=api_key,
                loading_parameter_digest=auto_negotiation_hash,
            )
            if model_from_access_manager:
                if point_model_directory is None:
                    return model_from_access_manager
                model_package_dir = _retrieve_access_manager_model_package_path(
                    model_access_manager=model_access_manager,
                    model=model_from_access_manager,
                    model_id=model_id_or_path,
                    package_id=model_package_id,
                    api_key=api_key,
                    loading_parameter_digest=auto_negotiation_hash,
                )
                if model_package_dir is not None:
                    point_model_directory(model_package_dir)
                    return model_from_access_manager
            matching_model_packages = negotiate_model_packages(
                model_architecture=model_metadata.model_architecture,
                task_type=model_metadata.task_type,
                model_packages=model_metadata.model_packages,
                requested_model_package_id=model_package_id,
                requested_backends=backend,
                requested_batch_size=batch_size,
                requested_quantization=quantization,
                device=device,
                onnx_execution_providers=onnx_execution_providers,
                allow_untrusted_packages=allow_untrusted_packages,
                trt_engine_host_code_allowed=trt_engine_host_code_allowed,
                nms_fusion_preferences=nms_fusion_preferences,
                verbose=verbose,
            )
            model_dependencies_instances = {}
            model_dependencies_directories = {}
            dependency_models_params = dependency_models_params or {}
            for model_dependency in model_dependencies:
                dependency_params = dict(
                    dependency_models_params.get(model_dependency.name, {})
                )
                dependency_params["model_id_or_path"] = model_dependency.model_id
                dependency_params["model_package_id"] = (
                    model_dependency.model_package_id
                )
                resolved_model_parameters = prepare_dependency_model_parameters(
                    model_parameters=dependency_params
                )
                verbose_info(
                    message=f"Initialising dependent model: {model_dependency.model_id}",
                    verbose_requested=verbose,
                )

                def model_directory_pointer(model_dir: str) -> None:
                    model_dependencies_directories[model_dependency.name] = model_dir

                for name, value in forwarded_kwargs_values.items():
                    if name not in resolved_model_parameters.model_extra:
                        resolved_model_parameters.model_extra[name] = value

                dependency_instance = AutoModel.from_pretrained(
                    model_id_or_path=resolved_model_parameters.model_id_or_path,
                    weights_provider=weights_provider,
                    api_key=api_key,
                    model_package_id=resolved_model_parameters.model_package_id,
                    backend=resolved_model_parameters.backend,
                    batch_size=resolved_model_parameters.batch_size,
                    quantization=resolved_model_parameters.quantization,
                    onnx_execution_providers=resolved_model_parameters.onnx_execution_providers,
                    device=resolved_model_parameters.device,
                    default_onnx_trt_options=resolved_model_parameters.default_onnx_trt_options,
                    max_package_loading_attempts=max_package_loading_attempts,
                    verbose=verbose,
                    model_download_file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
                    allow_untrusted_packages=allow_untrusted_packages,
                    trt_engine_host_code_allowed=trt_engine_host_code_allowed,
                    allow_local_code_packages=allow_local_code_packages,
                    verify_hash_while_download=verify_hash_while_download,
                    download_files_without_hash=download_files_without_hash,
                    use_auto_resolution_cache=use_auto_resolution_cache,
                    auto_resolution_cache=auto_resolution_cache,
                    allow_direct_local_storage_loading=allow_direct_local_storage_loading,
                    model_access_manager=model_access_manager,
                    nms_fusion_preferences=resolved_model_parameters.nms_fusion_preferences,
                    model_type=resolved_model_parameters.model_type,
                    task_type=resolved_model_parameters.task_type,
                    allow_loading_dependency_models=False,
                    dependency_models_params=None,
                    point_model_directory=model_directory_pointer,
                    weights_provider_extra_query_params=weights_provider_extra_query_params,
                    weights_provider_extra_headers=weights_provider_extra_headers,
                    **resolved_model_parameters.kwargs,
                )
                model_dependencies_instances[model_dependency.name] = (
                    dependency_instance
                )

            return attempt_loading_matching_model_packages(
                model_id=model_metadata.model_id,
                requested_model_id=model_id_or_path,
                model_architecture=model_metadata.model_architecture,
                task_type=model_metadata.task_type,
                matching_model_packages=matching_model_packages,
                model_init_kwargs=model_init_kwargs,
                model_access_manager=model_access_manager,
                auto_negotiation_hash=auto_negotiation_hash,
                offline_compatibility_hash=offline_compatibility_hash,
                api_key=api_key,
                model_dependencies=model_metadata.model_dependencies,
                model_dependencies_instances=model_dependencies_instances,
                model_dependencies_directories=model_dependencies_directories,
                recommended_parameters=model_metadata.recommended_parameters,
                max_package_loading_attempts=max_package_loading_attempts,
                model_download_file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
                verify_hash_while_download=verify_hash_while_download,
                download_files_without_hash=download_files_without_hash,
                auto_resolution_cache=auto_resolution_cache,
                use_auto_resolution_cache=use_auto_resolution_cache,
                point_model_directory=point_model_directory,
                verbose=verbose,
            )
        if not allow_direct_local_storage_loading:
            raise DirectLocalStorageAccessError(
                message="Attempted to load model directly pointing local path, rather than model ID. This "
                "operation is forbidden as AutoModel.from_pretrained(...) was used with "
                "`allow_direct_local_storage_loading=False`. If you are running `inference-models` outside Roboflow "
                "hosted solutions - verify your setup. If you see this error on Roboflow platform - this "
                "feature was disabled for security reason. In rare cases when you use valid model ID, the "
                "clash of ID with local path may cause this error - we ask you to report the issue here: "
                "https://github.com/roboflow/inference/issues.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#directlocalstorageaccesserror",
            )
        return attempt_loading_model_from_local_storage(
            model_dir_or_weights_path=model_id_or_path,
            allow_local_code_packages=allow_local_code_packages,
            model_init_kwargs=model_init_kwargs,
            model_type=model_type,
            task_type=task_type,
            backend_type=backend,
        )


def _verified_auto_cache_package_dir(
    cache_entry: AutoResolutionCacheEntry,
) -> Optional[str]:
    if (
        cache_entry.cache_attribution_version != CACHE_ATTRIBUTION_VERSION
        or not cache_entry.canonical_model_id
        or not cache_entry.canonical_model_id.strip()
        or not cache_entry.cache_model_id
        or not cache_entry.cache_model_id.strip()
        or not isinstance(cache_entry.package_manifest_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", cache_entry.package_manifest_hash) is None
    ):
        LOGGER.warning(
            "Ignoring invalid auto-load cache entry because canonical package "
            "attribution is missing or malformed."
        )
        return None
    try:
        package_dir = resolve_existing_model_package_cache_path(
            model_id=cache_entry.cache_model_id,
            package_id=cache_entry.model_package_id,
        )
        if package_dir is None:
            return None
        package_config = parse_model_config(
            config_path=os.path.join(package_dir, MODEL_CONFIG_FILE_NAME)
        )
    except Exception as error:
        LOGGER.warning(
            "Ignoring invalid auto-load cache entry because its package "
            "attribution manifest could not be verified: %s",
            error,
        )
        return None
    if (
        package_config.offline_manifest_version != OFFLINE_CACHE_MANIFEST_VERSION
        or package_config.model_id != cache_entry.cache_model_id
        or package_config.canonical_model_id != cache_entry.canonical_model_id
        or package_config.manifest_content_hash != cache_entry.package_manifest_hash
        or package_config.model_architecture != cache_entry.model_architecture
        or package_config.task_type != cache_entry.task_type
        or package_config.backend_type != cache_entry.backend_type
        or package_config.model_features != cache_entry.model_features
        or package_config.trusted_source != cache_entry.trusted_source
        or package_config.model_dependencies
        != [
            dependency.model_dump(mode="json")
            for dependency in (cache_entry.model_dependencies or [])
        ]
        or package_config.recommended_parameters
        != (
            cache_entry.recommended_parameters.model_dump(mode="json")
            if cache_entry.recommended_parameters is not None
            else None
        )
        or package_config.runtime_compatibility_hash
        != _runtime_compatibility_hash(runtime_x_ray=x_ray_runtime_environment())
    ):
        LOGGER.warning(
            "Ignoring invalid auto-load cache entry because its resolution "
            "metadata does not match the package manifest published by the "
            "successful warm."
        )
        return None
    if (
        not _validate_cached_package_artifacts(
            package_dir=package_dir,
            identities=package_config.package_artifacts,
        )
        or not _validate_cached_dependency_package_paths(
            package_dir=package_dir,
            identities=package_config.dependency_package_paths,
        )
        or not _validate_cached_package_layout(
            package_dir=package_dir,
            artifact_identities=package_config.package_artifacts,
            dependency_identities=package_config.dependency_package_paths,
        )
    ):
        return None
    return package_dir


def attempt_loading_model_with_auto_load_cache(
    use_auto_resolution_cache: bool,
    auto_resolution_cache: AutoResolutionCache,
    auto_negotiation_hash: str,
    model_access_manager: ModelAccessManager,
    model_name_or_path: str,
    model_init_kwargs: dict,
    api_key: Optional[str],
    allow_loading_dependency_models: bool,
    forwarded_kwargs_values: Dict[str, Any],
    verbose: bool = False,
    weights_provider: str = "roboflow",
    max_package_loading_attempts: Optional[int] = None,
    model_download_file_lock_acquire_timeout: int = FILE_LOCK_ACQUIRE_TIMEOUT,
    allow_untrusted_packages: bool = False,
    trt_engine_host_code_allowed: bool = True,
    allow_local_code_packages: bool = True,
    verify_hash_while_download: bool = True,
    download_files_without_hash: bool = False,
    allow_direct_local_storage_loading: bool = True,
    dependency_models_params: Optional[dict] = None,
    weights_provider_extra_query_params: Optional[List[Tuple[str, str]]] = None,
    weights_provider_extra_headers: Optional[Dict[str, str]] = None,
    expected_offline_compatibility_hash: Optional[str] = None,
) -> Optional[AnyModel]:
    if not use_auto_resolution_cache:
        return None
    verbose_info(
        message=f"Attempt to load model {model_name_or_path} using auto-load cache.",
        verbose_requested=verbose,
    )
    cache_entry = auto_resolution_cache.retrieve(
        auto_negotiation_hash=auto_negotiation_hash
    )
    if cache_entry is None:
        verbose_info(
            message=f"Could not find auto-load cache for model {model_name_or_path}.",
            verbose_requested=verbose,
        )
        return None
    cache_entry_invalidated = False

    def invalidate_cache_entry(reason: str) -> None:
        nonlocal cache_entry_invalidated
        if cache_entry_invalidated:
            return
        if OFFLINE_MODE:
            LOGGER.warning(
                "Preserving unusable auto-load cache entry for model %s in "
                "OFFLINE_MODE: %s",
                model_name_or_path,
                reason,
            )
            return
        LOGGER.warning(
            "Invalidating unusable auto-load cache entry for model %s: %s",
            model_name_or_path,
            reason,
        )
        auto_resolution_cache.invalidate(auto_negotiation_hash=auto_negotiation_hash)
        cache_entry_invalidated = True

    if cache_entry.model_id != model_name_or_path:
        LOGGER.warning(
            "Ignoring auto-load cache entry for model %s while loading %s.",
            cache_entry.model_id,
            model_name_or_path,
        )
        invalidate_cache_entry("cached requested-model identity does not match")
        return None
    if (
        expected_offline_compatibility_hash is not None
        and cache_entry.offline_compatibility_hash
        != expected_offline_compatibility_hash
    ):
        LOGGER.warning(
            "Ignoring auto-load cache entry for model %s because it was "
            "registered for different model-loading constraints.",
            model_name_or_path,
        )
        return None
    if not allow_untrusted_packages and cache_entry.trusted_source is not True:
        verbose_info(
            message=(
                f"Auto-load cache for {model_name_or_path} does not prove that "
                "the selected package came from a trusted source."
            ),
            verbose_requested=verbose,
        )
        return None
    model_package_cache_dir = _verified_auto_cache_package_dir(cache_entry=cache_entry)
    if model_package_cache_dir is None:
        invalidate_cache_entry("cached package verification failed")
        return None
    if not model_access_manager.is_model_package_access_granted(
        model_id=cache_entry.canonical_model_id,
        package_id=cache_entry.model_package_id,
        api_key=api_key,
    ):
        return None
    if not all_files_exist(files=cache_entry.resolved_files):
        verbose_info(
            message=f"Could not find all required files denoted in auto-load cache for model {model_name_or_path}.",
            verbose_requested=verbose,
        )
        invalidate_cache_entry("cached package is missing resolved files")
        return None
    try:
        model_dependencies = cache_entry.model_dependencies or []
        if model_dependencies and not allow_loading_dependency_models:
            LOGGER.warning(
                "Ignoring auto-load cache entry for %s because dependency "
                "loading is disabled for this request.",
                model_name_or_path,
            )
            return None
        package_config = parse_model_config(
            config_path=os.path.join(
                model_package_cache_dir,
                MODEL_CONFIG_FILE_NAME,
            )
        )
        expected_dependency_paths = _expected_dependency_package_paths(
            model_dependencies=model_dependencies,
            identities=package_config.dependency_package_paths,
        )
        model_dependencies_instances = {}
        dependency_models_params = dependency_models_params or {}
        for model_dependency in model_dependencies:
            dependency_params = dict(
                dependency_models_params.get(model_dependency.name, {})
            )
            dependency_params["model_id_or_path"] = model_dependency.model_id
            dependency_params["model_package_id"] = model_dependency.model_package_id
            resolved_model_parameters = prepare_dependency_model_parameters(
                model_parameters=dependency_params
            )

            for name, value in forwarded_kwargs_values.items():
                if name not in resolved_model_parameters.model_extra:
                    resolved_model_parameters.model_extra[name] = value
            verbose_info(
                message=f"Initialising dependent model: {model_dependency.model_id}",
                verbose_requested=verbose,
            )
            resolved_dependency_directories: List[str] = []

            def dependency_directory_pointer(model_dir: str) -> None:
                resolved_dependency_directories.append(os.path.realpath(model_dir))

            dependency_instance = AutoModel.from_pretrained(
                model_id_or_path=resolved_model_parameters.model_id_or_path,
                weights_provider=weights_provider,
                api_key=api_key,
                model_package_id=resolved_model_parameters.model_package_id,
                backend=resolved_model_parameters.backend,
                batch_size=resolved_model_parameters.batch_size,
                quantization=resolved_model_parameters.quantization,
                onnx_execution_providers=resolved_model_parameters.onnx_execution_providers,
                device=resolved_model_parameters.device,
                default_onnx_trt_options=resolved_model_parameters.default_onnx_trt_options,
                max_package_loading_attempts=max_package_loading_attempts,
                verbose=verbose,
                model_download_file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
                allow_untrusted_packages=allow_untrusted_packages,
                trt_engine_host_code_allowed=trt_engine_host_code_allowed,
                allow_local_code_packages=allow_local_code_packages,
                verify_hash_while_download=verify_hash_while_download,
                download_files_without_hash=download_files_without_hash,
                use_auto_resolution_cache=use_auto_resolution_cache,
                auto_resolution_cache=auto_resolution_cache,
                allow_direct_local_storage_loading=allow_direct_local_storage_loading,
                model_access_manager=model_access_manager,
                nms_fusion_preferences=resolved_model_parameters.nms_fusion_preferences,
                model_type=resolved_model_parameters.model_type,
                task_type=resolved_model_parameters.task_type,
                allow_loading_dependency_models=False,
                dependency_models_params=None,
                point_model_directory=dependency_directory_pointer,
                weights_provider_extra_query_params=weights_provider_extra_query_params,
                weights_provider_extra_headers=weights_provider_extra_headers,
                **resolved_model_parameters.kwargs,
            )
            if len(resolved_dependency_directories) != 1 or (
                _dependency_package_identity_for_path(
                    dependency_name=model_dependency.name,
                    dependency_directory=resolved_dependency_directories[0],
                )
                != expected_dependency_paths[model_dependency.name]
            ):
                raise CorruptedModelPackageError(
                    message=(
                        f"Cached dependency `{model_dependency.name}` resolved "
                        "to a different package directory than the parent manifest."
                    ),
                    help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
                )
            model_dependencies_instances[model_dependency.name] = dependency_instance
        model_class = resolve_model_class(
            model_architecture=cache_entry.model_architecture,
            task_type=cache_entry.task_type,
            backend=cache_entry.backend_type,
            model_features=(
                set(cache_entry.model_features) if cache_entry.model_features else None
            ),
        )
        model_init_kwargs[MODEL_DEPENDENCIES_KEY] = model_dependencies_instances
        # Cache stores the already-resolved (package-vs-model) value written
        # in initialize_model — no need to re-run resolve_recommended_parameters.
        if cache_entry.recommended_parameters is not None:
            model_init_kwargs["recommended_parameters"] = (
                cache_entry.recommended_parameters
            )
        model = model_class.from_pretrained(
            model_package_cache_dir,
            **_prepare_library_model_init_kwargs(
                model_class=model_class,
                model_init_kwargs=model_init_kwargs,
            ),
        )
        post_initialization_package_dir = _verified_auto_cache_package_dir(
            cache_entry=cache_entry
        )
        if post_initialization_package_dir is None or os.path.realpath(
            post_initialization_package_dir
        ) != os.path.realpath(model_package_cache_dir):
            raise CorruptedModelPackageError(
                message=(
                    "Cached model package attribution, artefacts, dependencies, "
                    "or layout changed while the model was being initialized."
                ),
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        _record_model_package_path(
            model=model,
            package_dir=model_package_cache_dir,
        )
        verbose_info(
            message=f"Successfully loaded model {model_name_or_path} using auto-loading cache.",
            verbose_requested=verbose,
        )
        return model
    except CorruptedModelPackageError as error:
        invalidate_cache_entry(str(error))
        return None
    except Exception as error:
        LOGGER.warning(
            f"Encountered error {error} of type {type(error)} when attempted to load model using "
            f"auto-load cache. Contact Roboflow submitting "
            f"issue under: https://github.com/roboflow/inference/issues/"
        )
        invalidate_cache_entry(str(error))
        return None


def _find_direct_cached_model_package_dir(model_id: str) -> Optional[str]:
    for package_dir in _iterate_cached_model_package_dirs(model_id=model_id):
        try:
            model_config = parse_model_config(
                config_path=os.path.join(package_dir, MODEL_CONFIG_FILE_NAME)
            )
        except Exception:
            continue
        if (
            model_config.offline_manifest_version != OFFLINE_CACHE_MANIFEST_VERSION
            or model_config.model_id != model_id
            or model_config.canonical_model_id != model_id
        ):
            continue
        if (
            not _validate_cached_package_artifacts(
                package_dir=package_dir,
                identities=model_config.package_artifacts,
            )
            or not _validate_cached_dependency_package_paths(
                package_dir=package_dir,
                identities=model_config.dependency_package_paths,
            )
            or not _validate_cached_package_layout(
                package_dir=package_dir,
                artifact_identities=model_config.package_artifacts,
                dependency_identities=model_config.dependency_package_paths,
            )
        ):
            continue
        if model_config.task_type is None:
            continue
        if not model_config.is_library_model() and (
            model_config.model_module is None or model_config.model_class is None
        ):
            continue
        return package_dir
    return None


def _cached_package_model_identity(
    package_dir: str,
) -> Optional[Tuple[str, TaskType, Optional[ModelArchitecture]]]:
    """Return current canonical metadata for an already-verified package."""

    try:
        package_config = parse_model_config(
            config_path=os.path.join(package_dir, MODEL_CONFIG_FILE_NAME)
        )
    except Exception:
        return None
    if (
        package_config.offline_manifest_version != OFFLINE_CACHE_MANIFEST_VERSION
        or not package_config.canonical_model_id
        or package_config.task_type is None
    ):
        return None
    return (
        package_config.canonical_model_id,
        package_config.task_type,
        package_config.model_architecture,
    )


def find_cached_model_package_dir(
    model_id: str,
    api_key: Optional[str] = None,
) -> Optional[str]:
    """Return the path to a locally-cached model package for *model_id*, or ``None``.

    Direct canonical packages are discovered from the model cache. Alias
    packages are discovered only through an auto-resolution entry written for
    this exact effective credential and verified against the package manifest.
    """
    _validate_remote_model_id(model_id=model_id)
    effective_api_key = _resolve_effective_api_key(
        api_key=api_key,
        provider_requires_network=True,
    )
    direct_package = _find_direct_cached_model_package_dir(model_id=model_id)
    direct_identity = (
        _cached_package_model_identity(package_dir=direct_package)
        if direct_package is not None
        else None
    )
    if direct_identity is None:
        direct_package = None

    auto_resolution_cache = BaseAutoLoadMetadataCache(
        file_lock_acquire_timeout=FILE_LOCK_ACQUIRE_TIMEOUT,
    )
    requested_credential_hash = (
        _credential_hash(api_key=effective_api_key) if effective_api_key else None
    )
    candidates = auto_resolution_cache.find_model_candidates(
        model_id=model_id,
        credential_hash=requested_credential_hash,
    )
    valid_candidates: List[Tuple[Tuple[object, ...], str]] = []
    for _, cache_entry in candidates:
        if (
            cache_entry.cache_attribution_version != CACHE_ATTRIBUTION_VERSION
            or not isinstance(cache_entry.credential_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", cache_entry.credential_hash) is None
            or (
                requested_credential_hash is not None
                and cache_entry.credential_hash != requested_credential_hash
            )
        ):
            continue
        package_dir = _verified_auto_cache_package_dir(cache_entry=cache_entry)
        if package_dir is None:
            continue
        valid_candidates.append(
            (
                (
                    cache_entry.canonical_model_id,
                    cache_entry.task_type,
                    cache_entry.model_architecture,
                ),
                package_dir,
            )
        )
    candidate_identities = {
        candidate_identity for candidate_identity, _ in valid_candidates
    }
    if effective_api_key:
        # A current credential-scoped resolution is more specific than a
        # package historically cached under the same requested string.
        if len(candidate_identities) == 1:
            return valid_candidates[0][1]
        if candidates:
            LOGGER.warning(
                "Ignoring cached alias metadata for %s because entries for the "
                "current credential are invalid or conflicting.",
                model_id,
            )
            return None
        return direct_package
    if direct_package is not None:
        valid_candidates.insert(0, (direct_identity, direct_package))
        candidate_identities.add(direct_identity)
    if len(candidate_identities) != 1:
        if valid_candidates:
            LOGGER.warning(
                "Ignoring cached alias metadata for %s because current entries "
                "resolve to conflicting canonical model metadata.",
                model_id,
            )
        return None
    return valid_candidates[0][1]


def _iterate_cached_model_package_dirs(model_id: str) -> Generator[str, None, None]:
    # model_id may originate from request data - resolve both roots and make
    # sure scanned paths cannot escape the models cache (also guards against
    # symlinked cache entries pointing outside of it).
    models_cache_root = os.path.realpath(generate_models_cache_dir())
    try:
        cache_root_candidates = generate_model_cache_root_candidates_for_model_id(
            model_id=model_id
        )
    except Exception:
        return
    package_ids_by_casefold: Dict[str, Set[str]] = {}
    for lexical_cache_root in cache_root_candidates:
        if os.path.islink(lexical_cache_root):
            continue
        cache_root = os.path.realpath(lexical_cache_root)
        if not cache_root.startswith(models_cache_root + os.sep):
            continue
        if not os.path.isdir(cache_root):
            continue
        try:
            entries = sorted(os.listdir(cache_root))
        except OSError:
            continue
        for entry in entries:
            package_ids_by_casefold.setdefault(entry.casefold(), set()).add(entry)
    seen_package_dirs = set()
    for casefold_package_id in sorted(package_ids_by_casefold):
        package_ids = package_ids_by_casefold[casefold_package_id]
        if len(package_ids) != 1:
            LOGGER.warning(
                "Ignoring case-ambiguous cached package IDs for model %s: %s",
                model_id,
                sorted(package_ids),
            )
            continue
        package_id = next(iter(package_ids))
        try:
            package_dir = resolve_existing_model_package_cache_path(
                model_id=model_id,
                package_id=package_id,
            )
        except Exception:
            continue
        if package_dir is None or package_dir in seen_package_dirs:
            continue
        seen_package_dirs.add(package_dir)
        yield package_dir


def attempt_loading_model_from_offline_cache(
    model_id: str,
    model_init_kwargs: dict,
    requested_model_package_id: Optional[str] = None,
    requested_backends: Optional[
        Union[str, BackendType, List[Union[str, BackendType]]]
    ] = None,
    requested_batch_size: Optional[Union[int, Tuple[int, int]]] = None,
    requested_quantization: Optional[
        Union[str, Quantization, List[Union[str, Quantization]]]
    ] = None,
    model_access_manager: Optional[ModelAccessManager] = None,
    api_key: Optional[str] = None,
    allow_local_code_packages: bool = True,
    allow_untrusted_packages: bool = False,
    allow_loading_dependency_models: bool = True,
    dependency_models_params: Optional[dict] = None,
    forwarded_kwargs_values: Optional[Dict[str, Any]] = None,
    weights_provider: str = "roboflow",
    auto_resolution_cache: Optional[AutoResolutionCache] = None,
    use_auto_resolution_cache: bool = True,
    max_package_loading_attempts: Optional[int] = None,
    model_download_file_lock_acquire_timeout: int = FILE_LOCK_ACQUIRE_TIMEOUT,
    trt_engine_host_code_allowed: bool = True,
    verify_hash_while_download: bool = True,
    download_files_without_hash: bool = False,
    allow_direct_local_storage_loading: bool = True,
    nms_fusion_preferences: Optional[Union[bool, dict]] = None,
    weights_provider_extra_query_params: Optional[List[Tuple[str, str]]] = None,
    weights_provider_extra_headers: Optional[Dict[str, str]] = None,
    verbose: bool = False,
    offline_compatibility_hash: Optional[str] = None,
) -> Optional[Tuple[AnyModel, str]]:
    """Try to load a model from local cache when the API is unreachable.

    Scans the model's cache root for package directories containing
    ``model_config.json`` and attempts to load each until one succeeds.
    Returns ``(model, package_dir)`` on success, ``None`` if no cached
    package could be loaded.
    """
    _validate_remote_model_id(model_id=model_id)
    provider_requires_network = model_provider_requires_network(
        provider=weights_provider
    )
    effective_cache_api_key = _resolve_effective_api_key(
        api_key=api_key,
        provider_requires_network=provider_requires_network,
    )
    if effective_cache_api_key:
        return None
    found_any_package = False
    current_runtime_compatibility_hash = _runtime_compatibility_hash(
        runtime_x_ray=x_ray_runtime_environment()
    )
    candidates: Dict[str, Tuple[str, InferenceModelConfig, ModelPackageMetadata]] = {}
    for package_dir in _iterate_cached_model_package_dirs(model_id=model_id):
        package_id = os.path.basename(package_dir)
        if (
            requested_model_package_id is not None
            and package_id != requested_model_package_id
        ):
            continue
        if (
            model_access_manager is not None
            and not model_access_manager.is_model_package_access_granted(
                model_id=model_id,
                package_id=package_id,
                api_key=api_key,
            )
        ):
            continue
        try:
            package_config = parse_model_config(
                config_path=os.path.join(
                    package_dir,
                    MODEL_CONFIG_FILE_NAME,
                )
            )
        except Exception as error:
            LOGGER.warning(
                f"Failed to inspect cached model package from {package_dir}: {error}"
            )
            continue
        is_versioned_manifest = (
            package_config.offline_manifest_version == OFFLINE_CACHE_MANIFEST_VERSION
        )
        has_canonical_attribution = (
            is_versioned_manifest
            and package_config.model_id == model_id
            and package_config.canonical_model_id == model_id
        )
        if not has_canonical_attribution:
            LOGGER.warning(
                "Ignoring cached package %s because it has no current canonical "
                "model attribution. Re-warm it with this inference-models version.",
                package_dir,
            )
            continue
        if (
            not _validate_cached_package_artifacts(
                package_dir=package_dir,
                identities=package_config.package_artifacts,
            )
            or not _validate_cached_dependency_package_paths(
                package_dir=package_dir,
                identities=package_config.dependency_package_paths,
            )
            or not _validate_cached_package_layout(
                package_dir=package_dir,
                artifact_identities=package_config.package_artifacts,
                dependency_identities=package_config.dependency_package_paths,
            )
        ):
            continue
        if (
            is_versioned_manifest
            and package_config.runtime_compatibility_hash
            != current_runtime_compatibility_hash
        ):
            LOGGER.warning(
                "Ignoring cached package %s because it was warmed in a different "
                "runtime environment.",
                package_dir,
            )
            continue
        if (
            offline_compatibility_hash is not None
            and package_config.offline_compatibility_hash != offline_compatibility_hash
        ):
            LOGGER.warning(
                "Ignoring cached package %s because it was not warmed for the "
                "current model-loading constraints.",
                package_dir,
            )
            continue
        if not allow_untrusted_packages and package_config.trusted_source is not True:
            continue
        if package_config.backend_type is None:
            continue
        if not model_implementation_exists(
            model_architecture=package_config.model_architecture,
            task_type=package_config.task_type,
            backend=package_config.backend_type,
            model_features=(
                set(package_config.model_features)
                if package_config.model_features
                else None
            ),
        ):
            continue
        try:
            package_quantization = (
                Quantization(package_config.quantization)
                if package_config.quantization is not None
                else Quantization.UNKNOWN
            )
        except ValueError:
            continue
        package_metadata = ModelPackageMetadata(
            package_id=package_id,
            backend=package_config.backend_type,
            package_artefacts=[],
            package_source=PackageSourceType.LOCAL_CACHE,
            quantization=package_quantization,
            dynamic_batch_size_supported=package_config.dynamic_batch_size_supported,
            static_batch_size=package_config.static_batch_size,
            trusted_source=package_config.trusted_source is True,
            model_features=package_config.model_features,
        )
        candidates[package_id] = (
            package_dir,
            package_config,
            package_metadata,
        )

    matching_packages = [candidate[2] for candidate in candidates.values()]
    if requested_model_package_id is None:
        feature_compatible_packages = []
        for package_metadata in matching_packages:
            package_config = candidates[package_metadata.package_id][1]
            try:
                compatible = [package_metadata]
                if requested_backends is not None:
                    compatible, _ = filter_model_packages_by_requested_backend(
                        model_packages=compatible,
                        requested_backends=requested_backends,
                        verbose=verbose,
                    )
                if requested_batch_size is not None:
                    compatible, _ = filter_model_packages_by_requested_batch_size(
                        model_packages=compatible,
                        requested_batch_size=requested_batch_size,
                        verbose=verbose,
                    )
                effective_quantization = requested_quantization
                default_quantization_used = False
                if effective_quantization is None:
                    default_quantization_used = True
                    effective_quantization = determine_default_allowed_quantization(
                        device=model_init_kwargs.get("device")
                    )
                if effective_quantization:
                    compatible, _ = filter_model_packages_by_requested_quantization(
                        model_packages=compatible,
                        requested_quantization=effective_quantization,
                        default_quantization_used=default_quantization_used,
                        verbose=verbose,
                    )
                compatible, _ = filter_model_packages_based_on_model_features(
                    model_packages=compatible,
                    nms_fusion_preferences=nms_fusion_preferences,
                    model_architecture=package_config.model_architecture,
                    task_type=package_config.task_type,
                )
                feature_compatible_packages.extend(compatible)
            except Exception as error:
                LOGGER.warning(
                    "Ignoring malformed offline package metadata in %s: %s",
                    candidates[package_metadata.package_id][0],
                    error,
                )
        matching_packages = rank_model_packages(
            model_packages=feature_compatible_packages,
            selected_device=model_init_kwargs.get("device"),
            nms_fusion_preferences=nms_fusion_preferences,
        )
    if max_package_loading_attempts is not None:
        matching_packages = matching_packages[:max_package_loading_attempts]

    dependency_models_params = dependency_models_params or {}
    forwarded_kwargs_values = forwarded_kwargs_values or {}
    for package_metadata in matching_packages:
        package_id = package_metadata.package_id
        package_dir, package_config, _ = candidates[package_id]
        found_any_package = True
        try:
            raw_dependencies = package_config.model_dependencies
            if raw_dependencies is None and not allow_loading_dependency_models:
                raise CorruptedModelPackageError(
                    message=(
                        f"Cannot verify whether cached package {package_id} "
                        "has dependencies while dependency loading is disabled."
                    ),
                    help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
                )
            model_dependencies = [
                ModelDependency.model_validate(dependency)
                for dependency in (raw_dependencies or [])
            ]
            expected_dependency_paths = _expected_dependency_package_paths(
                model_dependencies=model_dependencies,
                identities=package_config.dependency_package_paths,
            )
            if model_dependencies and not allow_loading_dependency_models:
                raise CorruptedModelPackageError(
                    message=(
                        f"Cannot load cached package {package_id} because it "
                        "requires dependency models and dependency loading is disabled."
                    ),
                    help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
                )
            dependency_instances = {}
            for dependency in model_dependencies:
                dependency_params = dict(
                    dependency_models_params.get(dependency.name, {})
                )
                dependency_params["model_id_or_path"] = dependency.model_id
                dependency_params["model_package_id"] = dependency.model_package_id
                resolved_parameters = prepare_dependency_model_parameters(
                    model_parameters=dependency_params
                )
                for name, value in forwarded_kwargs_values.items():
                    if name not in resolved_parameters.model_extra:
                        resolved_parameters.model_extra[name] = value
                resolved_dependency_directories: List[str] = []

                def dependency_directory_pointer(model_dir: str) -> None:
                    resolved_dependency_directories.append(os.path.realpath(model_dir))

                dependency_instances[dependency.name] = AutoModel.from_pretrained(
                    model_id_or_path=resolved_parameters.model_id_or_path,
                    weights_provider=weights_provider,
                    api_key=api_key,
                    model_package_id=resolved_parameters.model_package_id,
                    backend=resolved_parameters.backend,
                    batch_size=resolved_parameters.batch_size,
                    quantization=resolved_parameters.quantization,
                    onnx_execution_providers=resolved_parameters.onnx_execution_providers,
                    device=resolved_parameters.device,
                    default_onnx_trt_options=resolved_parameters.default_onnx_trt_options,
                    max_package_loading_attempts=max_package_loading_attempts,
                    verbose=verbose,
                    model_download_file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
                    allow_untrusted_packages=allow_untrusted_packages,
                    trt_engine_host_code_allowed=trt_engine_host_code_allowed,
                    allow_local_code_packages=allow_local_code_packages,
                    verify_hash_while_download=verify_hash_while_download,
                    download_files_without_hash=download_files_without_hash,
                    use_auto_resolution_cache=use_auto_resolution_cache,
                    auto_resolution_cache=auto_resolution_cache,
                    allow_direct_local_storage_loading=allow_direct_local_storage_loading,
                    model_access_manager=model_access_manager,
                    nms_fusion_preferences=resolved_parameters.nms_fusion_preferences,
                    model_type=resolved_parameters.model_type,
                    task_type=resolved_parameters.task_type,
                    allow_loading_dependency_models=False,
                    dependency_models_params=None,
                    point_model_directory=dependency_directory_pointer,
                    weights_provider_extra_query_params=weights_provider_extra_query_params,
                    weights_provider_extra_headers=weights_provider_extra_headers,
                    **resolved_parameters.kwargs,
                )
                if len(resolved_dependency_directories) != 1 or (
                    _dependency_package_identity_for_path(
                        dependency_name=dependency.name,
                        dependency_directory=resolved_dependency_directories[0],
                    )
                    != expected_dependency_paths[dependency.name]
                ):
                    raise CorruptedModelPackageError(
                        message=(
                            f"Cached dependency `{dependency.name}` resolved "
                            "to a different package directory than the parent manifest."
                        ),
                        help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
                    )
            package_init_kwargs = dict(model_init_kwargs)
            package_init_kwargs[MODEL_DEPENDENCIES_KEY] = dependency_instances
            if package_config.recommended_parameters is not None:
                package_init_kwargs["recommended_parameters"] = (
                    RecommendedParameters.model_validate(
                        package_config.recommended_parameters
                    )
                )
            model = attempt_loading_model_from_local_storage(
                model_dir_or_weights_path=package_dir,
                allow_local_code_packages=allow_local_code_packages,
                model_init_kwargs=package_init_kwargs,
            )
            _record_model_package_path(model=model, package_dir=package_dir)
            verbose_info(
                message=f"Loaded model {model_id} from offline cache at {package_dir}.",
                verbose_requested=verbose,
            )
            if model_access_manager is not None:
                model_access_manager.on_model_loaded(
                    model=model,
                    access_identifiers=AccessIdentifiers(
                        model_id=model_id,
                        package_id=package_id,
                        api_key=api_key,
                    ),
                    model_storage_path=package_dir,
                )
            return model, package_dir
        except Exception as error:
            LOGGER.warning(
                f"Failed to load cached model package from {package_dir}: {error}"
            )
    if not found_any_package:
        verbose_info(
            message=f"No offline cache packages found for model {model_id}.",
            verbose_requested=verbose,
        )
    else:
        verbose_info(
            message=f"No usable cached model package found for {model_id}.",
            verbose_requested=verbose,
        )
    return None


def all_files_exist(files: List[str]) -> bool:
    return all(os.path.exists(f) for f in files)


def _prepare_library_model_init_kwargs(
    model_class: Any, model_init_kwargs: dict
) -> dict:
    if not OFFLINE_MODE:
        return model_init_kwargs
    try:
        loader_parameters = inspect.signature(model_class.from_pretrained).parameters
    except (TypeError, ValueError):
        return model_init_kwargs
    local_files_only_parameter = loader_parameters.get("local_files_only")
    if local_files_only_parameter is None or local_files_only_parameter.kind is (
        inspect.Parameter.POSITIONAL_ONLY
    ):
        return model_init_kwargs
    return {**model_init_kwargs, "local_files_only": True}


def attempt_loading_matching_model_packages(
    model_id: str,
    model_architecture: ModelArchitecture,
    task_type: Optional[TaskType],
    matching_model_packages: List[ModelPackageMetadata],
    model_init_kwargs: dict,
    model_access_manager: ModelAccessManager,
    auto_resolution_cache: AutoResolutionCache,
    auto_negotiation_hash: str,
    api_key: Optional[str],
    model_dependencies: Optional[List[ModelDependency]],
    model_dependencies_instances: Dict[str, AnyModel],
    model_dependencies_directories: Dict[str, str],
    recommended_parameters: Optional[RecommendedParameters] = None,
    max_package_loading_attempts: Optional[int] = None,
    model_download_file_lock_acquire_timeout: int = FILE_LOCK_ACQUIRE_TIMEOUT,
    verbose: bool = True,
    verify_hash_while_download: bool = True,
    download_files_without_hash: bool = False,
    use_auto_resolution_cache: bool = True,
    point_model_directory: Optional[Callable[[str], None]] = None,
    requested_model_id: Optional[str] = None,
    offline_compatibility_hash: Optional[str] = None,
) -> AnyModel:
    if requested_model_id is None:
        requested_model_id = model_id
    if max_package_loading_attempts is not None:
        matching_model_packages = matching_model_packages[:max_package_loading_attempts]
    if not matching_model_packages:
        raise NoModelPackagesAvailableError(
            message=f"Cannot load model {model_id} - no matching model package candidates for given model "
            f"running in this environment.",
            help_url="https://inference-models.roboflow.com/errors/package-negotiation/#nomodelpackagesavailableerror",
        )
    failed_load_attempts: List[Tuple[str, Exception]] = []
    for idx, model_package in enumerate(matching_model_packages):
        access_identifiers = AccessIdentifiers(
            model_id=model_id,
            package_id=model_package.package_id,
            api_key=api_key,
        )
        verbose_info(
            message=f"Attempt to load model package: {model_package.get_summary()}",
            verbose_requested=verbose,
        )
        try:
            model, model_package_cache_dir = initialize_model(
                model_id=model_id,
                requested_model_id=requested_model_id,
                model_architecture=model_architecture,
                task_type=task_type,
                model_package=model_package,
                model_download_file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
                model_init_kwargs=model_init_kwargs,
                auto_resolution_cache=auto_resolution_cache,
                auto_negotiation_hash=auto_negotiation_hash,
                offline_compatibility_hash=offline_compatibility_hash,
                model_dependencies=model_dependencies,
                model_dependencies_instances=model_dependencies_instances,
                model_dependencies_directories=model_dependencies_directories,
                recommended_parameters=recommended_parameters,
                verify_hash_while_download=verify_hash_while_download,
                download_files_without_hash=download_files_without_hash,
                on_file_created=partial(
                    model_access_manager.on_file_created,
                    access_identifiers=access_identifiers,
                ),
                on_file_renamed=partial(
                    model_access_manager.on_file_renamed,
                    access_identifiers=access_identifiers,
                ),
                on_symlink_created=partial(
                    model_access_manager.on_symlink_created,
                    access_identifiers=access_identifiers,
                ),
                on_symlink_deleted=model_access_manager.on_symlink_deleted,
                use_auto_resolution_cache=use_auto_resolution_cache,
                api_key=api_key,
            )
            LOGGER.info(
                "Loaded model %s with backend %s (package %s)",
                model_id,
                model_package.backend.value,
                model_package.package_id,
            )
            model_access_manager.on_model_loaded(
                model=model,
                access_identifiers=access_identifiers,
                model_storage_path=model_package_cache_dir,
            )
            if point_model_directory:
                point_model_directory(model_package_cache_dir)
            return model
        except Exception as error:
            LOGGER.warning(
                f"Model package with id {model_package.package_id} that was selected to be loaded "
                f"failed to load with error: {error} of type {error.__class__.__name__}. This may "
                f"be caused several issues. If you see this warning after manually specifying model "
                f"package to be loaded - make sure that all required dependencies are installed. If "
                f"that warning is displayed when the model package was auto-selected - there is most "
                f"likely a bug in `inference-models` and you should raise an issue providing full context of "
                f"the event. https://github.com/roboflow/inference/issues"
            )
            next_idx = idx + 1
            if next_idx < len(matching_model_packages):
                next_backend = matching_model_packages[next_idx].backend.value
                LOGGER.warning(
                    "Falling back from %s to %s backend for model %s",
                    model_package.backend.value,
                    next_backend,
                    model_id,
                )
            failed_load_attempts.append((model_package.package_id, error))

    summary_of_errors = "\n".join(
        f"\t* model_package_id={model_package_id} error={error} error_type={error.__class__.__name__}"
        for model_package_id, error in failed_load_attempts
    )
    raise ModelPackageAlternativesExhaustedError(
        message=f"Could not load any of model package candidate for model {model_id}. This may "
        f"be caused several issues. If you see this warning after manually specifying model "
        f"package to be loaded - make sure that all required dependencies are installed. If "
        f"that warning is displayed when the model package was auto-selected - there is most "
        f"likely a bug in `inference-models` and you should raise an issue providing full context of "
        f"the event. https://github.com/roboflow/inference/issues\n\n"
        f"Here is the summary of errors for specific model packages:\n{summary_of_errors}\n\n",
        help_url="https://inference-models.roboflow.com/errors/model-loading/#modelpackagealternativesexhaustederror",
        alternatives_errors=[summary[1] for summary in failed_load_attempts],
    )


def _validate_existing_cache_package_attribution(
    package_dir: str,
    cache_model_id: str,
    canonical_model_id: str,
    expected_manifest_fields: Optional[dict] = None,
    package_artifact_declarations: Optional[List[dict]] = None,
    dependency_package_paths: Optional[List[dict]] = None,
    materialized_package_artifacts: Optional[List[dict]] = None,
) -> None:
    config_path = os.path.join(package_dir, MODEL_CONFIG_FILE_NAME)
    if not os.path.lexists(config_path):
        return
    if os.path.islink(config_path) or not os.path.isfile(config_path):
        raise CorruptedModelPackageError(
            message="Cached model metadata is not a regular file.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    try:
        content = read_json(path=config_path)
    except (OSError, ValueError) as error:
        raise CorruptedModelPackageError(
            message="Cached model metadata cannot be decoded.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        ) from error
    if not isinstance(content, dict):
        raise CorruptedModelPackageError(
            message="Cached model metadata is not a dictionary.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    existing_model_id = content.get("model_id")
    if existing_model_id is not None and (
        not isinstance(existing_model_id, str)
        or not existing_model_id
        or existing_model_id != cache_model_id
    ):
        raise CorruptedModelPackageError(
            message="Cached package is attributed to a different cache owner.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    existing_canonical_model_id = content.get("canonical_model_id")
    if existing_canonical_model_id is not None and (
        not isinstance(existing_canonical_model_id, str)
        or not existing_canonical_model_id
        or existing_canonical_model_id != canonical_model_id
    ):
        raise CorruptedModelPackageError(
            message="Cached package is attributed to a different canonical model.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    if content.get("offline_manifest_version") == OFFLINE_CACHE_MANIFEST_VERSION and (
        existing_model_id != cache_model_id
        or existing_canonical_model_id != canonical_model_id
    ):
        raise CorruptedModelPackageError(
            message="Current cached model metadata has incomplete attribution.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    if content.get("offline_manifest_version") != OFFLINE_CACHE_MANIFEST_VERSION:
        return
    existing_offline_compatibility_hash = content.get("offline_compatibility_hash")
    if existing_offline_compatibility_hash is not None and (
        not isinstance(existing_offline_compatibility_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", existing_offline_compatibility_hash) is None
    ):
        raise CorruptedModelPackageError(
            message=(
                "Current cached model manifest has invalid offline "
                "compatibility metadata."
            ),
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    try:
        existing_artifact_identities = _parse_package_artifact_identities(
            content.get("package_artifacts")
        )
        existing_dependency_paths = _parse_dependency_package_path_identities(
            content.get("dependency_package_paths")
        )
    except CorruptedModelPackageError as error:
        raise CorruptedModelPackageError(
            message="Current cached model manifest has invalid materialization identity.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        ) from error
    if (
        expected_manifest_fields is None
        or package_artifact_declarations is None
        or dependency_package_paths is None
        or any(
            content.get(field_name) != expected_value
            for field_name, expected_value in expected_manifest_fields.items()
        )
        or _artifact_declarations_from_identities(
            identities=existing_artifact_identities
        )
        != package_artifact_declarations
        or (
            materialized_package_artifacts is not None
            and existing_artifact_identities != materialized_package_artifacts
        )
        or existing_dependency_paths != dependency_package_paths
    ):
        raise CorruptedModelPackageError(
            message=(
                "Refusing to mutate a current cached package because the "
                "incoming provenance, selection metadata, or artefact identity "
                "does not match its published manifest."
            ),
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )


def initialize_model(
    model_id: str,
    model_architecture: ModelArchitecture,
    task_type: Optional[TaskType],
    model_package: ModelPackageMetadata,
    model_init_kwargs: dict,
    auto_resolution_cache: AutoResolutionCache,
    auto_negotiation_hash: str,
    model_dependencies: Optional[List[ModelDependency]],
    model_dependencies_instances: Dict[str, AnyModel],
    model_dependencies_directories: Dict[str, str],
    recommended_parameters: Optional[RecommendedParameters] = None,
    model_download_file_lock_acquire_timeout: int = FILE_LOCK_ACQUIRE_TIMEOUT,
    verify_hash_while_download: bool = True,
    download_files_without_hash: bool = False,
    on_file_created: Optional[Callable[[str], None]] = None,
    on_file_renamed: Optional[Callable[[str, str], None]] = None,
    on_symlink_created: Optional[Callable[[str, str], None]] = None,
    on_symlink_deleted: Optional[Callable[[str], None]] = None,
    use_auto_resolution_cache: bool = True,
    requested_model_id: Optional[str] = None,
    offline_compatibility_hash: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[AnyModel, str]:
    if not isinstance(model_id, str) or not model_id.strip():
        raise CorruptedModelPackageError(
            message="Cannot initialize a model without a canonical model identity.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    if requested_model_id is None:
        requested_model_id = model_id
    if not isinstance(requested_model_id, str) or not requested_model_id.strip():
        raise CorruptedModelPackageError(
            message="Cannot initialize a model without a requested model identity.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    cache_model_id = model_package.cache_model_id
    if cache_model_id is None:
        cache_model_id = model_id
    if not isinstance(cache_model_id, str) or not cache_model_id.strip():
        raise CorruptedModelPackageError(
            message="Cannot initialize a model without a valid cache owner identity.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    package_artifact_declarations = _package_artifact_declarations(
        package_artefacts=model_package.package_artefacts
    )
    if (
        model_package.package_source is not PackageSourceType.LOCAL_CACHE
        and not download_files_without_hash
        and any(
            declaration["unhashed"] for declaration in package_artifact_declarations
        )
    ):
        raise UntrustedFileError(
            message=(
                "Model package contains artefacts without MD5 identities while "
                "download_files_without_hash=False."
            ),
            help_url="https://inference-models.roboflow.com/errors/file-download/#untrustedfileerror",
        )
    dependency_package_paths = _dependency_package_path_identities(
        model_dependencies=model_dependencies,
        model_dependencies_directories=model_dependencies_directories,
    )
    model_dependencies_manifest = [
        dependency.model_dump(mode="json") for dependency in (model_dependencies or [])
    ]
    resolved_recommended_parameters = resolve_recommended_parameters(
        package_level=model_package.recommended_parameters,
        model_level=recommended_parameters,
    )
    recommended_parameters_manifest = (
        resolved_recommended_parameters.model_dump(mode="json")
        if resolved_recommended_parameters is not None
        else None
    )
    quantization = (
        model_package.quantization.value
        if model_package.quantization is not None
        else Quantization.UNKNOWN.value
    )
    runtime_compatibility_hash = _runtime_compatibility_hash(
        runtime_x_ray=x_ray_runtime_environment()
    )
    expected_manifest_fields = {
        "model_id": cache_model_id,
        "canonical_model_id": model_id,
        "model_architecture": model_architecture,
        "task_type": task_type,
        "backend_type": model_package.backend.value,
        "model_features": model_package.model_features,
        "trusted_source": model_package.trusted_source,
        "model_dependencies": model_dependencies_manifest,
        "recommended_parameters": recommended_parameters_manifest,
        "quantization": quantization,
        "dynamic_batch_size_supported": (model_package.dynamic_batch_size_supported),
        "static_batch_size": model_package.static_batch_size,
        "runtime_compatibility_hash": runtime_compatibility_hash,
    }
    if model_package.package_source == PackageSourceType.LOCAL_CACHE:
        package_path_for_lock = resolve_existing_model_package_cache_path(
            model_id=cache_model_id,
            package_id=model_package.package_id,
            allow_unattributed_local_cache=True,
        )
        if package_path_for_lock is None:
            raise CorruptedModelPackageError(
                message=(
                    f"Could not resolve attributed local cache package "
                    f"{model_package.package_id} for model {cache_model_id}."
                ),
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
    else:
        package_path_for_lock = generate_model_package_cache_path(
            model_id=cache_model_id,
            package_id=model_package.package_id,
        )
    package_parent = os.path.dirname(package_path_for_lock)
    package_lock_path = os.path.join(
        package_parent,
        f".{model_package.package_id}.materialization.lock",
    )
    held_package_locks = _HELD_PACKAGE_MATERIALIZATION_LOCKS.get()
    if OFFLINE_MODE:
        if model_package.package_source != PackageSourceType.LOCAL_CACHE:
            raise CorruptedModelPackageError(
                message=(
                    "OFFLINE_MODE can only initialize packages discovered from "
                    "the verified local cache."
                ),
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        if os.path.islink(package_parent) or not os.path.isdir(package_parent):
            raise CorruptedModelPackageError(
                message="Offline model package parent is not a safe directory.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
    else:
        os.makedirs(package_parent, exist_ok=True)
    if not OFFLINE_MODE and package_lock_path not in held_package_locks:
        if os.path.islink(package_parent) or os.path.islink(package_lock_path):
            raise CorruptedModelPackageError(
                message="Model package materialization lock cannot be a symbolic link.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        with FileLock(
            package_lock_path,
            timeout=model_download_file_lock_acquire_timeout,
        ):
            lock_state_token = _HELD_PACKAGE_MATERIALIZATION_LOCKS.set(
                held_package_locks | {package_lock_path}
            )
            try:
                return initialize_model(
                    model_id=model_id,
                    model_architecture=model_architecture,
                    task_type=task_type,
                    model_package=model_package,
                    model_init_kwargs=model_init_kwargs,
                    auto_resolution_cache=auto_resolution_cache,
                    auto_negotiation_hash=auto_negotiation_hash,
                    model_dependencies=model_dependencies,
                    model_dependencies_instances=model_dependencies_instances,
                    model_dependencies_directories=model_dependencies_directories,
                    recommended_parameters=recommended_parameters,
                    model_download_file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
                    verify_hash_while_download=verify_hash_while_download,
                    download_files_without_hash=download_files_without_hash,
                    on_file_created=on_file_created,
                    on_file_renamed=on_file_renamed,
                    on_symlink_created=on_symlink_created,
                    on_symlink_deleted=on_symlink_deleted,
                    use_auto_resolution_cache=use_auto_resolution_cache,
                    requested_model_id=requested_model_id,
                    offline_compatibility_hash=offline_compatibility_hash,
                    api_key=api_key,
                )
            finally:
                _HELD_PACKAGE_MATERIALIZATION_LOCKS.reset(lock_state_token)
    model_features = None
    if model_package.model_features:
        model_features = set(model_package.model_features.keys())
    model_class = resolve_model_class(
        model_architecture=model_architecture,
        task_type=task_type,
        backend=model_package.backend,
        model_features=model_features,
    )
    for artefact in model_package.package_artefacts:
        if artefact.file_handle == MODEL_CONFIG_FILE_NAME:
            raise CorruptedModelPackageError(
                message=f"For model with id=`{model_id}` and package={model_package.package_id} discovered "
                f"artefact named `{MODEL_CONFIG_FILE_NAME}` which collides with the config file that "
                f"inference is supposed to create for a model in order for compatibility with offline "
                f"loaders. This problem indicate a violation of model package contract and requires change in "
                f"model package structure. If you experience this issue using hosted Roboflow solution, contact "
                f"us to solve the problem.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
    if model_package.package_source == PackageSourceType.LOCAL_CACHE:
        model_package_cache_dir = resolve_existing_model_package_cache_path(
            model_id=cache_model_id,
            package_id=model_package.package_id,
            allow_unattributed_local_cache=True,
        )
        if model_package_cache_dir is None:
            raise CorruptedModelPackageError(
                message=(
                    f"Could not resolve attributed local cache package "
                    f"{model_package.package_id} for model {cache_model_id}."
                ),
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        _validate_existing_cache_package_attribution(
            package_dir=model_package_cache_dir,
            cache_model_id=cache_model_id,
            canonical_model_id=model_id,
            expected_manifest_fields=expected_manifest_fields,
            package_artifact_declarations=package_artifact_declarations,
            dependency_package_paths=dependency_package_paths,
        )
        _validate_package_directory_layout(
            package_dir=model_package_cache_dir,
            artifact_declarations=package_artifact_declarations,
            dependency_package_paths=dependency_package_paths,
        )
        shared_files_mapping = _resolve_local_cache_package_files(
            model_package_cache_dir=model_package_cache_dir,
            package_artefacts=model_package.package_artefacts,
        )
        model_specific_files_mapping: Dict[str, str] = {}
        symlinks_mapping = {
            handle: os.path.join(model_package_cache_dir, handle)
            for handle in shared_files_mapping
        }
    else:
        model_package_cache_dir = generate_model_package_cache_path(
            model_id=cache_model_id,
            package_id=model_package.package_id,
        )
        _validate_existing_cache_package_attribution(
            package_dir=model_package_cache_dir,
            cache_model_id=cache_model_id,
            canonical_model_id=model_id,
            expected_manifest_fields=expected_manifest_fields,
            package_artifact_declarations=package_artifact_declarations,
            dependency_package_paths=dependency_package_paths,
        )
        _validate_package_directory_layout(
            package_dir=model_package_cache_dir,
            artifact_declarations=package_artifact_declarations,
            dependency_package_paths=dependency_package_paths,
        )
        _remove_unattributed_unhashed_artifacts(
            package_dir=model_package_cache_dir,
            declarations=package_artifact_declarations,
        )
        os.makedirs(model_package_cache_dir, exist_ok=True)
        files_specs = [
            (artefact.file_handle, artefact.download_url, artefact.md5_hash)
            for artefact in model_package.package_artefacts
            if isinstance(artefact, FileDownloadSpecs)
        ]
        file_specs_with_hash = [spec for spec in files_specs if spec[2] is not None]
        file_specs_without_hash = [spec for spec in files_specs if spec[2] is None]
        shared_blobs_dir = generate_shared_blobs_path()
        shared_files_mapping = download_files_to_directory(
            target_dir=shared_blobs_dir,
            files_specs=file_specs_with_hash,
            file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
            verify_hash_while_download=verify_hash_while_download,
            download_files_without_hash=download_files_without_hash,
            name_after="md5_hash",
            on_file_created=on_file_created,
            on_file_renamed=on_file_renamed,
        )
        model_specific_files_mapping = download_files_to_directory(
            target_dir=model_package_cache_dir,
            files_specs=file_specs_without_hash,
            file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
            verify_hash_while_download=verify_hash_while_download,
            download_files_without_hash=download_files_without_hash,
            on_file_created=on_file_created,
            on_file_renamed=on_file_renamed,
        )
        symlinks_mapping = create_symlinks_to_shared_blobs(
            model_dir=model_package_cache_dir,
            shared_files_mapping=shared_files_mapping,
            model_download_file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
            on_symlink_created=on_symlink_created,
            on_symlink_deleted=on_symlink_deleted,
        )
    package_artifact_identities = _materialize_package_artifact_identities(
        package_dir=model_package_cache_dir,
        declarations=package_artifact_declarations,
    )
    config_path = os.path.join(model_package_cache_dir, MODEL_CONFIG_FILE_NAME)
    resolved_files = set(shared_files_mapping.values())
    resolved_files.update(model_specific_files_mapping.values())
    resolved_files.update(symlinks_mapping.values())
    if OFFLINE_MODE:
        dependencies_resolved_files = {
            os.path.join(
                model_package_cache_dir,
                MODEL_DEPENDENCIES_SUB_DIR,
                identity["name"],
            )
            for identity in dependency_package_paths
        }
        for identity in dependency_package_paths:
            dependencies_resolved_files.update(
                scan_dependency_directory_for_resolved_files(
                    dependency_directory=identity["target_path"]
                )
            )
    else:
        dependencies_resolved_files = handle_dependencies_directories_creation(
            model_package_cache_dir=model_package_cache_dir,
            model_dependencies_directories={
                identity["name"]: identity["target_path"]
                for identity in dependency_package_paths
            },
            model_download_file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
            on_symlink_created=on_symlink_created,
            on_symlink_deleted=on_symlink_deleted,
        )
    resolved_files.update(dependencies_resolved_files)
    if not _validate_cached_dependency_package_paths(
        package_dir=model_package_cache_dir,
        identities=dependency_package_paths,
    ):
        raise CorruptedModelPackageError(
            message="Model dependency package links failed identity validation.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    _validate_package_directory_layout(
        package_dir=model_package_cache_dir,
        artifact_declarations=package_artifact_declarations,
        dependency_package_paths=dependency_package_paths,
    )
    _validate_existing_cache_package_attribution(
        package_dir=model_package_cache_dir,
        cache_model_id=cache_model_id,
        canonical_model_id=model_id,
        expected_manifest_fields=expected_manifest_fields,
        package_artifact_declarations=package_artifact_declarations,
        dependency_package_paths=dependency_package_paths,
        materialized_package_artifacts=package_artifact_identities,
    )
    model_init_kwargs[MODEL_DEPENDENCIES_KEY] = model_dependencies_instances
    if resolved_recommended_parameters is not None:
        model_init_kwargs["recommended_parameters"] = resolved_recommended_parameters
    model = model_class.from_pretrained(
        model_package_cache_dir,
        **_prepare_library_model_init_kwargs(
            model_class=model_class,
            model_init_kwargs=model_init_kwargs,
        ),
    )
    post_initialization_artifact_identities = _materialize_package_artifact_identities(
        package_dir=model_package_cache_dir,
        declarations=package_artifact_declarations,
    )
    if post_initialization_artifact_identities != package_artifact_identities:
        raise CorruptedModelPackageError(
            message=(
                "Model package artefacts changed while the model was being "
                "initialized."
            ),
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    if not _validate_cached_dependency_package_paths(
        package_dir=model_package_cache_dir,
        identities=dependency_package_paths,
    ):
        raise CorruptedModelPackageError(
            message="Model dependencies changed while the model was being initialized.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    _validate_package_directory_layout(
        package_dir=model_package_cache_dir,
        artifact_declarations=package_artifact_declarations,
        dependency_package_paths=dependency_package_paths,
    )
    if OFFLINE_MODE:
        _validate_existing_cache_package_attribution(
            package_dir=model_package_cache_dir,
            cache_model_id=cache_model_id,
            canonical_model_id=model_id,
            expected_manifest_fields=expected_manifest_fields,
            package_artifact_declarations=package_artifact_declarations,
            dependency_package_paths=dependency_package_paths,
            materialized_package_artifacts=post_initialization_artifact_identities,
        )
        _record_model_package_path(
            model=model,
            package_dir=model_package_cache_dir,
        )
        return model, model_package_cache_dir
    # The versioned manifest is the marker that a package is eligible for raw
    # offline discovery.  Do not publish it until the package has initialized
    # successfully: a failed online candidate may leave downloaded artefacts
    # behind, but those partial artefacts must not be advertised as a warmed
    # offline package on the next restart.
    package_manifest_hash = dump_model_config_for_offline_use(
        config_path=config_path,
        model_architecture=model_architecture,
        task_type=task_type,
        backend_type=model_package.backend,
        file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
        model_id=cache_model_id,
        on_file_created=on_file_created,
        model_features=model_package.model_features,
        trusted_source=model_package.trusted_source,
        model_dependencies=model_dependencies_manifest,
        recommended_parameters=recommended_parameters_manifest,
        quantization=quantization,
        dynamic_batch_size_supported=model_package.dynamic_batch_size_supported,
        static_batch_size=model_package.static_batch_size,
        runtime_compatibility_hash=runtime_compatibility_hash,
        offline_compatibility_hash=offline_compatibility_hash,
        canonical_model_id=model_id,
        package_artifacts=package_artifact_identities,
        dependency_package_paths=dependency_package_paths,
    )
    resolved_files.add(config_path)
    dump_auto_resolution_cache(
        use_auto_resolution_cache=use_auto_resolution_cache,
        auto_resolution_cache=auto_resolution_cache,
        auto_negotiation_hash=auto_negotiation_hash,
        offline_compatibility_hash=offline_compatibility_hash,
        model_id=requested_model_id,
        cache_model_id=cache_model_id,
        canonical_model_id=model_id,
        model_package_id=model_package.package_id,
        model_architecture=model_architecture,
        task_type=task_type,
        backend_type=model_package.backend,
        resolved_files=resolved_files,
        model_dependencies=model_dependencies,
        model_features=model_package.model_features,
        recommended_parameters=resolved_recommended_parameters,
        trusted_source=model_package.trusted_source,
        package_manifest_hash=package_manifest_hash,
        api_key=api_key,
    )
    _record_model_package_path(
        model=model,
        package_dir=model_package_cache_dir,
    )
    return model, model_package_cache_dir


def create_symlinks_to_shared_blobs(
    model_dir: str,
    shared_files_mapping: Dict[FileHandle, str],
    model_download_file_lock_acquire_timeout: int = FILE_LOCK_ACQUIRE_TIMEOUT,
    on_symlink_created: Optional[Callable[[str, str], None]] = None,
    on_symlink_deleted: Optional[Callable[[str], None]] = None,
) -> Dict[str, str]:
    # this function will not override existing files
    os.makedirs(model_dir, exist_ok=True)
    result = {}
    for file_handle, source_path in shared_files_mapping.items():
        link_name = os.path.join(model_dir, file_handle)
        target_path = shared_files_mapping[file_handle]
        result[file_handle] = link_name
        if os.path.exists(link_name) and (
            not os.path.islink(link_name) or os.path.realpath(link_name) == target_path
        ):
            continue
        handle_symlink_creation(
            target_path=target_path,
            link_name=link_name,
            model_download_file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
            on_symlink_created=on_symlink_created,
            on_symlink_deleted=on_symlink_deleted,
        )
    return result


def handle_symlink_creation(
    target_path: str,
    link_name: str,
    model_download_file_lock_acquire_timeout: int = FILE_LOCK_ACQUIRE_TIMEOUT,
    on_symlink_created: Optional[Callable[[str, str], None]] = None,
    on_symlink_deleted: Optional[Callable[[str], None]] = None,
) -> None:
    link_dir, link_file_name = os.path.split(os.path.abspath(link_name))
    os.makedirs(link_dir, exist_ok=True)
    lock_path = os.path.join(link_dir, f".{link_file_name}.lock")
    with FileLock(lock_path, timeout=model_download_file_lock_acquire_timeout):
        if os.path.islink(link_name):
            # file does not exist, but is link = broken symlink - we should purge
            os.remove(link_name)
            if on_symlink_deleted:
                on_symlink_deleted(link_name)
        elif os.path.exists(link_name):
            # regular file exists at link location - do not overwrite
            LOGGER.debug(
                f"Regular file already exists at {link_name}, skipping symlink creation."
            )
            return
        try:
            os.symlink(target_path, link_name)
            if on_symlink_created:
                on_symlink_created(target_path, link_name)
        except FileExistsError:
            # Another process created the file/link between our check and symlink call
            LOGGER.debug(
                f"Symlink target {link_name} was created by another process, skipping."
            )
            return


def dump_model_config_for_offline_use(
    config_path: str,
    model_architecture: Optional[ModelArchitecture],
    task_type: TaskType,
    backend_type: Optional[BackendType],
    file_lock_acquire_timeout: int,
    on_file_created: Optional[Callable[[str], None]] = None,
    model_id: Optional[str] = None,
    model_features: Optional[dict] = None,
    trusted_source: Optional[bool] = None,
    model_dependencies: Optional[List[dict]] = None,
    recommended_parameters: Optional[dict] = None,
    quantization: Optional[str] = None,
    dynamic_batch_size_supported: Optional[bool] = None,
    static_batch_size: Optional[int] = None,
    runtime_compatibility_hash: Optional[str] = None,
    offline_compatibility_hash: Optional[str] = None,
    canonical_model_id: Optional[str] = None,
    package_artifacts: Optional[List[dict]] = None,
    dependency_package_paths: Optional[List[dict]] = None,
) -> str:
    """Persist a versioned manifest used for safe offline package loading."""
    if not isinstance(model_id, str) or not model_id.strip():
        raise CorruptedModelPackageError(
            message="Cannot publish cached model metadata without a cache owner.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    if not isinstance(canonical_model_id, str) or not canonical_model_id.strip():
        raise CorruptedModelPackageError(
            message="Cannot publish cached model metadata without a canonical model identity.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    if package_artifacts is None:
        package_artifacts = []
    package_artifacts = _parse_package_artifact_identities(package_artifacts)
    if dependency_package_paths is None:
        dependency_package_paths = []
    dependency_package_paths = _parse_dependency_package_path_identities(
        dependency_package_paths
    )
    backend_value = (
        backend_type.value if isinstance(backend_type, BackendType) else backend_type
    )
    published_fields = {
        "offline_manifest_version": OFFLINE_CACHE_MANIFEST_VERSION,
        "model_architecture": model_architecture,
        "task_type": task_type,
        "backend_type": backend_value,
        "model_features": model_features,
        "trusted_source": trusted_source,
        "model_dependencies": model_dependencies,
        "recommended_parameters": recommended_parameters,
        "quantization": quantization,
        "dynamic_batch_size_supported": dynamic_batch_size_supported,
        "static_batch_size": static_batch_size,
        "runtime_compatibility_hash": runtime_compatibility_hash,
        "offline_compatibility_hash": offline_compatibility_hash,
        "canonical_model_id": canonical_model_id,
        "package_artifacts": package_artifacts,
        "dependency_package_paths": dependency_package_paths,
        "model_id": model_id,
    }
    target_file_dir, target_file_name = os.path.split(config_path)
    target_file_dir = target_file_dir or "."
    lock_path = os.path.join(target_file_dir, f".{target_file_name}.lock")
    if os.path.islink(target_file_dir):
        raise CorruptedModelPackageError(
            message="Refusing to write model metadata through a symbolic link.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    os.makedirs(target_file_dir, exist_ok=True)
    if (
        os.path.islink(target_file_dir)
        or os.path.islink(config_path)
        or os.path.islink(lock_path)
    ):
        raise CorruptedModelPackageError(
            message="Refusing to write model metadata through a symbolic link.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    with FileLock(lock_path, timeout=file_lock_acquire_timeout):
        created = not os.path.exists(config_path)
        if os.path.exists(config_path):
            try:
                content = read_json(path=config_path)
            except (OSError, ValueError) as error:
                raise CorruptedModelPackageError(
                    message="Refusing to replace malformed cached model metadata.",
                    help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
                ) from error
            if not isinstance(content, dict):
                raise CorruptedModelPackageError(
                    message="Refusing to replace non-dictionary cached model metadata.",
                    help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
                )
        else:
            content = {}
        if "model_id" in content and content["model_id"] != model_id:
            raise CorruptedModelPackageError(
                message=(
                    "Refusing to publish model metadata over a package attributed "
                    f"to a different or invalid cache owner ({content['model_id']})."
                ),
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        if (
            "canonical_model_id" in content
            and content["canonical_model_id"] != canonical_model_id
        ):
            raise CorruptedModelPackageError(
                message=(
                    "Refusing to publish model metadata over a package attributed "
                    "to a different canonical model identity "
                    f"({content['canonical_model_id']})."
                ),
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        if content.get("offline_manifest_version") == OFFLINE_CACHE_MANIFEST_VERSION:
            existing_offline_compatibility_hash = content.get(
                "offline_compatibility_hash"
            )
            if existing_offline_compatibility_hash is not None and (
                not isinstance(existing_offline_compatibility_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", existing_offline_compatibility_hash)
                is None
            ):
                raise CorruptedModelPackageError(
                    message=(
                        "Refusing to reuse a current cached model manifest with "
                        "invalid offline compatibility metadata."
                    ),
                    help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
                )
            if any(
                content.get(field_name) != field_value
                for field_name, field_value in published_fields.items()
                if field_name != "offline_compatibility_hash"
            ):
                raise CorruptedModelPackageError(
                    message=(
                        "Refusing to replace a current cached model manifest "
                        "with different provenance, selection, or materialization metadata."
                    ),
                    help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
                )
            # This hash describes the request that first materialized the
            # immutable package, not the package itself. A provider alias can
            # legitimately resolve to the same canonical package with a
            # different request hash. Keep the original package manifest
            # stable; the alias-specific hash is verified on its separate
            # auto-resolution entry.
            published_fields["offline_compatibility_hash"] = (
                existing_offline_compatibility_hash
            )
            # Every other current-manifest field was just proven identical.
            # Do not replace the file merely to publish another request alias:
            # model_config.json is part of the immutable package, while the
            # alias-specific request hash belongs to its auto-resolution entry.
            return hash_dict_content(content=content)
        content.update(published_fields)
        manifest_content_hash = hash_dict_content(content=content)

        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=target_file_dir,
                prefix=f".{target_file_name}.",
                suffix=".tmp",
                delete=False,
            ) as file_handle:
                temporary_path = file_handle.name
                json.dump(content, file_handle)
                file_handle.flush()
                os.fsync(file_handle.fileno())
            if os.path.islink(config_path):
                raise CorruptedModelPackageError(
                    message="Refusing to replace model metadata through a symbolic link.",
                    help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
                )
            os.replace(temporary_path, config_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass
        if created and on_file_created:
            on_file_created(config_path)
        return manifest_content_hash


def handle_dependencies_directories_creation(
    model_package_cache_dir: str,
    model_dependencies_directories: Dict[str, str],
    model_download_file_lock_acquire_timeout: int = FILE_LOCK_ACQUIRE_TIMEOUT,
    on_symlink_created: Optional[Callable[[str, str], None]] = None,
    on_symlink_deleted: Optional[Callable[[str], None]] = None,
) -> Set[str]:
    resolved_files = set()
    if not model_dependencies_directories:
        return resolved_files
    for dependency_name, dependency_directory in model_dependencies_directories.items():
        _validate_portable_cache_name(
            value=dependency_name,
            kind="dependency name",
        )
        dependency_files = scan_dependency_directory_for_resolved_files(
            dependency_directory=dependency_directory
        )
        resolved_files.update(dependency_files)
        dependencies_sub_dir = os.path.join(
            model_package_cache_dir, MODEL_DEPENDENCIES_SUB_DIR
        )
        if os.path.islink(dependencies_sub_dir):
            raise CorruptedModelPackageError(
                message="Model dependencies directory cannot be a symbolic link.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        target_dependency_dir = os.path.join(dependencies_sub_dir, dependency_name)
        os.makedirs(dependencies_sub_dir, exist_ok=True)
        if os.path.islink(dependencies_sub_dir):
            raise CorruptedModelPackageError(
                message="Model dependencies directory cannot be a symbolic link.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        dependency_lock_path = os.path.join(
            dependencies_sub_dir, f".{dependency_name}.lock"
        )
        if os.path.islink(dependency_lock_path):
            raise CorruptedModelPackageError(
                message="Model dependency lock cannot be a symbolic link.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        with FileLock(
            dependency_lock_path, timeout=model_download_file_lock_acquire_timeout
        ):
            if os.path.islink(target_dependency_dir):
                if os.path.realpath(target_dependency_dir) != os.path.realpath(
                    dependency_directory
                ):
                    os.remove(target_dependency_dir)
                    if on_symlink_deleted:
                        on_symlink_deleted(target_dependency_dir)
            elif os.path.lexists(target_dependency_dir):
                raise CorruptedModelPackageError(
                    message=(
                        f"Dependency path `{target_dependency_dir}` is not a "
                        "symbolic link."
                    ),
                    help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
                )
            if not os.path.exists(target_dependency_dir):
                os.symlink(dependency_directory, target_dependency_dir)
                if on_symlink_created:
                    on_symlink_created(dependency_directory, target_dependency_dir)
            resolved_files.add(target_dependency_dir)
    return resolved_files


def scan_dependency_directory_for_resolved_files(
    dependency_directory: str,
) -> List[str]:
    # we do not follow symlinks here, as the assumption is that we only support one level of nesting
    # for packages, wo when we have dependency - this model must not have dependencies, so
    # we will not encounter directories which are symlinks to be followed.
    results = []
    for current_dir, _, files in os.walk(dependency_directory):
        for file in files:
            if file.startswith(".") and file.endswith(".lock"):
                continue
            full_path = os.path.abspath(os.path.join(current_dir, file))
            results.append(full_path)
            if os.path.islink(full_path):
                results.append(os.readlink(full_path))
    return results


def dump_auto_resolution_cache(
    use_auto_resolution_cache: bool,
    auto_resolution_cache: AutoResolutionCache,
    auto_negotiation_hash: str,
    model_id: str,
    model_package_id: str,
    model_architecture: Optional[ModelArchitecture],
    task_type: TaskType,
    backend_type: Optional[BackendType],
    resolved_files: Set[str],
    model_dependencies: Optional[List[ModelDependency]],
    model_features: Optional[dict],
    recommended_parameters: Optional[RecommendedParameters] = None,
    cache_model_id: Optional[str] = None,
    trusted_source: Optional[bool] = None,
    offline_compatibility_hash: Optional[str] = None,
    canonical_model_id: Optional[str] = None,
    package_manifest_hash: Optional[str] = None,
    api_key: Optional[str] = None,
) -> None:
    if not use_auto_resolution_cache:
        return None
    if not isinstance(model_id, str) or not model_id.strip():
        raise CorruptedModelPackageError(
            message="Cannot cache model resolution without a requested model identity.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    if (
        not isinstance(cache_model_id, str)
        or not cache_model_id.strip()
        or not isinstance(canonical_model_id, str)
        or not canonical_model_id.strip()
    ):
        raise CorruptedModelPackageError(
            message="Cannot cache model resolution without canonical cache attribution.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    if (
        not isinstance(package_manifest_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", package_manifest_hash) is None
    ):
        raise CorruptedModelPackageError(
            message="Cannot cache model resolution without a valid package manifest identity.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    cache_content = AutoResolutionCacheEntry(
        model_id=model_id,
        cache_model_id=cache_model_id,
        canonical_model_id=canonical_model_id,
        cache_attribution_version=CACHE_ATTRIBUTION_VERSION,
        credential_hash=_credential_hash(api_key=api_key),
        model_package_id=model_package_id,
        resolved_files=resolved_files,
        model_architecture=model_architecture,
        task_type=task_type,
        backend_type=backend_type,
        created_at=datetime.now(),
        model_dependencies=model_dependencies,
        model_features=model_features,
        recommended_parameters=recommended_parameters,
        offline_compatibility_hash=offline_compatibility_hash,
        trusted_source=trusted_source,
        package_manifest_hash=package_manifest_hash,
    )
    auto_resolution_cache.register(
        auto_negotiation_hash=auto_negotiation_hash, cache_entry=cache_content
    )


def _resolve_local_cache_package_files(
    model_package_cache_dir: str,
    package_artefacts: List[LocalFileArtefactSpecs],
) -> Dict[str, str]:
    shared_files_mapping: Dict[str, str] = {}
    for artefact in package_artefacts:
        if not isinstance(artefact, LocalFileArtefactSpecs):
            raise CorruptedModelPackageError(
                message=(
                    "Local cache model package contains non-local artefact specs. "
                    "All artefacts must be LocalFileArtefactSpecs."
                ),
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        if not is_valid_md5_hash(artefact.md5_hash):
            raise CorruptedModelPackageError(
                message=(
                    f"Local cache model package artefact `{artefact.file_handle}` has an "
                    f"invalid md5 hash `{artefact.md5_hash}`."
                ),
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        _validate_package_artifact_handle(
            value=artefact.file_handle,
            kind="local cache artefact file handle",
        )
        package_file_path = os.path.join(model_package_cache_dir, artefact.file_handle)
        if os.path.islink(package_file_path) or not os.path.isfile(package_file_path):
            raise CorruptedModelPackageError(
                message=(
                    f"Local cache model package is missing artefact `{artefact.file_handle}` "
                    f"at `{package_file_path}`."
                ),
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        if _md5_file(path=package_file_path) != artefact.md5_hash:
            raise CorruptedModelPackageError(
                message=(
                    f"Local cache model package artefact `{artefact.file_handle}` "
                    "changed after discovery or does not match its MD5 identity."
                ),
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        shared_files_mapping[artefact.file_handle] = package_file_path
    return shared_files_mapping


def attempt_loading_model_from_local_storage(
    model_dir_or_weights_path: str,
    allow_local_code_packages: bool,
    model_init_kwargs: dict,
    model_type: Optional[str] = None,
    task_type: Optional[str] = None,
    backend_type: Optional[
        Union[str, BackendType, List[Union[str, BackendType]]]
    ] = None,
) -> AnyModel:
    if os.path.isfile(model_dir_or_weights_path):
        return attempt_loading_model_from_checkpoint(
            checkpoint_path=model_dir_or_weights_path,
            model_init_kwargs=model_init_kwargs,
            model_type=model_type,
            task_type=task_type,
            backend_type=backend_type,
        )
    config_path = os.path.join(model_dir_or_weights_path, MODEL_CONFIG_FILE_NAME)
    model_config = parse_model_config(config_path=config_path)
    if model_config.is_library_model():
        return load_library_model_from_local_dir(
            model_dir=model_dir_or_weights_path,
            model_config=model_config,
            model_init_kwargs=model_init_kwargs,
        )
    if not allow_local_code_packages:
        raise ForbiddenLocalCodePackageAccessError(
            message=f"Attempted to load model from local package with arbitrary code. This is not allowed in "
            f"this environment. To let inference loading such models, use `allow_local_code_packages=True` "
            f"parameter of `AutoModel.from_pretrained(...)`. If you see this error while using one of Roboflow "
            f"hosted solution - contact us to solve the problem.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#forbiddenlocalcodepackageaccesserror",
        )
    return load_model_from_local_package_with_arbitrary_code(
        model_dir=model_dir_or_weights_path,
        model_config=model_config,
        model_init_kwargs=model_init_kwargs,
    )


def attempt_loading_model_from_checkpoint(
    checkpoint_path: str,
    model_init_kwargs: dict,
    model_type: Optional[str] = None,
    task_type: Optional[str] = None,
    backend_type: Optional[
        Union[str, BackendType, List[Union[str, BackendType]]]
    ] = None,
) -> AnyModel:
    model_architecture, task_type, backend_type = resolve_models_registry_entry(
        model_type=model_type,
        task_type=task_type,
        backend_type=backend_type,
    )
    model_init_kwargs["model_type"] = model_type
    model_class = resolve_model_class(
        model_architecture=model_architecture,
        task_type=task_type,
        backend=backend_type,
    )
    return model_class.from_pretrained(
        checkpoint_path,
        **_prepare_library_model_init_kwargs(
            model_class=model_class,
            model_init_kwargs=model_init_kwargs,
        ),
    )


def resolve_models_registry_entry(
    model_type: Optional[str],
    task_type: Optional[str] = None,
    backend_type: Optional[
        Union[str, BackendType, List[Union[str, BackendType]]]
    ] = None,
) -> Tuple[str, str, BackendType]:
    #  TODO: in the future this check will grow in size
    if not model_type:
        raise MissingModelInitParameterError(
            message="When loading model directly from checkpoint path, `model_type` parameter must be specified. "
            "Use one of the supported value, for example `rfdetr-nano` in case you refer checkpoint of "
            "RFDetr Nano model.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#missingmodelinitparametererror",
        )
    if model_type not in MODEL_TYPES_TO_LOAD_FROM_CHECKPOINT:
        raise InvalidModelInitParameterError(
            message="When loading model directly from checkpoint path, `model_type` parameter must define "
            "one of the type of model that support loading directly from the checkpoints. "
            f"Models supported in current version: {MODEL_TYPES_TO_LOAD_FROM_CHECKPOINT}",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#invalidmodelinitparametererror",
        )
    # a bit of hard coding here, over time we must maintain
    model_architecture = "rfdetr"
    if task_type is None:
        if model_type == "rfdetr-seg-preview":
            task_type = INSTANCE_SEGMENTATION_TASK
        else:
            task_type = OBJECT_DETECTION_TASK
    if task_type not in {OBJECT_DETECTION_TASK, INSTANCE_SEGMENTATION_TASK}:
        raise InvalidModelInitParameterError(
            message=f"When loading model directly from checkpoint path, set `model_type` as {model_type} and "
            f"`task_type` as {task_type}, whereas selected model do only support `{OBJECT_DETECTION_TASK}` "
            f"task while loading from checkpoint file.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#invalidmodelinitparametererror",
        )
    if backend_type is None:
        backend_type = BackendType.TORCH
    if isinstance(backend_type, list) and len(backend_type) != 1:
        if len(backend_type) != 1:
            raise InvalidModelInitParameterError(
                message=f"When loading model directly from checkpoint path, set `backend` parameter to be {backend_type}, "
                f"whereas it is only supported to pass a single value.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#invalidmodelinitparametererror",
            )
        backend_type = backend_type[0]
    if isinstance(backend_type, str):
        backend_type = parse_backend_type(value=backend_type)
    if backend_type is not BackendType.TORCH:
        raise InvalidModelInitParameterError(
            message=f"When loading model directly from checkpoint path, selected the following backend {backend_type}, "
            f"but the backend supported for model {model_type} is {BackendType.TORCH}",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#invalidmodelinitparametererror",
        )
    return model_architecture, task_type, backend_type


def parse_model_config(config_path: str) -> InferenceModelConfig:
    if not os.path.isfile(config_path):
        raise CorruptedModelPackageError(
            message=f"Could not find model config while attempting to load model from "
            f"local directory. This error may be caused by misconfiguration of model package (lack of config "
            f"file), as well as by clash between model_id or model alias and contents of local disc drive which "
            f"is possible when you have local directory in current dir which has the name colliding with the "
            f"model you attempt to load. If your intent was to load model from remote backend (not local "
            f"storage) - verify the contents of $PWD. If you see this problem while using one of Roboflow "
            f"hosted solutions - contact us to get help.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    try:
        raw_config = read_json(path=config_path)
    except ValueError as error:
        raise CorruptedModelPackageError(
            message=f"Could not decode model config while attempting to load model from "
            f"local directory. This error may be caused by corrupted config file. Validate the content of your "
            f"model package and check in documentation the required format of model config file. "
            f"If you see this problem while using one of Roboflow hosted solutions - contact us to get help.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        ) from error
    if not isinstance(raw_config, dict):
        raise CorruptedModelPackageError(
            message=f"While loading the model from local directory encountered corrupted model config file - config is "
            f"supposed to be a dictionary, instead decoded object of type: "
            f"{type(raw_config)}. If you see this problem while using one of Roboflow hosted solutions - "
            f"contact us to get help. Otherwise - verify the content of your model config.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    backend_type = None
    if "backend_type" in raw_config:
        raw_backend_type = raw_config["backend_type"]
        try:
            backend_type = BackendType(raw_backend_type)
        except (TypeError, ValueError) as e:
            raise CorruptedModelPackageError(
                message=f"While loading the model from local directory encountered corrupted model config "
                f"- declared `backend_type` ({raw_backend_type}) is not supported by inference. "
                f"Supported values: {list(t.value for t in BackendType)}. If you see this problem while using "
                f"one of Roboflow hosted solutions - contact us to get help. Otherwise - verify the content "
                f"of your model config.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            ) from e
    model_features = raw_config.get("model_features")
    model_dependencies = raw_config.get("model_dependencies")
    recommended_parameters = raw_config.get("recommended_parameters")
    trusted_source = raw_config.get("trusted_source")
    package_artifacts = raw_config.get("package_artifacts")
    dependency_package_paths = raw_config.get("dependency_package_paths")
    optional_string_fields = (
        "model_architecture",
        "task_type",
        "model_module",
        "model_class",
        "quantization",
        "runtime_compatibility_hash",
        "offline_compatibility_hash",
        "model_id",
        "canonical_model_id",
    )
    for field_name in optional_string_fields:
        field_value = raw_config.get(field_name)
        if field_value is not None and not isinstance(field_value, str):
            raise CorruptedModelPackageError(
                message=f"Cached model config contains invalid {field_name} metadata.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
    for hash_field_name in (
        "runtime_compatibility_hash",
        "offline_compatibility_hash",
    ):
        hash_field_value = raw_config.get(hash_field_name)
        if (
            hash_field_value is not None
            and re.fullmatch(r"[0-9a-f]{64}", hash_field_value) is None
        ):
            raise CorruptedModelPackageError(
                message=(
                    f"Cached model config contains invalid {hash_field_name} "
                    "metadata."
                ),
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
    if model_features is not None and not isinstance(model_features, dict):
        raise CorruptedModelPackageError(
            message="Cached model config contains invalid model_features metadata.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    if model_dependencies is not None and (
        not isinstance(model_dependencies, list)
        or not all(isinstance(item, dict) for item in model_dependencies)
    ):
        raise CorruptedModelPackageError(
            message="Cached model config contains invalid model_dependencies metadata.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    if recommended_parameters is not None and not isinstance(
        recommended_parameters, dict
    ):
        raise CorruptedModelPackageError(
            message="Cached model config contains invalid recommended_parameters metadata.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    if trusted_source is not None and not isinstance(trusted_source, bool):
        raise CorruptedModelPackageError(
            message="Cached model config contains invalid trusted_source metadata.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    if package_artifacts is not None:
        package_artifacts = _parse_package_artifact_identities(value=package_artifacts)
    if dependency_package_paths is not None:
        dependency_package_paths = _parse_dependency_package_path_identities(
            value=dependency_package_paths
        )
    dynamic_batch_size_supported = raw_config.get("dynamic_batch_size_supported")
    if dynamic_batch_size_supported is not None and not isinstance(
        dynamic_batch_size_supported, bool
    ):
        raise CorruptedModelPackageError(
            message=(
                "Cached model config contains invalid "
                "dynamic_batch_size_supported metadata."
            ),
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    static_batch_size = raw_config.get("static_batch_size")
    if static_batch_size is not None and (
        not isinstance(static_batch_size, int)
        or isinstance(static_batch_size, bool)
        or static_batch_size < 1
    ):
        raise CorruptedModelPackageError(
            message="Cached model config contains invalid static_batch_size metadata.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    offline_manifest_version = raw_config.get("offline_manifest_version")
    if offline_manifest_version is not None and (
        not isinstance(offline_manifest_version, int)
        or isinstance(offline_manifest_version, bool)
    ):
        raise CorruptedModelPackageError(
            message=(
                "Cached model config contains invalid offline_manifest_version "
                "metadata."
            ),
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    return InferenceModelConfig(
        model_architecture=raw_config.get("model_architecture"),
        task_type=raw_config.get("task_type"),
        backend_type=backend_type,
        model_module=raw_config.get("model_module"),
        model_class=raw_config.get("model_class"),
        model_features=model_features,
        trusted_source=trusted_source,
        model_dependencies=model_dependencies,
        recommended_parameters=recommended_parameters,
        quantization=raw_config.get("quantization"),
        dynamic_batch_size_supported=dynamic_batch_size_supported,
        static_batch_size=static_batch_size,
        runtime_compatibility_hash=raw_config.get("runtime_compatibility_hash"),
        offline_compatibility_hash=raw_config.get("offline_compatibility_hash"),
        offline_manifest_version=offline_manifest_version,
        model_id=raw_config.get("model_id"),
        canonical_model_id=raw_config.get("canonical_model_id"),
        package_artifacts=package_artifacts,
        dependency_package_paths=dependency_package_paths,
        manifest_content_hash=hash_dict_content(content=raw_config),
    )


def load_library_model_from_local_dir(
    model_dir: str,
    model_config: InferenceModelConfig,
    model_init_kwargs: dict,
) -> AnyModel:
    model_class = resolve_model_class(
        model_architecture=model_config.model_architecture,
        task_type=model_config.task_type,
        backend=model_config.backend_type,
        model_features=(
            set(model_config.model_features) if model_config.model_features else None
        ),
    )
    return model_class.from_pretrained(
        model_dir,
        **_prepare_library_model_init_kwargs(
            model_class=model_class,
            model_init_kwargs=model_init_kwargs,
        ),
    )


def load_model_from_local_package_with_arbitrary_code(
    model_dir: str,
    model_config: InferenceModelConfig,
    model_init_kwargs: dict,
) -> AnyModel:
    if model_config.model_module is None or model_config.model_class is None:
        raise CorruptedModelPackageError(
            message=f"While loading the model from local directory encountered corrupted model config file. "
            f"Config does not specify neither `model_module` name nor `model_class`, which are both  "
            f"required to load models provided with arbitrary code. If you see this problem while using "
            f"one of Roboflow hosted solutions - contact us to get help. Otherwise - verify the content "
            f"of your model config.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    model_module_path = os.path.join(model_dir, model_config.model_module)
    if not os.path.isfile(model_module_path):
        raise CorruptedModelPackageError(
            message=f"While loading the model from local directory encountered corrupted model config file. "
            f"Config pointed module {model_config.model_module}, but there is no file under "
            f"{model_module_path}. If you see this problem while using "
            f"one of Roboflow hosted solutions - contact us to get help. Otherwise - verify the content "
            f"of your model config.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    model_class = load_class_from_path(
        module_path=model_module_path, class_name=model_config.model_class
    )
    return model_class.from_pretrained(model_dir, **model_init_kwargs)


def load_class_from_path(module_path: str, class_name: str) -> AnyModel:
    if not os.path.exists(module_path):
        raise CorruptedModelPackageError(
            message=f"When loading local model with arbitrary code, encountered issue with loading the module. "
            "Could find the module under the path specified in model config. If you see this problem "
            f"while using one of Roboflow hosted solutions - contact us to get help. Otherwise - verify your "
            f"model package checking if you can load the module with model implementation within your "
            f"python environment.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise CorruptedModelPackageError(
            message=f"When loading local model with arbitrary code, encountered issue with loading the module. "
            "Could not build module specification. If you see this problem while using "
            f"one of Roboflow hosted solutions - contact us to get help. Otherwise - verify your "
            f"model package checking if you can load the module with model implementation within your "
            f"python environment.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    if loader is None or not hasattr(loader, "exec_module"):
        raise CorruptedModelPackageError(
            message=f"When loading local model with arbitrary code, encountered issue with loading the module. "
            "Could not execute module loader. If you see this problem while using "
            f"one of Roboflow hosted solutions - contact us to get help. Otherwise - verify your "
            f"model package checking if you can load the module with model implementation within your "
            f"python environment.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    try:
        loader.exec_module(module)
    except Exception as error:
        raise CorruptedModelPackageError(
            message=f"When loading local model with arbitrary code, encountered issue executing the module code "
            f"to retrieve model class. Details of the error: {error}. If you see this problem while using "
            f"one of Roboflow hosted solutions - contact us to get help. Otherwise - verify your "
            f"model package checking if you can load the module with model implementation within your "
            f"python environment.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    if not hasattr(module, class_name):
        raise CorruptedModelPackageError(
            message=f"When loading local model with arbitrary code, encountered issue with loading the module. "
            f"Module `{module_name}` has no class `{class_name}`. If you see this problem while using "
            f"one of Roboflow hosted solutions - contact us to get help. Otherwise - verify your "
            f"model package checking if you can load the module with model implementation within your "
            f"python environment. It may also be the case that configuration file of the model points "
            f"to invalid class name.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    return getattr(module, class_name)


def resolve_recommended_parameters(
    package_level: Optional[RecommendedParameters],
    model_level: Optional[RecommendedParameters],
) -> Optional[RecommendedParameters]:
    """Package-level recommended_parameters take priority over model-level."""
    return package_level if package_level is not None else model_level
