"""On-disk cache of saved Workflow definitions in the legacy server's layout.

A definition fetched from the platform is written under
``MODEL_CACHE_DIR/workflow/<workspace>/`` where the legacy server writes it
and is read back from the same places, in the same order, when the platform
cannot be reached.
"""

import hashlib
import hmac
import json
import logging
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any, List, Optional

from inference_server import configuration

logger = logging.getLogger(__name__)

MAX_CACHED_DEFINITION_BYTES = 16 * 1024 * 1024
LOCAL_API_KEY = "local"

_UNSAFE_PATH_CHARACTERS = re.compile(r"[^A-Za-z0-9_-]")
_WINDOWS_RESERVED_PATH_SEGMENTS = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)
_LOWER_HEX_DIGITS = frozenset("0123456789abcdef")
_PUBLIC_IDENTITY_HMAC_MESSAGE = b"inference-workflow-cache-public-identity-v2"
_LEGACY_CANONICAL_SEGMENT = re.compile(r"[a-z0-9-]+")
_CANONICAL_NAMESPACE = ".canonical-v2"
_TENANT_NAMESPACE = ".tenanted-v2"
_VERSIONS_NAMESPACE = "versions"
_INSECURE_LOCATION_MESSAGE = (
    "Detected attempt to save workflow definition in insecure location"
)


def _identity_fingerprint(value: str) -> str:
    identity = value.encode("utf-8", errors="surrogatepass")
    identity_key = len(identity).to_bytes(length=8, byteorder="big") + identity
    fingerprint = hmac.new(
        key=identity_key,
        msg=_PUBLIC_IDENTITY_HMAC_MESSAGE,
        digestmod=hashlib.sha256,
    ).hexdigest()

    return fingerprint


def _tenant_fingerprint(workspace_id: str, api_key: Optional[str]) -> str:
    effective_api_key = api_key if api_key and api_key != LOCAL_API_KEY else ""
    message = json.dumps(
        ["inference-workflow-tenant-v2", workspace_id],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8", errors="surrogatepass")
    fingerprint = hmac.new(
        key=effective_api_key.encode("utf-8", errors="surrogatepass"),
        msg=message,
        digestmod=hashlib.sha256,
    ).hexdigest()

    return fingerprint


def _has_ambiguous_legacy_filename_suffix(value: str) -> bool:
    match_end = len(value) - 1 if value.endswith("\n") else len(value)
    if match_end >= 65:
        fingerprint = value[match_end - 64 : match_end]
        if value[match_end - 65] == "_" and all(
            character in _LOWER_HEX_DIGITS for character in fingerprint
        ):
            return True

    terminal_line_start = value.rfind("\n", 0, match_end) + 1
    marker_index = value.find("_v", terminal_line_start, match_end)

    return marker_index >= 0 and marker_index + 2 < match_end


def _path_segment(value: str, *, reject_legacy_filename_shape: bool = False) -> str:
    sanitized_value = _UNSAFE_PATH_CHARACTERS.sub("_", value)
    if (
        sanitized_value == value
        and sanitized_value
        and len(sanitized_value) <= 96
        and value == value.lower()
        and value.upper() not in _WINDOWS_RESERVED_PATH_SEGMENTS
        and (
            not reject_legacy_filename_shape
            or not _has_ambiguous_legacy_filename_suffix(value)
        )
    ):
        return sanitized_value

    readable_prefix = sanitized_value[:48] or "empty"

    return f"~{readable_prefix}_{_identity_fingerprint(value)}"


def _versioned_stem(workflow_id: str, workflow_version_id: str) -> str:
    identity = json.dumps(
        [workflow_id, workflow_version_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    workflow_segment = _path_segment(workflow_id, reject_legacy_filename_shape=True)

    return f"{workflow_segment}_{_identity_fingerprint(identity)}"


def _cache_root() -> Path:
    return Path(os.path.abspath(Path(configuration.MODEL_CACHE_DIR) / "workflow"))


def _is_inside(candidate: Path, root: Path) -> bool:
    root_prefix = str(root).rstrip(os.sep) + os.sep
    if not str(candidate).startswith(root_prefix):
        return False
    try:
        return os.path.commonpath([root, candidate]) == str(root)
    except ValueError:
        return False


def cache_file_path(
    workspace_id: str,
    workflow_id: str,
    *,
    api_key: Optional[str],
    workflow_version_id: Optional[str],
) -> Path:
    """Return the file a definition is cached at, in the legacy layout.

    Args:
        workspace_id: Workspace the Workflow belongs to.
        workflow_id: Workflow identifier within the workspace.
        api_key: API key of the request; attributes the entry to a tenant
            unless ``SINGLE_TENANT_WORKFLOW_CACHE`` is on.
        workflow_version_id: Pinned version, or None/empty for the latest.

    Returns:
        Absolute path under ``MODEL_CACHE_DIR/workflow``.

    Raises:
        TypeError: If ``workflow_version_id`` is neither a string nor None.
        ValueError: If the path would leave ``MODEL_CACHE_DIR/workflow``.
    """
    workspace_segment = _path_segment(workspace_id)
    if not workflow_version_id:
        namespace_parts: List[str] = []
        cache_stem = _path_segment(workflow_id, reject_legacy_filename_shape=True)
    else:
        if not isinstance(workflow_version_id, str):
            raise TypeError("workflow_version_id must be a string or None")
        namespace_parts = [_VERSIONS_NAMESPACE]
        cache_stem = _versioned_stem(workflow_id, workflow_version_id)

    if configuration.SINGLE_TENANT_WORKFLOW_CACHE:
        namespace = _CANONICAL_NAMESPACE
        filename = f"{cache_stem}.json"
    else:
        namespace = _TENANT_NAMESPACE
        tenant_fingerprint = _tenant_fingerprint(workspace_id, api_key)
        filename = f"{cache_stem}_{tenant_fingerprint}.json"

    cache_root = _cache_root()
    cache_file = Path(
        os.path.abspath(
            cache_root.joinpath(
                workspace_segment, namespace, *namespace_parts, filename
            )
        )
    )
    if not _is_inside(cache_file, cache_root):
        raise ValueError(_INSECURE_LOCATION_MESSAGE)

    return cache_file


def _validated_path(candidate: Path) -> Optional[Path]:
    model_cache_root = os.path.abspath(configuration.MODEL_CACHE_DIR)
    cache_root = _cache_root()
    candidate = Path(os.path.abspath(candidate))
    if not _is_inside(candidate, cache_root):
        return None
    if candidate.suffix != ".json" or cache_root.is_symlink():
        return None
    is_junction = getattr(os.path, "isjunction", lambda _path: False)
    if is_junction(cache_root):
        return None

    resolved_root = os.path.realpath(cache_root)
    expected_resolved_root = os.path.normpath(
        Path(os.path.realpath(model_cache_root)) / "workflow"
    )
    if os.path.normcase(resolved_root) != os.path.normcase(expected_resolved_root):
        return None

    relative_path = candidate.relative_to(cache_root)
    if not relative_path.parts:
        return None
    current_path = cache_root
    for path_part in relative_path.parts:
        current_path = current_path / path_part
        if current_path.is_symlink():
            return None
    expected_resolved_path = os.path.normpath(Path(resolved_root) / relative_path)
    if os.path.realpath(candidate) != expected_resolved_path:
        return None

    return candidate


def _response_is_valid(response: object) -> bool:
    if not isinstance(response, dict):
        return False
    workflow = response.get("workflow")
    if not isinstance(workflow, dict) or not isinstance(workflow.get("config"), str):
        return False
    try:
        workflow_config = json.loads(workflow["config"])
    except (TypeError, ValueError):
        return False

    return isinstance(workflow_config, dict) and isinstance(
        workflow_config.get("specification"), dict
    )


def _read_json_regular_file_no_follow(path: Path) -> Any:
    path_status = os.lstat(path)
    if not stat.S_ISREG(path_status.st_mode):
        raise OSError(f"Refusing to read non-regular JSON file: {path}")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        descriptor_status = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_status.st_mode):
            raise OSError(f"Refusing to read non-regular JSON file: {path}")
        if (path_status.st_dev, path_status.st_ino) != (
            descriptor_status.st_dev,
            descriptor_status.st_ino,
        ):
            raise OSError(f"JSON file changed while it was being opened: {path}")
        if descriptor_status.st_size > MAX_CACHED_DEFINITION_BYTES:
            logger.warning("Ignoring an oversized Workflow cache file")
            raise OSError(f"Workflow cache file exceeds the size limit: {path}")
        file_handle = os.fdopen(descriptor, "rb")
        descriptor = -1
        with file_handle:
            content = file_handle.read(MAX_CACHED_DEFINITION_BYTES + 1)
        if len(content) > MAX_CACHED_DEFINITION_BYTES:
            logger.warning("Ignoring an oversized Workflow cache file")
            raise OSError(f"Workflow cache file exceeds the size limit: {path}")
        return json.loads(content.decode("utf-8"))
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _load_response_file(cache_file: Path) -> Optional[dict]:
    validated_cache_file = _validated_path(cache_file)
    if validated_cache_file is None:
        logger.warning("Refusing to read an unsafe Workflow cache file")
        return None
    try:
        response = _read_json_regular_file_no_follow(validated_cache_file)
        if not _response_is_valid(response):
            raise ValueError("Malformed Workflow cache response")
        return response
    except (OSError, TypeError, ValueError):
        return None


def _write_atomically(cache_file: Path, response: dict) -> None:
    validated_cache_file = _validated_path(cache_file)
    if validated_cache_file is None:
        raise ValueError("Refusing to write an unsafe Workflow cache path")
    cache_directory = validated_cache_file.parent
    cache_directory.mkdir(parents=True, exist_ok=True)
    validated_cache_file = _validated_path(validated_cache_file)
    if validated_cache_file is None:
        raise ValueError("Refusing to write an unsafe Workflow cache path")

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=cache_directory,
            prefix=".workflow.",
            suffix=".tmp",
            delete=False,
        ) as file_handle:
            temporary_path = file_handle.name
            json.dump(response, file_handle)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        validated_cache_file = _validated_path(validated_cache_file)
        if validated_cache_file is None:
            raise ValueError("Refusing to replace an unsafe Workflow cache path")
        os.replace(temporary_path, validated_cache_file)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def store_definition(
    workspace_id: str,
    workflow_id: str,
    *,
    api_key: Optional[str],
    workflow_version_id: Optional[str],
    response: dict,
) -> None:
    """Write a platform response to its cache file; failures only log.

    Args:
        workspace_id: Workspace the Workflow belongs to.
        workflow_id: Workflow identifier within the workspace.
        api_key: API key of the request that fetched the definition.
        workflow_version_id: Pinned version, or None/empty for the latest.
        response: Platform response as returned by the definition endpoint.
    """
    if not _response_is_valid(response):
        logger.warning("Refusing to cache a malformed Workflow response")
        return None
    try:
        cache_file = cache_file_path(
            workspace_id,
            workflow_id,
            api_key=api_key,
            workflow_version_id=workflow_version_id,
        )
        _write_atomically(cache_file, response)
    except (OSError, ValueError) as error:
        logger.warning(
            "Could not write the Workflow definition cache file: %s",
            type(error).__name__,
        )


def _find_offline_hashed_file(
    workspace_id: str,
    workflow_id: str,
    *,
    api_key: Optional[str],
    workflow_version_id: Optional[str],
) -> Optional[Path]:
    if (
        not configuration.LEGACY_OFFLINE_MODE
        or not configuration.SINGLE_TENANT_WORKFLOW_CACHE
    ):
        return None

    cache_file = cache_file_path(
        workspace_id,
        workflow_id,
        api_key=api_key,
        workflow_version_id=workflow_version_id,
    )
    validated_cache_file = _validated_path(cache_file)
    if validated_cache_file is None:
        return None
    cache_file = validated_cache_file

    if workflow_version_id:
        workspace_directory = cache_file.parent.parent.parent
        tenanted_directory = (
            workspace_directory / _TENANT_NAMESPACE / _VERSIONS_NAMESPACE
        )
    else:
        workspace_directory = cache_file.parent.parent
        tenanted_directory = workspace_directory / _TENANT_NAMESPACE
    validated_tenanted_probe = _validated_path(tenanted_directory / cache_file.name)
    if validated_tenanted_probe is None:
        return None
    tenanted_directory = validated_tenanted_probe.parent
    if not tenanted_directory.is_dir():
        return None

    filename_pattern = re.compile(rf"{re.escape(cache_file.stem)}_[0-9a-f]{{64}}\.json")
    candidates = []
    try:
        for candidate in tenanted_directory.iterdir():
            if not filename_pattern.fullmatch(candidate.name):
                continue
            validated_candidate = _validated_path(candidate)
            if validated_candidate is not None and validated_candidate.is_file():
                candidates.append(validated_candidate)
    except OSError:
        return None

    if api_key and api_key != LOCAL_API_KEY:
        tenant_fingerprint = _tenant_fingerprint(workspace_id, api_key)
        exact_filename = f"{cache_file.stem}_{tenant_fingerprint}.json"
        exact_match = next(
            (candidate for candidate in candidates if candidate.name == exact_filename),
            None,
        )
        return exact_match
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        logger.warning(
            "Cannot choose among %d hashed offline Workflow cache entries",
            len(candidates),
        )

    return None


def _case_swapped_sibling(path: Path) -> Optional[Path]:
    for index, character in enumerate(path.name):
        if "a" <= character <= "z":
            return path.with_name(
                path.name[:index] + character.upper() + path.name[index + 1 :]
            )
        if "A" <= character <= "Z":
            return path.with_name(
                path.name[:index] + character.lower() + path.name[index + 1 :]
            )

    return None


def _has_exact_case(path: Path) -> bool:
    try:
        if path.name not in os.listdir(path.parent):
            return False
    except OSError:
        return False
    case_alias = _case_swapped_sibling(path)

    return case_alias is None or not os.path.lexists(case_alias)


def _find_legacy_canonical_file(
    workspace_id: str,
    workflow_id: str,
    *,
    workflow_version_id: Optional[str],
) -> Optional[Path]:
    if (
        not configuration.SINGLE_TENANT_WORKFLOW_CACHE
        or workflow_version_id
        or _LEGACY_CANONICAL_SEGMENT.fullmatch(workspace_id) is None
        or _LEGACY_CANONICAL_SEGMENT.fullmatch(workflow_id) is None
    ):
        return None

    legacy_cache_file = Path(
        os.path.abspath(_cache_root() / workspace_id / f"{workflow_id}.json")
    )
    validated_legacy_cache_file = _validated_path(legacy_cache_file)
    if validated_legacy_cache_file is None:
        return None
    if not _has_exact_case(validated_legacy_cache_file.parent) or not _has_exact_case(
        validated_legacy_cache_file
    ):
        return None

    return validated_legacy_cache_file


def load_definition(
    workspace_id: str,
    workflow_id: str,
    *,
    api_key: Optional[str],
    workflow_version_id: Optional[str],
) -> Optional[dict]:
    """Read a cached platform response, trying the legacy layouts in order.

    Args:
        workspace_id: Workspace the Workflow belongs to.
        workflow_id: Workflow identifier within the workspace.
        api_key: API key of the request.
        workflow_version_id: Pinned version, or None/empty for the latest.

    Returns:
        The cached response, or None when no usable entry exists.
    """
    cache_file = cache_file_path(
        workspace_id,
        workflow_id,
        api_key=api_key,
        workflow_version_id=workflow_version_id,
    )
    validated_cache_file = _validated_path(cache_file)
    if validated_cache_file is None:
        return None
    if os.path.lexists(validated_cache_file):
        cached_response = _load_response_file(validated_cache_file)
        if cached_response is not None:
            return cached_response

    hashed_cache_file = _find_offline_hashed_file(
        workspace_id,
        workflow_id,
        api_key=api_key,
        workflow_version_id=workflow_version_id,
    )
    if hashed_cache_file is not None:
        cached_response = _load_response_file(hashed_cache_file)
        if cached_response is not None:
            return cached_response

    legacy_cache_file = _find_legacy_canonical_file(
        workspace_id,
        workflow_id,
        workflow_version_id=workflow_version_id,
    )
    if legacy_cache_file is None:
        return None
    cached_response = _load_response_file(legacy_cache_file)

    return cached_response
