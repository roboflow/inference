import errno
import hashlib
import json
import os.path
import re
import shutil
import stat
import time
from typing import List, Optional, Union

from filelock import FileLock

from inference.core.env import (
    ATOMIC_CACHE_WRITES_ENABLED,
    MODEL_CACHE_DIR,
    OFFLINE_MODE,
)
from inference.core.exceptions import ModelArtefactError
from inference.core.logger import logger
from inference.core.utils.file_system import (
    dump_bytes,
    dump_bytes_atomic,
    dump_json,
    dump_json_atomic,
    dump_text_lines,
    dump_text_lines_atomic,
    path_fits_os_limits,
    read_json,
    read_text_file,
)

MODEL_ID_CACHE_SLUG_PREFIX_LENGTH = 48
MODEL_ID_CACHE_SLUG_HASH_BYTES = 16
LEGACY_MODEL_ID_CACHE_SLUG_HASH_BYTES = 4
MODEL_ID_CACHE_SLUG_NAMESPACE_PREFIX = "~"
SPECIAL_CHAR_ONLY_MODEL_ID_SLUG = "special-char-only-model-id"
_PORTABLE_RAW_MODEL_ID_SEGMENT = re.compile(r"[a-z0-9._ -]+")
_LEGACY_MODEL_ID_CACHE_SLUG = re.compile(r"[A-Za-z0-9_-]{1,48}-[0-9a-f]{8}")
_WINDOWS_RESERVED_PATH_SEGMENTS = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}
RESERVED_CACHE_ROOT_NAMESPACES = {
    "_file_locks",
    "auto-resolution-cache",
    "hf_home",
    "huggingface",
    "lora-bases",
    "models-cache",
    "owl-v2-serialized-data",
    "shared-blobs",
    "usage.db",
    "workflow",
}


def initialise_cache(model_id: Optional[str] = None) -> None:
    if model_id is not None and OFFLINE_MODE:
        # An offline cache is immutable input. This also lets the later
        # required-artifact check produce a useful ModelArtefactError when the
        # current package is absent, instead of leaking a read-only-filesystem
        # error from mkdir.
        return
    cache_dir = get_cache_dir(model_id=model_id)
    os.makedirs(cache_dir, exist_ok=True)


def are_all_files_cached(
    files: List[Union[str, re.Pattern]], model_id: Optional[str] = None
) -> bool:
    return all(is_file_cached(file=file, model_id=model_id) for file in files)


def is_file_cached(
    file: Union[str, re.Pattern], model_id: Optional[str] = None
) -> bool:
    if isinstance(file, re.Pattern):
        return exists_file_matching_regex(file, model_id=model_id)

    cached_file_path = get_cache_file_path_for_read(file=file, model_id=model_id)
    return os.path.isfile(cached_file_path)


def exists_file_matching_regex(
    file: re.Pattern, model_id: Optional[str] = None
) -> bool:
    cache_dir = get_cache_dir_for_read(model_id=model_id)
    for filename in os.listdir(cache_dir):
        if file.match(filename):
            return True
    return False


def load_text_file_from_cache(
    file: str,
    model_id: Optional[str] = None,
    split_lines: bool = False,
    strip_white_chars: bool = False,
) -> Union[str, List[str]]:
    cached_file_path = get_cache_file_path_for_read(file=file, model_id=model_id)
    return read_text_file(
        path=cached_file_path,
        split_lines=split_lines,
        strip_white_chars=strip_white_chars,
    )


def load_json_from_cache(
    file: str, model_id: Optional[str] = None, **kwargs
) -> Optional[Union[dict, list]]:
    cached_file_path = get_cache_file_path_for_read(file=file, model_id=model_id)
    try:
        return read_json(path=cached_file_path, **kwargs)
    except json.JSONDecodeError as e:
        raise ModelArtefactError(f"Error loading JSON from cache: {e}")


def save_bytes_in_cache(
    content: bytes,
    file: str,
    model_id: Optional[str] = None,
    allow_override: bool = True,
) -> None:
    cached_file_path = get_cache_file_path(file=file, model_id=model_id)
    if ATOMIC_CACHE_WRITES_ENABLED:
        dump_bytes_atomic(
            path=cached_file_path, content=content, allow_override=allow_override
        )
    else:
        dump_bytes(
            path=cached_file_path, content=content, allow_override=allow_override
        )


def save_json_in_cache(
    content: Union[dict, list],
    file: str,
    model_id: Optional[str] = None,
    allow_override: bool = True,
    **kwargs,
) -> None:
    cached_file_path = get_cache_file_path(file=file, model_id=model_id)
    if ATOMIC_CACHE_WRITES_ENABLED:
        dump_json_atomic(
            path=cached_file_path,
            content=content,
            allow_override=allow_override,
            **kwargs,
        )
    else:
        dump_json(
            path=cached_file_path,
            content=content,
            allow_override=allow_override,
            **kwargs,
        )


def save_text_lines_in_cache(
    content: List[str],
    file: str,
    model_id: Optional[str] = None,
    allow_override: bool = True,
) -> None:
    cached_file_path = get_cache_file_path(file=file, model_id=model_id)
    if ATOMIC_CACHE_WRITES_ENABLED:
        dump_text_lines_atomic(
            path=cached_file_path, content=content, allow_override=allow_override
        )
    else:
        dump_text_lines(
            path=cached_file_path, content=content, allow_override=allow_override
        )


def get_cache_file_path(file: str, model_id: Optional[str] = None) -> str:
    cache_dir = get_cache_dir(model_id=model_id)
    return os.path.join(cache_dir, file)


def get_cache_file_path_for_read(file: str, model_id: Optional[str] = None) -> str:
    cache_dir = get_cache_dir_for_read(model_id=model_id)
    return os.path.join(cache_dir, file)


def _rmtree_onerror(func, path, exc_info):
    """Error handler for shutil.rmtree."""
    if exc_info[1].errno == errno.ENOTEMPTY:
        try:
            # Try deleting files within the directory first
            for filename in os.listdir(path):
                filepath = os.path.join(path, filename)
                try:
                    if os.path.isfile(filepath) or os.path.islink(filepath):
                        os.unlink(filepath)
                    elif os.path.isdir(filepath):
                        shutil.rmtree(filepath, onerror=_rmtree_onerror)
                except FileNotFoundError:
                    # Another process already removed the file, continue.
                    pass
            # Retry deleting the directory
            os.rmdir(path)
            return  # Success
        except FileNotFoundError:
            # Another process already removed the directory.
            return
        except OSError as e:
            print(f"Error during onerror handling: {e}")
            raise  # re-raise the error.
    else:
        print(f"Error during rmtree: {exc_info[1]}")
        raise  # re-raise the error.


def clear_cache(model_id: Optional[str] = None, delete_from_disk: bool = True) -> None:
    """Clear the cache for a specific model or the entire cache directory.

    Args:
        model_id (Optional[str], optional): The model ID to clear cache for. If None, clears entire cache. Defaults to None.
        delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to False.
    """
    if not delete_from_disk:
        return
    cache_dir = get_cache_dir(model_id=model_id)
    if not os.path.exists(cache_dir):
        return
    lock_dir = MODEL_CACHE_DIR + "/_file_locks"  # Dedicated lock directory
    os.makedirs(lock_dir, exist_ok=True)  # ensure lock directory exists.

    # Use the last 2 levels of the cache directory path as the lock file name suffix
    parts = os.path.normpath(cache_dir).split(os.sep)
    suffix = (
        os.path.join(*parts[-2:]) if len(parts) >= 2 else os.path.basename(cache_dir)
    )
    lock_file = os.path.join(lock_dir, f"{suffix}.lock")

    try:
        lock = FileLock(lock_file, timeout=10)  # 10 second timeout
        with lock:
            if not os.path.exists(cache_dir):  # Check again after acquiring lock
                return  # Already deleted by another process

            max_retries = 3
            retry_delay = 1  # Initial delay in seconds

            for attempt in range(max_retries):
                try:
                    shutil.rmtree(cache_dir, onerror=_rmtree_onerror)
                    return  # Success
                except FileNotFoundError:
                    return  # Already deleted by another process
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Error deleting cache %s: %s, retrying in %s seconds...",
                            cache_dir,
                            e,
                            retry_delay,
                        )
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logger.warning(
                            f"Error deleting cache %s: %s, max retries exceeded.",
                            cache_dir,
                            e,
                        )
                        return
    except Exception as e:
        logger.warning(
            f"Error acquiring lock for cache %s, skipping cache cleanup. %s",
            cache_dir,
            e,
        )


def get_cache_dir(
    model_id: Optional[str] = None, cache_dir_root: Optional[str] = None
) -> str:
    cache_dir_root = cache_dir_root if cache_dir_root is not None else MODEL_CACHE_DIR
    if model_id is not None:
        model_cache_path = get_model_id_cache_path(
            model_id=model_id, cache_dir_root=cache_dir_root
        )
        return os.path.join(cache_dir_root, model_cache_path)
    return cache_dir_root


def get_cache_dir_for_read(
    model_id: Optional[str] = None,
    cache_dir_root: Optional[str] = None,
) -> str:
    """Resolve an exact-owned legacy tree for offline reads only."""

    explicit_cache_dir_root = cache_dir_root
    cache_dir_root = cache_dir_root if cache_dir_root is not None else MODEL_CACHE_DIR
    if explicit_cache_dir_root is None:
        current_cache_dir = get_cache_dir(model_id=model_id)
    else:
        current_cache_dir = get_cache_dir(
            model_id=model_id,
            cache_dir_root=cache_dir_root,
        )
    if model_id is None or not OFFLINE_MODE:
        return current_cache_dir

    if os.path.lexists(current_cache_dir) and not _cache_directory_tree_is_safe(
        cache_dir=current_cache_dir,
        cache_dir_root=cache_dir_root,
    ):
        raise ModelArtefactError(
            f"Refusing unsafe offline cache tree for model {model_id}."
        )

    legacy_cache_path = get_legacy_model_id_cache_path(
        model_id=model_id,
        cache_dir_root=cache_dir_root,
    )
    if legacy_cache_path is None or _cache_directory_has_exact_owner(
        cache_dir=current_cache_dir,
        cache_dir_root=cache_dir_root,
        model_id=model_id,
    ):
        return current_cache_dir
    legacy_cache_dir = os.path.join(cache_dir_root, legacy_cache_path)
    if _cache_directory_has_exact_owner(
        cache_dir=legacy_cache_dir,
        cache_dir_root=cache_dir_root,
        model_id=model_id,
    ):
        return legacy_cache_dir
    return current_cache_dir


def _cache_directory_has_exact_owner(
    cache_dir: str,
    cache_dir_root: str,
    model_id: str,
) -> bool:
    """Validate exact ownership of a safe regular-file cache tree."""

    if not _cache_directory_tree_is_safe(
        cache_dir=cache_dir,
        cache_dir_root=cache_dir_root,
    ):
        return False

    absolute_root = os.path.abspath(cache_dir_root)
    absolute_cache_dir = os.path.abspath(cache_dir)
    if not cache_path_is_within_root(
        path=absolute_cache_dir,
        cache_dir_root=absolute_root,
    ):
        return False
    try:
        relative_cache_dir = os.path.relpath(absolute_cache_dir, absolute_root)
    except ValueError:
        return False
    if relative_cache_dir in {"", os.curdir}:
        return False

    current_path = absolute_root
    is_junction = getattr(os.path, "isjunction", lambda _path: False)
    for path_part in relative_cache_dir.split(os.sep):
        current_path = os.path.join(current_path, path_part)
        if os.path.islink(current_path) or is_junction(current_path):
            return False
    expected_resolved_cache_dir = os.path.normpath(
        os.path.join(os.path.realpath(absolute_root), relative_cache_dir)
    )
    if os.path.normcase(os.path.realpath(absolute_cache_dir)) != os.path.normcase(
        expected_resolved_cache_dir
    ):
        return False

    metadata_path = os.path.join(absolute_cache_dir, "model_type.json")
    try:
        path_status = os.lstat(metadata_path)
    except OSError:
        return False
    if not stat.S_ISREG(path_status.st_mode):
        return False

    descriptor = -1
    try:
        descriptor = os.open(
            metadata_path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        descriptor_status = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_status.st_mode) or (
            path_status.st_dev,
            path_status.st_ino,
        ) != (descriptor_status.st_dev, descriptor_status.st_ino):
            return False
        file_handle = os.fdopen(descriptor, encoding="utf-8")
        descriptor = -1
        with file_handle:
            metadata = json.load(file_handle)
    except (
        json.JSONDecodeError,
        OSError,
        RecursionError,
        TypeError,
        UnicodeError,
        ValueError,
    ):
        return False
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    return isinstance(metadata, dict) and metadata.get("model_id") == model_id


def _cache_directory_tree_is_safe(cache_dir: str, cache_dir_root: str) -> bool:
    """Reject path aliases, symlinks, junctions, and special artifact files."""

    absolute_root = os.path.abspath(cache_dir_root)
    absolute_cache_dir = os.path.abspath(cache_dir)
    if not cache_path_is_within_root(
        path=absolute_cache_dir,
        cache_dir_root=absolute_root,
    ):
        return False
    try:
        relative_cache_dir = os.path.relpath(absolute_cache_dir, absolute_root)
    except ValueError:
        return False
    if relative_cache_dir in {"", os.curdir}:
        return False

    is_junction = getattr(os.path, "isjunction", lambda _path: False)
    current_path = absolute_root
    for path_part in relative_cache_dir.split(os.sep):
        current_path = os.path.join(current_path, path_part)
        if os.path.islink(current_path) or is_junction(current_path):
            return False
    expected_resolved_cache_dir = os.path.normpath(
        os.path.join(os.path.realpath(absolute_root), relative_cache_dir)
    )
    if os.path.normcase(os.path.realpath(absolute_cache_dir)) != os.path.normcase(
        expected_resolved_cache_dir
    ):
        return False

    pending_directories = [absolute_cache_dir]
    while pending_directories:
        directory = pending_directories.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    entry_path = entry.path
                    if entry.is_symlink() or is_junction(entry_path):
                        return False
                    entry_status = entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(entry_status.st_mode):
                        pending_directories.append(entry_path)
                    elif not stat.S_ISREG(entry_status.st_mode):
                        return False
        except OSError:
            return False
    return True


def get_model_id_cache_path(model_id: str, cache_dir_root: str) -> str:
    validate_model_id_for_cache(model_id=model_id)
    legacy_cache_path = os.path.join(cache_dir_root, model_id)
    if _model_id_can_use_raw_cache_path(
        model_id=model_id,
        path=legacy_cache_path,
        cache_dir_root=cache_dir_root,
    ):
        return model_id
    return slugify_model_id_to_cache_key(model_id=model_id)


def get_legacy_model_id_cache_path(model_id: str, cache_dir_root: str) -> Optional[str]:
    """Return the pre-v2 path when the current cache path changed.

    Before V2, every model ID whose raw path fit the host limits used that
    path, including IDs that are case-, Unicode-, or platform-ambiguous. Those
    paths are returned only as a legacy candidate; callers must require exact
    ownership metadata before reading them.
    """

    validate_model_id_for_cache(model_id=model_id)
    raw_cache_path = os.path.join(cache_dir_root, model_id)
    current_cache_path = get_model_id_cache_path(
        model_id=model_id,
        cache_dir_root=cache_dir_root,
    )
    if _raw_cache_path_fits(path=raw_cache_path, cache_dir_root=cache_dir_root):
        return None if current_cache_path == model_id else model_id
    return _slugify_model_id_to_cache_key(
        model_id=model_id,
        digest_size=LEGACY_MODEL_ID_CACHE_SLUG_HASH_BYTES,
        namespace_prefix="",
    )


def validate_model_id_for_cache(model_id: str) -> None:
    if not isinstance(model_id, str):
        raise ValueError("Model ID used for cache access must be a string.")
    path_segments = re.split(r"[\\/]", model_id)
    if any(segment in {"", ".", ".."} for segment in path_segments):
        raise ValueError(
            f"Model ID {model_id!r} contains an unsafe or ambiguous path segment."
        )


def _model_id_can_use_raw_cache_path(
    model_id: str, path: str, cache_dir_root: str
) -> bool:
    path_segments = re.split(r"[\\/]", model_id)
    first_path_segment = path_segments[0]
    return (
        "\\" not in model_id
        and not first_path_segment.startswith(MODEL_ID_CACHE_SLUG_NAMESPACE_PREFIX)
        and first_path_segment not in RESERVED_CACHE_ROOT_NAMESPACES
        and _LEGACY_MODEL_ID_CACHE_SLUG.fullmatch(first_path_segment) is None
        and all(_path_segment_is_portable(segment) for segment in path_segments)
        and _raw_cache_path_fits(path=path, cache_dir_root=cache_dir_root)
    )


def _path_segment_is_portable(path_segment: str) -> bool:
    windows_device_name = path_segment.split(".", maxsplit=1)[0].rstrip(" ").lower()
    return (
        _PORTABLE_RAW_MODEL_ID_SEGMENT.fullmatch(path_segment) is not None
        and not path_segment.endswith((" ", "."))
        and windows_device_name not in _WINDOWS_RESERVED_PATH_SEGMENTS
    )


def _raw_cache_path_fits(path: str, cache_dir_root: str) -> bool:
    return cache_path_is_within_root(
        path=path, cache_dir_root=cache_dir_root
    ) and path_fits_os_limits(path=path)


def cache_path_is_within_root(path: str, cache_dir_root: str) -> bool:
    try:
        root = os.path.abspath(cache_dir_root)
        candidate = os.path.abspath(path)
        return os.path.commonpath([root, candidate]) == root
    except ValueError:
        return False


def slugify_model_id_to_cache_key(model_id: str) -> str:
    return _slugify_model_id_to_cache_key(
        model_id=model_id,
        digest_size=MODEL_ID_CACHE_SLUG_HASH_BYTES,
        namespace_prefix=MODEL_ID_CACHE_SLUG_NAMESPACE_PREFIX,
    )


def _slugify_model_id_to_cache_key(
    model_id: str, digest_size: int, namespace_prefix: str
) -> str:
    model_id_slug = re.sub(r"[^A-Za-z0-9_-]+", "-", model_id)
    model_id_slug = re.sub(r"[_-]{2,}", "-", model_id_slug)
    if not model_id_slug:
        model_id_slug = SPECIAL_CHAR_ONLY_MODEL_ID_SLUG
    if len(model_id_slug) > MODEL_ID_CACHE_SLUG_PREFIX_LENGTH:
        model_id_slug = model_id_slug[:MODEL_ID_CACHE_SLUG_PREFIX_LENGTH]
    digest = hashlib.blake2s(
        model_id.encode("utf-8"), digest_size=digest_size
    ).hexdigest()
    return f"{namespace_prefix}{model_id_slug}-{digest}"
