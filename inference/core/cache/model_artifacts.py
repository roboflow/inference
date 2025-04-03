import errno
import os.path
import re
import shutil
import time
from typing import List, Optional, Union

from filelock import FileLock

from inference.core.env import MODEL_CACHE_DIR
from inference.core.logger import logger
from inference.core.utils.file_system import (
    dump_bytes,
    dump_json,
    dump_text_lines,
    read_json,
    read_text_file,
)


def initialise_cache(model_id: Optional[str] = None) -> None:
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

    cached_file_path = get_cache_file_path(file=file, model_id=model_id)
    return os.path.isfile(cached_file_path)


def exists_file_matching_regex(
    file: re.Pattern, model_id: Optional[str] = None
) -> bool:
    cache_dir = get_cache_dir(model_id=model_id)
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
    cached_file_path = get_cache_file_path(file=file, model_id=model_id)
    return read_text_file(
        path=cached_file_path,
        split_lines=split_lines,
        strip_white_chars=strip_white_chars,
    )


def load_json_from_cache(
    file: str, model_id: Optional[str] = None, **kwargs
) -> Optional[Union[dict, list]]:
    cached_file_path = get_cache_file_path(file=file, model_id=model_id)
    return read_json(path=cached_file_path, **kwargs)


def save_bytes_in_cache(
    content: bytes,
    file: str,
    model_id: Optional[str] = None,
    allow_override: bool = True,
) -> None:
    cached_file_path = get_cache_file_path(file=file, model_id=model_id)
    dump_bytes(path=cached_file_path, content=content, allow_override=allow_override)


def save_json_in_cache(
    content: Union[dict, list],
    file: str,
    model_id: Optional[str] = None,
    allow_override: bool = True,
    **kwargs,
) -> None:
    cached_file_path = get_cache_file_path(file=file, model_id=model_id)
    dump_json(
        path=cached_file_path, content=content, allow_override=allow_override, **kwargs
    )


def save_text_lines_in_cache(
    content: List[str],
    file: str,
    model_id: Optional[str] = None,
    allow_override: bool = True,
) -> None:
    cached_file_path = get_cache_file_path(file=file, model_id=model_id)
    dump_text_lines(
        path=cached_file_path, content=content, allow_override=allow_override
    )


def get_cache_file_path(file: str, model_id: Optional[str] = None) -> str:
    cache_dir = get_cache_dir(model_id=model_id)
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
    finally:
        try:
            if os.path.exists(lock_file):
                os.unlink(lock_file)  # Clean up lock file
        except OSError:
            pass  # Best effort cleanup


def get_cache_dir(model_id: Optional[str] = None) -> str:
    if model_id is not None:
        return os.path.join(MODEL_CACHE_DIR, model_id)
    return MODEL_CACHE_DIR
