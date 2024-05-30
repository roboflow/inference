import os.path
import re
import shutil
from typing import List, Optional, Union

from inference.core.env import MODEL_CACHE_DIR
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


def clear_cache(model_id: Optional[str] = None) -> None:
    cache_dir = get_cache_dir(model_id=model_id)
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)


def get_cache_dir(model_id: Optional[str] = None) -> str:
    if model_id is not None:
        return os.path.join(MODEL_CACHE_DIR, model_id)
    return MODEL_CACHE_DIR
