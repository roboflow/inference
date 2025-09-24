import json
import os.path
import re
from typing import List, Optional, Union


def read_text_file(
    path: str,
    split_lines: bool = False,
    strip_white_chars: bool = False,
) -> Union[str, List[str]]:
    with open(path) as f:
        if split_lines:
            lines = list(f.readlines())
            if strip_white_chars:
                return [line.strip() for line in lines if len(line.strip()) > 0]
            else:
                return lines
        content = f.read()
        if strip_white_chars:
            content = content.strip()
        return content


def read_json(path: str, **kwargs) -> Optional[Union[dict, list]]:
    with open(path) as f:
        return json.load(f, **kwargs)


def dump_json(
    path: str, content: Union[dict, list], allow_override: bool = False, **kwargs
) -> None:
    ensure_write_is_allowed(path=path, allow_override=allow_override)
    ensure_parent_dir_exists(path=path)
    with open(path, "w") as f:
        json.dump(content, fp=f, **kwargs)


def dump_text_lines(
    path: str,
    content: List[str],
    allow_override: bool = False,
    lines_connector: str = "\n",
) -> None:
    ensure_write_is_allowed(path=path, allow_override=allow_override)
    ensure_parent_dir_exists(path=path)
    with open(path, "w") as f:
        f.write(lines_connector.join(content))


def dump_bytes(path: str, content: bytes, allow_override: bool = False) -> None:
    ensure_write_is_allowed(path=path, allow_override=allow_override)
    ensure_parent_dir_exists(path=path)
    with open(path, "wb") as f:
        f.write(content)


def ensure_parent_dir_exists(path: str) -> None:
    absolute_path = os.path.abspath(path)
    parent_dir = os.path.dirname(absolute_path)
    os.makedirs(parent_dir, exist_ok=True)


def ensure_write_is_allowed(path: str, allow_override: bool) -> None:
    if os.path.exists(path) and not allow_override:
        raise RuntimeError(f"File {path} exists and override is forbidden.")


def sanitize_path_segment(path_segment: str) -> str:
    # Keep only letters, numbers, underscores and dashes
    return re.sub(r"[^A-Za-z0-9_-]", "_", path_segment)
