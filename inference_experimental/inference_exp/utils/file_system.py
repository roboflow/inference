import json
import os
from typing import Callable, Generator, Optional, Union


def stream_file_lines(path: str) -> Generator[str, None, None]:
    with open(path, "r") as f:
        for line in f.readlines():
            stripped_line = line.strip()
            if stripped_line:
                yield stripped_line


def stream_file_bytes(path: str, chunk_size: int) -> Generator[bytes, None, None]:
    chunk_size = max(chunk_size, 1)
    with open(path, "rb") as f:
        chunk = f.read(chunk_size)
        while chunk:
            yield chunk
            chunk = f.read(chunk_size)


def read_json(path: str) -> Optional[Union[dict, list]]:
    with open(path) as f:
        return json.load(f)


def dump_json(path: str, content: Union[dict, list]) -> None:
    ensure_parent_dir_exists(path=path)
    with open(path, "w") as f:
        json.dump(content, f)


def pre_allocate_file(
    path: str, file_size: int, on_file_created: Optional[Callable[[str], None]] = None
) -> None:
    ensure_parent_dir_exists(path=path)
    with open(path, "wb") as f:
        if on_file_created:
            on_file_created(path)
        f.truncate(file_size)


def ensure_parent_dir_exists(path: str) -> None:
    parent_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent_dir, exist_ok=True)


def remove_file_if_exists(path: str) -> None:
    if os.path.isfile(path):
        os.remove(path)
