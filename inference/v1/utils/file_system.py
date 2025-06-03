import json
import os
from typing import Dict, Generator, List, Optional, Union


def index_directory(path: str) -> Dict[str, str]:
    index_base = os.path.abspath(path)
    return {content: os.path.join(index_base, content) for content in os.listdir(path)}


def stream_file_lines(path: str) -> Generator[str, None, None]:
    with open(path, "r") as f:
        for line in f.readlines():
            stripped_line = line.strip()
            if stripped_line:
                yield stripped_line


def read_json(path: str) -> Optional[Union[dict, list]]:
    with open(path) as f:
        return json.load(f)


def pre_allocate_file(path: str, file_size: int) -> None:
    ensure_parent_dir_exists(path=path)
    with open(path, "wb") as f:
        f.truncate(file_size)


def ensure_parent_dir_exists(path: str) -> None:
    parent_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent_dir, exist_ok=True)


def remove_file_if_exists(path: str) -> None:
    if os.path.isfile(path):
        os.remove(path)
