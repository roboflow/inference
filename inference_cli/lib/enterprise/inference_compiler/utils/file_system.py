import hashlib
import json
from typing import Generator


def read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)  # type: ignore


def dump_json(path: str, content: dict) -> None:
    with open(path, "w") as f:
        json.dump(content, f)


def calculate_local_file_md5(file_path: str) -> str:
    computed_hash = hashlib.md5()
    for file_chunk in stream_file_bytes(path=file_path):
        computed_hash.update(file_chunk)
    return computed_hash.hexdigest()


def stream_file_bytes(
    path: str, chunk_size: int = 16384
) -> Generator[bytes, None, None]:
    chunk_size = max(chunk_size, 1)
    with open(path, "rb") as f:
        chunk = f.read(chunk_size)
        while chunk:
            yield chunk
            chunk = f.read(chunk_size)


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def write_bytes(path: str, content: bytes) -> None:
    with open(path, "wb") as f:
        f.write(content)
