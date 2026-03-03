import json
import os
import os.path
import re
import tempfile
from typing import List, Optional, Union

_pattern = re.compile(r"[^A-Za-z0-9_-]")


class AtomicPath:
    """Context manager for atomic file writes.

    Ensures that files are either written completely or not at all,
    preventing partial/corrupted files from power failures or crashes.

    Usage:
        with AtomicPath(target_path, allow_override=False) as temp_path:
            # Write to temp_path
            with open(temp_path, 'w') as f:
                f.write(data)
        # File is atomically moved to target_path on successful exit
    """

    def __init__(self, target_path: str, allow_override: bool = False):
        self.target_path = target_path
        self.allow_override = allow_override
        self.temp_path: Optional[str] = None
        self.temp_file = None

    def __enter__(self) -> str:
        ensure_write_is_allowed(
            path=self.target_path, allow_override=self.allow_override
        )
        ensure_parent_dir_exists(path=self.target_path)

        dir_name = os.path.dirname(os.path.abspath(self.target_path))
        base_name = os.path.basename(self.target_path)
        self.temp_file = tempfile.NamedTemporaryFile(
            dir=dir_name, prefix=".tmp_", suffix="_" + base_name, delete=False
        )
        self.temp_path = self.temp_file.name
        self.temp_file.close()
        return self.temp_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            try:
                if os.name == "nt":  # Windows
                    if os.path.exists(self.target_path):
                        os.remove(self.target_path)
                    os.rename(self.temp_path, self.target_path)
                else:  # POSIX
                    os.replace(self.temp_path, self.target_path)
            except Exception:
                try:
                    os.unlink(self.temp_path)
                except OSError:
                    pass
                raise
        else:
            # Error occurred - clean up temp file
            try:
                os.unlink(self.temp_path)
            except OSError:
                pass
        return False  # Don't suppress exceptions


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
    path: str,
    content: Union[dict, list],
    allow_override: bool = False,
    fsync: bool = False,
    **kwargs,
) -> None:
    ensure_write_is_allowed(path=path, allow_override=allow_override)
    ensure_parent_dir_exists(path=path)
    with open(path, "w") as f:
        json.dump(content, fp=f, **kwargs)
        if fsync:
            os.fsync(f.fileno())


def dump_json_atomic(
    path: str, content: Union[dict, list], allow_override: bool = False, **kwargs
) -> None:
    with AtomicPath(path, allow_override=allow_override) as temp_path:
        dump_json(temp_path, content, allow_override=True, fsync=True, **kwargs)


def dump_text_lines(
    path: str,
    content: List[str],
    allow_override: bool = False,
    lines_connector: str = "\n",
    fsync: bool = False,
) -> None:
    ensure_write_is_allowed(path=path, allow_override=allow_override)
    ensure_parent_dir_exists(path=path)
    with open(path, "w") as f:
        f.write(lines_connector.join(content))
        if fsync:
            os.fsync(f.fileno())


def dump_text_lines_atomic(
    path: str,
    content: List[str],
    allow_override: bool = False,
    lines_connector: str = "\n",
) -> None:
    with AtomicPath(path, allow_override=allow_override) as temp_path:
        dump_text_lines(
            temp_path,
            content,
            allow_override=True,
            lines_connector=lines_connector,
            fsync=True,
        )


def dump_bytes(
    path: str, content: bytes, allow_override: bool = False, fsync: bool = False
) -> None:
    ensure_write_is_allowed(path=path, allow_override=allow_override)
    ensure_parent_dir_exists(path=path)
    with open(path, "wb") as f:
        f.write(content)
        if fsync:
            os.fsync(f.fileno())


def dump_bytes_atomic(path: str, content: bytes, allow_override: bool = False) -> None:
    with AtomicPath(path, allow_override=allow_override) as temp_path:
        dump_bytes(temp_path, content, allow_override=True, fsync=True)


def ensure_parent_dir_exists(path: str) -> None:
    absolute_path = os.path.abspath(path)
    parent_dir = os.path.dirname(absolute_path)
    os.makedirs(parent_dir, exist_ok=True)


def ensure_write_is_allowed(path: str, allow_override: bool) -> None:
    if os.path.exists(path) and not allow_override:
        raise RuntimeError(f"File {path} exists and override is forbidden.")


def sanitize_path_segment(path_segment: str) -> str:
    # Keep only letters, numbers, underscores and dashes
    return _pattern.sub("_", path_segment)
