"""Read package configuration as data, without importing package Python."""

import ast
import json
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator


@contextmanager
def data_only_config(path: str) -> Iterator[str]:
    with open(path, "rb") as config_file:
        source_bytes = config_file.read(1_000_001)
    if len(source_bytes) > 1_000_000:
        raise ValueError("GroundingDINO configuration exceeds 1 MB")
    source = source_bytes.decode("utf-8")
    values = {}
    for statement in ast.parse(source, filename="model config").body:
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and not statement.targets[0].id.startswith("_")
        ):
            raise ValueError(
                "GroundingDINO config.py must contain only named literal assignments; "
                "imports, executable expressions and inherited configs are unsupported"
            )
        try:
            values[statement.targets[0].id] = ast.literal_eval(statement.value)
        except (ValueError, TypeError) as error:
            raise ValueError(
                "GroundingDINO configuration values must be literals, not executable Python"
            ) from error
    # JSON also avoids SLConfig's Python import path. Disallow _base_ above because
    # SLConfig recursively loads it, including executable files outside this package.
    with TemporaryDirectory(prefix="grounding-dino-config-") as directory:
        safe_path = Path(directory) / "config.json"
        safe_path.write_text(json.dumps(values, allow_nan=False), encoding="utf-8")
        yield str(safe_path)
