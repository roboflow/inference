"""Supervise a single Databricks invocation in a disposable Python process."""

import json
import os
import stat
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from roboflow_workflows.utils.subprocess_supervisor import run_in_subprocess

_WORKER_MODULE = (
    "roboflow_workflows.core_steps.integrations.databricks.sql_image_source.worker"
)
_MAX_REQUEST_BYTES = 1024 * 1024
_CLEANUP_SECONDS = 35.0


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("Databricks query exceeded timeout_seconds")


def run_isolated(
    request: Dict[str, Any], *, limits: Dict[str, Any], deadline: float
) -> List[Dict[str, Any]]:
    """Run one image-source invocation without modifying parent connector state.

    Args:
        request (Dict[str, Any]): Resolved block inputs, including the credential.
        limits (Dict[str, Any]): Effective limits and destination policy from the parent host.
        deadline (float): Absolute monotonic deadline including launch and output loading.

    Returns:
        The block's ordinary image, name and metadata output rows.

    Raises:
        TimeoutError: The invocation expires, including while loading outputs.
        RuntimeError: The worker fails or returns an invalid result.
        ValueError: Inputs exceed the bounded worker request size.
    """
    from roboflow_workflows.execution_engine.entities.base import (
        ImageParentMetadata,
        WorkflowImageData,
    )

    def _read_result(directory: Path) -> List[Dict[str, Any]]:
        rows = _read_outputs(
            directory, max_rows=request["max_rows"], limits=limits, deadline=deadline
        )
        results = []
        for row in rows:
            _check_deadline(deadline)
            image = WorkflowImageData(
                parent_metadata=ImageParentMetadata(
                    parent_id=f"databricks.{row['image_names']}"
                ),
                numpy_image=row["images"],
            )
            results.append(
                dict(
                    images=image,
                    image_names=row["image_names"],
                    metadata=row["metadata"],
                )
            )

        return results

    try:
        results = run_in_subprocess(
            dict(version=1, request=request, limits=limits, deadline=deadline),
            worker_module=_WORKER_MODULE,
            deadline=deadline,
            read_result=_read_result,
            secrets=(request.get("access_token", ""),),
            max_request_bytes=_MAX_REQUEST_BYTES,
            cleanup_seconds=_CLEANUP_SECONDS,
        )
    except TimeoutError:
        raise TimeoutError("Databricks query exceeded timeout_seconds") from None

    return results


def _read_json(path: Path, *, max_bytes: int) -> Any:
    with _open_regular(path) as stream:
        content = stream.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise RuntimeError("Databricks worker manifest exceeds its limit")

    try:
        result = json.loads(content)
    except (ValueError, RecursionError) as error:
        raise RuntimeError("Invalid Databricks worker manifest") from error

    return result


def _open_regular(path: Path):
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    if path.is_symlink():
        raise RuntimeError("Databricks worker output cannot be a symlink")

    descriptor = os.open(path, flags)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise RuntimeError("Databricks worker output must be a regular file")

    stream = os.fdopen(descriptor, "rb")

    return stream


def _read_outputs(
    directory: Path, *, max_rows: int, limits: Dict[str, Any], deadline: float
) -> List[Dict[str, Any]]:
    _check_deadline(deadline)
    manifest = _read_json(
        directory / "manifest.json",
        max_bytes=min(
            limits["max_run_bytes"] + 1024 * 1024, max_rows * (65536 + 4096) + 1024
        ),
    )
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise RuntimeError("Unsupported Databricks worker output version")

    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) > max_rows:
        raise RuntimeError("Invalid Databricks worker output count")

    result = []
    used = 0
    for index, row in enumerate(rows):
        _check_deadline(deadline)
        if not isinstance(row, dict) or set(row) != {"file", "image_names", "metadata"}:
            raise RuntimeError("Invalid Databricks worker row")

        name, metadata = row["image_names"], row["metadata"]
        if (
            not isinstance(name, str)
            or len(name) > 255
            or not isinstance(metadata, dict)
        ):
            raise RuntimeError("Invalid Databricks worker name or metadata")

        metadata_bytes = len(json.dumps(metadata, ensure_ascii=False).encode("utf-8"))
        if metadata_bytes > 65536 or row["file"] != f"{index}.npy":
            raise RuntimeError("Invalid Databricks worker metadata or image path")

        with _open_regular(directory / f"{index}.npy") as stream:
            version = np.lib.format.read_magic(stream)
            if version != (1, 0):
                raise RuntimeError("Unsupported Databricks image file version")

            shape, fortran, dtype = np.lib.format.read_array_header_1_0(
                stream, max_header_size=1024
            )
            if (
                dtype != np.dtype("uint8")
                or fortran
                or len(shape) != 3
                or shape[2] != 3
                or shape[0] <= 0
                or shape[1] <= 0
                or shape[0] * shape[1] > limits["max_image_pixels"]
            ):
                raise RuntimeError("Invalid Databricks image array")

            nbytes = shape[0] * shape[1] * 3
            if os.fstat(stream.fileno()).st_size != stream.tell() + nbytes:
                raise RuntimeError("Invalid Databricks image file size")

            used += nbytes + len(name) + metadata_bytes
            if used > limits["max_run_bytes"]:
                raise RuntimeError(
                    "Databricks worker outputs exceed the invocation budget"
                )

            _check_deadline(deadline)
            stream.seek(0)
            image = np.load(stream, allow_pickle=False, max_header_size=1024)
        result.append(dict(images=image, image_names=name, metadata=metadata))
    _check_deadline(deadline)

    return result
