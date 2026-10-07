"""Error types of V2 output recording and retrospective processing.

Every recording error names the place it belongs to: the recording directory,
the group, the entry key and the chunk index, as far as they are known. A
message never contains payload values.

``RecordingDefinitionError`` is a compile-time error; every other class is an
execution error, because recordings are written and read while a workflow
runs.
"""

from pathlib import Path
from typing import Optional, Union

from roboflow_workflows.execution_engine.v2.errors import (
    WorkflowCompileError,
    WorkflowExecutionError,
)


class RecordingError(WorkflowExecutionError):
    """A recording cannot be written or read.

    Args:
        message: Human-readable explanation.
        directory: Recording directory, when known.
        group: Recorded output group, when known.
        key: Entry key inside the group, when known.
        chunk_index: Chunk index inside the group, when known.
    """

    def __init__(
        self,
        message: str,
        *,
        directory: Optional[Union[str, Path]] = None,
        group: Optional[str] = None,
        key: Optional[str] = None,
        chunk_index: Optional[int] = None,
    ):
        location = _format_location(
            directory=directory, group=group, key=key, chunk_index=chunk_index
        )
        super().__init__(f"{location}: {message}" if location else message)
        self.directory = Path(directory) if directory is not None else None
        self.group = group
        self.key = key
        self.chunk_index = chunk_index


class RecordingCodecError(RecordingError):
    """No codec handles a payload, a codec failed, or a codec name is unknown."""


class RecordingExistsError(RecordingError):
    """The destination already holds a recording or other files."""


class RecordingIncompleteError(RecordingError):
    """The recording was not finalized as ``complete`` or ``stopped``.

    Args:
        message: Human-readable explanation.
        directory: Recording directory.
        status: Status found in the manifest.
    """

    def __init__(self, message: str, *, directory: Union[str, Path], status: str):
        super().__init__(message, directory=directory)
        self.status = status


class RecordingSchemaError(RecordingError):
    """A recorded schema or format version does not match what the reader needs."""


class RecordingCorruptError(RecordingError):
    """Recorded bytes are truncated, altered or structurally invalid."""


class RecordingDefinitionError(WorkflowCompileError):
    """A ``recording`` or ``retrospective`` declaration cannot be compiled."""


class RetrospectiveError(WorkflowExecutionError):
    """A retrospective stage failed.

    Args:
        message: Human-readable explanation.
        chunk_index: Chunk being processed when the stage failed; ``None``
            when the failure does not belong to one chunk.
    """

    def __init__(self, message: str, *, chunk_index: Optional[int] = None):
        location = f"chunk {chunk_index}: " if chunk_index is not None else ""
        super().__init__(f"{location}{message}")
        self.chunk_index = chunk_index


def _format_location(
    *,
    directory: Optional[Union[str, Path]],
    group: Optional[str],
    key: Optional[str],
    chunk_index: Optional[int],
) -> str:
    parts = []
    if directory is not None:
        parts.append(f"recording {str(directory)!r}")
    if group is not None:
        parts.append(f"group {group!r}")
    if chunk_index is not None:
        parts.append(f"chunk {chunk_index}")
    if key is not None:
        parts.append(f"entry {key!r}")

    location = ", ".join(parts)

    return location
