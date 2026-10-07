"""Trusted Python retrospective stages and the result files they register.

The definition declares the stage; the host passes the callable. The engine
never imports code named by a definition or a recording::

    "retrospective": {"type": "python", "input_group": "analysis",
                      "access": "all" | "chunks",
                      "parameters": {"threshold": 0.8},
                      "result_directory": "$inputs.results_dir"}

    access "all"     function(group, results, parameters)    once
                     group.iter_chunks() is lazy; every call reads anew
    access "chunks"  function(chunk, results, parameters)    once per chunk,
                     in recording order, on the caller's thread

The function writes its own files with its own serialization and registers
them with ``results.register(path)``. The engine creates the result
directory, lists the registered files in ``registered.json`` once the stage
ends (also after a failure) and returns them in ``RetrospectiveOutcome``.
Registered paths are kept in memory, one per call the user makes.
"""

import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Tuple, Union

from roboflow_workflows.execution_engine.v2.recording.errors import (
    RecordingError,
    RetrospectiveError,
)
from roboflow_workflows.execution_engine.v2.recording.store import RecordedGroup

__all__ = [
    "PYTHON_ACCESS_MODES",
    "REGISTERED_FILES_NAME",
    "ResultFiles",
    "RetrospectiveOutcome",
    "run_python_stage",
]

PYTHON_ACCESS_MODES = ("all", "chunks")
REGISTERED_FILES_NAME = "registered.json"


class ResultFiles:
    """The result directory of one Python stage run, and its registered files.

    Args:
        directory: The stage's result directory; it exists.
    """

    def __init__(self, directory: Path):
        self._directory = directory.resolve()
        self._files: Dict[Path, None] = {}  # insertion-ordered set

    @property
    def directory(self) -> Path:
        """Absolute result directory; write result files inside it."""
        return self._directory

    @property
    def files(self) -> Tuple[Path, ...]:
        """Registered files, in registration order."""
        return tuple(self._files)

    def register(self, path: Union[str, Path]) -> Path:
        """Register a result file the stage wrote.

        Args:
            path: The file, absolute or relative to ``directory``.

        Returns:
            Its absolute path. Registering a file twice keeps one entry.

        Raises:
            RetrospectiveError: When the file does not exist, is not a
                regular file, lies outside ``directory`` or is the engine's
                ``registered.json``.
        """
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self._directory / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self._directory):
            raise RetrospectiveError(
                f"result file {str(path)!r} lies outside the result directory "
                f"{str(self._directory)!r}"
            )
        if resolved == self._directory / REGISTERED_FILES_NAME:
            raise RetrospectiveError(
                f"{REGISTERED_FILES_NAME} is written by the engine and cannot be "
                "registered"
            )
        if not resolved.is_file():
            raise RetrospectiveError(
                f"result file {str(path)!r} does not exist or is not a file; write "
                "it before registering it"
            )

        self._files.setdefault(resolved)

        return resolved

    def write_index(self) -> Path:
        """Replace ``registered.json`` with the files registered so far.

        Returns:
            Path of the index.
        """
        index = self._directory / REGISTERED_FILES_NAME
        listed = {
            "files": [
                file.relative_to(self._directory).as_posix() for file in self._files
            ]
        }
        temporary = index.with_name(f".{REGISTERED_FILES_NAME}.tmp")
        temporary.write_text(json.dumps(listed, indent=2), encoding="utf-8")
        os.replace(temporary, index)

        return index


@dataclass(frozen=True)
class RetrospectiveOutcome:
    """What one Python stage run produced.

    Args:
        files: Registered result files, in registration order.
        chunks: Chunks the stage function received (``chunks`` access), or
            the finalized chunk count of the group (``all`` access).
        recording_status: Status of the analysed recording.
        result_directory: The stage's result directory.
    """

    files: Tuple[Path, ...]
    chunks: int
    recording_status: str
    result_directory: Path


def run_python_stage(
    function: Callable[..., Any],
    *,
    access: str,
    group: RecordedGroup,
    recording_status: str,
    parameters: Mapping[str, Any],
    result_directory: Path,
) -> RetrospectiveOutcome:
    """Run a trusted host function over one recorded group.

    Args:
        function: Synchronous callable taking ``(data, results, parameters)``.
        access: ``all`` (one call with the group) or ``chunks`` (one call per
            chunk, in order).
        group: The opened, schema-checked recorded group.
        recording_status: Status of the recording, reported back.
        parameters: Resolved stage parameters.
        result_directory: Directory to create and register results in.

    Returns:
        The registered files and the chunk count.

    Raises:
        RetrospectiveError: When the function raises (with the chunk index in
            ``chunks`` access), returns an awaitable or registers an invalid
            file. Files registered before stay listed.
        RecordingError: When reading the recording fails, e.g. a corrupt
            chunk; it names the group and chunk.
    """
    result_directory.mkdir(parents=True, exist_ok=True)
    results = ResultFiles(result_directory)
    try:
        if access == "all":
            _call(function, group, results=results, parameters=parameters, index=None)
            chunks = len(group)
        else:
            chunks = 0
            for chunk in group.iter_chunks():
                _call(
                    function,
                    chunk,
                    results=results,
                    parameters=parameters,
                    index=chunk.index,
                )
                chunks += 1
    finally:
        results.write_index()

    outcome = RetrospectiveOutcome(
        files=results.files,
        chunks=chunks,
        recording_status=recording_status,
        result_directory=results.directory,
    )

    return outcome


def _call(
    function: Callable[..., Any],
    data: Any,
    *,
    results: ResultFiles,
    parameters: Mapping[str, Any],
    index: Any,
) -> None:
    try:
        returned = function(data, results, parameters)
    except RecordingError:
        # Located errors of the store, as raised.
        raise
    except RetrospectiveError as error:
        if index is None or error.chunk_index is not None:
            raise
        raise RetrospectiveError(str(error), chunk_index=index) from error
    except Exception as error:
        raise RetrospectiveError(
            f"retrospective function raised {type(error).__name__}: {error}",
            chunk_index=index,
        ) from error
    if inspect.isawaitable(returned):
        if inspect.iscoroutine(returned):
            returned.close()
        raise RetrospectiveError(
            "retrospective function returned an awaitable; Python stages call "
            "synchronous functions and never await their results",
            chunk_index=index,
        )
