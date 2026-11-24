"""Local file sink block: save text content to files on this machine.

Port of V1 ``roboflow_core/local_file_sink@v1``::

    separate_files  each call ─▶ new <prefix>_<local time>.<file_type>
    append_log      each call ─▶ one entry appended to the current file;
                    a new file after max_entries_per_file entries, or when
                    target_directory, file_name_prefix or file_type change

In ``append_log`` mode a CSV entry after the first loses its header line and
JSON content is rewritten on one line into a ``.jsonl`` file. Files are opened
per write and never kept open, so a dropped block instance leaks no handle.

The host decides access through resources: ``allow_access_to_file_system``
(``False`` by default, so writing needs an explicit opt-in),
``allowed_write_directory`` and ``disable_sinks``.

Differences from V1: containment follows symbolic links (V1 compared
``abspath`` text); a change of target starts a new append file (V1 kept
appending to the first one); new files never overwrite an existing file;
``max_entries_per_file`` accepts an integer selector (V1 declared a string);
files are written as UTF-8 (V1 used the platform's default encoding), and text
that cannot be encoded is a failed write.
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple
from pydantic import Field, StrictInt, StrictStr
from roboflow_workflows.v2.blocks.formatters._shared.values import WallClock
from roboflow_workflows.v2.blocks.sinks.runtime import (
    ERROR_STATUS_OUTPUT,
    MESSAGE_OUTPUT,
    disabled_sink_message,
)
from roboflow_workflows.v2.data.kinds import INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Ref


BLOCK_TYPE = "roboflow_core/local_file_sink@v1"

SAVED_MESSAGE = "Data saved successfully"

FILE_TIMESTAMP_FORMAT = "%Y_%m_%d_%H_%M_%S_%f"

FileType = Literal["csv", "json", "txt"]

AppendTarget = Tuple[Path, str, str]

class FileSystemAccessError(RuntimeError):
    """The host does not allow this sink to write where it was asked to."""

class LocalFileSinkBlock(Block):
    """Save string content (CSV, JSON or text) to local files.

    Example: ``content="$steps.csv_formatter.csv_content"``,
    ``output_mode="append_log"`` and ``target_directory="logs"`` append one
    CSV entry per call to ``logs/workflow_output_<time>.csv``; its header is
    written once.
    """

    type = BLOCK_TYPE
    metadata = {
        "name": "Local File Sink",
        "port_of": "roboflow_core/local_file_sink@v1",
        "section": "data_storage",
    }
    outputs = {"error_status": ERROR_STATUS_OUTPUT, "message": MESSAGE_OUTPUT}

    class Params(BlockParams):
        content: Ref(STRING_KIND) = Field(
            description=(
                "Text to save, e.g. a CSV or JSON document from a formatter. "
                "In append_log mode, CSV content repeats its header line."
            ),
            examples=["$steps.csv_formatter.csv_content"],
        )
        file_type: FileType = Field(
            default="csv",
            description=(
                "File extension and format: csv, json (jsonl in append_log "
                "mode, one JSON document per line) or txt."
            ),
            examples=["csv"],
        )
        output_mode: Literal["append_log", "separate_files"] = Field(
            description=(
                "append_log: append each call to a growing file. "
                "separate_files: one new file per call."
            ),
            examples=["append_log"],
        )
        target_directory: StrictStr | Ref(STRING_KIND) = Field(
            description=(
                "Directory of the files, created when missing. It must lie "
                "inside the host's allowed_write_directory, if one is set."
            ),
            examples=["some/location"],
        )
        file_name_prefix: StrictStr | Ref(STRING_KIND) = Field(
            default="workflow_output",
            description=(
                "File name start; a local timestamp and the extension follow, "
                "e.g. workflow_output_2024_10_18_14_09_57_622297.csv."
            ),
            examples=["my_file"],
        )
        max_entries_per_file: StrictInt | Ref(INTEGER_KIND) = Field(
            default=1024,
            ge=1,
            description=(
                "append_log only: entries per file before a new file starts. "
                "At least 1."
            ),
            examples=[1024],
        )

    def __init__(
        self,
        *,
        allow_access_to_file_system: bool = False,
        allowed_write_directory: Optional[str] = None,
        disable_sinks: bool = False,
        wall_clock: WallClock = datetime.now,
    ) -> None:
        self._allow_access = allow_access_to_file_system
        self._allowed_directory = (
            None
            if allowed_write_directory is None
            else Path(allowed_write_directory).resolve()
        )
        self._disable_sinks = disable_sinks
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        self._append_target: Optional[AppendTarget] = None
        self._append_path: Optional[Path] = None
        self._entries_in_file = 0

    def run(
        self,
        *,
        content: str,
        file_type: FileType,
        output_mode: str,
        target_directory: str,
        file_name_prefix: str,
        max_entries_per_file: int,
    ) -> Dict[str, Any]:
        """Save ``content`` and report the outcome.

        Args:
            content: Text to save.
            file_type: ``csv``, ``json`` or ``txt``.
            output_mode: ``append_log`` or ``separate_files``.
            target_directory: Directory of the files.
            file_name_prefix: Start of new file names.
            max_entries_per_file: Entries per append file.

        Returns:
            ``error_status`` and ``message``. A failed write is reported
            there, as in V1.

        Raises:
            FileSystemAccessError: When the host forbids file access or the
                directory lies outside ``allowed_write_directory``.
        """
        if self._disable_sinks:
            return {
                "error_status": False,
                "message": disabled_sink_message(by_policy=True),
            }

        if not self._allow_access:
            raise FileSystemAccessError(
                f"{BLOCK_TYPE} cannot write files: the host did not allow file "
                "system access (resource allow_access_to_file_system=True)"
            )
        directory = self._checked_directory(target_directory)

        if output_mode == "separate_files":
            result = self._save_separate_file(
                content,
                directory=directory,
                prefix=file_name_prefix,
                extension=file_type,
            )
            return result

        result = self._append(
            content,
            target=(directory, file_name_prefix, file_type),
            max_entries_per_file=max_entries_per_file,
        )

        return result

    def _checked_directory(self, target_directory: str) -> Path:
        directory = Path(target_directory).resolve()
        if self._allowed_directory is not None and not directory.is_relative_to(
            self._allowed_directory
        ):
            raise FileSystemAccessError(
                f"{BLOCK_TYPE} may not write to {target_directory!r}: it is not "
                f"inside the allowed write directory {str(self._allowed_directory)!r}"
            )

        return directory

    def _save_separate_file(
        self, content: str, *, directory: Path, prefix: str, extension: str
    ) -> Dict[str, Any]:
        try:
            path = self._create_new_file(directory, prefix=prefix, extension=extension)
            with path.open("a", encoding="utf-8") as file:
                file.write(with_line_end(content))
        except (OSError, ValueError) as error:
            return {"error_status": True, "message": str(error)}

        return {"error_status": False, "message": SAVED_MESSAGE}

    def _append(
        self, content: str, *, target: AppendTarget, max_entries_per_file: int
    ) -> Dict[str, Any]:
        directory, prefix, file_type = target
        if file_type == "json":
            try:
                content = json.dumps(json.loads(content))
            except ValueError:
                return {"error_status": True, "message": "Invalid JSON content"}

        with self._lock:
            starts_file = (
                self._append_path is None
                or self._append_target != target
                or self._entries_in_file >= max_entries_per_file
            )
            if starts_file:
                extension = "jsonl" if file_type == "json" else file_type
                try:
                    self._append_path = self._create_new_file(
                        directory, prefix=prefix, extension=extension
                    )
                except OSError:
                    self._append_path = None
                    return {
                        "error_status": True,
                        "message": "Could not create new sink file",
                    }
                self._append_target = target
                self._entries_in_file = 0
            elif file_type == "csv":
                content = without_header(content)

            try:
                with self._append_path.open("a", encoding="utf-8") as file:
                    file.write(with_line_end(content))
            except (OSError, ValueError):
                return {
                    "error_status": True,
                    "message": "Could not append content to append log",
                }
            self._entries_in_file += 1

        return {"error_status": False, "message": SAVED_MESSAGE}

    def _create_new_file(self, directory: Path, *, prefix: str, extension: str) -> Path:
        local_now = self._wall_clock(timezone.utc).astimezone()
        name = f"{prefix}_{local_now.strftime(FILE_TIMESTAMP_FORMAT)}"
        # The prefix may name a subdirectory, so check the file's own parent.
        parent = self._checked_directory(str((directory / name).parent))
        parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation: two files in one microsecond get a numeric
        # suffix instead of overwriting each other or an existing file.
        path = directory / f"{name}.{extension}"
        attempt = 0
        while True:
            try:
                path.open("x").close()
            except FileExistsError:
                attempt += 1
                path = directory / f"{name}_{attempt}.{extension}"
                continue

            return path

def with_line_end(content: str) -> str:
    """Return ``content`` ending with ``\\n``, as V1 wrote every entry.

    Args:
        content: Text to write.

    Returns:
        The text with a trailing newline.
    """
    if content.endswith("\n"):
        return content

    terminated = f"{content}\n"

    return terminated

def without_header(content: str) -> str:
    """Drop the first line of a CSV document, as V1 did on append.

    Args:
        content: CSV text with a header line.

    Returns:
        The text after the first line.
    """
    body = "\n".join(content.split("\n")[1:])

    return body
