"""Local file store of recorded output groups.

One recording is one directory, owned by exactly one writer::

    <directory>/manifest.json            format, status, schema, codecs, counts
    <directory>/groups/000/records.jsonl one record per delivered GroupResult
    <directory>/groups/000/blobs.bin     raw bytes referenced by the records

A record line is ``<json>\\t<crc32 hex>\\n``. One record is one chunk: the
values, statuses, filtered paths and metadata of one pulse of the group,
including wholly filtered pulses. Blob references are
``[offset, length, crc32]``. Layouts are stored once, in the schema.

Lifecycle::

    RecordingWriter.create   destination must be absent or empty; the manifest
                             is published with a no-replace link, status
                             "recording"
    write(result)            encode + append, synchronously, per-group lock
    finish(status)           flush, close, fsync once, replace the manifest
                             with the final status and counts

``open_recording`` accepts only ``complete`` and ``stopped`` recordings,
requires the group directories and files to resolve inside the recording
directory and checks file sizes against the manifest. Iteration is lazy: every
``iter_chunks`` call opens its own file handles, holds one record at a time and
raises ``RecordingCorruptError`` at the first torn, altered or out-of-bounds
record. Records already yielded stay yielded; there is no prefix recovery.
``inspect_recording`` reports any recording's status and counts without
decoding data, so partial recordings stay inspectable.
"""

import datetime
import json
import os
import stat
import threading
import uuid
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from roboflow_workflows.execution_engine.v2.data import (
    Batch,
    EntryLayout,
    EntryMetadata,
    Index,
    validate_entry,
)
from roboflow_workflows.execution_engine.v2.errors import ContractError
from roboflow_workflows.execution_engine.v2.recording.codecs import (
    BlobRef,
    CodecRegistry,
)
from roboflow_workflows.execution_engine.v2.recording.errors import (
    RecordingCorruptError,
    RecordingError,
    RecordingExistsError,
    RecordingIncompleteError,
    RecordingSchemaError,
)
from roboflow_workflows.execution_engine.v2.recording.metadata import (
    decode_entry_metadata,
    encode_entry_metadata,
)
from roboflow_workflows.execution_engine.v2.recording.schema import (
    FINAL_STATUSES,
    FORMAT_VERSION,
    RECORDING_STATUSES,
    REPLAYABLE_STATUSES,
    RecordedGroupSchema,
    RecordingSchema,
)

if TYPE_CHECKING:
    from roboflow_workflows.execution_engine.v2.execution.outputs import GroupResult
    from roboflow_workflows.execution_engine.v2.plan import PulseKey

MANIFEST_NAME = "manifest.json"
RECORDS_NAME = "records.jsonl"
BLOBS_NAME = "blobs.bin"
ENTRY_STATUSES = ("complete", "filtered")

_RECORD_KEYS = {"index", "pulse", "causes", "statuses", "filtered_paths", "entries"}


@dataclass(frozen=True)
class RecordedPulse:
    """Original identity of a recorded pulse (the JSON form of ``PulseKey``).

    Args:
        active_run_id: Active run that produced the pulse.
        source: Source or operator name of the pulse domain.
        sequence: Emission number of that domain in that run. Gaps are kept,
            e.g. frames a source dropped before admission.
    """

    active_run_id: str
    source: str
    sequence: int

    @classmethod
    def from_key(cls, key: "PulseKey") -> "RecordedPulse":
        """Copy a ``PulseKey``.

        Args:
            key: Pulse identity of a ``GroupResult``.

        Returns:
            The recorded identity.
        """
        pulse = cls(
            active_run_id=key.active_run_id, source=key.source, sequence=key.sequence
        )

        return pulse

    def to_json(self) -> List[Any]:
        """Return ``[active_run_id, source, sequence]``."""
        return [self.active_run_id, self.source, self.sequence]

    @classmethod
    def from_json(cls, value: Any) -> "RecordedPulse":
        """Rebuild a pulse from ``to_json`` output.

        Args:
            value: ``[active_run_id, source, sequence]``.

        Returns:
            The recorded identity.

        Raises:
            RecordingCorruptError: When the value is malformed.
        """
        if (
            type(value) is not list
            or len(value) != 3
            or type(value[0]) is not str
            or type(value[1]) is not str
            or type(value[2]) is not int
            or value[2] < 0
        ):
            raise RecordingCorruptError(
                "a recorded pulse must be [run id, domain, sequence]"
            )

        pulse = cls(active_run_id=value[0], source=value[1], sequence=value[2])

        return pulse


@dataclass(frozen=True)
class RecordedChunk:
    """One recorded ``GroupResult``: one pulse of the group's anchor domain.

    ``statuses``, ``filtered_paths``, ``values``, ``layouts`` and ``metadata``
    are keyed by entry key: the field name, or ``"<field>/<port>"`` for one
    port of a wildcard field. ``data`` is the field-keyed view of ``values``;
    read a wildcard port's metadata as ``metadata["<field>/<port>"]``.

    Args:
        group: Output group name.
        index: Position in the group's recording, from 0.
        pulse: Original pulse identity.
        causes: Original upstream pulses of an operator pulse; empty for a
            source pulse.
        statuses: ``complete`` or ``filtered`` per entry key, in schema order.
        filtered_paths: Minimal filtered index paths per entry key.
        values: Decoded value per complete entry key: a payload or a nested
            ``Batch`` with the entry's layout and metadata views attached.
            Tensors are on the CPU.
        layouts: Layout per complete entry key, from the schema.
        metadata: Metadata per complete entry key.
        schema: Schema of the group.
    """

    group: str
    index: int
    pulse: RecordedPulse
    causes: Tuple[RecordedPulse, ...]
    statuses: Mapping[str, str]
    filtered_paths: Mapping[str, Tuple[Index, ...]]
    values: Mapping[str, Any]
    layouts: Mapping[str, EntryLayout]
    metadata: Mapping[str, EntryMetadata]
    schema: RecordedGroupSchema = field(repr=False, compare=False)

    @property
    def is_filtered(self) -> bool:
        """Whether every entry of this chunk is filtered."""
        filtered = all(status == "filtered" for status in self.statuses.values())

        return filtered

    @property
    def data(self) -> Dict[str, Any]:
        """Complete values by field name.

        A wildcard field maps to ``{port: value}`` of its complete ports. A
        field whose entries are all filtered is absent.
        """
        data: Dict[str, Any] = {}
        for entry in self.schema.entries:
            if entry.key not in self.values:
                continue
            if entry.port is None:
                data[entry.field] = self.values[entry.key]
            else:
                data.setdefault(entry.field, {})[entry.port] = self.values[entry.key]

        return data


class RecordingWriter:
    """Single owner of a new recording directory.

    Use ``RecordingWriter.create``. ``write`` may be called from several
    threads; each group has its own lock, so groups never wait for each
    other's disk I/O, and records of one group keep call order.
    """

    def __init__(
        self,
        *,
        directory: Path,
        schema: RecordingSchema,
        codecs: CodecRegistry,
        manifest: Dict[str, Any],
        groups: Dict[str, "_GroupWriter"],
    ):
        self._directory = directory
        self._schema = schema
        self._codecs = codecs
        self._manifest = manifest
        self._groups = groups
        self._finish_lock = threading.Lock()
        self._status = "recording"

    @classmethod
    def create(
        cls,
        directory: Union[str, Path],
        *,
        schema: RecordingSchema,
        codecs: CodecRegistry,
        definition_digest: str,
    ) -> "RecordingWriter":
        """Claim a new recording directory and open its files.

        Args:
            directory: Destination; it must not exist or must be empty.
            schema: Recorded groups.
            codecs: Codecs of the capturing catalogue.
            definition_digest: Digest of the capture definition, kept as
                provenance only.

        Returns:
            A writer in status ``recording``.

        Raises:
            RecordingExistsError: When the destination holds anything, or
                another writer claimed it first.
            RecordingError: When the files cannot be created.
        """
        directory = Path(directory)
        if not isinstance(schema, RecordingSchema):
            raise RecordingSchemaError("RecordingWriter needs a RecordingSchema")
        if not isinstance(codecs, CodecRegistry):
            raise RecordingError("RecordingWriter needs a CodecRegistry")
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise RecordingExistsError(
                "the destination already exists and is not an empty directory; "
                "recordings are never overwritten or appended to",
                directory=directory,
            )

        manifest = {
            "format": FORMAT_VERSION,
            "status": "recording",
            "error": None,
            "created_at": _now(),
            "finished_at": None,
            "definition_digest": definition_digest,
            "codecs": [],
            "groups": [
                {
                    "name": name,
                    "directory": f"groups/{position:03d}",
                    "schema": group.to_json(),
                    "chunks": None,
                    "record_bytes": None,
                    "blob_bytes": None,
                }
                for position, (name, group) in enumerate(schema.groups.items())
            ],
        }
        try:
            directory.mkdir(parents=True, exist_ok=True)
            _publish_new(directory / MANIFEST_NAME, manifest)
        except FileExistsError as error:
            raise RecordingExistsError(
                "another writer claimed this recording directory", directory=directory
            ) from error
        except OSError as error:
            raise RecordingError(
                f"cannot create the manifest: {error}", directory=directory
            ) from error

        groups = {}
        try:
            for item in manifest["groups"]:
                groups[item["name"]] = _GroupWriter.create(
                    directory / item["directory"],
                    schema=schema.groups[item["name"]],
                    codecs=codecs,
                )
        except OSError as error:
            for group_writer in groups.values():
                group_writer.finish()
            raise RecordingError(
                f"cannot create the group files: {error}", directory=directory
            ) from error

        writer = cls(
            directory=directory,
            schema=schema,
            codecs=codecs,
            manifest=manifest,
            groups=groups,
        )

        return writer

    @property
    def directory(self) -> Path:
        """Recording directory."""
        return self._directory

    @property
    def schema(self) -> RecordingSchema:
        """Recorded groups."""
        return self._schema

    @property
    def status(self) -> str:
        """``recording`` until ``finish``, then the final status."""
        return self._status

    @property
    def counters(self) -> Mapping[str, Mapping[str, int]]:
        """Per group: ``chunks``, ``record_bytes`` and ``blob_bytes`` written."""
        counters = MappingProxyType(
            {name: group.counters() for name, group in self._groups.items()}
        )

        return counters

    def write(self, result: "GroupResult") -> None:
        """Encode and append one group result.

        Every byte is copied before this method returns, so later mutation of
        the result's payloads cannot change the record.

        Args:
            result: Delivered group result of a recorded group.

        Raises:
            RecordingError: When the writer is finished or a previous write
                of this group failed on disk.
            RecordingSchemaError: When the result does not match the group's
                schema (unknown group, missing or extra entries, layouts).
            RecordingCodecError: When a payload has no codec or its codec
                fails.
        """
        if self._status != "recording":
            raise RecordingError(
                f"cannot write after finish (status {self._status!r})",
                directory=self._directory,
                group=result.group,
            )

        group_writer = self._groups.get(result.group)
        if group_writer is None:
            raise RecordingSchemaError(
                f"group is not recorded; recorded groups: {list(self._groups)}",
                directory=self._directory,
                group=result.group,
            )

        group_writer.write(result, directory=self._directory)

    def finish(self, status: str, *, error: Optional[str] = None) -> None:
        """Close the files and publish the final manifest.

        Calling ``finish`` again is a no-op. A writer that had a disk failure
        is finished as ``failed`` whatever ``status`` says.

        Args:
            status: ``complete``, ``stopped``, ``failed`` or ``cancelled``.
            error: Failure text stored in the manifest.

        Raises:
            RecordingError: On an unknown status, or when the files or the
                manifest cannot be written; the manifest then says ``failed``
                when it can still be written.
        """
        if status not in FINAL_STATUSES:
            raise RecordingError(
                f"unknown final status {status!r}; expected one of {list(FINAL_STATUSES)}",
                directory=self._directory,
            )

        with self._finish_lock:
            if self._status != "recording":
                return

            close_failures = {}
            for name, group_writer in self._groups.items():
                failure = group_writer.finish()
                if failure is not None:
                    close_failures[name] = failure
            failure_text = "; ".join(
                f"group {name!r}: {failure}" for name, failure in close_failures.items()
            )
            if close_failures and status in REPLAYABLE_STATUSES:
                status = "failed"
                error = failure_text

            used_codecs = set()
            for item in self._manifest["groups"]:
                group_writer = self._groups[item["name"]]
                item.update(group_writer.counters())
                used_codecs.update(group_writer.used_codecs)
            self._manifest.update(
                status=status,
                error=error,
                finished_at=_now(),
                codecs=sorted(used_codecs),
            )
            try:
                _replace(self._directory / MANIFEST_NAME, self._manifest)
            except OSError as manifest_error:
                self._status = "failed"
                detail = f"; files not closed: {failure_text}" if close_failures else ""
                raise RecordingError(
                    f"cannot write the final manifest: {manifest_error}{detail}",
                    directory=self._directory,
                ) from manifest_error

            self._status = status
            if close_failures:
                raise RecordingError(
                    f"recording files could not be closed: {failure_text}",
                    directory=self._directory,
                    group=next(iter(close_failures)),
                )


class _GroupWriter:
    """Records and blobs of one group, guarded by one lock."""

    def __init__(
        self, *, schema: RecordedGroupSchema, codecs: CodecRegistry, records, blobs
    ):
        self.schema = schema
        self.used_codecs: set = set()
        self._codecs = codecs
        self._records = records
        self._blobs = _BlobWriter(blobs)
        self._lock = threading.Lock()
        self._chunks = 0
        self._record_bytes = 0
        self._broken: Optional[str] = None
        self._closed = False
        self._layouts = {entry.key: entry.layout for entry in schema.entries}

    @classmethod
    def create(
        cls, directory: Path, *, schema: RecordedGroupSchema, codecs: CodecRegistry
    ) -> "_GroupWriter":
        directory.mkdir(parents=True)
        records = open(directory / RECORDS_NAME, "xb")
        try:
            blobs = open(directory / BLOBS_NAME, "xb")
        except OSError:
            records.close()
            raise

        group_writer = cls(schema=schema, codecs=codecs, records=records, blobs=blobs)

        return group_writer

    def counters(self) -> Dict[str, int]:
        counters = {
            "chunks": self._chunks,
            "record_bytes": self._record_bytes,
            "blob_bytes": self._blobs.offset,
        }

        return counters

    def write(self, result: "GroupResult", *, directory: Path) -> None:
        self._check_shape(result, directory=directory)
        with self._lock:
            if self._broken is not None or self._closed:
                raise RecordingError(
                    self._broken or "the group is already finished",
                    directory=directory,
                    group=self.schema.name,
                )

            line = self._encode(result)
            try:
                self._records.write(line)
            except OSError as error:
                self._broken = f"a record write failed: {error}"
                raise RecordingError(
                    self._broken,
                    directory=directory,
                    group=self.schema.name,
                    chunk_index=self._chunks,
                ) from error

            self._chunks += 1
            self._record_bytes += len(line)

    def finish(self) -> Optional[str]:
        """Flush, fsync and close both files; every step is attempted.

        Returns:
            The first failure of this group, or ``None``.
        """
        with self._lock:
            if self._closed:
                return self._broken
            self._closed = True
            for file_name, handle in (
                (RECORDS_NAME, self._records),
                (BLOBS_NAME, self._blobs.handle),
            ):
                try:
                    handle.flush()
                    os.fsync(handle.fileno())
                except OSError as error:
                    self._broken = self._broken or f"{file_name} flush failed: {error}"
                try:
                    handle.close()
                except OSError as error:
                    self._broken = self._broken or f"{file_name} close failed: {error}"

        return self._broken

    def _check_shape(self, result: "GroupResult", *, directory: Path) -> None:
        keys = tuple(result.statuses)
        if set(keys) != set(self.schema.keys):
            raise RecordingSchemaError(
                f"result entries {sorted(keys)} differ from the recorded entries "
                f"{sorted(self.schema.keys)}",
                directory=directory,
                group=self.schema.name,
            )

        for key, status in result.statuses.items():
            if status not in ENTRY_STATUSES:
                raise RecordingSchemaError(
                    f"unknown entry status {status!r}",
                    directory=directory,
                    group=self.schema.name,
                    key=key,
                )
            present = key in result.outputs.data
            if present != (status == "complete"):
                raise RecordingSchemaError(
                    f"entry is {status} but its value is "
                    f"{'present' if present else 'absent'}",
                    directory=directory,
                    group=self.schema.name,
                    key=key,
                )
            if present and result.outputs.layout[key] != self._layouts[key]:
                raise RecordingSchemaError(
                    "entry layout differs from the recorded schema",
                    directory=directory,
                    group=self.schema.name,
                    key=key,
                )

    def _encode(self, result: "GroupResult") -> bytes:
        entries = {}
        for key in self.schema.keys:
            if result.statuses[key] != "complete":
                continue
            encoder = self._codecs.encoder(
                self._blobs, location=f"{self.schema.name}.{key}"
            )
            entries[key] = {
                "value": encoder.encode(result.outputs.data[key]),
                "metadata": encode_entry_metadata(
                    result.outputs.metadata[key], encoder=encoder
                ),
            }
            self.used_codecs.update(encoder.used_codecs)

        record = {
            "index": self._chunks,
            "pulse": RecordedPulse.from_key(result.pulse).to_json(),
            "causes": [
                RecordedPulse.from_key(cause).to_json() for cause in result.causes
            ],
            "statuses": {key: result.statuses[key] for key in self.schema.keys},
            "filtered_paths": {
                key: [list(path) for path in result.filtered_paths.get(key, ())]
                for key in self.schema.keys
            },
            "entries": entries,
        }
        text = json.dumps(record, separators=(",", ":"), allow_nan=False).encode(
            "ascii"
        )
        line = text + b"\t" + format(zlib.crc32(text), "08x").encode("ascii") + b"\n"

        return line


class _BlobWriter:
    """Append-only blob file; references carry offset, length and CRC-32."""

    def __init__(self, handle):
        self.handle = handle
        self.offset = 0

    def put(self, data: Union[bytes, bytearray, memoryview]) -> BlobRef:
        view = memoryview(data).cast("B") if not isinstance(data, bytes) else data
        length = len(view)
        checksum = zlib.crc32(view)
        self.handle.write(view)
        ref = [self.offset, length, checksum]
        self.offset += length

        return ref


@dataclass(frozen=True)
class RecordingInfo:
    """Manifest of a recording, read without decoding any data.

    Args:
        directory: Recording directory.
        status: One of ``RECORDING_STATUSES``.
        error: Failure text of a ``failed`` recording.
        created_at: ISO-8601 creation time.
        finished_at: ISO-8601 finalization time; ``None`` while recording.
        definition_digest: Digest of the capture definition (provenance).
        schema: Recorded groups.
        codecs: Names of the codecs the records use.
        counts: Per group: ``chunks``, ``record_bytes`` and ``blob_bytes``;
            ``None`` values until finalized.
    """

    directory: Path
    status: str
    error: Optional[str]
    created_at: str
    finished_at: Optional[str]
    definition_digest: str
    schema: RecordingSchema
    codecs: Tuple[str, ...]
    counts: Mapping[str, Mapping[str, Optional[int]]]
    group_directories: Mapping[str, Path] = field(repr=False)


def inspect_recording(directory: Union[str, Path]) -> RecordingInfo:
    """Read a recording's manifest, whatever its status.

    Args:
        directory: Recording directory.

    Returns:
        Status, schema and counts. No record is read.

    Raises:
        RecordingError: When the directory holds no manifest.
        RecordingSchemaError: On an unsupported format version.
        RecordingCorruptError: When the manifest is malformed.
    """
    directory = Path(directory)
    manifest_path = directory / MANIFEST_NAME
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise RecordingError(
            f"not a recording: {MANIFEST_NAME} is missing", directory=directory
        ) from error
    except OSError as error:
        raise RecordingError(
            f"cannot read {MANIFEST_NAME}: {error}", directory=directory
        ) from error

    try:
        manifest = json.loads(text)
    except ValueError as error:
        raise RecordingCorruptError(
            f"{MANIFEST_NAME} is not valid JSON: {error}", directory=directory
        ) from error

    info = _parse_manifest(manifest, directory=directory)

    return info


def open_recording(
    directory: Union[str, Path], *, codecs: CodecRegistry
) -> "Recording":
    """Open a finalized recording for reading.

    Args:
        directory: Recording directory.
        codecs: Codecs of the reading catalogue.

    Returns:
        The recording. No record is decoded yet.

    Raises:
        RecordingIncompleteError: When the status is not ``complete`` or
            ``stopped``; partial data stay on disk for ``inspect_recording``.
        RecordingCodecError: When a codec the records use is not registered.
        RecordingCorruptError: When a file is missing, is not a regular file,
            resolves outside the recording directory or its size differs from
            the manifest.
        RecordingSchemaError: On an unsupported format version.
    """
    info = inspect_recording(directory)
    if info.status not in REPLAYABLE_STATUSES:
        detail = f" ({info.error})" if info.error else ""
        raise RecordingIncompleteError(
            f"status is {info.status!r}{detail}; only {list(REPLAYABLE_STATUSES)} "
            "recordings can be read. The partial data are kept; "
            "inspect_recording() reports them",
            directory=info.directory,
            status=info.status,
        )

    codecs.require(info.codecs, directory=info.directory)
    root = info.directory.resolve()
    for name, counts in info.counts.items():
        group_directory = _contained(
            info.group_directories[name],
            what="the group directory",
            root=root,
            info=info,
            group=name,
        )
        if not group_directory.is_dir():
            raise RecordingCorruptError(
                "the group directory is not a directory",
                directory=info.directory,
                group=name,
            )

        for file_name, count_name in (
            (RECORDS_NAME, "record_bytes"),
            (BLOBS_NAME, "blob_bytes"),
        ):
            path = _contained(
                group_directory / file_name,
                what=file_name,
                root=root,
                info=info,
                group=name,
            )
            file_stat = path.stat()
            if not stat.S_ISREG(file_stat.st_mode):
                raise RecordingCorruptError(
                    f"{file_name} is not a regular file",
                    directory=info.directory,
                    group=name,
                )
            size = file_stat.st_size
            if size != counts[count_name]:
                raise RecordingCorruptError(
                    f"{file_name} has {size} bytes, the manifest says "
                    f"{counts[count_name]} (truncated or modified)",
                    directory=info.directory,
                    group=name,
                )

    recording = Recording(info, codecs=codecs)

    return recording


def _contained(
    path: Path, *, what: str, root: Path, info: RecordingInfo, group: str
) -> Path:
    """Resolve a group path and require it inside the resolved recording root.

    Symlinks inside the recording are followed; a target outside it is not
    part of the recording format. This validates the format, it is not a
    sandbox against concurrent changes of the directory.
    """
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise RecordingCorruptError(
            f"{what} is missing: {error}", directory=info.directory, group=group
        ) from error
    if not resolved.is_relative_to(root):
        raise RecordingCorruptError(
            f"{what} resolves to {str(resolved)!r}, outside the recording",
            directory=info.directory,
            group=group,
        )

    return resolved


class Recording:
    """Read side of a finalized recording. Holds no open files."""

    def __init__(self, info: RecordingInfo, *, codecs: CodecRegistry):
        self._info = info
        self._codecs = codecs

    @property
    def info(self) -> RecordingInfo:
        """The manifest."""
        return self._info

    @property
    def directory(self) -> Path:
        """Recording directory."""
        return self._info.directory

    @property
    def status(self) -> str:
        """``complete`` or ``stopped``."""
        return self._info.status

    @property
    def schema(self) -> RecordingSchema:
        """Recorded groups."""
        return self._info.schema

    @property
    def definition_digest(self) -> str:
        """Digest of the capture definition; provenance, not compatibility."""
        return self._info.definition_digest

    @property
    def groups(self) -> Tuple[str, ...]:
        """Recorded group names in declaration order."""
        return tuple(self._info.schema.groups)

    def group(self, name: str) -> "RecordedGroup":
        """Return the reader of one recorded group.

        Args:
            name: Group name.

        Returns:
            The group reader.

        Raises:
            RecordingSchemaError: When the group was not recorded.
        """
        schema = self._info.schema.group(name)
        reader = RecordedGroup(
            schema=schema,
            directory=self._info.group_directories[name],
            recording_directory=self._info.directory,
            counts=self._info.counts[name],
            codecs=self._codecs,
        )

        return reader

    def __repr__(self) -> str:
        return (
            f"Recording(directory={str(self.directory)!r}, status={self.status!r}, "
            f"groups={list(self.groups)})"
        )


class RecordedGroup:
    """Lazy, repeatable reader of one recorded group."""

    def __init__(
        self,
        *,
        schema: RecordedGroupSchema,
        directory: Path,
        recording_directory: Path,
        counts: Mapping[str, Optional[int]],
        codecs: CodecRegistry,
    ):
        self._schema = schema
        self._directory = directory
        self._recording_directory = recording_directory
        self._counts = counts
        self._codecs = codecs
        self._layouts = MappingProxyType(
            {entry.key: entry.layout for entry in schema.entries}
        )

    @property
    def name(self) -> str:
        """Group name."""
        return self._schema.name

    @property
    def schema(self) -> RecordedGroupSchema:
        """Recorded group schema."""
        return self._schema

    def __len__(self) -> int:
        return self._counts["chunks"]

    def iter_chunks(self) -> Iterator[RecordedChunk]:
        """Iterate the chunks in recording order.

        Each call opens its own file handles; iterators are independent and
        hold one record at a time. Values are decoded freshly per call.

        Returns:
            Iterator of chunks.

        Raises:
            RecordingCorruptError: At the first torn, altered, out-of-bounds
                or surplus record, or when records are missing at the end.
        """
        iterator = self._iterate()

        return iterator

    def __iter__(self) -> Iterator[RecordedChunk]:
        return self.iter_chunks()

    def _iterate(self) -> Iterator[RecordedChunk]:
        expected = self._counts["chunks"]
        with open(self._directory / RECORDS_NAME, "rb") as records, open(
            self._directory / BLOBS_NAME, "rb"
        ) as blobs:
            blob_reader = _BlobReader(blobs, size=self._counts["blob_bytes"])
            index = 0
            for line in records:
                if index >= expected:
                    raise self._corrupt(
                        f"surplus record beyond the {expected} finalized", index
                    )
                record = self._read_line(line, index=index)
                chunk = self._decode(record, index=index, blobs=blob_reader)
                index += 1
                yield chunk

        if index != expected:
            raise self._corrupt(
                f"has {index} records, the manifest says {expected}", index
            )

    def _read_line(self, line: bytes, *, index: int) -> Dict[str, Any]:
        if not line.endswith(b"\n"):
            raise self._corrupt("torn record (no line end)", index)

        text, separator, checksum = line[:-1].rpartition(b"\t")
        if not separator or checksum != format(zlib.crc32(text), "08x").encode("ascii"):
            raise self._corrupt("record checksum mismatch", index)

        try:
            record = json.loads(text)
        except ValueError as error:
            raise self._corrupt(f"record is not valid JSON: {error}", index) from error
        if type(record) is not dict or set(record) != _RECORD_KEYS:
            raise self._corrupt(
                f"record needs exactly the keys {sorted(_RECORD_KEYS)}", index
            )
        if record["index"] != index:
            raise self._corrupt(f"record says index {record['index']!r}", index)

        return record

    def _decode(
        self, record: Dict[str, Any], *, index: int, blobs: "_BlobReader"
    ) -> RecordedChunk:
        statuses = record["statuses"]
        if (
            type(statuses) is not dict
            or list(statuses) != list(self._schema.keys)
            or not all(status in ENTRY_STATUSES for status in statuses.values())
        ):
            raise self._corrupt("record statuses do not match the schema", index)

        entries = record["entries"]
        complete = [key for key in self._schema.keys if statuses[key] == "complete"]
        if type(entries) is not dict or sorted(entries) != sorted(complete):
            raise self._corrupt("record entries do not match their statuses", index)

        values = {}
        metadata = {}
        for key in complete:
            location = f"{self._schema.name}[{index}].{key}"
            decoder = self._codecs.decoder(blobs, location=location)
            try:
                entry = entries[key]
                if type(entry) is not dict or set(entry) != {"value", "metadata"}:
                    raise RecordingCorruptError("entry needs 'value' and 'metadata'")
                entry_metadata = decode_entry_metadata(
                    entry["metadata"], decoder=decoder
                )
                value = _attach_views(
                    decoder.decode(entry["value"]),
                    layout=self._layouts[key],
                    metadata=entry_metadata,
                )
                validate_entry(
                    value, layout=self._layouts[key], metadata=entry_metadata
                )
            except RecordingCorruptError as error:
                raise self._corrupt(str(error), index, key=key) from error
            except ContractError as error:
                raise self._corrupt(
                    f"invalid entry: {error}", index, key=key
                ) from error
            values[key] = value
            metadata[key] = entry_metadata

        try:
            chunk = RecordedChunk(
                group=self._schema.name,
                index=index,
                pulse=RecordedPulse.from_json(record["pulse"]),
                causes=tuple(
                    RecordedPulse.from_json(cause)
                    for cause in _list(record["causes"], what="causes")
                ),
                statuses=MappingProxyType(dict(statuses)),
                filtered_paths=MappingProxyType(
                    _filtered_paths(record["filtered_paths"], keys=self._schema.keys)
                ),
                values=MappingProxyType(values),
                layouts=MappingProxyType({key: self._layouts[key] for key in complete}),
                metadata=MappingProxyType(metadata),
                schema=self._schema,
            )
        except RecordingCorruptError as error:
            raise self._corrupt(str(error), index) from error

        return chunk

    def _corrupt(
        self, message: str, index: int, *, key: Optional[str] = None
    ) -> RecordingCorruptError:
        error = RecordingCorruptError(
            message,
            directory=self._recording_directory,
            group=self._schema.name,
            key=key,
            chunk_index=index,
        )

        return error

    def __repr__(self) -> str:
        return f"RecordedGroup(name={self.name!r}, chunks={len(self)})"


class _BlobReader:
    """Bounds- and checksum-verified reads from one blob file handle."""

    def __init__(self, handle, *, size: int):
        self._handle = handle
        self._size = size

    def get(self, ref: BlobRef) -> bytearray:
        if (
            type(ref) is not list
            or len(ref) != 3
            or not all(type(part) is int and part >= 0 for part in ref)
        ):
            raise RecordingCorruptError(
                "a blob reference must be [offset, length, crc32]"
            )

        offset, length, checksum = ref
        if offset + length > self._size:
            raise RecordingCorruptError(
                f"blob [{offset}, {offset + length}) is outside the {self._size}-byte blob file"
            )

        data = bytearray(length)
        self._handle.seek(offset)
        if self._handle.readinto(data) != length:
            raise RecordingCorruptError(f"blob at {offset} is truncated")
        if zlib.crc32(data) != checksum:
            raise RecordingCorruptError(f"blob at {offset} fails its checksum")

        return data


def _attach_views(value: Any, *, layout: EntryLayout, metadata: EntryMetadata) -> Any:
    if not layout.depth:
        return value

    attached = _attach_batch(
        value, layout=layout, metadata=metadata, depth=layout.depth
    )

    return attached


def _attach_batch(
    value: Any, *, layout: EntryLayout, metadata: EntryMetadata, depth: int
) -> Batch:
    if type(value) is not Batch:
        raise RecordingCorruptError(
            f"expected a batch for {depth} more axes, got {type(value).__name__}"
        )

    content = value.content
    if depth > 1:
        content = [
            _attach_batch(item, layout=layout, metadata=metadata, depth=depth - 1)
            for item in content
        ]
    batch = Batch(
        content,
        indices=value.indices,
        layout=layout,
        metadata=metadata,
        parent_index=value.parent_index,
    )

    return batch


def _filtered_paths(
    node: Any, *, keys: Tuple[str, ...]
) -> Dict[str, Tuple[Index, ...]]:
    if type(node) is not dict or list(node) != list(keys):
        raise RecordingCorruptError("record filtered_paths do not match the schema")

    paths = {}
    for key, items in node.items():
        paths[key] = tuple(
            tuple(_index_list(item)) for item in _list(items, what="filtered path")
        )

    return paths


def _index_list(value: Any) -> List[int]:
    if type(value) is not list or not all(
        type(component) is int and component >= 0 for component in value
    ):
        raise RecordingCorruptError("an index must be a list of non-negative integers")

    return value


def _list(value: Any, *, what: str) -> List[Any]:
    if type(value) is not list:
        raise RecordingCorruptError(f"record {what} must be a list")

    return value


def _parse_manifest(manifest: Any, *, directory: Path) -> RecordingInfo:
    if type(manifest) is not dict:
        raise RecordingCorruptError(
            f"{MANIFEST_NAME} must hold an object", directory=directory
        )
    if manifest.get("format") != FORMAT_VERSION:
        raise RecordingSchemaError(
            f"unsupported recording format {manifest.get('format')!r}; this reader "
            f"supports {FORMAT_VERSION}",
            directory=directory,
        )

    try:
        status = manifest["status"]
        if status not in RECORDING_STATUSES:
            raise RecordingCorruptError(f"unknown status {status!r}")

        groups = {}
        counts = {}
        group_directories = {}
        for item in manifest["groups"]:
            schema = RecordedGroupSchema.from_json(item["schema"])
            if schema.name != item["name"]:
                raise RecordingCorruptError(
                    f"group {item['name']!r} names another schema"
                )
            relative = Path(item["directory"])
            if relative.is_absolute() or ".." in relative.parts:
                raise RecordingCorruptError(
                    f"group directory {item['directory']!r} escapes"
                )
            group_counts = {
                name: item[name] for name in ("chunks", "record_bytes", "blob_bytes")
            }
            final = status != "recording"
            if final and not all(
                type(count) is int and count >= 0 for count in group_counts.values()
            ):
                raise RecordingCorruptError(
                    f"group {item['name']!r} counts are invalid"
                )
            groups[schema.name] = schema
            counts[schema.name] = MappingProxyType(group_counts)
            group_directories[schema.name] = directory / relative

        codecs = manifest["codecs"]
        if type(codecs) is not list or not all(type(name) is str for name in codecs):
            raise RecordingCorruptError("codecs must be a list of names")

        info = RecordingInfo(
            directory=directory,
            status=status,
            error=manifest["error"],
            created_at=manifest["created_at"],
            finished_at=manifest["finished_at"],
            definition_digest=manifest["definition_digest"],
            schema=RecordingSchema(groups),
            codecs=tuple(codecs),
            counts=MappingProxyType(counts),
            group_directories=MappingProxyType(group_directories),
        )
    except RecordingCorruptError as error:
        raise RecordingCorruptError(
            f"invalid {MANIFEST_NAME}: {error}", directory=directory
        ) from error
    except (KeyError, TypeError, RecordingSchemaError) as error:
        raise RecordingCorruptError(
            f"invalid {MANIFEST_NAME}: {error!r}", directory=directory
        ) from error

    return info


def _publish_new(path: Path, manifest: Dict[str, Any]) -> None:
    """Write a complete file, then link it into place without replacing."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    _write_synced(temporary, manifest)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()
    _sync_directory(path.parent)


def _replace(path: Path, manifest: Dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    _write_synced(temporary, manifest)
    os.replace(temporary, path)
    _sync_directory(path.parent)


def _write_synced(path: Path, manifest: Dict[str, Any]) -> None:
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _now() -> str:
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    return timestamp
