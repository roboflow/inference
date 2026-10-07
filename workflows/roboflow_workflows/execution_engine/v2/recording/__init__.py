"""Recording of V2 output groups into a local file store, and its read side.

::

    CodecRegistry(catalogue.codecs.values())
    writer = RecordingWriter.create(directory, schema=..., codecs=..., definition_digest=...)
    writer.write(group_result)  ...  writer.finish("complete")

    recording = open_recording(directory, codecs=...)   # complete/stopped only
    for chunk in recording.group("analysis").iter_chunks():   # lazy, repeatable
        chunk.data, chunk.metadata, chunk.pulse

    inspect_recording(directory)   # status and counts of any recording
"""

from roboflow_workflows.execution_engine.v2.recording.codecs import (
    BUILTIN_CODECS,
    BlobSink,
    BlobSource,
    CodecRegistry,
    PayloadCodec,
    PayloadDecoder,
    PayloadEncoder,
    type_name_of,
)
from roboflow_workflows.execution_engine.v2.recording.errors import (
    RecordingCodecError,
    RecordingCorruptError,
    RecordingDefinitionError,
    RecordingError,
    RecordingExistsError,
    RecordingIncompleteError,
    RecordingSchemaError,
    RetrospectiveError,
)
from roboflow_workflows.execution_engine.v2.recording.schema import (
    FINAL_STATUSES,
    FORMAT_VERSION,
    RECORDING_STATUSES,
    REPLAYABLE_STATUSES,
    RecordedEntrySchema,
    RecordedGroupSchema,
    RecordingSchema,
)
from roboflow_workflows.execution_engine.v2.recording.store import (
    RecordedChunk,
    RecordedGroup,
    RecordedPulse,
    Recording,
    RecordingInfo,
    RecordingWriter,
    inspect_recording,
    open_recording,
)

__all__ = [
    "BUILTIN_CODECS",
    "BlobSink",
    "BlobSource",
    "CodecRegistry",
    "FINAL_STATUSES",
    "FORMAT_VERSION",
    "PayloadCodec",
    "PayloadDecoder",
    "PayloadEncoder",
    "RECORDING_STATUSES",
    "REPLAYABLE_STATUSES",
    "RecordedChunk",
    "RecordedEntrySchema",
    "RecordedGroup",
    "RecordedGroupSchema",
    "RecordedPulse",
    "Recording",
    "RecordingCodecError",
    "RecordingCorruptError",
    "RecordingDefinitionError",
    "RecordingError",
    "RecordingExistsError",
    "RecordingIncompleteError",
    "RecordingInfo",
    "RecordingSchema",
    "RecordingSchemaError",
    "RecordingWriter",
    "RetrospectiveError",
    "inspect_recording",
    "open_recording",
    "type_name_of",
]
