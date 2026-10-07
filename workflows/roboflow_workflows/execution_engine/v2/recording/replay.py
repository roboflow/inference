"""The engine's replay source: one recorded group as the pulses of a workflow.

A retrospective workflow reads recorded fields as ``$sources.<group>.<field>``.
The compiler adds this source under the group's name; hosts never declare it::

    chunk i of the recording ──▶ pulse i of $sources.<group>
        complete entry           port restored: surviving tree, filtered
                                 positions, original sample/temporal context
        entry filtered inside    port restored with its known filtered nodes
        entry filtered as whole  port restored as filtered as a whole
        every entry filtered     Emission({}): explicitly filtered pulse

A port filtered as a whole is still emitted: the chunk recorded that field.
An omitted port would mean "no event on this port", so an OutputGroup
anchored on that field would skip the chunk and its present siblings.

Restoration is private to the engine (``sources._ReplaySource``): recorded
layouts keep their time axis and their filtered nodes, which an ordinary
source can never emit. Each port keeps the recorded layout, re-scoped to the
replay source, so fields that shared an axis still share it. Replay pulses
are numbered by chunk; the original pulse identity stays in the chunk.
"""

from typing import Any, Dict, Iterator, Optional, Tuple, Type

from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.data import EntryMetadata
from roboflow_workflows.execution_engine.v2.errors import ContractError
from roboflow_workflows.execution_engine.v2.recording.errors import (
    RecordingDefinitionError,
)
from roboflow_workflows.execution_engine.v2.recording.schema import (
    RecordedEntrySchema,
    RecordedGroupSchema,
)
from roboflow_workflows.execution_engine.v2.recording.store import (
    RecordedChunk,
    RecordedGroup,
)
from roboflow_workflows.execution_engine.v2.sources import (
    Emission,
    Source,
    SourceOutput,
    _ReplaySource,
    _RestoredPort,
)

__all__ = [
    "RECORDING_RESOURCE",
    "REPLAY_SOURCE_TYPE",
    "make_replay_source",
    "replay_emission",
]

REPLAY_SOURCE_TYPE = "v2/recorded_group@v1"
"""Type of the replay source; it exists only in a retrospective's catalogue."""

RECORDING_RESOURCE = "workflows_v2_recording"
"""Reserved resource carrying the opened recorded group into a replay session."""


class _RecordedGroupSource(_ReplaySource):
    """Replays the chunks of one recorded group in order, one pulse each."""

    entries: Tuple[RecordedEntrySchema, ...] = ()

    def __init__(self, workflows_v2_recording: RecordedGroup):
        self._group = workflows_v2_recording
        self._chunks: Optional[Iterator[RecordedChunk]] = None

    def open(self) -> None:
        self._chunks = iter(self._group.iter_chunks())

    def read(self) -> Optional[Emission]:
        chunk = next(self._chunks, None)
        if chunk is None:
            return None

        emission = replay_emission(chunk, entries=self.entries)

        return emission

    def close(self) -> None:
        close = getattr(self._chunks, "close", None)
        if close is not None:
            close()
        self._chunks = None


def make_replay_source(
    schema: RecordedGroupSchema, *, catalogue: Catalogue
) -> Type[Source]:
    """Declare the replay source of one recorded group.

    Args:
        schema: Recorded contract of the group.
        catalogue: Catalogue holding the recorded kinds (the primary plan's).

    Returns:
        A replay source class with one port per recorded field, typed by
        the recorded kinds and laid out on the recorded axes.

    Raises:
        RecordingDefinitionError: For a wildcard field, whose heterogeneous
            ports a replay port cannot carry, or a kind or layout the
            source cannot declare.
    """
    wildcard = [entry.field for entry in schema.entries if entry.port is not None]
    if wildcard:
        raise RecordingDefinitionError(
            f"recorded group {schema.name!r} selects wildcard field(s) "
            f"{sorted(set(wildcard))} ($steps.<step>.*); a retrospective workflow "
            "cannot replay them. Select the step outputs one by one, or use a "
            "Python retrospective stage, which receives them as {port: value}"
        )

    try:
        outputs = {
            entry.field: SourceOutput._restored(
                *(catalogue.kind(name) for name in entry.kinds), layout=entry.layout
            )
            for entry in schema.entries
        }
        source_class = type(
            "RecordedGroupSource",
            (_RecordedGroupSource,),
            {
                "__module__": __name__,
                "__doc__": f"Replay of recorded output group {schema.name!r}.",
                "type": REPLAY_SOURCE_TYPE,
                "outputs": outputs,
                "entries": schema.entries,
            },
        )
    except ContractError as error:
        raise RecordingDefinitionError(
            f"recorded group {schema.name!r} cannot be replayed: {error}"
        ) from error

    return source_class


def replay_emission(
    chunk: RecordedChunk, *, entries: Tuple[RecordedEntrySchema, ...]
) -> Emission:
    """The emission restoring one recorded chunk.

    Args:
        chunk: The recorded chunk.
        entries: Recorded entries of the group; none is a wildcard entry.

    Returns:
        One restored port per entry, also per entry filtered as a whole;
        ``Emission({})`` when every entry is filtered as a whole.
    """
    data: Dict[str, Any] = {}
    for entry in entries:
        filtered = tuple(chunk.filtered_paths[entry.key])
        if chunk.statuses[entry.key] == "complete":
            data[entry.field] = _RestoredPort(
                data=chunk.values[entry.key],
                filtered=filtered,
                metadata=chunk.metadata[entry.key],
            )
        else:
            data[entry.field] = _RestoredPort(
                data=None, filtered=filtered, metadata=EntryMetadata()
            )
    if all(port.filtered == ((),) for port in data.values()):
        data = {}
    emission = Emission(data)

    return emission
