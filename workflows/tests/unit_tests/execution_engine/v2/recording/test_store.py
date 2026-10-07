"""The recording file store: lifecycle, exactness, integrity and bounded reading."""

import dataclasses
import json
import threading
import tracemalloc
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import torch
from roboflow_workflows.execution_engine.v2.blocks import create_catalogue
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.data import (
    Axis,
    Batch,
    EntryLayout,
    EntryMetadata,
    SampleContext,
    TemporalContext,
    Timestamp,
    WorkflowsBuffer,
)
from roboflow_workflows.execution_engine.v2.execution.outputs import GroupResult
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND
from roboflow_workflows.execution_engine.v2.plan import PulseKey
from roboflow_workflows.execution_engine.v2.recording import (
    CodecRegistry,
    RecordedEntrySchema,
    RecordedGroupSchema,
    RecordedPulse,
    RecordingCodecError,
    RecordingCorruptError,
    RecordingError,
    RecordingExistsError,
    RecordingIncompleteError,
    RecordingSchema,
    RecordingSchemaError,
    RecordingWriter,
    inspect_recording,
    open_recording,
)
from roboflow_workflows.execution_engine.v2.sources import (
    Emission,
    Source,
    SourceOutput,
)

CODECS = CodecRegistry(create_catalogue().codecs.values())
CROPS = EntryLayout((Axis(id="steps.crop:crops", kind="dynamic_nesting"),))
GRID = EntryLayout(
    (
        Axis(id="steps.grid:rows", kind="static_nesting"),
        Axis(id="steps.grid:cells", kind="dynamic_nesting"),
    )
)
UNGROUPED = EntryLayout()
LAYOUTS = {
    "frame": UNGROUPED,
    "crops": CROPS,
    "grid": GRID,
    "stats/count": UNGROUPED,
    "stats/mean": UNGROUPED,
}
ANALYSIS = RecordedGroupSchema(
    name="analysis",
    anchor_domain="video",
    entries=(
        RecordedEntrySchema("frame", "frame", None, ("image",), UNGROUPED),
        RecordedEntrySchema("crops", "crops", None, ("image",), CROPS),
        RecordedEntrySchema("grid", "grid", None, ("*",), GRID),
        RecordedEntrySchema("stats/count", "stats", "count", ("integer",), UNGROUPED),
        RecordedEntrySchema("stats/mean", "stats", "mean", ("float",), UNGROUPED),
    ),
)
EVENTS = RecordedGroupSchema(
    name="events",
    anchor_domain="video",
    entries=(RecordedEntrySchema("value", "value", None, ("*",), UNGROUPED),),
)
SCHEMA = RecordingSchema([ANALYSIS, EVENTS])
CAMERA = SampleContext(
    source_id="camera-1", source_type="video", source_metadata={"fps": 30}
)


def media(ticks: int) -> TemporalContext:
    return TemporalContext(
        observed_coverage=Timestamp(ticks, Fraction(1, 1000), "engine"),
        media_coverage=Timestamp(ticks * 1001, Fraction(1, 30000), "media"),
    )


def image(seed: int, *, size=(3, 8, 10)) -> ImageData:
    pixels = torch.full(size, seed % 256, dtype=torch.uint8)
    return ImageData.from_tensor(pixels, image_id=f"frame-{seed}")


def result(
    group: str,
    sequence: int,
    values: Dict[str, Any],
    *,
    keys: Optional[List[str]] = None,
    filtered_paths: Optional[Dict[str, list]] = None,
    metadata: Optional[Dict[str, EntryMetadata]] = None,
    causes: tuple = (),
    layouts: Dict[str, EntryLayout] = LAYOUTS,
) -> GroupResult:
    keys = keys if keys is not None else list(SCHEMA.group(group).keys)
    metadata = metadata or {}
    outputs = WorkflowsBuffer(
        lineage_id="video",
        pulse_id=sequence,
        data=values,
        layout={key: layouts.get(key, UNGROUPED) for key in values},
        metadata={key: metadata.get(key, EntryMetadata()) for key in values},
    )
    group_result = GroupResult(
        group=group,
        source="video",
        pulse=PulseKey(active_run_id="run-1", source="video", sequence=sequence),
        run_id=f"pulse-{sequence}",
        session_id="session-1",
        fields=(),
        outputs=outputs,
        selections={},
        statuses={key: "complete" if key in values else "filtered" for key in keys},
        filtered_paths={
            key: tuple((filtered_paths or {}).get(key, ())) for key in keys
        },
        plan=None,
        causes=causes,
    )
    return group_result


def complete_values(seed: int) -> Dict[str, Any]:
    return {
        "frame": image(seed),
        "crops": Batch([image(seed + 1), image(seed + 2)], indices=[(0,), (1,)]),
        "grid": Batch(
            [Batch([1.5, None], indices=[(0, 0), (0, 1)], parent_index=(0,))],
            indices=[(0,)],
        ),
        "stats/count": 2,
        "stats/mean": np.float32(0.5),
    }


def sparse_values() -> Dict[str, Any]:
    """Crops filtered at index 1, grid row 0 gaps and an empty row 2."""
    return {
        "frame": image(9),
        "crops": Batch([image(10)], indices=[(0,)]),
        "grid": Batch(
            [
                Batch(["a", "c"], indices=[(0, 0), (0, 2)], parent_index=(0,)),
                Batch.empty(parent_index=(2,)),
            ],
            indices=[(0,), (2,)],
        ),
        "stats/count": 1,
    }


def sparse_metadata() -> Dict[str, EntryMetadata]:
    return {
        "frame": EntryMetadata(sample={(): CAMERA}, temporal={(): media(5)}),
        "grid": EntryMetadata(
            sample={(): CAMERA, (2,): None}, temporal={(0,): media(5), (0, 2): None}
        ),
    }


def write_recording(directory: Path, *, status: str = "complete") -> List[GroupResult]:
    results = [
        result("analysis", 0, complete_values(0)),
        result("analysis", 1, {}),
        result(
            "analysis",
            5,
            sparse_values(),
            filtered_paths={
                "crops": [(1,)],
                "grid": [(0, 1), (1,)],
                "stats/mean": [()],
            },
            metadata=sparse_metadata(),
            causes=(PulseKey("run-1", "video", 4),),
        ),
    ]
    writer = RecordingWriter.create(
        directory, schema=SCHEMA, codecs=CODECS, definition_digest="sha256:abc"
    )
    for item in results:
        writer.write(item)
    writer.write(result("events", 0, {"value": {"label": "start"}}))
    writer.finish(status)
    return results


def chunks(directory: Path, group: str = "analysis") -> list:
    return list(open_recording(directory, codecs=CODECS).group(group).iter_chunks())


def assert_images_equal(actual: ImageData, expected: ImageData) -> None:
    assert actual.image_id == expected.image_id
    assert actual.parent == expected.parent and actual.root == expected.root
    assert torch.equal(actual.tensor_image, expected.tensor_image)


def assert_chunk_matches(chunk, expected: GroupResult) -> None:
    assert chunk.pulse == RecordedPulse.from_key(expected.pulse)
    assert chunk.causes == tuple(
        RecordedPulse.from_key(cause) for cause in expected.causes
    )
    assert dict(chunk.statuses) == dict(expected.statuses)
    assert dict(chunk.filtered_paths) == dict(expected.filtered_paths)
    assert set(chunk.values) == set(expected.outputs.data)
    for key, value in expected.outputs.data.items():
        assert chunk.metadata[key] == expected.outputs.metadata[key]
        assert chunk.layouts[key] == expected.outputs.layout[key]
        if isinstance(value, ImageData):
            assert_images_equal(chunk.values[key], value)
        elif key == "crops":
            assert chunk.values[key].indices == value.indices
            for actual, wanted in zip(chunk.values[key], value):
                assert_images_equal(actual, wanted)
        else:
            assert chunk.values[key] == value
            assert type(chunk.values[key]) is type(value)


# Round trip ------------------------------------------------------------------------


def test_recorded_chunks_keep_values_statuses_paths_metadata_and_identity(
    tmp_path: Path,
) -> None:
    expected = write_recording(tmp_path / "rec")

    recorded = chunks(tmp_path / "rec")

    assert [chunk.index for chunk in recorded] == [0, 1, 2]
    for chunk, wanted in zip(recorded, expected):
        assert_chunk_matches(chunk, wanted)
    assert [chunk.pulse.sequence for chunk in recorded] == [0, 1, 5]
    assert [chunk.is_filtered for chunk in recorded] == [False, True, False]


def test_sparse_nested_entries_keep_gaps_empty_groups_and_barriers(
    tmp_path: Path,
) -> None:
    write_recording(tmp_path / "rec")

    sparse = chunks(tmp_path / "rec")[2]

    grid = sparse.values["grid"]
    assert grid.indices == ((0,), (2,))
    assert grid.content[0].indices == ((0, 0), (0, 2))
    assert grid.content[1].parent_index == (2,) and len(grid.content[1]) == 0
    assert sparse.metadata["grid"].sample_at((2, 0)) is None
    assert sparse.metadata["grid"].sample_at((0, 2)) == CAMERA
    assert sparse.metadata["grid"].temporal_at((0, 2)) is None
    assert sparse.metadata["frame"].temporal_at(()) == media(5)
    assert sparse.statuses["stats/mean"] == "filtered"


def test_batches_carry_the_entry_layout_and_metadata_views(tmp_path: Path) -> None:
    write_recording(tmp_path / "rec")

    sparse = chunks(tmp_path / "rec")[2]

    grid = sparse.values["grid"]
    for batch in (grid, *grid.content):
        assert batch.layout == GRID
        assert batch.metadata is sparse.metadata["grid"]


def test_data_groups_wildcard_ports_and_omits_filtered_fields(tmp_path: Path) -> None:
    write_recording(tmp_path / "rec")

    first, filtered, sparse = chunks(tmp_path / "rec")

    assert list(first.data) == ["frame", "crops", "grid", "stats"]
    assert first.data["stats"] == {"count": 2, "mean": np.float32(0.5)}
    assert filtered.data == {}
    assert sparse.data["stats"] == {"count": 1}


def test_groups_are_recorded_independently(tmp_path: Path) -> None:
    write_recording(tmp_path / "rec")
    recording = open_recording(tmp_path / "rec", codecs=CODECS)

    events = list(recording.group("events").iter_chunks())

    assert recording.groups == ("analysis", "events")
    assert len(recording.group("analysis")) == 3 and len(events) == 1
    assert events[0].data == {"value": {"label": "start"}}
    with pytest.raises(RecordingSchemaError, match="no recorded group 'other'"):
        recording.group("other")


def test_manifest_records_schema_provenance_counts_and_used_codecs(
    tmp_path: Path,
) -> None:
    write_recording(tmp_path / "rec")

    info = inspect_recording(tmp_path / "rec")

    assert info.status == "complete" and info.error is None
    assert info.schema == SCHEMA
    assert info.definition_digest == "sha256:abc"
    assert info.counts["analysis"]["chunks"] == 3
    assert set(info.codecs) == {
        "v2/image_data@1",
        "v2/torch_tensor@1",
        "v2/numpy_scalar@1",
    }
    assert info.finished_at is not None


def test_pixels_are_raw_binary_outside_the_records(tmp_path: Path) -> None:
    write_recording(tmp_path / "rec")

    records = (tmp_path / "rec" / "groups" / "000" / "records.jsonl").read_bytes()
    blobs = (tmp_path / "rec" / "groups" / "000" / "blobs.bin").read_bytes()

    assert image(0).tensor_image.numpy().tobytes() in blobs
    assert len(records) < 8_000


def test_writer_snapshots_values_before_later_mutation(tmp_path: Path) -> None:
    frame = image(1)
    array = np.zeros(4)
    writer = RecordingWriter.create(
        tmp_path / "rec", schema=SCHEMA, codecs=CODECS, definition_digest=""
    )

    writer.write(result("events", 0, {"value": {"frame": frame, "array": array}}))
    frame.tensor_image.fill_(255)
    array[:] = 7
    writer.finish("complete")

    (chunk,) = chunks(tmp_path / "rec", "events")
    assert int(chunk.data["value"]["frame"].tensor_image.max()) == 1
    assert not chunk.data["value"]["array"].any()


# Lazy repeatable reading ------------------------------------------------------------


def test_iteration_is_repeatable_and_iterators_are_independent(tmp_path: Path) -> None:
    write_recording(tmp_path / "rec")
    group = open_recording(tmp_path / "rec", codecs=CODECS).group("analysis")

    first, second = group.iter_chunks(), group.iter_chunks()
    a0, b0, a1 = next(first), next(second), next(first)
    rest_b = list(second)

    assert (a0.index, b0.index, a1.index) == (0, 0, 1)
    assert [chunk.index for chunk in rest_b] == [1, 2]
    a0.values["frame"].tensor_image.fill_(200)
    assert int(next(group.iter_chunks()).values["frame"].tensor_image.max()) == 0
    assert [chunk.pulse for chunk in group] == [chunk.pulse for chunk in group]


def test_abandoned_iterators_release_their_files(tmp_path: Path) -> None:
    write_recording(tmp_path / "rec")
    group = open_recording(tmp_path / "rec", codecs=CODECS).group("analysis")

    iterator = group.iter_chunks()
    next(iterator)
    iterator.close()

    assert list(iterator) == []


@pytest.mark.parametrize("count", [40, 160])
def test_reading_memory_does_not_grow_with_recording_length(
    tmp_path: Path, count: int
) -> None:
    schema = RecordingSchema([EVENTS])
    writer = RecordingWriter.create(
        tmp_path / "rec", schema=schema, codecs=CODECS, definition_digest=""
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for sequence in range(count):
        frame[0, 0, 0] = sequence % 256
        writer.write(result("events", sequence, {"value": frame}))
    writer.finish("complete")
    group = open_recording(tmp_path / "rec", codecs=CODECS).group("events")

    tracemalloc.start()
    try:
        seen = sum(int(chunk.values["value"][0, 0, 0]) for chunk in group.iter_chunks())
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert seen == sum(sequence % 256 for sequence in range(count))
    assert peak < 4 * frame.nbytes  # a few frames in flight, never all of them


# Ownership and lifecycle --------------------------------------------------------------


def test_a_destination_is_claimed_by_exactly_one_writer(tmp_path: Path) -> None:
    write_recording(tmp_path / "rec")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "notes.txt").write_text("user file")

    with pytest.raises(RecordingExistsError, match="never overwritten"):
        RecordingWriter.create(
            tmp_path / "rec", schema=SCHEMA, codecs=CODECS, definition_digest=""
        )
    with pytest.raises(RecordingExistsError):
        RecordingWriter.create(
            tmp_path / "other", schema=SCHEMA, codecs=CODECS, definition_digest=""
        )
    assert (tmp_path / "other" / "notes.txt").read_text() == "user file"
    assert len(chunks(tmp_path / "rec")) == 3


def test_racing_writers_cannot_share_a_destination(tmp_path: Path) -> None:
    outcomes: List[str] = []
    barrier = threading.Barrier(8)

    def claim() -> None:
        barrier.wait()
        try:
            writer = RecordingWriter.create(
                tmp_path / "rec", schema=SCHEMA, codecs=CODECS, definition_digest=""
            )
        except RecordingExistsError:
            outcomes.append("exists")
        else:
            outcomes.append("won")
            writer.finish("complete")

    threads = [threading.Thread(target=claim) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outcomes.count("won") == 1
    assert outcomes.count("exists") == 7


def test_finish_is_idempotent_and_closes_the_writer(tmp_path: Path) -> None:
    writer = RecordingWriter.create(
        tmp_path / "rec", schema=SCHEMA, codecs=CODECS, definition_digest=""
    )
    writer.write(result("events", 0, {"value": 1}))

    writer.finish("stopped")
    writer.finish("failed", error="late")

    assert writer.status == "stopped"
    assert inspect_recording(tmp_path / "rec").status == "stopped"
    with pytest.raises(RecordingError, match="cannot write after finish"):
        writer.write(result("events", 1, {"value": 2}))
    with pytest.raises(RecordingError, match="unknown final status 'done'"):
        writer.finish("done")
    assert dict(writer.counters["events"])["chunks"] == 1


class CloseFails:
    """A file handle whose close closes the file, then reports a failure."""

    def __init__(self, handle) -> None:
        self.handle = handle

    def __getattr__(self, name: str) -> Any:
        return getattr(self.handle, name)

    def close(self) -> None:
        self.handle.close()
        raise OSError("injected close failure")


def group_handles(writer: RecordingWriter) -> list:
    return [
        handle
        for group_writer in writer._groups.values()
        for handle in (group_writer._records, group_writer._blobs.handle)
    ]


@pytest.mark.parametrize("failing", ["records", "blobs"])
def test_a_close_failure_closes_every_file_and_publishes_failed(
    tmp_path: Path, failing: str
) -> None:
    writer = RecordingWriter.create(
        tmp_path / "rec", schema=SCHEMA, codecs=CODECS, definition_digest=""
    )
    writer.write(result("events", 0, {"value": 1}))
    analysis = writer._groups["analysis"]
    if failing == "records":
        analysis._records = CloseFails(analysis._records)
    else:
        analysis._blobs.handle = CloseFails(analysis._blobs.handle)

    with pytest.raises(RecordingError) as caught:
        writer.finish("complete")

    assert caught.value.group == "analysis"
    assert f"{failing}." in str(caught.value)
    assert "injected close failure" in str(caught.value)
    assert all(handle.closed for handle in group_handles(writer))
    assert writer.status == "failed"
    info = inspect_recording(tmp_path / "rec")
    assert info.status == "failed"
    assert "injected close failure" in info.error
    assert info.counts["events"]["chunks"] == 1
    with pytest.raises(RecordingIncompleteError):
        open_recording(tmp_path / "rec", codecs=CODECS)


def test_an_unwritable_final_manifest_keeps_the_close_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = RecordingWriter.create(
        tmp_path / "rec", schema=SCHEMA, codecs=CODECS, definition_digest=""
    )
    analysis = writer._groups["analysis"]
    analysis._records = CloseFails(analysis._records)

    def disk_full(path: Path, manifest: Dict[str, Any]) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(
        "roboflow_workflows.execution_engine.v2.recording.store._replace", disk_full
    )
    with pytest.raises(RecordingError) as caught:
        writer.finish("complete")

    assert "cannot write the final manifest: disk full" in str(caught.value)
    assert "injected close failure" in str(caught.value)
    assert all(handle.closed for handle in group_handles(writer))
    assert writer.status == "failed"
    with pytest.raises(RecordingIncompleteError, match="status is 'recording'"):
        open_recording(tmp_path / "rec", codecs=CODECS)


@pytest.mark.parametrize("status", ["complete", "stopped"])
def test_complete_and_stopped_recordings_are_readable(
    tmp_path: Path, status: str
) -> None:
    write_recording(tmp_path / "rec", status=status)

    assert open_recording(tmp_path / "rec", codecs=CODECS).status == status


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_failed_and_cancelled_recordings_are_rejected_but_inspectable(
    tmp_path: Path, status: str
) -> None:
    writer = RecordingWriter.create(
        tmp_path / "rec", schema=SCHEMA, codecs=CODECS, definition_digest=""
    )
    writer.write(result("events", 0, {"value": 1}))
    writer.finish(status, error="model crashed" if status == "failed" else None)

    with pytest.raises(RecordingIncompleteError) as caught:
        open_recording(tmp_path / "rec", codecs=CODECS)

    assert caught.value.status == status
    assert f"status is '{status}'" in str(caught.value)
    info = inspect_recording(tmp_path / "rec")
    assert info.status == status and info.counts["events"]["chunks"] == 1
    if status == "failed":
        assert "model crashed" in str(caught.value)


def test_an_unfinalized_recording_is_rejected(tmp_path: Path) -> None:
    writer = RecordingWriter.create(
        tmp_path / "rec", schema=SCHEMA, codecs=CODECS, definition_digest=""
    )
    writer.write(result("events", 0, {"value": 1}))

    with pytest.raises(RecordingIncompleteError, match="status is 'recording'"):
        open_recording(tmp_path / "rec", codecs=CODECS)
    assert inspect_recording(tmp_path / "rec").counts["events"]["chunks"] is None
    writer.finish("cancelled")


def test_concurrent_writers_of_different_groups_keep_per_group_order(
    tmp_path: Path,
) -> None:
    writer = RecordingWriter.create(
        tmp_path / "rec", schema=SCHEMA, codecs=CODECS, definition_digest=""
    )

    def produce(group: str, values) -> None:
        for sequence in range(50):
            writer.write(result(group, sequence, values(sequence)))

    threads = [
        threading.Thread(target=produce, args=("events", lambda s: {"value": s})),
        threading.Thread(
            target=produce, args=("analysis", lambda s: {"stats/count": s})
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    writer.finish("complete")

    events = chunks(tmp_path / "rec", "events")
    analysis = chunks(tmp_path / "rec", "analysis")
    assert [chunk.data["value"] for chunk in events] == list(range(50))
    assert [chunk.data["stats"]["count"] for chunk in analysis] == list(range(50))


# Explicit write failures -------------------------------------------------------------


def test_results_that_do_not_match_the_schema_are_rejected(tmp_path: Path) -> None:
    writer = RecordingWriter.create(
        tmp_path / "rec", schema=SCHEMA, codecs=CODECS, definition_digest=""
    )

    with pytest.raises(RecordingSchemaError, match="group 'unknown'"):
        writer.write(dataclasses.replace(result("events", 0, {}), group="unknown"))
    with pytest.raises(RecordingSchemaError, match="differ from the recorded entries"):
        writer.write(
            result("events", 0, {"value": 1, "extra": 2}, keys=["value", "extra"])
        )
    with pytest.raises(RecordingSchemaError, match="layout differs"):
        writer.write(
            result(
                "analysis",
                0,
                {"stats/count": Batch([1], indices=[(0,)])},
                layouts={"stats/count": CROPS},
            )
        )
    writer.finish("failed", error="schema")


def test_a_payload_without_codec_names_group_entry_and_path(tmp_path: Path) -> None:
    writer = RecordingWriter.create(
        tmp_path / "rec", schema=SCHEMA, codecs=CODECS, definition_digest=""
    )

    with pytest.raises(RecordingCodecError) as caught:
        writer.write(result("events", 0, {"value": {"handle": object()}}))

    assert "events.value['handle']: no recording codec for builtins.object" in str(
        caught.value
    )
    writer.write(result("events", 1, {"value": 1}))
    writer.finish("failed", error=str(caught.value))
    assert inspect_recording(tmp_path / "rec").counts["events"]["chunks"] == 1


# Integrity -----------------------------------------------------------------------------


def records_path(directory: Path, group: str = "000") -> Path:
    return directory / "groups" / group / "records.jsonl"


def rewrite_manifest(directory: Path, **changes) -> None:
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    for name, value in changes.items():
        if name == "analysis":
            manifest["groups"][0].update(value)
        else:
            manifest[name] = value
    path.write_text(json.dumps(manifest))


def test_a_truncated_file_is_rejected_on_open(tmp_path: Path) -> None:
    write_recording(tmp_path / "rec")
    path = records_path(tmp_path / "rec")
    path.write_bytes(path.read_bytes()[:-10])

    with pytest.raises(RecordingCorruptError, match="truncated or modified"):
        open_recording(tmp_path / "rec", codecs=CODECS)


def test_an_altered_record_fails_at_its_chunk_after_earlier_chunks(
    tmp_path: Path,
) -> None:
    write_recording(tmp_path / "rec")
    path = records_path(tmp_path / "rec")
    lines = path.read_bytes().split(b"\n")
    lines[1] = lines[1].replace(b'"index":1', b'"index":7')
    path.write_bytes(b"\n".join(lines))
    iterator = (
        open_recording(tmp_path / "rec", codecs=CODECS).group("analysis").iter_chunks()
    )

    assert next(iterator).index == 0
    with pytest.raises(RecordingCorruptError) as caught:
        next(iterator)

    assert caught.value.chunk_index == 1
    assert "group 'analysis', chunk 1" in str(caught.value)
    assert "checksum mismatch" in str(caught.value)


def test_an_altered_blob_fails_its_checksum(tmp_path: Path) -> None:
    write_recording(tmp_path / "rec")
    path = tmp_path / "rec" / "groups" / "000" / "blobs.bin"
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))

    with pytest.raises(RecordingCorruptError) as caught:
        chunks(tmp_path / "rec")

    assert caught.value.chunk_index == 2
    assert "fails its checksum" in str(caught.value)


def test_a_torn_last_record_is_an_error_not_a_shorter_recording(tmp_path: Path) -> None:
    write_recording(tmp_path / "rec")
    path = records_path(tmp_path / "rec")
    torn = path.read_bytes()[:-5]
    path.write_bytes(torn)
    rewrite_manifest(tmp_path / "rec", analysis={"record_bytes": len(torn)})

    with pytest.raises(RecordingCorruptError, match="chunk 2.*torn record"):
        chunks(tmp_path / "rec")


def test_missing_and_surplus_records_are_errors(tmp_path: Path) -> None:
    write_recording(tmp_path / "rec")
    path = records_path(tmp_path / "rec")
    lines = path.read_bytes().splitlines(keepends=True)
    rewrite_manifest(tmp_path / "rec", analysis={"chunks": 4})

    with pytest.raises(
        RecordingCorruptError, match="has 3 records, the manifest says 4"
    ):
        chunks(tmp_path / "rec")

    rewrite_manifest(tmp_path / "rec", analysis={"chunks": 2})
    with pytest.raises(RecordingCorruptError, match="surplus record"):
        chunks(tmp_path / "rec")
    assert len(lines) == 3


def test_out_of_bounds_blob_references_are_errors(tmp_path: Path) -> None:
    write_recording(tmp_path / "rec")
    path = tmp_path / "rec" / "groups" / "000" / "blobs.bin"
    path.write_bytes(path.read_bytes()[:-1])
    size = path.stat().st_size
    rewrite_manifest(tmp_path / "rec", analysis={"blob_bytes": size})

    with pytest.raises(RecordingCorruptError, match="outside the .*-byte blob file"):
        chunks(tmp_path / "rec")


def test_format_codecs_and_manifest_problems_are_explicit(tmp_path: Path) -> None:
    write_recording(tmp_path / "rec")

    with pytest.raises(
        RecordingCodecError, match=r"\['v2/image_data@1', 'v2/torch_tensor@1'\]"
    ):
        open_recording(tmp_path / "rec", codecs=CodecRegistry())

    rewrite_manifest(tmp_path / "rec", format=2)
    with pytest.raises(RecordingSchemaError, match="unsupported recording format 2"):
        open_recording(tmp_path / "rec", codecs=CODECS)

    (tmp_path / "rec" / "manifest.json").write_text("{")
    with pytest.raises(RecordingCorruptError, match="not valid JSON"):
        inspect_recording(tmp_path / "rec")
    with pytest.raises(RecordingError, match="not a recording"):
        inspect_recording(tmp_path / "missing")


def test_group_directories_cannot_escape_the_recording(tmp_path: Path) -> None:
    write_recording(tmp_path / "rec")
    rewrite_manifest(tmp_path / "rec", analysis={"directory": "../elsewhere"})

    with pytest.raises(RecordingCorruptError, match="escapes"):
        inspect_recording(tmp_path / "rec")


def test_a_group_directory_linked_outside_the_recording_is_rejected(
    tmp_path: Path,
) -> None:
    write_recording(tmp_path / "rec")
    group = tmp_path / "rec" / "groups" / "000"
    group.rename(tmp_path / "outside")
    group.symlink_to(tmp_path / "outside", target_is_directory=True)

    with pytest.raises(
        RecordingCorruptError,
        match="the group directory resolves to .*outside the recording",
    ) as caught:
        open_recording(tmp_path / "rec", codecs=CODECS)

    assert caught.value.group == "analysis"


@pytest.mark.parametrize("file_name", ["records.jsonl", "blobs.bin"])
def test_a_group_file_linked_outside_the_recording_is_rejected(
    tmp_path: Path, file_name: str
) -> None:
    write_recording(tmp_path / "rec")
    path = tmp_path / "rec" / "groups" / "000" / file_name
    path.rename(tmp_path / file_name)
    path.symlink_to(tmp_path / file_name)

    with pytest.raises(RecordingCorruptError, match=f"{file_name} resolves to"):
        open_recording(tmp_path / "rec", codecs=CODECS)


def test_a_group_file_that_is_not_a_regular_file_is_rejected(tmp_path: Path) -> None:
    write_recording(tmp_path / "rec")
    path = tmp_path / "rec" / "groups" / "001" / "blobs.bin"
    path.unlink()
    path.mkdir()

    with pytest.raises(RecordingCorruptError, match="blobs.bin is not a regular file"):
        open_recording(tmp_path / "rec", codecs=CODECS)


def test_links_inside_the_recording_and_a_linked_recording_are_readable(
    tmp_path: Path,
) -> None:
    expected = write_recording(tmp_path / "rec")
    group = tmp_path / "rec" / "groups" / "000"
    group.rename(tmp_path / "rec" / "moved")
    group.symlink_to(tmp_path / "rec" / "moved", target_is_directory=True)
    (tmp_path / "alias").symlink_to(tmp_path / "rec", target_is_directory=True)

    recorded = chunks(tmp_path / "alias")

    assert [chunk.index for chunk in recorded] == [0, 1, 2]
    assert_chunk_matches(recorded[0], expected[0])


# Real engine results -----------------------------------------------------------------


ITEMS = EntryLayout((Axis(id="items", kind="dynamic_nesting"),))


class Feed(Source):
    type = "test/recording_feed@v1"
    outputs = {
        "value": SourceOutput(FLOAT_KIND),
        "pair": SourceOutput(FLOAT_KIND, layout=ITEMS),
    }

    def __init__(self, *, feed: list) -> None:
        self.items = iter(feed)

    def open(self) -> None:
        pass

    def read(self) -> Optional[Emission]:
        return next(self.items, None)


def test_group_results_of_an_active_run_round_trip(tmp_path: Path) -> None:
    feed = [
        Emission(
            {
                "value": float(position),
                "pair": Batch([position, -1.0], indices=[(0,), (1,)]),
            },
            media=Timestamp(position * 40, Fraction(1, 1000), "media"),
            source_metadata={"frame": position},
        )
        for position in range(4)
    ]
    plan = compile_workflow(
        {
            "version": "2.0",
            "sources": [{"type": Feed.type, "name": "video"}],
            "steps": [],
            "outputs": [
                {
                    "type": "OutputGroup",
                    "name": "analysis",
                    "anchor": "$sources.video.value",
                    "outputs": [
                        {
                            "type": "JsonField",
                            "name": "value",
                            "selector": "$sources.video.value",
                        },
                        {
                            "type": "JsonField",
                            "name": "pair",
                            "selector": "$sources.video.pair",
                        },
                    ],
                }
            ],
        },
        catalogue=Catalogue(sources=[Feed]),
    )
    delivered: List[GroupResult] = []
    writers: Dict[str, RecordingWriter] = {}

    def record(group_result: GroupResult) -> None:
        if not writers:
            schema = RecordingSchema(
                [
                    RecordedGroupSchema(
                        name="analysis",
                        anchor_domain="video",
                        entries=tuple(
                            RecordedEntrySchema(
                                key,
                                key,
                                None,
                                ("float",),
                                group_result.outputs.layout[key],
                            )
                            for key in group_result.statuses
                        ),
                    )
                ]
            )
            writers["w"] = RecordingWriter.create(
                tmp_path / "rec", schema=schema, codecs=CODECS, definition_digest=""
            )
        writers["w"].write(group_result)
        delivered.append(group_result)

    run = plan.create_session(resources={"feed": feed}).start(
        handlers={"analysis": record}
    )
    assert run.wait(10.0)
    writers["w"].finish("complete")

    recorded = chunks(tmp_path / "rec")
    assert len(recorded) == len(delivered) == 4
    for chunk, expected in zip(recorded, delivered):
        assert chunk.pulse == RecordedPulse.from_key(expected.pulse)
        assert dict(chunk.values) == dict(expected.outputs.data)
        assert dict(chunk.metadata) == dict(expected.outputs.metadata)
        assert chunk.metadata["pair"].sample_at((0,)).source_metadata == {
            "frame": chunk.index
        }
