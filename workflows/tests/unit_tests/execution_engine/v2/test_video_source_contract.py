"""Conformance of the public source-author contract used by video sources.

Small fake sources stand in for real cameras; nothing decodes or runs a model.
``CameraSet`` collects one frame per ready camera into one emission over a
stationary camera axis and keeps the camera position as the index::

    pulse 0: cameras (0,) and (2,)        pulse 1: camera (1,)

    cams.frame[camera] ──▶ member   one call per camera, state.source
                       ├─▶ batch    one call per pulse (batch="always"), state.at
                       └─▶ w        v2/window@v1, size 2: camera axis, then T
    groups: frames (frame, seen, tally) and clips (clip)

Each member carries its own context, built by the source:

    SampleContext.source_id       "<source_name>/<camera>"
    TemporalContext.observed      engine_observation() when the frame arrived
    TemporalContext.media         PTS on clock "<source_name>/<camera>:pts"
    TemporalContext.capture       None: no physical capture time is known

The pulse context at ``()`` keeps the engine's observation of the whole
collection. Every wait is bounded by ``WAIT``; every run is finished or
cancelled before the test returns.
"""

import threading
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from roboflow_workflows.execution_engine.v2.active import execution
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.data import (
    Axis,
    Batch,
    EntryLayout,
    EntryMetadata,
    InputValue,
    SampleContext,
    TemporalContext,
    Timestamp,
)
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND, INTEGER_KIND
from roboflow_workflows.execution_engine.v2.operators.window import Window
from roboflow_workflows.execution_engine.v2.recording.codecs import CodecRegistry
from roboflow_workflows.execution_engine.v2.recording.store import open_recording
from roboflow_workflows.execution_engine.v2.sources import (
    ENGINE_CLOCK_ID,
    Emission,
    Source,
    SourceOutput,
    SourceParams,
    engine_observation,
)
from roboflow_workflows.execution_engine.v2.state import ManagedState

WAIT = 10.0
"""Upper bound of every wait, in seconds; the tests finish far earlier."""

COLLECTION_WAIT = 0.002
"""Seconds a camera set waits after its last member arrived, like a timeout."""

PTS = Fraction(1, 90_000)
FRAME_TICKS = 3_000
"""PTS ticks between frames of one camera (30 FPS on a 90 kHz clock)."""

CAMERAS = EntryLayout((Axis(id="camera", kind="sample", stationary=True),))

SPARSE_PULSES = [[(0, 10.0), (2, 12.0)], [(1, 21.0)]]
"""Two pulses of ``(camera, value)`` members: cameras 0 and 2, then camera 1."""


class Probe(Source):
    """One ungrouped value per pulse; logs where each lifecycle call runs."""

    type = "test/probe@v1"
    outputs = {"value": SourceOutput(FLOAT_KIND)}

    def __init__(self, *, log: List[Tuple[str, Optional[str], int]]) -> None:
        self.log = log
        self.log.append(("construct", getattr(self, "source_name", None), _thread()))
        self.remaining = 1

    def open(self) -> None:
        self.log.append(("open", self.source_name, _thread()))

    def read(self) -> Optional[Emission]:
        self.log.append(("read", self.source_name, _thread()))
        if self.remaining == 0:
            return None

        self.remaining -= 1

        return Emission({"value": 1.0})

    def close(self) -> None:
        self.log.append(("close", self.source_name, _thread()))


class CameraSet(Source):
    """A set of cameras; each emission holds the members that were ready."""

    type = "test/camera_set@v1"
    outputs = {"frame": SourceOutput(FLOAT_KIND, layout=CAMERAS)}

    class Params(SourceParams):
        script: str

    def __init__(self, *, scripts: Dict[str, List[list]]) -> None:
        self.scripts = scripts
        self.pulses: Iterator[list] = iter(())
        self.frames_read: Dict[int, int] = {}

    def open(self, *, script: str) -> None:
        self.pulses = iter(self.scripts[script])

    def read(self) -> Optional[Emission]:
        members = next(self.pulses, None)
        if members is None:
            return None

        values, indices, sample, temporal = [], [], {}, {}
        for camera, value in members:
            index = (camera,)
            member_id = f"{self.source_name}/{camera}"
            frame_number = self.frames_read.get(camera, 0)
            self.frames_read[camera] = frame_number + 1
            values.append(value)
            indices.append(index)
            sample[index] = SampleContext(
                source_id=member_id,
                source_type=self.type,
                source_metadata={"camera": camera, "frame": frame_number},
            )
            temporal[index] = TemporalContext(
                observed_coverage=engine_observation(),
                media_coverage=Timestamp(
                    frame_number * FRAME_TICKS, PTS, f"{member_id}:pts"
                ),
                capture_coverage=None,
            )
        self.stop_event.wait(COLLECTION_WAIT)

        frames = InputValue(
            Batch.of(values, indices=indices),
            EntryMetadata(sample=sample, temporal=temporal),
        )

        return Emission({"frame": frames})


class PerMember(Block):
    """Ordinary block: one call per camera, counting in its camera's state."""

    type = "test/per_member@v1"
    outputs = {"seen": Output(INTEGER_KIND)}

    class Params(BlockParams):
        frame: Ref(FLOAT_KIND)

    def __init__(self, *, managed_state: ManagedState) -> None:
        self.state = managed_state
        self.calls: List[float] = []

    def run(self, *, frame) -> dict:
        self.calls.append(frame)
        seen = self.state.source.incr("member_calls")

        return {"seen": seen}


class PerBatch(Block):
    """Batch block: one call per pulse with every present camera."""

    type = "test/per_batch@v1"
    outputs = {"tally": Output(INTEGER_KIND)}

    class Params(BlockParams):
        frames: Ref(FLOAT_KIND, batch="always")

    def __init__(self, *, managed_state: ManagedState) -> None:
        self.state = managed_state
        self.calls: List[Tuple[tuple, tuple]] = []

    def run(self, *, frames) -> list:
        self.calls.append((frames.indices, frames.content))
        tallies = [
            {"tally": self.state.at(index).incr("batch_members")}
            for index in frames.indices
        ]

        return tallies


CATALOGUE = Catalogue(
    [PerMember, PerBatch],
    sources=[Probe, CameraSet],
    operators=[Window],
)


def _thread() -> int:
    return threading.get_ident()


def field(name: str, selector: str) -> Dict[str, str]:
    return {"type": "JsonField", "name": name, "selector": selector}


def group(name: str, anchor: str, *fields: Dict[str, str]) -> Dict[str, Any]:
    return {
        "type": "OutputGroup",
        "name": name,
        "anchor": anchor,
        "outputs": list(fields),
    }


def camera_definition(*, recorded: bool = False) -> Dict[str, Any]:
    """The camera set feeding a per-member block, a batch block and a window."""
    definition = {
        "version": "2.0",
        "inputs": [],
        "sources": [{"type": CameraSet.type, "name": "cams", "script": "sparse"}],
        "operators": [
            {
                "type": "v2/window@v1",
                "name": "w",
                "size": 2,
                "collect": {"frame": "$sources.cams.frame"},
            }
        ],
        "steps": [
            {"type": PerMember.type, "name": "member", "frame": "$sources.cams.frame"},
            {"type": PerBatch.type, "name": "batch", "frames": "$sources.cams.frame"},
        ],
        "outputs": [
            group(
                "frames",
                "$sources.cams.frame",
                field("frame", "$sources.cams.frame"),
                field("seen", "$steps.member.seen"),
                field("tally", "$steps.batch.tally"),
            ),
            group("clips", "$operators.w.frame", field("clip", "$operators.w.frame")),
        ],
    }
    if recorded:
        definition["inputs"] = [{"type": "WorkflowParameter", "name": "capture_dir"}]
        definition["recording"] = {
            "type": "file",
            "directory": "$inputs.capture_dir",
            "groups": ["frames", "clips"],
        }
        definition["retrospective"] = {
            "type": "workflow",
            "input_group": "frames",
            "workflow": {
                "outputs": [
                    group(
                        "reviewed",
                        "$sources.frames.frame",
                        *(
                            field(name, f"$sources.frames.{name}")
                            for name in ("frame", "seen", "tally")
                        ),
                    )
                ]
            },
        }

    return definition


def run_to_end(
    start: Callable[..., Any], *, groups: Tuple[str, ...], **options: Any
) -> Dict[str, List[Any]]:
    """Start a run, wait for it to finish and return every delivered result.

    ``start`` is ``session.start`` or ``plan.retrospective.start``; it gets
    one collecting handler per group plus ``options``.
    """
    collected: Dict[str, List[Any]] = {name: [] for name in groups}
    handlers = {name: collected[name].append for name in groups}
    run = start(handlers=handlers, **options)
    try:
        assert run.wait(WAIT)
    finally:
        if not run.done:
            run.cancel()
            run.wait(WAIT)

    return collected


def camera_session(**resources: Any) -> Any:
    plan = compile_workflow(camera_definition(), catalogue=CATALOGUE)
    session = plan.create_session({"scripts": {"sparse": SPARSE_PULSES}, **resources})

    return session


def plain(value: Any) -> Any:
    """A batch as ``[(index, item), ...]``, recursively; other values unchanged."""
    if isinstance(value, Batch):
        return [(index, plain(item)) for index, item in value.iter_with_indices()]

    return value


# Public API and source identity ----------------------------------------------


def test_engine_clock_is_public_in_sources_and_still_importable_from_execution() -> (
    None
):
    stamp = engine_observation()

    assert ENGINE_CLOCK_ID == "engine.monotonic"
    assert stamp.clock_id == ENGINE_CLOCK_ID
    assert stamp.time_base == Fraction(1, 10**9)
    assert execution.engine_observation is engine_observation
    assert execution.ENGINE_CLOCK_ID is ENGINE_CLOCK_ID


def test_source_name_is_set_before_open_and_lifecycle_runs_on_the_reader_thread() -> (
    None
):
    log: List[Tuple[str, Optional[str], int]] = []
    plan = compile_workflow(
        {
            "version": "2.0",
            "inputs": [],
            "sources": [
                {"type": Probe.type, "name": "left"},
                {"type": Probe.type, "name": "right"},
            ],
            "steps": [],
            "outputs": [
                group("L", "$sources.left.value", field("v", "$sources.left.value")),
                group("R", "$sources.right.value", field("v", "$sources.right.value")),
            ],
        },
        catalogue=CATALOGUE,
    )
    session = plan.create_session({"log": log})

    collected = run_to_end(session.start, groups=("L", "R"))

    caller = _thread()
    constructed = [entry for entry in log if entry[0] == "construct"]
    assert constructed == [("construct", None, caller), ("construct", None, caller)]
    for name in ("left", "right"):
        calls = [(call, thread) for call, owner, thread in log if owner == name]
        assert [call for call, _ in calls] == ["open", "read", "read", "close"]
        (reader,) = {thread for _, thread in calls}
        assert reader != caller
    for group_name, name in (("L", "left"), ("R", "right")):
        (result,) = collected[group_name]
        sample = result.outputs.metadata["v"].sample_at(())
        assert (sample.source_id, sample.source_type) == (name, Probe.type)
        assert result.outputs.metadata["v"].temporal_at(()).capture_coverage is None


# Member timing ------------------------------------------------------------------


def test_member_observed_precedes_collection_with_unknown_capture_and_own_clocks() -> (
    None
):
    session = camera_session(managed_state=ManagedState())

    collected = run_to_end(session.start, groups=("frames",))

    media_clocks = set()
    for result, members in zip(collected["frames"], SPARSE_PULSES):
        metadata = result.outputs.metadata["frame"]
        pulse = metadata.temporal_at(())
        assert pulse.observed_coverage.clock_id == ENGINE_CLOCK_ID
        assert pulse.capture_coverage is None
        for camera, _ in members:
            member = metadata.temporal_at((camera,))
            assert member.observed_coverage.clock_id == ENGINE_CLOCK_ID
            assert member.observed_coverage.ticks < pulse.observed_coverage.ticks
            assert member.capture_coverage is None
            assert member.media_coverage.clock_id == f"cams/{camera}:pts"
            assert member.media_coverage.time_base == PTS
            media_clocks.add(member.media_coverage.clock_id)
    assert media_clocks == {"cams/0:pts", "cams/1:pts", "cams/2:pts"}


# Sparse stationary indices ----------------------------------------------------


def test_sparse_camera_indices_reach_per_member_and_batch_blocks_and_groups() -> None:
    session = camera_session(managed_state=ManagedState())

    collected = run_to_end(session.start, groups=("frames",))

    first, second = collected["frames"]
    assert [result.pulse.sequence for result in (first, second)] == [0, 1]
    assert first.outputs.data["frame"].indices == ((0,), (2,))
    assert second.outputs.data["frame"].indices == ((1,),)
    assert {name: plain(data) for name, data in first.outputs.data.items()} == {
        "frame": [((0,), 10.0), ((2,), 12.0)],
        "seen": [((0,), 1), ((2,), 1)],
        "tally": [((0,), 1), ((2,), 1)],
    }
    assert {name: plain(data) for name, data in second.outputs.data.items()} == {
        "frame": [((1,), 21.0)],
        "seen": [((1,), 1)],
        "tally": [((1,), 1)],
    }
    # rows() is the V1 positional view: an absent camera is an all-None row.
    assert second.rows() == [
        {"frame": None, "seen": None, "tally": None},
        {"frame": 21.0, "seen": 1, "tally": 1},
    ]
    for result in (first, second):
        assert result.statuses == {
            "frame": "complete",
            "seen": "complete",
            "tally": "complete",
        }
        for name in ("seen", "tally"):
            assert result.outputs.data[name].indices == (
                result.outputs.data["frame"].indices
            )
    assert session.instances[("member",)].calls == [10.0, 12.0, 21.0]
    assert session.instances[("batch",)].calls == [
        (((0,), (2,)), (10.0, 12.0)),
        (((1,),), (21.0,)),
    ]


def test_member_identity_is_one_string_in_sample_context_and_managed_state() -> None:
    state = ManagedState()
    session = camera_session(managed_state=state)

    first_run = run_to_end(session.start, groups=("frames",))
    second_run = run_to_end(session.start, groups=("frames",))

    for run in (first_run, second_run):
        for result in run["frames"]:
            metadata = result.outputs.metadata["frame"]
            for index in result.outputs.data["frame"].indices:
                sample = metadata.sample_at(index)
                assert sample.source_id == f"cams/{index[0]}"
                assert sample.source_type == CameraSet.type
                assert sample.source_metadata["camera"] == index[0]
    for camera in (0, 1, 2):
        scope = state.for_source(f"cams/{camera}")
        assert scope.get("member_calls") == 2
        assert scope.get("batch_members") == 2
    assert state.for_source("cams").get("member_calls") is None
    assert state.global_.get("member_calls") is None
    assert plain(second_run["frames"][0].outputs.data["seen"]) == [((0,), 2), ((2,), 2)]


# Recording, replay and windows -------------------------------------------------


def test_recording_and_replay_preserve_sparse_indices_and_member_metadata(
    tmp_path: Path,
) -> None:
    plan = compile_workflow(camera_definition(recorded=True), catalogue=CATALOGUE)
    session = plan.create_session(
        {"scripts": {"sparse": SPARSE_PULSES}, "managed_state": ManagedState()}
    )
    directory = tmp_path / "rec"

    delivered = run_to_end(
        session.start, groups=("frames",), inputs={"capture_dir": str(directory)}
    )["frames"]
    replayed = run_to_end(
        plan.retrospective.start, groups=("reviewed",), recording=directory
    )["reviewed"]

    recording = open_recording(
        directory, codecs=CodecRegistry(CATALOGUE.codecs.values())
    )
    chunks = list(recording.group("frames").iter_chunks())
    assert len(delivered) == len(replayed) == len(chunks) == 2
    for original, chunk, result in zip(delivered, chunks, replayed):
        for name in ("frame", "seen", "tally"):
            expected = plain(original.outputs.data[name])
            assert plain(chunk.values[name]) == expected
            assert plain(result.outputs.data[name]) == expected
            for restored in (chunk.metadata[name], result.outputs.metadata[name]):
                assert dict(restored.sample) == dict(
                    original.outputs.metadata[name].sample
                )
                assert dict(restored.temporal) == dict(
                    original.outputs.metadata[name].temporal
                )
    assert replayed[0].outputs.data["frame"].indices == ((0,), (2,))
    assert replayed[1].outputs.data["frame"].indices == ((1,),)
    member = replayed[0].outputs.metadata["frame"]
    assert member.sample_at((2,)).source_id == "cams/2"
    assert member.temporal_at((2,)).media_coverage.clock_id == "cams/2:pts"
    assert member.temporal_at((2,)).capture_coverage is None


def test_window_appends_time_after_the_stationary_camera_axis(tmp_path: Path) -> None:
    plan = compile_workflow(camera_definition(recorded=True), catalogue=CATALOGUE)
    session = plan.create_session(
        {"scripts": {"sparse": SPARSE_PULSES}, "managed_state": ManagedState()}
    )
    directory = tmp_path / "rec"

    (clip,) = run_to_end(
        session.start, groups=("clips",), inputs={"capture_dir": str(directory)}
    )["clips"]

    layout = clip.outputs.layout["clip"]
    assert [axis.kind for axis in layout.axes] == ["sample", "time"]
    assert layout.axes[0].stationary
    assert plain(clip.outputs.data["clip"]) == [
        ((0,), [((0, 0), 10.0)]),
        ((1,), [((1, 1), 21.0)]),
        ((2,), [((2, 0), 12.0)]),
    ]
    metadata = clip.outputs.metadata["clip"]
    for camera in (0, 1, 2):
        assert metadata.sample_at((camera,)).source_id == f"cams/{camera}"
    recording = open_recording(
        directory, codecs=CodecRegistry(CATALOGUE.codecs.values())
    )
    (chunk,) = recording.group("clips").iter_chunks()
    assert plain(chunk.values["clip"]) == plain(clip.outputs.data["clip"])
    assert dict(chunk.metadata["clip"].sample) == dict(metadata.sample)
