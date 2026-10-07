"""Engine-owned capture: what an active run records, and how a recording ends.

The fixtures here (scripted source, model-call tripwire, workflow builders)
are shared by the replay and Python stage tests of this package.
"""

import threading
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.data import Batch, Timestamp
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
    Select,
    StepRef,
    Stop,
)
from roboflow_workflows.execution_engine.v2.errors import ActiveRunError
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND, INTEGER_KIND
from roboflow_workflows.execution_engine.v2.operators.alignment import Align
from roboflow_workflows.execution_engine.v2.operators.window import Window
from roboflow_workflows.execution_engine.v2.pipelining.options import PipelineOptions
from roboflow_workflows.execution_engine.v2.recording.codecs import CodecRegistry
from roboflow_workflows.execution_engine.v2.recording.errors import (
    RecordingExistsError,
    RecordingIncompleteError,
)
from roboflow_workflows.execution_engine.v2.recording.store import (
    inspect_recording,
    open_recording,
)
from roboflow_workflows.execution_engine.v2.sources import (
    Emission,
    Source,
    SourceOutput,
    SourceParams,
)

from tests.unit_tests.execution_engine.v2.execution.blocks import (
    ContinueIf,
    Counter,
    Echo,
    Expand,
)

WAIT = 10.0
MEDIA = Fraction(1, 1000)
TRIPWIRE: Dict[str, int] = {}
"""Constructor and call counts of the primary workflow's model and source."""


def media(milliseconds: int) -> Timestamp:
    return Timestamp(milliseconds, MEDIA, "media")


class Feed(Source):
    """Emits a scripted list of emissions; ``threading.Event`` items block."""

    type = "test/recorded_feed@v1"
    outputs = {"value": SourceOutput(FLOAT_KIND)}

    class Params(SourceParams):
        feed: str

    def __init__(self, *, feeds: Dict[str, list]):
        TRIPWIRE["source_constructed"] = TRIPWIRE.get("source_constructed", 0) + 1
        self.feeds = feeds

    def open(self, *, feed) -> None:
        self.items = iter(self.feeds[feed])

    def read(self) -> Optional[Emission]:
        item = next(self.items, None)
        while isinstance(item, threading.Event):
            item.wait(WAIT)
            item = next(self.items, None)
        if isinstance(item, Exception):
            raise item

        return item


class Model(Block):
    """Stands in for an expensive model: counts constructions and calls."""

    type = "test/recorded_model@v1"
    outputs = {"score": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def __init__(self) -> None:
        TRIPWIRE["model_constructed"] = TRIPWIRE.get("model_constructed", 0) + 1

    def run(self, *, value) -> dict:
        TRIPWIRE["model_calls"] = TRIPWIRE.get("model_calls", 0) + 1
        return {"score": value / 10}


class Unrecordable(Block):
    """Outputs a payload type no recording codec handles."""

    type = "test/unrecordable@v1"
    outputs = {"thing": Output()}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, *, value) -> dict:
        return {"thing": object()}


class MutateList(Block):
    """Appends to a list payload in place (a later causal mutator)."""

    type = "test/mutate_list@v1"
    outputs = {"items": Output()}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, *, value) -> dict:
        return {"items": [value]}


class Nothing(Block):
    """Outputs ``None``: a delivered payload, neither empty nor filtered."""

    type = "test/nothing@v1"
    outputs = {"value": Output()}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, *, value) -> dict:
        return {"value": None}


class Above(Block):
    """Control block: continue when ``value > threshold``; the threshold may be selected."""

    type = "test/recorded_above@v1"

    class Params(BlockParams):
        value: Ref()
        threshold: float | Ref(FLOAT_KIND) = 0.0
        next_steps: List[StepRef]

    def run(self, *, value, threshold, next_steps):
        return Select(next_steps) if value > threshold else Stop()


class StateTally(Block):
    """Counts calls in the session's managed state."""

    type = "test/state_tally@v1"
    outputs = {"tally": Output(INTEGER_KIND)}

    class Params(BlockParams):
        value: Ref()

    def __init__(self, *, managed_state) -> None:
        self.state = managed_state

    def run(self, *, value) -> dict:
        return {"tally": self.state.global_.incr("seen")}


CATALOGUE = Catalogue(
    [
        Above,
        ContinueIf,
        Counter,
        Echo,
        Expand,
        Model,
        Unrecordable,
        MutateList,
        Nothing,
        StateTally,
    ],
    sources=[Feed],
    operators=[Align, Window],
)


@pytest.fixture(autouse=True)
def _reset_tripwire():
    TRIPWIRE.clear()
    yield


def field(name: str, selector: str) -> Dict[str, str]:
    return {"type": "JsonField", "name": name, "selector": selector}


def group(name: str, anchor: str, *fields: Dict[str, str]) -> Dict[str, Any]:
    return {
        "type": "OutputGroup",
        "name": name,
        "anchor": anchor,
        "outputs": list(fields),
    }


def primary_definition(**sections: Any) -> Dict[str, Any]:
    """Feed ``cam`` -> model, gated children with partial filtering, a window.

    Groups: ``frames`` (value, score, children: gated per pulse, kept:
    filtered per child, empty: always an empty group) and ``clips`` (a
    two-pulse window of values, a time axis).
    """
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "capture_dir"}],
        "sources": [{"type": "test/recorded_feed@v1", "name": "cam", "feed": "cam"}],
        "operators": [
            {
                "type": "v2/window@v1",
                "name": "w",
                "size": 2,
                "collect": {"v": "$sources.cam.value"},
            }
        ],
        "steps": [
            {
                "type": "test/recorded_model@v1",
                "name": "model",
                "value": "$sources.cam.value",
            },
            {
                "type": "test/continue_if@v1",
                "name": "gate",
                "value": "$sources.cam.value",
                "threshold": 1.5,
                "next_steps": ["$steps.expand"],
            },
            {
                "type": "test/expand@v1",
                "name": "expand",
                "value": "$sources.cam.value",
                "count": 3,
            },
            {
                "type": "test/continue_if@v1",
                "name": "keep",
                "value": "$steps.expand.children",
                "threshold": 2.5,
                "next_steps": ["$steps.kept"],
            },
            {"type": "test/echo@v1", "name": "kept", "value": "$steps.expand.children"},
            {
                "type": "test/expand@v1",
                "name": "none",
                "value": "$sources.cam.value",
                "count": 0,
            },
        ],
        "outputs": [
            group(
                "frames",
                "$sources.cam.value",
                field("value", "$sources.cam.value"),
                field("score", "$steps.model.score"),
                field("children", "$steps.expand.children"),
                field("kept", "$steps.kept.value"),
                field("empty", "$steps.none.children"),
            ),
            group("clips", "$operators.w.v", field("clip", "$operators.w.v")),
        ],
        "recording": {
            "type": "file",
            "directory": "$inputs.capture_dir",
            "groups": ["frames", "clips"],
        },
    }
    definition.update(sections)

    return definition


def feed_values(*values: Optional[float]) -> List[Emission]:
    """One emission per value at 100 ms steps; ``None`` is a filtered pulse."""
    emissions = []
    for position, value in enumerate(values):
        data = {} if value is None else {"value": value}
        emissions.append(
            Emission(data, media=media(100 * position), source_metadata={"n": position})
        )

    return emissions


def capture(
    plan,
    directory: Path,
    feeds: Dict[str, list],
    *,
    handlers: Optional[Dict[str, Any]] = None,
    pipeline: Optional[PipelineOptions] = None,
):
    session = plan.create_session({"feeds": feeds})
    run = session.start(
        {"capture_dir": str(directory)}, handlers=handlers or {}, pipeline=pipeline
    )
    run.wait(WAIT)

    return run


def read_chunks(directory: Path, group_name: str, *, catalogue: Catalogue = CATALOGUE):
    recording = open_recording(
        directory, codecs=CodecRegistry(catalogue.codecs.values())
    )
    chunks = list(recording.group(group_name).iter_chunks())

    return chunks


def comparable(chunk) -> Dict[str, Any]:
    """A chunk without its run identity, for serial/pipelined comparison."""
    return {
        "index": chunk.index,
        "pulse": (chunk.pulse.source, chunk.pulse.sequence),
        "causes": [(cause.source, cause.sequence) for cause in chunk.causes],
        "statuses": dict(chunk.statuses),
        "filtered_paths": dict(chunk.filtered_paths),
        "values": {key: _plain(value) for key, value in chunk.values.items()},
        "sample": {
            key: dict(metadata.sample) for key, metadata in chunk.metadata.items()
        },
        "media": {
            key: {
                index: None if temporal is None else temporal.media_coverage
                for index, temporal in metadata.temporal.items()
            }
            for key, metadata in chunk.metadata.items()
        },
    }


def _plain(value: Any) -> Any:
    if isinstance(value, Batch):
        return [(index, _plain(item)) for index, item in value.iter_with_indices()]

    return value


def test_capture_records_exactly_the_delivered_group_results(tmp_path: Path) -> None:
    plan = compile_workflow(primary_definition(), catalogue=CATALOGUE)
    delivered = []

    run = capture(
        plan,
        tmp_path / "rec",
        {"cam": feed_values(1.0, 2.0, None, 3.0)},
        handlers={"frames": delivered.append},
    )

    assert run.state == "finished"
    chunks = read_chunks(tmp_path / "rec", "frames")
    assert [chunk.index for chunk in chunks] == [0, 1, 2, 3]
    for result, chunk in zip(delivered, chunks):
        assert chunk.pulse.sequence == result.pulse.sequence
        assert chunk.pulse.active_run_id == run.run_id
        assert dict(chunk.statuses) == dict(result.statuses)
        assert dict(chunk.filtered_paths) == dict(result.filtered_paths)
        for key in result.outputs.data:
            assert _plain(chunk.values[key]) == _plain(result.outputs.data[key])
            assert dict(chunk.metadata[key].sample) == dict(
                result.outputs.metadata[key].sample
            )
    # pulse 0: gate filtered the children; pulse 2: an explicitly filtered pulse;
    # pulse 1: children 2, 3, 4 with "kept" filtered at child 0 (2.0 <= 2.5).
    assert chunks[0].statuses["children"] == "filtered"
    assert chunks[2].is_filtered
    assert chunks[1].filtered_paths["kept"] == ((0,),)
    assert _plain(chunks[1].values["kept"]) == [((1,), 3.0), ((2,), 4.0)]
    assert _plain(chunks[1].values["empty"]) == []
    assert chunks[1].metadata["value"].sample[()].source_metadata == {"n": 1}
    assert chunks[1].metadata["value"].temporal[()].media_coverage == media(100)


def test_a_recorded_group_needs_no_handler_and_handlers_still_get_every_result(
    tmp_path: Path,
) -> None:
    plan = compile_workflow(primary_definition(), catalogue=CATALOGUE)
    clips = []

    run = capture(
        plan,
        tmp_path / "rec",
        {"cam": feed_values(1.0, 2.0, 3.0, 4.0)},
        handlers={"clips": clips.append},
    )

    assert run.state == "finished"
    assert len(read_chunks(tmp_path / "rec", "frames")) == 4
    clip_chunks = read_chunks(tmp_path / "rec", "clips")
    assert len(clips) == len(clip_chunks) == 2
    assert clip_chunks[0].layouts["clip"].has_time
    assert _plain(clip_chunks[1].values["clip"]) == [((0,), 3.0), ((1,), 4.0)]
    assert run.recording_counters["frames"]["chunks"] == 4
    assert run.recording_counters["clips"]["chunks"] == 2
    assert run.counters["cam"].delivered == 4


def test_recording_snapshots_the_value_before_the_handler_mutates_it(
    tmp_path: Path,
) -> None:
    definition = primary_definition(
        steps=[
            {"type": "test/mutate_list@v1", "name": "m", "value": "$sources.cam.value"}
        ],
        operators=[],
        outputs=[
            group("items", "$sources.cam.value", field("items", "$steps.m.items"))
        ],
        recording={
            "type": "file",
            "directory": "$inputs.capture_dir",
            "groups": ["items"],
        },
    )
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    def mutate(result) -> None:
        result.outputs.data["items"].append("changed by the handler")

    capture(
        plan, tmp_path / "rec", {"cam": feed_values(1.0)}, handlers={"items": mutate}
    )

    assert read_chunks(tmp_path / "rec", "items")[0].values["items"] == [1.0]


def test_serial_and_pipelined_capture_record_identical_chunks(tmp_path: Path) -> None:
    plan = compile_workflow(primary_definition(), catalogue=CATALOGUE)
    values = feed_values(1.0, 2.0, None, 3.0, 4.0, 0.5)

    capture(plan, tmp_path / "serial", {"cam": values})
    pipelined = capture(
        plan,
        tmp_path / "pipelined",
        {"cam": values},
        pipeline=PipelineOptions(max_in_flight=3),
    )

    assert pipelined.state == "finished"
    for name in ("frames", "clips"):
        serial_chunks = [comparable(c) for c in read_chunks(tmp_path / "serial", name)]
        piped_chunks = [
            comparable(c) for c in read_chunks(tmp_path / "pipelined", name)
        ]
        assert serial_chunks == piped_chunks
        assert serial_chunks


def test_the_recording_is_final_before_wait_returns(tmp_path: Path) -> None:
    plan = compile_workflow(primary_definition(), catalogue=CATALOGUE)

    capture(plan, tmp_path / "rec", {"cam": feed_values(1.0, 2.0)})

    info = inspect_recording(tmp_path / "rec")
    assert info.status == "complete"


def test_stop_finishes_a_replayable_stopped_recording(tmp_path: Path) -> None:
    plan = compile_workflow(primary_definition(), catalogue=CATALOGUE)
    gate = threading.Event()
    session = plan.create_session(
        {"feeds": {"cam": [*feed_values(1.0, 2.0), gate, *feed_values(3.0)]}}
    )
    seen = []

    def on_frames(result) -> None:
        seen.append(result.pulse.sequence)
        if len(seen) == 2:
            session.stop()
            gate.set()

    run = session.start(
        {"capture_dir": str(tmp_path / "rec")}, handlers={"frames": on_frames}
    )
    run.wait(WAIT)

    assert run.state == "finished"
    assert inspect_recording(tmp_path / "rec").status == "stopped"
    assert [chunk.index for chunk in read_chunks(tmp_path / "rec", "frames")] == [0, 1]


def test_stop_is_recorded_when_the_source_then_returns_its_end(tmp_path: Path) -> None:
    plan = compile_workflow(primary_definition(), catalogue=CATALOGUE)
    gate = threading.Event()
    session = plan.create_session({"feeds": {"cam": [*feed_values(1.0), gate]}})

    def on_frames(result) -> None:
        session.stop()
        gate.set()  # the blocked read then returns None, like a cooperative camera

    run = session.start(
        {"capture_dir": str(tmp_path / "rec")}, handlers={"frames": on_frames}
    )
    run.wait(WAIT)

    assert run.counters["cam"].ended
    assert inspect_recording(tmp_path / "rec").status == "stopped"


def test_a_failed_run_finishes_a_failed_recording_that_replay_rejects(
    tmp_path: Path,
) -> None:
    plan = compile_workflow(primary_definition(), catalogue=CATALOGUE)
    feeds = {"cam": [*feed_values(1.0), RuntimeError("camera unplugged")]}

    with pytest.raises(ActiveRunError, match="camera unplugged"):
        capture(plan, tmp_path / "rec", feeds)

    info = inspect_recording(tmp_path / "rec")
    assert info.status == "failed"
    assert "camera unplugged" in info.error
    with pytest.raises(RecordingIncompleteError):
        open_recording(tmp_path / "rec", codecs=CodecRegistry())


def test_cancel_finishes_a_cancelled_recording(tmp_path: Path) -> None:
    plan = compile_workflow(primary_definition(), catalogue=CATALOGUE)
    gate = threading.Event()
    session = plan.create_session({"feeds": {"cam": [*feed_values(1.0), gate]}})
    run = session.start({"capture_dir": str(tmp_path / "rec")})

    run.cancel()
    gate.set()

    assert run.wait(WAIT)
    assert run.state == "cancelled"
    assert inspect_recording(tmp_path / "rec").status == "cancelled"


def test_an_unrecordable_payload_fails_the_run_at_the_recording_stage(
    tmp_path: Path,
) -> None:
    definition = primary_definition(
        steps=[
            {"type": "test/unrecordable@v1", "name": "u", "value": "$sources.cam.value"}
        ],
        operators=[],
        outputs=[group("odd", "$sources.cam.value", field("thing", "$steps.u.thing"))],
        recording={
            "type": "file",
            "directory": "$inputs.capture_dir",
            "groups": ["odd"],
        },
    )
    plan = compile_workflow(definition, catalogue=CATALOGUE)
    handled = []

    with pytest.raises(ActiveRunError) as raised:
        capture(
            plan,
            tmp_path / "rec",
            {"cam": feed_values(1.0)},
            handlers={"odd": handled.append},
        )

    assert raised.value.stage == "recording"
    assert raised.value.group == "odd"
    assert "thing" in str(raised.value) and "object" in str(raised.value)
    assert handled == []
    assert inspect_recording(tmp_path / "rec").status == "failed"


def test_an_existing_destination_fails_start_before_any_source_is_constructed(
    tmp_path: Path,
) -> None:
    plan = compile_workflow(primary_definition(), catalogue=CATALOGUE)
    capture(plan, tmp_path / "rec", {"cam": feed_values(1.0)})
    constructed = TRIPWIRE["source_constructed"]
    session = plan.create_session({"feeds": {"cam": feed_values(2.0)}})

    with pytest.raises(RecordingExistsError):
        session.start({"capture_dir": str(tmp_path / "rec")})

    assert TRIPWIRE["source_constructed"] == constructed + 1  # built, never opened
    assert inspect_recording(tmp_path / "rec").status == "complete"
    rerun = session.start({"capture_dir": str(tmp_path / "second")})
    assert rerun.wait(WAIT)


def test_a_plan_without_recording_records_nothing(tmp_path: Path) -> None:
    definition = primary_definition()
    del definition["recording"]
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    run = capture(plan, tmp_path / "rec", {"cam": feed_values(1.0)})

    assert plan.recording is None and plan.retrospective is None
    assert "recording" not in plan.describe() and "retrospective" not in plan.describe()
    assert run.recording_counters == {}
    assert not (tmp_path / "rec").exists()


def test_multi_source_group_records_causes_and_every_member_context(
    tmp_path: Path,
) -> None:
    plan = compile_workflow(multi_source_definition(), catalogue=CATALOGUE)
    session = plan.create_session(
        {"feeds": {"left": feed_values(1.0, 2.0), "right": feed_values(10.0, 20.0)}}
    )

    run = session.start({"capture_dir": str(tmp_path / "rec")})
    run.wait(WAIT)

    chunks = read_chunks(tmp_path / "rec", "pairs")
    assert len(chunks) == 2
    assert {cause.source for cause in chunks[0].causes} == {"left", "right"}
    samples = chunks[0].metadata["left"].sample
    assert {context.source_id for context in samples.values() if context} == {"left"}
    samples = chunks[0].metadata["right"].sample
    assert {context.source_id for context in samples.values() if context} == {"right"}


def multi_source_definition(**sections: Any) -> Dict[str, Any]:
    """Two feeds aligned on media time; group ``pairs`` holds both members."""
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "capture_dir"}],
        "sources": [
            {"type": "test/recorded_feed@v1", "name": "left", "feed": "left"},
            {"type": "test/recorded_feed@v1", "name": "right", "feed": "right"},
        ],
        "operators": [
            {
                "type": "v2/align@v1",
                "name": "pair",
                "clock": "media",
                "tolerance_ms": 10,
                "inputs": {
                    "left": "$sources.left.value",
                    "right": "$sources.right.value",
                },
            }
        ],
        "steps": [],
        "outputs": [
            group(
                "pairs",
                "$operators.pair.left",
                field("left", "$operators.pair.left"),
                field("right", "$operators.pair.right"),
            )
        ],
        "recording": {
            "type": "file",
            "directory": "$inputs.capture_dir",
            "groups": ["pairs"],
        },
    }
    definition.update(sections)

    return definition
