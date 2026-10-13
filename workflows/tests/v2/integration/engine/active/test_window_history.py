"""Per-camera window history in active runs over a sparse camera set.

A scripted source emits, per pulse, only the cameras that delivered a frame,
plus two stationary crops per present camera. Two windows read it:

    cams.frame (camera)          ──┬─▶ by_camera  align=position, clear per camera
    cams.crop  (camera, crop)    ──┤   (hold reset: camera level)
    cams.reset (camera)          ──┤
    cams.reset_all (ungrouped)   ──┴─▶ whole_set  align=position, clear everything
                                       (hold reset_all: root level)

Both collect a depth-1 and a depth-2 layout, so their rows align at the
shared camera axis. Serial and pipelined runs must deliver the same windows.
"""

from fractions import Fraction
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest
from roboflow_workflows.v2.data.batch import (
    Axis,
    Batch,
    EntryLayout,
    EntryMetadata,
    InputValue,
    SampleContext,
    TemporalContext,
    Timestamp,
)
from roboflow_workflows.v2.data.kinds import BOOLEAN_KIND, FLOAT_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.operators.window import Window
from roboflow_workflows.v2.engine.pipelining.options import PipelineOptions
from roboflow_workflows.v2.engine.plan import CompileOptions
from roboflow_workflows.v2.engine.sources import Emission, Source, SourceOutput

WAIT = 10.0
"""Upper bound of every wait, in seconds; the tests finish far earlier."""

CAMERA = Axis(id="camera", kind="sample", stationary=True)
CROP = Axis(id="crop", kind="static_nesting", stationary=True)
CAMERAS = EntryLayout((CAMERA,))
CROPS = EntryLayout((CAMERA, CROP))

PULSES: List[Dict[int, bool]] = [
    {0: False, 1: False},
    {0: False},
    {0: False},
    {1: False},
    {0: False, 1: True},
    {0: False},
    {0: False, 1: False},
]
"""Per pulse: present camera -> its reset flag. Camera 1 is absent from
pulses 1, 2 and 5 and resets at pulse 4."""

RESET_ALL = {5}
"""Pulses whose root-level reset clears every camera of ``whole_set``."""


def frame_value(sequence: int, camera: int) -> float:
    return 10.0 * sequence + camera


class SparseCameraSet(Source):
    """Present cameras of each pulse, each with two fixed crops."""

    type = "test/sparse_camera_set@v1"
    outputs = {
        "frame": SourceOutput(FLOAT_KIND, layout=CAMERAS),
        "crop": SourceOutput(FLOAT_KIND, layout=CROPS),
        "reset": SourceOutput(BOOLEAN_KIND, layout=CAMERAS),
        "reset_all": SourceOutput(BOOLEAN_KIND),
    }

    def open(self) -> None:
        self.pulses: Iterator[Tuple[int, Dict[int, bool]]] = iter(enumerate(PULSES))

    def read(self) -> Optional[Emission]:
        current = next(self.pulses, None)
        if current is None:
            return None

        sequence, members = current
        indices = [(camera,) for camera in sorted(members)]
        sample, temporal = {}, {}
        for index in indices:
            sample[index] = SampleContext(f"set/{index[0]}", self.type)
            temporal[index] = TemporalContext(
                observed_coverage=Timestamp(sequence, Fraction(1), "test:pulse")
            )
        metadata = EntryMetadata(sample=sample, temporal=temporal)
        frames = [frame_value(sequence, index[0]) for index in indices]
        crops = [
            Batch(
                [frame_value(sequence, index[0]) + region / 10 for region in (0, 1)],
                indices=[index + (0,), index + (1,)],
                layout=CROPS,
                parent_index=index,
            )
            for index in indices
        ]
        resets = [members[index[0]] for index in indices]

        return Emission(
            {
                "frame": InputValue(Batch.of(frames, indices=indices), metadata),
                "crop": InputValue(Batch.of(crops, indices=indices), metadata),
                "reset": InputValue(Batch.of(resets, indices=indices), metadata),
                "reset_all": sequence in RESET_ALL,
            }
        )


CATALOGUE = Catalogue(
    [], sources=[SparseCameraSet], operators=[Window], kinds=[FLOAT_KIND, BOOLEAN_KIND]
)


def window(name: str, *, hold: str) -> Dict[str, Any]:
    return {
        "type": "v2/window@v1",
        "name": name,
        "size": 3,
        "align": "position",
        "fill": "grow",
        "order": "newest_first",
        "collect": {"frame": "$sources.cams.frame", "crop": "$sources.cams.crop"},
        "hold": {hold: f"$sources.cams.{hold}"},
        "clear_when": hold,
    }


def group(name: str, operator: str) -> Dict[str, Any]:
    return {
        "type": "OutputGroup",
        "name": name,
        "anchor": f"$operators.{operator}.frame",
        "outputs": [
            {
                "type": "JsonField",
                "name": "frames",
                "selector": f"$operators.{operator}.frame",
            },
            {
                "type": "JsonField",
                "name": "crops",
                "selector": f"$operators.{operator}.crop",
            },
        ],
    }


DEFINITION = {
    "version": "2.0",
    "inputs": [],
    "sources": [{"type": SparseCameraSet.type, "name": "cams"}],
    "operators": [
        window("by_camera", hold="reset"),
        window("whole_set", hold="reset_all"),
    ],
    "steps": [],
    "outputs": [group("by_camera", "by_camera"), group("whole_set", "whole_set")],
}

PIPELINED = PipelineOptions(max_in_flight=3, max_workflow_parallelism=2)


class SparseCollectFields(Source):
    """A new camera arrives on x while y is absent from the entire pulse."""

    type = "test/sparse_collect_fields@v1"
    outputs = {
        "x": SourceOutput(FLOAT_KIND, layout=CAMERAS),
        "y": SourceOutput(FLOAT_KIND, layout=CAMERAS),
    }

    def open(self) -> None:
        self.sequence = 0

    def read(self) -> Optional[Emission]:
        self.sequence += 1
        if self.sequence == 1:
            return Emission(
                {
                    "x": Batch.of([10.0], indices=[(1,)]),
                    "y": Batch.of([100.0], indices=[(1,)]),
                }
            )
        if self.sequence == 2:
            return Emission({"x": Batch.of([20.0, 21.0], indices=[(0,), (1,)])})

        return None


@pytest.mark.parametrize("mode", ["run", "phases"])
@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("fill", ["grow", "pad"])
def test_missing_collect_port_keeps_another_cameras_retained_value(
    mode: str, width: int, fill: str
) -> None:
    definition = {
        "version": "2.0",
        "inputs": [],
        "sources": [{"type": SparseCollectFields.type, "name": "cams"}],
        "steps": [],
        "operators": [
            {
                "type": Window.type,
                "name": "history",
                "size": 2,
                "align": "position",
                "fill": fill,
                "collect": {"x": "$sources.cams.x", "y": "$sources.cams.y"},
            }
        ],
        "outputs": [
            {
                "type": "OutputGroup",
                "name": "out",
                "anchor": "$operators.history.x",
                "outputs": [
                    {
                        "type": "JsonField",
                        "name": name,
                        "selector": f"$operators.history.{name}",
                    }
                    for name in ("x", "y")
                ],
            }
        ],
    }
    plan = compile_workflow(
        definition,
        catalogue=Catalogue([], sources=[SparseCollectFields], operators=[Window]),
        options=CompileOptions(block_execution=mode),
    )
    session = plan.create_session()
    delivered = []
    active = session.start(
        handlers={"out": delivered.append},
        pipeline=PipelineOptions(max_in_flight=width),
    )
    try:
        assert active.wait(WAIT)
        assert len(delivered) == 2
        rows = delivered[1].rows()
        assert rows[1] == {"x": [10.0, 21.0], "y": [100.0, None]}
        assert rows[0]["y"] == []
        assert rows[0]["x"] == ([20.0] if fill == "grow" else [None, 20.0])
    finally:
        active.stop()
        active.wait(WAIT)
        session.close()


def run(pipeline: Optional[PipelineOptions]) -> Tuple[Dict[str, List[Any]], Any]:
    plan = compile_workflow(DEFINITION, catalogue=CATALOGUE)
    delivered: Dict[str, List[Any]] = {"by_camera": [], "whole_set": []}
    active = plan.create_session().start(
        handlers={name: results.append for name, results in delivered.items()},
        pipeline=pipeline,
    )
    try:
        assert active.wait(WAIT)
    finally:
        if not active.done:
            active.cancel()
            active.wait(WAIT)

    return delivered, active


def leaves(value: Any, prefix: Tuple[int, ...] = ()) -> Iterator[Tuple[tuple, Any]]:
    """Every ``(full index, payload)`` of a nested batch."""
    for index, member in value.iter_with_indices():
        if isinstance(member, Batch):
            yield from leaves(member, index)
        else:
            yield index, member


def histories(results: List[Any], name: str = "frames") -> List[Dict[tuple, Any]]:
    """Each delivered window as ``{full index: payload}``."""
    rendered = [dict(leaves(result.outputs.data[name])) for result in results]

    return rendered


@pytest.fixture(scope="module")
def serial() -> Tuple[Dict[str, List[Any]], Any]:
    return run(None)


def test_absent_cameras_keep_history_and_resets_follow_their_hold_level(
    serial,
) -> None:
    delivered, _ = serial

    by_camera = histories(delivered["by_camera"])
    assert by_camera[3] == {(1, 0): 31.0, (1, 1): 1.0}
    assert by_camera[4] == {(0, 0): 40.0, (0, 1): 20.0, (0, 2): 10.0, (1, 0): 41.0}
    assert by_camera[6] == {
        (0, 0): 60.0,
        (0, 1): 50.0,
        (0, 2): 40.0,
        (1, 0): 61.0,
        (1, 1): 41.0,
    }

    whole_set = histories(delivered["whole_set"])
    assert whole_set[4][(1, 0)] == 41.0 and whole_set[4][(1, 2)] == 1.0
    assert whole_set[5] == {(0, 0): 50.0}
    assert whole_set[6] == {(0, 0): 60.0, (0, 1): 50.0, (1, 0): 61.0}


def test_depth_two_crops_align_with_their_camera(serial) -> None:
    delivered, _ = serial

    crops = histories(delivered["by_camera"], "crops")[3]

    assert crops == {(1, 0, 0): 31.0, (1, 0, 1): 1.0, (1, 1, 0): 31.1, (1, 1, 1): 1.1}


def test_every_member_keeps_the_context_of_its_own_pulse(serial) -> None:
    delivered, _ = serial
    result = delivered["by_camera"][6]
    metadata = result.outputs.metadata["frames"]

    for index, value in leaves(result.outputs.data["frames"]):
        camera = index[0]
        sequence = int(value - camera) // 10
        assert metadata.sample_at(index).source_id == f"set/{camera}"
        assert metadata.temporal_at(index).observed_coverage.ticks == sequence


def test_pipelined_runs_deliver_the_serial_windows_with_bounded_emission(
    serial,
) -> None:
    delivered, active = serial
    piped, piped_active = run(PIPELINED)

    for name in delivered:
        assert histories(piped[name]) == histories(delivered[name])
        assert histories(piped[name], "crops") == histories(delivered[name], "crops")
    for counters_of in (active, piped_active):
        for name, counters in counters_of.operator_counters.items():
            assert counters.emitted <= len(PULSES)
            assert counters.emitted == counters.processed == counters.delivered
            assert counters.emitted == len(delivered[name])
            assert counters.peak_retained <= 3 * 2
    assert {n: vars(c) for n, c in active.operator_counters.items()} == {
        n: vars(c) for n, c in piped_active.operator_counters.items()
    }
    gauges = piped_active.pipeline_counters
    operators = len(DEFINITION["operators"])
    assert gauges.peak("pending_operator_pulses") <= operators  # one each at most
    assert gauges.current("pending_operator_pulses") == 0
