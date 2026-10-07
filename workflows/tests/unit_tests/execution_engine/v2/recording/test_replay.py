"""Normal replay: a retrospective workflow over one recorded group.

Replays must reproduce what the primary run delivered (layouts, filtered and
empty positions, time axes, original contexts), start fresh every time and
never touch the primary workflow's model, sources or resources.
"""

import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    WorkflowInputError,
)
from roboflow_workflows.execution_engine.v2.pipelining.options import PipelineOptions
from roboflow_workflows.execution_engine.v2.recording.errors import (
    RecordingDefinitionError,
    RecordingIncompleteError,
    RecordingSchemaError,
)
from roboflow_workflows.execution_engine.v2.state.api import ManagedState

from tests.unit_tests.execution_engine.v2.recording.test_capture_run import (
    CATALOGUE,
    TRIPWIRE,
    WAIT,
    _plain,
    _reset_tripwire,
    capture,
    feed_values,
    field,
    group,
    multi_source_definition,
    primary_definition,
    read_chunks,
)

FRAME_FIELDS = ("value", "score", "children", "kept", "empty")
__all__ = ["_reset_tripwire"]  # the autouse fixture applies here too


def with_retrospective(
    definition: Dict[str, Any], workflow: Dict[str, Any], *, input_group: str = "frames"
) -> Dict[str, Any]:
    definition = dict(definition)
    definition["retrospective"] = {
        "type": "workflow",
        "input_group": input_group,
        "workflow": workflow,
    }

    return definition


def passthrough_workflow() -> Dict[str, Any]:
    """Every recorded frame field, unchanged, plus a per-replay call counter."""
    return {
        "steps": [
            {
                "type": "test/counter@v1",
                "name": "count",
                "value": "$sources.frames.value",
            }
        ],
        "outputs": [
            group(
                "reviewed",
                "$sources.frames.value",
                *(field(name, f"$sources.frames.{name}") for name in FRAME_FIELDS),
                field("count", "$steps.count.count"),
            )
        ],
    }


def threshold_workflow() -> Dict[str, Any]:
    """Keeps recorded scores above a retrospective ``minimum`` input."""
    return {
        "inputs": [
            {"type": "WorkflowParameter", "name": "minimum", "default_value": 0.0}
        ],
        "steps": [
            {
                "type": "test/recorded_above@v1",
                "name": "gate",
                "value": "$sources.frames.score",
                "threshold": "$inputs.minimum",
                "next_steps": ["$steps.keep"],
            },
            {"type": "test/echo@v1", "name": "keep", "value": "$sources.frames.score"},
        ],
        "outputs": [
            group(
                "reviewed", "$sources.frames.value", field("score", "$steps.keep.value")
            )
        ],
    }


def replay(plan, directory: Path, **options: Any) -> List[Any]:
    results: List[Any] = []
    run = plan.retrospective.start(
        recording=directory, handlers={"reviewed": results.append}, **options
    )
    assert run.wait(WAIT)

    return results


@pytest.fixture()
def recorded(tmp_path: Path):
    """A finished recording of five pulses, and the results delivered live."""
    plan = compile_workflow(
        with_retrospective(primary_definition(), passthrough_workflow()),
        catalogue=CATALOGUE,
    )
    delivered: List[Any] = []
    capture(
        plan,
        tmp_path / "rec",
        {"cam": feed_values(1.0, 2.0, None, 3.0, 0.5)},
        handlers={"frames": delivered.append},
    )

    return plan, tmp_path / "rec", delivered


def test_replay_reproduces_layouts_filtered_and_empty_positions_and_context(
    recorded,
) -> None:
    plan, directory, delivered = recorded

    replayed = replay(plan, directory)

    assert [result.pulse.sequence for result in replayed] == [0, 1, 2, 3, 4]
    assert {result.pulse.source for result in replayed} == {"frames"}
    for original, result in zip(delivered, replayed):
        assert result.pulse.active_run_id != original.pulse.active_run_id
        for name in FRAME_FIELDS:
            assert result.statuses[name] == original.statuses[name], name
            assert result.filtered_paths[name] == original.filtered_paths[name], name
            if original.statuses[name] != "complete":
                continue
            assert _plain(result.outputs.data[name]) == _plain(
                original.outputs.data[name]
            )
            assert dict(result.outputs.metadata[name].sample) == dict(
                original.outputs.metadata[name].sample
            )
            assert dict(result.outputs.metadata[name].temporal) == dict(
                original.outputs.metadata[name].temporal
            )
            assert (
                result.outputs.layout[name].depth == original.outputs.layout[name].depth
            )
    assert replayed[2].is_filtered
    assert replayed[1].filtered_paths["kept"] == ((0,),)
    sample = replayed[1].outputs.metadata["value"].sample[()]
    assert (sample.source_id, sample.source_metadata) == ("cam", {"n": 1})


def test_replay_never_constructs_or_calls_the_primary_model_or_source(recorded) -> None:
    plan, directory, _ = recorded
    before = dict(TRIPWIRE)

    replay(plan, directory)
    replay(plan, directory, pipeline=PipelineOptions(max_in_flight=2))

    assert TRIPWIRE == before
    assert before["model_calls"] == 4  # the filtered pulse never reached the model


def test_changed_retrospective_parameters_change_results_without_rerunning_capture(
    tmp_path: Path,
) -> None:
    plan = compile_workflow(
        with_retrospective(primary_definition(), threshold_workflow()),
        catalogue=CATALOGUE,
    )
    capture(plan, tmp_path / "rec", {"cam": feed_values(1.0, 2.0, 3.0)})
    calls = TRIPWIRE["model_calls"]

    def kept(minimum: float) -> List[str]:
        results: List[Any] = []
        run = plan.retrospective.start(
            recording=tmp_path / "rec",
            inputs={"minimum": minimum},
            handlers={"reviewed": results.append},
        )
        run.wait(WAIT)
        return [result.statuses["score"] for result in results]

    assert kept(0.15) == ["filtered", "complete", "complete"]
    assert kept(0.25) == ["filtered", "filtered", "complete"]
    assert TRIPWIRE["model_calls"] == calls


def test_each_replay_starts_fresh_block_and_managed_state(tmp_path: Path) -> None:
    workflow = {
        "steps": [
            {
                "type": "test/counter@v1",
                "name": "count",
                "value": "$sources.frames.value",
            },
            {
                "type": "test/state_tally@v1",
                "name": "tally",
                "value": "$sources.frames.value",
            },
        ],
        "outputs": [
            group(
                "reviewed",
                "$sources.frames.value",
                field("count", "$steps.count.count"),
                field("tally", "$steps.tally.tally"),
            )
        ],
    }
    plan = compile_workflow(
        with_retrospective(primary_definition(), workflow), catalogue=CATALOGUE
    )
    capture(plan, tmp_path / "rec", {"cam": feed_values(1.0, 2.0, 3.0)})

    first = replay(plan, tmp_path / "rec")
    second = replay(plan, tmp_path / "rec", pipeline=PipelineOptions(max_in_flight=2))

    for results in (first, second):
        assert [result.outputs.data["count"] for result in results] == [1, 2, 3]
        assert [result.outputs.data["tally"] for result in results] == [1, 2, 3]


def test_replay_rejects_borrowed_state_without_touching_it(recorded) -> None:
    plan, directory, _ = recorded
    caller_state = ManagedState()
    caller_state.global_.set("kept", 7)

    for key in ("managed_state", "steps.managed_state", "workflows_v2_recording"):
        with pytest.raises(ContractError, match="cannot be passed to a replay"):
            plan.retrospective.start(recording=directory, resources={key: caller_state})

    assert caller_state.global_.get("kept") == 7


def test_operators_run_over_replayed_frames_and_time_axes_replay(
    tmp_path: Path,
) -> None:
    window_workflow = {
        "operators": [
            {
                "type": "v2/window@v1",
                "name": "w",
                "size": 2,
                "collect": {"v": "$sources.frames.value"},
            }
        ],
        "outputs": [
            group("reviewed", "$operators.w.v", field("clip", "$operators.w.v"))
        ],
    }
    plan = compile_workflow(
        with_retrospective(primary_definition(), window_workflow), catalogue=CATALOGUE
    )
    capture(plan, tmp_path / "rec", {"cam": feed_values(1.0, 2.0, 3.0, 4.0)})
    recorded_clips = [
        _plain(c.values["clip"]) for c in read_chunks(tmp_path / "rec", "clips")
    ]

    windows = [
        _plain(result.outputs.data["clip"]) for result in replay(plan, tmp_path / "rec")
    ]

    assert (
        windows
        == recorded_clips
        == [[((0,), 1.0), ((1,), 2.0)], [((0,), 3.0), ((1,), 4.0)]]
    )

    clip_workflow = {
        "steps": [
            {"type": "test/echo@v1", "name": "echo", "value": "$sources.clips.clip"}
        ],
        "outputs": [
            group(
                "reviewed",
                "$sources.clips.clip",
                field("clip", "$sources.clips.clip"),
                field("echo", "$steps.echo.value"),
            )
        ],
    }
    clip_plan = compile_workflow(
        with_retrospective(primary_definition(), clip_workflow, input_group="clips"),
        catalogue=CATALOGUE,
    )
    replayed = replay(clip_plan, tmp_path / "rec")

    assert [
        _plain(result.outputs.data["clip"]) for result in replayed
    ] == recorded_clips
    assert all(result.outputs.layout["clip"].has_time for result in replayed)
    assert [
        _plain(result.outputs.data["echo"]) for result in replayed
    ] == recorded_clips
    assert dict(replayed[1].outputs.metadata["clip"].temporal) == dict(
        read_chunks(tmp_path / "rec", "clips")[1].metadata["clip"].temporal
    )


def test_ordinary_sources_still_cannot_declare_time_axes() -> None:
    from roboflow_workflows.execution_engine.v2.data import Axis, EntryLayout
    from roboflow_workflows.execution_engine.v2.sources import (
        SourceDeclarationError,
        SourceOutput,
    )

    with pytest.raises(SourceDeclarationError, match="time axis"):
        SourceOutput(layout=EntryLayout((Axis("t", "time"),)))


def test_multi_source_group_replays_with_both_sources_contexts(tmp_path: Path) -> None:
    workflow = {
        "outputs": [
            group(
                "reviewed",
                "$sources.pairs.left",
                field("left", "$sources.pairs.left"),
                field("right", "$sources.pairs.right"),
            )
        ]
    }
    plan = compile_workflow(
        with_retrospective(multi_source_definition(), workflow, input_group="pairs"),
        catalogue=CATALOGUE,
    )
    session = plan.create_session(
        {"feeds": {"left": feed_values(1.0, 2.0), "right": feed_values(10.0, 20.0)}}
    )
    delivered: List[Any] = []
    session.start(
        {"capture_dir": str(tmp_path / "rec")}, handlers={"pairs": delivered.append}
    ).wait(WAIT)

    replayed = replay(plan, tmp_path / "rec")

    assert len(replayed) == len(delivered) == 2
    for original, result in zip(delivered, replayed):
        for name in ("left", "right"):
            assert _plain(result.outputs.data[name]) == _plain(
                original.outputs.data[name]
            )
            assert dict(result.outputs.metadata[name].sample) == dict(
                original.outputs.metadata[name].sample
            )


def test_a_changed_primary_schema_is_rejected_before_anything_is_built(
    recorded,
) -> None:
    _, directory, _ = recorded
    changed = primary_definition()
    changed["outputs"][0]["outputs"].append(field("extra", "$sources.cam.value"))
    plan = compile_workflow(
        with_retrospective(changed, passthrough_workflow()), catalogue=CATALOGUE
    )
    before = dict(TRIPWIRE)

    with pytest.raises(RecordingSchemaError, match="extra"):
        plan.retrospective.start(recording=directory)

    assert TRIPWIRE == before


def test_an_unfinished_recording_is_rejected(tmp_path: Path) -> None:
    plan = compile_workflow(
        with_retrospective(primary_definition(), passthrough_workflow()),
        catalogue=CATALOGUE,
    )
    with pytest.raises(Exception, match="unplugged"):
        capture(
            plan,
            tmp_path / "rec",
            {"cam": [*feed_values(1.0), RuntimeError("unplugged")]},
        )

    with pytest.raises(RecordingIncompleteError) as raised:
        plan.retrospective.start(recording=tmp_path / "rec")

    assert raised.value.status == "failed"


def test_root_inputs_locate_the_recording_and_stay_separate_from_stage_inputs(
    tmp_path: Path,
) -> None:
    plan = compile_workflow(
        with_retrospective(primary_definition(), threshold_workflow()),
        catalogue=CATALOGUE,
    )
    capture(plan, tmp_path / "rec", {"cam": feed_values(1.0, 3.0)})
    results: List[Any] = []

    run = plan.retrospective.start(
        root_inputs={"capture_dir": str(tmp_path / "rec")},
        inputs={"minimum": 0.2},
        handlers={"reviewed": results.append},
    )
    run.wait(WAIT)

    assert [result.statuses["score"] for result in results] == ["filtered", "complete"]
    with pytest.raises(
        ContractError,
        match=r"resolved to None; pass recording=<dir> or "
        r"root_inputs=\{'capture_dir': <dir>\}",
    ):
        plan.retrospective.start()
    overridden = plan.retrospective.start(
        recording=tmp_path / "rec",
        root_inputs={"capture_dir": str(tmp_path / "missing")},
    )
    assert overridden.wait(WAIT) and overridden.failure is None
    with pytest.raises(RecordingDefinitionError, match="minimum"):
        plan.retrospective.start(root_inputs={"minimum": 0.2})


def test_a_python_stage_cannot_be_started_as_a_workflow(recorded) -> None:
    _, directory, _ = recorded
    definition = primary_definition()
    definition["retrospective"] = {
        "type": "python",
        "input_group": "frames",
        "access": "all",
        "result_directory": str(directory.parent / "results"),
    }
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    with pytest.raises(ContractError, match="run_python"):
        plan.retrospective.start(recording=directory)


def test_an_ordinary_source_payload_is_never_read_as_a_restored_port(
    recorded,
) -> None:
    from roboflow_workflows.execution_engine.v2.active.execution import (
        engine_observation,
        port_entries,
    )
    from roboflow_workflows.execution_engine.v2.data import EntryMetadata
    from roboflow_workflows.execution_engine.v2.sources import Emission, _RestoredPort

    plan, _, _ = recorded
    restored = _RestoredPort(data=None, filtered=((0,),), metadata=EntryMetadata())

    with pytest.raises(WorkflowInputError, match="not a valid 'float'"):
        port_entries(
            plan,
            "cam",
            emission=Emission({"value": restored}),
            observed=engine_observation(),
        )


def anchored_on_children_definition() -> Dict[str, Any]:
    """Frames plus a ``None`` field, reviewed per pulse anchored on the children.

    Children are gated per pulse, so a chunk can hold them filtered as a whole
    while its value, ``None`` and empty fields are complete.
    """
    definition = primary_definition()
    definition["steps"] = [
        *definition["steps"],
        {"type": "test/nothing@v1", "name": "nothing", "value": "$sources.cam.value"},
    ]
    frames, clips = definition["outputs"]
    frames = {
        **frames,
        "outputs": [*frames["outputs"], field("nothing", "$steps.nothing.value")],
    }
    definition["outputs"] = [frames, clips]
    workflow = {
        "steps": [
            {
                "type": "test/echo@v1",
                "name": "seen",
                "value": "$sources.frames.children",
            }
        ],
        "outputs": [
            group(
                "reviewed",
                "$sources.frames.children",
                field("value", "$sources.frames.value"),
                field("children", "$sources.frames.children"),
                field("seen", "$steps.seen.value"),
                field("empty", "$sources.frames.empty"),
                field("nothing", "$sources.frames.nothing"),
            )
        ],
    }

    return with_retrospective(definition, workflow)


@pytest.mark.parametrize(
    "pipeline", [None, PipelineOptions(max_in_flight=2)], ids=["serial", "pipelined"]
)
def test_a_wholly_filtered_anchor_still_replays_its_chunk_and_present_fields(
    tmp_path: Path, pipeline
) -> None:
    plan = compile_workflow(anchored_on_children_definition(), catalogue=CATALOGUE)
    live: List[Any] = []
    capture(
        plan,
        tmp_path / "rec",
        {"cam": feed_values(1.0, 2.0, None)},
        handlers={"frames": live.append},
    )

    replayed = replay(plan, tmp_path / "rec", pipeline=pipeline)

    assert [result.pulse.sequence for result in live] == [0, 1, 2]
    assert live[0].statuses["children"] == "filtered"
    assert [result.pulse.sequence for result in replayed] == [0, 1, 2]
    first = replayed[0]
    assert dict(first.statuses) == {
        "value": "complete",
        "children": "filtered",
        "seen": "filtered",
        "empty": "complete",
        "nothing": "complete",
    }
    assert first.outputs.data["value"] == 1.0
    assert first.outputs.data["nothing"] is None
    assert _plain(first.outputs.data["empty"]) == _plain(live[0].outputs.data["empty"])
    assert _plain(first.outputs.data["empty"]) is not None
    assert replayed[1].statuses["seen"] == "complete"
    assert _plain(replayed[1].outputs.data["seen"]) == _plain(
        live[1].outputs.data["children"]
    )
    assert replayed[2].is_filtered


def rescoring_workflow() -> Dict[str, Any]:
    """Runs the counting model again over every recorded value."""
    return {
        "steps": [
            {
                "type": "test/recorded_model@v1",
                "name": "rescore",
                "value": "$sources.frames.value",
            }
        ],
        "outputs": [
            group(
                "reviewed",
                "$sources.frames.value",
                field("score", "$steps.rescore.score"),
            )
        ],
    }


@pytest.mark.parametrize(
    "options",
    [
        PipelineOptions(overload="latest"),
        PipelineOptions(source_overload={"frames": "latest"}),
    ],
    ids=["global", "per-source"],
)
def test_replay_rejects_a_latest_overload_before_anything_is_built(
    tmp_path: Path, options: PipelineOptions
) -> None:
    plan = compile_workflow(
        with_retrospective(primary_definition(), rescoring_workflow()),
        catalogue=CATALOGUE,
    )
    live = capture(
        plan,
        tmp_path / "rec",
        {"cam": feed_values(1.0, 2.0)},
        pipeline=PipelineOptions(overload="latest"),
    )
    assert live.failure is None  # a live source keeps its freshness policy
    TRIPWIRE.clear()

    with pytest.raises(ContractError, match="'latest'.*drop recorded chunks"):
        plan.retrospective.start(recording=tmp_path / "rec", pipeline=options)

    assert TRIPWIRE == {}


def test_a_blocking_replay_delivers_every_chunk_to_a_slow_handler(
    tmp_path: Path,
) -> None:
    plan = compile_workflow(
        with_retrospective(primary_definition(), rescoring_workflow()),
        catalogue=CATALOGUE,
    )
    capture(
        plan,
        tmp_path / "rec",
        {"cam": feed_values(*(float(position) for position in range(40)))},
    )
    assert len(read_chunks(tmp_path / "rec", "frames")) == 40
    delivered: List[int] = []

    def slow(result) -> None:
        time.sleep(0.001)
        delivered.append(result.pulse.sequence)

    run = plan.retrospective.start(
        recording=tmp_path / "rec",
        handlers={"reviewed": slow},
        pipeline=PipelineOptions(
            max_in_flight=1, overload="latest", source_overload={"frames": "block"}
        ),
    )

    assert run.wait(WAIT) and run.failure is None
    assert delivered == list(range(40))
