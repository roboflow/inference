"""Trusted Python retrospective stages: whole-data and per-chunk access."""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.errors import ContractError
from roboflow_workflows.execution_engine.v2.recording.errors import (
    RecordingSchemaError,
    RetrospectiveError,
)

from tests.unit_tests.execution_engine.v2.recording.test_capture_run import (
    CATALOGUE,
    TRIPWIRE,
    _plain,
    _reset_tripwire,
    capture,
    feed_values,
    field,
    primary_definition,
)

__all__ = ["_reset_tripwire"]  # the autouse fixture applies here too


def python_definition(access: str, **stage: Any) -> Dict[str, Any]:
    definition = primary_definition()
    definition["inputs"].append({"type": "WorkflowParameter", "name": "results_dir"})
    definition["retrospective"] = {
        "type": "python",
        "input_group": "frames",
        "access": access,
        "parameters": {"minimum": 0.15, "out": "$inputs.results_dir"},
        "result_directory": "$inputs.results_dir",
        **stage,
    }

    return definition


def recorded_plan(tmp_path: Path, access: str, **stage: Any):
    plan = compile_workflow(python_definition(access, **stage), catalogue=CATALOGUE)
    capture(plan, tmp_path / "rec", {"cam": feed_values(1.0, 2.0, None, 3.0)})

    return plan


def test_whole_data_access_is_lazy_repeatable_and_registers_user_files(
    tmp_path: Path,
) -> None:
    plan = recorded_plan(tmp_path, "all")
    calls = TRIPWIRE["model_calls"]

    def two_passes(group, results, parameters) -> None:
        scores = [c.values["score"] for c in group.iter_chunks() if "score" in c.values]
        mean = round(sum(scores) / len(scores), 6)
        above = [
            c.index
            for c in group.iter_chunks()
            if c.values.get("score", 0) > max(mean, parameters["minimum"])
        ]
        path = results.directory / "summary.json"
        path.write_text(json.dumps({"mean": mean, "above": above}))
        results.register("summary.json")

    outcome = plan.retrospective.run_python(
        two_passes,
        recording=tmp_path / "rec",
        root_inputs={"results_dir": str(tmp_path / "out")},
    )

    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary == {"mean": pytest.approx(0.2), "above": [3]}
    assert outcome.files == ((tmp_path / "out" / "summary.json").resolve(),)
    assert outcome.chunks == 4 and outcome.recording_status == "complete"
    listed = json.loads((tmp_path / "out" / "registered.json").read_text())
    assert listed == {"files": ["summary.json"]}
    assert TRIPWIRE["model_calls"] == calls


def test_chunk_access_calls_once_per_chunk_in_order_with_metadata(
    tmp_path: Path,
) -> None:
    plan = recorded_plan(tmp_path, "chunks")
    seen: List[Any] = []

    def per_chunk(chunk, results, parameters) -> None:
        seen.append(
            (
                chunk.index,
                chunk.pulse.sequence,
                chunk.is_filtered,
                _plain(chunk.data.get("children")),
                (
                    chunk.metadata["value"].sample[()].source_id
                    if "value" in chunk.metadata
                    else None
                ),
                parameters["out"],
            )
        )

    outcome = plan.retrospective.run_python(
        per_chunk,
        recording=tmp_path / "rec",
        root_inputs={"results_dir": str(tmp_path / "out")},
    )

    out = str(tmp_path / "out")
    assert seen == [
        (0, 0, False, None, "cam", out),
        (1, 1, False, [((0,), 2.0), ((1,), 3.0), ((2,), 4.0)], "cam", out),
        (2, 2, True, None, None, out),
        (3, 3, False, [((0,), 3.0), ((1,), 4.0), ((2,), 5.0)], "cam", out),
    ]
    assert outcome.chunks == 4 and outcome.files == ()


def test_a_failing_chunk_names_its_index_and_keeps_earlier_registrations(
    tmp_path: Path,
) -> None:
    plan = recorded_plan(tmp_path, "chunks")

    def fail_on_third(chunk, results, parameters) -> None:
        if chunk.index == 2:
            raise ValueError("bad chunk")
        path = results.directory / f"chunk-{chunk.index}.txt"
        path.write_text("ok")
        results.register(path)

    with pytest.raises(
        RetrospectiveError, match="chunk 2: .*ValueError: bad chunk"
    ) as raised:
        plan.retrospective.run_python(
            fail_on_third,
            recording=tmp_path / "rec",
            root_inputs={"results_dir": str(tmp_path / "out")},
        )

    assert raised.value.chunk_index == 2
    listed = json.loads((tmp_path / "out" / "registered.json").read_text())
    assert listed == {"files": ["chunk-0.txt", "chunk-1.txt"]}


def test_registration_keeps_first_registration_order_and_one_entry_per_file(
    tmp_path: Path,
) -> None:
    plan = recorded_plan(tmp_path, "chunks")

    def per_chunk(chunk, results, parameters) -> None:
        shared = results.directory / "index.txt"
        shared.write_text("shared")
        results.register("index.txt")
        path = results.directory / f"chunk-{chunk.index}.txt"
        path.write_text("ok")
        results.register(path)
        results.register(shared)

    outcome = plan.retrospective.run_python(
        per_chunk,
        recording=tmp_path / "rec",
        root_inputs={"results_dir": str(tmp_path / "out")},
    )

    names = [file.name for file in outcome.files]
    assert names == ["index.txt", *(f"chunk-{index}.txt" for index in range(4))]
    listed = json.loads((tmp_path / "out" / "registered.json").read_text())
    assert listed == {"files": names}


@pytest.mark.parametrize(
    "path, message",
    [
        ("missing.txt", "does not exist"),
        ("../outside.txt", "outside the result directory"),
        ("registered.json", "written by the engine"),
    ],
)
def test_invalid_registrations_are_rejected(
    tmp_path: Path, path: str, message: str
) -> None:
    plan = recorded_plan(tmp_path, "all")
    (tmp_path / "outside.txt").write_text("x")

    def register(group, results, parameters) -> None:
        (results.directory / "registered.json").write_text("{}")
        results.register(path)

    with pytest.raises(RetrospectiveError, match=message):
        plan.retrospective.run_python(
            register,
            recording=tmp_path / "rec",
            root_inputs={"results_dir": str(tmp_path / "out")},
        )


def test_python_stage_misuse_fails_before_reading(tmp_path: Path) -> None:
    plan = recorded_plan(tmp_path, "all")

    async def coroutine(group, results, parameters) -> None:
        return None

    with pytest.raises(ContractError, match="synchronous callable"):
        plan.retrospective.run_python(coroutine, recording=tmp_path / "rec")
    with pytest.raises(ContractError, match="start"):
        plan.retrospective.start(recording=tmp_path / "rec")


def test_a_changed_schema_is_rejected_before_the_function_runs(tmp_path: Path) -> None:
    recorded_plan(tmp_path, "all")
    changed = python_definition("all")
    changed["outputs"][0]["outputs"].pop()
    plan = compile_workflow(changed, catalogue=CATALOGUE)
    called = []

    with pytest.raises(RecordingSchemaError, match="empty"):
        plan.retrospective.run_python(
            lambda *arguments: called.append(arguments),
            recording=tmp_path / "rec",
            root_inputs={"results_dir": str(tmp_path / "out")},
        )

    assert called == []


def test_wildcard_fields_reach_python_stages_as_port_mappings(tmp_path: Path) -> None:
    definition = python_definition("chunks")
    definition["outputs"][0]["outputs"].append(field("all", "$steps.expand.*"))
    plan = compile_workflow(definition, catalogue=CATALOGUE)
    capture(plan, tmp_path / "rec", {"cam": feed_values(2.0)})
    seen = []

    plan.retrospective.run_python(
        lambda chunk, results, parameters: seen.append(chunk.data["all"]),
        recording=tmp_path / "rec",
        root_inputs={"results_dir": str(tmp_path / "out")},
    )

    assert set(seen[0]) == {"children", "count"} and seen[0]["count"] == 3
