"""Compile rules of the root ``recording`` and ``retrospective`` declarations."""

import copy
from typing import Any, Dict

import pytest
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.errors import (
    SelectorError,
    WorkflowCompileError,
)
from roboflow_workflows.execution_engine.v2.recording.errors import (
    RecordingDefinitionError,
)

from tests.unit_tests.execution_engine.v2.recording.test_capture_run import (
    CATALOGUE,
    field,
    group,
    primary_definition,
)

REVIEW = {
    "steps": [
        {"type": "test/echo@v1", "name": "echo", "value": "$sources.frames.score"}
    ],
    "outputs": [
        group("reviewed", "$sources.frames.value", field("score", "$steps.echo.value"))
    ],
}


def staged(**retrospective: Any) -> Dict[str, Any]:
    definition = primary_definition()
    definition["retrospective"] = {
        "type": "workflow",
        "input_group": "frames",
        "workflow": copy.deepcopy(REVIEW),
        **retrospective,
    }

    return definition


def test_recording_and_retrospective_compile_beside_the_primary_plan() -> None:
    plan = compile_workflow(staged(), catalogue=CATALOGUE)

    assert plan.recording.groups == ("frames", "clips")
    assert plan.recording.schema.group("frames").keys == (
        "value",
        "score",
        "children",
        "kept",
        "empty",
    )
    assert plan.recording.schema.group("clips").entry("clip").layout.has_time
    assert plan.retrospective.kind == "workflow"
    assert list(plan.retrospective.plan.sources) == ["frames"]
    description = plan.describe()
    assert description["recording"]["directory"] == "$inputs.capture_dir"
    assert description["retrospective"]["input_group"] == "frames"
    assert set(plan.retrospective.plan.describe()["sources"]) == {"frames"}


def test_the_digest_ignores_the_retrospective_stage() -> None:
    first = compile_workflow(staged(), catalogue=CATALOGUE)
    changed = staged()
    changed["retrospective"]["workflow"]["steps"][0]["name"] = "other"
    changed["retrospective"]["workflow"]["outputs"][0]["outputs"][0][
        "selector"
    ] = "$steps.other.value"
    second = compile_workflow(changed, catalogue=CATALOGUE)

    assert first.recording.definition_digest == second.recording.definition_digest
    primary = primary_definition()
    primary["steps"][0]["name"] = "model2"
    primary["outputs"][0]["outputs"][1]["selector"] = "$steps.model2.score"
    third = compile_workflow(primary, catalogue=CATALOGUE)
    assert third.recording.definition_digest != first.recording.definition_digest


def _child_with_recording() -> Dict[str, Any]:
    definition = primary_definition()
    del definition["recording"]
    definition["steps"].append(
        {
            "type": "roboflow_core/inner_workflow@v1",
            "name": "child",
            "workflow_definition": {
                "version": "2.0",
                "inputs": [{"type": "WorkflowParameter", "name": "x"}],
                "steps": [],
                "outputs": [
                    {"type": "JsonField", "name": "x", "selector": "$inputs.x"}
                ],
                "recording": {"type": "file", "directory": "d", "groups": ["g"]},
            },
            "parameter_bindings": {"x": "$sources.cam.value"},
        }
    )

    return definition


def _replace(definition: Dict[str, Any], **sections: Any) -> Dict[str, Any]:
    definition = copy.deepcopy(definition)
    definition.update(sections)

    return definition


def _recording(**changes: Any) -> Dict[str, Any]:
    recording = {
        "type": "file",
        "directory": "$inputs.capture_dir",
        "groups": ["frames"],
    }
    recording.update(changes)

    return recording


@pytest.mark.parametrize(
    "definition, message",
    [
        (_child_with_recording(), r"workflow_definition\.recording is root-only"),
        (
            staged(workflow={**REVIEW, "recording": _recording()}),
            r"retrospective\.workflow\.recording is root-only",
        ),
        (
            _replace(primary_definition(), recording=_recording(groups=["nope"])),
            "unknown output group 'nope'",
        ),
        (
            _replace(primary_definition(), recording=_recording(groups=[])),
            "non-empty list",
        ),
        (
            _replace(
                primary_definition(), recording=_recording(groups=["frames", "frames"])
            ),
            "repeats group",
        ),
        (
            _replace(primary_definition(), recording=_recording(type="s3")),
            r"recording\.type",
        ),
        (
            _replace(primary_definition(), recording=_recording(directory="")),
            "non-empty path",
        ),
        (
            _replace(
                primary_definition(),
                recording=_recording(directory="$steps.model.score"),
            ),
            r"only \$inputs",
        ),
        (
            _replace(
                primary_definition(), recording=_recording(directory="$inputs.missing")
            ),
            "root inputs",
        ),
        (
            _replace(primary_definition(), recording=_recording(extra=1)),
            "unsupported keys",
        ),
        (
            {
                "version": "2.0",
                "inputs": [{"type": "WorkflowParameter", "name": "x"}],
                "steps": [],
                "outputs": [
                    {"type": "JsonField", "name": "x", "selector": "$inputs.x"}
                ],
                "recording": _recording(),
            },
            "active definition with sources",
        ),
        (_replace(staged(), recording=None), "needs a recording declaration"),
        (staged(input_group="nope"), "one recorded group"),
        (staged(entrypoint="module:function"), "never imports code"),
        (staged(type="sql"), r"retrospective\.type"),
        (staged(access="all"), "unsupported keys"),
        (staged(workflow=[]), "workflow definition mapping"),
        (
            staged(
                workflow={
                    **REVIEW,
                    "sources": [
                        {"type": "test/recorded_feed@v1", "name": "x", "feed": "a"}
                    ],
                }
            ),
            "declares sources",
        ),
        (
            staged(
                workflow={
                    **REVIEW,
                    "outputs": [
                        *REVIEW["outputs"],
                        group(
                            "again",
                            "$sources.frames.value",
                            field("v", "$sources.frames.value"),
                        ),
                    ],
                }
            ),
            "exactly one OutputGroup",
        ),
        (
            staged(
                workflow={
                    **REVIEW,
                    "steps": [
                        {
                            "type": "test/echo@v1",
                            "name": "echo",
                            "value": "$sources.frames.nope",
                        }
                    ],
                }
            ),
            "nope",
        ),
        (
            staged(
                type="python",
                workflow=None,
            ),
            "unsupported keys",
        ),
    ],
)
def test_invalid_declarations_fail_compilation_with_a_located_message(
    definition: Dict[str, Any], message: str
) -> None:
    with pytest.raises(WorkflowCompileError, match=message):
        compile_workflow(definition, catalogue=CATALOGUE)


def test_a_wildcard_field_cannot_be_replayed_by_a_workflow() -> None:
    definition = staged()
    definition["outputs"][0]["outputs"].append(field("all", "$steps.expand.*"))

    with pytest.raises(
        RecordingDefinitionError, match=r"wildcard field\(s\) \['all'\]"
    ):
        compile_workflow(definition, catalogue=CATALOGUE)


@pytest.mark.parametrize(
    "stage, message",
    [
        ({"access": "sometimes", "result_directory": "out"}, r"retrospective\.access"),
        ({"access": "all"}, r"retrospective\.result_directory"),
        (
            {"access": "all", "result_directory": "out", "parameters": []},
            "parameters must be a mapping",
        ),
        (
            {
                "access": "all",
                "result_directory": "out",
                "parameters": {"p": "$steps.x.y"},
            },
            r"only \$inputs",
        ),
    ],
)
def test_invalid_python_stages_fail_compilation(
    stage: Dict[str, Any], message: str
) -> None:
    definition = primary_definition()
    definition["retrospective"] = {"type": "python", "input_group": "frames", **stage}

    with pytest.raises(RecordingDefinitionError, match=message):
        compile_workflow(definition, catalogue=CATALOGUE)


def test_retrospective_workflow_errors_name_the_stage_or_the_step() -> None:
    malformed = staged(workflow={**REVIEW, "steps": "not a list"})
    unknown_type = staged(
        workflow={**REVIEW, "steps": [{"type": "test/unknown@v1", "name": "u"}]}
    )

    with pytest.raises(WorkflowCompileError, match=r"^retrospective\.workflow\.steps"):
        compile_workflow(malformed, catalogue=CATALOGUE)
    with pytest.raises(WorkflowCompileError, match=r"^\$steps\.u .*test/unknown@v1"):
        compile_workflow(unknown_type, catalogue=CATALOGUE)


def test_a_retrospective_step_reading_a_primary_step_is_a_selector_error() -> None:
    """Recorded fields are ``$sources.<input_group>.<field>``, not primary steps."""
    definition = staged(
        workflow={
            **REVIEW,
            "steps": [
                {"type": "test/echo@v1", "name": "echo", "value": "$steps.model.score"}
            ],
        }
    )

    with pytest.raises(SelectorError, match="unknown step 'model'"):
        compile_workflow(definition, catalogue=CATALOGUE)
