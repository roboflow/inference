"""Dynamic block definitions composed from the root and nested workflows."""

import pytest
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.errors import WorkflowCompileError
from roboflow_workflows.execution_engine.v2.plan import CompileOptions, StepPort

from tests.unit_tests.execution_engine.v2.compilation.fixture_blocks import (
    CATALOGUE,
    nested,
    parameter,
    workflow,
)

# A module-level statement that raises proves the source is only parsed.
NEVER_EXECUTED = 'raise RuntimeError("submitted code ran during compilation")\n'


def dynamic_block(
    block_type, *, outputs, selector_types=("input_parameter", "step_output")
):
    run_code = NEVER_EXECUTED + (
        "def run(self, value) -> BlockResult:\n"
        f"    return {{name: value for name in {list(outputs)!r}}}\n"
    )
    definition = {
        "type": "DynamicBlockDefinition",
        "manifest": {
            "type": "ManifestDescription",
            "block_type": block_type,
            "inputs": {
                "value": {
                    "type": "DynamicInputDefinition",
                    "selector_types": list(selector_types),
                }
            },
            "outputs": {
                name: {"type": "DynamicOutputDefinition", "kind": []}
                for name in outputs
            },
        },
        "code": {"type": "PythonCode", "run_function_code": run_code},
    }

    return definition


def test_root_dynamic_block_compiles_like_a_catalogue_block_without_running_code() -> (
    None
):
    definition = workflow(
        [{"type": "CountingOffset", "name": "offset", "value": "$inputs.values"}],
        {"total": "$steps.offset.total"},
        dynamic_blocks=[dynamic_block("CountingOffset", outputs=["total", "calls"])],
    )

    plan = compile_workflow(definition, catalogue=CATALOGUE)
    allowed = compile_workflow(
        definition, catalogue=CATALOGUE, options=CompileOptions(allow_local_code=True)
    )

    offset = plan.step(("offset",))
    assert offset.namespace == "dynamic_workflows_blocks"
    assert list(offset.outputs) == ["total", "calls"]
    assert list(offset.invocation_layout.axis_ids) == ["inputs"]
    assert offset.binding_for("value").mode == "element"
    assert "CountingOffset" in plan.catalogue and "CountingOffset" not in CATALOGUE
    assert allowed.options.allow_local_code is True
    plan.describe()


def test_child_dynamic_blocks_compose_into_the_parent() -> None:
    child = workflow(
        [{"type": "InnerScalarEcho", "name": "pick", "value": "$inputs.child_msg"}],
        {"echo": "$steps.pick.output"},
        inputs=[parameter("child_msg", default="default-child")],
        dynamic_blocks=[
            dynamic_block(
                "InnerScalarEcho",
                outputs=["output"],
                selector_types=["input_parameter"],
            )
        ],
    )
    definition = workflow(
        [
            nested(
                "nested",
                workflow_definition=child,
                parameter_bindings={"child_msg": "$inputs.root_msg"},
            ),
            nested("again", workflow_definition=child),
        ],
        {"final": "$steps.nested.echo"},
        inputs=[parameter("root_msg", default="unused-root")],
    )

    plan = compile_workflow(definition, catalogue=CATALOGUE)

    assert [step.path for step in plan.steps] == [("nested", "pick"), ("again", "pick")]
    assert plan.outputs[0].source == StepPort(("nested", "pick"), "output")
    assert plan.warnings == (), "identical copies of one definition are not conflicts"


def test_conflicting_child_definition_is_dropped_with_a_warning() -> None:
    child = workflow(
        [{"type": "Shared", "name": "use", "value": "$inputs.x"}],
        {"y": "$steps.use.child_output"},
        inputs=[parameter("x", default=1)],
        dynamic_blocks=[dynamic_block("Shared", outputs=["child_output"])],
    )
    definition = workflow(
        [
            {"type": "Shared", "name": "root_use", "value": "$inputs.values"},
            nested("child", workflow_definition=child),
        ],
        dynamic_blocks=[dynamic_block("Shared", outputs=["root_output"])],
    )

    with pytest.raises(WorkflowCompileError, match="has no output 'child_output'"):
        compile_workflow(definition, catalogue=CATALOGUE)
    definition["steps"][1]["workflow_definition"]["outputs"] = []
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    assert list(plan.step(("child", "use")).outputs) == ["root_output"]
    assert len(plan.warnings) == 1
    assert (
        "steps[1].workflow_definition.dynamic_blocks_definitions[0]" in plan.warnings[0]
    )


def test_dynamic_block_type_colliding_with_the_catalogue_is_rejected() -> None:
    definition = workflow(
        [],
        dynamic_blocks=[dynamic_block("test/scale@v1", outputs=["scaled"])],
    )

    with pytest.raises(WorkflowCompileError, match="already registered"):
        compile_workflow(definition, catalogue=CATALOGUE)


def test_invalid_nested_dynamic_definition_names_its_location() -> None:
    broken = dynamic_block("Broken", outputs=["out"])
    broken["code"][
        "run_function_code"
    ] = "def run(self, value) BlockResult:\n    pass\n"
    child = workflow([step_of("Broken")], inputs=[], dynamic_blocks=[broken])
    definition = workflow([nested("child", workflow_definition=child)], inputs=[])

    with pytest.raises(WorkflowCompileError) as info:
        compile_workflow(definition, catalogue=CATALOGUE)

    location = "steps[0].workflow_definition.dynamic_blocks_definitions[0]"
    assert location in str(info.value)
    assert info.value.location == location
    assert "run_function_code" in str(info.value)


def step_of(block_type):
    return {"type": block_type, "name": "use", "value": 1}
