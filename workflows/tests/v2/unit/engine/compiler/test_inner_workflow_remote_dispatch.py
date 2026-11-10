"""Contract guard: remote dispatch stays unsupported; local resources stay local."""

import pytest
from roboflow_workflows.v2.blocks.control.inner_workflow_dispatch import (
    inner_workflow_step,
)
from roboflow_workflows.v2.data.kinds import FLOAT_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import WorkflowReference, compile_workflow
from roboflow_workflows.v2.engine.compilation.definition import (
    NestedStepDeclaration,
    parse_workflow,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from roboflow_workflows.v2.engine.errors import ResourceError, WorkflowCompileError


class LocalConsumer(Block):
    type = "test/local-consumer@v1"
    outputs = {"y": Output(FLOAT_KIND)}

    class Params(BlockParams):
        x: Ref(FLOAT_KIND)

    def __init__(self, *, local_scale):
        self.scale = local_scale

    def run(self, *, x):
        return {"y": self.scale(x)}


CATALOGUE = Catalogue([LocalConsumer], kinds=[FLOAT_KIND])
CHILD = {
    "version": "2.0",
    "inputs": [{"type": "WorkflowParameter", "name": "x", "kind": ["float"]}],
    "steps": [{"type": LocalConsumer.type, "name": "use", "x": "$inputs.x"}],
    "outputs": [{"type": "JsonField", "name": "y", "selector": "$steps.use.y"}],
}


def parent(child_step):
    return {
        "version": "2.0",
        "inputs": CHILD["inputs"],
        "steps": [child_step],
        "outputs": [{"type": "JsonField", "name": "y", "selector": "$steps.child.y"}],
    }


@pytest.mark.parametrize("mode", ["remote_dispatch", "other", None, False])
def test_non_embedded_modes_reject_before_saved_resolver(mode):
    calls = []
    child_step = inner_workflow_step("child", reference=WorkflowReference("local", "x"))
    child_step["execution_mode"] = mode
    with pytest.raises(WorkflowCompileError, match="only embeds child workflows"):
        compile_workflow(
            parent(child_step),
            catalogue=CATALOGUE,
            reference_resolver=lambda ref: calls.append(ref) or CHILD,
        )
    assert calls == []


def test_local_child_stays_nested_and_uses_parent_resource():
    child_step = inner_workflow_step(
        "child", workflow_definition=CHILD, parameter_bindings={"x": "$inputs.x"}
    )
    assert isinstance(
        parse_workflow(parent(child_step)).steps[0], NestedStepDeclaration
    )
    plan = compile_workflow(parent(child_step), catalogue=CATALOGUE)
    calls = []
    with plan.create_session(
        resources={"local_scale": lambda x: calls.append(x) or x * 4}
    ) as session:
        assert session.run({"x": 3.0}).rows() == [{"y": 12.0}]
    assert calls == [3.0]
    with pytest.raises(ResourceError):
        with plan.create_session() as session:
            session.run({"x": 3.0})


def test_remote_target_remains_unknown_field():
    child_step = inner_workflow_step("child", workflow_definition=CHILD)
    child_step["remote_target"] = "https://excluded.invalid"
    with pytest.raises(WorkflowCompileError, match="remote_target"):
        compile_workflow(parent(child_step), catalogue=CATALOGUE)
