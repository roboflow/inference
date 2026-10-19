"""Real V2 sessions keep phased native calls under opt-in model ownership."""

from types import SimpleNamespace

import pytest
from pydantic import Field

from roboflow_workflows.v2.blocks.models import managed
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.data.kinds import INTEGER_KIND
from roboflow_workflows.v2.engine import (
    Block,
    BlockParams,
    Catalogue,
    CompileOptions,
    ContractError,
    Output,
    Ref,
    compile_workflow,
)
from roboflow_workflows.v2.engine.phases import phase

REQUEST = ModelRequest("test-model", "exact-package")


class NativeModel:

    def __init__(self):
        self.calls = []

    def pre_process(self, value):
        self.calls.append("pre_process")
        return value + 1

    def forward(self, value):
        self.calls.append("forward")
        return value * 2

    def post_process(self, value):
        self.calls.append("post_process")
        return value - 1


class PhasedModel(Block):
    type = "test/managed_native@v1"
    outputs = {"value": Output(INTEGER_KIND)}

    class Params(BlockParams):
        value: Ref(INTEGER_KIND) = Field(description="Value sent to native phases.")

    def __init__(self, *, models):
        self.models = models

    @phase
    def network_input(self, *, value):
        model = self.models.get(REQUEST)
        return (model, model.pre_process(value))

    @phase
    def raw_output(self, *, network_input):
        model, value = network_input
        return model.forward(value)

    @phase
    def result(self, *, network_input, raw_output):
        return {"value": network_input[0].post_process(raw_output)}

    def run(self, *, value):
        raise AssertionError("The engine must preserve phased execution")


def plan():
    return compile_workflow(
        {
            "version": "2.0",
            "inputs": [
                {"type": "WorkflowParameter", "name": "value", "kind": ["integer"]}
            ],
            "steps": [
                {"type": PhasedModel.type, "name": "model", "value": "$inputs.value"}
            ],
            "outputs": [{"name": "value", "selector": "$steps.model.value"}],
        },
        catalogue=Catalogue([PhasedModel]),
        options=CompileOptions(block_execution="phases"),
    )


@pytest.fixture
def manager(monkeypatch):

    class Manager:

        def __init__(self):
            self.models = {}
            self.shutdowns = 0

        def load(self, route, **kwargs):
            assert kwargs["pinned"] and kwargs["backend"] == "direct"
            assert kwargs["model_package_id"] == REQUEST.model_package_id
            self.models[route] = NativeModel()

        def get_backend(self, route):
            return SimpleNamespace(model=self.models[route])

        def shutdown(self):
            self.shutdowns += 1
            self.models.clear()

    manager = Manager()
    monkeypatch.setattr(managed, "_new_manager", lambda: manager)
    return manager


def test_native_phases_and_outputs_match_existing_stub(manager):
    compiled = plan()
    baseline = NativeModel()
    with compiled.create_session(
        resources={"models": ModelCache(supplied={REQUEST: baseline})}
    ) as session:
        expected = session.run({"value": 7}).rows()
    with managed.ManagedModels(api_key="", device="cpu") as owner:
        model = owner.preload(REQUEST)
        session = owner.create_session(compiled)
        assert session.run({"value": 7}).rows() == expected == [{"value": 15}]
        assert (
            model.calls == baseline.calls == ["pre_process", "forward", "post_process"]
        )
    assert session.closed and manager.shutdowns == 1


def test_unfinished_pipeline_keeps_models_alive_until_host_finishes(manager):
    owner = managed.ManagedModels(api_key="", device="cpu")
    session = owner.create_session(plan())
    with session.pipeline() as pipeline:
        with pytest.raises(ContractError, match="pipeline"):
            owner.close()
        assert manager.shutdowns == 0
        assert pipeline.submit({"value": 4}).result(timeout=5).rows() == [{"value": 9}]
    owner.close()
    assert session.closed and manager.shutdowns == 1
