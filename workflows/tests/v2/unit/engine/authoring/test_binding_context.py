"""Field lookup preserves index guards and old custom scopes."""

from types import SimpleNamespace

import pytest
from pydantic import Field
from roboflow_workflows.v2.data.batch import SampleContext, TemporalContext, Timestamp
from roboflow_workflows.v2.engine.context import ExecutionContext
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Ref
from roboflow_workflows.v2.engine.errors import ContractError
from roboflow_workflows.v2.engine.reactions.testing import block_call


class Fields(Block):
    type = "test/binding_context_fields@v1"
    outputs = {}

    class Params(BlockParams):
        image: Ref()
        label: str = Field(default="hello", alias="display_label")

    def run(self, *, image, label):
        return {}


def test_old_scope_is_called_without_new_keywords_when_field_is_omitted():
    sample = SampleContext(source_id="old")
    temporal = TemporalContext(Timestamp(1, 1, "clock"))

    class OldScope:
        def sample_at(self, index):
            assert index == ()
            return sample

        def temporal_at(self, index):
            assert index == ()
            return temporal

    context = ExecutionContext(
        ("step",), "old", "session", run_id="run", indices=((),), call_scope=OldScope()
    )
    assert context.sample_at(()) is sample
    assert context.source_id_at(()) == "old"
    assert context.temporal_at(()) is temporal


def test_block_call_known_fields_share_supplied_context_and_aliases_are_unknown():
    temporal = TemporalContext(Timestamp(1, 1, "clock"))
    with block_call(
        Fields,
        sources={(0,): "a", (1,): None},
        temporal={(0,): temporal},
        indices=((0,), (1,)),
    ) as call:
        context = call.context
        for field in (None, "image", "label"):
            assert context.sample_at((0,), field=field).source_id == "a"
            assert context.temporal_at((0,), field=field) is temporal
            assert context.source_id_at((0,), field=field) == "a"
            assert context.sample_at((1,), field=field) is None
        for field in ("display_label", "$inputs.image", "unknown", 3):
            with pytest.raises(ContractError, match="unknown Params field"):
                context.sample_at((0,), field=field)
            with pytest.raises(ContractError, match="unknown Params field"):
                context.temporal_at((0,), field=field)
        with pytest.raises(ContractError, match="names no index"):
            context.sample_at((2,), field="image")
        with pytest.raises(ContractError, match="no single source"):
            context.source_id_at((1,), field="image")


def test_constructor_and_hand_built_context_cannot_invent_field_bindings():
    constructor = ExecutionContext(("step",), "type", "session")
    with pytest.raises(ContractError, match="names no index"):
        constructor.sample_at((), field="image")
    context = ExecutionContext(("step",), "type", "session", indices=((),))
    assert context.sample_at(()) is None
    with pytest.raises(ContractError, match="no bindings"):
        context.sample_at((), field="image")


def test_scope_resolves_only_queried_field_indices_and_reuses_sample_time_pair(
    monkeypatch,
):
    from roboflow_workflows.v2.engine.execution import steps

    sample = SampleContext(source_id="origin")
    temporal = TemporalContext(Timestamp(1, 1, "clock"))
    step = SimpleNamespace(
        spec=SimpleNamespace(fields={"label": object()}), bindings=(), path=("step",)
    )
    run = SimpleNamespace(origin=(sample, temporal))
    scope = steps._StepScope(run, step, leaves={(): (), (0,): ()}, entries=())
    calls = []
    original = steps._binding_contexts

    def record(*args, **kwargs):
        calls.append((args[0], kwargs["kind"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(steps, "_binding_contexts", record)
    assert scope._resolved is None and scope._field_resolved is None
    assert scope.sample_at(()) is sample
    assert scope._field_resolved is None
    assert len(calls) == 2
    assert scope.sample_at((), field="label") is sample
    assert scope.temporal_at((), field="label") is temporal
    assert scope.sample_at((), field="label") is sample
    assert len(calls) == 4
    assert set(scope._field_resolved) == {("label", ())}
    assert scope.temporal_at((0,), field="label") is temporal
    assert len(calls) == 6
