"""Kinds returned by ``describe_outputs`` keep their hooks in the compiled plan (CF-03)."""

from typing import Mapping

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
)
from roboflow_workflows.execution_engine.v2.errors import (
    KindMismatchError,
    StepExecutionError,
)
from roboflow_workflows.execution_engine.v2.kinds import Kind

from tests.unit_tests.execution_engine.v2.compilation.fixture_blocks import workflow

SERIALIZED = []
COUNTER = Kind(
    name="counter",
    validate=lambda payload: isinstance(payload, int),
    serialize=lambda payload: SERIALIZED.append(payload) or {"count": payload},
    convert_output=lambda payload, options: (
        payload + 100 if options.get("coordinates_system") == "parent" else payload
    ),
)


def configured(identity: str, kind: Kind) -> type:
    """A block whose one output is named by configuration and has ``kind``."""

    class Configured(Block):
        type = identity
        output_fields = ("key",)

        class Params(BlockParams):
            key: str = "count"
            value: object = 7

        @classmethod
        def describe_outputs(cls, params: BlockParams) -> Mapping[str, Output]:
            return {params.key: Output(kind)}

        def run(self, *, key, value) -> dict:
            return {key: value}

    return Configured


COUNTING = configured("kinds/counting@v1", COUNTER)


def _plan(catalogue, *steps, outputs=None, output_options=None):
    definition = workflow(list(steps), outputs or {}, inputs=[])
    for output in definition["outputs"]:
        output.update(output_options or {})
    plan = compile_workflow(definition, catalogue=catalogue)

    return plan


@pytest.fixture(autouse=True)
def _reset_serialized():
    SERIALIZED.clear()


def test_configured_kind_validates_serializes_and_converts_like_a_static_one() -> None:
    catalogue = Catalogue([COUNTING])
    good = _plan(
        catalogue,
        {"type": "kinds/counting@v1", "name": "c"},
        outputs={"count": "$steps.c.count"},
        output_options={"coordinates_system": "parent"},
    )
    bad = _plan(
        catalogue,
        {"type": "kinds/counting@v1", "name": "c", "value": "seven"},
        outputs={"count": "$steps.c.count"},
    )

    assert good.catalogue.kinds["counter"] is COUNTER
    assert good.create_session().run({}).rows(serialize=True) == [
        {"count": {"count": 107}}
    ], "converted with the output's options, then serialized"
    assert SERIALIZED == [107]
    with pytest.raises(StepExecutionError, match="counter"):
        bad.create_session().run({})


def test_the_callers_catalogue_is_never_changed() -> None:
    catalogue = Catalogue([COUNTING])
    known = Catalogue([COUNTING], kinds=[COUNTER])

    plan = _plan(catalogue, {"type": "kinds/counting@v1", "name": "c"})
    reused = _plan(known, {"type": "kinds/counting@v1", "name": "c"})

    assert "counter" not in catalogue.kinds, "a fresh catalogue holds the new kind"
    assert plan.catalogue is not catalogue
    assert reused.catalogue is known, "nothing new: the same catalogue is kept"


@pytest.mark.parametrize("other", ["catalogue", "configured-step"])
def test_a_different_kind_with_a_known_name_is_rejected_at_compile_time(other) -> None:
    impostor = Kind(name="counter", validate=lambda payload: isinstance(payload, str))
    rival = configured("kinds/rival@v1", impostor)
    if other == "catalogue":
        catalogue = Catalogue([rival], kinds=[COUNTER])
        steps = [{"type": "kinds/rival@v1", "name": "rival"}]
    else:
        catalogue = Catalogue([COUNTING, rival])
        steps = [
            {"type": "kinds/counting@v1", "name": "first"},
            {"type": "kinds/rival@v1", "name": "rival"},
        ]

    with pytest.raises(KindMismatchError, match="'counter'") as info:
        _plan(catalogue, *steps)

    assert info.value.step_path == ("rival",)
    assert "counter" not in catalogue.kinds or catalogue.kinds["counter"] is COUNTER
