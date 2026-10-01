"""In-place mutation of a workflow input that every pulse of an active run shares.

An active run prepares its workflow inputs once, so a step mutating one in
place changes it for later pulses (serially) and races with concurrent pulses
(pipelined). The compiler reports this by default and rejects it under
``mutation_conflicts="error"``; the serial run keeps its existing behaviour.
"""

from typing import List, Optional

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.errors import MutationConflictError
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND
from roboflow_workflows.execution_engine.v2.plan import CompileOptions
from roboflow_workflows.execution_engine.v2.sources import (
    Emission,
    Source,
    SourceOutput,
    SourceParams,
)

from tests.unit_tests.execution_engine.v2.compilation.fixture_blocks import (
    ALL_BLOCKS,
    nested,
    parameter,
    step,
)

WAIT = 10.0
STRICT = CompileOptions(mutation_conflicts="error")
SHARED = "an active run shares its workflow inputs with every pulse"


class Ticks(Source):
    """Emits ``count`` scalar ticks, then ends."""

    type = "test/ticks@v1"
    outputs = {"tick": SourceOutput(FLOAT_KIND)}

    class Params(SourceParams):
        count: int = 3

    def open(self, *, count: int) -> None:
        self.remaining = count

    def read(self) -> Optional[Emission]:
        if self.remaining == 0:
            return None

        self.remaining -= 1

        return Emission({"tick": float(self.remaining)})


CATALOGUE = Catalogue(ALL_BLOCKS, sources=[Ticks], namespace="test")
STATE = parameter("state", default={"count": 0}, kind=["dictionary"])


def _definition(steps: List[dict], *, sources: bool = True, outputs=()) -> dict:
    definition = {
        "version": "2.0",
        "inputs": [STATE],
        "steps": steps,
        "outputs": list(outputs),
    }
    if sources:
        definition["sources"] = [{"type": Ticks.type, "name": "ticks"}]

    return definition


def _counted_group() -> dict:
    return {
        "type": "OutputGroup",
        "name": "seen",
        "anchor": "$sources.ticks.tick",
        "outputs": [
            {"type": "JsonField", "name": "seen", "selector": "$steps.read.seen"},
            {"type": "JsonField", "name": "tick", "selector": "$sources.ticks.tick"},
        ],
    }


def test_mutating_a_shared_static_input_warns_in_an_active_definition() -> None:
    plan = compile_workflow(
        _definition([step("increment", "bump", value="$inputs.state")]),
        catalogue=CATALOGUE,
    )

    [warning] = plan.warnings
    assert warning.startswith(
        "$steps.bump value ($inputs.state) is mutated in place, but the payload "
        "passed ['$inputs.state']"
    )
    assert SHARED in warning


def test_strict_policy_rejects_the_shared_static_input_mutation() -> None:
    with pytest.raises(MutationConflictError, match=SHARED) as raised:
        compile_workflow(
            _definition([step("increment", "bump", value="$inputs.state")]),
            catalogue=CATALOGUE,
            options=STRICT,
        )

    assert raised.value.step_path == ("bump",)
    assert raised.value.field_path == ("value",)


def test_an_alias_through_a_declared_output_source_is_also_shared() -> None:
    plan = compile_workflow(
        _definition(
            [
                step("echo", "forward", value="$inputs.state"),
                step("increment", "bump", value="$steps.forward.value"),
            ]
        ),
        catalogue=CATALOGUE,
    )

    [warning] = plan.warnings
    assert warning.startswith("$steps.bump value ($steps.forward.value)")
    assert SHARED in warning


@pytest.mark.parametrize(
    ("steps", "sources"),
    [
        pytest.param(
            [step("increment", "bump", value="$inputs.state")],
            False,
            id="passive-inputs-have-no-declared-cross-run-alias",
        ),
        pytest.param(
            [
                nested(
                    "child",
                    workflow_definition={
                        "version": "2.0",
                        "inputs": [STATE],
                        "steps": [step("increment", "bump", value="$inputs.state")],
                        "outputs": [],
                    },
                )
            ],
            True,
            id="child-default-is-materialized-per-pulse",
        ),
        pytest.param(
            [step("reader", "read", value="$inputs.state")],
            True,
            id="readers-alone-share-safely",
        ),
    ],
)
def test_private_or_unmutated_payloads_compile_without_the_report(
    steps: List[dict], sources: bool
) -> None:
    plan = compile_workflow(
        _definition(steps, sources=sources), catalogue=CATALOGUE, options=STRICT
    )

    assert plan.warnings == ()


def test_serial_active_run_keeps_its_existing_cross_pulse_visibility() -> None:
    plan = compile_workflow(
        _definition(
            [
                step("increment", "bump", value="$inputs.state"),
                step("reader", "read", value="$steps.bump.value"),
            ],
            outputs=[_counted_group()],
        ),
        catalogue=CATALOGUE,
    )
    assert len(plan.warnings) == 1
    delivered = []

    session = plan.create_session()
    run = session.start({}, handlers={"seen": delivered.append})

    assert run.wait(timeout=WAIT) is True
    seen = [
        result.outputs.data[result.selections["seen"]["$steps.read.seen"]]
        for result in delivered
    ]
    assert seen == [1, 2, 3]
