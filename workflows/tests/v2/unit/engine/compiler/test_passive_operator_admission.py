"""Which operators a plan without sources admits (passive history).

A passive history plan runs each operator inline in the submission that fed
it, so compilation admits only operators that declare
``passive_max_emissions = 1``, read one upstream domain and, for a Window,
drop the partial tail at close. Nothing here runs a workflow.
"""

from dataclasses import replace
from typing import Any, Dict, List

import pytest
from roboflow_workflows.v2.engine import Catalogue, compile_workflow
from roboflow_workflows.v2.engine.errors import WorkflowCompileError
from roboflow_workflows.v2.engine.operators import (
    Operator,
    OperatorPort,
    OperatorPulse,
)
from roboflow_workflows.v2.engine.operators.contract import OperatorDeclarationError
from roboflow_workflows.v2.engine.operators.window import Window
from roboflow_workflows.v2.engine.updates.diff import _same


class Relay(Operator):
    """Forwards its first arrival: at most one pulse per push."""

    type = "test/relay@v1"
    input_roles = ("input",)
    passive_max_emissions = 1

    @classmethod
    def plan_ports(cls, name, params, inputs) -> Dict[str, OperatorPort]:
        return {"out": OperatorPort()}

    def push(self, arrivals) -> List[OperatorPulse]:
        first = arrivals[0]
        return [OperatorPulse(ports={"out": first.entry}, causes=(first.pulse,))]


class Uncapped(Relay):
    type = "test/uncapped@v1"
    passive_max_emissions = None


class Burst(Relay):
    type = "test/burst@v1"
    passive_max_emissions = 2


class Mute(Relay):
    type = "test/mute@v1"
    passive_max_emissions = 0


CATALOGUE = Catalogue([], operators=[Relay, Uncapped, Burst, Mute, Window])


def _definition(*operators: Dict[str, Any], selector: str) -> Dict[str, Any]:
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "x"}],
        "operators": list(operators),
        "steps": [],
        "outputs": [{"type": "JsonField", "name": "out", "selector": selector}],
    }

    return definition


def _relay(operator_type: str, name: str = "op", **inputs: str) -> Dict[str, Any]:
    return {"type": operator_type, "name": name, "inputs": inputs or {"x": "$inputs.x"}}


def test_capable_operators_compile_into_a_passive_history_plan() -> None:
    relay = _definition(_relay(Relay.type), selector="$operators.op.out")
    window = _definition(
        {"type": Window.type, "name": "op", "size": 2, "collect": {"x": "$inputs.x"}},
        selector="$operators.op.x",
    )

    for definition in (relay, window):
        plan = compile_workflow(definition, catalogue=CATALOGUE)
        assert plan.passive_history
        assert plan.domains == ("$passive", "op")
        assert plan.operator("op").upstream_domains == ("$passive",)


def test_plans_without_operators_stay_ordinary_passive_plans() -> None:
    definition = _definition(selector="$inputs.x")

    plan = compile_workflow(definition, catalogue=CATALOGUE)

    assert not plan.passive_history
    assert plan.domains == ()


@pytest.mark.parametrize(
    ("operator_type", "stated"),
    [
        (Uncapped.type, "(it declares none);"),
        (Mute.type, "(it declares 0);"),
        (Burst.type, "(it declares 2);"),
    ],
    ids=["undeclared", "zero", "two"],
)
def test_operators_without_the_one_emission_capability_fail_to_compile(
    operator_type: str, stated: str
) -> None:
    definition = _definition(_relay(operator_type), selector="$operators.op.out")

    with pytest.raises(WorkflowCompileError) as raised:
        compile_workflow(definition, catalogue=CATALOGUE)

    message = str(raised.value)
    assert stated in message
    assert "supports only passive_max_emissions = 1" in message


def test_window_that_emits_its_partial_tail_fails_to_compile() -> None:
    definition = _definition(
        {
            "type": Window.type,
            "name": "op",
            "size": 3,
            "partial": "emit",
            "collect": {"x": "$inputs.x"},
        },
        selector="$operators.op.x",
    )

    with pytest.raises(WorkflowCompileError, match="partial='drop'"):
        compile_workflow(definition, catalogue=CATALOGUE)


def test_operator_reading_two_upstream_domains_fails_to_compile() -> None:
    definition = _definition(
        _relay(Relay.type, name="first"),
        _relay(Relay.type, name="both", x="$inputs.x", y="$operators.first.out"),
        selector="$operators.both.out",
    )

    with pytest.raises(WorkflowCompileError, match="one upstream domain"):
        compile_workflow(definition, catalogue=CATALOGUE)


@pytest.mark.parametrize("value", [-1, 1.0, "1", True])
def test_capability_must_be_none_or_a_nonnegative_integer(value: Any) -> None:
    with pytest.raises(OperatorDeclarationError, match="passive_max_emissions"):
        type(
            "Invalid",
            (Relay,),
            {"type": "test/invalid@v1", "passive_max_emissions": value},
        )


def test_catalogue_describes_the_capability() -> None:
    described = {
        item["type"]: item["passive_max_emissions"]
        for item in CATALOGUE.describe()["operators"]
    }

    assert described == {
        Relay.type: 1,
        Uncapped.type: None,
        Burst.type: 2,
        Mute.type: 0,
        Window.type: 1,
    }


def test_capability_alone_makes_an_operator_spec_different_for_updates() -> None:
    spec = Window.__operator_spec__

    assert _same(spec, replace(spec, passive_max_emissions=1))
    assert not _same(spec, replace(spec, passive_max_emissions=None))
