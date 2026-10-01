"""An operator method composing its own phases with the generic primitive.

Scope, stated exactly: the engine does not schedule operator phases. The
operator's ``push`` calls ``run_phases`` itself, like library code would; the
runtime still calls ``push`` / ``end_input`` / ``finish`` / ``close`` as
before, and the operator alone owns what it retains between pushes. The test
shows that the primitive's per-call cleanup leaves that retained state intact
and that a phase failure surfaces through the unchanged operator attribution.
"""

import threading
from typing import Any, List, Optional

import pytest
from pydantic import Field
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.errors import ActiveRunError
from roboflow_workflows.execution_engine.v2.operators import (
    Operator,
    OperatorParams,
    OperatorPort,
    OperatorPulse,
)
from roboflow_workflows.execution_engine.v2.operators.contract import (
    OperatorDeclarationError,
)
from roboflow_workflows.execution_engine.v2.phases import (
    PhaseFailure,
    phase,
    read_phase_graph,
    run_phases,
)

from tests.unit_tests.execution_engine.v2.test_active_runtime import (
    WAIT,
    Collector,
    Log,
    Scripted,
    active,
    emit,
    group,
    source,
)

OPERATORS: List["Pairing"] = []
"""Every constructed ``Pairing``, so a test can inspect what it retained."""


class Pairing(Operator):
    """Emits (previous, current) for each present arrival after the first.

    present ──> pairs ──┐
       └────────────────┴─> emission      (result: pulses and the newest arrival)
    """

    type = "test/pairing@v1"
    input_roles = ("input",)

    class Params(OperatorParams):
        fail_at: Optional[float] = Field(
            default=None, description="Arrival value that makes a phase raise."
        )

    @classmethod
    def plan_ports(cls, name, params, inputs):
        kinds = inputs[0].kinds
        return {"previous": OperatorPort(*kinds), "current": OperatorPort(*kinds)}

    def __init__(self, **arguments: Any) -> None:
        super().__init__(**arguments)
        self.retained = None
        self.phases: List[str] = []
        self.closed = 0
        OPERATORS.append(self)

    @phase
    def present(self, *, arrivals):
        present = [
            item for item in arrivals if not item.entry.is_effectively_filtered()
        ]
        if any(item.entry.values[()] == self.params.fail_at for item in present):
            raise ValueError("unpairable arrival")
        return present

    @phase
    def pairs(self, *, present, retained):
        chain = ([retained] if retained is not None else []) + present
        return list(zip(chain, chain[1:]))

    @phase
    def emission(self, *, pairs, present):
        pulses = [
            OperatorPulse(
                ports={"previous": previous.entry, "current": current.entry},
                causes=(previous.pulse, current.pulse),
            )
            for previous, current in pairs
        ]
        return {"pulses": pulses, "newest": present[-1] if present else None}

    def push(self, arrivals):
        emission = run_phases(
            self,
            PAIRING_PHASES,
            {"arrivals": arrivals, "retained": self.retained},
            on_phase=self.phases.append,
        )
        if emission["newest"] is not None:
            self.retained = emission["newest"]
        return emission["pulses"]

    def end_input(self, name):
        return []

    def finish(self, reason):
        return []

    def close(self):
        self.closed += 1


PAIRING_PHASES = read_phase_graph(
    Pairing, external=("arrivals", "retained"), fail=OperatorDeclarationError
)

CATALOGUE = Catalogue([], sources=[Scripted], operators=[Pairing])


def start(feeds: dict, **operator_params: Any):
    definition = active(
        [source("a")],
        [],
        [
            group(
                "P",
                "$operators.pairs.current",
                previous="$operators.pairs.previous",
                current="$operators.pairs.current",
            )
        ],
    )
    definition["operators"] = [
        {
            "type": Pairing.type,
            "name": "pairs",
            "inputs": {"value": "$sources.a.value"},
            **operator_params,
        }
    ]
    started = threading.Event()
    started.set()
    session = compile_workflow(definition, catalogue=CATALOGUE).create_session(
        resources={"feeds": feeds, "log": Log(), "started": started}
    )
    collected = Collector()
    OPERATORS.clear()

    run = session.start(handlers=collected.handlers("P"))

    return run, collected


def test_push_composes_its_phases_and_keeps_what_the_operator_retains() -> None:
    run, collected = start({"a": [emit(value=1.0), emit(value=2.0), emit(value=3.0)]})

    assert run.wait(WAIT)
    assert collected.rows("P") == [
        {"previous": 1.0, "current": 2.0},
        {"previous": 2.0, "current": 3.0},
    ]
    (operator,) = OPERATORS
    assert operator.phases == ["present", "pairs", "emission"] * 3
    # Cleanup released the call's intermediates, not the operator's own state.
    assert operator.retained.entry.values[()] == 3.0
    assert operator.retained.pulse.sequence == 2
    assert collected.results["P"][1].causes[0] == collected.results["P"][0].causes[1]
    assert operator.closed == 1
    assert run.operator_counters["pairs"].closed


def test_phase_failure_in_push_keeps_the_operator_attribution_and_closes_once() -> None:
    run, collected = start(
        {"a": [emit(value=1.0), emit(value=2.0), emit(value=3.0)]}, fail_at=2.0
    )

    with pytest.raises(ActiveRunError) as caught:
        run.wait(WAIT)

    failure = caught.value
    assert (failure.stage, failure.operator, failure.source, failure.pulse) == (
        "operator",
        "pairs",
        "a",
        1,
    )
    assert "push raised PhaseFailure: phase 'present' failed" in str(failure)
    assert isinstance(failure.__cause__, PhaseFailure)
    assert failure.__cause__.phase == "present"
    (operator,) = OPERATORS
    assert operator.retained.entry.values[()] == 1.0
    assert operator.closed == 1
    assert collected.rows("P") == []
