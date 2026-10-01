"""Phased blocks inside active runs: per-pulse isolation, gates and attribution.

Sources are the scripted feeds of ``test_active_runtime``; each pulse is one
call of the phased ``Diamond`` block, compiled in run and phase mode.
"""

import gc
import threading
from typing import List

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.errors import (
    ActiveRunError,
    StepExecutionError,
)
from roboflow_workflows.execution_engine.v2.plan import CompileOptions

from tests.unit_tests.execution_engine.v2.execution.blocks import ContinueIf
from tests.unit_tests.execution_engine.v2.execution.test_phase_execution import Diamond
from tests.unit_tests.execution_engine.v2.test_active_runtime import (
    WAIT,
    Collector,
    Log,
    Scripted,
    active,
    emit,
    group,
    source,
    step,
)

CATALOGUE = Catalogue([ContinueIf, Diamond], sources=[Scripted])

MODES = ("run", "phases")


def started_session(definition: dict, feeds: dict, *, mode: str):
    plan = compile_workflow(
        definition,
        catalogue=CATALOGUE,
        options=CompileOptions(block_execution=mode),
    )
    started = threading.Event()
    started.set()
    session = plan.create_session(
        resources={"feeds": feeds, "log": Log(), "started": started}
    )

    return session


def gated_diamond() -> dict:
    definition = active(
        [source("a")],
        [
            step(
                ContinueIf,
                "admit",
                value="$sources.a.value",
                threshold=0.0,
                next_steps=["$steps.diamond"],
            ),
            step(Diamond, "diamond", value="$sources.a.value"),
        ],
        [group("A", "$sources.a.value", total="$steps.diamond.total")],
    )

    return definition


def phases_of(result) -> List[str]:
    names = [event["phase"] for event in result.trace if event["event"] == "phase"]

    return names


@pytest.mark.parametrize("mode", MODES)
def test_each_admitted_pulse_runs_every_phase_once_and_denied_pulses_none(
    mode,
) -> None:
    feeds = {"a": [emit(value=1.0, pts=10), emit(value=-1.0, pts=20), emit(value=2.0)]}
    session = started_session(gated_diamond(), feeds, mode=mode)
    collected = Collector()

    run = session.start(handlers=collected.handlers("A"))

    assert run.wait(WAIT)
    assert collected.rows("A") == [{"total": 4.0}, {"total": None}, {"total": 7.0}]
    block = session.instances[("diamond",)]
    assert block.calls == [
        ("base", 1.0),
        ("left", 1.0),
        ("right", 1.0),
        ("total", 2.0),
        ("base", 2.0),
        ("left", 2.0),
        ("right", 2.0),
        ("total", 4.0),
    ]
    expected = ["base", "left", "right", "total"] if mode == "phases" else []
    assert [phases_of(result) for result in collected.results["A"]] == [
        expected,
        [],
        expected,
    ]
    gc.collect()
    assert all(reference() is None for reference in block.intermediates)


@pytest.mark.parametrize("mode", MODES)
def test_phase_failure_is_attributed_to_its_pulse_step_and_phase(mode) -> None:
    feeds = {"a": [emit(value=1.0), emit(value=-1.0), emit(value=2.0)]}
    definition = active(
        [source("a")],
        [step(Diamond, "diamond", value="$sources.a.value")],
        [group("A", "$sources.a.value", total="$steps.diamond.total")],
    )
    session = started_session(definition, feeds, mode=mode)
    collected = Collector()

    run = session.start(handlers=collected.handlers("A"))

    with pytest.raises(ActiveRunError) as caught:
        run.wait(WAIT)
    failure = caught.value
    assert (failure.stage, failure.source, failure.pulse) == ("step", "a", 1)
    assert (failure.step_path, failure.phase) == (("diamond",), "base")
    assert "phase 'base'" in str(failure)
    assert isinstance(failure.__cause__, StepExecutionError)
    assert failure.__cause__.phase == "base"
    assert collected.rows("A") == [{"total": 4.0}]
