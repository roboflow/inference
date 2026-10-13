"""``v2/continue_if@v1`` lane records in actual active runs.

A grouped source emits members with their own source and observed time, as a
collecting source may. The gate selects a nested workflow::

    source.pair ─▶ v2/continue_if@v1 (stop_delay > 0) ─▶ inner_workflow(echo)

With ``stop_delay > 0`` each lane's observed times must not go backward, and
a new lane beyond ``max_sources`` fails. Both fail the pulse: a decision is
never silently answered from lost history.
"""

import threading
from fractions import Fraction
from typing import List, Optional, Sequence

import pytest
from roboflow_workflows.v2.blocks.control.conditions import (
    ContinueIfQueryBlock,
    HoldCapacityError,
    ObservationOrderError,
)
from roboflow_workflows.v2.data.batch import (
    Axis,
    Batch,
    EntryLayout,
    EntryMetadata,
    InputValue,
    SampleContext,
    TemporalContext,
    Timestamp,
)
from roboflow_workflows.v2.data.kinds import FLOAT_KIND
from roboflow_workflows.v2.engine.active.execution import ENGINE_CLOCK_ID
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import ActiveRunError
from roboflow_workflows.v2.engine.pipelining.options import PipelineOptions
from roboflow_workflows.v2.engine.plan import CompileOptions
from roboflow_workflows.v2.engine.sources import Emission, SourceOutput

from tests.v2.support.execution.blocks import Echo
from tests.v2.support.runtime.active import (
    WAIT,
    Collector,
    FeedSource,
    Log,
    active,
    group,
)

MODES = ["run", "phases"]
SCORE_AT_LEAST_HALF = {
    "type": "StatementGroup",
    "statements": [
        {
            "type": "BinaryStatement",
            "left_operand": {"type": "DynamicOperand", "operand_name": "score"},
            "comparator": {"type": "(Number) >="},
            "right_operand": {"type": "StaticOperand", "value": 0.5},
        }
    ],
}


class Members(FeedSource):
    """One grouped port: a batch of scores per pulse."""

    type = "test/members@v1"
    outputs = {
        "pair": SourceOutput(
            FLOAT_KIND, layout=EntryLayout((Axis(id="items", kind="dynamic_nesting"),))
        )
    }


def _members(
    scores: Sequence[float], sources: Sequence[str], seconds: Sequence[float]
) -> Emission:
    def stamp(value: float) -> Timestamp:
        return Timestamp(round(value * 1000), Fraction(1, 1000), ENGINE_CLOCK_ID)

    metadata = EntryMetadata(
        sample={(n,): SampleContext(source_id=s) for n, s in enumerate(sources)},
        temporal={
            (n,): TemporalContext(observed_coverage=stamp(t))
            for n, t in enumerate(seconds)
        },
    )
    emission = Emission({"pair": InputValue(Batch.of(list(scores)), metadata=metadata)})

    return emission


def _definition(*, stop_delay: float, max_sources: int) -> dict:
    gate = {
        "type": ContinueIfQueryBlock.type,
        "name": "gate",
        "condition_statement": SCORE_AT_LEAST_HALF,
        "evaluation_parameters": {"score": "$sources.cams.pair"},
        "next_steps": ["$steps.nested"],
        "stop_delay": stop_delay,
        "max_sources": max_sources,
    }
    nested = {
        "type": "roboflow_core/inner_workflow@v1",
        "name": "nested",
        "workflow_definition": {
            "version": "2.0",
            "inputs": [{"type": "WorkflowBatchInput", "name": "x"}],
            "steps": [{"type": Echo.type, "name": "echo", "value": "$inputs.x"}],
            "outputs": [
                {"type": "JsonField", "name": "y", "selector": "$steps.echo.value"}
            ],
        },
        "parameter_bindings": {"x": "$sources.cams.pair"},
    }
    definition = active(
        [{"type": Members.type, "name": "cams", "feed": "cams"}],
        [gate, nested],
        [group("out", "$sources.cams.pair", selected="$steps.nested.y")],
    )

    return definition


def _run(
    pulses: List[Emission],
    *,
    stop_delay: float,
    max_sources: int = 256,
    mode: str = "run",
) -> tuple:
    """Runs the pulses; returns selected values per pulse and the run failure."""
    plan = compile_workflow(
        _definition(stop_delay=stop_delay, max_sources=max_sources),
        catalogue=Catalogue([ContinueIfQueryBlock, Echo], sources=[Members]),
        options=CompileOptions(block_execution=mode),
    )
    started = threading.Event()
    session = plan.create_session(
        {"feeds": {"cams": pulses}, "started": started, "log": Log()}
    )
    collected = Collector()
    failure: Optional[ActiveRunError] = None
    run = session.start(
        handlers=collected.handlers("out"), pipeline=PipelineOptions(max_in_flight=3)
    )
    started.set()
    try:
        assert run.wait(WAIT)
    except ActiveRunError as error:
        failure = error
    finally:
        run.stop()
        session.close()
    selected = [
        [row["selected"] for row in result.rows()]
        for result in collected.results.get("out", [])
    ]

    return selected, failure


def _block_error(failure: ActiveRunError) -> BaseException:
    cause = failure.__cause__
    while cause is not None and not isinstance(
        cause, (HoldCapacityError, ObservationOrderError)
    ):
        cause = cause.__cause__

    return cause


CAM_A_TRUE_AT_0 = _members([1.0], ["cam-a"], [0])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cam_b_first", [True, False])
def test_a_new_lane_beyond_capacity_fails_the_pulse_in_any_member_order(
    mode: str, cam_b_first: bool
) -> None:
    # cam-a's false at 1 is inside its own hold (1 - 0 <= 2). cam-b's time 10
    # must not expire cam-a's hold, whichever member comes first.
    members = [(1.0, "cam-b", 10), (0.0, "cam-a", 1)]
    if not cam_b_first:
        members.reverse()
    scores, sources, seconds = zip(*members)

    selected, failure = _run(
        [CAM_A_TRUE_AT_0, _members(scores, sources, seconds)],
        stop_delay=2.0,
        max_sources=1,
        mode=mode,
    )

    assert selected == [[1.0]]
    assert failure is not None and failure.pulse == 1
    error = _block_error(failure)
    assert isinstance(error, HoldCapacityError)
    assert "lane ('source', 'cam-b') would be lane 2 of max_sources=1" in str(error)


@pytest.mark.parametrize("cam_b_first", [True, False])
def test_within_capacity_every_member_keeps_its_own_lanes_decision(
    cam_b_first: bool,
) -> None:
    members = [(1.0, "cam-b", 10), (0.0, "cam-a", 1)]
    if not cam_b_first:
        members.reverse()
    scores, sources, seconds = zip(*members)

    selected, failure = _run(
        [CAM_A_TRUE_AT_0, _members(scores, sources, seconds)],
        stop_delay=2.0,
        max_sources=2,
    )

    # Both members continue: cam-b on its true, cam-a on its hold.
    assert failure is None
    assert selected == [[1.0], list(scores)]


@pytest.mark.parametrize("mode", MODES)
def test_an_older_observation_after_newer_ones_fails_the_pulse(mode: str) -> None:
    selected, failure = _run(
        [
            _members([1.0, 1.0, 1.0], ["cam-a"] * 3, [0, 2, 3]),
            _members([0.0], ["cam-a"], [1]),
        ],
        stop_delay=5.0,
        mode=mode,
    )

    assert selected == [[1.0, 1.0, 1.0]]
    assert failure is not None and failure.pulse == 1
    error = _block_error(failure)
    assert isinstance(error, ObservationOrderError)
    assert "time 1 s after 3 s" in str(error)


def test_the_order_guard_survives_an_expired_hold() -> None:
    selected, failure = _run(
        [
            CAM_A_TRUE_AT_0,
            _members([0.0], ["cam-a"], [10]),
            _members([1.0], ["cam-a"], [1]),
        ],
        stop_delay=2.0,
    )

    assert selected == [[1.0], [None]]
    assert failure is not None and failure.pulse == 2
    assert isinstance(_block_error(failure), ObservationOrderError)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("first_frame", [[1.0, 0.0], [0.0, 1.0]])
def test_members_of_one_frame_are_held_only_by_earlier_frames(
    mode: str, first_frame: List[float]
) -> None:
    selected, failure = _run(
        [
            _members(first_frame, ["cam-a"] * 2, [0, 0]),
            _members([0.0, 0.0], ["cam-a"] * 2, [1, 1]),
            _members([0.0], ["cam-a"], [9]),
        ],
        stop_delay=5.0,
        mode=mode,
    )

    assert failure is None
    expected_first = [score if score else None for score in first_frame]
    assert selected == [expected_first, [0.0, 0.0], [None]]
