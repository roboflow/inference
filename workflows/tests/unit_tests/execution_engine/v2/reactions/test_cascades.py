"""Event cascades through machines and handlers: drain, cancel and batches.

A graceful stop drains every accepted cascade (async handler -> machine
setter -> machine event -> async handler) before the run is done; a cancel
discards queued work and wakes the drain. Batch calls name the member whose
source a per-source machine uses.
"""

import threading
from typing import Any

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.data import (
    EntryMetadata,
    InputValue,
    SampleContext,
)
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.errors import StepExecutionError
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND
from roboflow_workflows.execution_engine.v2.reactions.runtime import session_reactions

from .test_machines_runtime import REVIEW, Journal, inspection, start
from .test_runtime import WAIT, eventually
from .test_signals import lifecycle


def test_a_graceful_stop_drains_an_async_cascade_started_before_it():
    journal = Journal()
    decide_held = journal.hold("pick", 0.9)
    audit_held = journal.hold("audit", "approved")
    gate = threading.Event()
    definition = lifecycle(inspection(decide="async", audit="async"), ended="sync")
    run = start(definition, journal, {"cam_a": [0.9, gate]})
    assert journal.started("pick", 0.9).wait(WAIT)
    run.stop()
    gate.set()
    # The main flow ended (``ended`` ran) while decide still holds its event.
    assert journal.started("ended", "ended").wait(WAIT)
    assert not run.wait(0.05)
    decide_held.set()
    # decide's setter emits ``decided`` now, during the drain; it is accepted.
    assert journal.started("audit", "approved").wait(WAIT)
    assert not run.wait(0.05)
    audit_held.set()
    assert run.wait(WAIT)

    assert run.state == "finished"
    assert run.machine_state("inspection_state", source_id="cam_a") == ("approved", 2)
    counters = run.reaction_counters
    for name in ("$handlers.decide", "$handlers.audit"):
        assert (counters[name].completed, counters[name].discarded) == (1, 0)
    assert all(item.status == "completed" for item in run.reaction_outcomes())


def test_cancel_discards_queued_work_wakes_the_drain_and_publishes_no_ended():
    journal = Journal()
    decide_held = journal.hold("pick", 0.9)
    gate = threading.Event()
    definition = lifecycle(inspection(decide="async"), ended="sync")
    definition["state_machines"][0]["transitions"].append(
        {
            "name": "again",
            "from": ["reviewing"],
            "to": "reviewing",
            "on": "$steps.inspection.events.review_requested",
            "emit": {
                "name": "decision_requested",
                "fields": {"score": "$event.score", "frame": "$event.frame"},
            },
        }
    )
    run = start(definition, journal, {"cam_a": [0.9, 0.8, gate]})
    assert journal.started("pick", 0.9).wait(WAIT)
    eventually(lambda: run.reaction_counters["$handlers.decide"].pending == 1)
    run.cancel()
    gate.set()
    decide_held.set()
    assert run.wait(WAIT)

    assert run.state == "cancelled"
    decide = run.reaction_counters["$handlers.decide"]
    assert (decide.discarded, decide.pending) == (1, 0)
    assert journal.values("pick") == [0.9]
    assert journal.values("ended") == []
    assert journal.values("audit") == []


class BatchInspect(Block):
    """A batch-delivering block emitting ``review_requested`` for every member."""

    type = "test/batch_inspect@v1"
    outputs = {"score": Output(FLOAT_KIND)}
    events = {"review_requested": REVIEW}

    class Params(BlockParams):
        # Ref metadata strings are values; pyflakes treats them as forward refs.
        scores: Ref(FLOAT_KIND, batch="always")  # noqa: F821

    def run(self, scores):
        for index, score in zip(scores.indices, scores):
            self.emit("review_requested", at=index, score=score, frame=score)
        return [{"score": score} for score in scores]


BATCH = {
    "version": "2.0",
    "inputs": [{"type": "WorkflowBatchInput", "name": "scores", "kind": ["float"]}],
    "steps": [
        {"type": BatchInspect.type, "name": "inspection", "scores": "$inputs.scores"}
    ],
    "outputs": [
        {"type": "JsonField", "name": "score", "selector": "$steps.inspection.score"}
    ],
    "state_machines": [
        {
            "name": "toggle",
            "scope": "source",
            "initial_state": "off",
            "states": ["off", "on"],
            "transitions": [
                {"name": name, "from": [source], "to": target,
                 "on": "$steps.inspection.events.review_requested"}
                for name, source, target in (("on", "off", "on"), ("off", "on", "off"))
            ],
        }
    ],
}  # fmt: skip


def scores(*sources: Any) -> dict:
    sample = {(position,): None if name is None else SampleContext(source_id=name)
              for position, name in enumerate(sources)}  # fmt: skip
    value = InputValue(
        [float(position) for position in range(len(sources))],
        metadata=EntryMetadata(sample=sample),
    )

    return {"scores": value}


def test_each_member_of_a_batch_call_moves_the_machine_of_its_own_source():
    plan = compile_workflow(BATCH, catalogue=Catalogue([BatchInspect]))
    session = plan.create_session()
    session.run(scores("a", "b", "a"))
    machines = session_reactions(session)

    assert machines.machine_state("toggle", source_id="a") == ("off", 2)
    assert machines.machine_state("toggle", source_id="b") == ("on", 1)
    with pytest.raises(StepExecutionError, match="has no source"):
        session.run(scores("b", None))
    assert machines.machine_state("toggle", source_id="b") == ("off", 2)
