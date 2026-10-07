"""Compiling ``state_machines``: schema, scopes, setters and cascade checks.

Draft example 05, with its draft blocks replaced by fixture blocks::

    $steps.inspection.events.review_requested
        └─(start_review: idle → reviewing)─▶ inspection_state.decision_requested
             └─ handler decide ─(sets finish_review: reviewing → approved|rejected)
                  ─▶ inspection_state.decided ─▶ handler audit_decision
    $system.events.started ─(lifecycle.start)─▶ (no event)

Compilation only: these tests prove plan shapes and rejections, not run time.
"""

import copy
from dataclasses import replace
from typing import Any, Dict, List

import pytest
from roboflow_workflows.execution_engine.v2.blocks.state_machine import (
    StateMachineSetBlock,
)
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    KindMismatchError,
    NestedWorkflowError,
    SelectorError,
    WorkflowCompileError,
)
from roboflow_workflows.execution_engine.v2.events import Event
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND, STRING_KIND
from roboflow_workflows.execution_engine.v2.plan import (
    CompiledWorkflow,
    requests_managed_state,
)
from roboflow_workflows.execution_engine.v2.reactions import (
    STATE_MACHINE_SET_TYPE,
    SYSTEM_EVENTS,
    EventOrigin,
    PlannedMachine,
    PlannedTransition,
)
from roboflow_workflows.execution_engine.v2.sources import (
    Source,
    SourceOutput,
    SourceParams,
)


class Inspection(Block):
    type = "test/inspection_trigger@v1"
    outputs = {"seen": Output(FLOAT_KIND)}
    events = {
        "review_requested": Event({"frame": FLOAT_KIND, "score": FLOAT_KIND}),
        "session_ended": Event({}),
    }

    class Params(BlockParams):
        frame: Ref(FLOAT_KIND)

    def run(self, frame):
        return {"seen": frame}


class Review(Block):
    type = "test/review_decision@v1"
    outputs = {"next_state": Output(STRING_KIND)}

    class Params(BlockParams):
        image: Ref(FLOAT_KIND)

    def run(self, image):
        return {"next_state": "approved" if image > 0 else "rejected"}


class Audit(Block):
    type = "test/audit@v1"
    outputs = {"logged": Output(STRING_KIND)}

    class Params(BlockParams):
        state: Ref(STRING_KIND)

    def run(self, state):
        return {"logged": state}


class Video(Source):
    type = "test/video_source@v1"
    outputs = {"image": SourceOutput(FLOAT_KIND)}

    class Params(SourceParams):
        pass

    def open(self) -> None:
        pass

    def read(self):
        return None

    def close(self) -> None:
        pass


CATALOGUE = Catalogue(
    [Inspection, Review, Audit, StateMachineSetBlock], sources=[Video]
)
REQUESTED = EventOrigin("step", "review_requested", ("inspection",))
ENDED = EventOrigin("step", "session_ended", ("inspection",))


def setter(machine="inspection_state", transition="finish_review", **extra):
    step = {
        "type": STATE_MACHINE_SET_TYPE,
        "name": "save",
        "machine": machine,
        "transition": transition,
        "next_state": "$steps.review.next_state",
    }
    step.update(extra)

    return step


def decide_workflow(*steps: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "inputs": [{"name": "image", "kind": ["float"]}],
        "steps": [
            {"type": Review.type, "name": "review", "image": "$inputs.image"},
            *(steps or [setter()]),
        ],
        "outputs": [
            {"type": "JsonField", "name": "state", "selector": "$steps.save.state"}
        ],
    }


def decide(on="$state_machines.inspection_state.events.decision_requested", **extra):
    declared = {
        "name": "decide",
        "on": on,
        "bindings": {"image": "$event.image"},
        "workflow": decide_workflow(),
    }
    declared.update(extra)

    return declared


def inspection_machine(**extra: Any) -> Dict[str, Any]:
    machine = {
        "name": "inspection_state",
        "scope": "source",
        "initial_state": "idle",
        "states": ["idle", "reviewing", "approved", "rejected"],
        "transitions": [
            {
                "name": "start_review",
                "from": ["idle"],
                "on": "$steps.inspection.events.review_requested",
                "to": "reviewing",
                "emit": {
                    "name": "decision_requested",
                    "fields": {"image": "$event.frame"},
                },
            },
            {
                "name": "finish_review",
                "from": ["reviewing"],
                "handler": "decide",
                "to": ["approved", "rejected"],
                "emit": {"name": "decided", "fields": {"state": "$transition.to"}},
            },
            {
                "name": "reset",
                "from": ["approved", "rejected"],
                "on": "$steps.inspection.events.session_ended",
                "to": "idle",
            },
        ],
    }
    machine.update(extra)

    return machine


def example_05(**extra: Any) -> Dict[str, Any]:
    """Draft example 05 with fixture block types."""
    definition = {
        "version": "2.0",
        "sources": [{"type": Video.type, "name": "camera"}],
        "steps": [
            {
                "type": Inspection.type,
                "name": "inspection",
                "frame": "$sources.camera.image",
            }
        ],
        "outputs": [
            {
                "type": "OutputGroup",
                "name": "decisions",
                "anchor": "$handlers.decide.state",
                "outputs": [
                    {
                        "type": "JsonField",
                        "name": "state",
                        "selector": "$handlers.decide.state",
                    }
                ],
            }
        ],
        "state_machines": [
            inspection_machine(),
            {
                "name": "lifecycle",
                "scope": "global",
                "initial_state": "created",
                "states": ["created", "running"],
                "transitions": [
                    {
                        "name": "start",
                        "from": ["created"],
                        "on": "$system.events.started",
                        "to": "running",
                    }
                ],
            },
        ],
        "handlers": [
            decide(),
            {
                "name": "audit_decision",
                "on": "$state_machines.inspection_state.events.decided",
                "execution": {
                    "mode": "async",
                    "queue": {"max_depth": 8, "overflow": "synchronous"},
                },
                "bindings": {"state": "$event.state"},
                "workflow": {
                    "inputs": [{"name": "state", "kind": ["string"]}],
                    "steps": [
                        {"type": Audit.type, "name": "audit", "state": "$inputs.state"}
                    ],
                    "outputs": [],
                },
            },
        ],
    }
    definition.update(extra)

    return definition


def passive(**extra: Any) -> Dict[str, Any]:
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "frame"}],
        "steps": [
            {"type": Inspection.type, "name": "inspection", "frame": "$inputs.frame"}
        ],
        "outputs": [
            {"type": "JsonField", "name": "seen", "selector": "$steps.inspection.seen"}
        ],
    }
    definition.update(extra)

    return definition


def machine(name="m", scope="global", transitions=(), states=("a", "b"), **extra):
    declared = {
        "name": name,
        "scope": scope,
        "initial_state": states[0],
        "states": list(states),
        "transitions": list(transitions),
    }
    declared.update(extra)

    return declared


def fixed(name, on, source="a", target="b", emit=None, fields=None):
    transition = {"name": name, "from": [source], "on": on, "to": target}
    if emit is not None:
        transition["emit"] = {"name": emit, "fields": fields or {}}

    return transition


def audit(name: str, on: str, state: str = "$event.state") -> Dict[str, Any]:
    """Handler running ``Audit`` on one string; ``state`` may be a literal."""
    declared = {
        "name": name,
        "on": on,
        "bindings": {"state": state},
        "workflow": {
            "inputs": [{"name": "state", "kind": ["string"]}],
            "steps": [{"type": Audit.type, "name": "audit", "state": "$inputs.state"}],
            "outputs": [],
        },
    }

    return declared


def nested(name: str, definition: Dict[str, Any]) -> Dict[str, Any]:
    step = {
        "type": "roboflow_core/inner_workflow@v1",
        "name": name,
        "workflow_definition": copy.deepcopy(definition),
        "parameter_bindings": {"frame": "$inputs.frame"},
    }

    return step


def compile_(definition: Dict[str, Any]) -> CompiledWorkflow:
    return compile_workflow(definition, catalogue=CATALOGUE)


# --- Draft example 05 -------------------------------------------------------------


def test_draft_example_05_compiles_to_machines_handlers_and_edges() -> None:
    plan = compile_(example_05())
    reactions = plan.reactions

    inspection, lifecycle = reactions.machines
    assert inspection.path == ("inspection_state",)
    assert inspection.scope == "source"
    assert [item.name for item in inspection.transitions] == [
        "start_review",
        "finish_review",
        "reset",
    ]
    start, finish, reset = inspection.transitions
    assert start.trigger == REQUESTED
    assert start.targets == ("reviewing",)
    assert start.fields == {"image": ("event", "frame")}
    assert finish.handler == ("decide",)
    assert finish.targets == ("approved", "rejected")
    assert finish.fields == {"state": ("transition", "to")}
    assert reset.emits is None
    assert inspection.events["decision_requested"].kind_names("image") == ("float",)
    assert inspection.events["decided"].kind_names("state") == ("string",)
    assert lifecycle.transitions[0].trigger == EventOrigin("system", "started")

    decide_handler = reactions.handler(("decide",))
    assert decide_handler.origin == inspection.origin("decision_requested")
    assert reactions.causes(("decide",)) == {inspection.origin("decided")}
    assert reactions.causes(("audit_decision",)) == frozenset()
    assert reactions.setter(("decide",), "inspection_state", "finish_review") is finish
    assert STATE_MACHINE_SET_TYPE in {
        step.block_type for step in decide_handler.plan.steps
    }
    assert reactions.groups_of(("decide",))[0].fields == {"state": "state"}

    assert reactions.transitions_for(REQUESTED) == (start,)
    assert reactions.transitions_for(ENDED) == (reset,)
    assert reactions.transitions_for(EventOrigin("system", "started")) == (
        lifecycle.transitions[0],
    )
    assert reactions.emitting_steps == {("inspection",)}
    assert reactions.bound_fields(REQUESTED) == {"frame"}
    assert reactions.subscribed(ENDED)
    assert not reactions.subscribed(EventOrigin("system", "ended"))
    assert reactions.declared_event(inspection.origin("decided")) is (
        inspection.events["decided"]
    )
    assert reactions.declared_event(EventOrigin("system", "ended")) is (
        SYSTEM_EVENTS["ended"]
    )
    assert requests_managed_state(plan)

    described = plan.describe()["reactions"]["state_machines"]
    assert described["inspection_state"]["transitions"]["finish_review"] == {
        "from": ["reviewing"],
        "to": ["approved", "rejected"],
        "on": None,
        "handler": "decide",
        "emit": "$state_machines.inspection_state.events.decided",
        "fields": {"state": "$transition.to"},
    }


def test_fixed_transition_selection_is_deterministic_across_compiles() -> None:
    def definition():
        return passive(
            state_machines=[
                machine(
                    "first",
                    transitions=[
                        fixed("from_b", "$steps.inspection.events.session_ended", "b"),
                        fixed("from_a", "$steps.inspection.events.session_ended", "a"),
                    ],
                ),
                machine(
                    "second",
                    transitions=[
                        fixed("go", "$steps.inspection.events.session_ended"),
                    ],
                ),
            ]
        )

    orders = [
        [item.label for item in compile_(definition()).reactions.transitions_for(ENDED)]
        for _ in range(2)
    ]

    assert orders == [["first.from_b", "first.from_a", "second.go"]] * 2


def test_machine_only_subscribers_keep_the_event_and_union_the_demand() -> None:
    transitions = [
        fixed(
            "open",
            "$steps.inspection.events.review_requested",
            emit="opened",
            fields={"frame": "$event.frame", "by": "$transition.name", "n": 3},
        )
    ]
    plan = compile_(passive(state_machines=[machine(transitions=transitions)]))

    assert plan.reactions.handlers_for(REQUESTED) == ()
    assert plan.reactions.subscribed(REQUESTED)
    assert plan.reactions.emitting_steps == {("inspection",)}
    assert plan.reactions.bound_fields(REQUESTED) == {"frame"}
    assert plan.reactions.machines[0].events["opened"].describe()["fields"] == {
        "frame": ["float"],
        "by": ["string"],
        "n": ["integer"],
    }

    audit = {
        "name": "audit",
        "on": "$steps.inspection.events.review_requested",
        "bindings": {"image": "$event.score"},
        "workflow": {
            "inputs": [{"name": "image", "kind": ["float"]}],
            "steps": [
                {"type": Review.type, "name": "review", "image": "$inputs.image"}
            ],
            "outputs": [
                {
                    "type": "JsonField",
                    "name": "s",
                    "selector": "$steps.review.next_state",
                }
            ],
        },
    }
    both = compile_(
        passive(state_machines=[machine(transitions=transitions)], handlers=[audit])
    )

    assert both.reactions.bound_fields(REQUESTED) == {"frame", "score"}


def test_literal_and_shared_emitted_fields_get_their_kinds() -> None:
    transitions = [
        fixed(
            "one",
            "$steps.inspection.events.review_requested",
            emit="changed",
            fields={"v": "$transition.to", "flag": True, "any": None},
        ),
        fixed(
            "two",
            "$steps.inspection.events.session_ended",
            source="b",
            target="a",
            emit="changed",
            fields={"v": 2, "flag": False, "any": "x"},
        ),
    ]
    plan = compile_(passive(state_machines=[machine(transitions=transitions)]))

    assert plan.reactions.machines[0].events["changed"].describe()["fields"] == {
        "v": ["string", "integer"],
        "flag": ["boolean"],
        "any": ["*"],
    }


def test_event_kinds_propagate_through_machine_events_to_handler_bindings() -> None:
    definition = example_05()
    definition["handlers"].append(
        audit(
            "audit",
            "$state_machines.inspection_state.events.decision_requested",
            state="$event.image",
        )
    )

    with pytest.raises(KindMismatchError, match=r"\$event.image.*\['float'\]"):
        compile_(definition)


# --- Cycles -------------------------------------------------------------------------


def test_state_graph_cycles_on_independent_events_compile() -> None:
    transitions = [
        fixed("alert", "$steps.inspection.events.review_requested", "a", "b", "up"),
        fixed("calm", "$steps.inspection.events.session_ended", "b", "a", "down"),
    ]
    plan = compile_(passive(state_machines=[machine(transitions=transitions)]))

    assert [item.name for item in plan.reactions.machines[0].transitions] == [
        "alert",
        "calm",
    ]


@pytest.mark.parametrize(
    "machines, handlers, message",
    [
        (
            [machine(transitions=[fixed("t", "$state_machines.m.events.x", emit="x")])],
            [],
            r"\$state_machines.m.events.x -> \(transition m.t\) -> "
            r"\$state_machines.m.events.x",
        ),
        (
            [
                machine(
                    "a",
                    transitions=[fixed("t", "$state_machines.b.events.y", emit="x")],
                ),
                machine(
                    "b",
                    transitions=[fixed("t", "$state_machines.a.events.x", emit="y")],
                ),
            ],
            [],
            r"cascade cycle: .*\(transition a.t\).*\(transition b.t\)",
        ),
        (
            [
                machine(
                    transitions=[
                        fixed(
                            "t",
                            "$state_machines.m.events.x",
                            emit="x",
                            fields={"f": "$event.f"},
                        )
                    ]
                )
            ],
            [],
            r"State machine events form an automatic cascade cycle: "
            r"\$state_machines.m.events.x -> \$state_machines.m.events.x",
        ),
        (
            [
                machine(
                    states=("a", "b", "c"),
                    transitions=[
                        fixed(
                            "start",
                            "$steps.inspection.events.session_ended",
                            emit="x",
                        ),
                        {
                            "name": "again",
                            "from": ["b"],
                            "handler": "loop",
                            "to": ["c"],
                            "emit": {"name": "x", "fields": {}},
                        },
                    ],
                )
            ],
            [
                {
                    "name": "loop",
                    "on": "$state_machines.m.events.x",
                    "workflow": {
                        "inputs": [],
                        "steps": [
                            {
                                "type": STATE_MACHINE_SET_TYPE,
                                "name": "save",
                                "machine": "m",
                                "transition": "again",
                                "next_state": "c",
                            }
                        ],
                        "outputs": [],
                    },
                }
            ],
            r"\$state_machines.m.events.x -> \(handler loop sets m.again\) -> "
            r"\$state_machines.m.events.x",
        ),
    ],
    ids=["self", "ping-pong", "payload-self", "handler-selected"],
)
def test_automatic_cascade_cycles_are_rejected(machines, handlers, message) -> None:
    with pytest.raises(WorkflowCompileError, match=message):
        compile_(passive(state_machines=machines, handlers=handlers))


# --- Schema -------------------------------------------------------------------------


def _first_transition(**changes: Any) -> List[Dict[str, Any]]:
    transition = fixed("t", "$steps.inspection.events.session_ended", emit="x")
    transition.update(changes)
    return [transition]


@pytest.mark.parametrize(
    "declared, error, message",
    [
        (
            machine(transitions=_first_transition(handler="h")),
            WorkflowCompileError,
            "exactly one of 'on'",
        ),
        (
            machine(transitions=[{"name": "t", "from": ["a"], "to": "b"}]),
            WorkflowCompileError,
            "exactly one of 'on'",
        ),
        (
            machine(transitions=_first_transition(to=["b"])),
            WorkflowCompileError,
            "must be one state for a transition fired by an event",
        ),
        (
            machine(transitions=_first_transition(to="z")),
            WorkflowCompileError,
            r"unknown states \['z'\]",
        ),
        (
            machine(transitions=_first_transition(**{"from": ["z"]})),
            WorkflowCompileError,
            r"unknown states \['z'\]",
        ),
        (machine(initial_state="z"), WorkflowCompileError, "is not one of the states"),
        (machine(states=["a", "a"]), WorkflowCompileError, "repeats a state"),
        (machine(scope="frame"), WorkflowCompileError, "scope must be one of"),
        (
            machine(transitions=_first_transition() * 2),
            WorkflowCompileError,
            r"duplicate transitions \['t'\]",
        ),
        (
            machine(transitions=_first_transition(color="red")),
            WorkflowCompileError,
            "unsupported keys",
        ),
        (
            machine(
                transitions=_first_transition(
                    emit={"name": "x", "fields": {"v": "$transition.time"}}
                )
            ),
            SelectorError,
            r"\$transition.time",
        ),
        (
            machine(
                transitions=_first_transition(
                    emit={"name": "x", "fields": {"v": "$inputs.frame"}}
                )
            ),
            SelectorError,
            "an emitted field is",
        ),
        (
            machine(
                transitions=_first_transition(
                    emit={"name": "x", "fields": {"v": "$event.nope"}}
                )
            ),
            SelectorError,
            r"reads \$event.nope.*declares fields \[\]",
        ),
        (
            machine(
                transitions=[
                    *_first_transition(),
                    fixed(
                        "u",
                        "$steps.inspection.events.review_requested",
                        emit="x",
                        fields={"v": 1},
                    ),
                ]
            ),
            WorkflowCompileError,
            "one machine event has one field set",
        ),
        (
            machine(transitions=_first_transition(on="$steps.ghost.events.e")),
            SelectorError,
            "unknown step 'ghost'",
        ),
        (
            machine(transitions=_first_transition(on="$state_machines.ghost.events.e")),
            SelectorError,
            "unknown state machine 'ghost'",
        ),
        (
            machine(transitions=_first_transition(on="$state_machines.m.events.nope")),
            SelectorError,
            "emits no event 'nope'",
        ),
        (
            machine(transitions=_first_transition(on="$system.events.paused")),
            SelectorError,
            "unknown system event 'paused'",
        ),
    ],
)
def test_malformed_machines_are_rejected(declared, error, message) -> None:
    with pytest.raises(error, match=message):
        compile_(passive(state_machines=[declared]))


def test_handler_transitions_cannot_copy_event_fields() -> None:
    declared = inspection_machine()
    declared["transitions"][1]["emit"]["fields"] = {"image": "$event.image"}

    with pytest.raises(SelectorError, match="does not keep the payload"):
        compile_(example_05(state_machines=[declared]))


def test_overlapping_from_states_of_one_trigger_are_rejected() -> None:
    on = "$steps.inspection.events.session_ended"
    declared = machine(
        states=("a", "b", "c"),
        transitions=[
            {"name": "one", "from": ["a", "b"], "on": on, "to": "c"},
            {"name": "two", "from": ["b"], "on": on, "to": "a"},
        ],
    )

    with pytest.raises(WorkflowCompileError, match=r"from \['b'\]"):
        compile_(passive(state_machines=[declared]))


def test_duplicate_machine_names_and_unknown_handlers_are_rejected() -> None:
    with pytest.raises(WorkflowCompileError, match="duplicate state machine 'm'"):
        compile_(passive(state_machines=[machine(), machine()]))

    declared = inspection_machine()
    declared["transitions"][1]["handler"] = "ghost"
    with pytest.raises(SelectorError, match="handler of the same workflow"):
        compile_(example_05(state_machines=[declared]))


# --- System events ------------------------------------------------------------------


def test_system_events_need_an_active_run_and_global_machines() -> None:
    started = machine(
        transitions=[fixed("start", "$system.events.started")], scope="global"
    )
    with pytest.raises(SelectorError, match="system events belong to an active run"):
        compile_(passive(state_machines=[started]))

    on_ended = audit("bye", "$system.events.ended", state="done")
    with pytest.raises(SelectorError, match="system events belong to an active run"):
        compile_(passive(handlers=[on_ended]))

    per_source = dict(started, scope="source")
    with pytest.raises(SelectorError, match="is per source.*has no source"):
        compile_(
            example_05(state_machines=[per_source], handlers=[on_ended], outputs=[])
        )

    plan = compile_(
        example_05(state_machines=[started], handlers=[on_ended], outputs=[])
    )
    assert plan.reactions.handlers_for(EventOrigin("system", "ended"))[0].name == "bye"


# --- Setters ------------------------------------------------------------------------


def _with_decide_steps(*steps: Dict[str, Any]) -> Dict[str, Any]:
    definition = example_05()
    definition["handlers"][0]["workflow"] = decide_workflow(*steps)

    return definition


@pytest.mark.parametrize(
    "step, message",
    [
        (setter(machine="ghost"), "has no machine 'ghost'"),
        (setter(transition="ghost"), "has no transition 'ghost'"),
        (setter(transition="start_review"), "fires on .*review_requested"),
        (setter(next_state="idle"), r"may enter \['approved', 'rejected'\]"),
        (setter(machine="lifecycle", transition="start"), "fires on"),
    ],
    ids=["machine", "transition", "fixed", "target", "other-machine"],
)
def test_setter_steps_are_authorized_per_handler(step, message) -> None:
    with pytest.raises(WorkflowCompileError, match=message) as raised:
        compile_(_with_decide_steps(step))

    assert raised.value.step_path == ("decide", "save")


def test_setter_target_selectors_and_literal_targets_compile() -> None:
    plan = compile_(_with_decide_steps(setter(next_state="approved")))
    assert plan.reactions.causes(("decide",))


def test_only_the_named_handler_may_set_its_transition() -> None:
    definition = example_05()
    intruder = decide(name="intruder", workflow=decide_workflow())
    definition["handlers"].append(intruder)

    with pytest.raises(
        WorkflowCompileError, match="selected by handler 'decide'; handler 'intruder'"
    ):
        compile_(definition)


def test_main_flow_setters_are_rejected() -> None:
    definition = example_05()
    definition["steps"].append(
        setter(next_state="approved", trigger="$sources.camera.image")
    )

    with pytest.raises(WorkflowCompileError, match="main flow") as raised:
        compile_(definition)
    assert raised.value.step_path == ("save",)

    passive_setter = passive()
    passive_setter["steps"].append(
        setter(next_state="approved", trigger="$inputs.frame")
    )
    with pytest.raises(WorkflowCompileError, match="main flow"):
        compile_(passive_setter)


def test_setters_inside_children_of_a_handler_workflow_use_the_handler() -> None:
    inner = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "frame"}],
        "steps": [
            {"type": Review.type, "name": "review", "image": "$inputs.frame"},
            setter(),
        ],
        "outputs": [
            {"type": "JsonField", "name": "state", "selector": "$steps.save.state"}
        ],
    }
    workflow = {
        "inputs": [{"name": "image", "kind": ["float"]}],
        "steps": [
            {
                **nested("inner", inner),
                "parameter_bindings": {"frame": "$inputs.image"},
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": "state", "selector": "$steps.inner.state"}
        ],
    }
    definition = example_05()
    definition["handlers"][0]["workflow"] = workflow
    plan = compile_(definition)

    decide_plan = plan.reactions.handler(("decide",)).plan
    assert ("inner", "save") in {step.path for step in decide_plan.steps}

    definition["handlers"][0]["workflow"]["steps"][0]["workflow_definition"]["steps"][
        1
    ] = setter(next_state="idle")
    with pytest.raises(WorkflowCompileError) as raised:
        compile_(definition)
    assert raised.value.step_path == ("decide", "inner", "save")


def test_machines_inside_handler_workflows_are_rejected() -> None:
    definition = example_05()
    definition["handlers"][0]["workflow"]["state_machines"] = [machine()]

    with pytest.raises(NestedWorkflowError, match=r"declares \['state_machines'\]"):
        compile_(definition)


# --- Scopes -------------------------------------------------------------------------


def _reviewing_child() -> Dict[str, Any]:
    child = passive(
        state_machines=[inspection_machine()],
        handlers=[decide()],
    )

    return child


def test_diamond_children_get_their_own_machines_handlers_and_triggers() -> None:
    child = _reviewing_child()
    parent = passive(
        steps=[nested("a", child), nested("b", child)],
        outputs=[],
        handlers=[audit("audit", "$state_machines.a/inspection_state.events.decided")],
        state_machines=[
            machine(
                "watch",
                transitions=[fixed("go", "$steps.b/inspection.events.session_ended")],
            )
        ],
    )
    reactions = compile_(parent).reactions

    assert [item.path for item in reactions.machines] == [
        ("watch",),
        ("a", "inspection_state"),
        ("b", "inspection_state"),
    ]
    a_machine = reactions.machine(("a", "inspection_state"))
    assert a_machine.transitions[0].trigger == EventOrigin(
        "step", "review_requested", ("a", "inspection")
    )
    assert a_machine.transitions[1].handler == ("a", "decide")
    assert reactions.handler(("b", "decide")).origin == EventOrigin(
        "machine", "decision_requested", ("b", "inspection_state")
    )
    assert reactions.setter(("a", "decide"), "inspection_state", "finish_review") is (
        a_machine.transitions[1]
    )
    assert reactions.setter(("b", "decide"), "inspection_state", "finish_review") is (
        reactions.machine(("b", "inspection_state")).transitions[1]
    )
    with pytest.raises(ContractError, match="selected by handler 'a/decide'"):
        reactions.setter(("audit",), "a/inspection_state", "finish_review")
    assert reactions.handler(("audit",)).origin == a_machine.origin("decided")
    assert reactions.transitions_for(
        EventOrigin("step", "session_ended", ("b", "inspection"))
    ) == (
        reactions.machine(("watch",)).transitions[0],
        reactions.machine(("b", "inspection_state")).transitions[2],
    )
    assert reactions.machine(("watch",)).transitions[0].label == "watch.go"


def test_child_machine_events_resolve_only_downward() -> None:
    child = passive(handlers=[audit("up", "$state_machines.watch.events.x", "s")])
    parent = passive(
        steps=[nested("c", child)],
        outputs=[],
        state_machines=[
            machine(
                "watch",
                transitions=[
                    fixed("go", "$steps.c/inspection.events.session_ended", emit="x")
                ],
            )
        ],
    )

    with pytest.raises(SelectorError, match="unknown state machine 'c/watch'"):
        compile_(parent)


# --- Plan invariants -------------------------------------------------------------


def test_forged_reaction_plans_are_rejected() -> None:
    reactions = compile_(example_05()).reactions
    inspection, lifecycle = reactions.machines

    with pytest.raises(ContractError, match="declared twice"):
        replace(reactions, machines=(inspection, inspection))
    with pytest.raises(ContractError, match="not a declared machine event"):
        replace(reactions, machines=(lifecycle,))

    orphan = replace(inspection.transitions[1], handler=("ghost",))
    forged = replace(
        inspection,
        transitions=(inspection.transitions[0], orphan, inspection.transitions[2]),
    )
    with pytest.raises(ContractError, match="unknown handler 'ghost'"):
        replace(reactions, machines=(forged, lifecycle))

    wrong = replace(inspection.transitions[1], handler=("audit_decision",))
    forged = replace(
        inspection,
        transitions=(inspection.transitions[0], wrong, inspection.transitions[2]),
    )
    with pytest.raises(ContractError, match="Handler 'decide' step 'save'"):
        replace(reactions, machines=(forged, lifecycle))


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"trigger": None}, "exactly one of a trigger or a handler"),
        ({"targets": ("b", "a")}, "exactly one target"),
        ({"sources": frozenset()}, "needs 'from' and 'to'"),
        ({"emits": None}, "declares fields but emits no event"),
        ({"emits": EventOrigin("machine", "x", ("other",))}, "of its own machine"),
        ({"fields": {"v": ("transition", "when")}}, "known values"),
        ({"fields": {"v": ("literal", float("nan"))}}, "must be finite"),
        ({"fields": {"v": ("config", 1)}}, "unknown source"),
    ],
)
def test_planned_transitions_validate_their_shape(arguments, message) -> None:
    base = dict(
        name="t",
        machine=("m",),
        sources={"a"},
        targets=("b",),
        trigger=ENDED,
        emits=EventOrigin("machine", "x", ("m",)),
        fields={"v": ("event", "frame")},
    )
    base.update(arguments)

    with pytest.raises(ContractError, match=message):
        PlannedTransition(**base)


def test_handler_selected_transitions_do_not_read_the_trigger_payload() -> None:
    with pytest.raises(ContractError, match="does not retain the payload"):
        PlannedTransition(
            name="t",
            machine=("m",),
            sources={"a"},
            targets=("b", "c"),
            handler=("h",),
            emits=EventOrigin("machine", "x", ("m",)),
            fields={"v": ("event", "frame")},
        )


def test_planned_machines_validate_states_and_events() -> None:
    transition = PlannedTransition(
        name="t", machine=("m",), sources={"a"}, targets=("b",), trigger=ENDED
    )
    with pytest.raises(ContractError, match="unknown states"):
        PlannedMachine(
            path=("m",),
            scope="global",
            initial="a",
            states=("a",),
            transitions=(transition,),
        )
    with pytest.raises(ContractError, match="initial state"):
        PlannedMachine(path=("m",), scope="global", initial="z", states=("a",))
    emitting = replace(transition, emits=EventOrigin("machine", "x", ("m",)))
    with pytest.raises(ContractError, match="no such event"):
        PlannedMachine(
            path=("m",),
            scope="global",
            initial="a",
            states=("a", "b"),
            transitions=(emitting,),
        )
    system = replace(transition, trigger=EventOrigin("system", "started"))
    with pytest.raises(ContractError, match="is per source"):
        PlannedMachine(
            path=("m",),
            scope="source",
            initial="a",
            states=("a", "b"),
            transitions=(system,),
        )


def test_compiled_workflows_reject_forged_machine_origins() -> None:
    plan = compile_(example_05())
    reactions = plan.reactions
    inspection, lifecycle = reactions.machines
    ghost = replace(
        inspection.transitions[2],
        trigger=EventOrigin("step", "session_ended", ("ghost",)),
    )
    forged = replace(
        inspection,
        transitions=(inspection.transitions[0], inspection.transitions[1], ghost),
    )

    with pytest.raises(ContractError, match="plan has no step"):
        replace(plan, reactions=replace(reactions, machines=(forged, lifecycle)))

    passive_plan = compile_(passive(state_machines=[machine()]))
    with pytest.raises(ContractError, match="system events belong to an active run"):
        replace(
            passive_plan,
            reactions=replace(reactions, handlers=(), groups=(), machines=(lifecycle,)),
        )


def test_sessions_of_machine_plans_get_managed_state() -> None:
    plan = compile_(passive(state_machines=[machine()]))
    assert plan.reactions.machines[0].transitions == ()
    assert requests_managed_state(plan)

    session = plan.create_session()
    try:
        assert session.managed_state is not None
        assert session.owned_state is session.managed_state
    finally:
        session.close()
