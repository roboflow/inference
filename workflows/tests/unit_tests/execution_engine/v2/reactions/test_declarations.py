"""Event declarations and the compiled reaction contract, without the parser.

Plans here are compiled from minimal definitions, then reactions are attached
with ``dataclasses.replace`` so every invariant of ``PlannedHandler``,
``ReactionPlan`` and ``CompiledWorkflow`` is checked on forged input.
"""

import dataclasses

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
    spec_of,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    DeclarationError,
)
from roboflow_workflows.execution_engine.v2.events import Event, EventPayloadError
from roboflow_workflows.execution_engine.v2.kinds import (
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
    Kind,
)
from roboflow_workflows.execution_engine.v2.plan import NULL_OBSERVER
from roboflow_workflows.execution_engine.v2.reactions import (
    EventOrigin,
    PlannedHandler,
    PlannedHandlerGroup,
    PlannedMachine,
    PlannedSignal,
    PlannedTransition,
    QueuePolicy,
    ReactionPlan,
    StateDefaults,
)
from roboflow_workflows.execution_engine.v2.sources import (
    Source,
    SourceOutput,
    SourceParams,
)

ENTERED = Event(
    {"zone_id": STRING_KIND, "count": INTEGER_KIND, "frame": []},
    description="An object entered the zone.",
)


class Zone(Block):
    """Emits ``entered`` for every value."""

    type = "test/zone@v1"
    outputs = {"present": Output(FLOAT_KIND)}
    events = {"entered": ENTERED}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, value):
        self.emit("entered", zone_id="bay", count=1, frame=value)
        return {"present": value}


class Note(Block):
    """A handler step returning its text."""

    type = "test/note@v1"
    outputs = {"text": Output(STRING_KIND)}

    class Params(BlockParams):
        text: Ref(STRING_KIND)

    def run(self, text):
        return {"text": text}


class Feed(Source):
    """Never emits; only makes a definition active."""

    type = "test/feed@v1"
    outputs = {"value": SourceOutput(FLOAT_KIND)}

    class Params(SourceParams):
        pass

    def open(self) -> None:
        pass

    def read(self):
        return None

    def close(self) -> None:
        pass


CATALOGUE = Catalogue([Zone, Note], sources=[Feed])

PASSIVE = {
    "version": "2.0",
    "inputs": [{"type": "WorkflowParameter", "name": "value"}],
    "steps": [{"type": Zone.type, "name": "zone", "value": "$inputs.value"}],
    "outputs": [
        {"type": "JsonField", "name": "present", "selector": "$steps.zone.present"}
    ],
}

ACTIVE = {
    "version": "2.0",
    "sources": [{"type": Feed.type, "name": "cam"}],
    "steps": [{"type": Zone.type, "name": "zone", "value": "$sources.cam.value"}],
    "outputs": [
        {
            "type": "OutputGroup",
            "name": "frames",
            "anchor": "$sources.cam.value",
            "outputs": [
                {
                    "type": "JsonField",
                    "name": "present",
                    "selector": "$steps.zone.present",
                }
            ],
        }
    ],
}

NOTE = {
    "version": "2.0",
    "inputs": [{"name": "zone", "kind": ["string"]}],
    "steps": [{"type": Note.type, "name": "note", "text": "$inputs.zone"}],
    "outputs": [{"type": "JsonField", "name": "text", "selector": "$steps.note.text"}],
}

ZONE_ENTERED = EventOrigin(kind="step", event="entered", path=("zone",))


def _note_plan(catalogue: Catalogue = CATALOGUE, definition: dict = NOTE):
    return compile_workflow(definition, catalogue=catalogue)


def _handler(**overrides) -> PlannedHandler:
    arguments = dict(
        path=("notify",),
        origin=ZONE_ENTERED,
        event=ENTERED,
        plan=_note_plan(),
        bindings={"zone": "zone_id"},
    )
    arguments.update(overrides)

    return PlannedHandler(**arguments)


def _reacting(definition: dict, reactions: ReactionPlan):
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    return dataclasses.replace(plan, reactions=reactions)


# --- Event -----------------------------------------------------------------


def test_event_normalizes_kinds_and_describes_fields() -> None:
    assert ENTERED.kind_names("zone_id") == ("string",)
    assert ENTERED.kind_names("frame") == ("*",)
    assert ENTERED.describe() == {
        "fields": {"zone_id": ["string"], "count": ["integer"], "frame": ["*"]},
        "description": "An object entered the zone.",
    }


@pytest.mark.parametrize(
    "fields",
    [{"1st": STRING_KIND}, {"a.b": STRING_KIND}, {"x": "string"}, ["x"]],
)
def test_event_rejects_bad_field_names_and_kinds(fields) -> None:
    with pytest.raises(ContractError):
        Event(fields)


def test_check_payload_requires_the_exact_fields_and_accepted_kinds() -> None:
    ENTERED.check_payload("entered", {"zone_id": "bay", "count": 2, "frame": object()})

    with pytest.raises(EventPayloadError, match="missing \\['count'\\]") as missing:
        ENTERED.check_payload("entered", {"zone_id": "bay", "frame": 1})
    assert missing.value.event == "entered"
    with pytest.raises(EventPayloadError, match="unknown \\['extra'\\]"):
        ENTERED.check_payload(
            "entered", {"zone_id": "bay", "count": 1, "frame": 1, "extra": 0}
        )
    with pytest.raises(EventPayloadError, match="field 'count'"):
        ENTERED.check_payload("entered", {"zone_id": "bay", "count": "1", "frame": 1})


def test_block_events_reach_the_spec_and_its_description() -> None:
    spec = spec_of(Zone)

    assert dict(spec.events) == {"entered": ENTERED}
    assert spec.describe()["events"] == {"entered": ENTERED.describe()}
    assert "events" not in spec_of(Note).describe()


def test_kinds_used_only_by_events_join_the_block_kinds() -> None:
    reviewed = Kind(name="test_reviewed_score")

    class Reviewer(Block):
        type = "test/reviewer@v1"
        outputs = {"present": Output(FLOAT_KIND)}
        events = {"reviewed": Event({"score": reviewed, "zone_id": STRING_KIND})}

        class Params(BlockParams):
            value: Ref(FLOAT_KIND)

        def run(self, value):
            return {"present": value}

    kinds = {kind.name: kind for kind in spec_of(Reviewer).kinds}

    assert kinds["test_reviewed_score"] is reviewed
    assert kinds["string"] is STRING_KIND


def test_event_kinds_share_the_same_name_conflict_check() -> None:
    impostor = Kind(name=FLOAT_KIND.name, description="another float")

    with pytest.raises(DeclarationError, match="two different kinds named 'float'"):

        class Broken(Block):
            type = "test/broken@v1"
            outputs = {"value": Output(FLOAT_KIND)}
            events = {"entered": Event({"value": impostor})}

            class Params(BlockParams):
                value: Ref(FLOAT_KIND)

            def run(self, value):
                return {"value": value}


@pytest.mark.parametrize(
    "events, message",
    [
        ({"bad name": ENTERED}, "event name"),
        ({"entered": {"zone_id": STRING_KIND}}, "must be an Event"),
        (["entered"], "must be a mapping"),
    ],
)
def test_block_events_are_validated_at_class_creation(events, message) -> None:
    declared = events
    with pytest.raises(DeclarationError, match=message):

        class Broken(Block):
            type = "test/broken@v1"
            outputs = {"value": Output(FLOAT_KIND)}
            events = declared

            class Params(BlockParams):
                value: Ref(FLOAT_KIND)

            def run(self, value):
                return {"value": value}


# --- Small contract types ----------------------------------------------------


def test_event_origin_selectors_and_validation() -> None:
    assert ZONE_ENTERED.selector == "$steps.zone.events.entered"
    nested = EventOrigin(kind="step", event="hit", path=["a", "b"])
    assert nested.path == ("a", "b")
    assert nested.selector == "$steps.a/b.events.hit"
    assert EventOrigin(kind="signal", event="ack").selector == "$signals.ack"
    assert EventOrigin(kind="system", event="ended").selector == "$system.events.ended"
    machine = EventOrigin(kind="machine", event="decided", path=("c", "m"))
    assert machine.selector == "$state_machines.c/m.events.decided"

    with pytest.raises(ContractError, match="needs a path"):
        EventOrigin(kind="step", event="hit")
    with pytest.raises(ContractError, match="needs a path"):
        EventOrigin(kind="machine", event="hit")
    with pytest.raises(ContractError, match="has no path"):
        EventOrigin(kind="signal", event="ack", path=("a",))
    with pytest.raises(ContractError, match="has no path"):
        EventOrigin(kind="system", event="started", path=("a",))
    with pytest.raises(ContractError, match="System event must be one of"):
        EventOrigin(kind="system", event="paused")
    with pytest.raises(ContractError, match="kind must be"):
        EventOrigin(kind="timer", event="tick")


@pytest.mark.parametrize(
    "arguments",
    [{"max_depth": 0}, {"max_depth": True}, {"max_depth": 1.5}, {"overflow": "block"}],
)
def test_queue_policy_rejects_bad_values(arguments) -> None:
    with pytest.raises(ContractError):
        QueuePolicy(**arguments)


def test_queue_policy_defaults() -> None:
    assert QueuePolicy().describe() == {"max_depth": 16, "overflow": "synchronous"}


@pytest.mark.parametrize(
    "global_, source",
    [({"": 1}, {}), ({"x": float("nan")}, {}), ({}, {"x": object()}), ({}, {1: 2})],
)
def test_state_defaults_accept_only_portable_json(global_, source) -> None:
    with pytest.raises(ContractError):
        StateDefaults(global_=global_, source=source)


def test_state_defaults_describe_and_emptiness() -> None:
    defaults = StateDefaults(global_={"n": 0}, source={"on": True, "seen": None})

    assert not defaults.is_empty
    assert StateDefaults().is_empty
    assert defaults.describe() == {
        "global": {"n": 0},
        "source": {"on": True, "seen": None},
    }


# --- PlannedHandler ------------------------------------------------------------


def test_planned_handler_exposes_identity_bindings_and_inputs() -> None:
    handler = _handler(path=("child", "notify"), constants={})

    assert handler.name == "notify"
    assert handler.scope == ("child",)
    assert handler.bound_fields == frozenset({"zone_id"})
    assert handler.outputs == ("text",)
    assert handler.inputs_for({"zone_id": "bay", "count": 3}) == {"zone": "bay"}


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"path": ()}, "must not be empty"),
        ({"mode": "async"}, "needs a QueuePolicy"),
        ({"queue": QueuePolicy()}, "must not declare a queue"),
        ({"mode": "later"}, "mode must be one of"),
        ({"bindings": {"other": "zone_id"}}, "not an input"),
        ({"bindings": {}}, "leaves required input 'zone' unbound"),
        ({"bindings": {"zone": "missing"}}, "declares fields"),
        ({"bindings": {"zone": "count"}}, "of kinds \\['integer'\\]"),
        (
            {"bindings": {"zone": "zone_id"}, "constants": {"zone": "x"}},
            "twice",
        ),
    ],
)
def test_planned_handler_rejects_inconsistent_declarations(overrides, message) -> None:
    with pytest.raises(ContractError, match=message):
        _handler(**overrides)


def test_planned_handler_rejects_active_handler_plans_and_inputs_with_axes() -> None:
    with pytest.raises(ContractError, match="handler workflows are passive"):
        _handler(plan=compile_workflow(ACTIVE, catalogue=CATALOGUE), bindings={})

    batched = {
        **NOTE,
        "inputs": [{"type": "WorkflowBatchInput", "name": "zone", "kind": ["string"]}],
    }
    with pytest.raises(ContractError, match="declares axes"):
        _handler(plan=_note_plan(definition=batched))


def test_planned_handler_rejects_reacting_handler_plans() -> None:
    inner = _handler()
    reacting = _reacting(PASSIVE, ReactionPlan(handlers=(inner,)))
    with pytest.raises(ContractError, match="cannot declare reactions"):
        _handler(plan=reacting, bindings={}, constants={"value": 1.0})


# --- ReactionPlan --------------------------------------------------------------


def test_empty_reaction_plan_answers_lookups_cheaply() -> None:
    empty = ReactionPlan.EMPTY

    assert empty.is_empty
    assert empty.emitting_steps == frozenset()
    assert empty.step_handlers(("zone",), "entered") == ()
    assert empty.bound_fields(ZONE_ENTERED) == frozenset()
    assert empty.groups_of(("notify",)) == ()


def test_reaction_plan_lookups_follow_declaration_order() -> None:
    first = _handler(path=("first",))
    second = _handler(path=("second",), bindings={}, constants={"zone": "fixed"})
    group = PlannedHandlerGroup(
        name="notes", handler=("first",), fields={"text": "text"}
    )
    reactions = ReactionPlan(handlers=(first, second), groups=(group,))

    assert reactions.handlers_for(ZONE_ENTERED) == (first, second)
    assert reactions.step_handlers(("zone",), "entered") == (first, second)
    assert reactions.bound_fields(ZONE_ENTERED) == frozenset({"zone_id"})
    assert reactions.emitting_steps == frozenset({("zone",)})
    assert reactions.handler(["second"]) is second
    assert reactions.groups_of(("first",)) == (group,)
    assert reactions.groups_of(("second",)) == ()
    with pytest.raises(KeyError):
        reactions.handler(("third",))


def test_reaction_plan_rejects_repeated_paths_and_bad_groups() -> None:
    handler = _handler()
    with pytest.raises(ContractError, match="declared twice"):
        ReactionPlan(handlers=(handler, handler))
    with pytest.raises(ContractError, match="unknown handler"):
        ReactionPlan(
            handlers=(handler,),
            groups=(
                PlannedHandlerGroup(name="g", handler=("x",), fields={"t": "text"}),
            ),
        )
    with pytest.raises(ContractError, match="does not declare"):
        ReactionPlan(
            handlers=(handler,),
            groups=(
                PlannedHandlerGroup(
                    name="g", handler=("notify",), fields={"t": "nope"}
                ),
            ),
        )
    group = PlannedHandlerGroup(name="g", handler=("notify",), fields={"t": "text"})
    with pytest.raises(ContractError, match="declared twice"):
        ReactionPlan(handlers=(handler,), groups=(group, group))
    with pytest.raises(ContractError, match="selects no outputs"):
        PlannedHandlerGroup(name="g", handler=("notify",), fields={})


def test_signal_subscriptions_need_the_declared_signal_payload() -> None:
    ack = Event({"zone_id": STRING_KIND})
    signal = PlannedSignal(name="ack", event=ack)
    handler = _handler(origin=signal.origin, event=ack)

    assert ReactionPlan(handlers=(handler,), signals={"ack": signal}).signals == {
        "ack": signal
    }
    with pytest.raises(ContractError, match="not a declared signal"):
        ReactionPlan(handlers=(handler,))
    other = PlannedSignal(name="ack", event=Event({"zone_id": [], "extra": []}))
    with pytest.raises(ContractError, match="not a declared signal"):
        ReactionPlan(handlers=(handler,), signals={"ack": other})
    with pytest.raises(ContractError, match="keyed as"):
        ReactionPlan(signals={"other": signal})


def test_cascade_cycles_are_rejected_but_chains_are_not() -> None:
    # Handler-caused edges come only from the transitions a handler sets.
    payload = Event({"zone_id": STRING_KIND})
    x = EventOrigin(kind="machine", event="x", path=("m",))
    y = EventOrigin(kind="machine", event="y", path=("m",))

    def sets(handler: str, emits: EventOrigin) -> PlannedTransition:
        return PlannedTransition(
            name=f"{handler}_t",
            machine=("m",),
            sources=frozenset({"a"}),
            targets=("a",),
            handler=(handler,),
            emits=emits,
            fields={"zone_id": ("literal", "z")},
        )

    def machine(*transitions: PlannedTransition) -> PlannedMachine:
        events = {transition.emits.event: payload for transition in transitions}
        return PlannedMachine(
            path=("m",),
            scope="global",
            initial="a",
            states=("a",),
            transitions=transitions,
            events=events,
        )

    chain = _handler(path=("chain",))
    reactions = ReactionPlan(handlers=(chain,), machines=(machine(sets("chain", x)),))
    assert reactions.causes(("chain",)) == {x}
    assert reactions.describe()["handlers"]["chain"]["causes"] == [x.selector]

    chain = _handler(path=("chain",), origin=y, event=payload)
    back = _handler(path=("back",), origin=x, event=payload)
    with pytest.raises(
        ContractError,
        match=r"cascade cycle: .*m.events.y -> \(handler chain sets m.chain_t\) -> "
        r".*m.events.x -> \(handler back sets m.back_t\) -> .*m.events.y",
    ):
        ReactionPlan(
            handlers=(chain, back),
            machines=(machine(sets("chain", x), sets("back", y)),),
        )


# --- CompiledWorkflow invariants -------------------------------------------------


def test_compiled_workflow_checks_handler_origins_and_paths() -> None:
    with pytest.raises(ContractError, match="same path as a step"):
        _reacting(PASSIVE, ReactionPlan(handlers=(_handler(path=("zone",)),)))
    other = _handler(origin=EventOrigin(kind="step", event="entered", path=("gone",)))
    with pytest.raises(ContractError, match="no step"):
        _reacting(PASSIVE, ReactionPlan(handlers=(other,)))
    wrong = _handler(
        origin=EventOrigin(kind="step", event="left", path=("zone",)),
        event=Event({"zone_id": STRING_KIND}),
    )
    with pytest.raises(ContractError, match="declares events \\['entered'\\]"):
        _reacting(PASSIVE, ReactionPlan(handlers=(wrong,)))


def test_compiled_workflow_rejects_async_passive_foreign_catalogue_and_groups() -> None:
    queued = _handler(mode="async", queue=QueuePolicy(max_depth=2))
    with pytest.raises(ContractError, match="async handlers need an active run"):
        _reacting(PASSIVE, ReactionPlan(handlers=(queued,)))
    assert not _reacting(ACTIVE, ReactionPlan(handlers=(queued,))).reactions.is_empty

    foreign = _handler(plan=_note_plan(catalogue=Catalogue([Zone, Note])))
    with pytest.raises(ContractError, match="another catalogue"):
        _reacting(PASSIVE, ReactionPlan(handlers=(foreign,)))

    group = PlannedHandlerGroup(name="notes", handler=("notify",), fields={"t": "text"})
    with pytest.raises(ContractError, match="need sources"):
        _reacting(PASSIVE, ReactionPlan(handlers=(_handler(),), groups=(group,)))
    clash = PlannedHandlerGroup(
        name="frames", handler=("notify",), fields={"t": "text"}
    )
    with pytest.raises(ContractError, match="repeats an output group name"):
        _reacting(ACTIVE, ReactionPlan(handlers=(_handler(),), groups=(clash,)))


def test_describe_lists_reactions_only_when_declared() -> None:
    plan = compile_workflow(PASSIVE, catalogue=CATALOGUE)
    assert "reactions" not in plan.describe()

    reacting = _reacting(PASSIVE, ReactionPlan(handlers=(_handler(),)))
    described = reacting.describe()["reactions"]["handlers"]["notify"]
    assert described["on"] == "$steps.zone.events.entered"
    assert described["bindings"] == {"zone": "$event.zone_id"}
    assert described["mode"] == "sync" and described["queue"] is None


# --- Sessions ----------------------------------------------------------------------


def test_handler_sessions_are_persistent_and_do_not_share_the_parent_observer() -> None:
    reacting = _reacting(PASSIVE, ReactionPlan(handlers=(_handler(),)))
    marker = object()
    session = reacting.create_session(
        observer=NULL_OBSERVER.__class__(),
        error_handler=lambda error: None,
        reaction_observer=marker,
    )

    handler_session = session.handler_sessions[("notify",)]
    assert set(session.handler_sessions) == {("notify",)}
    assert handler_session.plan is reacting.reactions.handler(("notify",)).plan
    assert handler_session.observer is NULL_OBSERVER
    assert handler_session.error_handler is None
    assert session.reaction_observer is marker
    assert session.managed_state is None and session.owned_state is None
    assert handler_session.run({"zone": "bay"}).rows() == [{"text": "bay"}]


def test_plain_sessions_have_no_reaction_observer() -> None:
    session = compile_workflow(PASSIVE, catalogue=CATALOGUE).create_session()

    assert session.reaction_observer is None
    assert dict(session.handler_sessions) == {}
