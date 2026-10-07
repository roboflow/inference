"""Compiling ``state``, ``signals``, ``handlers`` and handler groups from JSON.

Scope rules under test::

    root      handlers: $steps.zone…   $steps.child/zone…   $signals.ack
    └ child   handlers: $steps.zone…   (its own zone; never the parent's)
      used twice (diamond) → two handler identities, two emitters
"""

import copy
from typing import Any, Dict

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.errors import (
    CatalogueError,
    KindMismatchError,
    NestedWorkflowError,
    SelectorError,
    UnknownBlockError,
    WorkflowCompileError,
    WorkflowInputError,
)
from roboflow_workflows.execution_engine.v2.events import Event
from roboflow_workflows.execution_engine.v2.kinds import (
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
    WILDCARD_KIND,
    Kind,
)
from roboflow_workflows.execution_engine.v2.plan import requests_managed_state
from roboflow_workflows.execution_engine.v2.reactions import (
    EventOrigin,
    QueuePolicy,
    ReactionPlan,
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
    """Emits ``entered`` for every value, when somebody listens."""

    type = "test/zone@v1"
    outputs = {"present": Output(FLOAT_KIND)}
    events = {"entered": ENTERED}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, value):
        if self.has_subscribers("entered"):
            self.emit("entered", zone_id="bay", count=int(value), frame=value)
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


def note_workflow(**extra: Any) -> Dict[str, Any]:
    """Handler workflow ``zone (string) -> note -> text``; no version (draft)."""
    workflow = {
        "inputs": [{"name": "zone", "kind": ["string"]}],
        "steps": [{"type": Note.type, "name": "note", "text": "$inputs.zone"}],
        "outputs": [
            {"type": "JsonField", "name": "text", "selector": "$steps.note.text"}
        ],
    }
    workflow.update(extra)

    return workflow


def handler(name: str = "notify", on: str = "$steps.zone.events.entered", **extra):
    declared = {
        "name": name,
        "on": on,
        "bindings": {"zone": "$event.zone_id"},
        "workflow": note_workflow(),
    }
    declared.update(extra)

    return declared


def passive(**extra: Any) -> Dict[str, Any]:
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "value"}],
        "steps": [{"type": Zone.type, "name": "zone", "value": "$inputs.value"}],
        "outputs": [
            {"type": "JsonField", "name": "present", "selector": "$steps.zone.present"}
        ],
    }
    definition.update(extra)

    return definition


def active(**extra: Any) -> Dict[str, Any]:
    definition = {
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
    definition.update(extra)

    return definition


def child(**extra: Any) -> Dict[str, Any]:
    """A nested workflow step whose child has its own ``zone``."""
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "value"}],
        "steps": [{"type": Zone.type, "name": "zone", "value": "$inputs.value"}],
        "outputs": [
            {"type": "JsonField", "name": "present", "selector": "$steps.zone.present"}
        ],
    }
    definition.update(extra)

    return definition


def nested_step(name: str, definition: Dict[str, Any], value: str = "$inputs.value"):
    step = {
        "type": "roboflow_core/inner_workflow@v1",
        "name": name,
        "workflow_definition": definition,
        "parameter_bindings": {"value": value},
    }

    return step


def compile_(definition: Dict[str, Any]):
    return compile_workflow(definition, catalogue=CATALOGUE)


def handler_group(name: str = "notes", anchor: str = "$handlers.notify", **fields):
    selected = fields or {
        "text": f"{anchor.split('.', 2)[0]}.{anchor.split('.')[1]}.text"
    }
    group = {
        "type": "OutputGroup",
        "name": name,
        "anchor": anchor,
        "outputs": [
            {"type": "JsonField", "name": field, "selector": selector}
            for field, selector in selected.items()
        ],
    }

    return group


# --- The draft shape ------------------------------------------------------------


def test_draft_shape_compiles_to_handlers_state_and_a_handler_group() -> None:
    definition = active(
        state={"global": {"entry_count": 0}, "source": {"enabled": True, "last": ""}},
        handlers=[
            handler("count", execution={"mode": "sync"}),
            handler(
                "notify",
                execution={
                    "mode": "async",
                    "queue": {"max_depth": 4, "overflow": "leaky"},
                },
                bindings={"zone": "$event.zone_id"},
            ),
        ],
    )
    definition["outputs"].append(handler_group(anchor="$handlers.notify.text"))

    plan = compile_(definition)
    reactions = plan.reactions

    count, notify = reactions.handlers
    assert count.path == ("count",) and count.mode == "sync" and count.queue is None
    assert notify.queue == QueuePolicy(max_depth=4, overflow="leaky")
    assert notify.origin == EventOrigin(kind="step", event="entered", path=("zone",))
    assert notify.event is ENTERED
    assert dict(notify.bindings) == {"zone": "zone_id"}
    assert reactions.step_handlers(("zone",), "entered") == (count, notify)
    assert reactions.bound_fields(notify.origin) == frozenset({"zone_id"})
    assert reactions.state.describe() == {
        "global": {"entry_count": 0},
        "source": {"enabled": True, "last": ""},
    }
    (group,) = reactions.groups
    assert (group.name, group.handler, dict(group.fields)) == (
        "notes",
        ("notify",),
        {"text": "text"},
    )
    assert [item.name for item in plan.output_groups] == ["frames"]
    assert notify.plan.catalogue is plan.catalogue
    assert not notify.plan.is_active and notify.plan.reactions.is_empty
    assert requests_managed_state(plan)
    assert plan.describe()["reactions"]["handler_groups"]["notes"] == {
        "anchor": "$handlers.notify",
        "fields": {"text": "$handlers.notify.text"},
    }


def test_workflow_without_reactions_keeps_the_empty_plan() -> None:
    plan = compile_(passive())

    assert plan.reactions is ReactionPlan.EMPTY
    assert not requests_managed_state(plan)


def test_defaults_constants_and_versioned_handler_workflows() -> None:
    plan = compile_(
        passive(
            handlers=[
                handler(
                    bindings={"zone": "fixed"},
                    workflow=note_workflow(version="2.0"),
                )
            ]
        )
    )
    (notify,) = plan.reactions.handlers

    assert notify.mode == "sync"
    assert dict(notify.constants) == {"zone": "fixed"} and not notify.bindings
    assert notify.inputs_for({}) == {"zone": "fixed"}

    with pytest.raises(WorkflowCompileError, match="version must be '2.0'"):
        compile_(passive(handlers=[handler(workflow=note_workflow(version="1.0"))]))


def test_async_queue_defaults_to_a_bounded_synchronous_queue() -> None:
    plan = compile_(active(handlers=[handler(execution={"mode": "async"})]))

    assert plan.reactions.handlers[0].queue == QueuePolicy()


def test_passive_sync_handler_runs_with_its_persistent_session() -> None:
    plan = compile_(passive(handlers=[handler()]))
    session = plan.create_session()
    handler_session = session.handler_sessions[("notify",)]

    assert session.run({"value": 2.0}).rows() == [{"present": 2.0}]
    assert session.handler_sessions[("notify",)] is handler_session
    assert handler_session.run({"zone": "bay"}).rows() == [{"text": "bay"}]


# --- Scopes: nested, diamond, parent to child ---------------------------------------


def test_child_handlers_subscribe_to_their_own_scope() -> None:
    inner = child(handlers=[handler()])
    plan = compile_(passive(steps=[*passive()["steps"], nested_step("child", inner)]))

    (notify,) = plan.reactions.handlers
    assert notify.path == ("child", "notify")
    assert notify.origin.path == ("child", "zone")
    assert notify.origin.selector == "$steps.child/zone.events.entered"


def test_diamond_reuse_gives_each_use_its_own_handler_identity() -> None:
    inner = child(handlers=[handler()], state={"source": {"seen": 0}})
    plan = compile_(
        passive(
            steps=[
                *passive()["steps"],
                nested_step("a", inner),
                nested_step("b", inner),
            ]
        )
    )

    paths = [item.path for item in plan.reactions.handlers]
    emitters = [item.origin.path for item in plan.reactions.handlers]
    assert paths == [("a", "notify"), ("b", "notify")]
    assert emitters == [("a", "zone"), ("b", "zone")]
    assert plan.reactions.handlers[0].plan is not plan.reactions.handlers[1].plan
    assert dict(plan.reactions.state.source) == {"seen": 0}


def test_parent_handlers_subscribe_to_child_and_grandchild_steps() -> None:
    grandchild = child()
    middle = child(steps=[*child()["steps"], nested_step("deep", grandchild)])
    plan = compile_(
        passive(
            steps=[*passive()["steps"], nested_step("mid", middle)],
            handlers=[
                handler("near", on="$steps.mid/zone.events.entered"),
                handler("far", on="$steps.mid/deep/zone.events.entered"),
            ],
        )
    )

    near, far = plan.reactions.handlers
    assert (near.path, near.origin.path) == (("near",), ("mid", "zone"))
    assert (far.path, far.origin.path) == (("far",), ("mid", "deep", "zone"))


def test_child_handlers_cannot_reach_parent_steps() -> None:
    inner = child(
        steps=[{"type": Note.type, "name": "note", "text": "$inputs.value"}],
        outputs=[{"type": "JsonField", "name": "text", "selector": "$steps.note.text"}],
        handlers=[handler()],
    )
    with pytest.raises(
        SelectorError, match="unknown step 'zone' of nested workflow"
    ) as error:
        compile_(passive(steps=[*passive()["steps"], nested_step("child", inner)]))
    assert error.value.step_path == ("child", "notify")


@pytest.mark.parametrize(
    "on, message",
    [
        ("$steps.child.events.entered", "names nested workflow step 'child'"),
        ("$steps.zone/inner.events.entered", "'zone' is not a nested workflow step"),
        (
            "$steps.zone.events.left",
            "declares no event 'left'; it declares \\['entered'\\]",
        ),
        ("$steps.ghost.events.entered", "unknown step 'ghost' of the root workflow"),
        ("$steps.zone.entered", "must be \\$steps.<step>.events.<event>"),
        ("$inputs.value", "must be \\$steps.<step>.events.<event>"),
    ],
)
def test_bad_subscriptions_name_the_problem(on, message) -> None:
    definition = passive(
        steps=[*passive()["steps"], nested_step("child", child())],
        handlers=[handler(on=on)],
    )
    with pytest.raises(SelectorError, match=message):
        compile_(definition)


# --- Signals ------------------------------------------------------------------------


def test_root_signals_are_workflow_global() -> None:
    inner = child(handlers=[handler("on_ack", on="$signals.ack")])
    plan = compile_(
        passive(
            steps=[*passive()["steps"], nested_step("child", inner)],
            signals=[
                {"name": "ack", "fields": {"zone_id": ["string"], "extra": []}},
                {"name": "ping", "description": "No payload."},
            ],
            handlers=[handler("root_ack", on="$signals.ack")],
        )
    )

    reactions = plan.reactions
    assert set(reactions.signals) == {"ack", "ping"}
    assert reactions.signals["ack"].event.kind_names("extra") == ("*",)
    assert reactions.signals["ping"].event.describe() == {
        "fields": {},
        "description": "No payload.",
    }
    ack = EventOrigin(kind="signal", event="ack")
    assert [item.path for item in reactions.handlers_for(ack)] == [
        ("root_ack",),
        ("child", "on_ack"),
    ]


def test_signal_declarations_are_checked() -> None:
    with pytest.raises(SelectorError, match="undeclared signal; .* signals \\[\\]"):
        compile_(passive(handlers=[handler(on="$signals.ack")]))
    with pytest.raises(WorkflowCompileError, match="unknown kinds \\['nope'\\]"):
        compile_(passive(signals=[{"name": "ack", "fields": {"x": "nope"}}]))
    with pytest.raises(WorkflowCompileError, match="duplicate signal name"):
        compile_(passive(signals=[{"name": "ack"}, {"name": "ack"}]))
    with pytest.raises(
        NestedWorkflowError, match="only the root workflow declares signals"
    ):
        inner = child(signals=[{"name": "ack"}])
        compile_(passive(steps=[*passive()["steps"], nested_step("child", inner)]))


# --- Bindings and kinds -------------------------------------------------------


@pytest.mark.parametrize(
    "bindings, error, message",
    [
        ({"zone": "$event.missing"}, SelectorError, "declares fields"),
        ({"zone": "$event.count"}, KindMismatchError, "of kinds \\['integer'\\]"),
        ({}, WorkflowCompileError, "leaves required input 'zone' unbound"),
        ({"zone": "$event.zone_id", "other": 1}, WorkflowCompileError, "not an input"),
        ({"zone": "$inputs.value"}, SelectorError, "binds only \\$event.<field>"),
        ({"zone": "$event.a.b"}, SelectorError, "binds only \\$event.<field>"),
    ],
)
def test_binding_diagnostics(bindings, error, message) -> None:
    with pytest.raises(error, match=message) as raised:
        compile_(passive(handlers=[handler(bindings=bindings)]))
    if error is not SelectorError or "declares fields" in message:
        assert raised.value.step_path == ("notify",)
        assert raised.value.field_path == ("bindings", next(reversed(bindings), "zone"))


def test_handler_inputs_cannot_declare_axes() -> None:
    workflow = note_workflow(
        inputs=[{"type": "WorkflowBatchInput", "name": "zone", "kind": ["string"]}]
    )
    with pytest.raises(WorkflowCompileError, match="declares axes"):
        compile_(passive(handlers=[handler(workflow=workflow)]))


# --- Kinds declared only by events ------------------------------------------------

SCORE_KIND = Kind(
    name="test_review_score",
    validate=lambda value: isinstance(value, int) and 0 <= value <= 10,
)


class Reviewer(Block):
    """Emits ``reviewed``; no input or output uses its ``score`` kind."""

    type = "test/reviewer@v1"
    outputs = {"present": Output(FLOAT_KIND)}
    events = {"reviewed": Event({"score": SCORE_KIND})}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, value):
        self.emit("reviewed", score=int(value))
        return {"present": value}


class Echo(Block):
    """A handler step accepting any kind."""

    type = "test/echo@v1"
    outputs = {"value": Output(WILDCARD_KIND)}

    class Params(BlockParams):
        value: Ref(WILDCARD_KIND)

    def run(self, value):
        return {"value": value}


def review_workflow() -> Dict[str, Any]:
    tally = {
        "name": "tally",
        "on": "$steps.review.events.reviewed",
        "bindings": {"score": "$event.score"},
        "workflow": {
            "inputs": [{"name": "score", "kind": [SCORE_KIND.name]}],
            "steps": [{"type": Echo.type, "name": "echo", "value": "$inputs.score"}],
            "outputs": [
                {"type": "JsonField", "name": "score", "selector": "$steps.echo.value"}
            ],
        },
    }
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "value"}],
        "steps": [{"type": Reviewer.type, "name": "review", "value": "$inputs.value"}],
        "outputs": [
            {
                "type": "JsonField",
                "name": "present",
                "selector": "$steps.review.present",
            }
        ],
        "handlers": [tally],
    }

    return definition


def test_event_only_kind_registers_and_keeps_its_validator_in_the_handler() -> None:
    plan = compile_workflow(review_workflow(), catalogue=Catalogue([Reviewer, Echo]))
    tally = plan.reactions.handlers[0]
    handler_session = plan.create_session().handler_sessions[("tally",)]

    assert tally.plan.inputs["score"].kinds == (SCORE_KIND.name,)
    assert tally.plan.catalogue.kinds[SCORE_KIND.name] is SCORE_KIND
    assert handler_session.run({"score": 3}).rows() == [{"score": 3}]
    with pytest.raises(WorkflowInputError, match="not a valid 'test_review_score'"):
        handler_session.run({"score": 11})


def test_event_kind_reusing_a_registered_name_is_an_ordinary_kind_conflict() -> None:
    class Scorer(Block):
        type = "test/scorer@v1"
        outputs = {"score": Output(Kind(name=SCORE_KIND.name))}

        class Params(BlockParams):
            value: Ref(FLOAT_KIND)

        def run(self, value):
            return {"score": int(value)}

    with pytest.raises(CatalogueError, match="kinds are named 'test_review_score'"):
        Catalogue([Reviewer, Scorer])


# --- Execution modes ----------------------------------------------------------


@pytest.mark.parametrize(
    "execution, message",
    [
        ({"mode": "async"}, "async handlers need an active run"),
        ({"mode": "sync", "queue": {"max_depth": 2}}, "declares a queue for a sync"),
        ({"mode": "later"}, "mode must be one of"),
        ({"mode": "async", "queue": {"overflow": "block"}}, "queue is invalid"),
        ({"mode": "async", "queue": {"size": 2}}, "unsupported keys \\['size'\\]"),
        ({"retries": 3}, "unsupported keys \\['retries'\\]"),
    ],
)
def test_execution_diagnostics(execution, message) -> None:
    with pytest.raises(WorkflowCompileError, match=message):
        compile_(passive(handlers=[handler(execution=execution)]))


# --- Handler workflows --------------------------------------------------------


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"sources": [{"type": Feed.type, "name": "cam"}]}, "declares \\['sources'\\]"),
        ({"handlers": [handler()]}, "declares \\['handlers'\\]; handler workflows"),
        ({"state": {"global": {"n": 0}}}, "declares \\['state'\\]"),
        ({"steps": []}, "handler workflow has no steps"),
    ],
)
def test_handler_workflow_restrictions(extra, message) -> None:
    with pytest.raises(NestedWorkflowError, match=message) as error:
        compile_(passive(handlers=[handler(workflow=note_workflow(**extra))]))
    assert error.value.step_path == ("notify",)


def test_handler_workflow_children_cannot_react_either() -> None:
    reacting_child = child(state={"source": {"n": 0}})
    workflow = note_workflow(
        steps=[*note_workflow()["steps"], nested_step("inner", reacting_child, "1.0")]
    )
    with pytest.raises(
        NestedWorkflowError, match="declares \\['state'\\] \\(in \\$steps.inner\\)"
    ):
        compile_(passive(handlers=[handler(workflow=workflow)]))


def test_handler_workflow_may_embed_ordinary_nested_workflows() -> None:
    workflow = note_workflow(
        steps=[*note_workflow()["steps"], nested_step("inner", child(), 1.0)]
    )
    plan = compile_(passive(handlers=[handler(workflow=workflow)]))

    paths = [step.path for step in plan.reactions.handlers[0].plan.steps]
    assert ("inner", "zone") in paths


def test_errors_inside_handler_workflows_name_the_handler() -> None:
    workflow = note_workflow(
        steps=[{"type": "test/ghost@v1", "name": "ghost", "text": "$inputs.zone"}]
    )
    inner = child(handlers=[handler(workflow=workflow)])
    with pytest.raises(
        UnknownBlockError, match="\\$handlers.child/notify workflow:"
    ) as error:
        compile_(passive(steps=[*passive()["steps"], nested_step("child", inner)]))
    assert error.value.step_path == ("child", "notify", "ghost")


def test_handler_names_are_unique_and_differ_from_steps() -> None:
    with pytest.raises(WorkflowCompileError, match="duplicate handler name"):
        compile_(passive(handlers=[handler(), handler()]))
    with pytest.raises(WorkflowCompileError, match="has the name of \\$steps.zone"):
        compile_(passive(handlers=[handler("zone")]))


@pytest.mark.parametrize(
    "declared, message",
    [
        ({"name": "x", "on": "$steps.zone.events.entered"}, "non-empty workflow"),
        ({**handler(), "priority": 1}, "unsupported keys \\['priority'\\]"),
        ({**handler(), "bindings": ["zone"]}, "must map handler workflow inputs"),
    ],
)
def test_handler_structure_diagnostics(declared, message) -> None:
    with pytest.raises(WorkflowCompileError, match=message):
        compile_(passive(handlers=[declared]))


# --- State --------------------------------------------------------------------


def test_state_of_all_scopes_merges_into_one_namespace() -> None:
    inner = child(state={"global": {"total": 0}, "source": {"seen": 0}})
    plan = compile_(
        passive(
            steps=[*passive()["steps"], nested_step("child", inner)],
            state={"global": {"total": 0, "mode": "idle"}},
        )
    )

    assert plan.reactions.state.describe() == {
        "global": {"total": 0, "mode": "idle"},
        "source": {"seen": 0},
    }
    assert not plan.reactions.handlers


@pytest.mark.parametrize("value", [0.0, False, "0", None, 1])
def test_state_conflicts_are_rejected_with_both_locations(value) -> None:
    inner = child(state={"global": {"total": value}})
    definition = passive(
        steps=[*passive()["steps"], nested_step("child", inner)],
        state={"global": {"total": 0}},
    )
    with pytest.raises(
        WorkflowCompileError, match="state.global.total starts at"
    ) as error:
        compile_(definition)
    assert "steps[1].workflow_definition.state.global.total" in str(error.value)


@pytest.mark.parametrize(
    "state, message",
    [
        ({"global": {"x": float("nan")}}, "must be finite"),
        ({"global": {"x": {1, 2}}}, "portable JSON"),
        ({"local": {}}, "unsupported keys \\['local'\\]"),
        ({"source": []}, "must map keys to values"),
        ([], "must be a mapping"),
    ],
)
def test_state_values_must_be_portable(state, message) -> None:
    with pytest.raises(WorkflowCompileError, match=message):
        compile_(passive(state=state))


def test_empty_state_machines_keep_the_reaction_plan_empty() -> None:
    inner = child(state_machines=[])
    plan = compile_(
        passive(
            state_machines=[],
            steps=[*passive()["steps"], nested_step("child", inner)],
        )
    )

    assert plan.reactions is ReactionPlan.EMPTY


# --- Handler groups -----------------------------------------------------------


def test_child_handler_groups_use_the_scoped_handler_path() -> None:
    inner = child(handlers=[handler()])
    definition = active(
        steps=[*active()["steps"], nested_step("c", inner, "$sources.cam.value")]
    )
    definition["outputs"].append(
        handler_group(anchor="$handlers.c/notify", text="$handlers.c/notify.text")
    )

    (group,) = compile_(definition).reactions.groups
    assert group.handler == ("c", "notify")


@pytest.mark.parametrize(
    "group, error, message",
    [
        (
            handler_group(anchor="$handlers.ghost", text="$handlers.ghost.text"),
            SelectorError,
            "unknown handler \\$handlers.ghost",
        ),
        (
            handler_group(anchor="$handlers.notify.nope"),
            SelectorError,
            "outputs \\['nope'\\]",
        ),
        (
            handler_group(text="$handlers.notify.nope"),
            SelectorError,
            "outputs \\['nope'\\]",
        ),
        (
            handler_group(text="$handlers.other.text"),
            WorkflowCompileError,
            "carries outputs of that handler only",
        ),
        (
            handler_group(text="$steps.zone.present"),
            SelectorError,
            "must be \\$handlers.<handler>",
        ),
        (
            handler_group(anchor="$handlers.a.b.c"),
            SelectorError,
            "must be \\$handlers.<handler>",
        ),
        (
            handler_group(name="frames"),
            WorkflowCompileError,
            "duplicate output group names \\['frames'\\]",
        ),
        ({**handler_group(), "outputs": []}, WorkflowCompileError, "non-empty list"),
    ],
)
def test_handler_group_diagnostics(group, error, message) -> None:
    definition = active(handlers=[handler()])
    definition["outputs"].append(group)
    with pytest.raises(error, match=message):
        compile_(definition)


def test_handler_group_fields_reject_output_options() -> None:
    group = handler_group()
    group["outputs"][0]["coordinates_system"] = "parent"
    definition = active(handlers=[handler()])
    definition["outputs"].append(group)
    with pytest.raises(WorkflowCompileError, match="sets \\['coordinates_system'\\]"):
        compile_(definition)


def test_handler_outputs_stay_out_of_pulse_groups() -> None:
    definition = active(handlers=[handler()])
    definition["outputs"][0]["outputs"].append(
        {"type": "JsonField", "name": "text", "selector": "$handlers.notify.text"}
    )
    with pytest.raises(WorkflowCompileError, match="group anchored at the handler"):
        compile_(definition)


def test_passive_workflows_and_children_cannot_declare_handler_groups() -> None:
    definition = passive(handlers=[handler()], outputs=[handler_group()])
    with pytest.raises(WorkflowCompileError, match="declares no sources; a passive"):
        compile_(definition)

    inner = child(handlers=[handler()], outputs=[handler_group()])
    with pytest.raises(
        NestedWorkflowError, match="nested workflow declares output groups"
    ):
        compile_(passive(steps=[*passive()["steps"], nested_step("child", inner)]))


# --- Sessions -----------------------------------------------------------------


def test_sessions_create_state_for_declared_defaults_and_keep_caller_state() -> None:
    from roboflow_workflows.execution_engine.v2.state import MISSING, ManagedState

    plan = compile_(passive(state={"global": {"total": 5}}, handlers=[handler()]))

    session = plan.create_session()
    assert session.owned_state is not None
    assert session.managed_state.global_.get("total", MISSING) == 5
    assert session.handler_sessions[("notify",)].plan is plan.reactions.handlers[0].plan
    session.close()

    provided = ManagedState()
    provided.global_.set("total", 9)
    caller_session = plan.create_session(resources={"managed_state": provided})
    assert caller_session.owned_state is None
    assert caller_session.managed_state.global_.get("total", MISSING) == 9
    caller_session.close()
    assert provided.global_.get("total", MISSING) == 9


def test_definitions_are_not_modified() -> None:
    inner = child(handlers=[handler()], state={"source": {"n": 0}})
    definition = passive(
        steps=[*passive()["steps"], nested_step("a", inner), nested_step("b", inner)],
        handlers=[handler(workflow=note_workflow())],
    )
    snapshot = copy.deepcopy(definition)

    compile_(definition)

    assert definition == snapshot
    assert "version" not in definition["handlers"][0]["workflow"]


def test_draft_style_state_handler_counts_entries_through_managed_state() -> None:
    from roboflow_workflows.execution_engine.v2.blocks.state import StateIncrementBlock
    from roboflow_workflows.execution_engine.v2.state import MISSING

    catalogue = Catalogue([Zone, Note, StateIncrementBlock], sources=[Feed])
    count_entry = {
        "name": "count_entry",
        "on": "$steps.zone.events.entered",
        "execution": {"mode": "sync"},
        "bindings": {"zone": "$event.zone_id"},
        "workflow": {
            "inputs": [{"name": "zone", "kind": ["string"]}],
            "steps": [
                {
                    "type": "v2/state_increment",
                    "name": "global_count",
                    "trigger": "$inputs.zone",
                    "scope": "global",
                    "key": "entry_count",
                }
            ],
            "outputs": [
                {
                    "type": "JsonField",
                    "name": "entry_count",
                    "selector": "$steps.global_count.value",
                }
            ],
        },
    }
    plan = compile_workflow(
        passive(state={"global": {"entry_count": 10}}, handlers=[count_entry]),
        catalogue=catalogue,
    )
    session = plan.create_session()

    session.run({"value": 1.0})
    session.run({"value": 2.0})

    assert session.managed_state.global_.get("entry_count", MISSING) == 12
    session.close()


class FailingStateUser(Block):
    """Takes managed state in its constructor, keeps it, then fails."""

    type = "test/failing_state_user@v1"
    outputs = {"value": Output(FLOAT_KIND)}
    seen: list = []

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def __init__(self, *, managed_state) -> None:
        FailingStateUser.seen.append(managed_state)
        raise RuntimeError("constructor fails")

    def run(self, value):
        return {"value": value}


def test_failed_session_construction_closes_engine_owned_state() -> None:
    from roboflow_workflows.execution_engine.v2.errors import ResourceError
    from roboflow_workflows.execution_engine.v2.state import StateBackendError

    catalogue = Catalogue([Zone, Note, FailingStateUser], sources=[Feed])
    definition = passive(
        steps=[
            {"type": FailingStateUser.type, "name": "user", "value": "$inputs.value"}
        ],
        outputs=[
            {"type": "JsonField", "name": "value", "selector": "$steps.user.value"}
        ],
    )
    plan = compile_workflow(definition, catalogue=catalogue)
    FailingStateUser.seen.clear()

    with pytest.raises(ResourceError, match="constructor fails"):
        plan.create_session()

    (owned,) = FailingStateUser.seen
    with pytest.raises(StateBackendError, match="closed"):
        owned.global_.get("anything")
