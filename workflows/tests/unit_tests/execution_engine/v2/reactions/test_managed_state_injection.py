"""Managed state injection through normal resource resolution (decision 009).

Every compiled plan here has one parent step and one handler step requesting
``managed_state``::

    parent   ping ──entered──▶ handler "react"
             keep (custom)              keep (custom | root namespace)

The session, the parent step and the handler step must see one service: the
view with the declared defaults of whatever the resolver chose.
"""

import os
import subprocess
import sys
import textwrap
from typing import Any, Dict, List

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.errors import ResourceError
from roboflow_workflows.execution_engine.v2.events import Event
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND
from roboflow_workflows.execution_engine.v2.resources import Factory
from roboflow_workflows.execution_engine.v2.sources import (
    Source,
    SourceOutput,
    SourceParams,
)
from roboflow_workflows.execution_engine.v2.state import (
    MISSING,
    ManagedState,
    StateBackendError,
)

PARENT = ("keep",)
HANDLER = ("react",)
DEFAULTS = {"global": {"seed": 7}, "source": {"visits": 0}}


class Ping(Block):
    type = "test/ping@v1"
    outputs = {"value": Output(FLOAT_KIND)}
    events = {"entered": Event({"value": FLOAT_KIND})}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, value):
        self.emit("entered", value=value)
        return {"value": value}


class Keep(Block):
    """Keeps its managed state; registered in namespace ``custom``."""

    type = "test/keep@v1"
    outputs = {"value": Output(FLOAT_KIND)}
    seen: List[Any] = []

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def __init__(self, *, managed_state) -> None:
        Keep.seen.append(managed_state)

    def run(self, value):
        return {"value": value}


class RootKeep(Keep):
    """Root-namespace ``Keep`` whose resource has a constructor default."""

    type = "test/root_keep@v1"

    def __init__(self, *, managed_state=None) -> None:
        self.managed_state = managed_state


class Breaks(Keep):
    type = "test/breaks@v1"

    def __init__(self) -> None:
        raise RuntimeError("constructor fails")


class StatefulFeed(Source):
    type = "test/stateful_feed@v1"
    outputs = {"value": SourceOutput(FLOAT_KIND)}

    class Params(SourceParams):
        pass

    def __init__(self, *, managed_state) -> None:
        self.managed_state = managed_state

    def open(self) -> None:
        pass

    def read(self):
        return None

    def close(self) -> None:
        pass


def keep_with_default(default: Any, type_: str = "test/default_keep@v1") -> type:
    """Return a custom ``Keep`` whose ``managed_state`` defaults to ``default``."""

    class DefaultKeep(Keep):
        type = type_

        def __init__(self, *, managed_state=default) -> None:
            Keep.seen.append(managed_state)

    return DefaultKeep


def catalogue(*custom: type, custom_sources=(), **providers: Any) -> Catalogue:
    merged = Catalogue.merge(
        Catalogue([Ping, RootKeep, Breaks], sources=[StatefulFeed]),
        Catalogue(
            [Keep, *custom],
            sources=custom_sources,
            namespace="custom",
            providers=providers,
        ),
    )

    return merged


def definition(
    handler_step: str = Keep.type, parent_step: str = Keep.type, **extra: Any
) -> Dict[str, Any]:
    react = {
        "name": "react",
        "on": "$steps.ping.events.entered",
        "bindings": {"value": "$event.value"},
        "workflow": {
            "inputs": [{"name": "value", "kind": ["float"]}],
            "steps": [{"type": handler_step, "name": "keep", "value": "$inputs.value"}],
            "outputs": [
                {"type": "JsonField", "name": "value", "selector": "$steps.keep.value"}
            ],
        },
    }
    declared = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "value"}],
        "steps": [
            {"type": Ping.type, "name": "ping", "value": "$inputs.value"},
            {"type": parent_step, "name": "keep", "value": "$steps.ping.value"},
        ],
        "outputs": [
            {"type": "JsonField", "name": "value", "selector": "$steps.keep.value"}
        ],
        "handlers": [react],
        "state": DEFAULTS,
    }
    declared.update(extra)

    return declared


def compile_(
    handler_step: str = Keep.type,
    *,
    parent_step: str = Keep.type,
    custom=(),
    custom_sources=(),
    providers=None,
    **extra: Any,
):
    plan = compile_workflow(
        definition(handler_step, parent_step, **extra),
        catalogue=catalogue(
            *custom, custom_sources=custom_sources, **(providers or {})
        ),
    )

    return plan


def compile_source_plan(feed: type):
    """Compile the plan with a ``cam`` source of type ``feed`` feeding ``ping``."""
    output_group = {
        "type": "OutputGroup",
        "name": "frames",
        "anchor": "$sources.cam.value",
        "outputs": [
            {"type": "JsonField", "name": "value", "selector": "$steps.keep.value"}
        ],
    }
    plan = compile_(
        custom_sources=[] if feed is StatefulFeed else [feed],
        sources=[{"type": feed.type, "name": "cam"}],
        inputs=[],
        steps=[
            {"type": Ping.type, "name": "ping", "value": "$sources.cam.value"},
            {"type": Keep.type, "name": "keep", "value": "$steps.ping.value"},
        ],
        outputs=[output_group],
    )

    return plan


def injected(session) -> List[Any]:
    """Return the managed state of the parent step and of the handler step."""
    handler_session = session.handler_sessions[HANDLER]
    values = [
        session.resources[PARENT]["managed_state"].value,
        handler_session.resources[PARENT]["managed_state"].value,
    ]

    return values


def assert_one_configured_service(session, original: ManagedState) -> None:
    parent, handler = injected(session)
    assert parent is handler is session.managed_state
    assert session.handler_sessions[HANDLER].managed_state is session.managed_state
    # The view carries the defaults; they were seeded into the chosen service.
    assert session.managed_state is not original
    assert original.global_.get("seed", MISSING) == 7


def test_unscoped_caller_state_is_seeded_and_shared() -> None:
    caller = ManagedState()
    plan = compile_()

    session = plan.create_session(resources={"managed_state": caller})

    assert_one_configured_service(session, caller)
    assert session.owned_state is None
    session.close()
    caller.global_.set("after_close", 1)


def test_namespaced_caller_state_reaches_blocks_through_the_view() -> None:
    caller = ManagedState()
    plan = compile_()

    session = plan.create_session(resources={"custom.managed_state": caller})

    assert_one_configured_service(session, caller)
    assert session.owned_state is None
    sources = {
        session.resources[PARENT]["managed_state"].source,
        session.handler_sessions[HANDLER].resources[PARENT]["managed_state"].source,
    }
    assert sources == {"provided:custom.managed_state"}


def test_caller_factory_is_created_once_and_borrowed() -> None:
    created: List[ManagedState] = []

    def create() -> ManagedState:
        created.append(ManagedState())
        return created[-1]

    plan = compile_()

    session = plan.create_session(resources={"managed_state": Factory(create)})

    (state,) = created
    assert_one_configured_service(session, state)
    assert session.owned_state is None
    session.close()
    state.global_.set("after_close", 1)


def test_catalogue_provider_is_resolved_and_not_masked() -> None:
    created: List[ManagedState] = []

    def create() -> ManagedState:
        created.append(ManagedState())
        return created[-1]

    plan = compile_(providers={"managed_state": Factory(create)})

    session = plan.create_session()

    (state,) = created
    assert_one_configured_service(session, state)
    assert session.owned_state is None


def test_engine_state_is_owned_seeded_and_shared() -> None:
    plan = compile_()

    session = plan.create_session()

    assert_one_configured_service(session, session.owned_state)
    session.close()
    with pytest.raises(StateBackendError, match="closed"):
        session.managed_state.global_.get("seed")


def test_demands_without_a_choice_share_the_chosen_service() -> None:
    caller = ManagedState()
    plan = compile_(RootKeep.type)

    session = plan.create_session(resources={"custom.managed_state": caller})

    assert_one_configured_service(session, caller)


def test_different_explicit_services_are_rejected_before_construction() -> None:
    plan = compile_(RootKeep.type)
    resources = {
        "custom.managed_state": ManagedState(),
        "managed_state": ManagedState(),
    }

    with pytest.raises(ResourceError, match="different managed state services"):
        plan.create_session(resources=resources)

    same = ManagedState()
    session = plan.create_session(
        resources={"custom.managed_state": same, "managed_state": same}
    )
    assert_one_configured_service(session, same)


def test_constructor_default_is_the_borrowed_shared_service() -> None:
    default = ManagedState()
    block = keep_with_default(default)
    plan = compile_(block.type, parent_step=block.type, custom=[block])

    session = plan.create_session()

    assert_one_configured_service(session, default)
    assert session.owned_state is None
    session.close()
    default.global_.set("after_close", 1)


def test_constructor_default_factory_is_created_once_and_borrowed() -> None:
    created: List[ManagedState] = []

    def create() -> ManagedState:
        created.append(ManagedState())
        return created[-1]

    block = keep_with_default(Factory(create))
    plan = compile_(block.type, parent_step=block.type, custom=[block])

    session = plan.create_session()

    (state,) = created
    assert_one_configured_service(session, state)
    assert session.owned_state is None


@pytest.mark.parametrize("via_provider", [False, True])
def test_caller_and_provider_services_override_a_constructor_default(
    via_provider,
) -> None:
    default, chosen = ManagedState(), ManagedState()
    block = keep_with_default(default)
    providers = {"managed_state": chosen} if via_provider else {}
    resources = {} if via_provider else {"managed_state": chosen}
    plan = compile_(
        block.type, parent_step=block.type, custom=[block], providers=providers
    )

    session = plan.create_session(resources=resources)

    assert_one_configured_service(session, chosen)
    assert default.global_.get("seed", MISSING) is MISSING


def test_none_default_beside_a_default_service_shares_it() -> None:
    default = ManagedState()
    block = keep_with_default(default)
    plan = compile_(RootKeep.type, parent_step=block.type, custom=[block])

    session = plan.create_session()

    assert_one_configured_service(session, default)


def test_different_constructor_defaults_conflict_and_aliases_share() -> None:
    first = keep_with_default(ManagedState())
    other = keep_with_default(ManagedState(), type_="test/other_keep@v1")
    plan = compile_(other.type, parent_step=first.type, custom=[first, other])

    with pytest.raises(ResourceError, match="constructor default of"):
        plan.create_session()

    same = ManagedState()
    first, alias = keep_with_default(same), keep_with_default(same, other.type)
    plan = compile_(alias.type, parent_step=first.type, custom=[first, alias])
    session = plan.create_session()
    assert_one_configured_service(session, same)


def test_constructor_default_conflicts_with_a_namespaced_caller_service() -> None:
    default = ManagedState()
    block = keep_with_default(default)
    root = keep_with_default(ManagedState(), type_="test/root_default@v1")
    # "custom.managed_state" wins for the parent; the root default differs.
    plan = compile_workflow(
        definition(root.type, block.type),
        catalogue=Catalogue.merge(catalogue(block), Catalogue([root])),
    )

    with pytest.raises(ResourceError, match="different managed state services"):
        plan.create_session(resources={"custom.managed_state": ManagedState()})

    assert default.global_.get("seed", MISSING) is MISSING


def test_source_constructor_default_is_the_session_service() -> None:
    default = ManagedState()

    class DefaultFeed(StatefulFeed):
        type = "test/default_feed@v1"

        def __init__(self, *, managed_state=default) -> None:
            self.managed_state = managed_state

    session = compile_source_plan(DefaultFeed).create_session()

    source_state = session.source_resources["cam"]["managed_state"].value
    assert source_state is session.managed_state
    assert_one_configured_service(session, default)


def test_unused_keys_and_other_factories_are_untouched() -> None:
    def fail() -> Any:
        raise AssertionError("an unused factory was created")

    caller = ManagedState()
    plan = compile_()
    resources = {
        "managed_state": caller,
        "other.managed_state": Factory(fail),
        "model": Factory(fail),
    }

    session = plan.create_session(resources=resources)

    assert_one_configured_service(session, caller)
    assert isinstance(resources["model"], Factory)


def test_source_constructor_receives_the_configured_service() -> None:
    caller = ManagedState()
    plan = compile_source_plan(StatefulFeed)

    session = plan.create_session(resources={"custom.managed_state": caller})

    source_state = session.source_resources["cam"]["managed_state"].value
    assert source_state is session.managed_state
    assert_one_configured_service(session, caller)


@pytest.mark.parametrize("chooser", ["engine", "caller", "default"])
def test_later_constructor_failure_closes_only_engine_state(chooser) -> None:
    borrowed = ManagedState()
    resources = {"managed_state": borrowed} if chooser == "caller" else {}
    block = keep_with_default(borrowed if chooser == "default" else None)
    plan = compile_(Breaks.type, parent_step=block.type, custom=[block])
    Keep.seen.clear()

    # The parent step is built before the handler session whose step fails.
    with pytest.raises(ResourceError, match="constructor fails"):
        plan.create_session(resources=resources)

    (view,) = Keep.seen
    if chooser != "engine":
        assert view.global_.get("seed", MISSING) == 7
        assert borrowed.global_.get("seed", MISSING) == 7
        return
    with pytest.raises(StateBackendError, match="closed"):
        view.global_.get("seed")


LAZY_IMPORT_PROBE = textwrap.dedent("""
    import sys
    from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
    from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
    from roboflow_workflows.execution_engine.v2.declaration import (
        Block, BlockParams, Output, Ref,
    )
    from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND

    class Echo(Block):
        type = "test/echo@v1"
        outputs = {"value": Output(FLOAT_KIND)}

        class Params(BlockParams):
            value: Ref(FLOAT_KIND)

        def run(self, value):
            return {"value": value}

    def loaded():
        prefix = "roboflow_workflows.execution_engine.v2.state"
        return sorted(name for name in sys.modules if name.startswith(prefix))

    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "value"}],
        "steps": [{"type": Echo.type, "name": "echo", "value": "$inputs.value"}],
        "outputs": [
            {"type": "JsonField", "name": "value", "selector": "$steps.echo.value"}
        ],
    }
    plan = compile_workflow(definition, catalogue=Catalogue([Echo]))
    plan.create_session().run({"value": 1.0})
    print(loaded())
    stateful = compile_workflow(
        {**definition, "state": {"global": {"n": 0}}}, catalogue=Catalogue([Echo])
    )
    stateful.create_session().close()
    print("roboflow_workflows.execution_engine.v2.state.session" in loaded())
    """)


def test_plans_without_state_never_import_the_state_package() -> None:
    completed = subprocess.run(
        [sys.executable, "-B", "-c", LAZY_IMPORT_PROBE],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == ["[]", "True"]
