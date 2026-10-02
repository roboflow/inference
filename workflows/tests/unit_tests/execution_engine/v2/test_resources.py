"""Tests of V2 constructor resources and execution-session construction."""

from typing import Optional, Tuple

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.data import EntryLayout
from roboflow_workflows.execution_engine.v2.declaration import Block, spec_of
from roboflow_workflows.execution_engine.v2.errors import ResourceError
from roboflow_workflows.execution_engine.v2.implementations import Implementation
from roboflow_workflows.execution_engine.v2.plan import CompiledWorkflow, PlannedStep
from roboflow_workflows.execution_engine.v2.resources import (
    Factory,
    ResourceResolver,
    ResourceSpec,
    read_resource_specs,
)


class Handle:
    """A resource that must never be copied."""

    def __deepcopy__(self, memo):
        raise AssertionError("resources must not be deep-copied")


class UsesAudit(Block):
    """Records its constructor arguments."""

    type = "test/uses_audit@v1"
    constructions = []

    def __init__(self, *, audit, api_key=None, callback=None):
        self.audit = audit
        self.api_key = api_key
        self.callback = callback
        UsesAudit.constructions.append(self)

    def run(self) -> dict:
        return {}


class NoResources(Block):
    """Has no constructor."""

    type = "test/no_resources@v1"

    def run(self) -> dict:
        return {}


class BrokenConstructor(Block):
    """Fails while being constructed."""

    type = "test/broken_constructor@v1"

    def __init__(self):
        raise RuntimeError("cannot connect")

    def run(self) -> dict:
        return {}


def _plan(
    *steps: Tuple[str, type], namespace: str = "core", providers=None
) -> CompiledWorkflow:
    catalogue = Catalogue(
        {block_class for _, block_class in steps},
        namespace=namespace,
        providers=providers,
    )
    planned = [
        PlannedStep(
            path=(name,),
            spec=spec_of(block_class),
            namespace=namespace,
            params=spec_of(block_class).validate_params({}),
            bindings=(),
            invocation_layout=EntryLayout(),
            outputs={},
        )
        for name, block_class in steps
    ]
    plan = CompiledWorkflow(
        inputs={}, steps=tuple(planned), outputs=(), catalogue=catalogue
    )

    return plan


def test_resource_specs_come_from_the_constructor_signature() -> None:
    specs = read_resource_specs(UsesAudit)

    assert [spec.name for spec in specs] == ["audit", "api_key", "callback"]
    assert [spec.required for spec in specs] == [True, False, False]
    assert read_resource_specs(NoResources) == ()


def test_precedence_scoped_then_generic_then_catalogue_then_default() -> None:
    specs = (ResourceSpec(name="token", required=False, default="default"),)
    providers = {"core": {"token": "catalogue"}}

    def resolve(provided):
        resolver = ResourceResolver(provided=provided, providers=providers)
        resolved = resolver.resolve(
            specs, namespace="core", step_path=("s",), block_type="t"
        )
        return resolved["token"]

    scoped = resolve({"core.token": "scoped", "token": "generic"})
    generic = resolve({"token": "generic"})
    catalogue = resolve({})
    default = ResourceResolver().resolve(
        specs, namespace="core", step_path=("s",), block_type="t"
    )["token"]

    assert (scoped.value, scoped.source) == ("scoped", "provided:core.token")
    assert (generic.value, generic.source) == ("generic", "provided:token")
    assert (catalogue.value, catalogue.source) == ("catalogue", "catalogue:core.token")
    assert (default.value, default.source) == ("default", "default")


def test_other_namespaces_do_not_leak() -> None:
    specs = (ResourceSpec(name="token", required=True),)
    resolver = ResourceResolver(
        provided={"other.token": "x"}, providers={"other": {"token": "y"}}
    )

    with pytest.raises(ResourceError, match="'core.token' or 'token'"):
        resolver.resolve(specs, namespace="core", step_path=("s",), block_type="t")


def test_values_keep_identity_and_callables_pass_through() -> None:
    handle = Handle()
    callback = lambda: "not called"  # noqa: E731
    plan = _plan(("a", UsesAudit))

    session = plan.create_session({"audit": handle, "callback": callback})

    instance = session.instances[("a",)]
    assert instance.audit is handle
    assert instance.callback is callback
    assert session.resources[("a",)]["audit"].source == "provided:audit"


def test_session_factory_is_created_once_and_shared_by_steps() -> None:
    created = []
    plan = _plan(("a", UsesAudit), ("b", UsesAudit))

    session = plan.create_session(
        {"audit": Factory(lambda: created.append(1) or object())}
    )

    first, second = session.instances[("a",)], session.instances[("b",)]
    assert len(created) == 1
    assert first.audit is second.audit


def test_step_factory_is_created_per_step() -> None:
    plan = _plan(("a", UsesAudit), ("b", UsesAudit))

    session = plan.create_session({"audit": Factory(list, scope="step")})

    assert session.instances[("a",)].audit is not session.instances[("b",)].audit


def test_catalogue_providers_supply_plugin_defaults() -> None:
    audit = []
    plan = _plan(("a", UsesAudit), providers={"audit": audit})

    session = plan.create_session()

    assert session.instances[("a",)].audit is audit
    assert session.resources[("a",)]["audit"].source == "catalogue:core.audit"


def test_each_step_is_constructed_once_per_session_and_sessions_are_independent() -> (
    None
):
    UsesAudit.constructions.clear()
    plan = _plan(("a", UsesAudit), ("b", UsesAudit))

    first = plan.create_session({"audit": []})
    second = plan.create_session({"audit": []})

    assert len(UsesAudit.constructions) == 4
    assert first.instances[("a",)] is not second.instances[("a",)]
    assert first.instances[("a",)] is not first.instances[("b",)]
    assert first.session_id != second.session_id


def test_plan_description_constructs_nothing() -> None:
    UsesAudit.constructions.clear()
    plan = _plan(("a", UsesAudit))

    description = plan.describe()

    assert UsesAudit.constructions == []
    assert description["steps"][0]["path"] == "$steps.a"


def test_missing_resource_names_step_and_parameter() -> None:
    plan = _plan(("child_step", UsesAudit))

    with pytest.raises(ResourceError) as error:
        plan.create_session()

    assert error.value.step_path == ("child_step",)
    assert error.value.parameter == "audit"
    assert "$steps.child_step" in str(error.value)


def test_failing_factory_and_constructor_report_context() -> None:
    def fail():
        raise OSError("no disk")

    with pytest.raises(
        ResourceError, match="factory for resource 'audit'"
    ) as factory_error:
        _plan(("a", UsesAudit)).create_session({"audit": Factory(fail)})
    with pytest.raises(ResourceError, match="cannot connect") as constructor_error:
        _plan(("a", BrokenConstructor)).create_session()

    assert isinstance(factory_error.value.__cause__, OSError)
    assert constructor_error.value.block_type == "test/broken_constructor@v1"


def test_factory_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="callable"):
        Factory("not callable")
    with pytest.raises(ValueError, match="scope"):
        Factory(list, scope="run")


class OptionalHandles(Block):
    """V1 object-detection shape: Optional without a default is required."""

    type = "test/optional_handles@v1"

    def __init__(self, *, api_key: Optional[str], manager, executor=None):
        self.api_key = api_key
        self.manager = manager
        self.executor = executor

    def run(self) -> dict:
        return {}


def test_precedence_is_observed_by_identity_at_the_constructor() -> None:
    scoped, generic, catalogue_value = object(), object(), object()

    def construct(provided):
        plan = _plan(("a", OptionalHandles), providers={"manager": catalogue_value})
        session = plan.create_session({"api_key": None, **provided})
        return session.instances[("a",)].manager

    assert construct({"core.manager": scoped, "manager": generic}) is scoped
    assert construct({"manager": generic}) is generic
    assert construct({}) is catalogue_value


def test_optional_without_default_is_required_and_explicit_none_is_kept() -> None:
    plan = _plan(("a", OptionalHandles))

    with pytest.raises(ResourceError, match="'api_key'"):
        plan.create_session({"manager": object()})
    session = plan.create_session({"api_key": None, "manager": object()})

    assert session.instances[("a",)].api_key is None
    assert session.resources[("a",)]["api_key"].source == "provided:api_key"
    assert session.resources[("a",)]["executor"].source == "default"


def test_session_factories_are_not_shared_between_sessions() -> None:
    plan = _plan(("a", UsesAudit), ("b", UsesAudit))
    provided = {"audit": Factory(list)}

    first = plan.create_session(provided)
    second = plan.create_session(provided)

    assert first.instances[("a",)].audit is first.instances[("b",)].audit
    assert first.instances[("a",)].audit is not second.instances[("a",)].audit


class FactoryDefaults(Block):
    """Constructor defaults: session and step factories, and a plain callable."""

    type = "test/factory_defaults@v1"

    def __init__(
        self,
        *,
        shared=Factory(dict),
        own=Factory(list, scope="step"),
        callback=print,
    ):
        self.shared = shared
        self.own = own
        self.callback = callback

    def run(self) -> dict:
        return {}


def test_factory_defaults_are_materialized_with_their_scope() -> None:
    plan = _plan(("a", FactoryDefaults), ("b", FactoryDefaults))

    first = plan.create_session()
    second = plan.create_session()

    a, b = first.instances[("a",)], first.instances[("b",)]
    assert isinstance(a.shared, dict) and a.shared is b.shared
    assert isinstance(a.own, list) and a.own is not b.own
    assert a.callback is print
    assert second.instances[("a",)].shared is not a.shared
    assert first.resources[("a",)]["shared"].source == "default"


def test_provided_value_replaces_a_factory_default() -> None:
    shared = {"key": "value"}
    plan = _plan(("a", FactoryDefaults))

    session = plan.create_session({"shared": shared})

    assert session.instances[("a",)].shared is shared


class Plain(Block):
    """Ordinary block without constructor resources."""

    type = "test/plain@v1"

    def run(self) -> dict:
        return {}


class Remote(Implementation):
    name = "remote"

    def __init__(self, *, endpoint) -> None:
        self.endpoint = endpoint

    def run(self) -> dict:
        return {}


class Local(Implementation):
    name = "local"

    def run(self) -> dict:
        return {}


class Served(Block):
    """Contract block: each implementation declares its own resources."""

    type = "test/served@v1"
    implementations = (Remote, Local)


def test_ordinary_block_resources_are_its_only_implementations_resources() -> None:
    audited, plain = spec_of(UsesAudit), spec_of(Plain)

    assert audited.resources is audited.implementations[0].resources
    assert [item.name for item in audited.resources] == [
        "audit",
        "api_key",
        "callback",
    ]
    # Known to need nothing, which differs from "not described here".
    assert plain.resources == ()
    assert plain.describe()["resources"] == []


def test_contract_block_resources_are_read_per_implementation() -> None:
    spec = spec_of(Served)
    remote, local = spec.implementations

    assert spec.resources is None
    assert "resources" not in spec.describe()
    assert [item.name for item in remote.resources] == ["endpoint"]
    assert local.resources == ()
    assert [item["resources"] for item in spec.describe()["implementations"]] == [
        [{"name": "endpoint", "required": True, "annotation": None}],
        [],
    ]
