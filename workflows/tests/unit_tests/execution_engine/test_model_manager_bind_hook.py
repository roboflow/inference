"""The `__workflows_bind__` hook contract, exercised without server imports.

The engine invokes an optional class-level `__workflows_bind__` on the
effective ``workflows_core.model_manager`` (namespaced-key precedence, MRO
lookup) to let a host fill in bindings a raw manager cannot provide. Nothing
here imports ``inference.core.interfaces.*`` or ``inference.core.managers.*``
- a hook that only works because the server is already imported would still
pass on the server-side suite; here the workflows package stands alone.
"""

from types import MethodType
from unittest.mock import MagicMock

import pytest
from roboflow_workflows.execution_engine.core import ExecutionEngine
from roboflow_workflows.execution_engine.v1.core import (
    DEFAULT_WORKFLOWS_STEP_ERROR_HANDLER,
    OMITTED_STEP_ERROR_HANDLER,
    REGISTERED_STEP_ERROR_HANDLERS,
    ExecutionEngineV1,
)

_TRIVIAL_WORKFLOW = {"version": "1.0", "inputs": [], "steps": [], "outputs": []}


class _HostManager:
    """Raw-manager stand-in exposing the class-level hook."""

    def __init__(self, effective_handler=OMITTED_STEP_ERROR_HANDLER):
        self._effective_handler = effective_handler
        self.received_step_error_handler = None

    def __workflows_bind__(self, init_parameters, step_error_handler):
        self.received_step_error_handler = step_error_handler
        init_parameters["workflows_core.model_manager"] = _StubProvider(self)
        init_parameters.setdefault("workflows_core.custom_marker", "installed")
        return self._effective_handler


class _StubProvider:
    """Provider written back by the hook. No ``__workflows_bind__``: reusing
    the same manager across engines cannot re-fire, and an explicit provider
    supplied by the caller reaches the engine unchanged."""

    def __init__(self, model_manager):
        self.model_manager = model_manager


def _v1(engine):
    return engine._engine  # type: ignore[attr-defined]


def _stored(engine):
    return _v1(engine)._compiled_workflow.init_parameters


# Effective-manager lookup: namespaced/bare precedence, class-level, no invoke.


@pytest.mark.parametrize(
    "init_parameters",
    [
        # Namespaced key wins over bare.
        {"workflows_core.model_manager": _HostManager(), "model_manager": object()},
        # Bare key alone still triggers the hook; the bare binding survives.
        {"model_manager": _HostManager()},
    ],
)
def test_hook_fires_for_effective_manager(init_parameters):
    bare_before = init_parameters.get("model_manager")
    engine = ExecutionEngine.init(
        workflow_definition=_TRIVIAL_WORKFLOW, init_parameters=init_parameters
    )
    stored = _stored(engine)
    assert isinstance(stored["workflows_core.model_manager"], _StubProvider)
    if bare_before is not None:
        assert stored["model_manager"] is bare_before


def test_explicit_none_on_namespaced_key_disables_hook_over_bare():
    # An explicit ``None`` means "no manager"; the bare binding must NOT be
    # promoted by the hook path.
    bare = _HostManager()
    engine = ExecutionEngine.init(
        workflow_definition=_TRIVIAL_WORKFLOW,
        init_parameters={
            "workflows_core.model_manager": None,
            "model_manager": bare,
        },
    )
    stored = _stored(engine)
    assert stored["workflows_core.model_manager"] is None
    assert stored["model_manager"] is bare


def test_permissive_getattr_does_not_opt_in():
    # The mock's TYPE is ``MagicMock`` (no ``__workflows_bind__``); instance
    # attribute access is permissive but the class-level lookup must miss.
    permissive = MagicMock()
    engine = ExecutionEngine.init(
        workflow_definition=_TRIVIAL_WORKFLOW,
        init_parameters={"workflows_core.model_manager": permissive},
    )
    assert _stored(engine)["workflows_core.model_manager"] is permissive


def test_instance_only_attribute_does_not_opt_in():
    fresh = _HostManager()

    class _Bare:
        pass

    bare = _Bare()
    bare.__workflows_bind__ = MethodType(  # type: ignore[attr-defined]
        _HostManager.__workflows_bind__, fresh
    )
    engine = ExecutionEngine.init(
        workflow_definition=_TRIVIAL_WORKFLOW,
        init_parameters={"workflows_core.model_manager": bare},
    )
    assert _stored(engine)["workflows_core.model_manager"] is bare


def test_inherited_hook_still_fires():
    # ``ModelManagerDecorator`` subclasses ``ModelManager``; a subclass that
    # inherits ``__workflows_bind__`` must still opt in.
    class _Sub(_HostManager):
        pass

    sub = _Sub()
    engine = ExecutionEngine.init(
        workflow_definition=_TRIVIAL_WORKFLOW,
        init_parameters={"workflows_core.model_manager": sub},
    )
    assert isinstance(_stored(engine)["workflows_core.model_manager"], _StubProvider)


# step_error_handler semantics through the hook seam.


def test_engine_uses_hook_returned_handler_for_omitted_argument():
    def handler(step_name, error):
        return None

    manager = _HostManager(effective_handler=handler)
    engine = ExecutionEngine.init(
        workflow_definition=_TRIVIAL_WORKFLOW,
        init_parameters={"workflows_core.model_manager": manager},
    )
    assert manager.received_step_error_handler is OMITTED_STEP_ERROR_HANDLER
    assert _v1(engine)._step_error_handler is handler


def test_v1_direct_entry_point_also_fires_hook_and_uses_returned_handler():
    def handler(step_name, error):
        return None

    manager = _HostManager(effective_handler=handler)
    v1_engine = ExecutionEngineV1.init(
        workflow_definition=_TRIVIAL_WORKFLOW,
        init_parameters={"workflows_core.model_manager": manager},
    )
    assert v1_engine._step_error_handler is handler


def test_engine_forwards_explicit_callable_to_hook_and_preserves_identity():
    def explicit(step_name, error):
        return None

    manager = _HostManager(effective_handler=explicit)
    engine = ExecutionEngine.init(
        workflow_definition=_TRIVIAL_WORKFLOW,
        init_parameters={"workflows_core.model_manager": manager},
        step_error_handler=explicit,
    )
    assert manager.received_step_error_handler is explicit
    assert _v1(engine)._step_error_handler is explicit


def test_standalone_default_applies_when_no_hook_and_omitted():
    engine = ExecutionEngine.init(workflow_definition=_TRIVIAL_WORKFLOW)
    expected = REGISTERED_STEP_ERROR_HANDLERS[DEFAULT_WORKFLOWS_STEP_ERROR_HANDLER]
    assert _v1(engine)._step_error_handler is expected


# Non-hook providers, private dict, per-engine lifetime.


def test_explicit_provider_reaches_engine_unchanged():
    provider = _StubProvider(model_manager=object())
    caller_dict = {
        "workflows_core.model_manager": provider,
        "workflows_core.custom_marker": "caller-value",
    }
    engine = ExecutionEngine.init(
        workflow_definition=_TRIVIAL_WORKFLOW, init_parameters=caller_dict
    )
    stored = _stored(engine)
    assert stored["workflows_core.model_manager"] is provider
    assert stored["workflows_core.custom_marker"] == "caller-value"


def test_caller_dictionary_is_never_mutated():
    manager = _HostManager()
    caller_dict = {"workflows_core.model_manager": manager}
    ExecutionEngine.init(
        workflow_definition=_TRIVIAL_WORKFLOW, init_parameters=caller_dict
    )
    assert set(caller_dict) == {"workflows_core.model_manager"}
    assert caller_dict["workflows_core.model_manager"] is manager
