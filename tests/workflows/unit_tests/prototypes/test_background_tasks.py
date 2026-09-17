import inspect

from fastapi import BackgroundTasks

from inference.core.workflows.prototypes.background_tasks import BackgroundTaskScheduler


def test_protocol_declares_add_task() -> None:
    assert hasattr(BackgroundTaskScheduler, "add_task")


def test_fastapi_background_tasks_satisfies_the_protocol() -> None:
    # FastAPI's BackgroundTasks is what the HTTP layer injects in production.
    assert hasattr(BackgroundTasks, "add_task")


def _params(method):
    # Skip `self`; the port is the source of truth for which names to iterate.
    return list(inspect.signature(method).parameters.values())[1:]


def test_add_task_signature_matches_fastapi_background_tasks() -> None:
    # Annotations deliberately not compared: the port uses Any, fastapi uses
    # ParamSpec-typed Callable/args/kwargs.
    port_params = _params(BackgroundTaskScheduler.add_task)
    real_params = _params(BackgroundTasks.add_task)
    assert [p.name for p in port_params] == [p.name for p in real_params]
    for port_param, real_param in zip(port_params, real_params):
        assert port_param.kind == real_param.kind
        assert port_param.default == real_param.default
