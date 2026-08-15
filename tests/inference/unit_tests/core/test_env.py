import importlib
import os

from inference.core import env as env_module


def _resolve_runs_on_jetson(monkeypatch, runs_on_jetson=None, running_on_jetson=None):
    with monkeypatch.context() as env_context:
        if runs_on_jetson is None:
            env_context.delenv("RUNS_ON_JETSON", raising=False)
        else:
            env_context.setenv("RUNS_ON_JETSON", runs_on_jetson)
        if running_on_jetson is None:
            env_context.delenv("RUNNING_ON_JETSON", raising=False)
        else:
            env_context.setenv("RUNNING_ON_JETSON", running_on_jetson)
        importlib.reload(env_module)
        result = env_module.RUNS_ON_JETSON
    importlib.reload(env_module)
    return result


def test_runs_on_jetson_uses_running_on_jetson_fallback_when_unset(
    monkeypatch,
) -> None:
    assert (
        _resolve_runs_on_jetson(
            monkeypatch, runs_on_jetson=None, running_on_jetson="True"
        )
        is True
    )


def test_runs_on_jetson_explicit_false_overrides_running_on_jetson(
    monkeypatch,
) -> None:
    assert (
        _resolve_runs_on_jetson(
            monkeypatch, runs_on_jetson="False", running_on_jetson="True"
        )
        is False
    )


def test_runs_on_jetson_preserves_false_default_when_both_names_unset(
    monkeypatch,
) -> None:
    assert (
        _resolve_runs_on_jetson(
            monkeypatch, runs_on_jetson=None, running_on_jetson=None
        )
        is False
    )


def test_assume_identity_service_access_token_uses_roboflow_env_name() -> None:
    original_scoped_token = os.environ.get(
        "ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN"
    )
    original_token = os.environ.get("ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN")
    try:
        os.environ.pop("ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN", None)
        os.environ["ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN"] = "assume-token"

        importlib.reload(env_module)

        assert (
            env_module.ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN == "assume-token"
        )
    finally:
        if original_scoped_token is None:
            os.environ.pop("ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN", None)
        else:
            os.environ["ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN"] = (
                original_scoped_token
            )
        if original_token is None:
            os.environ.pop("ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN", None)
        else:
            os.environ["ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN"] = original_token
        importlib.reload(env_module)
