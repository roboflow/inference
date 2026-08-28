"""Shared harness for tests that load the real ``modal/modal_app.py``.

``modal_app.py`` is deployed separately (``modal deploy``) and is not part of
the ``inference`` package, so it is imported here by path with a stubbed
``modal`` module. Loading the real server module is what makes the websocket
protocol tests genuine behavior tests rather than mock theater.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


class _FakeModalImage:
    @classmethod
    def debian_slim(cls, *args, **kwargs):
        return cls()

    @classmethod
    def from_registry(cls, *args, **kwargs):
        return cls()

    def apt_install(self, *args, **kwargs):
        return self

    def pip_install(self, *args, **kwargs):
        return self

    def entrypoint(self, *args, **kwargs):
        return self


class _FakeModalApp:
    def __init__(self, *args, **kwargs):
        pass

    def cls(self, *args, **kwargs):
        return lambda cls: cls


def _identity_decorator(*args, **kwargs):
    return lambda obj: obj


def load_modal_app(monkeypatch, module_name: str):
    """Import ``modal/modal_app.py`` fresh, with ``modal`` stubbed out.

    A fresh module per test keeps container-local state (namespaces, session
    and dedup registries) from leaking between tests.
    """
    fake_modal = ModuleType("modal")
    fake_modal.App = _FakeModalApp
    fake_modal.Image = _FakeModalImage
    fake_modal.parameter = lambda *args, **kwargs: None
    fake_modal.enter = _identity_decorator
    fake_modal.fastapi_endpoint = _identity_decorator
    fake_modal.asgi_app = _identity_decorator
    fake_modal.concurrent = _identity_decorator
    monkeypatch.setitem(sys.modules, "modal", fake_modal)

    modal_app_path = Path(__file__).resolve().parents[5] / "modal" / "modal_app.py"
    spec = importlib.util.spec_from_file_location(module_name, modal_app_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_ws_app(modal_app, run_user_code):
    """Build the websocket ASGI app with only user-code execution stubbed."""
    cls = modal_app.Executor
    user_cls = cls._get_user_cls() if hasattr(cls, "_get_user_cls") else cls
    executor = user_cls.__new__(user_cls)
    executor.workspace_id = "test-ws"
    user_cls.identify(executor)
    user_cls._run_user_code_ws = staticmethod(run_user_code).__func__
    return executor, user_cls.wsapp(executor)


@pytest.fixture()
def modal_app(monkeypatch):
    return load_modal_app(monkeypatch, "modal_app_under_test")
