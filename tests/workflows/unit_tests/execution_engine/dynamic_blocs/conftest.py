"""Shared scaffolding for tests that load ``modal/modal_app.py`` directly.

``modal_app`` imports the real ``modal`` package at module scope and decorates
``Executor`` with ``@app.cls`` / ``@modal.concurrent`` / ``@modal.fastapi_endpoint``.
Stubbing those out lets the sandbox-side code be imported and driven in-process,
so tests can exercise the shipped path instead of reimplementing it.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


class FakeModalImage:
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


class FakeModalApp:
    def __init__(self, name: str):
        self.name = name

    def cls(self, *args, **kwargs):
        return lambda cls: cls


def identity_decorator(*args, **kwargs):
    return lambda obj: obj


@pytest.fixture()
def modal_app_with_fake_modal(monkeypatch):
    """Import ``modal/modal_app.py`` with a stubbed ``modal`` package."""
    fake_modal = ModuleType("modal")
    fake_modal.App = FakeModalApp
    fake_modal.Image = FakeModalImage
    fake_modal.parameter = lambda *args, **kwargs: None
    fake_modal.enter = identity_decorator
    fake_modal.fastapi_endpoint = identity_decorator
    fake_modal.asgi_app = identity_decorator
    fake_modal.concurrent = identity_decorator
    monkeypatch.setitem(sys.modules, "modal", fake_modal)

    modal_app_path = Path(__file__).resolve().parents[5] / "modal" / "modal_app.py"
    spec = importlib.util.spec_from_file_location(
        "modal_app_under_test", modal_app_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
