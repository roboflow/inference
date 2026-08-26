"""Result-serialisation contract for the Modal sandbox's HTTP endpoint.

A ``BlockResult`` is a LIST whenever the block increases output dimensionality
(offset-1) or declares ``batch_oriented_parameters`` — one entry per element.
The sandbox used to hand that straight to a dict-only serialiser, so those
blocks failed on the return trip with
``AttributeError: 'list' object has no attribute 'items'``.

These drive the real ``Executor.execute_block`` end to end (user code is
compiled and run in-process) so the assertion covers the shipped path rather
than a reimplementation of it.
"""

import asyncio
import importlib.util
import json
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
    def __init__(self, name: str):
        self.name = name

    def cls(self, *args, **kwargs):
        return lambda cls: cls


def _identity_decorator(*args, **kwargs):
    return lambda obj: obj


@pytest.fixture()
def modal_app_with_fake_modal(monkeypatch):
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
    spec = importlib.util.spec_from_file_location(
        "modal_app_result_serialization_test", modal_app_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeRequest:
    """Minimal stand-in for starlette's Request: body() + headers.get()."""

    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()
        self.headers = {}

    async def body(self) -> bytes:
        return self._body


def _run_block(module, code: str) -> dict:
    executor = module.Executor.__new__(module.Executor)
    executor._code_namespaces = {}
    executor._shared_globals = {}
    request = _FakeRequest(
        {
            "code_str": code,
            "imports": [],
            "run_function_name": "run",
            "inputs_json": "{}",
        }
    )
    return asyncio.run(executor.execute_block(request))


def test_execute_block_serialises_list_shaped_result(
    modal_app_with_fake_modal,
) -> None:
    """offset-1 / batch-oriented blocks return one entry per element."""
    response = _run_block(
        modal_app_with_fake_modal,
        "def run():\n"
        "    return [{'measurement': {'w': 1}}, {'measurement': {'w': 2}}]",
    )

    assert response["success"] is True, response.get("error")
    assert json.loads(response["result"]) == [
        {"measurement": {"w": 1}},
        {"measurement": {"w": 2}},
    ]


def test_execute_block_list_result_survives_nested_values(
    modal_app_with_fake_modal,
) -> None:
    response = _run_block(
        modal_app_with_fake_modal,
        "def run():\n    return [{'v': [1, 2]}, {'v': []}]",
    )

    assert response["success"] is True, response.get("error")
    assert json.loads(response["result"]) == [{"v": [1, 2]}, {"v": []}]


def test_execute_block_dict_shaped_result_is_unchanged(
    modal_app_with_fake_modal,
) -> None:
    """The ordinary (offset-0) contract must keep byte-identical behaviour."""
    response = _run_block(
        modal_app_with_fake_modal,
        "def run():\n    return {'measurement': {'w': 1}}",
    )

    assert response["success"] is True, response.get("error")
    assert json.loads(response["result"]) == {"measurement": {"w": 1}}
