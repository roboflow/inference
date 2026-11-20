"""The sandbox's ``self`` stays per call on both transports.

Every call gets a class of its own, so writes to ``type(self)`` cannot reach
later calls of the same or another block in the container, and ``self`` can be
weak-referenced. These drive the real ``Executor.execute_block`` (HTTP) and
``Executor._run_user_code_ws`` handlers on one executor.
"""

import asyncio
import json

import pytest


class _FakeRequest:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()
        self.headers = {}

    async def body(self) -> bytes:
        return self._body


def _new_executor(modal_app):
    executor = modal_app.Executor.__new__(modal_app.Executor)
    executor._code_namespaces = {}
    executor._shared_globals = {}
    executor._namespace_lock = None
    return executor


def _run_http(modal_app, executor, code: str, block_state_json=None) -> dict:
    payload = {
        "code_str": code,
        "imports": [],
        "run_function_name": "run",
        "inputs_json": "{}",
    }
    if block_state_json is not None:
        payload["block_state_json"] = block_state_json
    return asyncio.run(executor.execute_block(_FakeRequest(payload)))


def _run_ws(modal_app, executor, code: str, block_state_json=None) -> dict:
    return modal_app.Executor._run_user_code_ws(
        executor,
        code,
        [],
        "run",
        {},
        block_state_json=block_state_json,
    )


def _result(response: dict) -> dict:
    assert response["success"] is True, response.get("error")
    result = response["result"]
    return json.loads(result) if isinstance(result, str) else result


def _echo(response: dict) -> dict:
    state = response["block_state"]
    return json.loads(state) if isinstance(state, str) else state


WRITE_CLASS = (
    "def run(self):\n"
    "    type(self).cache = {'secret': 'from-block-A'}\n"
    "    type(self).n = property(lambda s: 0)\n"
    "    return {'ok': True}\n"
)
READ_CLASS = (
    "def run(self):\n"
    "    self.n = 1\n"
    "    return {'cache': getattr(type(self), 'cache', 'NOT VISIBLE')}\n"
)
WEAKREF_SELF = (
    "import weakref\n"
    "def run(self):\n"
    "    ref = weakref.ref(self)\n"
    "    self.n = getattr(self, 'n', 0) + 1\n"
    "    return {'alive': ref() is self}\n"
)


@pytest.mark.parametrize("transport", [_run_http, _run_ws], ids=["http", "ws"])
@pytest.mark.parametrize("state", [None, "{}"], ids=["no_state", "with_state"])
def test_class_writes_do_not_reach_another_block(
    modal_app_with_fake_modal, transport, state
) -> None:
    executor = _new_executor(modal_app_with_fake_modal)

    _result(transport(modal_app_with_fake_modal, executor, WRITE_CLASS, state))
    response = transport(modal_app_with_fake_modal, executor, READ_CLASS, state)

    assert _result(response) == {"cache": "NOT VISIBLE"}


@pytest.mark.parametrize("transport", [_run_http, _run_ws], ids=["http", "ws"])
def test_class_writes_do_not_reach_the_next_call_of_the_same_block(
    modal_app_with_fake_modal, transport
) -> None:
    executor = _new_executor(modal_app_with_fake_modal)
    code = (
        "def run(self):\n"
        "    seen = getattr(type(self), 'calls', 0)\n"
        "    type(self).calls = seen + 1\n"
        "    return {'seen': seen}\n"
    )

    first = transport(modal_app_with_fake_modal, executor, code, "{}")
    second = transport(modal_app_with_fake_modal, executor, code, "{}")

    assert _result(first) == {"seen": 0}
    assert _result(second) == {"seen": 0}


@pytest.mark.parametrize("transport", [_run_http, _run_ws], ids=["http", "ws"])
@pytest.mark.parametrize(
    "state, expected_echo",
    [(None, None), ('{"n": 7}', {"n": 8})],
    ids=["no_state", "with_state"],
)
def test_self_can_be_weak_referenced(
    modal_app_with_fake_modal, transport, state, expected_echo
) -> None:
    executor = _new_executor(modal_app_with_fake_modal)

    response = transport(modal_app_with_fake_modal, executor, WEAKREF_SELF, state)

    assert _result(response) == {"alive": True}
    if expected_echo is not None:
        assert _echo(response) == expected_echo
