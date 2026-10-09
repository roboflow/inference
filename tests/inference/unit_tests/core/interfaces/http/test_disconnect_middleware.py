"""Request-body ownership and watcher lifetime regressions."""

import anyio
import pytest

from inference.core.interfaces.http.middlewares.disconnect import (
    REQUEST_DISCONNECT_STATE_KEY,
    RequestDisconnectMiddleware,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_inactive_monitor_preserves_chunked_body_and_response():
    chunks = [
        {"type": "http.request", "body": b"first", "more_body": True},
        {"type": "http.request", "body": b"second", "more_body": False},
    ]
    observed = []
    outgoing = []
    index = 0

    async def receive():
        nonlocal index
        assert index < len(chunks), "An inactive watcher read beyond the body"
        message = chunks[index]
        index += 1
        return message

    async def app(scope, receive, send):
        observed.extend([await receive(), await receive()])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def send(message):
        outgoing.append(message)

    await RequestDisconnectMiddleware(app)({"type": "http"}, receive, send)
    assert observed == chunks
    assert outgoing[-1]["body"] == b"ok"


@pytest.mark.anyio
@pytest.mark.parametrize("raise_error", [False, True])
async def test_pending_watcher_is_closed_on_response_or_error(raise_error):
    receiving = anyio.Event()
    receive_closed = anyio.Event()
    failure = ValueError("endpoint failed")

    async def receive():
        receiving.set()
        try:
            await anyio.sleep_forever()
        finally:
            receive_closed.set()

    async def app(scope, receive, send):
        state = scope["state"][REQUEST_DISCONNECT_STATE_KEY]
        state.start_monitoring()
        await receiving.wait()
        if raise_error:
            raise failure

        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def send(message):
        pass

    with anyio.fail_after(2):
        if raise_error:
            with pytest.raises(ValueError) as error:
                await RequestDisconnectMiddleware(app)({"type": "http"}, receive, send)
            assert error.value is failure
        else:
            await RequestDisconnectMiddleware(app)({"type": "http"}, receive, send)
    assert receive_closed.is_set()


@pytest.mark.anyio
async def test_lifespan_and_websocket_are_not_wrapped():
    async def receive():
        return {}

    async def send(message):
        pass

    async def app(scope, app_receive, app_send):
        assert "state" not in scope
        assert app_receive is receive
        assert app_send is send

    middleware = RequestDisconnectMiddleware(app)
    await middleware({"type": "lifespan"}, receive, send)
    await middleware({"type": "websocket"}, receive, send)


@pytest.mark.anyio
async def test_monitor_handoff_does_not_read_past_an_observed_disconnect():
    receiving = anyio.Event()
    release_receive = anyio.Event()
    delivered = anyio.Event()
    receive_calls = 0

    async def receive():
        nonlocal receive_calls
        receive_calls += 1
        receiving.set()
        await release_receive.wait()
        return {"type": "http.disconnect"}

    async def app(scope, receive, send):
        state = scope["state"][REQUEST_DISCONNECT_STATE_KEY]

        async def read_disconnect():
            assert (await receive())["type"] == "http.disconnect"
            delivered.set()

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(read_disconnect)
            await receiving.wait()
            state.start_monitoring()
            # Let the watcher block behind the application's pending read.
            await anyio.sleep(0)
            release_receive.set()
            await delivered.wait()
            assert state.disconnected.is_set()
            await anyio.sleep(0)

    async def send(message):
        pass

    with anyio.fail_after(2):
        await RequestDisconnectMiddleware(app)({"type": "http"}, receive, send)
    assert receive_calls == 1
