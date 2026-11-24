"""Observe connection closure outside the BaseHTTPMiddleware receive wrappers."""

from threading import Event

import anyio
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_DISCONNECT_STATE_KEY = "_inference_request_disconnect"


class _RequestDisconnectState:
    def __init__(self, receive: Receive):
        self.disconnected = Event()
        self._receive = receive
        self._receive_lock = anyio.Lock()
        self._monitor_requested = anyio.Event()
        self._disconnect_observed = anyio.Event()

    def start_monitoring(self) -> None:
        """Start observing after the route has parsed or elected to ignore its body."""
        self._monitor_requested.set()

    def _observe(self, message: Message) -> None:
        if message["type"] == "http.disconnect":
            self.disconnected.set()
            self._disconnect_observed.set()

    async def receive(self) -> Message:
        """Keep one reader of the raw receive channel, including during handoff."""
        async with self._receive_lock:
            if not self._monitor_requested.is_set():
                message = await self._receive()
                self._observe(message)
                return message

        # Once processing starts, the body is no longer needed by the route.
        # Middleware disconnect listeners share the signal rather than racing
        # the watcher for raw ASGI events.
        await self._disconnect_observed.wait()
        return {"type": "http.disconnect"}

    async def watch_disconnect(self) -> None:
        """Drain any ignored body and keep receiving while model work runs."""
        await self._monitor_requested.wait()
        while True:
            async with self._receive_lock:
                if self.disconnected.is_set():
                    return

                message = await self._receive()
                self._observe(message)


class RequestDisconnectMiddleware:
    """Provide request-scoped connection monitoring for cooperative processing.

    Monitoring is opt-in after the endpoint parses or ignores its request body.
    Normal receive calls and the watcher serialize access to the raw channel;
    no request-body queue or extra body buffering is introduced.
    """

    def __init__(self, app: ASGIApp):
        """Wrap the application outside its HTTP middleware.

        Args:
            app (ASGIApp): Application receiving the request-scoped state.
        """
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Run the application and close its watcher on every exit path.

        Args:
            scope (Scope): ASGI connection scope.
            receive (Receive): Server's incoming event channel.
            send (Send): Server's outgoing event channel.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        state = _RequestDisconnectState(receive)
        scope.setdefault("state", {})[REQUEST_DISCONNECT_STATE_KEY] = state
        app_error = None
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(state.watch_disconnect)
            try:
                await self.app(scope, state.receive, send)
            except BaseException as error:
                # Preserve the application's exception rather than changing
                # it into a task-group ExceptionGroup for outer error handlers.
                app_error = error
            finally:
                tasks.cancel_scope.cancel()

        if app_error is not None:
            raise app_error
