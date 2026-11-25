import asyncio

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from inference.core.logger import logger

EXEMPT_PATHS = frozenset(
    {
        "/",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/info",
        "/healthz",
        "/readiness",
        "/metrics",
        "/logs",
    }
)


class ConcurrencyLimitMiddleware:
    """Caps concurrently executing HTTP requests, queueing the excess.

    Pure ASGI, so a waiting request leaves its body unread in the socket and the
    slot is held until the response is fully sent. Requests that cannot get a
    slot within the queue timeout receive 503.

    Args:
        app: Wrapped ASGI application.
        max_concurrent_requests: Number of requests allowed to execute at once.
        queue_timeout: Seconds a request waits for a slot; 0 rejects immediately.
    """

    def __init__(
        self,
        app: ASGIApp,
        max_concurrent_requests: int,
        queue_timeout: float,
    ) -> None:
        self.app = app
        self._max_concurrent_requests = max_concurrent_requests
        self._queue_timeout = queue_timeout
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] == "OPTIONS"
            or self._is_exempt(scope=scope)
        ):
            await self.app(scope, receive, send)
            return None

        if not await self._acquire_slot():
            logger.warning(
                "Rejecting request to %s: concurrent request limit %s reached",
                scope["path"],
                self._max_concurrent_requests,
            )
            response = JSONResponse(
                status_code=503,
                content={
                    "message": "Server is at its concurrent request limit - retry request."
                },
                headers={"Retry-After": "1"},
            )
            await response(scope, receive, send)
            return None

        try:
            await self.app(scope, receive, send)
        finally:
            self._semaphore.release()

    @staticmethod
    def _is_exempt(scope: Scope) -> bool:
        path = scope["path"]
        root_path = scope.get("root_path", "")
        if root_path and path.startswith(root_path):
            path = path[len(root_path) :]
        is_exempt = path in EXEMPT_PATHS

        return is_exempt

    async def _acquire_slot(self) -> bool:
        if self._queue_timeout <= 0:
            if self._semaphore.locked():
                return False

            await self._semaphore.acquire()
            return True

        acquisition = asyncio.ensure_future(self._semaphore.acquire())
        try:
            await asyncio.wait({acquisition}, timeout=self._queue_timeout)
        except BaseException:
            self._abandon(acquisition=acquisition)
            raise

        if acquisition.done():
            acquisition.result()
            return True

        self._abandon(acquisition=acquisition)

        return False

    def _abandon(self, acquisition: "asyncio.Future[bool]") -> None:
        acquisition.cancel()
        if acquisition.done() and not acquisition.cancelled():
            if acquisition.exception() is None:
                self._semaphore.release()
