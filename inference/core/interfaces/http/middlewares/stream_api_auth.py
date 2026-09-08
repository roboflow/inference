"""Authenticate the device-wide pipeline management API with a local admin token."""

import hmac

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class StreamAPIAuthMiddleware:
    def __init__(self, app: ASGIApp, token: str):
        self.app = app
        self.token = token.encode("utf-8")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        root_path = scope.get("root_path", "").rstrip("/")
        if root_path and path.startswith(root_path + "/"):
            path = path[len(root_path) :]
        if scope["type"] == "http" and (
            path == "/inference_pipelines" or path.startswith("/inference_pipelines/")
        ):
            tokens = [
                value
                for name, value in scope.get("headers", [])
                if name.lower() == b"x-stream-api-key"
            ]
            if (
                not self.token
                or len(tokens) != 1
                or not hmac.compare_digest(tokens[0], self.token)
            ):
                await JSONResponse(
                    status_code=401,
                    content={"message": "Unauthorized stream management request"},
                )(scope, receive, send)
                return
        await self.app(scope, receive, send)
