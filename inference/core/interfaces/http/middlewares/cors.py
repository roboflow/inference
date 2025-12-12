from __future__ import annotations

import re
import typing

from starlette.datastructures import Headers, MutableHeaders
from starlette.middleware.cors import CORSMiddleware as StarletteCORSMiddleware
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class PathAwareCORSMiddleware(StarletteCORSMiddleware):
    """
    Extends Starlette's CORSMiddleware to allow specifying a regex of paths that
    this middleware should apply to.
    If 'match_paths' is given, only requests matching that regex will have CORS
    headers applied.

    Also supports Private Network Access (PNA) for local development, allowing
    requests from public websites to localhost.
    """

    def __init__(
        self,
        app: ASGIApp,
        match_paths: str | None = None,
        allow_origins: typing.Sequence[str] = (),
        allow_methods: typing.Sequence[str] = ("GET",),
        allow_headers: typing.Sequence[str] = (),
        allow_credentials: bool = False,
        allow_origin_regex: str | None = None,
        expose_headers: typing.Sequence[str] = (),
        max_age: int = 600,
        allow_private_network: bool = False,
    ) -> None:
        super().__init__(
            app=app,
            allow_origins=allow_origins,
            allow_methods=allow_methods,
            allow_headers=allow_headers,
            allow_credentials=allow_credentials,
            allow_origin_regex=allow_origin_regex,
            expose_headers=expose_headers,
            max_age=max_age,
        )
        self.match_paths_regex = re.compile(match_paths) if match_paths else None
        self.allow_private_network = allow_private_network
        # Store these for PNA preflight handling (not exposed by parent class)
        self._max_age = max_age
        self._allow_methods = allow_methods
        self._allow_headers = allow_headers
        self._allow_credentials = allow_credentials

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """
        Only apply the CORS logic if the path matches self.match_paths_regex
        (when provided). Otherwise, just call the wrapped 'app'.
        """
        # If it's not an HTTP request, skip the CORS processing:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # If match_paths was supplied, check if the current path matches
        if self.match_paths_regex is not None:
            path = scope.get("path", "")
            if not self.match_paths_regex.match(path):
                # If it does NOT match, just run the app without CORS
                await self.app(scope, receive, send)
                return

        # Handle Private Network Access preflight requests
        if self.allow_private_network:
            headers = Headers(scope=scope)
            if (
                scope["method"] == "OPTIONS"
                and "access-control-request-private-network" in headers
            ):
                await self._handle_pna_preflight(scope, receive, send, headers)
                return

        # If we got here, apply the normal Starlette CORSMiddleware behavior
        await super().__call__(scope, receive, send)

    async def _handle_pna_preflight(
        self, scope: Scope, receive: Receive, send: Send, request_headers: Headers
    ) -> None:
        """
        Handle preflight requests that include Private Network Access header.
        """
        origin = request_headers.get("origin", "")
        if self.is_allowed_origin(origin=origin):
            response_headers = {
                "access-control-allow-origin": origin,
                "access-control-allow-private-network": "true",
                "access-control-allow-methods": ", ".join(self._allow_methods),
                "access-control-max-age": str(self._max_age),
            }
            if self._allow_headers and "*" not in self._allow_headers:
                response_headers["access-control-allow-headers"] = ", ".join(
                    self._allow_headers
                )
            elif "*" in self._allow_headers:
                requested_headers = request_headers.get(
                    "access-control-request-headers", ""
                )
                if requested_headers:
                    response_headers["access-control-allow-headers"] = requested_headers
            if self._allow_credentials:
                response_headers["access-control-allow-credentials"] = "true"
            response = PlainTextResponse(
                "OK", status_code=200, headers=response_headers
            )
        else:
            response = PlainTextResponse(
                "Disallowed CORS origin", status_code=400, headers={"vary": "Origin"}
            )
        await response(scope, receive, send)
