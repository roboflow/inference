from __future__ import annotations

import re
import typing

from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware as StarletteCORSMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send


class PathAwareCORSMiddleware(StarletteCORSMiddleware):
    """
    Extends Starlette's CORSMiddleware to allow specifying a regex of paths that
    this middleware should apply to.
    If 'match_paths' is given, only requests matching that regex will have CORS
    headers applied.
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

        # If we got here, apply the normal Starlette CORSMiddleware behavior
        await super().__call__(scope, receive, send)
