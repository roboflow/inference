"""Theme 7: auth transport failures, middleware path/scope handling,
shared bearer extraction."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

import inference_server.auth as auth_mod
from inference_server.app import _AuthMiddleware
from inference_server.auth import extract_bearer, validate_api_key
from inference_server.errors import AuthBackendUnavailable


class TestExtractBearer:
    def test_canonical(self):
        assert extract_bearer("Bearer abc123") == "abc123"

    def test_case_insensitive_scheme(self):
        assert extract_bearer("bearer abc123") == "abc123"
        assert extract_bearer("BEARER abc123") == "abc123"

    def test_other_scheme_rejected(self):
        assert extract_bearer("Token abc123") == ""
        assert extract_bearer("") == ""


class TestAuthBackendFailure:
    def test_network_error_raises_and_caches_nothing(self, monkeypatch):
        monkeypatch.setattr(auth_mod, "_cache", {})

        class _BoomSession:
            def get(self, *a, **k):
                raise OSError("connection refused")

        monkeypatch.setattr(auth_mod, "_get_session", lambda: _BoomSession())
        with pytest.raises(AuthBackendUnavailable):
            asyncio.run(validate_api_key("some-key"))
        assert auth_mod._cache == {}  # no negative-cache poisoning


def _run_middleware(scope, validate=None):
    """Drive _AuthMiddleware with a fake downstream app; returns sent messages
    and whether downstream ran."""
    sent: list[dict] = []
    state = {"downstream": False}

    async def downstream(scope, receive, send):
        state["downstream"] = True

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    mw = _AuthMiddleware(downstream)
    ctx = (
        patch("inference_server.app.validate_api_key", new=validate)
        if validate is not None
        else patch(
            "inference_server.app.validate_api_key",
            new=lambda token: _never_called(),
        )
    )
    with ctx:
        asyncio.run(mw(scope, receive, send))
    return sent, state["downstream"]


def _never_called():
    raise AssertionError("validate_api_key should not be called")


def _http_scope(path, headers=()):
    return {"type": "http", "path": path, "headers": list(headers)}


class TestAuthMiddleware:
    def test_root_path_skips_auth(self):
        sent, downstream = _run_middleware(_http_scope("/"))
        assert downstream is True

    def test_websocket_scope_rejected(self):
        sent, downstream = _run_middleware({"type": "websocket", "path": "/ws"})
        assert downstream is False
        assert sent[0]["type"] == "websocket.close"

    def test_auth_backend_outage_returns_503_not_403(self):
        async def _unavailable(token):
            raise AuthBackendUnavailable("api down")

        sent, downstream = _run_middleware(
            _http_scope("/v2/models", [(b"authorization", b"Bearer k")]),
            validate=_unavailable,
        )
        assert downstream is False
        start = [m for m in sent if m["type"] == "http.response.start"][0]
        assert start["status"] == 503

    def test_lowercase_bearer_accepted(self):
        async def _ok(token):
            assert token == "k"
            return True, "ws"

        sent, downstream = _run_middleware(
            _http_scope("/v2/models", [(b"authorization", b"bearer k")]),
            validate=_ok,
        )
        assert downstream is True
