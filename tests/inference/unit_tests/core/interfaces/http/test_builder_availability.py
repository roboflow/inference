"""Regression tests for https://github.com/roboflow/inference/issues/2993.

The landing page (inference/landing/src/app/page.tsx) advertises a link to
the Workflows builder at /build. The builder_router above is only mounted by
inference/core/interfaces/http/http_api.py when ENABLE_BUILDER is truthy
(default: False), so a manually-run server otherwise serves a link that
404s.

These tests pin down two things at the router/config level, independent of
the frontend:

1. Feature-off vs feature-on is a real, observable HTTP difference: with the
   builder router unmounted, /build simply does not exist (404). This is
   the same "not included" branch http_api.py takes when ENABLE_BUILDER is
   falsy -- we exercise it directly instead of importing the whole
   http_api module (which pulls in the full model-serving stack).
2. GET is the only safe availability probe for /build. The builder module
   registers a GET-only handler for the builder root path, with no HEAD
   support. Verified against a real running server (see
   roboflow/evidence/inference-builder/): even when the builder IS enabled,
   a HEAD request to /build does not return 200 (it falls through to the
   landing page's static-export catch-all in the full app, or gets rejected
   by routing here). A landing page probe that reused the dashboard's HEAD
   check verbatim would therefore hide a working builder link. This is the
   regression the fix in page.tsx (GET instead of HEAD) guards against.
"""

from fastapi import FastAPI
from starlette.status import HTTP_200_OK
from starlette.testclient import TestClient

# Reuse the existing builder_app fixture (a FastAPI app with only the builder
# router mounted, editor.html templated out) instead of duplicating it.
from .test_builder import builder_app  # noqa: F401


def test_builder_route_absent_when_router_not_mounted():
    """Mirrors the `if ENABLE_BUILDER: app.include_router(...)` branch in
    http_api.py taking the False path: no builder routes are registered at
    all, so /build is simply unknown to the app and both GET and HEAD 404.
    """
    app = FastAPI()

    client = TestClient(app)
    get_response = client.get("/build")
    head_response = client.head("/build")

    assert get_response.status_code == 404
    assert head_response.status_code == 404


def test_get_probes_builder_availability_when_mounted(builder_app):
    """GET /build is the request the builder router actually implements, and
    is what the landing page now uses to decide whether to show the link."""
    client = TestClient(builder_app)

    response = client.get("/build")

    assert response.status_code == HTTP_200_OK
    assert "text/html" in response.headers["content-type"]


def test_head_does_not_reliably_report_builder_availability(builder_app):
    """Regression guard for the HEAD-probe trap: even though GET /build
    succeeds on this exact mounted router, HEAD /build must NOT be treated
    as an availability signal, because the builder only implements GET.

    If this assertion ever starts failing because HEAD now returns 200, the
    frontend's GET-based probe in page.tsx is no longer the only safe
    option and this comment (and the matching comment in page.tsx) should
    be revisited -- but until then, a HEAD-based probe would silently hide
    a working builder link, which is the exact bug this change fixes.
    """
    client = TestClient(builder_app)

    response = client.head("/build")

    assert response.status_code != HTTP_200_OK
