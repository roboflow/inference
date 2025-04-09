from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from inference.core.interfaces.http.middlewares.cors import PathAwareCORSMiddleware


def homepage(request):
    return PlainTextResponse("Hello, world!")


def create_app(match_paths=None, allow_origins=None):
    """
    Utility to create a Starlette test app with our custom PathAwareCORSMiddleware.
    """
    routes = [
        Route("/", endpoint=homepage),
        Route("/foo", endpoint=homepage),
        Route("/bar/baz", endpoint=homepage),
    ]

    app = Starlette(routes=routes)

    app.add_middleware(
        PathAwareCORSMiddleware,
        match_paths=match_paths,
        allow_origins=allow_origins or ["http://example.com"],  # Example
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
        allow_credentials=True,
    )
    return app


def test_no_match_paths_applies_cors_everywhere():
    """
    With no match_paths specified, the middleware should apply CORS to all HTTP requests.
    """
    app = create_app(match_paths=None)
    client = TestClient(app)

    # Example 1: Checking GET on "/"
    response = client.get("/", headers={"Origin": "http://example.com"})
    assert response.status_code == 200
    # Middleware should set CORS header
    assert response.headers["access-control-allow-origin"] == "http://example.com"

    # Example 2: Checking GET on "/foo"
    response = client.get("/foo", headers={"Origin": "http://example.com"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://example.com"


def test_regex_does_not_match_path():
    """
    With match_paths provided, if the requested path doesn't match, CORS is NOT applied.
    """
    # Regex only matches /foo and /foo/anything...
    app = create_app(match_paths=r"^/foo.*")
    client = TestClient(app)

    # /bar/baz does not match the regex
    response = client.get("/bar/baz", headers={"Origin": "http://example.com"})
    assert response.status_code == 200
    # Ensure no CORS headers exist
    assert "access-control-allow-origin" not in response.headers


def test_regex_matches_path():
    """
    With match_paths provided, if the requested path DOES match, CORS is applied.
    """
    # Regex only matches /foo
    app = create_app(match_paths=r"^/foo$")
    client = TestClient(app)

    # /foo does match the regex
    response = client.get("/foo", headers={"Origin": "http://example.com"})
    assert response.status_code == 200
    # Should include CORS header
    assert response.headers["access-control-allow-origin"] == "http://example.com"


def test_cors_preflight_with_matching_path():
    """
    Test an OPTIONS preflight request on a path that DOES match the regex.
    """
    app = create_app(match_paths=r"^/foo$")
    client = TestClient(app)

    response = client.options(
        "/foo",
        headers={
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert response.status_code == 200
    # Check typical CORS preflight response headers:
    # Access-Control-Allow-Origin, Access-Control-Allow-Methods, etc.
    assert response.headers["access-control-allow-origin"] == "http://example.com"
    assert "access-control-allow-methods" in response.headers
    assert "access-control-allow-headers" in response.headers
    # Because we allowed credentials, the response should have that header too
    assert response.headers.get("access-control-allow-credentials") == "true"
