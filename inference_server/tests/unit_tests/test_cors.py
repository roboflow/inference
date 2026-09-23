from fastapi import FastAPI
from starlette.testclient import TestClient

from inference_server.cors import PathAwareCORSMiddleware

ALLOWED_ORIGIN = "https://app.example"


def _build_app() -> FastAPI:
    app = FastAPI()

    @app.get("/build/api/x")
    async def _build_api():
        return {"ok": True}

    @app.get("/workflows/run")
    async def _workflows_run():
        return {"ok": True}

    @app.get("/other")
    async def _other():
        return {"ok": True}

    app.add_middleware(
        PathAwareCORSMiddleware,
        match_paths=r"^/(build/api|workflows/).*",
        allow_origins=[ALLOWED_ORIGIN],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
        allow_private_network=True,
    )
    return app


def test_private_network_preflight_on_matched_path():
    client = TestClient(_build_app())

    response = client.options(
        "/build/api/x",
        headers={
            "Origin": ALLOWED_ORIGIN,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Private-Network": "true",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGIN
    assert response.headers["access-control-allow-private-network"] == "true"


def test_preflight_on_unmatched_path_gets_no_cors_headers():
    client = TestClient(_build_app())

    response = client.options(
        "/other",
        headers={
            "Origin": ALLOWED_ORIGIN,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Private-Network": "true",
        },
    )

    assert response.status_code in (404, 405)
    assert "access-control-allow-origin" not in response.headers
    assert "access-control-allow-private-network" not in response.headers


def test_simple_request_on_matched_path_carries_cors_headers():
    client = TestClient(_build_app())

    response = client.get("/workflows/run", headers={"Origin": ALLOWED_ORIGIN})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGIN


def test_disallowed_origin_private_network_preflight_is_rejected():
    client = TestClient(_build_app())

    response = client.options(
        "/build/api/x",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Private-Network": "true",
        },
    )

    assert response.status_code == 400
    assert "access-control-allow-private-network" not in response.headers
