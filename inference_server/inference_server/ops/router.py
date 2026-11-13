import asyncio
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse, Response

from inference_server import configuration
from inference_server.ops.docker_stats import (
    CONTAINER_STATS_ERROR_MESSAGE,
    ContainerStatsError,
    get_container_stats,
    is_docker_socket_mounted,
)
from inference_server.ops.memory_logs import DEFAULT_LOG_LIMIT, get_recent_logs
from inference_server.ops.notebook import (
    NOTEBOOK_START_ERROR_MESSAGE,
    NotebookStartError,
    notebook_token,
    start_notebook,
)
from inference_server.ops.secure_gateway import (
    SecureGatewayHealthResponse,
    get_secure_gateway_base_url,
    probe_secure_gateway_health,
)

NOTEBOOK_REDIRECT_DELAY_S = 2.0
DEVICE_STATS_NOT_CONFIGURED = {
    "error": "Device statistics endpoint is not enabled.",
    "hint": "Mount the Docker socket and point its location when running the docker "
    "container to collect device stats "
    "(i.e. `docker run ... -v /var/run/docker.sock:/var/run/docker.sock "
    "-e DOCKER_SOCKET_PATH=/var/run/docker.sock ...`).",
}

logs_router = APIRouter(tags=["legacy"])
local_router = APIRouter(tags=["legacy"])
secure_gateway_router = APIRouter(tags=["legacy"])


def include_ops_routers(app: FastAPI) -> None:
    app.include_router(logs_router)
    if configuration.LAMBDA or configuration.GCP_SERVERLESS:
        return

    app.include_router(local_router)
    if configuration.SECURE_GATEWAY_HEALTH_ENDPOINT_ENABLED:
        app.include_router(secure_gateway_router)


@logs_router.get(
    "/logs",
    summary="Get Recent Logs",
    description="Get recent application logs for debugging",
)
async def get_logs(
    limit: Optional[int] = Query(
        DEFAULT_LOG_LIMIT, description="Maximum number of log entries to return"
    ),
    level: Optional[str] = Query(
        None,
        description="Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    ),
    since: Optional[str] = Query(
        None, description="Return logs since this ISO timestamp"
    ),
) -> JSONResponse:
    if not configuration.ENABLE_IN_MEMORY_LOGS:
        raise HTTPException(status_code=404, detail="Logs endpoint not available")

    logs = get_recent_logs(limit or DEFAULT_LOG_LIMIT, level=level, since=since)

    return JSONResponse(content={"logs": logs, "total_count": len(logs)})


@local_router.get("/device/stats")
async def device_stats() -> JSONResponse:
    docker_socket_path = configuration.DOCKER_SOCKET_PATH
    if not docker_socket_path:
        return JSONResponse(status_code=404, content=DEVICE_STATS_NOT_CONFIGURED)
    if not is_docker_socket_mounted(docker_socket_path):
        return JSONResponse(status_code=500, content=DEVICE_STATS_NOT_CONFIGURED)

    try:
        container_stats = await asyncio.to_thread(
            get_container_stats, docker_socket_path
        )
    except ContainerStatsError:
        return JSONResponse(
            status_code=500, content={"error": CONTAINER_STATS_ERROR_MESSAGE}
        )

    return JSONResponse(status_code=200, content=container_stats)


@secure_gateway_router.get(
    "/secure-gateway/health",
    response_model=SecureGatewayHealthResponse,
    responses={
        404: {
            "model": SecureGatewayHealthResponse,
            "description": "SECURE_GATEWAY is not configured on this server.",
        },
        502: {
            "model": SecureGatewayHealthResponse,
            "description": "Gateway answered, but not with 2xx (includes redirects).",
        },
        503: {
            "model": SecureGatewayHealthResponse,
            "description": "Gateway unreachable or TLS handshake failed.",
        },
        504: {
            "model": SecureGatewayHealthResponse,
            "description": "Gateway did not answer within SECURE_GATEWAY_HEALTH_CHECK_TIMEOUT.",
        },
    },
    summary="Secure gateway health",
    description="Probe the /health route of the configured SECURE_GATEWAY "
    "(legacy LICENSE_SERVER) and report whether the proxy is reachable "
    "from this server. Opt-in via SECURE_GATEWAY_HEALTH_ENDPOINT_ENABLED.",
)
async def secure_gateway_health() -> JSONResponse:
    status_code, payload = await asyncio.to_thread(
        probe_secure_gateway_health,
        get_secure_gateway_base_url(),
        timeout=configuration.SECURE_GATEWAY_HEALTH_CHECK_TIMEOUT,
        verify_ssl=configuration.ROBOFLOW_API_VERIFY_SSL,
    )

    return JSONResponse(status_code=status_code, content=payload.model_dump())


@local_router.get(
    "/notebook/start",
    summary="Jupyter Lab Server Start",
    description="Starts a jupyter lab server for running development code",
)
async def notebook_start(browserless: bool = False) -> Response:
    if not configuration.NOTEBOOK_ENABLED:
        if browserless:
            return JSONResponse(
                content={
                    "success": False,
                    "message": "Notebook server is not enabled. Enable notebooks via the NOTEBOOK_ENABLED environment variable.",
                }
            )
        return RedirectResponse("/notebook-instructions.html")

    try:
        await asyncio.to_thread(start_notebook)
    except NotebookStartError:
        return JSONResponse(
            status_code=500, content={"message": NOTEBOOK_START_ERROR_MESSAGE}
        )

    port = configuration.NOTEBOOK_PORT
    token = quote(notebook_token(), safe="")
    if browserless:
        return JSONResponse(
            content={
                "success": True,
                "message": f"Jupyter Lab server started at http://localhost:{port}?token={token}",
            }
        )

    await asyncio.sleep(NOTEBOOK_REDIRECT_DELAY_S)

    return RedirectResponse(
        f"http://localhost:{port}/lab/tree/quickstart.ipynb?token={token}"
    )
