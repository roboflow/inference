from fastapi import APIRouter, Depends
from typing import Any, Optional
from starlette.responses import JSONResponse

from inference.core.env import DOCKER_SOCKET_PATH
from inference.core.managers.metrics import get_container_stats
from inference.core.utils.container import is_docker_socket_mounted


def create_health_router(model_init_state: Optional[Any] = None) -> APIRouter:
    router = APIRouter()

    @router.get("/device/stats", summary="Device/container statistics")
    def device_stats():
        not_configured_error_message = {
            "error": "Device statistics endpoint is not enabled.",
            "hint": (
                "Mount the Docker socket and point its location when running the docker "
                "container to collect device stats "
                "(i.e. `docker run ... -v /var/run/docker.sock:/var/run/docker.sock "
                "-e DOCKER_SOCKET_PATH=/var/run/docker.sock ...`)."
            ),
        }
        if not DOCKER_SOCKET_PATH:
            return JSONResponse(
                status_code=404,
                content=not_configured_error_message,
            )
        if not is_docker_socket_mounted(docker_socket_path=DOCKER_SOCKET_PATH):
            return JSONResponse(
                status_code=500,
                content=not_configured_error_message,
            )

        container_stats = get_container_stats(docker_socket_path=DOCKER_SOCKET_PATH)
        return JSONResponse(status_code=200, content=container_stats)
    
    @router.get("/readiness", status_code=200)
    def readiness(state: Any = Depends(lambda: model_init_state)):
        """Readiness endpoint for Kubernetes readiness probe."""
        if state is None:
            return {"status": "ready"}
        with state.lock:
            if state.is_ready:
                return {"status": "ready"}
            return JSONResponse(
                content={"status": "not ready"}, status_code=503
            )

    @router.get("/healthz", status_code=200)
    def healthz():
        """Health endpoint for Kubernetes liveness probe."""
        return {"status": "healthy"}

    return router