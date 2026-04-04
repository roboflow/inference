"""Jupyter notebook server HTTP routes."""

from fastapi import APIRouter
from inference.core.env import NOTEBOOK_ENABLED, NOTEBOOK_PASSWORD, NOTEBOOK_PORT
from inference.core.interfaces.http.error_handlers import with_route_exceptions
from inference.core.utils.notebooks import start_notebook
from time import sleep
from starlette.responses import RedirectResponse
from inference.core import logger


def create_notebook_router() -> APIRouter:
    router = APIRouter()
    @router.get(
        "/notebook/start",
        summary="Jupyter Lab Server Start",
        description="Starts a jupyter lab server for running development code",
    )
    @with_route_exceptions
    def notebook_start(browserless: bool = False):
        """Starts a jupyter lab server for running development code.

        Args:
            inference_request (NotebookStartRequest): The request containing the necessary details for starting a jupyter lab server.
            background_tasks: (BackgroundTasks) pool of fastapi background tasks

        Returns:
            NotebookStartResponse: The response containing the URL of the jupyter lab server.
        """
        logger.debug(f"Reached /notebook/start")
        if NOTEBOOK_ENABLED:
            start_notebook()
            if browserless:
                return {
                    "success": True,
                    "message": f"Jupyter Lab server started at http://localhost:{NOTEBOOK_PORT}?token={NOTEBOOK_PASSWORD}",
                }
            else:
                sleep(2)
                return RedirectResponse(
                    f"http://localhost:{NOTEBOOK_PORT}/lab/tree/quickstart.ipynb?token={NOTEBOOK_PASSWORD}"
                )
        else:
            if browserless:
                return {
                    "success": False,
                    "message": "Notebook server is not enabled. Enable notebooks via the NOTEBOOK_ENABLED environment variable.",
                }
            else:
                return RedirectResponse(f"/notebook-instructions.html")

    return router
