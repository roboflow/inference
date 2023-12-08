from functools import wraps
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference.core import logger

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def with_route_exceptions(route: callable) -> Callable[[Any], Awaitable[JSONResponse]]:
    @wraps(route)
    async def wrapped_route(*args, **kwargs):
        try:
            return await route(*args, **kwargs)
        except Exception:
            resp = JSONResponse(status_code=500, content={"message": "Internal error."})
            logger.exception("Internal error in API")
            return resp

    return wrapped_route


@app.post("/list_pipelines")
async def list_pipelines(_: Request) -> JSONResponse:
    # TODO: maybe authentication should be added?
    pass
