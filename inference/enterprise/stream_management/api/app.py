import os
from functools import wraps
from typing import Any, Awaitable, Callable

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference.core import logger
from inference.enterprise.stream_management.api.entities import (
    CommandResponse,
    InferencePipelineStatusResponse,
    ListPipelinesResponse,
    PipelineInitialisationRequest,
)
from inference.enterprise.stream_management.api.errors import (
    ConnectivityError,
    ProcessesManagerAuthorisationError,
    ProcessesManagerClientError,
    ProcessesManagerInvalidPayload,
    ProcessesManagerNotFoundError,
)
from inference.enterprise.stream_management.api.stream_manager_client import (
    StreamManagerClient,
)
from inference.enterprise.stream_management.manager.entities import (
    STATUS_KEY,
    OperationStatus,
)

API_HOST = os.getenv("STREAM_MANAGEMENT_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("STREAM_MANAGEMENT_API_PORT", "8080"))

OPERATIONS_TIMEOUT = os.getenv("STREAM_MANAGER_OPERATIONS_TIMEOUT")
if OPERATIONS_TIMEOUT is not None:
    OPERATIONS_TIMEOUT = float(OPERATIONS_TIMEOUT)

STREAM_MANAGER_CLIENT = StreamManagerClient.init(
    host=os.getenv("STREAM_MANAGER_HOST", "127.0.0.1"),
    port=int(os.getenv("STREAM_MANAGER_PORT", "7070")),
    operations_timeout=OPERATIONS_TIMEOUT,
)

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
        except ProcessesManagerInvalidPayload as error:
            resp = JSONResponse(
                status_code=400,
                content={STATUS_KEY: OperationStatus.FAILURE, "message": str(error)},
            )
            logger.exception("Processes Manager - invalid payload error")
            return resp
        except ProcessesManagerAuthorisationError as error:
            resp = JSONResponse(
                status_code=401,
                content={STATUS_KEY: OperationStatus.FAILURE, "message": str(error)},
            )
            logger.exception("Processes Manager - authorisation error")
            return resp
        except ProcessesManagerNotFoundError as error:
            resp = JSONResponse(
                status_code=404,
                content={STATUS_KEY: OperationStatus.FAILURE, "message": str(error)},
            )
            logger.exception("Processes Manager - not found error")
            return resp
        except ConnectivityError as error:
            resp = JSONResponse(
                status_code=503,
                content={STATUS_KEY: OperationStatus.FAILURE, "message": str(error)},
            )
            logger.exception("Processes Manager connectivity error occurred")
            return resp
        except ProcessesManagerClientError as error:
            resp = JSONResponse(
                status_code=500,
                content={STATUS_KEY: OperationStatus.FAILURE, "message": str(error)},
            )
            logger.exception("Processes Manager error occurred")
            return resp
        except Exception:
            resp = JSONResponse(
                status_code=500,
                content={
                    STATUS_KEY: OperationStatus.FAILURE,
                    "message": "Internal error.",
                },
            )
            logger.exception("Internal error in API")
            return resp

    return wrapped_route


@app.get(
    "/list_pipelines",
    response_model=ListPipelinesResponse,
    summary="List active pipelines",
    description="Listing all active pipelines in the state of ProcessesManager being queried.",
)
@with_route_exceptions
async def list_pipelines(_: Request) -> ListPipelinesResponse:
    return await STREAM_MANAGER_CLIENT.list_pipelines()


@app.get(
    "/status/{pipeline_id}",
    response_model=InferencePipelineStatusResponse,
    summary="Get status of pipeline",
    description="Returns detailed statis of Inference Pipeline in the state of ProcessesManager being queried.",
)
@with_route_exceptions
async def get_status(pipeline_id: str) -> InferencePipelineStatusResponse:
    return await STREAM_MANAGER_CLIENT.get_status(pipeline_id=pipeline_id)


@app.post(
    "/initialise",
    response_model=CommandResponse,
    summary="Initialise the pipeline",
    description="Starts new Inference Pipeline within the state of ProcessesManager being queried.",
)
@with_route_exceptions
async def initialise(request: PipelineInitialisationRequest) -> CommandResponse:
    return await STREAM_MANAGER_CLIENT.initialise_pipeline(
        initialisation_request=request
    )


@app.post(
    "/pause/{pipeline_id}",
    response_model=CommandResponse,
    summary="Pauses the pipeline processing",
    description="Mutes the VideoSource of Inference Pipeline within the state of ProcessesManager being queried.",
)
@with_route_exceptions
async def pause(pipeline_id: str) -> CommandResponse:
    return await STREAM_MANAGER_CLIENT.pause_pipeline(pipeline_id=pipeline_id)


@app.post(
    "/resume/{pipeline_id}",
    response_model=CommandResponse,
    summary="Resumes the pipeline processing",
    description="Resumes the VideoSource of Inference Pipeline within the state of ProcessesManager being queried.",
)
@with_route_exceptions
async def resume(pipeline_id: str) -> CommandResponse:
    return await STREAM_MANAGER_CLIENT.resume_pipeline(pipeline_id=pipeline_id)


@app.post(
    "/terminate/{pipeline_id}",
    response_model=CommandResponse,
    summary="Terminates the pipeline processing",
    description="Terminates the VideoSource of Inference Pipeline within the state of ProcessesManager being queried.",
)
@with_route_exceptions
async def terminate(pipeline_id: str) -> CommandResponse:
    return await STREAM_MANAGER_CLIENT.terminate_pipeline(pipeline_id=pipeline_id)


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
