from __future__ import annotations

import functools
import logging
from typing import Callable

from fastapi.responses import JSONResponse
from roboflow_workflows.http_contract.errors import workflow_error_payload
from starlette.exceptions import HTTPException

from inference_server.legacy.errors import legacy_error_response

logger = logging.getLogger(__name__)


def with_workflow_errors(fn: Callable) -> Callable:
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except HTTPException:
            raise
        except Exception as error:
            payload = workflow_error_payload(error)
            if payload is not None:
                status_code, content = payload
                logger.error("Workflow route error", exc_info=error)
                return JSONResponse(status_code=status_code, content=content)
            return legacy_error_response(error)

    return wrapper
