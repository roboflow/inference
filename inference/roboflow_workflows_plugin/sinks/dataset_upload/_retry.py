from typing import Callable, ParamSpec, TypeVar

import backoff
from requests.exceptions import HTTPError

from inference.core.exceptions import (
    RoboflowAPIConnectionError,
    RoboflowAPITimeoutError,
    RoboflowAPIUnsuccessfulRequestError,
)

_P = ParamSpec("_P")
_T = TypeVar("_T")
_TRANSIENT_STATUS_CODES = {500, 502, 503, 504}


def _give_up(error: Exception) -> bool:
    if isinstance(error, (RoboflowAPIConnectionError, RoboflowAPITimeoutError)):
        return False

    cause = error.__cause__
    retryable = (
        isinstance(cause, HTTPError)
        and cause.response is not None
        and cause.response.status_code in _TRANSIENT_STATUS_CODES
    )

    return not retryable


@backoff.on_exception(
    backoff.expo,
    exception=(
        RoboflowAPIConnectionError,
        RoboflowAPITimeoutError,
        RoboflowAPIUnsuccessfulRequestError,
    ),
    max_tries=3,
    giveup=_give_up,
    logger=None,
)
def _retry_api_request(
    operation: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs
) -> _T:
    """Retry one upload stage without replaying earlier successful stages."""
    response = operation(*args, **kwargs)

    return response
