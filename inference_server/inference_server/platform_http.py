"""Outbound HTTP calls to the Roboflow platform shared by server modules."""

import json
import logging
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

from inference_models.utils.environment import get_float_from_env
from inference_models.weights_providers.roboflow import (
    roboflow_secure_gateway_proxy_url_builder,
)
from inference_server import configuration
from inference_server.hosted.assume_identity import add_assume_identity_headers
from inference_server.legacy.errors import LegacyHTTPError

logger = logging.getLogger(__name__)

API_REQUEST_TIMEOUT_S = get_float_from_env(
    "ROBOFLOW_API_REQUEST_TIMEOUT", default=120.0
)


def _add_params_to_url(url: str, params: List[Tuple[str, str]]) -> str:
    if not params:
        return url

    query = "&".join(
        f"{name}={urllib.parse.quote_plus(value)}" for name, value in params
    )
    return f"{url}?{query}"


def build_api_headers(
    explicit_headers: Optional[Dict[str, Union[str, List[str]]]] = None,
) -> Dict[str, Union[str, List[str]]]:
    """Build the headers every platform request carries.

    Args:
        explicit_headers: Headers that override the defaults and
            ``ROBOFLOW_API_EXTRA_HEADERS``.

    Returns:
        Header mapping with the server version and chunked-response markers.
    """
    headers: Dict[str, Union[str, List[str]]] = {
        "x-roboflow-inference-version": configuration.SERVER_VERSION,
        "x-allow-chunked-response": "true",
    }
    if configuration.ROBOFLOW_API_EXTRA_HEADERS:
        try:
            headers.update(json.loads(configuration.ROBOFLOW_API_EXTRA_HEADERS))
        except ValueError:
            logger.warning("Could not decode ROBOFLOW_API_EXTRA_HEADERS")
    headers.update(explicit_headers or {})

    return headers


def wrap_url(url: str) -> str:
    """Route a platform URL through ``SECURE_GATEWAY`` when one is configured.

    Args:
        url: Fully built platform URL.

    Returns:
        The URL to request, proxied when a secure gateway is set.
    """
    wrapped_url = roboflow_secure_gateway_proxy_url_builder(url, None)

    return wrapped_url


def tls_verification_options() -> Dict[str, bool]:
    if configuration.ROBOFLOW_API_VERIFY_SSL:
        return {}

    return {"verify": False}


def _platform_request(
    method: str, url: str, *, assume_identity: bool = False, **kwargs: Any
) -> requests.Response:
    """Send one request to the Roboflow platform, mapping transport failures.

    ``assume_identity`` attaches the assume-identity headers of the current
    request. Legacy sends them at two outbound sites only, the model stat lookup
    and the workflows model-metadata registry lookup, never on the usage check
    or on workflow-definition fetches, so it defaults to False.

    Args:
        method: ``requests`` function name, e.g. ``"get"``.
        url: Fully built and gateway-wrapped URL.
        assume_identity: Whether to add the assume-identity headers.
        **kwargs: Passed through to ``requests``; a caller-supplied ``verify``
            wins over ``ROBOFLOW_API_VERIFY_SSL``.

    Returns:
        The platform response.

    Raises:
        LegacyHTTPError: 504 on timeout, 503 on any other transport failure.
    """
    headers = dict(kwargs.pop("headers", None) or {})
    kwargs = {**tls_verification_options(), **kwargs}
    if assume_identity:
        add_assume_identity_headers(headers)
    try:
        return getattr(requests, method)(url=url, headers=headers, **kwargs)
    except requests.exceptions.Timeout as error:
        raise LegacyHTTPError(
            504, "Timeout when attempting to connect to Roboflow API."
        ) from error
    except requests.exceptions.RequestException as error:
        raise LegacyHTTPError(
            503, "Internal error. Could not connect to Roboflow API."
        ) from error
