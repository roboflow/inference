import inspect
from threading import Lock
from typing import Any, Callable, Dict, Iterable, Optional
from urllib.parse import urlparse

from inference.core.logger import logger

signatures = {}
lock = Lock()

TRUTHY_STRINGS = {"true", "1", "yes", "on"}
FALSY_STRINGS = {"false", "0", "no", "off"}


def coerce_optional_bool(value: Any) -> Optional[bool]:
    """Interpret a tri-state flag that may arrive as a bool or as a string.

    Query parameters reach handlers already coerced by FastAPI, but the same
    flags are also read straight off raw request params (authorization
    middleware) and off values that round-tripped through a payload, where they
    are still strings.

    Args:
        value: Raw flag value.

    Returns:
        The boolean the value denotes, or None when it denotes neither.
    """
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return None

    normalized_value = value.strip().lower()
    if normalized_value in TRUTHY_STRINGS:
        return True
    if normalized_value in FALSY_STRINGS:
        return False

    return None


def ssl_verify_for_endpoint(url: str) -> bool:
    """TLS verification is skipped only for local development endpoints.

    Judges the hostname of the URL that will actually be requested, so
    gateway-wrapped URLs (``SECURE_GATEWAY``) are evaluated by the gateway
    host - not by substrings of the percent-encoded embedded target.
    """
    try:
        hostname = urlparse(url).hostname or ""
    except ValueError:
        return True
    return hostname.lower() not in {"localhost", "127.0.0.1"}


def get_signature(func: Callable[[Any], Any]) -> inspect.Signature:
    with lock:
        if func not in signatures:
            signatures[func] = inspect.signature(func)
        return signatures[func]


def collect_func_params(
    func: Callable[[Any], Any], args: Iterable[Any], kwargs: Dict[Any, Any]
) -> Dict[str, Any]:
    signature = get_signature(func)
    parameters = signature.parameters

    # Initialize params with positional arguments
    params = {param: arg_value for param, arg_value in zip(parameters.keys(), args)}

    # Update params with keyword arguments
    params.update(kwargs)

    # Set default values for missing arguments
    defaults = {
        param: param_obj.default
        for param, param_obj in parameters.items()
        if param not in params and param_obj.default is not inspect.Parameter.empty
    }
    params.update(defaults)

    # Verify against function signature parameters
    signature_keys = set(parameters.keys())
    if params.keys() != signature_keys:
        if "kwargs" in signature_keys:
            params["kwargs"] = kwargs
        if "args" in signature_keys:
            params["args"] = args
        if not set(params).issuperset(signature_keys):
            logger.error("Params mismatch for %s.%s", func.__module__, func.__name__)

    return params
