import inspect
from threading import Lock
from typing import Any, Callable, Dict, Iterable
from urllib.parse import urlparse

from inference.core.logger import logger

signatures = {}
lock = Lock()


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
