import inspect
from threading import Lock
from typing import Any, Callable, Dict, Iterable

from inference.core.logger import logger


signatures = {}
lock = Lock()


def get_signature(func: Callable[[Any], Any]) -> inspect.Signature:
    with lock:
        if func not in signatures:
            signatures[func] = inspect.signature(func)
        return signatures[func]


def collect_func_params(
    func: Callable[[Any], Any], args: Iterable[Any], kwargs: Dict[Any, Any]
) -> Dict[str, Any]:
    signature = get_signature(func)

    params = {}
    if args:
        for param, arg_value in zip(signature.parameters.keys(), args):
            params[param] = arg_value
    if kwargs:
        params = {**params, **kwargs}
    defaults = set(signature.parameters.keys()).difference(set(params.keys()))
    for default_arg in defaults:
        params[default_arg] = signature.parameters[default_arg].default

    signature_params = set(signature.parameters)
    if set(params) != signature_params:
        if "kwargs" in signature_params:
            params["kwargs"] = kwargs
        if "args" in signature_params:
            params["args"] = args
        if not set(params).issuperset(signature_params):
            logger.error("Params mismatch for %s.%s", func.__module__, func.__name__)

    return params
