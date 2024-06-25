import inspect
from typing import Any, Callable, Dict, Iterable

from inference.core.logger import logger


def collect_func_params(
    func: Callable[[Any], Any], args: Iterable[Any], kwargs: Dict[Any, Any]
) -> Dict[str, Any]:
    signature = inspect.signature(func)

    params = {}
    if args:
        for param, arg_value in zip(signature.parameters.keys(), args):
            params[param] = arg_value
    if kwargs:
        params = {**params, **kwargs}
    defaults = set(signature.parameters.keys()).difference(set(params.keys()))
    for default_arg in defaults:
        params[default_arg] = signature.parameters[default_arg].default

    if set(params) != set(signature.parameters):
        logger.error("Params mismatch for %s.%s", func.__module__, func.__name__)

    return params
