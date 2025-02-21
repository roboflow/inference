import inspect
from typing import Any, Callable, Dict, Iterable

from inference.core.logger import logger


def collect_func_params(
    func: Callable[[Any], Any], args: Iterable[Any], kwargs: Dict[Any, Any]
) -> Dict[str, Any]:
    signature = func.__code__.co_varnames
    params = dict(zip(signature, args))

    for param in signature[len(args) :]:
        if param in kwargs:
            params[param] = kwargs[param]

    # Add missing parameters with default values
    for i, param in enumerate(signature):
        if param not in params:
            try:
                default_index = i - (func.__code__.co_argcount - len(func.__defaults__))
                if default_index >= 0:
                    params[param] = func.__defaults__[default_index]
            except (AttributeError, TypeError):
                # There is no default value for this parameter
                continue

    # Add *args and **kwargs if they exist in the function definition
    if func.__code__.co_flags & 0x04:  # CO_VARARGS flag
        params["args"] = args[len(signature) :]
    if func.__code__.co_flags & 0x08:  # CO_VARKEYWORDS flag
        params["kwargs"] = {k: v for k, v in kwargs.items() if k not in signature}

    if not set(params).issuperset(signature):
        logger.error("Params mismatch for %s.%s", func.__module__, func.__name__)

    return params
