import functools
import warnings

from inference_sdk.config import InferenceSDKDeprecationWarning


def deprecated(reason: str):
    """Create a decorator that marks functions as deprecated.

    This decorator will emit a warning when the decorated function is called,
    indicating that the function is deprecated and providing a reason.

    Args:
        reason (str): The reason why the function is deprecated.

    Returns:
        callable: A decorator function that can be applied to mark functions as deprecated.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated: {reason}",
                category=InferenceSDKDeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def experimental(info: str):
    """Create a decorator that marks functions as experimental.

    This decorator will emit a warning when the decorated function is called,
    indicating that the function is experimental and providing additional information.

    Args:
        info (str): Information about the experimental status of the function.

    Returns:
        callable: A decorator function that can be applied to mark functions as experimental.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is experimental: {info}",
                category=InferenceSDKDeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator
