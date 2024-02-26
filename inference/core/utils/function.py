import functools

from inference.core import logger


def deprecated(reason: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.warn(
                f"{func.__name__} is deprecated: {reason}",
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator
