"""Copy of `inference.core.utils.function.experimental`.

Warns with the historical `InferenceExperimentalFeatureWarning` category
(re-exported by `inference.core.warnings`), not the Workflows variant.
"""

import functools
import warnings

from inference.core.interfaces.stream.warnings import (
    InferenceExperimentalFeatureWarning,
)


def experimental(reason: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is experimental: {reason}",
                category=InferenceExperimentalFeatureWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator
