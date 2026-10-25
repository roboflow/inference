"""Exceptions raised by the camera, stream and stream-manager runtime.

`inference.core.exceptions` imports these names back, so each historical name
is exactly one class object: `except` clauses, `isinstance` checks and the
class name the stream manager reports as `error_class` do not change.

Keep this module free of imports - `inference.core.exceptions` loads it while
`inference.core` is still being imported.
"""


class InvalidEnvironmentVariableError(Exception):
    """Raised when an environment variable is invalid.

    Attributes:
        message (str): Optional message describing the error.
    """


class MissingApiKeyError(Exception):
    """Raised when the API key is missing.

    Attributes:
        message (str): Optional message describing the error.
    """


class CannotInitialiseModelError(Exception):
    pass


class WebRTCConfigurationError(Exception):
    pass
