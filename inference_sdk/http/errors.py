from typing import Optional


class HTTPClientError(Exception):
    """Base class for HTTP client errors."""

    pass


class HTTPCallErrorError(HTTPClientError):
    """Error for HTTP call errors.

    Attributes:
        description: The description of the error.
        status_code: The status code of the error.
        api_message: The API message of the error.
    """

    def __init__(
        self,
        description: str,
        status_code: int,
        api_message: Optional[str],
    ):
        super().__init__(description)
        self.__description = description
        self.__api_message = api_message
        self.__status_code = status_code

    @property
    def description(self) -> str:
        """The description of the error."""
        return self.__description

    @property
    def api_message(self) -> str:
        """The API message of the error."""
        return self.__api_message

    @property
    def status_code(self) -> int:
        """The status code of the error."""
        return self.__status_code

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"description='{self.description}', "
            f"api_message='{self.api_message}',"
            f"status_code={self.__status_code})"
        )

    def __str__(self) -> str:
        return self.__repr__()


class InvalidInputFormatError(HTTPClientError):
    """Error for invalid input format."""

    pass


class InvalidModelIdentifier(HTTPClientError):
    """Error for invalid model identifier."""

    pass


class ModelNotInitializedError(HTTPClientError):
    """Error for model not initialized."""

    pass


class ModelTaskTypeNotSupportedError(HTTPClientError):
    """Error for model task type not supported."""

    pass


class ModelNotSelectedError(HTTPClientError):
    """Error for model not selected."""

    pass


class APIKeyNotProvided(HTTPClientError):
    """Error for API key not provided."""

    pass


class EncodingError(HTTPClientError):
    """Error for encoding errors."""

    pass


class WrongClientModeError(HTTPClientError):
    """Error for wrong client mode."""

    pass


class InvalidParameterError(HTTPClientError):
    """Error for invalid parameter."""

    pass
