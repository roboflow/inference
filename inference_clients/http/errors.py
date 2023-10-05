from typing import Optional


class HTTPClientError(Exception):
    pass


class HTTPCallErrorError(HTTPClientError):
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
        return self.__description

    @property
    def api_message(self) -> str:
        return self.__api_message

    @property
    def status_code(self) -> int:
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
    pass


class InvalidModelIdentifier(HTTPClientError):
    pass


class ModelNotInitializedError(HTTPClientError):
    pass


class ModelTaskTypeNotSupportedError(HTTPClientError):
    pass


class ModelNotSelectedError(HTTPClientError):
    pass


class EncodingError(HTTPClientError):
    pass


class WrongClientModeError(HTTPClientError):
    pass
