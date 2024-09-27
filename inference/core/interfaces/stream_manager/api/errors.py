from typing import Optional


class ProcessesManagerClientError(Exception):

    def __init__(
        self,
        private_message: str,
        public_message: Optional[str] = None,
        inner_error: Optional[Exception] = None,
    ):
        super().__init__(private_message)
        self._public_message = public_message
        self._inner_error = inner_error

    @property
    def public_message(self) -> str:
        return self._public_message

    @property
    def inner_error_type(self) -> Optional[str]:
        if self._inner_error is None:
            return None
        return self._inner_error.__class__.__name__

    @property
    def inner_error(self) -> Optional[Exception]:
        return self._inner_error


class ConnectivityError(ProcessesManagerClientError):
    pass


class ProcessesManagerInternalError(ProcessesManagerClientError):
    pass


class ProcessesManagerOperationError(ProcessesManagerClientError):
    pass


class ProcessesManagerInvalidPayload(ProcessesManagerClientError):
    pass


class ProcessesManagerNotFoundError(ProcessesManagerClientError):
    pass


class ProcessesManagerAuthorisationError(ProcessesManagerClientError):
    pass
