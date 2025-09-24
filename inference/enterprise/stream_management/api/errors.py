class ProcessesManagerClientError(Exception):
    pass


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
