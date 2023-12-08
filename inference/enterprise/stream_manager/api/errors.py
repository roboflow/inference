class ProcessesManagerClientError(Exception):
    pass


class ProcessesManagerInternalError(ProcessesManagerClientError):
    pass


class ProcessesManagerOperationError(ProcessesManagerClientError):
    pass


class ProcessesManagerInvalidPayload(ProcessesManagerClientError):
    pass


class ProcessesManagerNotFound(ProcessesManagerClientError):
    pass


class ProcessesManagerAuthorisationError(ProcessesManagerClientError):
    pass
