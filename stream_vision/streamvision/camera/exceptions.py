class StreamError(Exception):
    pass


class StreamOperationNotAllowedError(StreamError):
    pass


class EndOfStreamError(StreamError):
    pass


class SourceConnectionError(StreamError):
    pass
