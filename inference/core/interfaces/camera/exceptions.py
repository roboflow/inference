class StreamError(Exception):
    pass


class StreamOperationNotAllowedError(StreamError):
    pass


class StreamReadNotFeasibleError(StreamError):
    pass


class EndOfStreamError(StreamError):
    pass
