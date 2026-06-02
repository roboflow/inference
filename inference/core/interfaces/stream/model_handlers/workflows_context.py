from contextlib import contextmanager
import threading

_WORKFLOW_STREAM_CONTEXT = threading.local()


def is_workflow_stream_flush_active() -> bool:
    return bool(getattr(_WORKFLOW_STREAM_CONTEXT, "flush_active", False))


@contextmanager
def workflow_stream_flush_context():
    previous = is_workflow_stream_flush_active()
    _WORKFLOW_STREAM_CONTEXT.flush_active = True
    try:
        yield
    finally:
        _WORKFLOW_STREAM_CONTEXT.flush_active = previous
