"""Unit tests for the `bind_capture_to` helper used by the Modal watchdog."""

import threading
import time
from io import StringIO

from inference.core.workflows.execution_engine.v1.dynamic_blocks.error_utils import (
    _thread_local,
    bind_capture_to,
    capture_output,
)


def test_bind_capture_to_captures_writes_from_calling_thread() -> None:
    stdout_buf, stderr_buf = StringIO(), StringIO()

    import sys

    with bind_capture_to(stdout_buf, stderr_buf):
        sys.stdout.write("from main\n")
        sys.stderr.write("err line\n")

    assert "from main" in stdout_buf.getvalue()
    assert "err line" in stderr_buf.getvalue()


def test_bind_capture_to_clears_thread_local_on_exit() -> None:
    stdout_buf, stderr_buf = StringIO(), StringIO()

    with bind_capture_to(stdout_buf, stderr_buf):
        assert getattr(_thread_local, "_capture_stdout", None) is stdout_buf
        assert getattr(_thread_local, "_capture_stderr", None) is stderr_buf

    assert getattr(_thread_local, "_capture_stdout", None) is None
    assert getattr(_thread_local, "_capture_stderr", None) is None


def test_bind_capture_to_does_not_leak_to_other_threads() -> None:
    """Writes from a concurrent unrelated thread MUST NOT appear in the bound buffers."""

    bound_stdout, bound_stderr = StringIO(), StringIO()
    other_stdout, other_stderr = StringIO(), StringIO()

    import sys

    bind_thread_done = threading.Event()
    other_thread_done = threading.Event()
    bind_thread_started = threading.Event()

    def bind_thread():
        with bind_capture_to(bound_stdout, bound_stderr):
            bind_thread_started.set()
            other_thread_done.wait(timeout=5)
            sys.stdout.write("bind-thread message\n")
        bind_thread_done.set()

    def other_thread():
        # Wait until the bind thread is inside its `with` so the dispatcher
        # is installed and `_thread_local` is set on the bind thread (not on us).
        bind_thread_started.wait(timeout=5)
        with capture_output() as (os_, es_):
            sys.stdout.write("other-thread message\n")
            sys.stderr.write("other-thread err\n")
            other_stdout.write(os_.getvalue())
            other_stderr.write(es_.getvalue())
        other_thread_done.set()

    t1 = threading.Thread(target=bind_thread)
    t2 = threading.Thread(target=other_thread)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert bind_thread_done.is_set()
    assert other_thread_done.is_set()

    # The bind thread's buffers contain only its own writes.
    assert "bind-thread message" in bound_stdout.getvalue()
    assert "other-thread message" not in bound_stdout.getvalue()
    assert "other-thread err" not in bound_stderr.getvalue()

    # And the other thread's writes were captured by its own capture_output().
    assert "other-thread message" in other_stdout.getvalue()
    assert "other-thread err" in other_stderr.getvalue()


def test_bind_capture_to_snapshot_safe_while_worker_still_writing() -> None:
    """Simulates the watchdog scenario: main reads `getvalue()` while a worker thread
    is still writing into the same buffer. CPython's GIL makes these calls
    individually atomic — we should observe a coherent snapshot."""

    import sys

    stdout_buf, stderr_buf = StringIO(), StringIO()
    worker_ready = threading.Event()
    main_read_done = threading.Event()

    def worker():
        with bind_capture_to(stdout_buf, stderr_buf):
            sys.stdout.write("part-1\n")
            worker_ready.set()
            # Wait for the main thread to read getvalue() before writing more.
            main_read_done.wait(timeout=5)
            sys.stdout.write("part-2\n")

    t = threading.Thread(target=worker)
    t.start()
    worker_ready.wait(timeout=5)
    # Reading from main thread while worker is bound to the buffer must not raise.
    snapshot = stdout_buf.getvalue()
    main_read_done.set()
    t.join(timeout=5)

    assert "part-1" in snapshot
    # part-2 is written after our read, so snapshot must not have observed it.
    assert "part-2" not in snapshot
    # After the worker finishes, the full content is present.
    assert "part-2" in stdout_buf.getvalue()
