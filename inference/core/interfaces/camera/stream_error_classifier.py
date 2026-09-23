"""Classify producer failures into structured stream error codes."""

from __future__ import annotations

import array
import os
import re
import select
import threading
import time
from contextlib import contextmanager
from typing import Iterator, List, Optional, Set, Tuple

try:  # POSIX only: lets the drain be settled without writing into the pipe
    import fcntl
    import termios

    _FIONREAD: Optional[int] = termios.FIONREAD
except (ImportError, AttributeError):  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]
    _FIONREAD = None

from inference.core.env import DISABLE_NATIVE_STDERR_CAPTURE
from inference.core.interfaces.camera.exceptions import SourceConnectionError
from inference.core.interfaces.camera.source_reference_sanitizer import (
    redact_credentials_in_text,
)
from inference.core.interfaces.camera.stream_error_codes import StreamErrorCode

_AUTH_STATUS_PATTERN = re.compile(r"\b401\b|\b403\b")
_NOT_FOUND_STATUS_PATTERN = re.compile(r"\b404\b")
_STREAM_OPEN_ERROR_HINTS = (
    "error",
    "failed",
    "unauthorized",
    "forbidden",
    "not found",
    "timeout",
    "timed out",
    "handshake",
    "certificate",
    "401",
    "403",
    "404",
)


STDERR_FILENO = 2

# fd 2 is a process-wide resource, so the redirection below is shared by every
# thread that happens to be opening a native video source at the same time.
# Retaining a quarter of a megabyte is far more than the handful of FFmpeg
# lines the classifier needs while bounding memory when a chatty backend spams.
_MAX_CAPTURED_BYTES = 256 * 1024
_DRAIN_CHUNK_BYTES = 65536
_DRAIN_POLL_SECONDS = 0.2

# How long a participant will wait for the drain thread to catch up before it
# settles for what has been buffered so far. This is on the reconnect path - a
# source retrying every second against a 30 s open timeout crosses it twice per
# attempt - and the text it delimits is only a hint for error classification,
# so a fraction of a second is the right order of magnitude. It is only ever
# reached when some other thread is writing to stderr continuously; normally
# the wait ends on the first notification, in well under a millisecond.
_QUIESCE_TIMEOUT_SECONDS = 0.25
_QUIESCE_POLL_SECONDS = 0.02

# Capability marker for integrators that used to serialise native opens behind
# a lock of their own to work around the historical non-reentrancy of this
# helper: it is also set on `capture_process_stderr` itself, so such a
# workaround can detect a build that no longer needs it and disable itself.
NATIVE_STDERR_CAPTURE_HANDLES_CONCURRENT_OPENS = True

_capture_state_lock = threading.Lock()
_active_capture: Optional["_SharedStderrCapture"] = None

# Every capture that still owns a descriptor, whether or not it is the one
# currently installed: a capture is released - fd 2 restored, write end closed -
# before its drain thread has finished with the read end. Guarded by
# ``_capture_state_lock``, which is also the fork lock, so ``fork()`` cannot land
# between a drain's final close and its removal from here. The forked child has
# no drain threads, so it uses this to close what it inherited.
_captures_owning_descriptors: "Set[_SharedStderrCapture]" = set()


if hasattr(select, "poll"):

    def _wait_readable(read_fd: int, timeout: float) -> bool:
        # poll() rather than select(): select() raises ValueError once a
        # descriptor number reaches FD_SETSIZE (1024), which a server holding
        # many cameras and sockets reaches easily, and a capture that cannot
        # test for readability never drains its pipe.
        poller = select.poll()
        poller.register(read_fd, select.POLLIN | select.POLLHUP | select.POLLERR)
        try:
            return bool(poller.poll(timeout * 1000.0))
        except (OSError, ValueError):
            return False

else:  # pragma: no cover - Windows has neither poll() nor pipe-capable select()

    def _wait_readable(read_fd: int, timeout: float) -> bool:
        # Fall back to a blocking read, ended by EOF when the write ends close.
        return True


def _pipe_unread_bytes(read_fd: Optional[int]) -> Optional[int]:
    """Bytes sitting in the pipe, or None where that cannot be asked."""
    if read_fd is None or fcntl is None or _FIONREAD is None:
        return None
    try:
        buffer = array.array("i", [0])
        fcntl.ioctl(read_fd, _FIONREAD, buffer, True)
        return int(buffer[0])
    except (OSError, ValueError):
        return None


class _SharedStderrCapture:
    """One redirection of fd 2 into a pipe drained by a background thread.

    Instances are shared by every concurrent participant and reference counted:
    the first entrant installs the redirection, the last leaver removes it. All
    fd manipulation happens under ``_capture_state_lock`` and never spans the
    body of the context manager, so a slow native open never blocks anybody.

    Descriptors live in ``_fds`` and are closed through :meth:`_close_owned`,
    which pops the entry before closing. ``dict.pop`` is atomic, so each is
    closed exactly once however many cleanup paths run - a closed number is
    immediately reusable, and closing it twice shuts an unrelated file.
    """

    def __init__(self, read_fd: int, write_fd: int, restore_fd: int) -> None:
        self._fds = {"read": read_fd, "write": write_fd, "restore": restore_fd}
        self._condition = threading.Condition()
        self._buffer = bytearray()
        self._buffer_start = 0  # absolute offset of self._buffer[0]
        self._total_bytes = 0  # absolute offset just past self._buffer[-1]
        self._idle = False
        self._stop = threading.Event()
        self._restored = False
        self._drain_entered = False
        # Set only by the at-fork child handler, and only read - never written -
        # by a context manager unwinding in the child.
        self.discarded = False
        self.ref_count = 0
        self._drain_thread = threading.Thread(
            target=self._drain,
            name="native-stderr-capture",
            daemon=True,
        )

    def _close_owned(self, name: str) -> None:
        file_descriptor = self._fds.pop(name, None)
        if file_descriptor is None:
            return
        try:
            os.close(file_descriptor)
        except OSError:
            pass

    @classmethod
    def start(cls) -> Optional["_SharedStderrCapture"]:
        """Install the redirection, or return None if fd 2 cannot be captured.

        Caller holds ``_capture_state_lock``.
        """
        # Duplicate the real stderr *before* allocating the pipe. If fd 2 is
        # closed, os.pipe() would be handed descriptor 2 and the capture would
        # later "restore" a pipe end onto it; duplicating first fails cleanly
        # instead, and the caller carries on with no capture at all.
        try:
            restore_fd = os.dup(STDERR_FILENO)
        except OSError:
            return None
        capture = None
        read_fd = write_fd = None
        try:
            read_fd, write_fd = os.pipe()
            capture = cls(read_fd=read_fd, write_fd=write_fd, restore_fd=restore_fd)
            _captures_owning_descriptors.add(capture)
            os.dup2(write_fd, STDERR_FILENO)
            capture._drain_thread.start()
            return capture
        except BaseException as error:
            # Includes an interrupt landing inside Thread.start(), which may
            # already have spawned the thread: if it is running, it owns the
            # read end and closes it itself, so only take it back when it is
            # certainly not.
            if capture is None:
                for file_descriptor in (read_fd, write_fd, restore_fd):
                    if file_descriptor is not None:
                        try:
                            os.close(file_descriptor)
                        except OSError:
                            pass
            else:
                capture.restore_stderr()
                capture._stop.set()
                capture._close_owned("write")
                if not capture._drain_entered and not capture._drain_thread.is_alive():
                    capture._close_owned("read")
                    _captures_owning_descriptors.discard(capture)
            if isinstance(error, Exception):
                # Capturing stderr is a diagnostics nicety, never a reason to
                # fail an open. Interrupts still propagate.
                return None
            raise

    def _drain(self) -> None:
        global _active_capture
        self._drain_entered = True
        try:
            while not self._stop.is_set():
                read_fd = self._fds.get("read")
                if read_fd is None:
                    break
                if not self._wait_for_input(read_fd):
                    continue
                try:
                    chunk = os.read(read_fd, _DRAIN_CHUNK_BYTES)
                except (OSError, ValueError):
                    break
                if not chunk:  # EOF: every write end is gone
                    break
                self._append(chunk)
        finally:
            with _capture_state_lock:
                # Only matters if the loop stopped while still installed: fd 2
                # must not point at a pipe nobody reads, or a full pipe would
                # block every native writer.
                if _active_capture is self:
                    self.restore_stderr()
                    _active_capture = None
                # Under the lock, so the last close and giving up the claim are
                # one step as far as fork() is concerned.
                self._close_owned("read")
                _captures_owning_descriptors.discard(self)
            with self._condition:
                self._idle = True
                self._condition.notify_all()

    def _wait_for_input(self, read_fd: int) -> bool:
        with self._condition:
            self._idle = True
            self._condition.notify_all()
        try:
            return _wait_readable(read_fd, _DRAIN_POLL_SECONDS)
        finally:
            with self._condition:
                self._idle = False

    def _append(self, chunk: bytes) -> None:
        with self._condition:
            self._buffer += chunk
            self._total_bytes += len(chunk)
            overflow = len(self._buffer) - _MAX_CAPTURED_BYTES
            if overflow > 0:
                del self._buffer[:overflow]
                self._buffer_start += overflow
            self._condition.notify_all()

    def quiesce(self) -> int:
        """Absolute offset once everything written so far has been buffered.

        Nothing is written into the pipe to find this out - the pipe is asked
        how much it still holds, and the drain thread reports when it is idle.
        Bounded by ``_QUIESCE_TIMEOUT_SECONDS``; on timeout the current offset
        is returned, which can only make a window slightly less complete.
        """
        deadline = time.monotonic() + _QUIESCE_TIMEOUT_SECONDS
        previous_total: Optional[int] = None
        while True:
            pending = _pipe_unread_bytes(self._fds.get("read"))
            with self._condition:
                total = self._total_bytes
                if not self._drain_thread.is_alive():
                    return total
                if pending == 0 and self._idle:
                    return total
                if pending is None and previous_total == total:
                    # No way to ask the pipe: settle for the byte counter
                    # having stopped moving across one poll interval.
                    return total
                previous_total = total
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return total
                self._condition.wait(min(remaining, _QUIESCE_POLL_SECONDS))

    def text_between(self, start: int, end: int) -> str:
        with self._condition:
            begin = max(start, self._buffer_start)
            stop = max(begin, min(end, self._total_bytes))
            raw = bytes(
                self._buffer[begin - self._buffer_start : stop - self._buffer_start]
            )
        return raw.decode("utf-8", errors="replace")

    def restore_stderr(self) -> None:
        """Point fd 2 back at the real stderr.

        Caller holds ``_capture_state_lock`` (the at-fork child handler is the
        exception: that process has exactly one thread). One-shot, so a late
        cleanup path can never dup a stale descriptor over a redirection
        installed by a newer capture.
        """
        if self._restored:
            return
        self._restored = True
        restore_fd = self._fds.get("restore")
        if restore_fd is not None:
            try:
                os.dup2(restore_fd, STDERR_FILENO)
            except OSError:
                pass
        self._close_owned("restore")

    def shut_down(self) -> None:
        """Stop the drain and hand the pipe back. Caller holds the state lock.

        The read end is left to the drain thread, the only owner that knows
        when it has stopped using it; it notices the stop within one poll
        interval, or at once through EOF when this closes the last write end.
        """
        self._stop.set()
        self._close_owned("write")

    def discard_in_forked_child(self) -> None:
        """Undo an inherited redirection. Runs in a single-threaded child.

        Touches no per-instance lock: another thread could have been holding
        the condition when ``fork()`` ran, and it will never be released here.
        """
        self.discarded = True
        self._stop.set()
        self.restore_stderr()
        self._close_owned("write")
        self._close_owned("read")


def _reset_after_fork() -> None:
    # The drain thread does not survive fork(), so the inherited redirection
    # would silently swallow the child's stderr forever.
    global _active_capture
    try:
        capture = _active_capture
        _active_capture = None
        if capture is not None:
            capture.discard_in_forked_child()
        # A capture released moments before the fork has already given fd 2
        # back, but its drain thread - which does not exist here - still owned
        # the read end. Nothing drains it in the child, so just close it.
        for orphan in list(_captures_owning_descriptors):
            if orphan is not capture:
                orphan.discard_in_forked_child()
        _captures_owning_descriptors.clear()
    finally:
        _capture_state_lock.release()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_capture_state_lock.acquire,
        after_in_parent=_capture_state_lock.release,
        after_in_child=_reset_after_fork,
    )


def _reset_capture_state_for_tests() -> None:
    """Tear down any capture left behind by a failed test."""
    global _active_capture
    with _capture_state_lock:
        capture, _active_capture = _active_capture, None
        if capture is not None:
            capture.restore_stderr()
            capture.shut_down()


def _join_or_start_capture() -> Optional[Tuple["_SharedStderrCapture", int]]:
    global _active_capture
    with _capture_state_lock:
        capture = _active_capture
        if capture is None:
            capture = _SharedStderrCapture.start()
            if capture is None:
                return None
            _active_capture = capture
            capture.ref_count = 1
            # Nothing can have been written yet, so no settling needed.
            return capture, 0
        capture.ref_count += 1
    try:
        start_offset = capture.quiesce()
    except BaseException:
        # An interruption between taking the reference and returning it to the
        # caller would otherwise strand the capture, and fd 2 with it.
        _release_capture(capture)
        raise
    return capture, start_offset


def _release_capture(capture: "_SharedStderrCapture") -> None:
    global _active_capture
    with _capture_state_lock:
        capture.ref_count -= 1
        if capture.ref_count > 0:
            return
        if _active_capture is capture:
            _active_capture = None
        # Restoring fd 2 under the lock is what makes the next entrant dup the
        # *real* stderr rather than this capture's pipe.
        capture.restore_stderr()
        capture.shut_down()
    # Deliberately no join: the drain owns and closes the read end itself, the
    # next capture allocates fresh descriptors, and making every open wait out
    # a thread exit would add latency to the reconnect path for no guarantee a
    # bounded join could actually give.


@contextmanager
def capture_process_stderr() -> Iterator[List[str]]:
    """Capture OS stderr (fd 2) for native backends such as FFmpeg/GStreamer.

    fd 2 is process-wide, so concurrent callers share a single redirection that
    is reference counted: the first entrant installs it, the last leaver takes
    it down and fd 2 is restored even when the body raises. A background thread
    drains the pipe continuously, so a backend that outpaces the pipe buffer
    cannot block the caller, and window boundaries are found by asking the pipe
    how much it still holds rather than by writing anything into it.

    Because the redirection is shared, overlapping captures may see each
    other's lines. That is acceptable: the text is only a hint used to classify
    a failed open (see ``extract_stream_open_error``), and it is consulted only
    when the open actually failed, so a successful open is never turned into an
    error by another source's output.

    Set ``DISABLE_NATIVE_STDERR_CAPTURE=True`` to leave fd 2 alone entirely.

    ``capture_process_stderr.handles_concurrent_opens`` marks a build on which
    concurrent entry is safe, for integrators that previously serialised native
    opens around this helper and want to drop that workaround.

    Known limitation, unchanged from before this was made concurrency-safe:
    while a capture is active fd 2 is the pipe, so stderr written by the rest
    of the process is captured rather than logged, and a subprocess spawned
    during a capture inherits the pipe and sees EPIPE on stderr writes made
    after the capture ends.
    """
    captured_chunks: List[str] = []
    if DISABLE_NATIVE_STDERR_CAPTURE:
        yield captured_chunks
        return
    joined = _join_or_start_capture()
    if joined is None:
        yield captured_chunks
        return
    capture, start_offset = joined
    # Nothing between here and the `try` may touch state the `finally` has to
    # undo. An asynchronous exception can still land on any bytecode boundary;
    # that cannot be made airtight in pure Python, and in practice the only
    # such exception, KeyboardInterrupt, is delivered to the main thread while
    # these opens run on per-pipeline worker threads.
    try:
        yield captured_chunks
    finally:
        if not capture.discarded:
            try:
                end_offset = capture.quiesce()
                captured = capture.text_between(start_offset, end_offset)
                if captured:
                    captured_chunks.append(captured)
            finally:
                # Unconditional: an interruption anywhere in the bookkeeping
                # above must still give the reference back and restore fd 2.
                _release_capture(capture)
        # else: a fork happened inside the body. The child inherited this
        # object but not its drain thread, and another thread may have been
        # holding its condition when fork() ran, so touch none of it - the
        # at-fork handler has already put the child's fd 2 back.


capture_process_stderr.handles_concurrent_opens = (
    NATIVE_STDERR_CAPTURE_HANDLES_CONCURRENT_OPENS
)


def extract_stream_open_error(stderr: str) -> str:
    if not stderr:
        return ""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    for line in reversed(lines):
        lowered = line.lower()
        if any(hint in lowered for hint in _STREAM_OPEN_ERROR_HINTS):
            return line
    return lines[-1] if lines else ""


def build_source_connection_error_message(
    source_reference: str, underlying_error: str = ""
) -> str:
    summary = f"Cannot connect to video source under reference: {source_reference}"
    detail = redact_credentials_in_text((underlying_error or "").strip())
    if detail:
        return f"{detail}: {summary}"
    return summary


def classify_stream_error_message(message: str) -> StreamErrorCode:
    lowered = (message or "").lower()
    if (
        _AUTH_STATUS_PATTERN.search(lowered)
        or "unauthorized" in lowered
        or "authentication" in lowered
        or "forbidden" in lowered
    ):
        return StreamErrorCode.STREAM_AUTH_FAILED
    if (
        ("handshake" in lowered and ("tls" in lowered or "ssl" in lowered))
        or "ssl handshake failed" in lowered
        or "tls handshake failed" in lowered
    ):
        return StreamErrorCode.STREAM_TLS_HANDSHAKE
    if any(
        token in lowered
        for token in ("certificate verify failed", "x509", "tls certificate")
    ):
        return StreamErrorCode.STREAM_TLS_CERTIFICATE
    if "timed out" in lowered or "timeout" in lowered:
        return StreamErrorCode.STREAM_TIMEOUT
    if _NOT_FOUND_STATUS_PATTERN.search(lowered) or "not found" in lowered:
        return StreamErrorCode.STREAM_NOT_FOUND
    if "codec" in lowered and "unsupported" in lowered:
        return StreamErrorCode.STREAM_CODEC_UNSUPPORTED
    return StreamErrorCode.STREAM_CONNECTION_FAILED


def wrap_source_connection_error(
    message: str,
    source_reference: str = "",
    classification_text: Optional[str] = None,
) -> SourceConnectionError:
    # classify from the raw text: redaction must not influence the error
    # taxonomy (redacted messages e.g. lose ffmpeg's '?timeout=0' query)
    code = classify_stream_error_message(
        message if classification_text is None else classification_text
    )
    error = SourceConnectionError(message)
    setattr(error, "code", code)
    setattr(error, "source_reference", source_reference)
    return error
