"""Utility functions for error formatting in dynamic blocks."""

import sys
import threading
import traceback
from contextlib import contextmanager
from io import StringIO
from typing import Generator, Optional, Tuple

from inference.core.workflows.errors import DynamicBlockCodeError

_thread_local = threading.local()
_install_lock = threading.Lock()


class _ThreadDispatchStream:
    """Stream wrapper that tees writes into a per-thread StringIO buffer
    (when one is active) while still forwarding them to the original stream.

    Threads that are not capturing see normal stdout/stderr behaviour; threads
    that are capturing get both: the buffer keeps an in-memory copy for error
    payloads, and the original stream still receives the bytes so ``print()``
    output continues to reach Docker / the process stdout.
    """

    def __init__(self, original, attr_name: str):
        object.__setattr__(self, "_original", original)
        object.__setattr__(self, "_attr_name", attr_name)

    def _get_buffer(self):
        return getattr(_thread_local, self._attr_name, None)

    def write(self, data):
        buf = self._get_buffer()
        if buf is not None:
            try:
                buf.write(data)
            except Exception:
                pass
        return self._original.write(data)

    def flush(self):
        buf = self._get_buffer()
        if buf is not None:
            try:
                buf.flush()
            except Exception:
                pass
        return self._original.flush()

    def fileno(self):
        return self._original.fileno()

    def isatty(self):
        return self._original.isatty()

    def __getattr__(self, name):
        return getattr(self._original, name)


def _install_dispatchers() -> None:
    if isinstance(sys.stdout, _ThreadDispatchStream):
        return
    with _install_lock:
        if isinstance(sys.stdout, _ThreadDispatchStream):
            return
        sys.stdout = _ThreadDispatchStream(sys.stdout, "_capture_stdout")
        sys.stderr = _ThreadDispatchStream(sys.stderr, "_capture_stderr")


@contextmanager
def capture_output() -> Generator[Tuple[StringIO, StringIO], None, None]:
    """Context manager to capture stdout and stderr for the current thread.

    Uses per-thread buffers via ``threading.local`` so concurrent calls in
    different threads capture independently without any global lock.
    """
    _install_dispatchers()
    stdout_buf, stderr_buf = StringIO(), StringIO()
    _thread_local._capture_stdout = stdout_buf
    _thread_local._capture_stderr = stderr_buf
    try:
        yield stdout_buf, stderr_buf
    finally:
        _thread_local._capture_stdout = None
        _thread_local._capture_stderr = None


def extract_code_snippet(
    code: Optional[str], error_line: int, context: int = 10
) -> str:
    """Extract a code snippet around the error line with markers."""
    if not code:
        return ""

    lines = code.splitlines()
    error_idx = error_line - 1
    start = max(0, error_idx - context)
    end = min(len(lines), error_idx + context + 1)
    snippet_lines = [
        f"{'>>>' if i == error_idx else '   '} {i + 1}: {lines[i]}"
        for i in range(start, end)
    ]
    return "\n" + "\n".join(snippet_lines)


def build_traceback_string(
    code: Optional[str],
    line_number: int,
    function_name: str,
    error_type: str,
    error_msg: str,
) -> str:
    """Build a traceback string from structured error data."""
    code_lines = (code or "").splitlines()
    code_line = (
        code_lines[line_number - 1].strip()
        if 0 < line_number <= len(code_lines)
        else ""
    )

    lines = [
        "Traceback (most recent call last):",
        f'  File "Python Block", line {line_number}, in {function_name}',
    ]
    if code_line:
        lines.append(f"    {code_line}")
    lines.append(f"{error_type}: {error_msg}")
    return "\n".join(lines)


def _create_clean_traceback(
    error: Exception,
    user_code: str,
    import_lines_count: int,
) -> str:
    """Create a clean traceback showing only user code with adjusted line numbers."""
    tb = traceback.extract_tb(error.__traceback__)
    code_lines = user_code.splitlines()

    lines = ["Traceback (most recent call last):"]
    for frame in tb:
        if frame.filename == "<string>":
            adjusted_line = frame.lineno - import_lines_count
            code_line = (
                code_lines[adjusted_line - 1]
                if 0 < adjusted_line <= len(code_lines)
                else ""
            )
            lines.append(
                f'  File "Python Block", line {adjusted_line}, in {frame.name}'
            )
            if code_line:
                lines.append(f"    {code_line.strip()}")

    lines.append(f"{error.__class__.__name__}: {error}")
    return "\n".join(lines)


def create_dynamic_block_code_error(
    error: Exception,
    user_code: str,
    import_lines_count: int,
    stdout: Optional[str] = None,
    stderr: Optional[str] = None,
    block_type_name: Optional[str] = None,
) -> DynamicBlockCodeError:
    """Create a DynamicBlockCodeError with structured code context.

    Args:
        error: The exception that was raised.
        user_code: The user's Python code (run_function_code).
        import_lines_count: Number of import lines prepended to the code.
        stdout: Captured stdout, if any.
        stderr: Captured stderr, if any.
        block_type_name: The dynamic block's type identifier.
    """
    tb = traceback.extract_tb(error.__traceback__)
    if not tb:
        return DynamicBlockCodeError(
            public_message=f"{error.__class__.__name__}: {error}",
            inner_error=error,
            block_type_name=block_type_name,
            stdout=stdout,
            stderr=stderr,
        )

    frame = tb[-1]
    line_number = frame.lineno - import_lines_count

    code_snippet = extract_code_snippet(user_code, line_number)
    message = f"Error in line {line_number}, in {frame.name}: {error.__class__.__name__}: {error}"
    clean_traceback = _create_clean_traceback(error, user_code, import_lines_count)

    return DynamicBlockCodeError(
        public_message=message,
        inner_error=error,
        block_type_name=block_type_name,
        error_line=line_number,
        code_snippet=code_snippet.lstrip("\n") if code_snippet else None,
        traceback_str=clean_traceback,
        stdout=stdout,
        stderr=stderr,
    )
