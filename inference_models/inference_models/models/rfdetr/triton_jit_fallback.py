"""Detect Triton JIT failures and fall back to reference RF-DETR paths."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import warnings
from typing import List, Optional, Sequence, Set

logger = logging.getLogger(__name__)

_triton_jit_exception_types: list[type[BaseException]] = []

try:
    from triton.runtime.errors import OutOfResources as _OutOfResources

    _triton_jit_exception_types.append(_OutOfResources)
except ImportError:  # pragma: no cover - optional at import time
    pass

try:
    from triton.runtime.errors import PTXASError as _PTXASError

    _triton_jit_exception_types.append(_PTXASError)
except ImportError:  # pragma: no cover - optional at import time
    pass

try:
    from triton.compiler.errors import CompilationError as _CompilationError

    _triton_jit_exception_types.append(_CompilationError)
except ImportError:  # pragma: no cover - optional at import time
    pass

_TRITON_JIT_EXCEPTION_TYPES = tuple(_triton_jit_exception_types)

# Do not replace runtime classification with a compiler preflight. Triton may use
# CC, cc, GCC, Clang, bundled tooling, or version-specific compilation paths.
# Finding an executable still does not prove that headers, linker libraries,
# architecture support, or the actual specialized kernel compilation will work.
_TRITON_RUNTIME_ERROR_MARKERS = (
    "c compiler",
    "ptxas",
    "ptx codegen",
    "triton ptx",
    "out of resource",
)
_GENERIC_TOOLCHAIN_ERROR_MARKERS = (
    "cannot find -l",
    "cannot open shared object file",
    "libcuda.so",
    "linker command failed",
    "undefined reference",
)
_JIT_CONTEXT_MARKERS = ("triton", "jit", "compiler", "ptxas", "ptx codegen")
_COMPILER_AND_LINKER_EXECUTABLES = frozenset(
    {
        "cc",
        "gcc",
        "g++",
        "clang",
        "clang++",
        "nvcc",
        "ptxas",
        "ld",
        "ld.lld",
        "lld",
    }
)
_VERSIONED_EXECUTABLE_PREFIXES = ("gcc-", "g++-", "clang-", "clang++-", "nvcc-")
_SHELL_EXECUTABLES = frozenset({"bash", "dash", "sh", "zsh"})


def _decode_subprocess_value(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _split_command(command: object) -> List[str]:
    if isinstance(command, (str, bytes)):
        try:
            return shlex.split(_decode_subprocess_value(command))
        except ValueError:
            return [_decode_subprocess_value(command)]
    if isinstance(command, Sequence):
        return [_decode_subprocess_value(part) for part in command]
    return [_decode_subprocess_value(command)]


def _is_compiler_or_linker_executable(executable: str) -> bool:
    name = os.path.basename(executable).lower()
    return name in _COMPILER_AND_LINKER_EXECUTABLES or name.startswith(
        _VERSIONED_EXECUTABLE_PREFIXES
    )


def _called_process_is_jit_related(error: subprocess.CalledProcessError) -> bool:
    command = _split_command(error.cmd)
    if command and _is_compiler_or_linker_executable(command[0]):
        return True
    if (
        len(command) >= 3
        and os.path.basename(command[0]).lower() in _SHELL_EXECUTABLES
        and command[1] == "-c"
    ):
        nested_command = _split_command(command[2])
        if nested_command and _is_compiler_or_linker_executable(nested_command[0]):
            return True

    process_details = " ".join(
        _decode_subprocess_value(value)
        for value in (error, error.output, error.stderr)
        if value is not None
    ).lower()
    return _message_is_jit_related(process_details)


def _message_is_jit_related(message: str) -> bool:
    if any(marker in message for marker in _TRITON_RUNTIME_ERROR_MARKERS):
        return True
    has_generic_toolchain_error = any(
        marker in message for marker in _GENERIC_TOOLCHAIN_ERROR_MARKERS
    )
    has_jit_context = any(marker in message for marker in _JIT_CONTEXT_MARKERS)

    return has_generic_toolchain_error and has_jit_context


def is_triton_jit_failure(exc: BaseException) -> bool:
    """Return whether ``exc`` looks like a Triton compile-time failure."""
    current: Optional[BaseException] = exc
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if _TRITON_JIT_EXCEPTION_TYPES and isinstance(
            current, _TRITON_JIT_EXCEPTION_TYPES
        ):
            return True
        if isinstance(
            current, subprocess.CalledProcessError
        ) and _called_process_is_jit_related(current):
            return True
        message = str(current).lower()
        if isinstance(current, RuntimeError) and _message_is_jit_related(message):
            return True
        # Explicit causes preserve intentional compiler-error wrapping. Implicit
        # context may be an unrelated error that happened to be under handling.
        current = current.__cause__

    return False


def warn_triton_jit_fallback(
    *,
    path: str,
    exc: BaseException,
    warned_reasons: Set[str],
    stacklevel: int = 4,
) -> None:
    """Log and warn once per distinct JIT failure reason."""
    reason = f"{type(exc).__name__}: {exc}"
    if reason in warned_reasons:
        return
    warned_reasons.add(reason)
    logger.error(
        "RF-DETR Triton %s JIT compilation failed; falling back to reference path: %s",
        path,
        exc,
        exc_info=exc,
    )
    warnings.warn(
        "RF-DETR Triton "
        f"{path} path failed during JIT compilation; using reference "
        f"implementation ({exc})",
        RuntimeWarning,
        stacklevel=stacklevel,
    )
