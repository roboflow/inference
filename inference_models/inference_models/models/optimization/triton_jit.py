"""Detect Triton JIT failures and fall back to reference inference paths."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TritonJITFailureDiagnostic:
    """Conservative operator guidance for a recognized Triton JIT failure."""

    category: str
    guidance: str


_MISSING_COMPILER_DIAGNOSTIC = TritonJITFailureDiagnostic(
    category="missing_compiler",
    guidance=(
        "Verify that a supported C compiler is installed and that CC resolves "
        "to a working executable."
    ),
)
_MISSING_DRIVER_LIBRARY_DIAGNOSTIC = TritonJITFailureDiagnostic(
    category="missing_driver_library",
    guidance=(
        "Verify that the NVIDIA driver and container GPU runtime expose the "
        "host driver libraries."
    ),
)
_MISSING_RUNTIME_LIBRARY_DIAGNOSTIC = TritonJITFailureDiagnostic(
    category="missing_runtime_library",
    guidance=(
        "Verify that the required CUDA or system runtime library is installed "
        "and visible to the dynamic linker."
    ),
)
_LINKER_FAILURE_DIAGNOSTIC = TritonJITFailureDiagnostic(
    category="linker_failure",
    guidance=(
        "Inspect the preceding linker diagnostics and verify library versions, "
        "search paths, and ABI compatibility."
    ),
)
_PTX_TOOLCHAIN_DIAGNOSTIC = TritonJITFailureDiagnostic(
    category="ptx_toolchain_mismatch",
    guidance=(
        "Verify that Triton, the CUDA toolkit, the NVIDIA driver, and the target "
        "GPU architecture are mutually compatible."
    ),
)
_KERNEL_RESOURCE_DIAGNOSTIC = TritonJITFailureDiagnostic(
    category="kernel_resource_limit",
    guidance=(
        "Use the reference fallback or adjust the kernel launch configuration "
        "to reduce shared-memory or register pressure."
    ),
)
_COMPILATION_FAILURE_DIAGNOSTIC = TritonJITFailureDiagnostic(
    category="compilation_failure",
    guidance=(
        "Inspect the original compiler diagnostics and verify the Triton, CUDA, "
        "compiler, and target-GPU configuration."
    ),
)

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


def _called_process_jit_diagnostic(
    error: subprocess.CalledProcessError,
) -> Optional[TritonJITFailureDiagnostic]:
    command = _split_command(error.cmd)
    if command and _is_compiler_or_linker_executable(command[0]):
        return _executable_diagnostic(command[0])
    if (
        len(command) >= 3
        and os.path.basename(command[0]).lower() in _SHELL_EXECUTABLES
        and command[1] == "-c"
    ):
        nested_command = _split_command(command[2])
        if nested_command and _is_compiler_or_linker_executable(nested_command[0]):
            return _executable_diagnostic(nested_command[0])

    process_details = " ".join(
        _decode_subprocess_value(value)
        for value in (error, error.output, error.stderr)
        if value is not None
    ).lower()
    return _message_jit_diagnostic(process_details)


def _executable_diagnostic(executable: str) -> TritonJITFailureDiagnostic:
    name = os.path.basename(executable).lower()
    if name == "ptxas":
        return _PTX_TOOLCHAIN_DIAGNOSTIC
    if name in {"ld", "ld.lld", "lld"}:
        return _LINKER_FAILURE_DIAGNOSTIC

    return _COMPILATION_FAILURE_DIAGNOSTIC


def _message_jit_diagnostic(
    message: str,
) -> Optional[TritonJITFailureDiagnostic]:
    has_triton_runtime_error = any(
        marker in message for marker in _TRITON_RUNTIME_ERROR_MARKERS
    )
    has_generic_toolchain_error = any(
        marker in message for marker in _GENERIC_TOOLCHAIN_ERROR_MARKERS
    )
    has_jit_context = any(marker in message for marker in _JIT_CONTEXT_MARKERS)
    if not has_triton_runtime_error and not (
        has_generic_toolchain_error and has_jit_context
    ):
        return None
    if "out of resource" in message:
        return _KERNEL_RESOURCE_DIAGNOSTIC
    if any(marker in message for marker in ("ptxas", "ptx codegen", "triton ptx")):
        return _PTX_TOOLCHAIN_DIAGNOSTIC
    if "c compiler" in message:
        return _MISSING_COMPILER_DIAGNOSTIC
    if "libcuda.so" in message:
        return _MISSING_DRIVER_LIBRARY_DIAGNOSTIC
    if "cannot open shared object file" in message:
        return _MISSING_RUNTIME_LIBRARY_DIAGNOSTIC
    if any(
        marker in message
        for marker in (
            "cannot find -l",
            "linker command failed",
            "undefined reference",
        )
    ):
        return _LINKER_FAILURE_DIAGNOSTIC

    return _COMPILATION_FAILURE_DIAGNOSTIC


def classify_triton_jit_failure(
    exc: BaseException,
) -> Optional[TritonJITFailureDiagnostic]:
    """Classify a Triton JIT failure and provide conservative operator guidance."""
    current: Optional[BaseException] = exc
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if _TRITON_JIT_EXCEPTION_TYPES and isinstance(
            current, _TRITON_JIT_EXCEPTION_TYPES
        ):
            diagnostic = _message_jit_diagnostic(str(current).lower())
            return diagnostic or _COMPILATION_FAILURE_DIAGNOSTIC
        if isinstance(current, subprocess.CalledProcessError):
            diagnostic = _called_process_jit_diagnostic(current)
            if diagnostic is not None:
                return diagnostic
        message = str(current).lower()
        if isinstance(current, RuntimeError):
            diagnostic = _message_jit_diagnostic(message)
            if diagnostic is not None:
                return diagnostic
        # Explicit causes preserve intentional compiler-error wrapping. Implicit
        # context may be an unrelated error that happened to be under handling.
        current = current.__cause__

    return None


def is_triton_jit_failure(exc: BaseException) -> bool:
    """Return whether ``exc`` looks like a Triton compile-time failure."""
    return classify_triton_jit_failure(exc) is not None


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
    diagnostic = classify_triton_jit_failure(exc)
    guidance = (
        f" Category: {diagnostic.category}. Suggested action: {diagnostic.guidance}"
        if diagnostic is not None
        else ""
    )
    logger.error(
        "RF-DETR Triton %s JIT compilation failed; falling back to reference "
        "path: %s%s",
        path,
        exc,
        guidance,
        exc_info=exc,
    )
    warnings.warn(
        "RF-DETR Triton "
        f"{path} path failed during JIT compilation; using reference "
        f"implementation ({exc}).{guidance}",
        RuntimeWarning,
        stacklevel=stacklevel,
    )
