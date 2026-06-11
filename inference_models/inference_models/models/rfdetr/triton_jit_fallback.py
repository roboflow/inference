"""Detect Triton JIT failures and fall back to reference RF-DETR paths."""

from __future__ import annotations

import logging
import warnings
from typing import Set

logger = logging.getLogger(__name__)

try:
    from triton.runtime.errors import PTXASError as _PTXASError
except ImportError:  # pragma: no cover - optional at import time
    _PTXASError = ()

_RUNTIME_ERROR_MARKERS = (
    "c compiler",
    "ptxas",
    "ptx codegen",
    "triton ptx",
)


def is_triton_jit_failure(exc: BaseException) -> bool:
    """Return whether ``exc`` looks like a Triton compile-time failure."""
    if _PTXASError and isinstance(exc, _PTXASError):
        return True
    if isinstance(exc, RuntimeError):
        message = str(exc).lower()
        return any(marker in message for marker in _RUNTIME_ERROR_MARKERS)
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
