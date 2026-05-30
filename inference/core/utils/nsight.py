import os
import threading
from contextlib import contextmanager
from typing import Optional

_TRACE_CONTEXT = threading.local()
_NVTX = None
_NVTX_INIT_ATTEMPTED = False


def nsight_markers_enabled() -> bool:
    return os.getenv("RFDETR_NSIGHT_MARKERS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _get_nvtx():
    global _NVTX, _NVTX_INIT_ATTEMPTED
    if _NVTX_INIT_ATTEMPTED:
        return _NVTX
    _NVTX_INIT_ATTEMPTED = True
    try:
        import torch

        _NVTX = torch.cuda.nvtx
    except Exception:
        _NVTX = None
    return _NVTX


def nsight_mark(message: str) -> None:
    if not nsight_markers_enabled():
        return
    nvtx = _get_nvtx()
    if nvtx is None:
        return
    try:
        nvtx.mark(message)
    except Exception:
        return


def nsight_range_push(message: str) -> None:
    if not nsight_markers_enabled():
        return
    nvtx = _get_nvtx()
    if nvtx is None:
        return
    try:
        nvtx.range_push(message)
    except Exception:
        return


def nsight_range_pop() -> None:
    if not nsight_markers_enabled():
        return
    nvtx = _get_nvtx()
    if nvtx is None:
        return
    try:
        nvtx.range_pop()
    except Exception:
        return


@contextmanager
def nsight_range(message: str):
    nsight_range_push(message)
    try:
        yield
    finally:
        nsight_range_pop()


def nsight_current_frame_id() -> Optional[str]:
    return getattr(_TRACE_CONTEXT, "frame_id", None)


@contextmanager
def nsight_frame_context(frame_id: Optional[str]):
    previous = getattr(_TRACE_CONTEXT, "frame_id", None)
    _TRACE_CONTEXT.frame_id = frame_id
    try:
        yield
    finally:
        _TRACE_CONTEXT.frame_id = previous


def nsight_frame_label(frame_id: Optional[str], event: str) -> str:
    if frame_id is None:
        return f"rfdetr.{event}"
    return f"rfdetr.frame={frame_id}.{event}"
