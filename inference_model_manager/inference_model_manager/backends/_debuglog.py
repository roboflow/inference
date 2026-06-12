"""TEMPORARY diagnostic logging for the OOM / slot-race investigation (ISSUES.md).

To remove: delete this file and every line tagged `# DEBUGLOG`:
    grep -rn DEBUGLOG inference_model_manager/

All messages go to the 'inference_model_manager.debuglog' logger and start
with 'DBG' so they are easy to grep in server logs.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import zlib
from typing import Any, List, Optional, Tuple

log = logging.getLogger("inference_model_manager.debuglog")

_PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def rss_mb() -> float:
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * _PAGE / 1e6
    except Exception:
        return -1.0


def free_vram_mb() -> float:
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return -1.0
        free, _ = torch.cuda.mem_get_info()
        return free / 1e6
    except Exception:
        return -1.0


def cuda_mem_mb() -> Tuple[float, float]:
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return -1.0, -1.0
        return (
            torch.cuda.memory_allocated() / 1e6,
            torch.cuda.memory_reserved() / 1e6,
        )
    except Exception:
        return -1.0, -1.0


def jpeg_sof_dims(data: Any) -> Optional[Tuple[int, int, int]]:
    """(height, width, n_components) from a JPEG SOF header, no decode.

    Scans the first 64 KiB only. None if no SOF marker found.
    """
    buf = bytes(data[:65536])
    i = 2
    while i + 9 < len(buf):
        if buf[i] != 0xFF:
            i += 1
            continue
        marker = buf[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            h = int.from_bytes(buf[i + 5 : i + 7], "big")
            w = int.from_bytes(buf[i + 7 : i + 9], "big")
            return h, w, buf[i + 9]
        i += 2 + int.from_bytes(buf[i + 2 : i + 4], "big")
    return None


def decode_batch_entry(mvs: List[Any], jpeg_idx: List[int]) -> None:
    """One line per decode batch: sizes, SOF dims, estimated decoded MB, free VRAM."""
    dims = []
    est_bytes = 0
    for i in jpeg_idx:
        d = jpeg_sof_dims(mvs[i])
        dims.append(d)
        if d:
            est_bytes += d[0] * d[1] * 3
    log.info(
        "DBG decode batch: n=%d jpeg=%d compressed_mb=%.1f sof_dims=%s "
        "est_decoded_mb=%.0f free_vram_mb=%.0f",
        len(mvs),
        len(jpeg_idx),
        sum(len(m) for m in mvs) / 1e6,
        dims,
        est_bytes / 1e6,
        free_vram_mb(),
    )


def decode_batch_failure() -> None:
    """Call from inside an except block: exception type + free VRAM at failure."""
    exc = sys.exc_info()[1]
    msg = str(exc).splitlines()[0] if exc is not None and str(exc) else ""
    log.info(
        "DBG nvjpeg batch failure: %s: %s | free_vram_mb=%.0f",
        type(exc).__name__ if exc is not None else "?",
        msg,
        free_vram_mb(),
    )


def fallback_failure(slot_index: int, mv: Any) -> None:
    log.info(
        "DBG cpu fallback failure: slot_index=%d bytes=%d sof_dims=%s free_vram_mb=%.0f",
        slot_index,
        len(mv),
        jpeg_sof_dims(mv),
        free_vram_mb(),
    )


def capture_slots(batch: List[tuple], mvs: List[Any]) -> List[Tuple[int, int]]:
    """Record (input_size, crc32 of first 64 KiB) per slot + a batch-start line."""
    state = [(len(mv), zlib.crc32(mv[:65536]) if len(mv) else 0) for mv in mvs]
    alloc, reserved = cuda_mem_mb()
    log.info(
        "DBG worker batch start: n=%d slots=%s req_ids=%s sizes=%s rss_mb=%.0f "
        "cuda_alloc_mb=%.0f cuda_reserved_mb=%.0f free_vram_mb=%.0f",
        len(batch),
        [b[0] for b in batch],
        [b[1] for b in batch],
        [s for s, _ in state],
        rss_mb(),
        alloc,
        reserved,
        free_vram_mb(),
    )
    return state


def check_slots(
    pool: Any, batch: List[tuple], mvs: List[Any], state: List[Tuple[int, int]]
) -> None:
    """Detect slot mutation between capture_slots and now (torn read evidence)."""
    for i, (slot_id, req_id, _) in enumerate(batch):
        try:
            hdr = pool.read_header(slot_id)
        except Exception:
            continue
        size0, crc0 = state[i]
        if hdr.request_id != req_id or hdr.input_size != size0:
            log.error(
                "DBG TORN SLOT: slot=%d req_id %d->%d input_size %d->%d status=%d",
                slot_id,
                req_id,
                hdr.request_id,
                size0,
                hdr.input_size,
                hdr.status,
            )
        elif len(mvs[i]) and zlib.crc32(mvs[i][:65536]) != crc0:
            log.error(
                "DBG TORN SLOT: slot=%d req_id=%d payload crc changed mid-batch",
                slot_id,
                req_id,
            )


def batch_done(n: int, t0: float) -> None:
    alloc, reserved = cuda_mem_mb()
    log.info(
        "DBG worker batch done: n=%d t_ms=%.0f rss_mb=%.0f cuda_alloc_mb=%.0f "
        "cuda_reserved_mb=%.0f free_vram_mb=%.0f",
        n,
        (time.monotonic() - t0) * 1000,
        rss_mb(),
        alloc,
        reserved,
        free_vram_mb(),
    )
