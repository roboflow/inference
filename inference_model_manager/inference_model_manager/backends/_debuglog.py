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


def mem_split_mb() -> Tuple[float, float, float]:
    """(RssAnon, RssShmem, VmLck) MB from /proc/self/status.

    RssAnon = private heap (true leak measure); RssShmem = shared SHM-pool
    pages first-touched by this process (counted per-process, one physical
    copy); VmLck = page-locked (CUDA-pinned) host memory.
    """
    anon = shmem = lck = -1.0
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("RssAnon:"):
                    anon = int(line.split()[1]) / 1e3
                elif line.startswith("RssShmem:"):
                    shmem = int(line.split()[1]) / 1e3
                elif line.startswith("VmLck:"):
                    lck = int(line.split()[1]) / 1e3
    except Exception:
        pass
    return anon, shmem, lck


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
        "DBG decode batch: pid=%d n=%d jpeg=%d compressed_mb=%.1f sof_dims=%s "
        "est_decoded_mb=%.0f rss_mb=%.0f free_vram_mb=%.0f",
        os.getpid(),
        len(mvs),
        len(jpeg_idx),
        sum(len(m) for m in mvs) / 1e6,
        dims,
        est_bytes / 1e6,
        rss_mb(),
        free_vram_mb(),
    )


def decode_batch_failure() -> None:
    """Call from inside an except block: exception type + free VRAM at failure."""
    exc = sys.exc_info()[1]
    msg = str(exc).splitlines()[0] if exc is not None and str(exc) else ""
    log.info(
        "DBG nvjpeg batch failure: pid=%d %s: %s | rss_mb=%.0f free_vram_mb=%.0f",
        os.getpid(),
        type(exc).__name__ if exc is not None else "?",
        msg,
        rss_mb(),
        free_vram_mb(),
    )


def fallback_failure(slot_index: int, mv: Any) -> None:
    log.info(
        "DBG cpu fallback failure: pid=%d slot_index=%d bytes=%d sof_dims=%s "
        "rss_mb=%.0f free_vram_mb=%.0f",
        os.getpid(),
        slot_index,
        len(mv),
        jpeg_sof_dims(mv),
        rss_mb(),
        free_vram_mb(),
    )


def stage(name: str) -> None:
    """RSS/VRAM snapshot at a pipeline stage boundary — attributes leak growth."""
    alloc, reserved = cuda_mem_mb()
    anon, shmem, lck = mem_split_mb()
    log.info(
        "DBG stage %s: pid=%d rss_mb=%.0f anon_mb=%.0f shm_mb=%.0f locked_mb=%.0f "
        "cuda_alloc_mb=%.0f cuda_reserved_mb=%.0f free_vram_mb=%.0f",
        name,
        os.getpid(),
        rss_mb(),
        anon,
        shmem,
        lck,
        alloc,
        reserved,
        free_vram_mb(),
    )


def capture_slots(
    pool: Any, batch: List[tuple], mvs: List[Any]
) -> List[Tuple[int, int, int, int]]:
    """Record (header_req_id, status, input_size, crc32 of first 64 KiB) per slot.

    NOTE: header request_id is the client's ALLOC req_id; the batch carries the
    client's SUBMIT req_id — different id spaces, never compare across them.
    """
    state = []
    for (slot_id, _, _), mv in zip(batch, mvs):
        try:
            hdr = pool.read_header(slot_id)
            state.append(
                (
                    hdr.request_id,
                    hdr.status,
                    len(mv),
                    zlib.crc32(mv[:65536]) if len(mv) else 0,
                )
            )
        except Exception:
            state.append((0, -1, len(mv), 0))
    alloc, reserved = cuda_mem_mb()
    log.info(
        "DBG worker batch start: pid=%d n=%d slots=%s req_ids=%s sizes=%s rss_mb=%.0f "
        "cuda_alloc_mb=%.0f cuda_reserved_mb=%.0f free_vram_mb=%.0f",
        os.getpid(),
        len(batch),
        [b[0] for b in batch],
        [b[1] for b in batch],
        [s for _, _, s, _ in state],
        rss_mb(),
        alloc,
        reserved,
        free_vram_mb(),
    )
    return state


def check_slots(
    pool: Any,
    batch: List[tuple],
    mvs: List[Any],
    state: List[Tuple[int, int, int, int]],
) -> None:
    """Detect slot mutation between capture_slots and now (torn read evidence).

    Compares header-at-start vs header-now — same id space, immune to the
    alloc-vs-submit req_id split.
    """
    for i, (slot_id, req_id, _) in enumerate(batch):
        try:
            hdr = pool.read_header(slot_id)
        except Exception:
            continue
        hdr_req0, status0, size0, crc0 = state[i]
        if hdr.request_id != hdr_req0 or hdr.input_size != size0:
            log.error(
                "DBG TORN SLOT: pid=%d slot=%d batch_req_id=%d hdr_req_id %d->%d "
                "input_size %d->%d status %d->%d",
                os.getpid(),
                slot_id,
                req_id,
                hdr_req0,
                hdr.request_id,
                size0,
                hdr.input_size,
                status0,
                hdr.status,
            )
        elif len(mvs[i]) and zlib.crc32(mvs[i][:65536]) != crc0:
            log.error(
                "DBG TORN SLOT: pid=%d slot=%d batch_req_id=%d payload crc changed "
                "mid-batch",
                os.getpid(),
                slot_id,
                req_id,
            )


def batch_done(n: int, t0: float) -> None:
    alloc, reserved = cuda_mem_mb()
    anon, shmem, lck = mem_split_mb()
    log.info(
        "DBG worker batch done: pid=%d n=%d t_ms=%.0f rss_mb=%.0f anon_mb=%.0f "
        "shm_mb=%.0f locked_mb=%.0f cuda_alloc_mb=%.0f cuda_reserved_mb=%.0f "
        "free_vram_mb=%.0f",
        os.getpid(),
        n,
        (time.monotonic() - t0) * 1000,
        rss_mb(),
        anon,
        shmem,
        lck,
        alloc,
        reserved,
        free_vram_mb(),
    )


def slot_freed(reason: str, slot_id: int, req_id: Any) -> None:
    """Attribute every MMP free_slot call site."""
    log.info("DBG MMP free: reason=%s slot=%d req_id=%s", reason, slot_id, req_id)


def client_mem(pending: int) -> None:
    """Sampled memory snapshot for the uvicorn-side MMP client process."""
    anon, shmem, _ = mem_split_mb()
    log.info(
        "DBG client mem: pid=%d rss_mb=%.0f anon_mb=%.0f shm_mb=%.0f pending=%d",
        os.getpid(),
        rss_mb(),
        anon,
        shmem,
        pending,
    )
