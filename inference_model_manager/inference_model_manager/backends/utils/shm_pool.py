"""Fixed-size SHM slot pool with typed 64-byte headers.

Created by ModelManager (in-process) or ModelManagerProcess (orchestrated).
FastAPI workers and backend workers attach by name and access slot memory
directly — zero copy on the data path.

Slot layout (per slot):
    [HEADER 64B | DATA data_slot_bytes]

DATA area is shared: input written first (image bytes), then overwritten with
result (pickled predictions) after inference. Input is dead after decode — the
worker holds GPU tensors at that point. This halves SHM usage vs separate
INPUT + RESULT areas.

Header layout (little-endian, 64 bytes total):
    offset  size  field
       0       1  status      (SlotStatus)
       1       1  error_code
       2       2  _pad
       4       4  input_size  (bytes written to data area as input)
       8       4  result_size (bytes written to data area as result)
      12       4  _pad
      16       8  request_id  (UUID int — never use slot_id as key, slots recycled)
      24       8  timestamp_ns (monotonic_ns at alloc — stale detection)
      32       4  owner_pid
      36      28  _pad
"""

from __future__ import annotations

import struct
import threading
import time
from collections import deque
from enum import IntEnum
from multiprocessing.shared_memory import SharedMemory
from typing import NamedTuple, Optional

# ---------------------------------------------------------------------------
# Header format
# ---------------------------------------------------------------------------

_HEADER_FMT = "<BBxxIIxxxxQQi28x"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
assert _HEADER_SIZE == 64, f"Header must be 64 bytes, got {_HEADER_SIZE}"

# Field byte offsets within a slot header (for fast partial writes)
_OFF_STATUS = 0
_OFF_ERROR = 1
_OFF_INPUT_SZ = 4
_OFF_RESULT_SZ = 8
_OFF_REQ_ID = 16
_OFF_TS_NS = 24
_OFF_OWNER_PID = 32


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class SlotStatus(IntEnum):
    FREE = 0
    ALLOCATED = 1
    WRITTEN = 2
    PROCESSING = 3
    DONE = 4
    ERROR = 5


class SlotHeader(NamedTuple):
    status: int
    error_code: int
    input_size: int
    result_size: int
    request_id: int
    timestamp_ns: int
    owner_pid: int


# ---------------------------------------------------------------------------
# SHMPool
# ---------------------------------------------------------------------------


class SHMPool:
    """Fixed pool of SHM slots. Only the creator (owner) manages allocation.

    Usage — creator (ModelManager / MMP)::

        pool = SHMPool.create(n_slots=256, input_mb=20)
        # share pool.name with workers via env / config
        slot_id = pool.alloc_slot()
        pool.mark_allocated(slot_id, request_id=req_id)
        pool.data_memoryview(slot_id)[:n] = image_bytes
        pool.mark_written(slot_id, n)
        # ... inference done (result overwrites same data area) ...
        result = bytes(pool.data_memoryview(slot_id)[:result_sz])
        pool.free_slot(slot_id)

    Usage — worker / FastAPI (attached, read-only allocation)::

        pool = SHMPool.attach("inference_pool_abc123", n_slots=256,
                              input_mb=20)
        view = pool.data_memoryview(slot_id)   # zero-copy
        pool.close()  # does NOT unlink
    """

    def __init__(
        self,
        shm: SharedMemory,
        n_slots: int,
        data_slot_bytes: int,
        *,
        owner: bool,
    ) -> None:
        self._shm = shm
        self._n_slots = n_slots
        self._data_slot_bytes = data_slot_bytes
        self._owner = owner
        self._slot_bytes = _HEADER_SIZE + data_slot_bytes

        # Allocation state — only owner ever calls alloc/free
        self._free: deque[int] = deque(range(n_slots)) if owner else deque()
        self._allocated: set[int] = (
            set()
        )  # slots currently in use (for double-free guard)
        self._cond: threading.Condition = threading.Condition(threading.Lock())

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        n_slots: int,
        input_mb: float,
        *,
        name: Optional[str] = None,
    ) -> "SHMPool":
        """Create a new SHM pool. Caller is the owner and must call close()."""
        data_bytes = int(input_mb * 1024 * 1024)
        total_bytes = n_slots * (_HEADER_SIZE + data_bytes)

        shm = SharedMemory(name=name, create=True, size=total_bytes)
        # Zero only the headers (64B × n_slots) so status=FREE everywhere.
        slot_bytes = _HEADER_SIZE + data_bytes
        for i in range(n_slots):
            shm.buf[i * slot_bytes : i * slot_bytes + _HEADER_SIZE] = (
                b"\x00" * _HEADER_SIZE
            )

        return cls(shm, n_slots, data_bytes, owner=True)

    @classmethod
    def attach(
        cls,
        name: str,
        n_slots: int,
        input_mb: float,
    ) -> "SHMPool":
        """Attach to an existing pool. Does NOT unlink on close()."""
        data_bytes = int(input_mb * 1024 * 1024)
        shm = SharedMemory(name=name, create=False)
        try:
            return cls(shm, n_slots, data_bytes, owner=False)
        except Exception:
            shm.close()
            raise

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """SHM block name — share this with workers so they can attach."""
        return self._shm.name

    @property
    def n_slots(self) -> int:
        return self._n_slots

    @property
    def slot_bytes(self) -> int:
        """Total bytes per slot (header + data area)."""
        return self._slot_bytes

    @property
    def data_slot_bytes(self) -> int:
        """Bytes available per slot for input or result data."""
        return self._data_slot_bytes

    @property
    def free_count(self) -> int:
        return len(self._free)

    # ------------------------------------------------------------------
    # Slot allocation (owner only)
    # ------------------------------------------------------------------

    def alloc_slot(self, timeout: float = 2.0) -> int:
        """Pop a free slot. Blocks until one is available or timeout expires.

        Only the pool creator (owner) should call this.

        Returns:
            slot_id (0-based index).

        Raises:
            RuntimeError: If called on an attached (non-owner) pool.
            TimeoutError: If no slot is available within *timeout* seconds.
        """
        if not self._owner:
            raise RuntimeError("Only the pool creator can allocate slots")
        with self._cond:
            deadline = time.monotonic() + timeout
            while not self._free:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._cond.wait(timeout=remaining):
                    raise TimeoutError(f"No free SHM slots (pool size={self._n_slots})")
            slot_id = self._free.popleft()
            self._allocated.add(slot_id)
            return slot_id

    def free_slot(self, slot_id: int) -> None:
        """Return slot to the pool and zero its header.

        Only the pool creator (owner) should call this.
        Thread-safe: lock protects both header write and free-list append.
        """
        if not self._owner:
            raise RuntimeError("Only the pool creator can free slots")
        with self._cond:
            if slot_id not in self._allocated:
                return  # double-free guard — silently ignore
            self._allocated.discard(slot_id)
            # Zero header inside lock so no reader sees partial state
            off = self._slot_offset(slot_id)
            struct.pack_into(
                _HEADER_FMT, self._shm.buf, off, SlotStatus.FREE, 0, 0, 0, 0, 0, 0
            )
            self._free.append(slot_id)
            self._cond.notify()

    # ------------------------------------------------------------------
    # Header access
    # ------------------------------------------------------------------

    def read_header(self, slot_id: int) -> SlotHeader:
        """Read all header fields. Non-blocking, no lock."""
        off = self._slot_offset(slot_id)
        status, error_code, input_size, result_size, request_id, ts_ns, pid = (
            struct.unpack_from(_HEADER_FMT, self._shm.buf, off)
        )
        return SlotHeader(
            status=status,
            error_code=error_code,
            input_size=input_size,
            result_size=result_size,
            request_id=request_id,
            timestamp_ns=ts_ns,
            owner_pid=pid,
        )

    def mark_allocated(self, slot_id: int, request_id: int) -> None:
        """Write full header: status=ALLOCATED, req_id, timestamp."""
        off = self._slot_offset(slot_id)
        struct.pack_into(
            _HEADER_FMT,
            self._shm.buf,
            off,
            SlotStatus.ALLOCATED,
            0,
            0,
            0,
            request_id,
            time.monotonic_ns(),
            0,
        )

    def mark_written(self, slot_id: int, input_size: int) -> None:
        """Fast update: status=WRITTEN + input_size. Data is already in slot.
        Size written first so cross-process readers never see WRITTEN with stale size.
        """
        off = self._slot_offset(slot_id)
        struct.pack_into("<I", self._shm.buf, off + _OFF_INPUT_SZ, input_size)
        self._shm.buf[off + _OFF_STATUS] = SlotStatus.WRITTEN

    def mark_processing(self, slot_id: int, owner_pid: int) -> None:
        """Worker sets status=PROCESSING and records its pid."""
        off = self._slot_offset(slot_id)
        self._shm.buf[off + _OFF_STATUS] = SlotStatus.PROCESSING
        struct.pack_into("<i", self._shm.buf, off + _OFF_OWNER_PID, owner_pid)

    def mark_done(self, slot_id: int, result_size: int) -> None:
        """Worker sets status=DONE + result_size after writing result area.
        Size written first so cross-process readers never see DONE with stale size.
        """
        off = self._slot_offset(slot_id)
        struct.pack_into("<I", self._shm.buf, off + _OFF_RESULT_SZ, result_size)
        self._shm.buf[off + _OFF_STATUS] = SlotStatus.DONE

    def mark_error(
        self, slot_id: int, error_code: int = 1, error_size: int = 0
    ) -> None:
        """Set status=ERROR + error_code. error_size = bytes of error detail in DATA area."""
        off = self._slot_offset(slot_id)
        struct.pack_into("<I", self._shm.buf, off + _OFF_RESULT_SZ, error_size)
        self._shm.buf[off + _OFF_STATUS] = SlotStatus.ERROR
        self._shm.buf[off + _OFF_ERROR] = error_code

    # ------------------------------------------------------------------
    # Data area access (zero-copy)
    # ------------------------------------------------------------------

    def data_memoryview(self, slot_id: int) -> memoryview:
        """Zero-copy view of the data area for this slot.

        Used for both input (write image bytes) and result (write pickled
        predictions). Input is overwritten by result after inference.
        """
        start = self._slot_offset(slot_id) + _HEADER_SIZE
        return self._shm.buf[start : start + self._data_slot_bytes]

    # ------------------------------------------------------------------
    # Stale slot detection
    # ------------------------------------------------------------------

    def stale_slots(self, max_age_s: float = 30.0) -> list[int]:
        """Slot IDs that have been non-FREE longer than max_age_s.

        Used by MMP's reaper to reclaim orphaned slots when a FastAPI worker
        or backend worker crashes without sending T_FREE.
        Only returns slots that are actually allocated (not in free list).
        """
        now_ns = time.monotonic_ns()
        max_ns = int(max_age_s * 1_000_000_000)
        with self._cond:
            allocated = set(self._allocated)
        result = []
        for slot_id in allocated:
            hdr = self.read_header(slot_id)
            if (
                hdr.status != SlotStatus.FREE
                and hdr.timestamp_ns > 0
                and now_ns - hdr.timestamp_ns > max_ns
            ):
                result.append(slot_id)
        return result

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Detach from SHM. Owner also unlinks (destroys) the block. Idempotent."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self._shm.close()
        if self._owner:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _slot_offset(self, slot_id: int) -> int:
        if slot_id < 0 or slot_id >= self._n_slots:
            raise IndexError(f"slot_id {slot_id} out of range [0, {self._n_slots})")
        return slot_id * self._slot_bytes
