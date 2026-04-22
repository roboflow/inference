"""Shared-memory serializer for the SubprocessBackend data path.

Extracts ``np.ndarray`` and ``torch.Tensor`` objects from an arbitrary Python
structure, writes their raw bytes into a pre-allocated shared memory buffer,
and replaces them with lightweight :class:`_ShmRef` markers.  The modified
structure (now tiny — no bulk data) is pickled normally.

On the receiving side, :func:`unpack` reverses the process: unpickles the
marker-ified object, walks it to find :class:`_ShmRef` instances, and
reconstructs the original arrays / tensors from SHM.

Everything that is *not* an ndarray or Tensor is handled by pickle as usual.
"""

from __future__ import annotations

import dataclasses
import pickle
from typing import Any

import numpy as np

try:
    import torch as _torch
except ImportError:
    _torch = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Marker — survives pickle round-trip
# ---------------------------------------------------------------------------


class _ShmRef:
    """Placeholder for an array whose data lives in shared memory."""

    __slots__ = ("index",)

    def __init__(self, index: int) -> None:
        self.index = index


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pack(
    obj: Any,
    shm_buf: memoryview,
    offset: int = 0,
) -> tuple[list[dict], bytes, int]:
    """Extract arrays from *obj*, write them to *shm_buf*, pickle the rest.

    Returns
    -------
    (descriptors, pickled, shm_bytes_written)
        *descriptors* — list of dicts, one per extracted array, each with
        keys ``d`` (dtype str), ``s`` (shape list), ``o`` (byte offset),
        ``n`` (nbytes), ``T`` (bool, True if originally a torch.Tensor).

        *pickled* — the structure with arrays replaced by :class:`_ShmRef`,
        serialized via :func:`pickle.dumps`.

        *shm_bytes_written* — total bytes written to *shm_buf*.
    """
    descriptors: list[dict] = []
    current_offset = offset
    buf_len = len(shm_buf)

    def _write_array(arr: np.ndarray, *, is_tensor: bool) -> _ShmRef:
        nonlocal current_offset
        arr = np.ascontiguousarray(arr)
        # Align offset to dtype's alignment requirement
        alignment = arr.dtype.alignment
        if alignment > 1:
            current_offset = (current_offset + alignment - 1) & ~(alignment - 1)
        nbytes = arr.nbytes
        if current_offset + nbytes > buf_len:
            raise ValueError(
                f"Data ({current_offset + nbytes} bytes needed) exceeds "
                f"shared memory buffer ({buf_len} bytes)"
            )
        target = np.ndarray(
            arr.shape,
            dtype=arr.dtype,
            buffer=shm_buf,
            offset=current_offset,
        )
        np.copyto(target, arr)
        idx = len(descriptors)
        descriptors.append(
            {
                "d": str(arr.dtype),
                "s": list(arr.shape),
                "o": current_offset,
                "n": nbytes,
                "T": is_tensor,
            }
        )
        current_offset += nbytes
        return _ShmRef(idx)

    def _replace(o: Any) -> Any:
        if _torch is not None and isinstance(o, _torch.Tensor):
            if o.is_cuda:
                o = o.cpu()
            return _write_array(o.detach().numpy(), is_tensor=True)

        if isinstance(o, np.ndarray):
            return _write_array(o, is_tensor=False)

        # Walk standard containers to find nested arrays
        if isinstance(o, list):
            return [_replace(x) for x in o]
        if isinstance(o, tuple):
            return tuple(_replace(x) for x in o)
        if isinstance(o, dict):
            return {k: _replace(v) for k, v in o.items()}
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            fields = {
                f.name: _replace(getattr(o, f.name)) for f in dataclasses.fields(o)
            }
            return type(o)(**fields)

        # Everything else — left as-is for pickle
        return o

    modified = _replace(obj)
    pickled = pickle.dumps(modified, protocol=pickle.HIGHEST_PROTOCOL)
    return descriptors, pickled, current_offset - offset


def unpack(
    descriptors: list[dict],
    pickled: bytes,
    shm_buf: memoryview,
    *,
    copy: bool = True,
) -> Any:
    """Reconstruct the original object from pickled remainder + SHM arrays.

    Parameters
    ----------
    descriptors:
        Array descriptors produced by :func:`pack`.
    pickled:
        Pickled object with :class:`_ShmRef` markers.
    shm_buf:
        Readable memoryview backed by the same shared memory written by
        :func:`pack`.
    copy:
        If ``True`` (default), array data is copied out of *shm_buf* so the
        caller owns the memory.  ``False`` yields zero-copy views — only
        safe when *shm_buf* will not be overwritten before the caller is
        done with the data.
    """
    obj = pickle.loads(pickled)

    def _restore(o: Any) -> Any:
        if isinstance(o, _ShmRef):
            desc = descriptors[o.index]
            arr = np.ndarray(
                shape=desc["s"],
                dtype=np.dtype(desc["d"]),
                buffer=shm_buf,
                offset=desc["o"],
            )
            if copy:
                arr = arr.copy()
            if desc["T"] and _torch is not None:
                return _torch.from_numpy(arr)
            return arr

        if isinstance(o, list):
            return [_restore(x) for x in o]
        if isinstance(o, tuple):
            return tuple(_restore(x) for x in o)
        if isinstance(o, dict):
            return {k: _restore(v) for k, v in o.items()}
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            fields = {
                f.name: _restore(getattr(o, f.name)) for f in dataclasses.fields(o)
            }
            return type(o)(**fields)

        return o

    return _restore(obj)
