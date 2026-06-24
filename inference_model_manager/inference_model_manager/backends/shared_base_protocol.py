"""Wire protocol + head-index bookkeeping for the shared-base subprocess backend.

A shared-base worker hosts one base model (e.g. OWLv2) and many heads. The parent
and worker exchange head lifecycle over the existing ZMQ PAIR socket with dedicated
control tags, and route per-slot inference to a head via a head_index carried in the
slot frame. Only the shared-base backend speaks this protocol; plain SubprocessBackend
is unchanged.
"""

import json
import struct
from typing import Any, Dict, Optional, Tuple

# Control tags (parent↔worker). Disjoint from subproc.py's \x01-\x05.
MSG_LOAD_HEAD = b"\x10"  # parent→worker: JSON {req_id, head_id, api_key, device, dep_*}
MSG_LOAD_HEAD_ACK = b"\x11"  # worker→parent: JSON {req_id, ok, head_index|error, meta}
MSG_DROP_HEAD = b"\x12"  # parent→worker: JSON {req_id, head_id}
MSG_DROP_HEAD_ACK = b"\x13"  # worker→parent: JSON {req_id, ok, error?}
MSG_HEAD_SLOT_READY = b"\x14"  # parent→worker: struct(">IQH", slot_id, req_id, head_idx)

_HEAD_SLOT = struct.Struct(">IQH")  # 14 bytes
HEAD_SLOT_SIZE = _HEAD_SLOT.size


def pack_head_slot(slot_id: int, req_id: int, head_index: int) -> bytes:
    return _HEAD_SLOT.pack(slot_id, req_id, head_index)


def unpack_head_slot(buf: bytes) -> Tuple[int, int, int]:
    return _HEAD_SLOT.unpack(buf[:HEAD_SLOT_SIZE])


def encode_control(req_id: int, **fields: Any) -> bytes:
    return json.dumps({"req_id": req_id, **fields}).encode("utf-8")


def decode_control(payload: bytes) -> Dict[str, Any]:
    return json.loads(payload.decode("utf-8"))


class HeadIndexRegistry:
    """Owner-local head_id ↔ head_index ↔ head-object map, held in the worker.

    Indexes are monotonic and NEVER reused: a dropped head retires its index for
    good, so any in-flight slot still carrying that index resolves to nothing and is
    rejected rather than silently hitting a different head.
    """

    def __init__(self) -> None:
        self._index_by_id: Dict[str, int] = {}
        self._entry_by_index: Dict[int, Tuple[str, Any]] = {}
        self._next_index = 0

    def add(self, head_id: str, head: Any) -> int:
        if head_id in self._index_by_id:
            raise ValueError(f"head '{head_id}' already registered")
        index = self._next_index
        self._next_index += 1
        self._index_by_id[head_id] = index
        self._entry_by_index[index] = (head_id, head)
        return index

    def get(self, head_index: int) -> Optional[Any]:
        entry = self._entry_by_index.get(head_index)
        return entry[1] if entry is not None else None

    def head_id_for(self, head_index: int) -> Optional[str]:
        entry = self._entry_by_index.get(head_index)
        return entry[0] if entry is not None else None

    def remove(self, head_id: str) -> Optional[int]:
        index = self._index_by_id.pop(head_id, None)
        if index is None:
            return None
        self._entry_by_index.pop(index, None)
        return index

    def __contains__(self, head_id: str) -> bool:
        return head_id in self._index_by_id

    def __len__(self) -> int:
        return len(self._index_by_id)
