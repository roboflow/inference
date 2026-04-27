"""Unit tests for SubprocessBackend v2 helpers — _to_bytes, wire formats."""

from __future__ import annotations

import io
import pickle
import struct

import numpy as np
import pytest

from inference_model_manager.backends.subproc import (
    _MSG_RESULT,
    _MSG_SLOT_READY,
    _NP_MAGIC,
    _to_bytes,
)

# ---------------------------------------------------------------------------
# _to_bytes
# ---------------------------------------------------------------------------


class TestToBytes:
    def test_bytes_passthrough(self) -> None:
        data = b"\xff\xd8\xff"  # JPEG header
        assert _to_bytes(data) == data

    def test_bytearray_passthrough(self) -> None:
        data = bytearray(b"hello")
        result = _to_bytes(data)
        assert result == b"hello"
        assert isinstance(result, bytes)

    def test_memoryview_passthrough(self) -> None:
        buf = bytearray(b"world")
        result = _to_bytes(memoryview(buf))
        assert result == b"world"
        assert isinstance(result, bytes)

    def test_numpy_array_magic_prefix(self) -> None:
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        result = _to_bytes(arr)
        assert result[:6] == _NP_MAGIC, "must start with numpy .npy magic"
        # round-trip
        reloaded = np.load(io.BytesIO(result), allow_pickle=False)
        np.testing.assert_array_equal(reloaded, arr)

    def test_numpy_2d_array(self) -> None:
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        result = _to_bytes(arr)
        assert result[:6] == _NP_MAGIC
        reloaded = np.load(io.BytesIO(result), allow_pickle=False)
        np.testing.assert_array_equal(reloaded, arr)

    def test_arbitrary_object_pickle(self) -> None:
        obj = {"key": [1, 2, 3]}
        result = _to_bytes(obj)
        assert pickle.loads(result) == obj

    def test_none_pickle(self) -> None:
        result = _to_bytes(None)
        assert pickle.loads(result) is None


# ---------------------------------------------------------------------------
# Wire format: T_SLOT_READY  [msg_type][slot_id(4B big-endian) | req_id(8B big-endian)]
# ---------------------------------------------------------------------------


class TestSlotReadyFormat:
    def test_pack_unpack_roundtrip(self) -> None:
        slot_id = 7
        req_id = 0xDEAD_BEEF_CAFE_1234

        payload = struct.pack(">IQ", slot_id, req_id)
        assert len(payload) == 12, "T_SLOT_READY payload must be 12 bytes"

        s, r = struct.unpack(">IQ", payload)
        assert s == slot_id
        assert r == req_id

    def test_zero_values(self) -> None:
        payload = struct.pack(">IQ", 0, 0)
        s, r = struct.unpack(">IQ", payload)
        assert s == 0
        assert r == 0

    def test_max_values(self) -> None:
        slot_id = 0xFFFF_FFFF
        req_id = 0xFFFF_FFFF_FFFF_FFFF
        payload = struct.pack(">IQ", slot_id, req_id)
        s, r = struct.unpack(">IQ", payload)
        assert s == slot_id
        assert r == req_id

    def test_msg_type_constant(self) -> None:
        assert _MSG_SLOT_READY == b"\x01"


# ---------------------------------------------------------------------------
# Wire format: T_RESULT  [msg_type][req_id(8B) | slot_id(4B) | result_sz(4B)]
# ---------------------------------------------------------------------------


class TestResultFormat:
    def test_pack_unpack_roundtrip(self) -> None:
        req_id = 0xCAFE_BABE_0000_0001
        slot_id = 3
        result_sz = 1024

        payload = struct.pack(">QII", req_id, slot_id, result_sz)
        assert len(payload) == 16, "T_RESULT payload must be 16 bytes"

        r, s, sz = struct.unpack(">QII", payload)
        assert r == req_id
        assert s == slot_id
        assert sz == result_sz

    def test_zero_result_sz_signals_error(self) -> None:
        # result_sz=0 means worker inference failed
        payload = struct.pack(">QII", 42, 1, 0)
        _, _, sz = struct.unpack(">QII", payload)
        assert sz == 0

    def test_msg_type_constant(self) -> None:
        assert _MSG_RESULT == b"\x02"


# ---------------------------------------------------------------------------
# Magic byte constant
# ---------------------------------------------------------------------------


class TestNumpyMagic:
    def test_magic_matches_npy_format(self) -> None:
        arr = np.array([1.0])
        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        assert buf.getvalue()[:6] == _NP_MAGIC

    def test_raw_jpeg_header_not_magic(self) -> None:
        jpeg_start = b"\xff\xd8\xff\xe0\x00\x10"
        assert jpeg_start[:6] != _NP_MAGIC

    def test_raw_png_header_not_magic(self) -> None:
        png_start = b"\x89PNG\r\n\x1a\n"
        assert png_start[:6] != _NP_MAGIC
