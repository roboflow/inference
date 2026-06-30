import pytest

from inference_model_manager.backends.shared_base_protocol import (
    HEAD_SLOT_SIZE,
    HeadIndexRegistry,
    decode_control,
    encode_control,
    pack_head_slot,
    unpack_head_slot,
)


def test_head_slot_roundtrip():
    buf = pack_head_slot(slot_id=7, req_id=123456789, head_index=3)
    assert len(buf) == HEAD_SLOT_SIZE == 14
    assert unpack_head_slot(buf) == (7, 123456789, 3)


def test_unpack_ignores_trailing_bytes():
    buf = pack_head_slot(1, 2, 4) + b"junk"
    assert unpack_head_slot(buf) == (1, 2, 4)


def test_control_roundtrip():
    payload = encode_control(99, head_id="head/1", api_key="k", device="cuda:0")
    decoded = decode_control(payload)
    assert decoded == {
        "req_id": 99,
        "head_id": "head/1",
        "api_key": "k",
        "device": "cuda:0",
    }


def test_registry_add_assigns_monotonic_index():
    reg = HeadIndexRegistry()
    assert reg.add("a", object()) == 0
    assert reg.add("b", object()) == 1
    assert len(reg) == 2


def test_registry_get_and_head_id_for():
    reg = HeadIndexRegistry()
    head = object()
    idx = reg.add("a", head)
    assert reg.get(idx) is head
    assert reg.head_id_for(idx) == "a"
    assert "a" in reg


def test_registry_rejects_duplicate_add():
    reg = HeadIndexRegistry()
    reg.add("a", object())
    with pytest.raises(ValueError):
        reg.add("a", object())


def test_registry_unknown_index_returns_none():
    reg = HeadIndexRegistry()
    assert reg.get(42) is None
    assert reg.head_id_for(42) is None


def test_registry_removed_index_is_retired_not_recycled():
    reg = HeadIndexRegistry()
    idx_a = reg.add("a", object())
    reg.add("b", object())
    assert reg.remove("a") == idx_a
    assert "a" not in reg
    # stale slot still carrying idx_a resolves to nothing — rejected, not misrouted.
    assert reg.get(idx_a) is None
    # next head gets a fresh index, never the retired one.
    idx_c = reg.add("c", object())
    assert idx_c == 2
    assert idx_c != idx_a


def test_registry_remove_unknown_returns_none():
    reg = HeadIndexRegistry()
    assert reg.remove("ghost") is None
