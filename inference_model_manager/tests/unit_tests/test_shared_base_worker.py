from inference_model_manager.backends.shared_base_protocol import HeadIndexRegistry
from inference_model_manager.backends.shared_base_worker import (
    group_pending_by_head,
    handle_drop_head,
    handle_load_head,
)


def test_group_pending_routes_known_heads():
    reg = HeadIndexRegistry()
    reg.add("a", object())  # index 0
    reg.add("b", object())  # index 1
    pending = [
        (10, 100, 0, b"{}"),
        (11, 101, 1, b"{}"),
        (12, 102, 0, b"{}"),
    ]

    groups, unknown = group_pending_by_head(pending, reg)

    assert groups == {0: [(10, 100, b"{}"), (12, 102, b"{}")], 1: [(11, 101, b"{}")]}
    assert unknown == []


def test_group_pending_collects_unknown_head_index():
    reg = HeadIndexRegistry()
    reg.add("a", object())  # index 0
    reg.remove("a")  # index 0 retired
    pending = [(10, 100, 0, b"{}"), (11, 101, 7, b"{}")]

    groups, unknown = group_pending_by_head(pending, reg)

    assert groups == {}
    # both the retired index and the never-seen index are rejected, not misrouted.
    assert sorted(unknown) == [(10, 100), (11, 101)]


def test_handle_load_head_registers_and_acks():
    reg = HeadIndexRegistry()
    model = object()

    def load_fn(payload):
        return model, {"model_mro_names": ["H"], "max_batch_size": 8, "class_names": None}

    ack = handle_load_head(
        {"req_id": 1, "head_id": "h1", "api_key": "k"}, reg, load_fn
    )

    assert ack == {
        "req_id": 1,
        "ok": True,
        "head_index": 0,
        "model_mro_names": ["H"],
        "max_batch_size": 8,
        "class_names": None,
    }
    assert reg.get(0) is model


def test_handle_load_head_failure_is_isolated():
    reg = HeadIndexRegistry()
    reg.add("existing", object())

    def load_fn(payload):
        raise RuntimeError("CUDA out of memory")

    ack = handle_load_head({"req_id": 2, "head_id": "h2"}, reg, load_fn)

    assert ack["ok"] is False
    assert "CUDA out of memory" in ack["error"]
    # base/other heads untouched: failed head not registered, existing head intact.
    assert "h2" not in reg
    assert "existing" in reg


def test_handle_load_head_rejects_duplicate():
    reg = HeadIndexRegistry()
    reg.add("h1", object())

    ack = handle_load_head(
        {"req_id": 3, "head_id": "h1"}, reg, lambda p: (object(), {})
    )

    assert ack["ok"] is False
    assert "already loaded" in ack["error"]


def test_handle_drop_head_removes_then_acks():
    reg = HeadIndexRegistry()
    reg.add("h1", object())
    removed = []

    ack = handle_drop_head(
        {"req_id": 4, "head_id": "h1"}, reg, on_removed=lambda idx, hid: removed.append((idx, hid))
    )

    assert ack == {"req_id": 4, "ok": True}
    assert "h1" not in reg
    assert removed == [(0, "h1")]


def test_handle_drop_unknown_head_acks_ok():
    reg = HeadIndexRegistry()
    ack = handle_drop_head({"req_id": 5, "head_id": "ghost"}, reg)
    assert ack == {"req_id": 5, "ok": True}
