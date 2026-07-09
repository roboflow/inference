import msgpack
import pytest

from inference.core.workflows.execution_engine.v1.dynamic_blocks.modal_executor import (
    _WS_MAX_FRAME_BYTES,
    WebSocketModalExecutor,
    _split_ws_frames,
)


class _FakeWS:
    def __init__(self, frames):
        self._frames = list(frames)

    def recv(self):
        return self._frames.pop(0)


def test_small_frame_is_not_chunked() -> None:
    payload = msgpack.packb({"result": "ok"}, use_bin_type=True)

    frames = _split_ws_frames(payload, msgpack)

    assert frames == [payload]


def test_large_frame_is_chunked_below_limit() -> None:
    payload = msgpack.packb(
        {"result": b"x" * (3 * _WS_MAX_FRAME_BYTES + 100)}, use_bin_type=True
    )

    frames = _split_ws_frames(payload, msgpack)

    head = msgpack.unpackb(frames[0], raw=False)
    assert head == {"_chunked": len(frames) - 1}
    assert all(len(frame) <= _WS_MAX_FRAME_BYTES for frame in frames)
    assert b"".join(frames[1:]) == payload


@pytest.mark.parametrize("size", [10, _WS_MAX_FRAME_BYTES, 5 * _WS_MAX_FRAME_BYTES])
def test_recv_reassembled_round_trip(size: int) -> None:
    payload = msgpack.packb({"result": b"x" * size}, use_bin_type=True)
    executor = WebSocketModalExecutor(workspace_id="test")
    executor._ws = _FakeWS(_split_ws_frames(payload, msgpack))

    assert executor._recv_reassembled(msgpack) == payload


def test_recv_reassembled_leaves_small_responses_untouched() -> None:
    payload = msgpack.packb({"success": True}, use_bin_type=True)
    executor = WebSocketModalExecutor(workspace_id="test")
    executor._ws = _FakeWS([payload])

    assert executor._recv_reassembled(msgpack) == payload
