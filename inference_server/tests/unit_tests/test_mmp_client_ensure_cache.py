"""ensure_loaded warm-cache must be per-(routing_key, api_key).

Regression: the client _loaded_cache was keyed by routing_key alone, so within
the TTL a request from tenant B reused tenant A's warm cache and skipped
T_ENSURE_LOADED. MMP then never recorded B as an owner and tenant-scoped
/v2/server/metrics hid the model from B, a real user of it.
"""

from __future__ import annotations

import asyncio
import struct

from inference_server.proxies.mmp_client import T_ENSURE_LOADED, MMPClient


class _AutoReadySock:
    """Resolves each T_ENSURE_LOADED future to model_ready, records sends."""

    def __init__(self, client: MMPClient) -> None:
        self._client = client
        self.sent: list[list[bytes]] = []

    async def send_multipart(self, frames: list[bytes]) -> None:
        self.sent.append(frames)
        if frames[0] == T_ENSURE_LOADED:
            req_id = struct.unpack_from(">Q", frames[1])[0]
            fut = self._client._pending.get(req_id)
            if fut is not None and not fut.done():
                fut.set_result(("model_ready",))


def _api_keys_sent(sock: _AutoReadySock) -> list[str]:
    keys = []
    for frames in sock.sent:
        if frames[0] != T_ENSURE_LOADED:
            continue
        payload = frames[1]
        _req_id, _wait_ms, mid_len = struct.unpack_from(">QIH", payload)
        off = 14 + mid_len
        klen = struct.unpack_from(">H", payload, off)[0]
        keys.append(payload[off + 2 : off + 2 + klen].decode())
    return keys


def test_ensure_loaded_cache_keyed_per_api_key():
    async def _run() -> list[str]:
        client = MMPClient(mmp_addr="inproc://test", shm_name="test", shm_data_size=1024)
        client.ensure_cache_ttl_s = 60.0  # cache must be active for this test
        sock = _AutoReadySock(client)
        client._sock = sock

        await client.ensure_loaded("m", api_key="A")  # cold → sends
        await client.ensure_loaded("m", api_key="A")  # warm for A → no send
        await client.ensure_loaded("m", api_key="B")  # other tenant → must send
        return _api_keys_sent(sock)

    keys = asyncio.run(_run())
    # A sends once (second call served from cache); B is never masked by A's cache.
    assert keys == ["A", "B"]
