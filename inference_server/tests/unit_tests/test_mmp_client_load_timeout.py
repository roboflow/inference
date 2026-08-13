"""Timeout budget forwarding for MMPClient.load lifecycle requests."""

from __future__ import annotations

import asyncio
import inspect

from inference_server.proxies.mmp_client import T_LOAD, MMPClient


def _client_with_lifecycle_spy(recorded: list) -> MMPClient:
    client = MMPClient(mmp_addr="inproc://test")

    async def spy(*args, **kwargs):
        bound = inspect.signature(MMPClient._lifecycle_req).bind(
            client, *args, **kwargs
        )
        bound.apply_defaults()
        recorded.append((bound.arguments["msg_type"], bound.arguments["timeout_s"]))
        return ("ok",)

    client._lifecycle_req = spy
    return client


def test_load_without_timeout_uses_lifecycle_default():
    recorded = []
    client = _client_with_lifecycle_spy(recorded)

    result = asyncio.run(client.load("ws/1", "key"))

    assert result == ("ok",)
    assert recorded == [(T_LOAD, 30.0)]


def test_load_with_explicit_timeout_passes_it_to_lifecycle_req():
    recorded = []
    client = _client_with_lifecycle_spy(recorded)

    result = asyncio.run(client.load("ws/1", "key", timeout_s=600.0))

    assert result == ("ok",)
    assert recorded == [(T_LOAD, 600.0)]
