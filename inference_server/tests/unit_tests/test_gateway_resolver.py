import importlib.metadata as md

import pytest

from inference_server import gateway_resolver as gr


def test_default_is_direct(monkeypatch):
    monkeypatch.delenv("INFERENCE_GATEWAY", raising=False)
    gw = gr.resolve_gateway()
    from inference_server.gateway import ModelManagerGateway

    assert isinstance(gw, ModelManagerGateway)


def test_entry_point_gateway_wins(monkeypatch):
    class FakeGateway:
        pass

    ep = md.EntryPoint(
        name="fake", value="tests_fake_mod:factory", group="inference_server.gateway"
    )
    monkeypatch.setattr(ep.__class__, "load", lambda self: (lambda: FakeGateway()))
    monkeypatch.setattr(gr, "_iter_gateway_entry_points", lambda: [ep])
    monkeypatch.setenv("INFERENCE_GATEWAY", "fake")
    gr.GATEWAY_FACTORIES.pop("fake", None)
    gr._reset_entry_point_cache_for_tests()
    assert isinstance(gr.resolve_gateway(), FakeGateway)


def test_unknown_gateway_errors_with_names(monkeypatch):
    monkeypatch.setenv("INFERENCE_GATEWAY", "nope")
    gr._reset_entry_point_cache_for_tests()
    with pytest.raises(RuntimeError, match="direct"):
        gr.resolve_gateway()
