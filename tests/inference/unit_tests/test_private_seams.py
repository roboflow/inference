def test_modal_spawner_without_provider_raises(monkeypatch):
    import importlib.metadata as md

    import pytest

    from inference.core.interfaces import webrtc_worker

    monkeypatch.setattr(md, "entry_points", lambda group=None: [])
    with pytest.raises(ImportError, match="enterprise runtime"):
        webrtc_worker._resolve_modal_spawner()
