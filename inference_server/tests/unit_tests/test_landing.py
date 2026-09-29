import importlib

from fastapi.testclient import TestClient

from tests.unit_tests.legacy.conftest import FakeGateway


def _reloaded_app(monkeypatch, **overrides):
    import inference_server.app as app_mod
    from inference_server import configuration

    monkeypatch.setattr(
        "inference_model_manager.watchdogs.start_enabled_watchdogs", lambda: []
    )
    monkeypatch.delenv("INFERENCE_PRELOAD_MODELS", raising=False)
    for name, value in overrides.items():
        monkeypatch.setattr(configuration, name, value)
    return importlib.reload(app_mod)


def _make_landing_dir(tmp_path):
    (tmp_path / "index.html").write_text("landing")
    (tmp_path / "static").mkdir()
    (tmp_path / "static" / "x.txt").write_text("x")
    (tmp_path / "_next" / "static").mkdir(parents=True)
    (tmp_path / "_next" / "static" / "y.js").write_text("y")
    (tmp_path / "dashboard.html").write_text("dashboard")
    return tmp_path


def test_landing_page_and_static_mounts_are_served(tmp_path, monkeypatch):
    landing_dir = _make_landing_dir(tmp_path)
    module = _reloaded_app(monkeypatch, LANDING_DIR=str(landing_dir))
    try:
        with TestClient(module.app) as client:
            root = client.get("/")
            assert root.status_code == 200
            assert "landing" in root.text
            assert client.get("/static/x.txt").status_code == 200
            assert client.get("/_next/static/y.js").status_code == 200
    finally:
        monkeypatch.undo()
        importlib.reload(module)


def test_dashboard_html_is_404_by_default(tmp_path, monkeypatch):
    landing_dir = _make_landing_dir(tmp_path)
    module = _reloaded_app(
        monkeypatch, LANDING_DIR=str(landing_dir), ENABLE_DASHBOARD=False
    )
    try:
        with TestClient(module.app) as client:
            assert client.get("/dashboard.html").status_code == 404
            assert client.head("/dashboard.html").status_code == 404
    finally:
        monkeypatch.undo()
        importlib.reload(module)


def test_dashboard_html_is_served_when_enabled(tmp_path, monkeypatch):
    landing_dir = _make_landing_dir(tmp_path)
    module = _reloaded_app(
        monkeypatch, LANDING_DIR=str(landing_dir), ENABLE_DASHBOARD=True
    )
    try:
        with TestClient(module.app) as client:
            assert client.get("/dashboard.html").status_code == 200
    finally:
        monkeypatch.undo()
        importlib.reload(module)


def test_legacy_catch_all_still_reachable_alongside_root_mount(
    tmp_path, monkeypatch, fake_stat
):
    landing_dir = _make_landing_dir(tmp_path)
    module = _reloaded_app(monkeypatch, LANDING_DIR=str(landing_dir))
    fake_stat["some-workspace/3"] = ("object-detection", "infer")
    try:
        monkeypatch.setattr(
            "inference_server.gateway_resolver.resolve_gateway",
            lambda: FakeGateway(),
        )
        with TestClient(module.app, raise_server_exceptions=False) as client:
            response = client.post("/some-workspace/3?api_key=k")
            assert response.status_code == 400
    finally:
        monkeypatch.undo()
        importlib.reload(module)


def test_missing_landing_dir_does_not_mount_root(tmp_path, monkeypatch):
    missing_dir = tmp_path / "does-not-exist"
    module = _reloaded_app(monkeypatch, LANDING_DIR=str(missing_dir))
    try:
        assert module._LANDING_ASSETS_MOUNTED is False
        assert module._LANDING_ROOT_MOUNTED is False
        with TestClient(module.app) as client:
            assert client.get("/").status_code == 404
    finally:
        monkeypatch.undo()
        importlib.reload(module)
