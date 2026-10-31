import json

from inference_server import configuration, platform_http


def test_build_api_headers_without_extras(monkeypatch):
    monkeypatch.setattr(configuration, "SERVER_VERSION", "9.9.9")
    monkeypatch.setattr(configuration, "ROBOFLOW_API_EXTRA_HEADERS", None)

    headers = platform_http.build_api_headers()

    assert headers == {
        "x-roboflow-inference-version": "9.9.9",
        "X-Allow-Chunked": "true",
    }


def test_build_api_headers_explicit_overrides_extras(monkeypatch):
    monkeypatch.setattr(configuration, "SERVER_VERSION", "9.9.9")
    monkeypatch.setattr(
        configuration,
        "ROBOFLOW_API_EXTRA_HEADERS",
        json.dumps({"x-shared": "from-extra", "x-extra": "1"}),
    )

    headers = platform_http.build_api_headers(
        explicit_headers={"x-shared": "from-explicit", "x-explicit": "2"}
    )

    assert headers == {
        "x-roboflow-inference-version": "9.9.9",
        "X-Allow-Chunked": "true",
        "x-shared": "from-explicit",
        "x-extra": "1",
        "x-explicit": "2",
    }


def test_build_api_headers_markers_win_over_explicit_and_extras(monkeypatch):
    monkeypatch.setattr(configuration, "SERVER_VERSION", "9.9.9")
    monkeypatch.setattr(
        configuration,
        "ROBOFLOW_API_EXTRA_HEADERS",
        json.dumps(
            {"x-roboflow-inference-version": "extra", "X-Allow-Chunked": "extra"}
        ),
    )

    headers = platform_http.build_api_headers(
        explicit_headers={
            "x-roboflow-inference-version": "explicit",
            "X-Allow-Chunked": "explicit",
        }
    )

    assert headers == {
        "x-roboflow-inference-version": "9.9.9",
        "X-Allow-Chunked": "true",
    }


def test_build_api_headers_ignores_undecodable_extras(monkeypatch):
    monkeypatch.setattr(configuration, "SERVER_VERSION", "9.9.9")
    monkeypatch.setattr(configuration, "ROBOFLOW_API_EXTRA_HEADERS", "{not json")

    headers = platform_http.build_api_headers(explicit_headers={"x-a": "1"})

    assert headers == {
        "x-roboflow-inference-version": "9.9.9",
        "X-Allow-Chunked": "true",
        "x-a": "1",
    }
