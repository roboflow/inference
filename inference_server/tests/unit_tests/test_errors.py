"""Unit tests for structured error responses."""

import json

from inference_server.errors import error_response


def test_basic_error():
    resp = error_response(400, "MISSING_PARAM", "model_id required")
    assert resp.status_code == 400
    body = json.loads(resp.body)
    assert body["error_code"] == "MISSING_PARAM"
    assert body["description"] == "model_id required"
    assert "actionable_follow_up" not in body
    assert "help_url" not in body


def test_error_with_follow_up():
    resp = error_response(
        404, "NOT_FOUND", "model not loaded", follow_up="load it first"
    )
    body = json.loads(resp.body)
    assert body["actionable_follow_up"] == "load it first"


def test_error_with_help_url():
    resp = error_response(500, "ERR", "failed", help_url="https://docs.example.com")
    body = json.loads(resp.body)
    assert body["help_url"] == "https://docs.example.com"


def test_error_with_headers():
    resp = error_response(503, "LOADING", "loading", headers={"Retry-After": "5"})
    assert resp.headers.get("Retry-After") == "5"


def test_content_type_is_json():
    resp = error_response(400, "ERR", "msg")
    assert resp.media_type == "application/json"
