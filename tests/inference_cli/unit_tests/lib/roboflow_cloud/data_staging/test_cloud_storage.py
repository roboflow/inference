import pytest
from inference_cli.lib.roboflow_cloud.data_staging.api_operations import (
    _parse_bucket_path,
    _get_fs_kwargs,
)


def test_parse_bucket_path_without_glob():
    base, pattern = _parse_bucket_path("s3://bucket/path/")
    assert base == "s3://bucket/path/"
    assert pattern is None


def test_parse_bucket_path_with_glob():
    base, pattern = _parse_bucket_path("s3://bucket/path/**/*.jpg")
    assert base == "s3://bucket/path/"
    assert pattern == "**/*.jpg"


def test_get_fs_kwargs_with_endpoint(monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://r2.example.com")
    kwargs = _get_fs_kwargs()
    assert kwargs["client_kwargs"]["endpoint_url"] == "https://r2.example.com"


def test_get_fs_kwargs_without_endpoint(monkeypatch):
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    kwargs = _get_fs_kwargs()
    assert kwargs == {}


def test_get_fs_kwargs_with_profile(monkeypatch):
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.setenv("AWS_PROFILE", "my-profile")
    kwargs = _get_fs_kwargs()
    assert kwargs["profile"] == "my-profile"


def test_get_fs_kwargs_with_endpoint_and_profile(monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://r2.example.com")
    monkeypatch.setenv("AWS_PROFILE", "my-profile")
    kwargs = _get_fs_kwargs()
    assert kwargs["client_kwargs"]["endpoint_url"] == "https://r2.example.com"
    assert kwargs["profile"] == "my-profile"
