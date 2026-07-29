"""Network-free tests of the approved-fonts resolution and download logic.

All download operations are monkeypatched - these tests must never touch the
network. End-to-end download verification lives in the `bundled_fonts`
fixture used by rendering tests.
"""

import hashlib
from pathlib import Path

import pytest

import inference.core.workflows.core_steps.visualizations.common.fonts as fonts
from inference.core.workflows.core_steps.visualizations.common.fonts import (
    FONTS_REGISTRY,
    resolve_font_path,
)
from inference.core.workflows.core_steps.visualizations.common.fonts.downloader import (
    FontDownloadError,
    download_file_with_checksum,
)

FONT_ID = "geist_mono"
FONT_METADATA = FONTS_REGISTRY[FONT_ID]
FAKE_FONT_BYTES = b"not-a-real-font-but-good-enough-for-resolution-tests"
FAKE_FONT_SHA256 = hashlib.sha256(FAKE_FONT_BYTES).hexdigest()


@pytest.fixture()
def isolated_font_dirs(tmp_path, monkeypatch):
    assets_dir = tmp_path / "assets"
    cache_dir = tmp_path / "cache"
    assets_dir.mkdir()
    cache_dir.mkdir()
    monkeypatch.setattr(fonts, "ASSETS_DIR", assets_dir)
    monkeypatch.setattr(fonts, "FONTS_CACHE_DIR", cache_dir)
    monkeypatch.setattr(fonts, "ALLOW_WORKFLOWS_FONTS_DOWNLOAD", True)
    monkeypatch.setattr(fonts, "OFFLINE_MODE", False)
    monkeypatch.setattr(fonts, "SECURE_GATEWAY", None)

    return assets_dir, cache_dir


def _write_fake_font(base_dir: Path, content: bytes = FAKE_FONT_BYTES) -> Path:
    font_path = base_dir / FONT_ID / FONT_METADATA.file_name
    font_path.parent.mkdir(parents=True, exist_ok=True)
    font_path.write_bytes(content)

    return font_path


def _fake_download_factory(calls: list, content: bytes = FAKE_FONT_BYTES):
    def _fake_download(url, *, destination, expected_sha256, **kwargs):
        calls.append(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        return destination

    return _fake_download


def test_packaged_asset_takes_priority_over_cache_and_download(
    isolated_font_dirs, monkeypatch
) -> None:
    # given
    assets_dir, cache_dir = isolated_font_dirs
    packaged_path = _write_fake_font(assets_dir)
    _write_fake_font(cache_dir, content=b"cached-bytes")
    download_calls = []
    monkeypatch.setattr(
        fonts, "download_file_with_checksum", _fake_download_factory(download_calls)
    )

    # when
    resolved_path = resolve_font_path(FONT_ID)

    # then
    assert resolved_path == packaged_path
    assert download_calls == [], "No download should happen for packaged assets"


def test_missing_asset_triggers_approved_download(
    isolated_font_dirs, monkeypatch
) -> None:
    # given
    _assets_dir, cache_dir = isolated_font_dirs
    download_calls = []
    monkeypatch.setattr(
        fonts, "download_file_with_checksum", _fake_download_factory(download_calls)
    )

    # when
    resolved_path = resolve_font_path(FONT_ID)

    # then - the download must target the registry-pinned URL only
    assert download_calls == [FONT_METADATA.source_url]
    assert resolved_path == cache_dir / FONT_ID / FONT_METADATA.file_name
    assert resolved_path.read_bytes() == FAKE_FONT_BYTES


def test_corrupted_cache_entry_is_replaced_by_fresh_download(
    isolated_font_dirs, monkeypatch
) -> None:
    # given
    _assets_dir, cache_dir = isolated_font_dirs
    _write_fake_font(cache_dir, content=b"tampered-bytes")
    download_calls = []
    monkeypatch.setattr(
        fonts, "download_file_with_checksum", _fake_download_factory(download_calls)
    )

    # when
    resolved_path = resolve_font_path(FONT_ID)

    # then
    assert download_calls == [FONT_METADATA.source_url]
    assert resolved_path.read_bytes() == FAKE_FONT_BYTES


@pytest.mark.parametrize(
    "gate_overrides, expected_reason",
    [
        ({"ALLOW_WORKFLOWS_FONTS_DOWNLOAD": False}, "ALLOW_WORKFLOWS_FONTS_DOWNLOAD"),
        ({"OFFLINE_MODE": True}, "OFFLINE_MODE"),
        ({"SECURE_GATEWAY": "https://gateway.internal"}, "SECURE_GATEWAY"),
    ],
)
def test_missing_asset_with_downloads_gated_fails_fast_without_download(
    isolated_font_dirs, monkeypatch, gate_overrides: dict, expected_reason: str
) -> None:
    # given
    for attribute, value in gate_overrides.items():
        monkeypatch.setattr(fonts, attribute, value)
    download_calls = []
    monkeypatch.setattr(
        fonts, "download_file_with_checksum", _fake_download_factory(download_calls)
    )

    # when
    with pytest.raises(RuntimeError) as error:
        _ = resolve_font_path(FONT_ID)

    # then - actionable error, no download attempt
    assert expected_reason in str(error.value)
    assert "download_fonts" in str(error.value)
    assert download_calls == []


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload
        self._offset = 0

    def read(self, num_bytes: int) -> bytes:
        chunk = self._payload[self._offset : self._offset + num_bytes]
        self._offset += num_bytes
        return chunk

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args) -> None:
        pass


def test_download_with_checksum_mismatch_fails_hard_and_cleans_up(
    tmp_path, monkeypatch
) -> None:
    # given
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda url, timeout: _FakeResponse(b"evil-bytes")
    )
    destination = tmp_path / "font.ttf"

    # when
    with pytest.raises(FontDownloadError) as error:
        _ = download_file_with_checksum(
            "https://example.com/font.ttf",
            destination=destination,
            expected_sha256=FAKE_FONT_SHA256,
        )

    # then
    assert "Checksum mismatch" in str(error.value)
    assert not destination.exists(), "Unverified file must not be left behind"
    assert not destination.with_name("font.ttf.part").exists()


def test_download_refuses_non_https_url(tmp_path) -> None:
    with pytest.raises(FontDownloadError) as error:
        _ = download_file_with_checksum(
            "http://example.com/font.ttf",
            destination=tmp_path / "font.ttf",
            expected_sha256=FAKE_FONT_SHA256,
        )

    assert "non-HTTPS" in str(error.value)


def test_download_enforces_size_cap(tmp_path, monkeypatch) -> None:
    # given
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url, timeout: _FakeResponse(b"x" * 4096),
    )
    destination = tmp_path / "font.ttf"

    # when
    with pytest.raises(FontDownloadError) as error:
        _ = download_file_with_checksum(
            "https://example.com/font.ttf",
            destination=destination,
            expected_sha256=FAKE_FONT_SHA256,
            max_bytes=1024,
        )

    # then
    assert "maximum allowed" in str(error.value)
    assert not destination.exists()
    assert not destination.with_name("font.ttf.part").exists()
