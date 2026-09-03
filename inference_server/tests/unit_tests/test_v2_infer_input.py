"""Unit tests for v2 infer input extraction (multipart, JSON+base64, raw, batch)."""

from __future__ import annotations

import base64

import pytest

# Minimal JPEG (smallest valid JPEG — JFIF header + EOI)
_JPEG = bytes(
    [
        0xFF,
        0xD8,
        0xFF,
        0xE0,
        0x00,
        0x10,
        0x4A,
        0x46,
        0x49,
        0x46,
        0x00,
        0x01,
        0x01,
        0x00,
        0x00,
        0x01,
        0x00,
        0x01,
        0x00,
        0x00,
        0xFF,
        0xD9,
    ]
)


class _FakeUploadFile:
    def __init__(self, data: bytes):
        self._data = data

    async def read(self) -> bytes:
        return self._data


class _FakeRequest:
    def __init__(
        self, content_type: str = "", body: bytes = b"", form_data=None, json_body=None
    ):
        self.headers = {"content-type": content_type}
        self._body = body
        self._form_data = form_data
        self._json_body = json_body

    async def form(self):
        return self._form_data or _FakeFormData({})

    async def json(self):
        if self._json_body is not None:
            return self._json_body
        raise ValueError("no json")

    async def stream(self):
        if self._body:
            yield self._body


class _FakeFormData(dict):
    def multi_items(self):
        for k, v in self.items():
            if isinstance(v, list):
                for item in v:
                    yield k, item
            else:
                yield k, v


# ---------------------------------------------------------------------------
# Single-image tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raw_body():
    from inference_server.framework.input_parsers import extract_images_and_params

    req = _FakeRequest(content_type="image/jpeg", body=_JPEG)
    imgs, params, err = await extract_images_and_params(req)
    assert err is None
    assert len(imgs) == 1
    assert imgs[0] == _JPEG
    assert params == {}


@pytest.mark.asyncio
async def test_multipart_form_single():
    from inference_server.framework.input_parsers import extract_images_and_params

    form = _FakeFormData({"image": _FakeUploadFile(_JPEG), "confidence": "0.5"})
    req = _FakeRequest(content_type="multipart/form-data", form_data=form)
    imgs, params, err = await extract_images_and_params(req)
    assert err is None
    assert len(imgs) == 1
    assert imgs[0] == _JPEG
    assert params["confidence"] == "0.5"


@pytest.mark.asyncio
async def test_multipart_form_with_inputs_json():
    from inference_server.framework.input_parsers import extract_images_and_params

    form = _FakeFormData(
        {
            "image": _FakeUploadFile(_JPEG),
            "inputs": '{"confidence": 0.3, "iou": 0.5}',
        }
    )
    req = _FakeRequest(content_type="multipart/form-data", form_data=form)
    imgs, params, err = await extract_images_and_params(req)
    assert err is None
    assert len(imgs) == 1
    assert params["confidence"] == 0.3
    assert params["iou"] == 0.5


@pytest.mark.asyncio
async def test_multipart_form_missing_image_passes_through_params():
    # Zero image parts is legal at the extractor layer (params-only requests);
    # each handler-family parser enforces its own image requirement.
    from inference_server.framework.input_parsers import extract_images_and_params

    form = _FakeFormData({"confidence": "0.5"})
    req = _FakeRequest(content_type="multipart/form-data", form_data=form)
    imgs, params, err = await extract_images_and_params(req)
    assert err is None
    assert imgs == []
    assert params["confidence"] == "0.5"


@pytest.mark.asyncio
async def test_json_base64_single():
    from inference_server.framework.input_parsers import extract_images_and_params

    b64 = base64.b64encode(_JPEG).decode()
    body = {"inputs": {"image": {"type": "base64", "value": b64}, "confidence": 0.5}}
    req = _FakeRequest(content_type="application/json", json_body=body)
    imgs, params, err = await extract_images_and_params(req)
    assert err is None
    assert len(imgs) == 1
    assert imgs[0] == _JPEG
    assert params["confidence"] == 0.5


@pytest.mark.asyncio
async def test_json_missing_image():
    from inference_server.framework.input_parsers import extract_images_and_params

    body = {"inputs": {"confidence": 0.5}}
    req = _FakeRequest(content_type="application/json", json_body=body)
    imgs, params, err = await extract_images_and_params(req)
    assert err is not None
    assert err.status_code == 400


@pytest.mark.asyncio
async def test_json_invalid_base64():
    from inference_server.framework.input_parsers import extract_images_and_params

    body = {
        "inputs": {"image": {"type": "base64", "value": "!!!===not valid base64===!!!"}}
    }
    req = _FakeRequest(content_type="application/json", json_body=body)
    imgs, params, err = await extract_images_and_params(req)
    # base64.b64decode is lenient — no crash is the requirement
    assert err is None or err.status_code == 400


# ---------------------------------------------------------------------------
# Batch tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multipart_form_batch():
    from inference_server.framework.input_parsers import extract_images_and_params

    jpeg2 = _JPEG + b"\x00"  # slightly different
    form = _FakeFormData({"image": [_FakeUploadFile(_JPEG), _FakeUploadFile(jpeg2)]})
    req = _FakeRequest(content_type="multipart/form-data", form_data=form)
    imgs, params, err = await extract_images_and_params(req)
    assert err is None
    assert len(imgs) == 2
    assert imgs[0] == _JPEG
    assert imgs[1] == jpeg2


@pytest.mark.asyncio
async def test_json_base64_batch():
    from inference_server.framework.input_parsers import extract_images_and_params

    b64 = base64.b64encode(_JPEG).decode()
    body = {
        "inputs": {
            "image": [
                {"type": "base64", "value": b64},
                {"type": "base64", "value": b64},
            ]
        }
    }
    req = _FakeRequest(content_type="application/json", json_body=body)
    imgs, params, err = await extract_images_and_params(req)
    assert err is None
    assert len(imgs) == 2


# ---------------------------------------------------------------------------
# URL fetch tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_image_from_url_invalid_scheme():
    from inference_server.framework.input_parsers import fetch_image_from_url

    _, err = await fetch_image_from_url("ftp://example.com/image.jpg")
    assert err is not None
    assert err.status_code == 400


@pytest.mark.asyncio
async def test_fetch_image_from_url_bad_domain():
    from inference_server.framework.input_parsers import fetch_image_from_url

    _, err = await fetch_image_from_url(
        "https://this-domain-does-not-exist-12345.invalid/img.jpg"
    )
    assert err is not None
    assert err.status_code in (502, 504)


class _FakeContent:
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    def iter_chunked(self, size: int):
        return self._aiter()

    async def _aiter(self):
        for chunk in self._chunks:
            yield chunk


class _FakeResp:
    def __init__(self, chunks: list[bytes], content_length=None, status=200, headers=None):
        self.status = status
        self.content_length = content_length
        self.content = _FakeContent(chunks)
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeSession:
    def __init__(self, resp, redirects=None):
        self._resp = resp
        self._redirects = dict(redirects or {})
        self.requested: list[str] = []

    def get(self, url, allow_redirects=True):
        self.requested.append(url)
        location = self._redirects.get(url)
        if location is not None:
            return _FakeResp([], status=302, headers={"location": location})
        return self._resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _patch_public_dns(addresses=("93.184.216.34",)):
    from unittest.mock import patch

    async def _resolve(host: str):
        return list(addresses)

    return patch(
        "inference_server.framework.input_parsers.url_fetch.resolve_host", _resolve
    )


def _patch_http(chunks: list[bytes], content_length=None, redirects=None):
    """Patch aiohttp AND DNS: every unit test here must stay off the network."""
    import contextlib
    from unittest.mock import patch

    resp = _FakeResp(chunks, content_length=content_length)
    session = _FakeSession(resp, redirects=redirects)

    @contextlib.contextmanager
    def _both():
        with _patch_public_dns(), patch(
            "inference_server.framework.input_parsers.url_fetch.aiohttp.ClientSession",
            return_value=session,
        ):
            yield session

    return _both()


@pytest.mark.asyncio
async def test_fetch_image_from_url_chunked_over_cap_413():
    """No Content-Length on the response: cap must fire while streaming."""
    from unittest.mock import patch

    from inference_server.framework.input_parsers import fetch_image_from_url

    with _patch_http([b"x" * 8] * 10), patch(
        "inference_server.framework.input_parsers.url_fetch.URL_FETCH_MAX_BYTES", 16
    ):
        data, err = await fetch_image_from_url("https://example.com/img.jpg")
    assert data is None
    assert err.status_code == 413


@pytest.mark.asyncio
async def test_fetch_images_from_urls_too_many_urls_400():
    from unittest.mock import patch

    from inference_server.framework.input_parsers import fetch_images_from_urls

    with patch(
        "inference_server.configuration.MAX_IMAGES_PER_REQUEST",
        2,
    ):
        images, err = await fetch_images_from_urls(
            ["https://e.com/1.jpg", "https://e.com/2.jpg", "https://e.com/3.jpg"]
        )
    assert images is None
    assert err.status_code == 400


@pytest.mark.asyncio
async def test_fetch_images_from_urls_aggregate_over_budget_413():
    """Each URL is under the per-fetch cap but the sum exceeds the budget."""
    from unittest.mock import patch

    from inference_server.framework.input_parsers import fetch_images_from_urls

    with _patch_http([b"x" * 8]), patch(
        "inference_server.framework.input_parsers.url_fetch.configuration.MAX_BODY_BYTES",
        20,
    ):
        images, err = await fetch_images_from_urls(
            ["https://e.com/1.jpg", "https://e.com/2.jpg", "https://e.com/3.jpg"]
        )
    assert images is None
    assert err.status_code == 413


@pytest.mark.asyncio
async def test_fetch_images_from_urls_under_limits_ok():
    from inference_server.framework.input_parsers import fetch_images_from_urls

    with _patch_http([b"x" * 8]):
        images, err = await fetch_images_from_urls(
            ["https://e.com/1.jpg", "https://e.com/2.jpg"]
        )
    assert err is None
    assert images == [b"x" * 8, b"x" * 8]


# ---------------------------------------------------------------------------
# URL destination guarding (SSRF)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.1.2.3",
        "172.16.9.9",
        "192.168.0.5",
        "169.254.169.254",  # cloud instance metadata
        "0.0.0.0",
        "::1",
        "fd00::1",
        "::ffff:127.0.0.1",  # IPv4-mapped loopback
    ],
)
@pytest.mark.asyncio
async def test_fetch_image_from_url_rejects_non_global_destination(address):
    from inference_server.framework.input_parsers import fetch_image_from_url

    with _patch_public_dns((address,)):
        data, err = await fetch_image_from_url("https://internal.example/img.jpg")
    assert data is None
    assert err.status_code == 403


@pytest.mark.asyncio
async def test_fetch_image_from_url_rejects_when_any_address_is_internal():
    """Split-horizon DNS answers must not slip through on one public entry."""
    from inference_server.framework.input_parsers import fetch_image_from_url

    with _patch_public_dns(("93.184.216.34", "127.0.0.1")):
        data, err = await fetch_image_from_url("https://mixed.example/img.jpg")
    assert data is None
    assert err.status_code == 403


@pytest.mark.asyncio
async def test_fetch_image_from_url_allows_public_destination():
    from inference_server.framework.input_parsers import fetch_image_from_url

    with _patch_http([b"x" * 8]):
        data, err = await fetch_image_from_url("https://example.com/img.jpg")
    assert err is None
    assert data == b"x" * 8


@pytest.mark.asyncio
async def test_fetch_image_from_url_revalidates_redirect_hops():
    from unittest.mock import patch

    from inference_server.framework.input_parsers import fetch_image_from_url

    async def _resolve(host: str):
        return ["127.0.0.1"] if host == "metadata.internal" else ["93.184.216.34"]

    resp = _FakeResp([b"secret"])
    session = _FakeSession(
        resp,
        redirects={"https://example.com/img.jpg": "http://metadata.internal/creds"},
    )
    with patch(
        "inference_server.framework.input_parsers.url_fetch.resolve_host", _resolve
    ), patch(
        "inference_server.framework.input_parsers.url_fetch.aiohttp.ClientSession",
        return_value=session,
    ):
        data, err = await fetch_image_from_url("https://example.com/img.jpg")
    assert data is None
    assert err.status_code == 403
    # The internal URL was never requested.
    assert session.requested == ["https://example.com/img.jpg"]


@pytest.mark.asyncio
async def test_fetch_image_from_url_follows_allowed_redirect():
    from inference_server.framework.input_parsers import fetch_image_from_url

    with _patch_http(
        [b"ok"],
        redirects={"https://example.com/img.jpg": "https://cdn.example.com/img.jpg"},
    ) as session:
        data, err = await fetch_image_from_url("https://example.com/img.jpg")
    assert err is None
    assert data == b"ok"
    assert session.requested == [
        "https://example.com/img.jpg",
        "https://cdn.example.com/img.jpg",
    ]


@pytest.mark.asyncio
async def test_fetch_image_from_url_bounds_redirect_chain():
    from unittest.mock import patch

    from inference_server.framework.input_parsers import fetch_image_from_url

    hops = {
        f"https://example.com/{i}": f"https://example.com/{i + 1}" for i in range(50)
    }
    with _patch_http([b"ok"], redirects=hops), patch(
        "inference_server.configuration.MAX_IMAGE_URL_REDIRECTS", 2
    ) as _:
        data, err = await fetch_image_from_url("https://example.com/0")
    assert data is None
    assert err.status_code == 502


@pytest.mark.asyncio
async def test_fetch_image_from_url_allowlist_permits_internal_host():
    from unittest.mock import patch

    from inference_server.framework.input_parsers import fetch_image_from_url

    with _patch_http([b"ok"]), patch(
        "inference_server.configuration.WHITELISTED_DESTINATIONS_FOR_URL_INPUT",
        frozenset({"images.internal"}),
    ):
        data, err = await fetch_image_from_url("https://images.internal/img.jpg")
    assert err is None
    assert data == b"ok"


@pytest.mark.asyncio
async def test_fetch_image_from_url_allowlist_rejects_other_hosts():
    from unittest.mock import patch

    from inference_server.framework.input_parsers import fetch_image_from_url

    with _patch_http([b"ok"]), patch(
        "inference_server.configuration.WHITELISTED_DESTINATIONS_FOR_URL_INPUT",
        frozenset({"images.internal"}),
    ):
        data, err = await fetch_image_from_url("https://example.com/img.jpg")
    assert data is None
    assert err.status_code == 403


@pytest.mark.asyncio
async def test_fetch_image_from_url_blocklist_rejects_public_host():
    from unittest.mock import patch

    from inference_server.framework.input_parsers import fetch_image_from_url

    with _patch_http([b"ok"]), patch(
        "inference_server.configuration.BLACKLISTED_DESTINATIONS_FOR_URL_INPUT",
        frozenset({"example.com"}),
    ):
        data, err = await fetch_image_from_url("https://example.com/img.jpg")
    assert data is None
    assert err.status_code == 403


@pytest.mark.asyncio
async def test_fetch_image_from_url_non_global_opt_in():
    from unittest.mock import patch

    from inference_server.framework.input_parsers import fetch_image_from_url

    with _patch_http([b"ok"]), patch(
        "inference_server.configuration.ALLOW_URL_TO_NON_GLOBAL_ADDRESSES", True
    ), _patch_public_dns(("127.0.0.1",)):
        data, err = await fetch_image_from_url("http://localhost/img.jpg")
    assert err is None
    assert data == b"ok"


# ---------------------------------------------------------------------------
# DNS rebinding — the connection is pinned to the validated addresses
# ---------------------------------------------------------------------------


def _patch_pinning(answers: list[list[str]]):
    """Serve ``answers`` in order to resolve_host and capture the connector."""
    import contextlib
    from unittest.mock import patch

    captured: dict = {}
    session = _FakeSession(_FakeResp([b"ok"]))

    async def _resolve(host: str):
        return list(answers.pop(0)) if answers else ["169.254.169.254"]

    def _fake_connector(**kwargs):
        captured.update(kwargs)
        return object()

    @contextlib.contextmanager
    def _all():
        with patch(
            "inference_server.framework.input_parsers.url_fetch.resolve_host", _resolve
        ), patch(
            "inference_server.framework.input_parsers.url_fetch.aiohttp.TCPConnector",
            _fake_connector,
        ), patch(
            "inference_server.framework.input_parsers.url_fetch.aiohttp.ClientSession",
            return_value=session,
        ):
            yield captured

    return _all()


@pytest.mark.asyncio
async def test_fetch_image_from_url_pins_validated_addresses():
    """A second, hostile DNS answer must never reach the connection."""
    import socket

    from inference_server.framework.input_parsers import fetch_image_from_url

    with _patch_pinning([["93.184.216.34"], ["169.254.169.254"]]) as captured:
        data, err = await fetch_image_from_url("https://example.com/img.jpg")
        assert err is None
        assert data == b"ok"
        # Nothing but the resolver and the cache switch — no ssl / hostname
        # override that could relax certificate verification.
        assert set(captured) == {"resolver", "use_dns_cache"}
        assert captured["use_dns_cache"] is False
        resolved = await captured["resolver"].resolve("example.com", 443)

    assert [entry["host"] for entry in resolved] == ["93.184.216.34"]
    assert resolved[0]["hostname"] == "example.com"
    assert resolved[0]["port"] == 443
    assert resolved[0]["family"] == socket.AF_INET


@pytest.mark.asyncio
async def test_fetch_image_from_url_pins_allowlisted_host_too():
    """The allowlist skips the address check, not the pinning."""
    from unittest.mock import patch

    from inference_server.framework.input_parsers import fetch_image_from_url

    with _patch_pinning([["10.0.0.5"], ["169.254.169.254"]]) as captured, patch(
        "inference_server.configuration.WHITELISTED_DESTINATIONS_FOR_URL_INPUT",
        frozenset({"images.internal"}),
    ):
        data, err = await fetch_image_from_url("https://images.internal/img.jpg")
        assert err is None
        assert data == b"ok"
        resolved = await captured["resolver"].resolve("images.internal", 443)

    assert [entry["host"] for entry in resolved] == ["10.0.0.5"]


@pytest.mark.asyncio
async def test_pinned_resolver_fails_for_unvalidated_host():
    from inference_server.framework.input_parsers.url_fetch import _PinnedResolver

    resolver = _PinnedResolver({"example.com": ["93.184.216.34"]})
    with pytest.raises(OSError):
        await resolver.resolve("metadata.internal", 80)


@pytest.mark.asyncio
async def test_pinned_resolver_reports_ipv6_family():
    import socket

    from inference_server.framework.input_parsers.url_fetch import _PinnedResolver

    resolver = _PinnedResolver({"example.com": ["2606:2800:220:1:248:1893:25c8:1946"]})
    resolved = await resolver.resolve("example.com", 443)
    assert resolved[0]["family"] == socket.AF_INET6


@pytest.mark.asyncio
async def test_ip_literal_url_is_validated_and_bypasses_the_resolver():
    """aiohttp connects to an IP literal without consulting the resolver, so
    the literal itself must be (and is) validated by the destination check."""
    import aiohttp

    from inference_server.framework.input_parsers import fetch_image_from_url
    from inference_server.framework.input_parsers.url_fetch import _PinnedResolver

    with _patch_public_dns(("169.254.169.254",)):
        data, err = await fetch_image_from_url("https://169.254.169.254/img.jpg")
    assert data is None
    assert err.status_code == 403

    connector = aiohttp.TCPConnector(
        resolver=_PinnedResolver({}), use_dns_cache=False
    )
    try:
        hosts = await connector._resolve_host("93.184.216.34", 443)
    finally:
        await connector.close()
    assert [entry["host"] for entry in hosts] == ["93.184.216.34"]


# ---------------------------------------------------------------------------
# Per-request image count cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_json_base64_rejects_more_images_than_cap():
    from unittest.mock import patch

    from inference_server.framework.input_parsers import extract_images_and_params

    b64 = base64.b64encode(_JPEG).decode()
    body = {"inputs": {"image": [{"type": "base64", "value": b64}] * 4}}
    req = _FakeRequest(content_type="application/json", json_body=body)
    with patch("inference_server.configuration.MAX_IMAGES_PER_REQUEST", 3):
        imgs, params, err = await extract_images_and_params(req)
    assert imgs == []
    assert err.status_code == 400


@pytest.mark.asyncio
async def test_json_base64_allows_images_up_to_cap():
    from unittest.mock import patch

    from inference_server.framework.input_parsers import extract_images_and_params

    b64 = base64.b64encode(_JPEG).decode()
    body = {"inputs": {"image": [{"type": "base64", "value": b64}] * 3}}
    req = _FakeRequest(content_type="application/json", json_body=body)
    with patch("inference_server.configuration.MAX_IMAGES_PER_REQUEST", 3):
        imgs, params, err = await extract_images_and_params(req)
    assert err is None
    assert len(imgs) == 3


@pytest.mark.asyncio
async def test_multipart_rejects_more_images_than_cap():
    from unittest.mock import patch

    from inference_server.framework.input_parsers import extract_images_and_params

    form = _FakeFormData({"image": [_FakeUploadFile(_JPEG)] * 4})
    req = _FakeRequest(content_type="multipart/form-data", form_data=form)
    with patch("inference_server.configuration.MAX_IMAGES_PER_REQUEST", 3):
        imgs, params, err = await extract_images_and_params(req)
    assert imgs == []
    assert err.status_code == 400


@pytest.mark.asyncio
async def test_multipart_allows_images_up_to_cap():
    from unittest.mock import patch

    from inference_server.framework.input_parsers import extract_images_and_params

    form = _FakeFormData({"image": [_FakeUploadFile(_JPEG)] * 3})
    req = _FakeRequest(content_type="multipart/form-data", form_data=form)
    with patch("inference_server.configuration.MAX_IMAGES_PER_REQUEST", 3):
        imgs, params, err = await extract_images_and_params(req)
    assert err is None
    assert len(imgs) == 3
