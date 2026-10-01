import asyncio
import io

import aiohttp
import pytest
from PIL import Image

from inference_server.framework.input_parsers.url_fetch import fetch_images_from_urls
from tests.unit_tests.legacy.conftest import FakeGateway
from tests.unit_tests.test_v2_infer_input import _FakeResp

PREFIX = "Could not load input image. Cause: "
URL_INVALID = "Provided image URL is invalid"
URL_NON_HTTPS = (
    "Providing images via non https:// URL is not supported in this configuration "
    "of `inference`."
)
URL_WHITELIST = (
    "It is not allowed to reach image URL - prohibited by whitelisted destinations."
)
URL_BLACKLIST = (
    "It is not allowed to reach image URL - prohibited by blacklisted destinations."
)
URL_DESTINATION = "URL points to a network destination that is not allowed."
URL_FETCH = "Data pointed by URL could not be decoded into image."

PUBLIC_ADDRESS = "93.184.216.34"
FORBIDDEN_BODY = (
    b'{"error_code": "URL_DESTINATION_FORBIDDEN", '
    b'"description": "image URL destination"}'
)
FETCH_FAILED_BODY = (
    b'{"error_code": "URL_FETCH_FAILED", "description": "fetching image URL failed"}'
)
INVALID_SCHEME_BODY = (
    b'{"error_code": "INVALID_URL", '
    b'"description": "image URL must start with http:// or https://"}'
)
TOO_LARGE_BODY = (
    b'{"error_code": "URL_IMAGE_TOO_LARGE", '
    b'"description": "image at URL exceeds 0MB limit"}'
)


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6)).save(buffer, format="JPEG")

    return buffer.getvalue()


class _Session:
    def __init__(self, responses=None, redirects=None, error=None):
        self._responses = dict(responses or {})
        self._redirects = dict(redirects or {})
        self._error = error

    def get(self, url, allow_redirects=True):
        if self._error is not None:
            raise self._error
        location = self._redirects.get(url)
        if location is not None:
            return _FakeResp([], status=302, headers={"location": location})
        if url in self._responses:
            return self._responses[url]

        return _FakeResp([_jpeg()])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _configure(
    monkeypatch,
    *,
    session=None,
    blocked=None,
    allowed=None,
    addresses=None,
    max_redirects=3,
    max_fetch_bytes=None,
    max_body_bytes=None,
    max_images=None,
    offline=False,
    allow_url_input=True,
):
    addresses = addresses or {}

    async def _resolve(host):
        return list(addresses.get(host, (PUBLIC_ADDRESS,)))

    configuration = "inference_server.configuration"
    url_fetch = "inference_server.framework.input_parsers.url_fetch"
    monkeypatch.setattr(
        f"{configuration}.BLACKLISTED_DESTINATIONS_FOR_URL_INPUT", blocked
    )
    monkeypatch.setattr(
        f"{configuration}.WHITELISTED_DESTINATIONS_FOR_URL_INPUT", allowed
    )
    monkeypatch.setattr(f"{configuration}.ALLOW_URL_TO_NON_GLOBAL_ADDRESSES", False)
    monkeypatch.setattr(f"{configuration}.MAX_IMAGE_URL_REDIRECTS", max_redirects)
    monkeypatch.setattr(f"{configuration}.OFFLINE_MODE", offline)
    monkeypatch.setattr(f"{configuration}.ALLOW_URL_INPUT", allow_url_input)
    if max_body_bytes is not None:
        monkeypatch.setattr(f"{configuration}.MAX_BODY_BYTES", max_body_bytes)
    if max_images is not None:
        monkeypatch.setattr(f"{configuration}.MAX_IMAGES_PER_REQUEST", max_images)
    if max_fetch_bytes is not None:
        monkeypatch.setattr(f"{url_fetch}.URL_FETCH_MAX_BYTES", max_fetch_bytes)
    monkeypatch.setattr(f"{url_fetch}.resolve_host", _resolve)
    monkeypatch.setattr(
        f"{url_fetch}.aiohttp.ClientSession",
        lambda *args, **kwargs: session if session is not None else _Session(),
    )


A = "https://example.com/a.jpg"
B = "https://example.com/b.jpg"
REDIRECT_CHAIN = {
    f"https://example.com/{i}": f"https://example.com/{i + 1}" for i in range(9)
}

FAILURES = {
    "url-input-disabled": (
        [A],
        {"allow_url_input": False},
        403,
        b'{"error_code": "URL_INPUT_DISABLED", '
        b'"description": "loading images from URLs is disabled on this server"}',
    ),
    "offline": (
        [A],
        {"offline": True},
        403,
        b'{"error_code": "URL_INPUT_DISABLED", '
        b'"description": "loading images from URLs is disabled on this server"}',
    ),
    "other-scheme": (["ftp://example.com/a.jpg"], {}, 400, INVALID_SCHEME_BODY),
    "no-scheme": (["example.com/a.jpg"], {}, 400, INVALID_SCHEME_BODY),
    "no-host": (
        ["https:///a.jpg"],
        {},
        400,
        b'{"error_code": "INVALID_URL", "description": "image URL has no host"}',
    ),
    "blacklisted": ([A], {"blocked": frozenset({"example.com"})}, 403, FORBIDDEN_BODY),
    "not-whitelisted": (
        [A],
        {"allowed": frozenset({"images.internal"})},
        403,
        FORBIDDEN_BODY,
    ),
    "non-global": (
        [A],
        {"addresses": {"example.com": ("127.0.0.1",)}},
        403,
        FORBIDDEN_BODY,
    ),
    "redirect-to-blacklisted": (
        [A],
        {
            "session": _Session(redirects={A: "https://blocked.example/a.jpg"}),
            "blocked": frozenset({"blocked.example"}),
        },
        403,
        FORBIDDEN_BODY,
    ),
    "dns-failure": ([A], {"addresses": {"example.com": ()}}, 502, FETCH_FAILED_BODY),
    "redirect-without-location": (
        [A],
        {"session": _Session(responses={A: _FakeResp([], status=302)})},
        502,
        FETCH_FAILED_BODY,
    ),
    "upstream-status": (
        [A],
        {"session": _Session(responses={A: _FakeResp([], status=404)})},
        502,
        b'{"error_code": "URL_FETCH_FAILED", '
        b'"description": "fetching image URL returned status 404"}',
    ),
    "redirect-limit": (
        ["https://example.com/0"],
        {"session": _Session(redirects=REDIRECT_CHAIN), "max_redirects": 2},
        502,
        b'{"error_code": "URL_FETCH_FAILED", '
        b'"description": "too many redirects fetching image URL"}',
    ),
    "timeout": (
        [A],
        {"session": _Session(error=asyncio.TimeoutError())},
        504,
        b'{"error_code": "URL_FETCH_TIMEOUT", '
        b'"description": "fetching image URL timed out after 10s"}',
    ),
    "client-error": (
        [A],
        {"session": _Session(error=aiohttp.ClientError("boom"))},
        502,
        FETCH_FAILED_BODY,
    ),
    "oversize-declared": (
        [A],
        {
            "session": _Session(
                responses={A: _FakeResp([b"x" * 8], content_length=64)}
            ),
            "max_fetch_bytes": 16,
        },
        413,
        TOO_LARGE_BODY,
    ),
    "oversize-streamed": (
        [A],
        {
            "session": _Session(responses={A: _FakeResp([b"x" * 8] * 10)}),
            "max_fetch_bytes": 16,
        },
        413,
        TOO_LARGE_BODY,
    ),
    "combined-oversize": (
        [A, B],
        {
            "session": _Session(
                responses={A: _FakeResp([b"x" * 8]), B: _FakeResp([b"x" * 8])}
            ),
            "max_body_bytes": 12,
        },
        413,
        b'{"error_code": "PAYLOAD_TOO_LARGE", '
        b'"description": "combined size of images at URLs exceeds 12 byte limit"}',
    ),
    "too-many-images": (
        [A, B],
        {"max_images": 1},
        400,
        b'{"error_code": "TOO_MANY_IMAGES", '
        b'"description": "at most 1 images per request"}',
    ),
    "batch-non-global-then-blacklisted": (
        ["https://internal.example/a.jpg", "https://blocked.example/b.jpg"],
        {
            "addresses": {"internal.example": ("127.0.0.1",)},
            "blocked": frozenset({"blocked.example"}),
        },
        403,
        FORBIDDEN_BODY,
    ),
    "batch-fine-then-invalid": ([A, "example.com/b.jpg"], {}, 400, INVALID_SCHEME_BODY),
}


@pytest.mark.parametrize("kind", sorted(FAILURES))
@pytest.mark.asyncio
async def test_v2_url_fetch_failure_body_is_unchanged(monkeypatch, kind):
    urls, settings, status, body = FAILURES[kind]
    _configure(monkeypatch, **settings)

    images, error = await fetch_images_from_urls(urls)

    assert images is None
    assert error.status_code == status
    assert error.body == body
    assert error.media_type == "application/json"


LEGACY_ANSWERS = {
    "other-scheme": (400, f"{PREFIX}{URL_NON_HTTPS}"),
    "no-scheme": (400, f"{PREFIX}{URL_INVALID}"),
    "no-host": (400, f"{PREFIX}{URL_INVALID}"),
    "blacklisted": (400, f"{PREFIX}{URL_BLACKLIST}"),
    "not-whitelisted": (400, f"{PREFIX}{URL_WHITELIST}"),
    "non-global": (400, f"{PREFIX}{URL_DESTINATION}"),
    "redirect-to-blacklisted": (400, f"{PREFIX}{URL_BLACKLIST}"),
    "dns-failure": (400, f"{PREFIX}{URL_FETCH}"),
    "redirect-without-location": (400, f"{PREFIX}{URL_FETCH}"),
    "upstream-status": (400, f"{PREFIX}{URL_FETCH}"),
    "redirect-limit": (400, f"{PREFIX}{URL_FETCH}"),
    "timeout": (400, f"{PREFIX}{URL_FETCH}"),
    "client-error": (400, f"{PREFIX}{URL_FETCH}"),
    "oversize-declared": (413, "image at URL exceeds 0MB limit"),
    "oversize-streamed": (413, "image at URL exceeds 0MB limit"),
    "batch-non-global-then-blacklisted": (400, f"{PREFIX}{URL_DESTINATION}"),
    "batch-fine-then-invalid": (400, f"{PREFIX}{URL_INVALID}"),
}


def _post_urls(legacy_client, fake_stat, urls):
    fake_stat["ds/1"] = ("object-detection", "infer")
    client = legacy_client(FakeGateway(model_info={"ds/1": {"class_names": ["cat"]}}))
    images = [{"type": "url", "value": url} for url in urls]
    response = client.post(
        "/infer/object_detection",
        json={
            "model_id": "ds/1",
            "api_key": "k",
            "image": images if len(images) > 1 else images[0],
        },
    )

    return response


@pytest.mark.parametrize("kind", sorted(LEGACY_ANSWERS))
def test_url_fetch_failure_answers_like_legacy_on_the_route(
    legacy_client, fake_stat, monkeypatch, kind
):
    urls, settings, _, _ = FAILURES[kind]
    status, message = LEGACY_ANSWERS[kind]
    _configure(monkeypatch, **settings)

    response = _post_urls(legacy_client, fake_stat, urls)

    assert response.status_code == status, response.text
    assert response.json() == {"message": message}


@pytest.mark.parametrize(
    "urls,settings,public_message",
    [
        (["ftp://["], {}, URL_INVALID),
        (
            ["https://blocked.example\\@blocked.example/a.jpg"],
            {"blocked": frozenset({"blocked.example"})},
            URL_INVALID,
        ),
        (
            ["https://*.example.com/a.jpg"],
            {"blocked": frozenset({"*.example.com"})},
            URL_INVALID,
        ),
        (
            ["https://example.com:abc/a.jpg"],
            {"blocked": frozenset({"example.com"})},
            URL_INVALID,
        ),
        (
            ["https://example.com:abc/a.jpg"],
            {"allowed": frozenset({"images.internal"})},
            URL_INVALID,
        ),
        (["https://["], {}, URL_INVALID),
        ([A, "https://["], {}, URL_INVALID),
        (
            ["https://internal.example/a.jpg", "https://["],
            {"addresses": {"internal.example": ("127.0.0.1",)}},
            URL_DESTINATION,
        ),
        (
            [A],
            {
                "session": _Session(redirects={A: "ftp://other.example/a.jpg"}),
            },
            URL_NON_HTTPS,
        ),
        (
            [A],
            {
                "session": _Session(redirects={A: "https://other.example/a.jpg"}),
                "allowed": frozenset({"example.com"}),
            },
            URL_WHITELIST,
        ),
        (
            [A],
            {
                "session": _Session(redirects={A: "https://other.example/a.jpg"}),
                "allowed": frozenset({"example.com"}),
                "blocked": frozenset({"other.example"}),
            },
            URL_WHITELIST,
        ),
    ],
)
def test_url_failure_names_the_url_that_failed(
    legacy_client, fake_stat, monkeypatch, urls, settings, public_message
):
    _configure(monkeypatch, **settings)

    response = _post_urls(legacy_client, fake_stat, urls)

    assert response.status_code == 400, response.text
    assert response.json() == {"message": f"{PREFIX}{public_message}"}


def test_redirect_to_an_unparsable_location_answers_like_legacy(
    legacy_client, fake_stat, monkeypatch
):
    _configure(monkeypatch, session=_Session(redirects={A: "https://["}))

    response = _post_urls(legacy_client, fake_stat, [A])

    assert response.status_code == 500
    assert response.json() == {"message": "Internal error."}


def test_fetched_content_that_is_not_an_image_answers_like_legacy_on_the_route(
    legacy_client, fake_stat, monkeypatch
):
    _configure(monkeypatch, session=_Session(responses={A: _FakeResp([b"hello"])}))

    response = _post_urls(legacy_client, fake_stat, [A])

    assert response.status_code == 400, response.text
    assert response.json() == {"message": f"{PREFIX}Data is not image."}
