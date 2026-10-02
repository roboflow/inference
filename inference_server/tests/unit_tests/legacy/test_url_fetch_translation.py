import asyncio
import io

from types import SimpleNamespace

import aiohttp
import numpy as np
import pytest
from PIL import Image

from inference_server.framework.input_parsers.url_fetch import fetch_images_from_urls
from inference_server.legacy.bridge import LegacyModelBridge
from inference_server.legacy.errors import LegacyHTTPError
from tests.unit_tests.legacy.conftest import FakeGateway
from tests.unit_tests.test_v2_infer_input import _FakeResp

PREFIX = "Could not load input image. Cause: "
URL_INVALID = "Provided image URL is invalid"
URL_NON_HTTPS = (
    "Providing images via non https:// URL is not supported in this configuration "
    "of `inference`."
)
URL_WITHOUT_FQDN = (
    "Providing images via URL without FQDN is not supported in this configuration "
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
        self.requested = []

    def get(self, url, allow_redirects=True):
        self.requested.append(url)
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
    allow_non_global=False,
    allow_non_https=False,
    allow_without_fqdn=False,
    validate_redirects=False,
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
    monkeypatch.setattr(
        f"{configuration}.ALLOW_URL_TO_NON_GLOBAL_ADDRESSES", allow_non_global
    )
    monkeypatch.setattr(f"{configuration}.ALLOW_NON_HTTPS_URL_INPUT", allow_non_https)
    monkeypatch.setattr(
        f"{configuration}.ALLOW_URL_INPUT_WITHOUT_FQDN", allow_without_fqdn
    )
    monkeypatch.setattr(
        f"{configuration}.VALIDATE_IMAGE_URL_REDIRECTS", validate_redirects
    )
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


WITHOUT_FQDN = {"allow_without_fqdn": True}
VALIDATING = {"validate_redirects": True}

LEGACY_ANSWERS = {
    "other-scheme": (400, f"{PREFIX}{URL_NON_HTTPS}", {}),
    "no-scheme": (400, f"{PREFIX}{URL_INVALID}", {}),
    "no-host": (400, f"{PREFIX}{URL_INVALID}", {}),
    "blacklisted": (400, f"{PREFIX}{URL_BLACKLIST}", {}),
    "not-whitelisted": (400, f"{PREFIX}{URL_WHITELIST}", {}),
    "non-global": (400, f"{PREFIX}{URL_DESTINATION}", {}),
    "redirect-to-blacklisted": (
        400,
        f"{PREFIX}{URL_BLACKLIST}",
        {**WITHOUT_FQDN, **VALIDATING},
    ),
    "dns-failure": (400, f"{PREFIX}{URL_FETCH}", {}),
    "redirect-without-location": (400, f"{PREFIX}{URL_FETCH}", {}),
    "upstream-status": (400, f"{PREFIX}{URL_FETCH}", {}),
    "redirect-limit": (400, f"{PREFIX}{URL_FETCH}", {}),
    "timeout": (400, f"{PREFIX}{URL_FETCH}", {}),
    "client-error": (400, f"{PREFIX}{URL_FETCH}", {}),
    "oversize-declared": (413, "image at URL exceeds 0MB limit", {}),
    "oversize-streamed": (413, "image at URL exceeds 0MB limit", {}),
    "batch-non-global-then-blacklisted": (
        400,
        f"{PREFIX}{URL_DESTINATION}",
        WITHOUT_FQDN,
    ),
    "batch-fine-then-invalid": (400, f"{PREFIX}{URL_INVALID}", {}),
}


def _detections():
    return SimpleNamespace(
        xyxy=np.array([[1, 1, 3, 5]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )


def _post_urls(legacy_client, fake_stat, urls):
    fake_stat["ds/1"] = ("object-detection", "infer")
    client = legacy_client(
        FakeGateway(
            predictions={("ds/1", "infer"): _detections()},
            model_info={"ds/1": {"class_names": ["cat"], "actions": {"infer": {}}}},
        )
    )
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
    status, message, switches = LEGACY_ANSWERS[kind]
    _configure(monkeypatch, **settings, **switches)

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
            {"addresses": {"internal.example": ("127.0.0.1",)}, **WITHOUT_FQDN},
            URL_DESTINATION,
        ),
        (
            [A],
            {
                "session": _Session(redirects={A: "ftp://other.example/a.jpg"}),
                **VALIDATING,
            },
            URL_NON_HTTPS,
        ),
        (
            [A],
            {
                "session": _Session(redirects={A: "https://other.example/a.jpg"}),
                "allowed": frozenset({"example.com"}),
                **WITHOUT_FQDN,
                **VALIDATING,
            },
            URL_WHITELIST,
        ),
        (
            [A],
            {
                "session": _Session(redirects={A: "https://other.example/a.jpg"}),
                "allowed": frozenset({"example.com"}),
                "blocked": frozenset({"other.example"}),
                **WITHOUT_FQDN,
                **VALIDATING,
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


HTTP_URL = "http://example.com/a.jpg"
PRIVATE_HOP = "http://10.0.0.5/a.jpg"
FILES_URL = "https://files.example.com/a.jpg"
BLOCKED_URL = "https://blocked.example.com/a.jpg"
OPEN = {"allow_non_global": True}


def test_non_https_url_is_refused_by_default(legacy_client, fake_stat, monkeypatch):
    session = _Session()
    _configure(monkeypatch, session=session, **OPEN)

    response = _post_urls(legacy_client, fake_stat, [HTTP_URL])

    assert response.status_code == 400, response.text
    assert response.json() == {"message": f"{PREFIX}{URL_NON_HTTPS}"}
    assert session.requested == []


def test_non_https_url_is_fetched_when_allowed(legacy_client, fake_stat, monkeypatch):
    session = _Session()
    _configure(monkeypatch, session=session, allow_non_https=True, **OPEN)

    response = _post_urls(legacy_client, fake_stat, [HTTP_URL])

    assert response.status_code == 200, response.text
    assert session.requested == [HTTP_URL]


@pytest.mark.parametrize(
    "url",
    [
        "https://192.168.1.5/a.jpg",
        "https://localhost/a.jpg",
        "https://myhost/a.jpg",
        "https://[::1]/a.jpg",
    ],
)
def test_url_without_domain_name_is_refused_by_default(
    legacy_client, fake_stat, monkeypatch, url
):
    session = _Session()
    _configure(monkeypatch, session=session, **OPEN)

    response = _post_urls(legacy_client, fake_stat, [url])

    assert response.status_code == 400, response.text
    assert response.json() == {"message": f"{PREFIX}{URL_WITHOUT_FQDN}"}
    assert session.requested == []


@pytest.mark.parametrize(
    "url",
    ["https://192.168.1.5/a.jpg", "https://localhost/a.jpg", "https://myhost/a.jpg"],
)
def test_url_without_domain_name_is_fetched_when_allowed(
    legacy_client, fake_stat, monkeypatch, url
):
    session = _Session()
    _configure(monkeypatch, session=session, allow_without_fqdn=True, **OPEN)

    response = _post_urls(legacy_client, fake_stat, [url])

    assert response.status_code == 200, response.text
    assert session.requested == [url]


@pytest.mark.parametrize("url", ["HTTPS://Example.COM/a.jpg", "  " + A])
def test_scheme_case_and_leading_whitespace_are_normalised_like_legacy(
    legacy_client, fake_stat, monkeypatch, url
):
    session = _Session()
    _configure(monkeypatch, session=session, **OPEN)

    response = _post_urls(legacy_client, fake_stat, [url])

    assert response.status_code == 200, response.text
    assert session.requested == [A]


def test_scheme_rule_precedes_domain_name_rule(legacy_client, fake_stat, monkeypatch):
    _configure(monkeypatch, **OPEN)

    response = _post_urls(legacy_client, fake_stat, ["http://localhost/a.jpg"])

    assert response.status_code == 400, response.text
    assert response.json() == {"message": f"{PREFIX}{URL_NON_HTTPS}"}


def test_domain_name_rule_precedes_the_lists(legacy_client, fake_stat, monkeypatch):
    _configure(monkeypatch, blocked=frozenset({"localhost"}), **OPEN)

    response = _post_urls(legacy_client, fake_stat, ["https://localhost/a.jpg"])

    assert response.status_code == 400, response.text
    assert response.json() == {"message": f"{PREFIX}{URL_WITHOUT_FQDN}"}


def test_redirect_to_non_https_private_address_is_followed_by_default(
    legacy_client, fake_stat, monkeypatch
):
    session = _Session(redirects={FILES_URL: PRIVATE_HOP})
    _configure(
        monkeypatch,
        session=session,
        addresses={"10.0.0.5": ("10.0.0.5",)},
        **OPEN,
    )

    response = _post_urls(legacy_client, fake_stat, [FILES_URL])

    assert response.status_code == 200, response.text
    assert session.requested == [FILES_URL, PRIVATE_HOP]


def test_redirect_to_non_https_private_address_is_refused_when_validating(
    legacy_client, fake_stat, monkeypatch
):
    session = _Session(redirects={FILES_URL: PRIVATE_HOP})
    _configure(
        monkeypatch,
        session=session,
        addresses={"10.0.0.5": ("10.0.0.5",)},
        **VALIDATING,
        **OPEN,
    )

    response = _post_urls(legacy_client, fake_stat, [FILES_URL])

    assert response.status_code == 400, response.text
    assert response.json() == {"message": f"{PREFIX}{URL_NON_HTTPS}"}
    assert session.requested == [FILES_URL]


def test_redirect_to_host_without_domain_name_is_refused_when_validating(
    legacy_client, fake_stat, monkeypatch
):
    session = _Session(redirects={FILES_URL: "https://10.0.0.5/a.jpg"})
    _configure(monkeypatch, session=session, **VALIDATING, **OPEN)

    response = _post_urls(legacy_client, fake_stat, [FILES_URL])

    assert response.status_code == 400, response.text
    assert response.json() == {"message": f"{PREFIX}{URL_WITHOUT_FQDN}"}
    assert session.requested == [FILES_URL]


def test_redirect_to_deny_listed_host_is_followed_by_default(
    legacy_client, fake_stat, monkeypatch
):
    session = _Session(redirects={A: BLOCKED_URL})
    _configure(
        monkeypatch,
        session=session,
        blocked=frozenset({"blocked.example.com"}),
        **OPEN,
    )

    response = _post_urls(legacy_client, fake_stat, [A])

    assert response.status_code == 200, response.text
    assert session.requested == [A, BLOCKED_URL]


def test_redirect_to_deny_listed_host_is_refused_when_validating(
    legacy_client, fake_stat, monkeypatch
):
    session = _Session(redirects={A: BLOCKED_URL})
    _configure(
        monkeypatch,
        session=session,
        blocked=frozenset({"blocked.example.com"}),
        **VALIDATING,
        **OPEN,
    )

    response = _post_urls(legacy_client, fake_stat, [A])

    assert response.status_code == 400, response.text
    assert response.json() == {"message": f"{PREFIX}{URL_BLACKLIST}"}
    assert session.requested == [A]


def test_redirect_to_other_scheme_is_a_failed_fetch_by_default(
    legacy_client, fake_stat, monkeypatch
):
    session = _Session(redirects={A: "ftp://files.example.com/a.jpg"})
    _configure(monkeypatch, session=session, **OPEN)

    response = _post_urls(legacy_client, fake_stat, [A])

    assert response.status_code == 400, response.text
    assert response.json() == {"message": f"{PREFIX}{URL_FETCH}"}
    assert session.requested == [A]


def test_first_url_resolving_to_private_address_is_refused_when_non_global_is_off(
    legacy_client, fake_stat, monkeypatch
):
    session = _Session()
    _configure(monkeypatch, session=session, addresses={"example.com": ("10.0.0.5",)})

    response = _post_urls(legacy_client, fake_stat, [A])

    assert response.status_code == 400, response.text
    assert response.json() == {"message": f"{PREFIX}{URL_DESTINATION}"}
    assert session.requested == []


def test_hop_resolving_to_private_address_is_refused_when_non_global_is_off(
    legacy_client, fake_stat, monkeypatch
):
    session = _Session(redirects={A: BLOCKED_URL})
    _configure(
        monkeypatch,
        session=session,
        blocked=frozenset({"blocked.example.com"}),
        addresses={"blocked.example.com": ("10.0.0.5",)},
    )

    response = _post_urls(legacy_client, fake_stat, [A])

    assert response.status_code == 400, response.text
    assert response.json() == {"message": f"{PREFIX}{URL_DESTINATION}"}
    assert session.requested == [A]


def test_hop_to_private_address_literal_is_refused_when_non_global_is_off(
    legacy_client, fake_stat, monkeypatch
):
    session = _Session(redirects={A: PRIVATE_HOP})
    _configure(monkeypatch, session=session, addresses={"10.0.0.5": ("10.0.0.5",)})

    response = _post_urls(legacy_client, fake_stat, [A])

    assert response.status_code == 400, response.text
    assert response.json() == {"message": f"{PREFIX}{URL_DESTINATION}"}
    assert session.requested == [A]


def test_allow_listed_host_at_private_address_is_refused_on_the_route(
    legacy_client, fake_stat, monkeypatch
):
    session = _Session()
    _configure(
        monkeypatch,
        session=session,
        allowed=frozenset({"example.com"}),
        addresses={"example.com": ("10.0.0.5",)},
    )

    response = _post_urls(legacy_client, fake_stat, [A])

    assert response.status_code == 400, response.text
    assert response.json() == {"message": f"{PREFIX}{URL_DESTINATION}"}
    assert session.requested == []


@pytest.mark.asyncio
async def test_allow_listed_host_at_private_address_is_fetched_by_the_shared_fetch(
    monkeypatch,
):
    session = _Session()
    _configure(
        monkeypatch,
        session=session,
        allowed=frozenset({"example.com"}),
        addresses={"example.com": ("10.0.0.5",)},
    )

    images, error = await fetch_images_from_urls([A])

    assert error is None
    assert images == [_jpeg()]
    assert session.requested == [A]


@pytest.mark.parametrize("switches", [{}, VALIDATING], ids=["default", "validating"])
def test_redirect_limit_answers_like_legacy(
    legacy_client, fake_stat, monkeypatch, switches
):
    session = _Session(redirects=REDIRECT_CHAIN)
    _configure(monkeypatch, session=session, max_redirects=2, **switches)

    response = _post_urls(legacy_client, fake_stat, ["https://example.com/0"])

    assert response.status_code == 400, response.text
    assert response.json() == {"message": f"{PREFIX}{URL_FETCH}"}
    assert session.requested == [f"https://example.com/{i}" for i in range(3)]


@pytest.mark.parametrize("switches", [{}, VALIDATING], ids=["default", "validating"])
def test_redirects_up_to_the_limit_are_followed(
    legacy_client, fake_stat, monkeypatch, switches
):
    session = _Session(redirects={A: B, B: FILES_URL})
    _configure(monkeypatch, session=session, max_redirects=2, **switches)

    response = _post_urls(legacy_client, fake_stat, [A])

    assert response.status_code == 200, response.text
    assert session.requested == [A, B, FILES_URL]


def test_refused_redirect_at_the_limit_answers_with_its_rule_when_validating(
    legacy_client, fake_stat, monkeypatch
):
    session = _Session(redirects={A: B, B: HTTP_URL})
    _configure(monkeypatch, session=session, max_redirects=1, **VALIDATING)

    response = _post_urls(legacy_client, fake_stat, [A])

    assert response.status_code == 400, response.text
    assert response.json() == {"message": f"{PREFIX}{URL_NON_HTTPS}"}


def test_refused_url_after_a_fetched_one_answers_after_the_fetch(
    legacy_client, fake_stat, monkeypatch
):
    session = _Session()
    _configure(monkeypatch, session=session, **OPEN)

    response = _post_urls(legacy_client, fake_stat, [A, HTTP_URL])

    assert response.status_code == 400, response.text
    assert response.json() == {"message": f"{PREFIX}{URL_NON_HTTPS}"}
    assert session.requested == [A]


@pytest.mark.parametrize(
    "url,settings,status",
    [
        (HTTP_URL, {}, 400),
        ("https://192.168.1.5/a.jpg", {}, 400),
        ("https://localhost/a.jpg", {}, 400),
        ("https://myhost/a.jpg", {}, 400),
        ("https://[", {}, 400),
        (A, {"blocked": frozenset({"example.com"})}, 403),
        (A, {"allowed": frozenset({"images.example.com"})}, 403),
    ],
)
@pytest.mark.asyncio
async def test_workflow_image_fetch_refuses_like_the_route(
    monkeypatch, url, settings, status
):
    session = _Session()
    _configure(monkeypatch, session=session, **settings, **OPEN)

    with pytest.raises(LegacyHTTPError) as error:
        await LegacyModelBridge(FakeGateway()).fetch_image(url)

    assert error.value.status_code == status
    assert error.value.message == "Could not fetch image from URL."
    assert session.requested == []


@pytest.mark.parametrize(
    "url,switches",
    [
        (HTTP_URL, {"allow_non_https": True}),
        ("https://192.168.1.5/a.jpg", WITHOUT_FQDN),
        (A, {}),
    ],
)
@pytest.mark.asyncio
async def test_workflow_image_fetch_fetches_what_the_route_fetches(
    monkeypatch, url, switches
):
    session = _Session()
    _configure(monkeypatch, session=session, **switches, **OPEN)

    data = await LegacyModelBridge(FakeGateway()).fetch_image(url)

    assert data == _jpeg()
    assert session.requested == [url]


@pytest.mark.parametrize(
    "switches,requested,status",
    [({}, [A, BLOCKED_URL], None), (VALIDATING, [A], 403)],
    ids=["default", "validating"],
)
@pytest.mark.asyncio
async def test_workflow_image_fetch_follows_redirects_like_the_route(
    monkeypatch, switches, requested, status
):
    session = _Session(redirects={A: BLOCKED_URL})
    _configure(
        monkeypatch,
        session=session,
        blocked=frozenset({"blocked.example.com"}),
        **switches,
        **OPEN,
    )
    bridge = LegacyModelBridge(FakeGateway())

    if status is None:
        assert await bridge.fetch_image(A) == _jpeg()
    else:
        with pytest.raises(LegacyHTTPError) as error:
            await bridge.fetch_image(A)
        assert error.value.status_code == status
    assert session.requested == requested


@pytest.mark.parametrize(
    "url", [HTTP_URL, "https://93.184.216.34/a.jpg", "https://localhost/a.jpg"]
)
@pytest.mark.asyncio
async def test_shared_fetch_keeps_fetching_non_https_and_bare_hosts(monkeypatch, url):
    session = _Session()
    _configure(monkeypatch, session=session)

    images, error = await fetch_images_from_urls([url])

    assert error is None
    assert images == [_jpeg()]
    assert session.requested == [url]
