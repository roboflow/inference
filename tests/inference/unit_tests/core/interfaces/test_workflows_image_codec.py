"""Security parity for the injected image codec.

Every guard `inference/core/workflows` used to reach directly is exercised HERE,
through `ServerImageCodec`, never through `image_utils` directly - so an adapter
that stopped forwarding, or started reimplementing, fails these tests.

Idioms are the ones the existing image-loading suite uses
(`tests/inference/unit_tests/core/utils/test_image_utils.py`):

* env values are module-level constants captured at import, so
  `mock.patch.object(image_utils, "<CONST>", value)` is what takes effect;
* `requests_mock` intercepts at the transport-adapter level, which REPLACES the
  SSRF-protected adapter - so address-classification tests must not use it;
* `mock.patch.object(url_input.socket, "getaddrinfo", ...)` covers DNS.

Hostnames are subdomains of `example.com`, never `*.example`: under the packaged
suffix list the latter has an EMPTY fqdn and is rejected at `image_utils.py:454`
before any allow/deny-list check ever runs (verified:
`tldextract.TLDExtract(suffix_list_urls=())("cdn.allowed.example").fqdn == ""`).
"""

import socket
from unittest import mock

import cv2
import numpy as np
import pytest
from requests_mock import Mocker
from urllib3.connectionpool import HTTPSConnectionPool

from inference.core.exceptions import InputImageLoadError, InvalidImageTypeDeclared
from inference.core.interfaces.workflows_image_codec import (
    GUARDED_IMAGE_CODEC,
    ServerImageCodec,
    bind_image_codec,
    install_guarded_image_codec,
    resolve_image_codec,
)
from inference.core.utils import image_utils, url_input
from inference.core.workflows.prototypes.image_codec import (
    ImageCodec,
    get_image_codec,
    reset_image_codec,
)

CODEC = ServerImageCodec()

ALLOWED_HOST = "cdn.allowed.example.com"
DENIED_HOST = "metadata.internal.example.com"
REBINDING_HOST = "evil.example.com"


@pytest.fixture(autouse=True)
def _clean_registry_and_proxy_env(monkeypatch):
    # A developer's HTTP(S)_PROXY defers IP pinning and emits a warning
    # (url_input.py:225-227), which would make the pinning test pass vacuously.
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")
    reset_image_codec()
    yield
    reset_image_codec()


def _fake_getaddrinfo(ip: str):
    def _inner(host, port, *args, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port))
        ]

    return _inner


# --------------------------------------------------------------------------
# URL policy: enablement
# --------------------------------------------------------------------------


@mock.patch.object(image_utils, "ALLOW_URL_INPUT", False)
def test_adapter_refuses_url_when_url_input_is_disabled() -> None:
    with pytest.raises(InvalidImageTypeDeclared):
        CODEC.fetch_url(f"https://{ALLOWED_HOST}/image.jpg")


@mock.patch.object(image_utils, "_fetch_image_bytes_from_url")
@mock.patch.object(image_utils, "_validate_url_destination")
@mock.patch.object(image_utils, "OFFLINE_MODE", True)
def test_adapter_refuses_url_in_offline_mode_before_touching_the_network(
    validate_url_destination_mock: mock.MagicMock,
    fetch_image_bytes_mock: mock.MagicMock,
) -> None:
    with pytest.raises(InputImageLoadError, match="OFFLINE_MODE"):
        CODEC.fetch_url(f"https://{ALLOWED_HOST}/image.jpg")
    validate_url_destination_mock.assert_not_called()
    fetch_image_bytes_mock.assert_not_called()


# --------------------------------------------------------------------------
# URL policy: scheme
# --------------------------------------------------------------------------


@mock.patch.object(image_utils, "ALLOW_URL_INPUT", True)
@mock.patch.object(image_utils, "ALLOW_NON_HTTPS_URL_INPUT", False)
@mock.patch.object(image_utils, "ALLOW_URL_INPUT_WITHOUT_FQDN", True)
@mock.patch.object(image_utils, "WHITELISTED_DESTINATIONS_FOR_URL_INPUT", None)
@mock.patch.object(image_utils, "BLACKLISTED_DESTINATIONS_FOR_URL_INPUT", None)
@pytest.mark.parametrize(
    "url",
    [
        "http://cdn.allowed.example.com/image.jpg",
        "ftp://cdn.allowed.example.com/image.jpg",
        "file:///etc/passwd",
        "gopher://127.0.0.1:11211/_stats",
    ],
)
def test_adapter_rejects_non_https_schemes(url: str) -> None:
    with pytest.raises(InputImageLoadError, match="non https"):
        CODEC.fetch_url(url)


# --------------------------------------------------------------------------
# URL policy: FQDN, allow-list, deny-list
# --------------------------------------------------------------------------


@mock.patch.object(image_utils, "ALLOW_URL_INPUT", True)
@mock.patch.object(image_utils, "ALLOW_NON_HTTPS_URL_INPUT", False)
@mock.patch.object(image_utils, "ALLOW_URL_INPUT_WITHOUT_FQDN", False)
@mock.patch.object(image_utils, "WHITELISTED_DESTINATIONS_FOR_URL_INPUT", None)
@mock.patch.object(image_utils, "BLACKLISTED_DESTINATIONS_FOR_URL_INPUT", None)
def test_adapter_rejects_url_without_fqdn() -> None:
    with pytest.raises(InputImageLoadError, match="FQDN"):
        CODEC.fetch_url("https://127.0.0.1/image.jpg")


@mock.patch.object(image_utils, "ALLOW_URL_INPUT", True)
@mock.patch.object(image_utils, "ALLOW_NON_HTTPS_URL_INPUT", False)
@mock.patch.object(image_utils, "ALLOW_URL_INPUT_WITHOUT_FQDN", False)
@mock.patch.object(image_utils, "BLACKLISTED_DESTINATIONS_FOR_URL_INPUT", None)
@mock.patch.object(
    image_utils, "WHITELISTED_DESTINATIONS_FOR_URL_INPUT", {ALLOWED_HOST}
)
def test_adapter_rejects_a_destination_outside_the_allow_list() -> None:
    with pytest.raises(InputImageLoadError, match="whitelisted"):
        CODEC.fetch_url("https://not.allowed.example.com/image.jpg")


@mock.patch.object(image_utils, "VALIDATE_IMAGE_URL_REDIRECTS", False)
@mock.patch.object(image_utils, "ALLOW_URL_TO_NON_GLOBAL_ADDRESSES", True)
@mock.patch.object(image_utils, "ALLOW_URL_INPUT", True)
@mock.patch.object(image_utils, "ALLOW_NON_HTTPS_URL_INPUT", False)
@mock.patch.object(image_utils, "ALLOW_URL_INPUT_WITHOUT_FQDN", False)
@mock.patch.object(image_utils, "BLACKLISTED_DESTINATIONS_FOR_URL_INPUT", None)
@mock.patch.object(
    image_utils, "WHITELISTED_DESTINATIONS_FOR_URL_INPUT", {ALLOWED_HOST}
)
def test_adapter_accepts_an_allow_listed_destination(
    requests_mock: Mocker,
    image_as_numpy: np.ndarray,
    image_as_png_bytes: bytes,
) -> None:
    url = f"https://{ALLOWED_HOST}/image.png"
    requests_mock.get(url, content=image_as_png_bytes)

    result = CODEC.fetch_url(url)

    assert np.allclose(image_as_numpy, result)


@mock.patch.object(image_utils, "ALLOW_URL_INPUT", True)
@mock.patch.object(image_utils, "ALLOW_NON_HTTPS_URL_INPUT", False)
@mock.patch.object(image_utils, "ALLOW_URL_INPUT_WITHOUT_FQDN", False)
@mock.patch.object(image_utils, "WHITELISTED_DESTINATIONS_FOR_URL_INPUT", None)
@mock.patch.object(image_utils, "BLACKLISTED_DESTINATIONS_FOR_URL_INPUT", {DENIED_HOST})
def test_adapter_rejects_a_deny_listed_destination() -> None:
    with pytest.raises(InputImageLoadError, match="blacklisted"):
        CODEC.fetch_url(f"https://{DENIED_HOST}/latest/meta-data")


@mock.patch.object(image_utils, "ALLOW_URL_INPUT", True)
@mock.patch.object(image_utils, "ALLOW_NON_HTTPS_URL_INPUT", False)
@mock.patch.object(image_utils, "ALLOW_URL_INPUT_WITHOUT_FQDN", True)
@mock.patch.object(image_utils, "BLACKLISTED_DESTINATIONS_FOR_URL_INPUT", None)
@mock.patch.object(
    image_utils, "WHITELISTED_DESTINATIONS_FOR_URL_INPUT", {ALLOWED_HOST}
)
def test_adapter_rejects_the_backslash_authority_allow_list_bypass() -> None:
    # `https://localhost:6666\@cdn.allowed.example.com/x` parses one way in a
    # browser and another in urllib; the guard rejects backslashes in the
    # authority (image_utils.py:454-455, raising `ValueError("URL authority
    # contains a backslash")`, converted at :459-464 into
    # `InputImageLoadError("Provided image URL is invalid")` with `from
    # error`). Pinned to that specific guard - not just "any
    # InputImageLoadError" - so the test cannot pass for the wrong reason
    # (e.g. the allow-list rejecting the `localhost` authority instead).
    with mock.patch.object(image_utils, "_fetch_image_bytes_from_url") as fetch_mock:
        with pytest.raises(
            InputImageLoadError, match="Provided image URL is invalid"
        ) as error:
            CODEC.fetch_url(f"https://localhost:6666\\@{ALLOWED_HOST}/image.jpg")
        fetch_mock.assert_not_called()
    assert isinstance(error.value.__cause__, ValueError)
    assert "backslash" in str(error.value.__cause__)


# --------------------------------------------------------------------------
# URL policy: redirects
# --------------------------------------------------------------------------


@mock.patch.object(image_utils, "VALIDATE_IMAGE_URL_REDIRECTS", True)
@mock.patch.object(image_utils, "ALLOW_URL_TO_NON_GLOBAL_ADDRESSES", True)
@mock.patch.object(image_utils, "ALLOW_URL_INPUT", True)
@mock.patch.object(image_utils, "ALLOW_NON_HTTPS_URL_INPUT", False)
@mock.patch.object(image_utils, "ALLOW_URL_INPUT_WITHOUT_FQDN", False)
@mock.patch.object(image_utils, "WHITELISTED_DESTINATIONS_FOR_URL_INPUT", None)
@mock.patch.object(image_utils, "BLACKLISTED_DESTINATIONS_FOR_URL_INPUT", {DENIED_HOST})
def test_adapter_re_validates_every_redirect_hop(requests_mock: Mocker) -> None:
    start = f"https://{ALLOWED_HOST}/image.jpg"
    internal = f"https://{DENIED_HOST}/latest/meta-data"
    requests_mock.get(start, status_code=302, headers={"Location": internal})

    with pytest.raises(InputImageLoadError, match="blacklisted"):
        CODEC.fetch_url(start)


@mock.patch.object(image_utils, "VALIDATE_IMAGE_URL_REDIRECTS", True)
@mock.patch.object(image_utils, "ALLOW_URL_TO_NON_GLOBAL_ADDRESSES", True)
@mock.patch.object(image_utils, "ALLOW_URL_INPUT", True)
@mock.patch.object(image_utils, "ALLOW_NON_HTTPS_URL_INPUT", False)
@mock.patch.object(image_utils, "ALLOW_URL_INPUT_WITHOUT_FQDN", False)
@mock.patch.object(image_utils, "WHITELISTED_DESTINATIONS_FOR_URL_INPUT", None)
@mock.patch.object(image_utils, "BLACKLISTED_DESTINATIONS_FOR_URL_INPUT", None)
@mock.patch.object(image_utils, "MAX_IMAGE_URL_REDIRECTS", 2)
def test_adapter_enforces_the_redirect_hop_cap(requests_mock: Mocker) -> None:
    for index in range(6):
        requests_mock.get(
            f"https://hop{index}.example.com/image.jpg",
            status_code=302,
            headers={"Location": f"https://hop{index + 1}.example.com/image.jpg"},
        )

    # `fetch_url_content_validating_redirects` raises `TooManyRedirects`
    # (url_input.py:386-387), and `load_image_from_url`'s
    # `except (RequestException, ConnectionError)` (image_utils.py:437-441)
    # folds it into the message WITHOUT `from` - so pin on the surfaced text
    # rather than `__cause__`, which is not set here.
    with pytest.raises(InputImageLoadError, match="Exceeded maximum of 2 redirects"):
        CODEC.fetch_url("https://hop0.example.com/image.jpg")

    # `range(max_redirects + 1)` (url_input.py:366) allows 3 requests before
    # raising: hop0, hop1, hop2.
    assert requests_mock.call_count == 3


# --------------------------------------------------------------------------
# URL policy: address classification.
# NO requests_mock here: it replaces the transport adapter, and the SSRF adapter
# would never run.
# --------------------------------------------------------------------------


@mock.patch.object(image_utils, "VALIDATE_IMAGE_URL_REDIRECTS", False)
@mock.patch.object(image_utils, "ALLOW_URL_TO_NON_GLOBAL_ADDRESSES", False)
@mock.patch.object(image_utils, "ALLOW_URL_INPUT", True)
@mock.patch.object(image_utils, "ALLOW_NON_HTTPS_URL_INPUT", False)
@mock.patch.object(image_utils, "ALLOW_URL_INPUT_WITHOUT_FQDN", True)
@mock.patch.object(image_utils, "WHITELISTED_DESTINATIONS_FOR_URL_INPUT", None)
@mock.patch.object(image_utils, "BLACKLISTED_DESTINATIONS_FOR_URL_INPUT", None)
@pytest.mark.parametrize(
    "url",
    [
        "https://10.0.0.1/image.jpg",
        "https://127.0.0.1/image.jpg",
        "https://169.254.169.254/latest/meta-data",
        "https://[::1]/image.jpg",
        "https://[::ffff:127.0.0.1]/image.jpg",
        "https://100.64.0.1/image.jpg",
    ],
)
def test_adapter_blocks_non_global_address_literals(url: str) -> None:
    with pytest.raises(InputImageLoadError) as error:
        CODEC.fetch_url(url)
    assert "not allowed" in str(error.value).lower()


@mock.patch.object(image_utils, "VALIDATE_IMAGE_URL_REDIRECTS", False)
@mock.patch.object(image_utils, "ALLOW_URL_TO_NON_GLOBAL_ADDRESSES", False)
@mock.patch.object(image_utils, "ALLOW_URL_INPUT", True)
@mock.patch.object(image_utils, "ALLOW_NON_HTTPS_URL_INPUT", False)
@mock.patch.object(image_utils, "ALLOW_URL_INPUT_WITHOUT_FQDN", False)
@mock.patch.object(image_utils, "WHITELISTED_DESTINATIONS_FOR_URL_INPUT", None)
@mock.patch.object(image_utils, "BLACKLISTED_DESTINATIONS_FOR_URL_INPUT", None)
def test_adapter_blocks_a_public_hostname_resolving_to_loopback(monkeypatch) -> None:
    resolved = []

    def _rebinding_getaddrinfo(host, port, *args, **kwargs):
        resolved.append(host)
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("127.0.0.1", port),
            )
        ]

    monkeypatch.setattr(url_input.socket, "getaddrinfo", _rebinding_getaddrinfo)

    with pytest.raises(InputImageLoadError) as error:
        CODEC.fetch_url(f"https://{REBINDING_HOST}/image.jpg")

    assert "not allowed" in str(error.value).lower()
    # The patch is process-global and background threads may resolve unrelated
    # hosts; assert on the target host only.
    assert resolved.count(REBINDING_HOST) == 1


@mock.patch.object(image_utils, "VALIDATE_IMAGE_URL_REDIRECTS", False)
@mock.patch.object(image_utils, "ALLOW_URL_TO_NON_GLOBAL_ADDRESSES", False)
@mock.patch.object(image_utils, "ALLOW_URL_INPUT", True)
@mock.patch.object(image_utils, "ALLOW_NON_HTTPS_URL_INPUT", False)
@mock.patch.object(image_utils, "ALLOW_URL_INPUT_WITHOUT_FQDN", False)
@mock.patch.object(image_utils, "WHITELISTED_DESTINATIONS_FOR_URL_INPUT", None)
@mock.patch.object(image_utils, "BLACKLISTED_DESTINATIONS_FOR_URL_INPUT", None)
def test_adapter_pins_the_connection_and_preserves_the_tls_hostname(
    monkeypatch,
) -> None:
    # Round-1 Defect 7: rejecting loopback does not prove pinning. This asserts
    # the pool the fetch actually opens is pinned to the validated IP while
    # certificate verification, SNI and the `Host` header still target the
    # original hostname (url_input.py:159-167, :195-219).
    captured = {}

    class _StopBeforeSocket(Exception):
        pass

    def _capturing_urlopen(self, *args, **kwargs):
        captured["host"] = self.host
        captured["assert_hostname"] = getattr(self, "assert_hostname", None)
        # `conn_kw["server_hostname"]` IS the SNI setting (url_input.py:217).
        captured["server_hostname"] = self.conn_kw.get("server_hostname")
        # The adapter rewrites the request's `Host` header to the original
        # hostname before dialling the pinned IP (url_input.py:159-167); the
        # pool receives it in `urlopen(headers=...)`.
        captured["host_header"] = kwargs["headers"].get("Host")
        raise _StopBeforeSocket()

    monkeypatch.setattr(
        url_input.socket, "getaddrinfo", _fake_getaddrinfo("93.184.216.34")
    )
    monkeypatch.setattr(HTTPSConnectionPool, "urlopen", _capturing_urlopen)

    with pytest.raises(_StopBeforeSocket):
        CODEC.fetch_url(f"https://{ALLOWED_HOST}/image.jpg")

    # Pinned to the validated IP...
    assert captured["host"] == "93.184.216.34", captured
    # ...while certificate verification and SNI both still target the hostname.
    assert captured["assert_hostname"] == ALLOWED_HOST, captured
    assert captured["server_hostname"] == ALLOWED_HOST, captured
    # ...and the vhost the origin sees is still the hostname, not the IP
    # (round-3 Defect 2).
    assert captured["host_header"] == ALLOWED_HOST, captured


# --------------------------------------------------------------------------
# Local filesystem policy - one owner, in image_utils
# --------------------------------------------------------------------------


@mock.patch.object(image_utils, "ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM", True)
def test_adapter_permits_local_file_loads_when_the_flag_is_on(
    image_as_local_path: str,
) -> None:
    assert CODEC.ensure_local_file_load_allowed(image_as_local_path) is None


@mock.patch.object(image_utils, "ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM", False)
def test_adapter_refuses_local_file_loads_when_the_flag_is_off(
    image_as_local_path: str,
) -> None:
    with pytest.raises(InputImageLoadError, match="local filesystem"):
        CODEC.ensure_local_file_load_allowed(image_as_local_path)


def test_adapter_calls_the_single_owner_rather_than_restating_the_rule() -> None:
    # Round-1 Defect 6: the adapter must not hold its own flag snapshot.
    with mock.patch.object(image_utils, "ensure_local_file_load_allowed") as guard:
        CODEC.ensure_local_file_load_allowed("/tmp/whatever.jpg")
    guard.assert_called_once_with()


@mock.patch.object(image_utils, "ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM", False)
def test_adapter_load_image_refuses_a_declared_file_payload_when_disabled(
    image_as_local_path: str,
) -> None:
    with pytest.raises(InputImageLoadError, match="local filesystem"):
        CODEC.load_image({"type": "file", "value": image_as_local_path})


# --------------------------------------------------------------------------
# Pickle gate
# --------------------------------------------------------------------------


@mock.patch.object(image_utils, "ALLOW_NUMPY_INPUT", False)
def test_adapter_decode_string_keeps_the_pickle_gate_shut_by_default(
    image_as_base64_encoded_pickled_bytes: bytes,
) -> None:
    with pytest.raises(InputImageLoadError):
        CODEC.decode_string(image_as_base64_encoded_pickled_bytes)


@mock.patch.object(image_utils, "ALLOW_NUMPY_INPUT", True)
def test_adapter_decode_string_honours_an_opened_pickle_gate(
    image_as_base64_encoded_pickled_bytes: bytes,
    image_as_numpy: np.ndarray,
) -> None:
    decoded, is_bgr = CODEC.decode_string(image_as_base64_encoded_pickled_bytes)
    assert is_bgr is True
    assert np.allclose(decoded, image_as_numpy)


@mock.patch.object(image_utils, "ALLOW_NUMPY_INPUT", False)
def test_adapter_load_image_refuses_a_declared_numpy_payload_when_disabled(
    image_as_base64_encoded_pickled_bytes: bytes,
) -> None:
    with pytest.raises(InvalidImageTypeDeclared):
        CODEC.load_image(
            {"type": "numpy", "value": image_as_base64_encoded_pickled_bytes}
        )


# --------------------------------------------------------------------------
# Happy paths, flags and wiring
# --------------------------------------------------------------------------


def test_adapter_decodes_base64_like_the_server(
    image_as_jpeg_base64_string: str,
) -> None:
    decoded, is_bgr = CODEC.decode_string(image_as_jpeg_base64_string)
    reference, reference_is_bgr = image_utils.attempt_loading_image_from_string(
        value=image_as_jpeg_base64_string
    )
    assert is_bgr == reference_is_bgr
    assert np.array_equal(decoded, reference)


@pytest.mark.parametrize("disable_preproc_auto_orient", [True, False])
def test_adapter_forwards_decoding_flags(
    disable_preproc_auto_orient: bool, image_as_jpeg_base64_string: str
) -> None:
    flags = image_utils.choose_image_decoding_flags(
        disable_preproc_auto_orient=disable_preproc_auto_orient
    )
    ours, _ = CODEC.decode_string(image_as_jpeg_base64_string, cv_imread_flags=flags)
    theirs, _ = image_utils.attempt_loading_image_from_string(
        value=image_as_jpeg_base64_string, cv_imread_flags=flags
    )
    assert np.array_equal(ours, theirs)

    ours_dispatch, _ = CODEC.load_image(
        {"type": "base64", "value": image_as_jpeg_base64_string},
        disable_preproc_auto_orient=disable_preproc_auto_orient,
    )
    theirs_dispatch, _ = image_utils.load_image(
        {"type": "base64", "value": image_as_jpeg_base64_string},
        disable_preproc_auto_orient=disable_preproc_auto_orient,
    )
    assert np.array_equal(ours_dispatch, theirs_dispatch)


@pytest.mark.parametrize(
    "cv_imread_flags",
    [cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED],
)
def test_adapter_decode_string_returns_exactly_what_the_server_decoder_returns(
    cv_imread_flags: int, image_as_jpeg_base64_string: str
) -> None:
    # Round-2 Defect 8, adapter side: the forward must not reshape the result.
    ours, ours_is_bgr = CODEC.decode_string(
        image_as_jpeg_base64_string, cv_imread_flags=cv_imread_flags
    )
    theirs, theirs_is_bgr = image_utils.attempt_loading_image_from_string(
        value=image_as_jpeg_base64_string, cv_imread_flags=cv_imread_flags
    )
    assert ours.shape == theirs.shape
    assert ours_is_bgr == theirs_is_bgr
    assert np.array_equal(ours, theirs)


def test_adapter_load_image_matches_the_server_loader_for_numpy_payloads(
    image_as_numpy: np.ndarray,
) -> None:
    ours = CODEC.load_image({"type": "numpy_object", "value": image_as_numpy})
    theirs = image_utils.load_image({"type": "numpy_object", "value": image_as_numpy})
    assert np.array_equal(ours[0], theirs[0])
    assert ours[1] == theirs[1]


@pytest.mark.parametrize(
    "name",
    ["load_image", "fetch_url", "decode_string", "ensure_local_file_load_allowed"],
)
def test_adapter_satisfies_the_port_signatures(name: str) -> None:
    import inspect

    port = inspect.signature(getattr(ImageCodec, name))
    impl = inspect.signature(getattr(ServerImageCodec, name))
    assert [(p.name, p.kind, p.default) for p in port.parameters.values()] == [
        (p.name, p.kind, p.default) for p in impl.parameters.values()
    ]


def test_bind_image_codec_writes_and_installs_the_same_object() -> None:
    init_parameters = {"workflows_core.api_key": "k"}
    bound = bind_image_codec(init_parameters)
    assert bound is GUARDED_IMAGE_CODEC
    assert init_parameters["workflows_core.image_codec"] is GUARDED_IMAGE_CODEC
    assert get_image_codec() is GUARDED_IMAGE_CODEC


def test_bind_image_codec_honours_a_caller_override_on_both_paths() -> None:
    # Round-2 Defect 1: an override must move BOTH paths, not just Path A.
    from inference.core.workflows.prototypes.image_codec import WorkflowsLocalImageCodec

    override = WorkflowsLocalImageCodec()
    init_parameters = {"workflows_core.image_codec": override}
    bound = bind_image_codec(init_parameters)
    assert bound is override
    assert init_parameters["workflows_core.image_codec"] is override
    assert get_image_codec() is override


def test_bind_image_codec_refuses_an_override_that_conflicts_with_an_install() -> None:
    from inference.core.workflows.errors import WorkflowEnvironmentConfigurationError
    from inference.core.workflows.prototypes.image_codec import WorkflowsLocalImageCodec

    install_guarded_image_codec()
    with pytest.raises(WorkflowEnvironmentConfigurationError):
        bind_image_codec({"workflows_core.image_codec": WorkflowsLocalImageCodec()})
    assert get_image_codec() is GUARDED_IMAGE_CODEC


def test_resolve_and_install_share_one_singleton() -> None:
    assert resolve_image_codec() is GUARDED_IMAGE_CODEC
    install_guarded_image_codec()
    assert get_image_codec() is GUARDED_IMAGE_CODEC
    install_guarded_image_codec()  # idempotent
    assert get_image_codec() is GUARDED_IMAGE_CODEC


def test_adapter_forwards_rather_than_reimplements() -> None:
    # Round-1 Defect 3: the round-0 adapter held function ALIASES, so these
    # patches did not land and the real guards ran instead. The adapter must
    # import the MODULE and call attributes.
    sentinel_image = np.zeros((2, 2, 3), dtype=np.uint8)

    with mock.patch.object(image_utils, "load_image_from_url") as forwarded:
        forwarded.return_value = sentinel_image
        assert CODEC.fetch_url(f"https://{ALLOWED_HOST}/i.jpg") is sentinel_image
    forwarded.assert_called_once_with(
        value=f"https://{ALLOWED_HOST}/i.jpg",
        cv_imread_flags=image_utils.cv2.IMREAD_COLOR,
    )

    with mock.patch.object(
        image_utils, "attempt_loading_image_from_string"
    ) as forwarded:
        forwarded.return_value = (sentinel_image, True)
        assert CODEC.decode_string("payload")[0] is sentinel_image
    forwarded.assert_called_once_with(
        value="payload", cv_imread_flags=image_utils.cv2.IMREAD_COLOR
    )

    with mock.patch.object(image_utils, "load_image") as forwarded:
        forwarded.return_value = (sentinel_image, True)
        assert CODEC.load_image({"type": "base64", "value": "x"})[0] is sentinel_image
    forwarded.assert_called_once_with(
        {"type": "base64", "value": "x"}, disable_preproc_auto_orient=False
    )
