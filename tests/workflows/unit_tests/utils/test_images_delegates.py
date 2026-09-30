import inspect
from unittest import mock

import cv2
import numpy as np
import pytest

from inference.core.utils import image_utils as server_image_utils
from inference.core.workflows.errors import WorkflowImageLoadError
from inference.core.workflows.prototypes.image_codec import (
    reset_image_codec,
    set_image_codec,
)
from inference.core.workflows.utils import images as workflows_images
from inference.core.workflows.utils.images import (
    attempt_loading_image_from_string,
    encode_image_to_jpeg_bytes,
    ensure_local_image_load_allowed,
    load_image,
    load_image_from_url,
)

IMAGE = np.zeros((8, 12, 3), dtype=np.uint8)


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_image_codec()
    yield
    reset_image_codec()


class _RecordingCodec:
    def __init__(self):
        self.calls = []

    def load_image(self, value, disable_preproc_auto_orient=False):
        self.calls.append(("load_image", value, disable_preproc_auto_orient))
        return IMAGE, True

    def fetch_url(self, value, cv_imread_flags=cv2.IMREAD_COLOR):
        self.calls.append(("fetch_url", value, cv_imread_flags))
        return IMAGE

    def decode_string(self, value, cv_imread_flags=cv2.IMREAD_COLOR):
        self.calls.append(("decode_string", value, cv_imread_flags))
        return IMAGE, True

    def ensure_local_file_load_allowed(self, path):
        self.calls.append(("ensure_local_file_load_allowed", path))


@pytest.mark.parametrize(
    "name",
    [
        "load_image",
        "load_image_from_url",
        "attempt_loading_image_from_string",
        "encode_image_to_jpeg_bytes",
    ],
)
def test_delegate_names_shadow_the_server_names_exactly(name: str) -> None:
    # The codemod in Task 10.8 rewrites only the module path in each import
    # statement, so every name the repointed files import must exist here.
    assert hasattr(workflows_images, name)


@pytest.mark.parametrize(
    "name",
    ["load_image", "load_image_from_url", "attempt_loading_image_from_string"],
)
def test_delegate_signatures_match_the_server_functions(name: str) -> None:
    # Round-1 Defect 5: round-0 dropped cv_imread_flags from two of the three.
    ours = inspect.signature(getattr(workflows_images, name))
    theirs = inspect.signature(getattr(server_image_utils, name))
    assert [(p.name, p.kind, p.default) for p in ours.parameters.values()] == [
        (p.name, p.kind, p.default) for p in theirs.parameters.values()
    ]


def test_load_image_from_url_keeps_the_value_keyword() -> None:
    # base.py:573/627 and deserializers.py:123 call it as
    # `load_image_from_url(value=...)` and existing tests assert on that keyword.
    assert list(inspect.signature(load_image_from_url).parameters)[0] == "value"


def test_delegates_route_through_the_installed_codec() -> None:
    codec = _RecordingCodec()
    set_image_codec(codec)

    assert load_image({"type": "numpy_object", "value": IMAGE})[1] is True
    assert load_image_from_url(value="https://example.com/i.jpg") is IMAGE
    assert attempt_loading_image_from_string("payload")[0] is IMAGE
    ensure_local_image_load_allowed("/tmp/x.jpg")

    assert codec.calls == [
        ("load_image", {"type": "numpy_object", "value": IMAGE}, False),
        ("fetch_url", "https://example.com/i.jpg", cv2.IMREAD_COLOR),
        ("decode_string", "payload", cv2.IMREAD_COLOR),
        ("ensure_local_file_load_allowed", "/tmp/x.jpg"),
    ]


def test_delegates_forward_decoding_flags() -> None:
    codec = _RecordingCodec()
    set_image_codec(codec)
    flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION

    load_image_from_url(value="https://example.com/i.jpg", cv_imread_flags=flags)
    attempt_loading_image_from_string("payload", cv_imread_flags=flags)
    load_image({"type": "base64", "value": "x"}, disable_preproc_auto_orient=True)

    assert codec.calls == [
        ("fetch_url", "https://example.com/i.jpg", flags),
        ("decode_string", "payload", flags),
        ("load_image", {"type": "base64", "value": "x"}, True),
    ]


def test_delegates_refuse_when_no_host_codec_is_installed() -> None:
    with pytest.raises(WorkflowImageLoadError):
        load_image_from_url(value="https://example.com/i.jpg")
    with pytest.raises(WorkflowImageLoadError):
        ensure_local_image_load_allowed("/etc/passwd")
    with pytest.raises(WorkflowImageLoadError):
        load_image({"type": "url", "value": "https://example.com/i.jpg"})


def test_encode_is_not_routed_through_the_codec() -> None:
    # D3: JPEG encoding is pure cv2, vendored rather than injected. It must work
    # with no codec installed at all.
    assert encode_image_to_jpeg_bytes(
        IMAGE, jpeg_quality=95
    ) == server_image_utils.encode_image_to_jpeg_bytes(IMAGE, jpeg_quality=95)


def test_module_level_patching_still_works() -> None:
    # Four existing tests do `mock.patch.object(<module>, "load_image_from_url")`.
    # That relies on the name being a module attribute, not a method lookup.
    with mock.patch.object(workflows_images, "load_image_from_url") as patched:
        patched.return_value = IMAGE
        assert (
            workflows_images.load_image_from_url(value="https://x.example.com/y.jpg")
            is IMAGE
        )
    patched.assert_called_once_with(value="https://x.example.com/y.jpg")
