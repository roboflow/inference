import ast
import base64
import inspect
import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from inference.core.utils import image_utils as server_image_utils
from inference.core.workflows.errors import WorkflowImageLoadError
from inference.core.workflows.utils.image_encoding import (
    choose_image_decoding_flags,
    convert_gray_image_to_bgr,
    decode_base64_image,
    decode_encoded_image_bytes,
    encode_image_to_jpeg_bytes,
    ensure_valid_numpy_image,
)

# 48 rows x 64 cols, deliberately NOT square so an EXIF rotation is visible.
COLOUR_IMAGE = np.dstack(
    [
        np.tile(np.arange(64, dtype=np.uint8), (48, 1)),
        np.full((48, 64), 17, dtype=np.uint8),
        np.tile(np.arange(48, dtype=np.uint8).reshape(48, 1), (1, 64)),
    ]
)
GRAY_IMAGE = np.tile(np.arange(64, dtype=np.uint8), (48, 1))


def _jpeg_with_exif_orientation_6() -> bytes:
    """A 48x64 JPEG whose EXIF says 'rotate 90 CW'.

    cv2.IMREAD_COLOR honours that tag and returns (64, 48, 3);
    cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION returns (48, 64, 3).
    That difference is the whole point of `cv_imread_flags`.
    """
    pillow_image = Image.fromarray(COLOUR_IMAGE[:, :, ::-1])  # BGR -> RGB
    exif = pillow_image.getexif()
    exif[274] = 6  # Orientation
    buffer = io.BytesIO()
    pillow_image.save(buffer, format="JPEG", exif=exif, quality=95)
    return buffer.getvalue()


@pytest.mark.parametrize("jpeg_quality", [1, 30, 90, 95, 100])
def test_vendored_jpeg_encoder_is_byte_identical_to_the_server_encoder(
    jpeg_quality: int,
) -> None:
    # A JPEG encoder that "looks right" but differs by a byte changes every
    # base64 payload every VLM block sends. Byte equality, not visual equality.
    for image in (COLOUR_IMAGE, GRAY_IMAGE):
        assert encode_image_to_jpeg_bytes(
            image, jpeg_quality=jpeg_quality
        ) == server_image_utils.encode_image_to_jpeg_bytes(
            image, jpeg_quality=jpeg_quality
        )


@pytest.mark.parametrize(
    "ours,theirs",
    [
        ("encode_image_to_jpeg_bytes", "encode_image_to_jpeg_bytes"),
        ("choose_image_decoding_flags", "choose_image_decoding_flags"),
        ("convert_gray_image_to_bgr", "convert_gray_image_to_bgr"),
        ("decode_base64_image", "load_image_base64"),
        ("decode_encoded_image_bytes", "load_image_from_encoded_bytes"),
    ],
)
def test_vendored_signatures_match_the_server_originals(ours: str, theirs: str) -> None:
    import inference.core.workflows.utils.image_encoding as vendored

    mine = inspect.signature(getattr(vendored, ours))
    reference = inspect.signature(getattr(server_image_utils, theirs))
    assert [(p.name, p.kind, p.default) for p in mine.parameters.values()] == [
        (p.name, p.kind, p.default) for p in reference.parameters.values()
    ]


def test_vendored_decoding_flags_match_the_server_helper() -> None:
    for disable in (True, False):
        assert choose_image_decoding_flags(
            disable_preproc_auto_orient=disable
        ) == server_image_utils.choose_image_decoding_flags(
            disable_preproc_auto_orient=disable
        )


def test_vendored_gray_to_bgr_matches_the_server_helper() -> None:
    for image in (COLOUR_IMAGE, GRAY_IMAGE, GRAY_IMAGE.reshape(48, 64, 1)):
        assert np.array_equal(
            convert_gray_image_to_bgr(image.copy()),
            server_image_utils.convert_gray_image_to_bgr(image.copy()),
        )


def test_vendored_base64_decoder_matches_the_server_decoder() -> None:
    payload = base64.b64encode(
        server_image_utils.encode_image_to_jpeg_bytes(COLOUR_IMAGE)
    ).decode("ascii")
    for value in (
        payload,
        payload.encode("ascii"),
        f"data:image/jpeg;base64,{payload}",
    ):
        assert np.array_equal(
            decode_base64_image(value),
            server_image_utils.load_image_base64(value),
        )


@pytest.mark.parametrize("disable_preproc_auto_orient", [True, False])
def test_vendored_base64_decoder_honours_exif_flags_exactly(
    disable_preproc_auto_orient: bool,
) -> None:
    # Round-1 Defect 5: dropping cv_imread_flags silently rotated images.
    payload = base64.b64encode(_jpeg_with_exif_orientation_6()).decode("ascii")
    flags = choose_image_decoding_flags(
        disable_preproc_auto_orient=disable_preproc_auto_orient
    )
    ours = decode_base64_image(payload, cv_imread_flags=flags)
    theirs = server_image_utils.load_image_base64(payload, cv_imread_flags=flags)
    assert ours.shape == theirs.shape
    assert np.array_equal(ours, theirs)


def test_the_exif_fixture_actually_distinguishes_the_two_flag_settings() -> None:
    # Guards the guard: if Pillow ever stopped writing the tag, the test above
    # would pass vacuously.
    payload = base64.b64encode(_jpeg_with_exif_orientation_6()).decode("ascii")
    honoured = decode_base64_image(
        payload, cv_imread_flags=choose_image_decoding_flags(False)
    )
    ignored = decode_base64_image(
        payload, cv_imread_flags=choose_image_decoding_flags(True)
    )
    assert honoured.shape[:2] != ignored.shape[:2]


def test_vendored_encoded_bytes_decoder_matches_the_server_decoder() -> None:
    raw = server_image_utils.encode_image_to_jpeg_bytes(COLOUR_IMAGE)
    assert np.array_equal(
        decode_encoded_image_bytes(raw),
        server_image_utils.load_image_from_encoded_bytes(raw),
    )


def test_vendored_decoders_raise_a_workflows_error_not_a_server_error() -> None:
    with pytest.raises(WorkflowImageLoadError):
        decode_base64_image("!!!not base64!!!")
    with pytest.raises(WorkflowImageLoadError):
        decode_base64_image("")
    with pytest.raises(WorkflowImageLoadError):
        decode_encoded_image_bytes(b"not an image")


def test_ensure_valid_numpy_image_accepts_and_rejects_like_the_server() -> None:
    assert ensure_valid_numpy_image(COLOUR_IMAGE) is COLOUR_IMAGE
    assert ensure_valid_numpy_image(GRAY_IMAGE) is GRAY_IMAGE
    for bad in (np.zeros((4,), dtype=np.uint8), np.zeros((4, 4, 5), dtype=np.uint8)):
        with pytest.raises(WorkflowImageLoadError):
            ensure_valid_numpy_image(bad)
        with pytest.raises(Exception):
            server_image_utils.validate_numpy_image(bad)


def test_module_does_not_import_the_server_package_or_do_io() -> None:
    source = Path("inference/core/workflows/utils/image_encoding.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    forbidden = {
        m
        for m in modules
        if (m == "inference" or m.startswith("inference."))
        and not m.startswith("inference.core.workflows")
    }
    assert not forbidden, forbidden
    assert not modules & {"requests", "socket", "urllib", "pickle", "tldextract", "os"}
