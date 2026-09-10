import ast
import base64
import inspect
import io
import pickle
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from inference.core.workflows.errors import (
    WorkflowEnvironmentConfigurationError,
    WorkflowImageLoadError,
)
from inference.core.workflows.prototypes.image_codec import (
    ImageCodec,
    WorkflowsLocalImageCodec,
    get_image_codec,
    reset_image_codec,
    set_image_codec,
)
from inference.core.workflows.utils.image_encoding import (
    choose_image_decoding_flags,
    encode_image_to_jpeg_bytes,
)

IMAGE = np.dstack(
    [
        np.tile(np.arange(64, dtype=np.uint8), (48, 1)),
        np.full((48, 64), 17, dtype=np.uint8),
        np.tile(np.arange(48, dtype=np.uint8).reshape(48, 1), (1, 64)),
    ]
)


@pytest.fixture(autouse=True)
def _clean_registry():
    # The registry is process-wide and set-once; every test here must start and
    # finish with it empty, or the next test inherits an installed codec.
    reset_image_codec()
    yield
    reset_image_codec()


def _b64() -> str:
    return base64.b64encode(encode_image_to_jpeg_bytes(IMAGE)).decode("ascii")


def _b64_with_exif_orientation_6() -> str:
    pillow_image = Image.fromarray(IMAGE[:, :, ::-1])
    exif = pillow_image.getexif()
    exif[274] = 6
    buffer = io.BytesIO()
    pillow_image.save(buffer, format="JPEG", exif=exif, quality=95)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_protocol_declares_exactly_the_four_injected_capabilities() -> None:
    declared = {
        name
        for name, value in vars(ImageCodec).items()
        if not name.startswith("_") and callable(value)
    }
    assert declared == {
        "load_image",
        "fetch_url",
        "decode_string",
        "ensure_local_file_load_allowed",
    }


@pytest.mark.parametrize(
    "name",
    ["load_image", "fetch_url", "decode_string", "ensure_local_file_load_allowed"],
)
def test_default_codec_satisfies_the_protocol_signatures(name: str) -> None:
    port = inspect.signature(getattr(ImageCodec, name))
    impl = inspect.signature(getattr(WorkflowsLocalImageCodec, name))
    assert [(p.name, p.kind, p.default) for p in port.parameters.values()] == [
        (p.name, p.kind, p.default) for p in impl.parameters.values()
    ]


def test_port_signatures_mirror_the_server_functions_they_front() -> None:
    # The delegates in utils/images.py are import-compatible replacements for
    # the image_utils names, so the port must carry cv_imread_flags too.
    from inference.core.utils import image_utils as server_image_utils

    def _tail(signature):
        return [(p.name, p.kind, p.default) for p in signature.parameters.values()][
            1:
        ]  # drop `self`

    assert _tail(inspect.signature(ImageCodec.fetch_url)) == [
        (p.name, p.kind, p.default)
        for p in inspect.signature(
            server_image_utils.load_image_from_url
        ).parameters.values()
    ]
    assert _tail(inspect.signature(ImageCodec.decode_string)) == [
        (p.name, p.kind, p.default)
        for p in inspect.signature(
            server_image_utils.attempt_loading_image_from_string
        ).parameters.values()
    ]
    assert _tail(inspect.signature(ImageCodec.load_image)) == [
        (p.name, p.kind, p.default)
        for p in inspect.signature(server_image_utils.load_image).parameters.values()
    ]


def test_default_refuses_url_fetching() -> None:
    with pytest.raises(WorkflowImageLoadError, match="URL"):
        WorkflowsLocalImageCodec().fetch_url("https://example.com/image.jpg")


def test_default_refuses_local_filesystem_loads(tmp_path) -> None:
    real_file = tmp_path / "present.jpg"
    real_file.write_bytes(encode_image_to_jpeg_bytes(IMAGE))
    with pytest.raises(WorkflowImageLoadError, match="local filesystem"):
        WorkflowsLocalImageCodec().ensure_local_file_load_allowed(str(real_file))


def test_default_load_image_refuses_url_and_file_declarations() -> None:
    codec = WorkflowsLocalImageCodec()
    with pytest.raises(WorkflowImageLoadError, match="URL"):
        codec.load_image({"type": "url", "value": "https://example.com/i.jpg"})
    with pytest.raises(WorkflowImageLoadError, match="local filesystem"):
        codec.load_image({"type": "file", "value": "/etc/passwd"})
    with pytest.raises(WorkflowImageLoadError, match="URL"):
        codec.load_image("https://example.com/i.jpg")


def test_default_refuses_pickled_numpy_payloads() -> None:
    # image_utils.load_image_from_numpy_str reaches pickle.loads behind
    # ALLOW_NUMPY_INPUT. The workflows-local default has no such switch: it
    # never unpickles, whatever the payload claims to be.
    payload = base64.b64encode(pickle.dumps(IMAGE)).decode("ascii")
    codec = WorkflowsLocalImageCodec()
    with pytest.raises(WorkflowImageLoadError):
        codec.load_image({"type": "numpy", "value": payload})
    with pytest.raises(WorkflowImageLoadError):
        codec.decode_string(payload)


def test_default_never_imports_or_calls_pickle() -> None:
    # Round-2 Defect 3: a prose scan (`"pickle" not in source`) fails on the
    # module's own docstrings, which name the gate they refuse to own. Check the
    # AST instead: no import of `pickle`, and no attribute call on a name
    # `pickle`.
    tree = ast.parse(
        Path("inference/core/workflows/prototypes/image_codec.py").read_text(
            encoding="utf-8"
        )
    )
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "pickle" not in imported

    called_on = {
        node.func.value.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
    }
    assert "pickle" not in called_on


def test_default_decodes_base64_numpy_and_raw_bytes() -> None:
    codec = WorkflowsLocalImageCodec()

    decoded, is_bgr = codec.decode_string(_b64())
    assert is_bgr is True and decoded.shape == IMAGE.shape

    decoded, is_bgr = codec.decode_string(encode_image_to_jpeg_bytes(IMAGE))
    assert is_bgr is True and decoded.shape == IMAGE.shape

    decoded, is_bgr = codec.load_image({"type": "base64", "value": _b64()})
    assert is_bgr is True and decoded.shape == IMAGE.shape

    decoded, is_bgr = codec.load_image({"type": "numpy_object", "value": IMAGE})
    assert is_bgr is True and decoded is IMAGE

    decoded, is_bgr = codec.load_image(IMAGE)
    assert is_bgr is True and decoded is IMAGE


@pytest.mark.parametrize("disable_preproc_auto_orient", [True, False])
def test_default_forwards_decoding_flags_on_every_in_memory_path(
    disable_preproc_auto_orient: bool,
) -> None:
    # Round-1 Defect 5: the round-0 default computed flags then dropped them for
    # bare strings, silently re-enabling EXIF auto-orientation. Compare against
    # the server loader, which is the behaviour being replaced.
    from inference.core.utils import image_utils as server_image_utils

    codec = WorkflowsLocalImageCodec()
    payload = _b64_with_exif_orientation_6()
    flags = choose_image_decoding_flags(
        disable_preproc_auto_orient=disable_preproc_auto_orient
    )

    declared, _ = codec.load_image(
        {"type": "base64", "value": payload},
        disable_preproc_auto_orient=disable_preproc_auto_orient,
    )
    inferred, _ = codec.load_image(
        payload, disable_preproc_auto_orient=disable_preproc_auto_orient
    )
    reference, _ = server_image_utils.load_image(
        {"type": "base64", "value": payload},
        disable_preproc_auto_orient=disable_preproc_auto_orient,
    )
    direct, _ = codec.decode_string(payload, cv_imread_flags=flags)

    assert declared.shape == inferred.shape == reference.shape == direct.shape
    assert np.array_equal(declared, reference)
    assert np.array_equal(inferred, reference)


@pytest.mark.parametrize(
    "cv_imread_flags",
    [cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED],
)
def test_default_decode_string_does_not_promote_like_the_server_decoder(
    cv_imread_flags: int,
) -> None:
    # Round-2 Defect 8: `attempt_loading_image_from_string` (image_utils.py:238)
    # returns the decoder's result untouched; the grayscale->BGR promotion is a
    # `load_image` concern (image_utils.py:111). Promoting inside decode_string
    # made IMREAD_GRAYSCALE return (6, 9, 3) where the original returns (6, 9).
    from inference.core.utils import image_utils as server_image_utils

    codec = WorkflowsLocalImageCodec()
    payload = _b64()
    ours, ours_is_bgr = codec.decode_string(payload, cv_imread_flags=cv_imread_flags)
    theirs, theirs_is_bgr = server_image_utils.attempt_loading_image_from_string(
        value=payload, cv_imread_flags=cv_imread_flags
    )
    assert ours.shape == theirs.shape
    assert ours_is_bgr == theirs_is_bgr
    assert np.array_equal(ours, theirs)


def test_default_load_image_promotes_grayscale_to_bgr() -> None:
    gray = np.tile(np.arange(64, dtype=np.uint8), (48, 1))
    decoded, _ = WorkflowsLocalImageCodec().load_image(
        {"type": "numpy_object", "value": gray}
    )
    assert decoded.shape == (48, 64, 3)


def test_registry_returns_the_refusing_default_until_a_host_installs_one() -> None:
    assert isinstance(get_image_codec(), WorkflowsLocalImageCodec)


def test_installing_a_codec_makes_it_the_process_codec() -> None:
    codec = WorkflowsLocalImageCodec()
    set_image_codec(codec)
    assert get_image_codec() is codec


def test_installing_the_same_codec_twice_is_a_no_op() -> None:
    codec = WorkflowsLocalImageCodec()
    set_image_codec(codec)
    set_image_codec(codec)
    assert get_image_codec() is codec


def test_installing_a_conflicting_codec_is_an_error() -> None:
    first = WorkflowsLocalImageCodec()
    set_image_codec(first)
    with pytest.raises(WorkflowEnvironmentConfigurationError):
        set_image_codec(WorkflowsLocalImageCodec())
    assert get_image_codec() is first


def test_two_threads_installing_different_codecs_leave_exactly_one_winner() -> None:
    # Round-1 Defect 2: the round-0 setter's check-and-set was unsynchronised and
    # a controlled two-thread run installed BOTH codecs. The barrier maximises
    # the overlap; the lock must make exactly one call win and the other raise.
    barrier = threading.Barrier(2)
    outcomes = []
    outcomes_lock = threading.Lock()

    def worker(codec) -> None:
        barrier.wait(timeout=5)
        try:
            set_image_codec(codec)
            result = ("installed", codec)
        except WorkflowEnvironmentConfigurationError:
            result = ("refused", codec)
        with outcomes_lock:
            outcomes.append(result)

    candidates = [WorkflowsLocalImageCodec(), WorkflowsLocalImageCodec()]
    threads = [threading.Thread(target=worker, args=(c,)) for c in candidates]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert sorted(status for status, _ in outcomes) == ["installed", "refused"]
    winner = next(codec for status, codec in outcomes if status == "installed")
    assert get_image_codec() is winner


def test_many_threads_installing_the_same_codec_all_succeed() -> None:
    # The production case: every request re-installs the same server singleton.
    codec = WorkflowsLocalImageCodec()
    barrier = threading.Barrier(8)
    failures = []

    def worker() -> None:
        barrier.wait(timeout=5)
        try:
            set_image_codec(codec)
        except Exception as error:  # noqa: BLE001 - the test is the assertion
            failures.append(error)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert failures == []
    assert get_image_codec() is codec


def test_module_stays_free_of_the_server_package_and_of_io() -> None:
    source = Path("inference/core/workflows/prototypes/image_codec.py").read_text(
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
